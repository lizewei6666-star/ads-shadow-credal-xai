# -*- coding: utf-8 -*-
"""Shadow-calibrated, specification-robust interpretation of ADS crashes.

The workflow admits models by nested cross-validation, computes four distinct
class-specific explanation statistics, calibrates each statistic against a
feature-matched replacement-shadow reference, and derives credal relevance
bounds and strict partial orders. The paper configuration uses five outer
folds, five inner folds, and 49 shadow replicates.

Execution fingerprints, atomic checkpoints, and deterministic equivalence
checks support interruption-safe single- or multi-computer runs without
changing the statistical contract.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import importlib.metadata as importlib_metadata
import io
import json
import multiprocessing as mp
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import platform
import random
import shutil
import sys
import tempfile
import time
import traceback
import uuid
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# USER CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data and output paths. Environment-variable overrides let both computers use
# the same byte-identical script without editing its statistical method.
EXCEL_PATH = os.environ.get(
    "ADS_XAI_EXCEL_PATH",
    str(PROJECT_ROOT / "data" / "ads_crashes.xlsx"),
)
BASE_OUTPUT_DIR = os.environ.get(
    "ADS_XAI_BASE_OUTPUT_DIR",
    str(PROJECT_ROOT / "outputs"),
)

# Set to an existing directory created by this script to resume. Checkpoints
# from the earlier weighted-consensus workflow are deliberately incompatible.
_DEFAULT_RESUME_RUN_DIR = ""
_RESUME_RUN_DIR_TEXT = os.environ.get(
    "ADS_XAI_RESUME_RUN_DIR",
    _DEFAULT_RESUME_RUN_DIR,
).strip()
RESUME_RUN_DIR: Optional[str] = (
    None
    if _RESUME_RUN_DIR_TEXT.lower() in {"", "none", "null"}
    else _RESUME_RUN_DIR_TEXT
)

# Distributed execution has three explicit modes:
#   single:    original one-computer workflow;
#   worker:    compute checkpoints for only the assigned outer partitions;
#   aggregate: load all merged checkpoints and produce the final analysis.
# Partition assignment changes scheduling only. It is excluded from the
# statistical method digest and never changes a fitted or explained estimand.
DISTRIBUTED_EXECUTION_MODE = os.environ.get(
    "ADS_XAI_DISTRIBUTED_MODE", "single"
).strip().lower()
DISTRIBUTED_WORKER_ID = os.environ.get(
    "ADS_XAI_DISTRIBUTED_WORKER_ID", "local"
).strip()
DISTRIBUTED_PARTITIONS_TEXT = os.environ.get(
    "ADS_XAI_DISTRIBUTED_PARTITIONS", ""
).strip()

# The second study excludes the eight records with unknown vehicle age.
# Keep these settings aligned with the 1,469-record analytical sample.
EXCLUDE_UNKNOWN_VEHICLE_AGE = True
EXPECTED_N_SAMPLES = 1469

# Dormant legacy compatibility settings. TabPFN is excluded from
# CANDIDATE_MODELS below and is not imported, fitted, or explained in the
# paper-facing workflow.
TABPFN_LOCAL_CKPT = os.environ.get("ADS_XAI_TABPFN_CKPT", "")

# "smoke" checks the execution path; "full" reproduces the paper analysis.
RUN_PROFILE = os.environ.get("ADS_XAI_RUN_PROFILE", "full").strip().lower()

# If True, TabPFN must run on CUDA. If CUDA is unavailable, the script stops.
FORCE_TABPFN_CUDA = True

# TabPFN permutation SHAP is executed in a fresh spawned process. Repeated
# model-agnostic masking can otherwise leave a fatal CUDA kernel error in the
# main process and invalidate every subsequent explanation call. TabPFN uses
# FP32 inference and its own memory-saving computations. External row batching
# is accepted only after an explicit probability-equivalence check; otherwise
# the worker explains the unbatched prediction function.
ISOLATE_TABPFN_SHAP = True
TABPFN_FORCE_FP32_INFERENCE = True
TABPFN_INTERNAL_MEMORY_SAVING = True
TABPFN_SHAP_INITIAL_PREDICT_BATCH_SIZE = 2048
TABPFN_SHAP_MIN_PREDICT_BATCH_SIZE = 256
TABPFN_SHAP_UNBATCHED_RETRIES = 2
TABPFN_SHAP_RECONSTRUCTION_TOL = 1e-5
TABPFN_SHAP_BATCH_EQUIVALENCE_PROBE_ROWS = 512
TABPFN_SHAP_ALLOW_CPU_FALLBACK = False

# Long TabPFN SHAP jobs must not communicate through the Windows system temp
# directory. Successful call-level results are cached under the run directory,
# and completed outer partitions are checkpointed atomically. These controls
# alter execution only; they do not alter the explained prediction function or
# any statistical estimand.
ENABLE_TABPFN_SHAP_CACHE = True
KEEP_FAILED_TABPFN_WORK_DIRS = True
ENABLE_OUTER_PARTITION_CHECKPOINTS = True
CHECKPOINT_FORMAT_VERSION = 1

# Candidate prediction models. Optional backends are skipped when not installed.
# TabPFN is intentionally excluded from the formal interpretation workflow.
# The retained models cover linear, bagging, randomized-tree, histogram boosting,
# and major gradient-boosting families while supporting the frozen four-explainer
# execution contract.
CANDIDATE_MODELS = ["lr", "rf", "et", "hgb", "xgb", "lgbm", "catboost"]

# Explanation design. The paper-facing default explains every model admitted by
# the split-specific one-standard-error rule. No fixed model-count cap is used.
# Set EXPLAIN_ONLY_BEST_MODEL=True only for a quick diagnostic run.
EXPLAIN_ONLY_BEST_MODEL = False
USE_ONE_SE_ADMISSIBLE_MODEL_SET = True

# Stratified outer cross-fitting. Five folds retain approximately six to seven
# Moderate-plus records in each outer evaluation fold. The paper-facing run uses
# one complete five-fold pass; it therefore estimates fold variation but not
# variation across repeated fold assignments.
OUTER_N_SPLITS_FULL = 5
OUTER_N_REPEATS_FULL = 1
OUTER_N_SPLITS_SMOKE = 3
OUTER_N_REPEATS_SMOKE = 1
OUTER_RANDOM_SEED = 42
INNER_CV_FOLDS = 5

# The same deterministic inner folds support one-SE assessment, training-only
# explanation construction, and shadow-null calibration. The tasks are
# computed separately and the outer holdout is absent from all three.
SHADOW_INNER_CV_FOLDS = INNER_CV_FOLDS

# Computational controls.
MODEL_N_JOBS = 1
USE_BALANCED_SAMPLE_WEIGHT = True
ENABLE_SHAP = True
ENABLE_PFI = True
ENABLE_LOCO = True
ENABLE_FIRM = True
ENABLE_PDP = True
ENABLE_ICE = True
OUTPUT_FIGURES = True

# Frozen credal-inference contract.
CREDAL_SOURCE_NAMES = ("SHAP", "PFI", "LOCO", "FIRM")
SHADOW_NULL_MODE = "raw_feature_replacement_empirical_marginal"
SHADOW_REPLICATES_FULL = 49
SHADOW_REPLICATES_SMOKE = 2
SHADOW_NUMERICAL_TOLERANCE = 1e-12
SHADOW_CHECKPOINT_FORMAT_VERSION = 3
ENABLE_SHADOW_UNIT_CHECKPOINTS = True
# Windows ``spawn`` gives every worker its own Python/model state. Twelve
# simultaneous four-explainer jobs can exceed physical memory even when each
# estimator is single-threaded, which terminates a child process without a
# Python traceback and surfaces as BrokenProcessPool. Four workers keep the
# execution memory-bounded; this is a scheduling choice only.
SHADOW_PARALLEL_WORKERS = int(
    os.environ.get("ADS_XAI_SHADOW_PARALLEL_WORKERS", "4")
)
SHADOW_MAX_PENDING_TASKS = int(
    os.environ.get("ADS_XAI_SHADOW_MAX_PENDING_TASKS", "8")
)
SHADOW_WORKER_GC_AT_TASK_END = True

# Execution-only accelerations. These settings do not change any statistical
# estimand. PFI keeps the same raw-feature permutations and evaluation order,
# but transforms and predicts them in fixed batches. LOCO reduced models are
# fitted once per outer partition, model, omitted feature, and inner fold, then
# reused across all shadow replicates because deleting the replaced feature
# leaves identical reduced training and validation tables.
ENABLE_PFI_BATCH_PREDICTION = True
PFI_BATCH_PREDICTION_ROWS = int(
    os.environ.get("ADS_XAI_PFI_BATCH_PREDICTION_ROWS", "8192")
)
PFI_EQUIVALENCE_ATOL = 1e-10
PFI_EQUIVALENCE_RTOL = 1e-8
PFI_EQUIVALENCE_FEATURES = 2
ENABLE_LOCO_REDUCED_PROBABILITY_CACHE = True
LOCO_CACHE_PARALLEL_WORKERS = int(
    os.environ.get("ADS_XAI_LOCO_CACHE_PARALLEL_WORKERS", "4")
)
LOCO_CACHE_MAX_PENDING_TASKS = int(
    os.environ.get("ADS_XAI_LOCO_CACHE_MAX_PENDING_TASKS", "8")
)
LOCO_CACHE_CONTRACT_VERSION = "loco_reduced_probability_v1"

# Statistical checkpoints from the immediately preceding target-coordinate
# SHAP implementation may be migrated only after payload and numerical
# compatibility audits pass. The hash identifies the known source file, not an
# arbitrary older workflow.
STATISTICAL_CONTRACT_VERSION = "credal_shadow_statistical_contract_v3"
EXECUTION_CONTRACT_VERSION = "memory_bounded_parallel_execution_v2"
PREDICTIVE_SPECIFICATION_CONTRACT_VERSION = (
    "fixed_model_factory_20260729_v1"
)
PREPROCESSING_CONTRACT_VERSION = (
    "raw_numeric_scale_categorical_onehot_20260729_v1"
)
SOURCE_STATISTIC_CONTRACT_VERSION = (
    "shap_pfi_loco_firm_statistics_20260729_v1"
)
RANDOMIZATION_CONTRACT_VERSION = (
    "stable_keyed_randomization_20260729_v1"
)
LEGACY_TARGET_SHAP_V1_SCRIPT_SHA256 = (
    "a6510ab5f5cce68573a0011819111571f709bc51ca2460d7ae43eedef99c0e62"
)
# The immediately preceding target-coordinate SHAP script may have been saved
# with different line endings on another machine. Because that legacy script
# embedded its raw file hash in every checkpoint fingerprint, byte-identical
# Python semantics can otherwise appear incompatible. Candidate legacy scripts
# are therefore discovered in the current script directory and validated
# against the frozen legacy contract before their actual file hashes are used
# to reconstruct possible checkpoint fingerprints.
AUTO_DISCOVER_LEGACY_RESUME_SCRIPTS = True
LEGACY_RESUME_SCRIPT_PATHS: List[str] = []

# An unknown fingerprint is never trusted. "recompute" records the rejected
# marker and rebuilds only that unit; "error" retains the previous fail-closed
# behavior. Checksum or payload corruption remains fatal under both policies.
INCOMPATIBLE_CHECKPOINT_POLICY = "recompute"
ENABLE_TRAINING_ONLY_UTILITY_VALIDATION = True
ENABLE_LEAVE_ONE_EXPLAINER_OUT_SENSITIVITY = True
ENABLE_REPEAT_CONSISTENCY_DIAGNOSTIC = False

# Full execution is intentionally strict. A source/model result is valid only
# when every requested class and original feature is finite and all B shadow
# replicates are present. Failures reduce common support; they are never
# replaced by available-case source averaging.
FAIL_IF_ANY_PRIMARY_COMMON_SUPPORT_EMPTY = True
FAIL_IF_UTILITY_VALIDATION_INCOMPLETE = True

# The ADS analysis table contains discrete or grouped variables. Incident month
# is stored numerically but remains a 12-level factor. Add a feature name here
# only when the analytical data dictionary defines it as genuinely continuous.
FIRM_CONTINUOUS_FEATURES: List[str] = []

# Probability diagnostics and explanation stress tests.
ENABLE_CALIBRATION_DIAGNOSTICS = True
CALIBRATION_N_BINS = 10
ENABLE_NEGATIVE_CONTROL = True
NEGATIVE_CONTROL_FEATURE = "Random_Noise_Control"
NEGATIVE_CONTROL_PERMUTATION_REPEATS = 30
SOURCE_WEIGHT_EPS = 1e-12

# Every explanation source evaluates the complete holdout partition. SHAP alone
# needs a training-only masking reference. A fixed 120-record background is used
# throughout; this limits only the masking reference, not the evaluation sample.
SHAP_BACKGROUND_SIZE = 120
ENABLE_SHAP_BACKGROUND_SENSITIVITY = False
SHAP_BACKGROUND_SENSITIVITY_SIZES = [120]
SHAP_EVAL_SIZE = None
# Five forward-and-reverse antithetic cycles reduce permutation-order noise.
# PFI uses 30 permutations per feature and run. These are numerical integration
# settings, not fitted thresholds; their run-level dispersion is exported.
SHAP_PERMUTATION_CYCLES = 5
TARGET_SHAP_CONTRACT_VERSION = "target_coordinate_permutation_shap_v1"
TARGET_SHAP_PERMUTATION_PLAN_VERSION = "common_plan_v1"
TARGET_SHAP_SENSITIVITY_PLAN_VERSION = "common_plan_v2"
TARGET_SHAP_PREDICTION_BATCH_ROWS = int(
    os.environ.get("ADS_XAI_TARGET_SHAP_PREDICTION_BATCH_ROWS", "8192")
)
TARGET_SHAP_EQUIVALENCE_RECORDS = 12
TARGET_SHAP_EQUIVALENCE_ATOL = 1e-7
TARGET_SHAP_EQUIVALENCE_RTOL = 1e-6
ENABLE_TARGET_SHAP_EQUIVALENCE_AUDIT = True
ENABLE_TARGET_SHAP_PLAN_SENSITIVITY_AUDIT = True
PERMUTATION_REPEATS = 30
DISPLAY_TOP_K_FEATURES = 6
EXPLAIN_ALL_FEATURES = True
PDP_MAX_VALUES = None
PDP_MIN_TRAIN_COUNT = 1

# No source weights are used in the primary method.
ENABLE_WEIGHT_SENSITIVITY_TABLE = False
WEIGHT_SENSITIVITY_TOP_K = DISPLAY_TOP_K_FEATURES

# Method-validation controls. This lightweight known-truth experiment validates
# the aggregation rule itself under complementary, noisy evidence sources. It
# does not claim that any explainer recovers a causal data-generating mechanism.
ENABLE_FUSION_RECOVERY_SIMULATION = False
FUSION_SIMULATION_REPLICATES_FULL = 200
FUSION_SIMULATION_REPLICATES_SMOKE = 12
FUSION_SIMULATION_PARTITIONS = 5
FUSION_SIMULATION_MODELS = 3
FUSION_SIMULATION_SEED = 20260715

# Data-driven structural compatibility audit. This module is diagnostic and
# remains outside feature-ranking fusion. FCI is repeated across the same
# outer-training partitions and a small multiverse of conventional G-square
# settings so no single alpha or conditioning-depth choice determines the
# conclusion. Label permutations retain the observed class counts and provide
# a matched null benchmark. No expert-supplied edge is required or accepted.
ENABLE_STRUCTURAL_COMPATIBILITY_AUDIT = True
STRUCTURAL_TARGET_NODE = "Injury_3Class"
STRUCTURAL_CI_TEST = "gsq"
STRUCTURAL_ALPHA_GRID = [0.01, 0.05, 0.10]
STRUCTURAL_DEPTH_GRID = [1, 2, -1]
STRUCTURAL_MAX_PATH_LENGTH = -1
STRUCTURAL_NULL_REPLICATES_FULL = 99
STRUCTURAL_NULL_REPLICATES_SMOKE = 3
STRUCTURAL_RANDOM_SEED = 20260723
STRUCTURAL_SHOW_FCI_PROGRESS = False
STRUCTURAL_PARALLEL_WORKERS = int(
    os.environ.get("ADS_XAI_STRUCTURAL_PARALLEL_WORKERS", "4")
)
STRUCTURAL_MAX_PENDING_TASKS = int(
    os.environ.get("ADS_XAI_STRUCTURAL_MAX_PENDING_TASKS", "8")
)
ENABLE_STRUCTURAL_TASK_CHECKPOINTS = True
STRUCTURAL_TASK_CONTRACT_VERSION = "parallel_fci_group_v1"

# Lightweight deterministic checks for agreement, source availability,
# fallbacks, PAG endpoint decoding, categorical encoding, and entropy.
# These checks do not fit a prediction or graphical model.
RUN_INTERNAL_METHOD_CHECKS = True

# Stable analysis-record identifiers are generated after all cleaning rules and
# before any outer or inner split. They are never supplied to a predictor.
STABLE_RECORD_ID_COL = "_stable_record_id"


# =============================================================================
# ENVIRONMENT
# =============================================================================

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ads_reliable_xai_mpl_cache")
)

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


# =============================================================================
# TASK CONFIG
# =============================================================================

TARGET_COL = "Injury_3Class"
TARGET_MAP = {
    "No_Injury": 0,
    "Minor_Injury": 1,
    "ModeratePlus_Injury": 2,
}
TARGET_NAMES = {
    0: "No injury",
    1: "Minor injury",
    2: "Moderate-plus injury",
}

DEFAULT_FEATURE_COLUMNS = [
    "Incident_Month",
    "Is_Weekend",
    "Incident_Hour",
    "RoadwayType_Group",
    "RoadSurface_Group",
    "RoadCondition_Group",
    "Lighting_Group",
    "Weather_Group",
    "CrashWith_Group",
    "CPMovement_Group",
    "SVMovement_Group",
    "Mileage_Group",
    "VehicleAge_Group",
    "PostedSpeedLimit_Group",
]


@dataclass(frozen=True)
class FeatureSchema:
    feature_cols: List[str]
    numeric_cols: List[str]
    categorical_cols: List[str]


@dataclass
class EncodedData:
    X_train: np.ndarray
    X_test: np.ndarray
    encoded_feature_names: List[str]
    original_feature_groups: Dict[str, List[int]]
    preprocessor: object
    categorical_feature_indices: List[int]


@dataclass(frozen=True)
class OuterPartition:
    """One outer cross-fitting partition and its reproducibility metadata."""

    partition_id: int
    repeat_id: int
    fold_id: int
    split_seed: int
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class RawFeatureCodec:
    """Reversible numeric representation used only by the SHAP masker."""

    feature_cols: List[str]
    numeric_cols: List[str]
    categorical_levels: Dict[str, Tuple[str, ...]]
    unknown_token: str = "__SHAP_UNKNOWN__"

    @classmethod
    def fit(cls, X_train: pd.DataFrame, schema: FeatureSchema) -> "RawFeatureCodec":
        levels = {
            feature: tuple(
                sorted(X_train[feature].astype(str).dropna().unique().tolist())
            )
            for feature in schema.categorical_cols
        }
        return cls(
            feature_cols=list(schema.feature_cols),
            numeric_cols=list(schema.numeric_cols),
            categorical_levels=levels,
        )

    def encode(self, X: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(X), len(self.feature_cols)), dtype=float)
        numeric_set = set(self.numeric_cols)
        for pos, feature in enumerate(self.feature_cols):
            if feature in numeric_set:
                out[:, pos] = pd.to_numeric(X[feature], errors="coerce").to_numpy(
                    dtype=float
                )
                continue
            mapping = {
                value: idx for idx, value in enumerate(self.categorical_levels[feature])
            }
            out[:, pos] = (
                X[feature].astype(str).map(mapping).fillna(-1).to_numpy(dtype=float)
            )
        return out

    def decode(self, X: np.ndarray) -> pd.DataFrame:
        arr = np.asarray(X, dtype=float)
        data = {}
        numeric_set = set(self.numeric_cols)
        for pos, feature in enumerate(self.feature_cols):
            if feature in numeric_set:
                data[feature] = arr[:, pos].astype(float)
                continue
            levels = self.categorical_levels[feature]
            codes = np.rint(arr[:, pos]).astype(int)
            data[feature] = [
                levels[code] if 0 <= int(code) < len(levels) else self.unknown_token
                for code in codes
            ]
        return pd.DataFrame(data, columns=self.feature_cols)


# =============================================================================
# SMALL UTILITIES
# =============================================================================


def cprint(msg: str = "") -> None:
    print(msg, flush=True)


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def safe_json_dump(obj: dict, path: Path) -> None:
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=default)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def atomic_json_dump(obj: dict, path: Path) -> None:
    """Write JSON through a same-directory temporary file and atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the temporary name short.  On Windows, repeating the destination
    # filename plus a full UUID can push an otherwise valid checkpoint path
    # beyond the traditional MAX_PATH limit.
    temp_path = path.parent / f".tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    try:
        temp_path.write_text(
            json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_pickle_dump(obj, path: Path) -> None:
    """Write a pickle atomically so interrupted jobs cannot look complete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    try:
        pd.to_pickle(obj, temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_dataframe_to_csv(
    frame: pd.DataFrame,
    path: Path,
    encoding: str = "utf-8-sig",
) -> None:
    """Write a CSV atomically so a stopped run cannot leave a partial audit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    try:
        frame.to_csv(temp_path, index=False, encoding=encoding)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def stable_json_digest(obj: object) -> str:
    payload = json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataframe_digest(frame: pd.DataFrame) -> str:
    """Hash values, index, columns, and dtypes for cache validation."""
    normalized = frame.reset_index(drop=True)
    h = hashlib.sha256()
    h.update(stable_json_digest(list(normalized.columns)).encode("ascii"))
    h.update(
        stable_json_digest([str(dtype) for dtype in normalized.dtypes]).encode("ascii")
    )
    values_hash = pd.util.hash_pandas_object(
        normalized, index=True, categorize=True
    ).to_numpy(dtype=np.uint64, copy=False)
    h.update(values_hash.tobytes())
    return h.hexdigest()


def stable_record_hash_table(
    data: pd.DataFrame,
    schema: FeatureSchema,
) -> pd.DataFrame:
    """Map private sequential analysis IDs to canonical cleaned-row hashes."""
    columns = list(schema.feature_cols) + [TARGET_COL]
    rows = []
    for _, row in data[[STABLE_RECORD_ID_COL] + columns].iterrows():
        payload = {
            column: row[column]
            for column in columns
        }
        rows.append(
            {
                STABLE_RECORD_ID_COL: int(row[STABLE_RECORD_ID_COL]),
                "source_row_sha256": stable_json_digest(payload),
            }
        )
    return pd.DataFrame(rows)


def array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    h = hashlib.sha256()
    h.update(str(array.dtype).encode("ascii"))
    h.update(stable_json_digest(list(array.shape)).encode("ascii"))
    h.update(array.tobytes())
    return h.hexdigest()


def file_sha256(path_like: str, chunk_size: int = 1024 * 1024) -> Optional[str]:
    path = Path(path_like)
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_output_manifest(output_dir: Path) -> Path:
    """Write checksums for exported analysis artifacts, excluding checkpoints."""
    output_dir = Path(output_dir).resolve()
    manifest_path = output_dir / "output_manifest.csv"
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        relative = path.relative_to(output_dir)
        if relative.parts and relative.parts[0].lower() == "checkpoints":
            continue
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(str(path)),
            }
        )
    pd.DataFrame(
        rows, columns=["relative_path", "size_bytes", "sha256"]
    ).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return manifest_path


def package_version(package_name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(package_name)
    except Exception:
        return None


def collect_environment_metadata() -> dict:
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "shap",
        "xgboost",
        "lightgbm",
        "catboost",
        "matplotlib",
        "openpyxl",
        "causal-learn",
    ]
    tabpfn_enabled = "tabpfn" in [name.lower() for name in CANDIDATE_MODELS]
    if tabpfn_enabled:
        packages.extend(["tabpfn", "torch"])
    meta = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "package_versions": {pkg: package_version(pkg) for pkg in packages},
        "tabpfn_backend_enabled": bool(tabpfn_enabled),
    }
    if tabpfn_enabled:
        meta["tabpfn_checkpoint_sha256"] = file_sha256(TABPFN_LOCAL_CKPT)
    if tabpfn_enabled and module_available("torch"):
        try:
            import torch

            meta["torch_cuda_available"] = bool(torch.cuda.is_available())
            meta["torch_cuda_device_count"] = int(torch.cuda.device_count())
            meta["torch_cuda_device_name"] = (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() and torch.cuda.device_count() > 0
                else None
            )
            meta["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        except Exception as exc:
            meta["torch_metadata_error"] = str(exc)
    return meta


def validate_runtime_requirements() -> None:
    """Fail early when the configured analysis cannot be reproduced."""
    source_flags = {
        "SHAP": bool(ENABLE_SHAP),
        "PFI": bool(ENABLE_PFI),
        "LOCO": bool(ENABLE_LOCO),
        "FIRM": bool(ENABLE_FIRM),
    }
    disabled_sources = [
        source for source in CREDAL_SOURCE_NAMES if not source_flags.get(source, False)
    ]
    if disabled_sources:
        raise RuntimeError(
            "The frozen primary credal contract requires all four sources. "
            f"Disabled sources: {disabled_sources}"
        )
    if int(shadow_replicate_count(RUN_PROFILE)) < 1:
        raise RuntimeError("At least one replacement-shadow replicate is required.")
    if int(SHADOW_PARALLEL_WORKERS) < 1:
        raise RuntimeError("SHADOW_PARALLEL_WORKERS must be at least one.")
    if int(SHADOW_MAX_PENDING_TASKS) < int(SHADOW_PARALLEL_WORKERS):
        raise RuntimeError(
            "SHADOW_MAX_PENDING_TASKS must be at least "
            "SHADOW_PARALLEL_WORKERS."
        )
    if int(PFI_BATCH_PREDICTION_ROWS) < 1:
        raise RuntimeError("PFI_BATCH_PREDICTION_ROWS must be positive.")
    if int(LOCO_CACHE_PARALLEL_WORKERS) < 1:
        raise RuntimeError(
            "LOCO_CACHE_PARALLEL_WORKERS must be at least one."
        )
    if int(STRUCTURAL_PARALLEL_WORKERS) < 1:
        raise RuntimeError(
            "STRUCTURAL_PARALLEL_WORKERS must be at least one."
        )
    if int(TARGET_SHAP_PREDICTION_BATCH_ROWS) < int(SHAP_BACKGROUND_SIZE):
        raise RuntimeError(
            "TARGET_SHAP_PREDICTION_BATCH_ROWS must fit at least one "
            "complete background-integrated coalition state."
        )
    requirements = {
        "scikit-learn": "sklearn",
        "SciPy": "scipy",
        "openpyxl": "openpyxl",
    }
    if ENABLE_SHAP:
        requirements["SHAP"] = "shap"
    if OUTPUT_FIGURES:
        requirements["Matplotlib"] = "matplotlib"
    if ENABLE_STRUCTURAL_COMPATIBILITY_AUDIT:
        requirements["causal-learn"] = "causallearn"
    missing = [
        label for label, module in requirements.items() if not module_available(module)
    ]
    if missing:
        raise RuntimeError(
            "Missing dependencies required by the configured workflow: "
            + ", ".join(missing)
        )


def sample_indices(n: int, max_n: Optional[int], seed: int) -> np.ndarray:
    idx = np.arange(n)
    if max_n is None or n <= int(max_n):
        return idx
    rs = np.random.RandomState(int(seed))
    # A seeded permutation provides a reproducible training-only subset. If
    # multiple limits are configured in a future sensitivity run, prefixes of
    # the same permutation also make those subsets nested.
    out = rs.permutation(idx)[: int(max_n)]
    return np.sort(out)


def require_full_evaluation(
    table: pd.DataFrame,
    count_column: str,
    expected_n: int,
    source_name: str,
) -> None:
    """Fail fast if a paper-facing explanation source drops holdout rows."""
    if table is None or table.empty:
        return
    if count_column not in table.columns:
        raise RuntimeError(f"{source_name} output is missing {count_column}.")
    observed = set(
        pd.to_numeric(table[count_column], errors="coerce").dropna().astype(int)
    )
    if observed != {int(expected_n)}:
        raise RuntimeError(
            f"{source_name} did not use the complete holdout partition: "
            f"expected {expected_n}, observed {sorted(observed)}."
        )


def stable_seed_from_key(*parts: object) -> int:
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and request stable FP32 CUDA math."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    if module_available("torch"):
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def make_outer_partitions(y: np.ndarray, run_profile: str) -> List[OuterPartition]:
    """Create repeated stratified outer folds used by every downstream stage."""
    from sklearn.model_selection import RepeatedStratifiedKFold

    profile = str(run_profile).lower()
    n_splits = OUTER_N_SPLITS_SMOKE if profile == "smoke" else OUTER_N_SPLITS_FULL
    n_repeats = OUTER_N_REPEATS_SMOKE if profile == "smoke" else OUTER_N_REPEATS_FULL
    y = np.asarray(y, dtype=int)
    min_class_count = int(pd.Series(y).value_counts().min())
    if int(n_splits) > min_class_count:
        raise RuntimeError(
            f"OUTER_N_SPLITS={n_splits} exceeds the smallest class count={min_class_count}."
        )

    splitter = RepeatedStratifiedKFold(
        n_splits=int(n_splits),
        n_repeats=int(n_repeats),
        random_state=int(OUTER_RANDOM_SEED),
    )
    dummy = np.zeros((len(y), 1), dtype=float)
    partitions: List[OuterPartition] = []
    for zero_based_id, (train_idx, test_idx) in enumerate(splitter.split(dummy, y)):
        repeat_id = zero_based_id // int(n_splits) + 1
        fold_id = zero_based_id % int(n_splits) + 1
        partition_id = zero_based_id + 1
        partitions.append(
            OuterPartition(
                partition_id=int(partition_id),
                repeat_id=int(repeat_id),
                fold_id=int(fold_id),
                split_seed=stable_seed_from_key(
                    "outer_partition", OUTER_RANDOM_SEED, repeat_id, fold_id
                ),
                train_idx=np.asarray(train_idx, dtype=int),
                test_idx=np.asarray(test_idx, dtype=int),
            )
        )
    return partitions


def outer_partition_manifest(
    partitions: List[OuterPartition], y: np.ndarray
) -> pd.DataFrame:
    """Record fold sizes and class support without exporting row identifiers."""
    y = np.asarray(y, dtype=int)
    rows = []
    for part in partitions:
        row = {
            "outer_partition": int(part.partition_id),
            "outer_repeat": int(part.repeat_id),
            "outer_fold": int(part.fold_id),
            "split_seed": int(part.split_seed),
            "n_train": int(len(part.train_idx)),
            "n_holdout": int(len(part.test_idx)),
        }
        for class_id in sorted(np.unique(y)):
            row[f"n_train_class_{int(class_id)}"] = int(
                np.sum(y[part.train_idx] == int(class_id))
            )
            row[f"n_holdout_class_{int(class_id)}"] = int(
                np.sum(y[part.test_idx] == int(class_id))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def attach_outer_partition_metadata(
    table: pd.DataFrame,
    partitions: List[OuterPartition],
) -> pd.DataFrame:
    """Attach repeat and fold identifiers to run-level outputs keyed by ``seed``."""
    if table is None or table.empty:
        return pd.DataFrame() if table is None else table.copy()
    if "seed" not in table.columns:
        raise ValueError(
            "Run-level explanation table is missing the partition key 'seed'."
        )
    metadata = pd.DataFrame(
        [
            {
                "seed": int(part.partition_id),
                "outer_repeat": int(part.repeat_id),
                "outer_fold": int(part.fold_id),
                "split_seed": int(part.split_seed),
            }
            for part in partitions
        ]
    )
    out = table.drop(
        columns=["outer_repeat", "outer_fold", "split_seed"], errors="ignore"
    ).merge(metadata, on="seed", how="left", validate="many_to_one")
    if out[["outer_repeat", "outer_fold", "split_seed"]].isna().any().any():
        raise RuntimeError(
            "Some explanation rows could not be mapped to an outer partition."
        )
    return out


def class_balanced_evaluation_weights(y: np.ndarray, class_id: int) -> np.ndarray:
    """Give the focal class and its complement equal empirical mass."""
    y = np.asarray(y, dtype=int)
    positive = y == int(class_id)
    n_positive = int(np.sum(positive))
    n_negative = int(len(y) - n_positive)
    if n_positive == 0 or n_negative == 0:
        raise ValueError(
            f"Class-balanced evaluation requires both outcomes for class_id={class_id}; "
            f"observed n_positive={n_positive}, n_negative={n_negative}."
        )
    weights = np.empty(len(y), dtype=float)
    weights[positive] = 0.5 / float(n_positive)
    weights[~positive] = 0.5 / float(n_negative)
    return weights


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    total = float(np.sum(weights[valid])) if np.any(valid) else 0.0
    if total <= SOURCE_WEIGHT_EPS:
        return np.nan
    return float(np.sum(values[valid] * weights[valid]) / total)


def map_hour_to_group(hour: int) -> str:
    hour = int(hour)
    if 0 <= hour <= 5:
        return "Early morning"
    if 6 <= hour <= 11:
        return "Morning"
    if 12 <= hour <= 17:
        return "Afternoon"
    if 18 <= hour <= 23:
        return "Evening"
    raise ValueError(f"Incident_Hour is outside 0-23: {hour}")


# =============================================================================
# DATA
# =============================================================================


def load_ads_table(excel_path: str, use_hour_group: bool = True) -> pd.DataFrame:
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"ADS Excel file was not found: {path}")

    df = pd.read_excel(path, sheet_name=0)
    n_rows_in_workbook = int(len(df))
    required = list(DEFAULT_FEATURE_COLUMNS) + [TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].copy()
    n_removed_unknown_vehicle_age = 0
    if EXCLUDE_UNKNOWN_VEHICLE_AGE:
        unknown_vehicle_age = (
            df["VehicleAge_Group"].astype(str).str.strip().str.lower().eq("unknown")
        )
        n_removed_unknown_vehicle_age = int(unknown_vehicle_age.sum())
        df = df.loc[~unknown_vehicle_age].copy()
    df = df.loc[df[TARGET_COL].notna()].copy()
    if df[TARGET_COL].dtype.kind not in "iu":
        unknown = sorted(set(df[TARGET_COL].astype(str)) - set(TARGET_MAP))
        if unknown:
            raise ValueError(f"Unknown target labels in {TARGET_COL}: {unknown}")
        df[TARGET_COL] = df[TARGET_COL].map(TARGET_MAP)
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    for col in ["Incident_Month", "Is_Weekend", "Incident_Hour"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Incident_Month", "Is_Weekend", "Incident_Hour"]).copy()
    df["Incident_Month"] = df["Incident_Month"].astype(int)
    df["Is_Weekend"] = df["Is_Weekend"].astype(int)
    df["Incident_Hour"] = df["Incident_Hour"].astype(int)

    if use_hour_group:
        df["Incident_Hour_Group"] = df["Incident_Hour"].apply(map_hour_to_group)
        df = df.drop(columns=["Incident_Hour"])

    # Month is discrete and cyclic. Encoding it as a category avoids imposing a
    # linear January-to-December effect on Logistic Regression while preserving
    # the original 14-variable analysis scale used by all explainers.
    df["Incident_Month"] = df["Incident_Month"].astype(str)

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    for col in feature_cols:
        if col not in ["Incident_Month", "Is_Weekend", "Incident_Hour"]:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace({"Unknown": "Other_Unknown", "Other": "Other_Unknown"})
            )

    out = df.reset_index(drop=True)
    out.insert(
        0,
        STABLE_RECORD_ID_COL,
        np.arange(len(out), dtype=np.int64),
    )
    out.attrs["n_rows_in_workbook"] = n_rows_in_workbook
    out.attrs["n_removed_unknown_vehicle_age"] = n_removed_unknown_vehicle_age
    out.attrs["exclude_unknown_vehicle_age"] = bool(EXCLUDE_UNKNOWN_VEHICLE_AGE)
    return out


def get_feature_schema(df: pd.DataFrame) -> FeatureSchema:
    feature_cols = [
        c
        for c in df.columns
        if c not in {TARGET_COL, STABLE_RECORD_ID_COL}
    ]
    numeric_cols = [c for c in ["Is_Weekend", "Incident_Hour"] if c in feature_cols]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return FeatureSchema(
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )


def split_xy(
    df: pd.DataFrame, schema: FeatureSchema
) -> Tuple[pd.DataFrame, np.ndarray]:
    columns = list(schema.feature_cols)
    if STABLE_RECORD_ID_COL in df.columns:
        columns.append(STABLE_RECORD_ID_COL)
    X = df[columns].copy()
    y = df[TARGET_COL].astype(int).to_numpy()
    return X, y


def summarize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for label, count in df[TARGET_COL].value_counts().sort_index().items():
        rows.append(
            {
                "class_id": int(label),
                "class_name": TARGET_NAMES.get(int(label), str(label)),
                "n": int(count),
                "share": float(count / max(n, 1)),
            }
        )
    return pd.DataFrame(rows)


def category_support_table(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    rows = []
    for col in schema.feature_cols:
        vc = df[col].astype(str).value_counts(dropna=False)
        for value, count in vc.items():
            rows.append(
                {
                    "feature": col,
                    "value": str(value),
                    "n": int(count),
                    "share": float(count / max(len(df), 1)),
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# PREPROCESSING
# =============================================================================


def make_one_hot_encoder():
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(schema: FeatureSchema, model_name: str = ""):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler

    transformers = []
    if str(model_name).lower() == "tabpfn":
        # TabPFN performs its own feature transformations. Categorical variables
        # are represented by one training-fitted code per original variable and
        # their encoded column positions are passed to the estimator.
        if schema.numeric_cols:
            transformers.append(("num", "passthrough", schema.numeric_cols))
        if schema.categorical_cols:
            transformers.append(
                (
                    "cat",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        dtype=np.float64,
                    ),
                    schema.categorical_cols,
                )
            )
        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=True,
        )

    if schema.numeric_cols:
        transformers.append(
            ("num", Pipeline([("scaler", StandardScaler())]), schema.numeric_cols)
        )
    if schema.categorical_cols:
        transformers.append(("cat", make_one_hot_encoder(), schema.categorical_cols))
    if not transformers:
        raise ValueError("No feature columns are available for preprocessing.")
    return ColumnTransformer(
        transformers=transformers, remainder="drop", verbose_feature_names_out=True
    )


def fit_transform_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    schema: FeatureSchema,
    model_name: str = "",
) -> EncodedData:
    preprocessor = build_preprocessor(schema, model_name=model_name)
    X_train_enc = np.asarray(preprocessor.fit_transform(X_train), dtype=float)
    X_test_enc = np.asarray(preprocessor.transform(X_test), dtype=float)
    names = get_encoded_feature_names(preprocessor, schema)
    groups = build_original_feature_groups(names, schema)
    categorical_feature_indices = sorted(
        idx for feature in schema.categorical_cols for idx in groups.get(feature, [])
    )
    return EncodedData(
        X_train=X_train_enc,
        X_test=X_test_enc,
        encoded_feature_names=names,
        original_feature_groups=groups,
        preprocessor=preprocessor,
        categorical_feature_indices=categorical_feature_indices,
    )


def get_encoded_feature_names(preprocessor, schema: FeatureSchema) -> List[str]:
    try:
        return [str(x) for x in preprocessor.get_feature_names_out()]
    except Exception:
        names: List[str] = []
        for col in schema.numeric_cols:
            names.append(f"num__{col}")
        if schema.categorical_cols:
            cat = preprocessor.named_transformers_.get("cat")
            try:
                cat_names = cat.get_feature_names_out(schema.categorical_cols)
                names.extend([f"cat__{x}" for x in cat_names])
            except Exception:
                names.extend([f"cat__{c}" for c in schema.categorical_cols])
        return names


def build_original_feature_groups(
    encoded_feature_names: List[str], schema: FeatureSchema
) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {c: [] for c in schema.feature_cols}
    for idx, name in enumerate(encoded_feature_names):
        original = None
        if name.startswith("num__"):
            original = name.replace("num__", "", 1)
        elif name.startswith("cat__"):
            stripped = name.replace("cat__", "", 1)
            matches = [
                c
                for c in schema.categorical_cols
                if stripped == c or stripped.startswith(c + "_")
            ]
            if matches:
                original = max(matches, key=len)
        if original is None:
            matches = [
                c
                for c in schema.feature_cols
                if name == c or name.endswith("__" + c) or name.startswith(c + "_")
            ]
            if matches:
                original = max(matches, key=len)
        if original in groups:
            groups[original].append(idx)
    empty = [k for k, v in groups.items() if not v]
    if empty:
        raise RuntimeError(
            f"Failed to map encoded columns back to original features: {empty}"
        )
    return groups


def transform_raw(preprocessor, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(preprocessor.transform(X), dtype=float)


# =============================================================================
# MODELS
# =============================================================================


MODEL_DISPLAY_NAMES = {
    "lr": "Logistic regression",
    "rf": "Random forest",
    "et": "Extra trees",
    "hgb": "Histogram gradient boosting",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "catboost": "CatBoost",
    "tabpfn": "TabPFN",
}

MODEL_DEPENDENCIES = {
    "xgb": "xgboost",
    "lgbm": "lightgbm",
    "catboost": "catboost",
    "tabpfn": "tabpfn",
}

# Used only inside an isolated TabPFN SHAP worker. The main process always uses
# ``default_tabpfn_device`` without an override.
_TABPFN_DEVICE_OVERRIDE: Optional[str] = None


def preprocessing_mode(model_name: str) -> str:
    return (
        "training-fitted ordinal coding without scaling"
        if str(model_name).lower() == "tabpfn"
        else "training-fitted standardization and one-hot encoding"
    )


def imbalance_handling_mode(model_name: str) -> str:
    if not USE_BALANCED_SAMPLE_WEIGHT:
        return "none"
    return (
        "TabPFN balance_probabilities"
        if str(model_name).lower() == "tabpfn"
        else "balanced training-sample weights"
    )


def available_model_names(candidate_names: Iterable[str]) -> List[str]:
    out = []
    missing_dependencies = []
    for name in candidate_names:
        name = str(name).lower()
        if name not in MODEL_DISPLAY_NAMES:
            continue
        dep = MODEL_DEPENDENCIES.get(name)
        if dep is None or module_available(dep):
            out.append(name)
        else:
            missing_dependencies.append((name, dep))
    if missing_dependencies and RUN_PROFILE.lower() == "full":
        details = ", ".join(
            f"{name} requires {dependency}" for name, dependency in missing_dependencies
        )
        raise RuntimeError(
            "A full paper run requires every configured candidate model. "
            f"Missing dependencies: {details}."
        )
    for name, dep in missing_dependencies:
        cprint(f"[WARN] smoke run skips model={name}; dependency not installed: {dep}")
    return out


def default_tabpfn_device() -> str:
    if _TABPFN_DEVICE_OVERRIDE is not None:
        override = str(_TABPFN_DEVICE_OVERRIDE).lower()
        if override not in {"cpu", "cuda"}:
            raise ValueError(f"Unsupported TabPFN device override: {override}")
        if override == "cpu":
            return "cpu"
        if not module_available("torch"):
            raise RuntimeError("TabPFN CUDA worker requires torch.")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("TabPFN CUDA worker cannot access a CUDA device.")
        return "cuda"
    if not module_available("torch"):
        if FORCE_TABPFN_CUDA:
            raise RuntimeError("FORCE_TABPFN_CUDA=True, but torch is not installed.")
        return "cpu"
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if FORCE_TABPFN_CUDA:
        raise RuntimeError(
            "FORCE_TABPFN_CUDA=True, but torch.cuda.is_available() is False. "
            "Please install a CUDA-enabled PyTorch build or use a CUDA environment."
        )
    return "cpu"


def build_estimator(
    model_name: str,
    seed: int = 42,
    n_jobs: int = 1,
    categorical_feature_indices: Optional[List[int]] = None,
):
    name = str(model_name).lower()
    set_global_seed(seed)

    if name == "lr":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=3000, solver="lbfgs", random_state=seed)

    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=600,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=n_jobs,
        )

    if name == "et":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_estimators=800,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=n_jobs,
        )

    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=350,
            max_depth=None,
            random_state=seed,
        )

    if name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.90,
            colsample_bytree=0.90,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=n_jobs,
            tree_method="hist",
        )

    if name == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.90,
            subsample_freq=1,
            colsample_bytree=0.90,
            objective="multiclass",
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=-1,
        )

    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=max(1, int(n_jobs)),
        )

    if name == "tabpfn":
        from tabpfn import TabPFNClassifier
        import torch

        ckpt = Path(TABPFN_LOCAL_CKPT)
        if not ckpt.exists():
            raise FileNotFoundError(f"TabPFN checkpoint was not found: {ckpt}")
        device = default_tabpfn_device()
        cprint(f"[TABPFN] using device={device} | checkpoint={ckpt}")
        tabpfn_input_options = {
            "categorical_features_indices": list(categorical_feature_indices or []),
            "balance_probabilities": True,
            "fit_mode": "fit_preprocessors",
            "memory_saving_mode": bool(TABPFN_INTERNAL_MEMORY_SAVING),
            "n_preprocessing_jobs": 1,
        }
        if TABPFN_FORCE_FP32_INFERENCE:
            tabpfn_input_options["inference_precision"] = torch.float32
        try:
            return TabPFNClassifier(
                model_path=str(ckpt),
                device=device,
                random_state=int(seed),
                **tabpfn_input_options,
            )
        except TypeError as exc:
            raise TypeError(
                "The installed TabPFN interface cannot enforce the paper-facing "
                "random seed, FP32 inference, internal memory saving, and categorical "
                "metadata. Install a compatible TabPFN 2.6 interface rather than "
                f"silently changing those settings. Original error: {exc}"
            ) from exc

    raise ValueError(f"Unsupported model name: {model_name}")


def make_balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    from sklearn.utils.class_weight import compute_sample_weight

    return np.asarray(
        compute_sample_weight(class_weight="balanced", y=np.asarray(y, dtype=int)),
        dtype=float,
    )


def fit_estimator(
    estimator, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
):
    # TabPFN versions commonly do not support sample_weight.
    if sample_weight is None or estimator.__class__.__name__.lower().startswith(
        "tabpfn"
    ):
        if (
            sample_weight is not None
            and estimator.__class__.__name__.lower().startswith("tabpfn")
        ):
            cprint(
                "[INFO] TabPFN does not accept sample_weight in this interface; "
                "fit is unweighted and balance_probabilities=True is retained "
                "for class-probability adjustment."
            )
        estimator.fit(X, y)
        return estimator
    try:
        estimator.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        cprint(
            f"[WARN] {estimator.__class__.__name__} rejected sample_weight and was fitted unweighted."
        )
        estimator.fit(X, y)
    return estimator


def predict_proba_aligned(
    estimator, X: np.ndarray, classes_expected: Iterable[int]
) -> np.ndarray:
    classes_expected = np.asarray(list(classes_expected), dtype=int)
    proba = np.asarray(estimator.predict_proba(X), dtype=float)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)
    try:
        classes = np.asarray(estimator.classes_).astype(int)
    except Exception:
        classes = np.arange(proba.shape[1], dtype=int)
    out = np.zeros((X.shape[0], len(classes_expected)), dtype=float)
    for j, cls in enumerate(classes_expected):
        pos = np.where(classes == int(cls))[0]
        if len(pos) > 0:
            out[:, j] = proba[:, pos[0]]
    return out


def predict_proba_aligned_batched(
    estimator,
    X: np.ndarray,
    classes_expected: Iterable[int],
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """Predict aligned probabilities in external row batches.

    Some estimators can depend numerically on the composition or shape of a
    prediction batch. Callers that use this helper for explanation must first
    verify equivalence to the unbatched estimator on a prespecified probe.
    """
    X = np.asarray(X)
    classes_expected = np.asarray(list(classes_expected), dtype=int)
    if X.shape[0] == 0:
        return np.zeros((0, len(classes_expected)), dtype=float)
    if batch_size is None or int(batch_size) <= 0 or X.shape[0] <= int(batch_size):
        return predict_proba_aligned(estimator, X, classes_expected)
    parts = []
    step = int(batch_size)
    for start in range(0, X.shape[0], step):
        stop = min(start + step, X.shape[0])
        parts.append(predict_proba_aligned(estimator, X[start:stop], classes_expected))
    return np.vstack(parts)


def clear_cuda_cache() -> None:
    """Release unused CUDA allocations in a healthy process."""
    if not module_available("torch"):
        return
    import torch

    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


# =============================================================================
# METRICS
# =============================================================================


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Iterable[int]
) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        f1_score,
        matthews_corrcoef,
        recall_score,
    )

    labels = list(labels)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    for cls in labels:
        out[f"recall_class_{int(cls)}"] = float(
            recall_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)[0]
            if np.any(y_true == cls)
            else 0.0
        )
    return out


def compute_probability_diagnostics(
    y_true: np.ndarray,
    proba: np.ndarray,
    labels: Iterable[int],
    n_bins: int = 10,
) -> Dict[str, float]:
    from sklearn.metrics import log_loss

    labels = np.asarray(list(labels), dtype=int)
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    proba = np.clip(proba, 1e-12, 1.0)
    proba = proba / np.maximum(proba.sum(axis=1, keepdims=True), 1e-12)

    y_onehot = np.zeros((len(y_true), len(labels)), dtype=float)
    label_to_pos = {int(c): i for i, c in enumerate(labels)}
    for i, y in enumerate(y_true):
        pos = label_to_pos.get(int(y))
        if pos is not None:
            y_onehot[i, pos] = 1.0

    pred_pos = np.argmax(proba, axis=1)
    confidence = proba[np.arange(len(proba)), pred_pos]
    correctness = (labels[pred_pos] == y_true).astype(float)

    ece = 0.0
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi >= 1.0:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        if not np.any(mask):
            continue
        ece += float(
            mask.mean() * abs(correctness[mask].mean() - confidence[mask].mean())
        )

    out = {
        "log_loss": float(log_loss(y_true, proba, labels=labels.tolist())),
        "multiclass_brier": float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1))),
        "ece": float(ece),
        "mean_confidence": float(np.mean(confidence)),
        "mean_predicted_probability_true_class": float(
            np.mean(np.sum(proba * y_onehot, axis=1))
        ),
    }

    for class_pos, class_id in enumerate(labels):
        p = proba[:, class_pos]
        y_bin = (y_true == int(class_id)).astype(float)
        class_ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            if hi >= 1.0:
                mask = (p >= lo) & (p <= hi)
            else:
                mask = (p >= lo) & (p < hi)
            if not np.any(mask):
                continue
            class_ece += float(mask.mean() * abs(y_bin[mask].mean() - p[mask].mean()))
        out[f"ece_class_{int(class_id)}"] = float(class_ece)
        out[f"brier_class_{int(class_id)}"] = float(np.mean((p - y_bin) ** 2))
    return out


def compute_calibration_curve_rows(
    y_true: np.ndarray,
    proba: np.ndarray,
    labels: Iterable[int],
    model_name: str,
    seed: int,
    n_bins: int = 10,
) -> pd.DataFrame:
    labels = np.asarray(list(labels), dtype=int)
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    proba = np.clip(proba, 1e-12, 1.0)
    proba = proba / np.maximum(proba.sum(axis=1, keepdims=True), 1e-12)
    pred_pos = np.argmax(proba, axis=1)
    confidence = proba[np.arange(len(proba)), pred_pos]
    correctness = (labels[pred_pos] == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    rows = []

    for bin_id, (lo, hi) in enumerate(zip(bins[:-1], bins[1:]), 1):
        mask = (
            (confidence >= lo) & (confidence <= hi)
            if hi >= 1.0
            else (confidence >= lo) & (confidence < hi)
        )
        rows.append(
            {
                "seed": int(seed),
                "model": str(model_name),
                "curve_type": "top_label",
                "class_id": -1,
                "bin_id": int(bin_id),
                "bin_left": float(lo),
                "bin_right": float(hi),
                "n_bin": int(mask.sum()),
                "mean_predicted_probability": float(np.mean(confidence[mask]))
                if np.any(mask)
                else np.nan,
                "observed_frequency": float(np.mean(correctness[mask]))
                if np.any(mask)
                else np.nan,
            }
        )

    for class_pos, class_id in enumerate(labels):
        p = proba[:, class_pos]
        observed = (y_true == int(class_id)).astype(float)
        for bin_id, (lo, hi) in enumerate(zip(bins[:-1], bins[1:]), 1):
            mask = (p >= lo) & (p <= hi) if hi >= 1.0 else (p >= lo) & (p < hi)
            rows.append(
                {
                    "seed": int(seed),
                    "model": str(model_name),
                    "curve_type": "classwise",
                    "class_id": int(class_id),
                    "bin_id": int(bin_id),
                    "bin_left": float(lo),
                    "bin_right": float(hi),
                    "n_bin": int(mask.sum()),
                    "mean_predicted_probability": float(np.mean(p[mask]))
                    if np.any(mask)
                    else np.nan,
                    "observed_frequency": float(np.mean(observed[mask]))
                    if np.any(mask)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_metrics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    metric_cols = [
        c
        for c in df.columns
        if c not in {"seed", "model", "model_display", "n_train", "n_test"}
        and df[c].dtype.kind in "fiu"
    ]
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row["n_rep"] = int(len(group))
        for metric in metric_cols:
            row[f"mean_{metric}"] = float(group[metric].mean())
            row[f"std_{metric}"] = (
                float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            )
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["mean_balanced_accuracy", "mean_macro_f1", "mean_accuracy"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        out["rank_balanced_accuracy"] = np.arange(1, len(out) + 1)
    return out


# =============================================================================
# EXPLANATION EVIDENCE
# =============================================================================


def compute_shap_importance(
    estimator,
    preprocessor,
    X_train_raw: pd.DataFrame,
    X_eval_raw: pd.DataFrame,
    y_eval: np.ndarray,
    schema: FeatureSchema,
    classes: Iterable[int],
    model_name: str,
    seed: int,
    background_size: Optional[int] = 120,
    eval_size: Optional[int] = None,
    max_evals: Optional[int] = None,
    predict_batch_size: Optional[int] = None,
) -> pd.DataFrame:
    if not module_available("shap"):
        cprint("[WARN] SHAP is not installed; skip SHAP evidence.")
        return pd.DataFrame()

    import shap

    codec = RawFeatureCodec.fit(X_train_raw, schema)
    X_train_mask = codec.encode(X_train_raw.reset_index(drop=True))
    X_eval_mask = codec.encode(X_eval_raw.reset_index(drop=True))
    bg_idx = sample_indices(X_train_mask.shape[0], background_size, seed)
    ev_idx = sample_indices(X_eval_mask.shape[0], eval_size, seed + 1009)
    background = X_train_mask[bg_idx]
    X_eval = X_eval_mask[ev_idx]
    y_explained = np.asarray(y_eval, dtype=int)[ev_idx]
    classes = np.asarray(list(classes), dtype=int)

    def predict_fn(x):
        raw = codec.decode(np.asarray(x, dtype=float))
        encoded = transform_raw(preprocessor, raw)
        probability = predict_proba_aligned_batched(
            estimator,
            encoded,
            classes,
            batch_size=predict_batch_size,
        )
        if str(model_name).lower() == "tabpfn":
            clear_cuda_cache()
        return probability

    kwargs = {}
    if max_evals is not None:
        kwargs["max_evals"] = int(max_evals)
    # Passing a matrix directly would let IndependentMasker silently reduce a
    # background larger than its default max_samples=100. Constructing the
    # masker explicitly guarantees that the reported background size is the
    # size actually integrated by the permutation explainer.
    masker = shap.maskers.Independent(background, max_samples=int(len(background)))
    explainer = shap.Explainer(
        predict_fn,
        masker,
        algorithm="permutation",
        seed=int(seed),
    )
    result = explainer(X_eval, **kwargs)
    values = np.asarray(result.values, dtype=float)

    if values.ndim == 2:
        values = values[:, :, None]
    if values.ndim != 3:
        raise RuntimeError(f"Unexpected SHAP value shape: {values.shape}")
    if values.shape[1] == len(classes) and values.shape[2] == len(schema.feature_cols):
        values = np.transpose(values, (0, 2, 1))
    if values.shape[1] != len(schema.feature_cols) or values.shape[2] != len(classes):
        raise RuntimeError(
            "SHAP output does not match the raw feature and class dimensions: "
            f"values={values.shape}, features={len(schema.feature_cols)}, classes={len(classes)}."
        )

    rows = []
    for class_pos, class_id in enumerate(classes):
        balanced_weights = class_balanced_evaluation_weights(y_explained, int(class_id))
        focal_mask = np.asarray(y_explained, dtype=int) == int(class_id)
        other_mask = ~focal_mask
        for feature_pos, feature in enumerate(schema.feature_cols):
            signed = values[:, feature_pos, class_pos]
            balanced_abs = weighted_mean(np.abs(signed), balanced_weights)
            balanced_signed = weighted_mean(signed, balanced_weights)
            prevalence_abs = float(np.mean(np.abs(signed)))
            prevalence_signed = float(np.mean(signed))
            focal_abs = (
                float(np.mean(np.abs(signed[focal_mask])))
                if np.any(focal_mask)
                else np.nan
            )
            other_abs = (
                float(np.mean(np.abs(signed[other_mask])))
                if np.any(other_mask)
                else np.nan
            )
            focal_signed = (
                float(np.mean(signed[focal_mask]))
                if np.any(focal_mask)
                else np.nan
            )
            other_signed = (
                float(np.mean(signed[other_mask]))
                if np.any(other_mask)
                else np.nan
            )
            rows.append(
                {
                    "seed": int(seed),
                    "model": str(model_name),
                    "class_id": int(class_id),
                    "feature": str(feature),
                    # The legacy names retain compatibility with plotting and
                    # negative-control code; both now denote the prespecified
                    # class-balanced estimand.
                    "mean_abs_shap": balanced_abs,
                    "mean_signed_shap": balanced_signed,
                    "balanced_mean_abs_shap": balanced_abs,
                    "balanced_mean_signed_shap": balanced_signed,
                    "prevalence_mean_abs_shap": prevalence_abs,
                    "prevalence_mean_signed_shap": prevalence_signed,
                    # Sufficient OOF primitives for exact class-balanced
                    # cross-fold pooling. These are retained separately so
                    # fold-level class proportions cannot alter the statistic.
                    "focal_mean_abs_shap": focal_abs,
                    "other_mean_abs_shap": other_abs,
                    "focal_mean_signed_shap": focal_signed,
                    "other_mean_signed_shap": other_signed,
                    "n_focal": int(np.sum(focal_mask)),
                    "n_other": int(np.sum(other_mask)),
                    "evaluation_measure": "class_balanced_one_vs_rest",
                    "n_eval": int(len(ev_idx)),
                    "n_background": int(len(bg_idx)),
                    "background_size_requested": (
                        int(background_size)
                        if background_size is not None
                        else int(X_train_mask.shape[0])
                    ),
                    "n_permutation_cycles": int(
                        max(1, int(max_evals) // (2 * len(schema.feature_cols) + 1))
                    )
                    if max_evals is not None
                    else np.nan,
                    "prediction_batch_size": (
                        int(predict_batch_size)
                        if predict_batch_size is not None
                        else np.nan
                    ),
                    "execution_mode": "in_process",
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["rank"] = out.groupby(["seed", "model", "class_id"])["mean_abs_shap"].rank(
            ascending=False, method="average"
        )
    return out


def _target_shap_background_seed(
    phase_scope: str,
    outer_partition: int,
    inner_fold: int,
) -> int:
    """Return a model-independent seed for shared raw background row positions."""
    return stable_seed_from_key(
        "target_shap_background_rows_v1",
        str(phase_scope),
        int(outer_partition),
        int(inner_fold),
    )


def _keyed_master_permutation(
    plan_version: str,
    phase_scope: str,
    outer_partition: int,
    inner_fold: int,
    stable_record_id: int,
    permutation_cycle: int,
    feature_count: int,
) -> np.ndarray:
    """Generate one full-feature common-random-number permutation."""
    seed = stable_seed_from_key(
        str(plan_version),
        str(phase_scope),
        int(outer_partition),
        int(inner_fold),
        int(stable_record_id),
        int(permutation_cycle),
    )
    return np.random.default_rng(seed).permutation(int(feature_count))


def _coalition_mask(indices: Iterable[int]) -> int:
    mask = 0
    for index in indices:
        mask |= 1 << int(index)
    return int(mask)


def _mask_positions(mask: int, feature_count: int) -> np.ndarray:
    return np.asarray(
        [bool((int(mask) >> index) & 1) for index in range(int(feature_count))],
        dtype=bool,
    )


def _make_raw_probability_function(
    estimator,
    preprocessor,
    codec: RawFeatureCodec,
    classes: np.ndarray,
):
    def predict_fn(masked_rows: np.ndarray) -> np.ndarray:
        raw = codec.decode(np.asarray(masked_rows, dtype=float))
        encoded = transform_raw(preprocessor, raw)
        return predict_proba_aligned(
            estimator,
            encoded,
            np.asarray(classes, dtype=int),
        )

    return predict_fn


def _evaluate_coalition_states(
    state_keys: Sequence[Tuple[int, int]],
    X_eval_mask: np.ndarray,
    background: np.ndarray,
    predict_fn,
    class_count: int,
    prediction_batch_rows: int,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[str, int]]:
    """Evaluate unique Independent-masker coalition values in fixed batches."""
    ordered_states = sorted(set((int(i), int(mask)) for i, mask in state_keys))
    if not ordered_states:
        return {}, {
            "unique_mask_states": 0,
            "background_rows_predicted": 0,
            "python_prediction_calls": 0,
        }
    background_rows = int(len(background))
    if background_rows < 1:
        raise RuntimeError("Target-coordinate SHAP received an empty background.")
    states_per_call = max(1, int(prediction_batch_rows) // background_rows)
    values: Dict[Tuple[int, int], np.ndarray] = {}
    n_calls = 0
    n_rows = 0
    feature_count = int(X_eval_mask.shape[1])

    for start in range(0, len(ordered_states), states_per_call):
        chunk = ordered_states[start : start + states_per_call]
        masked_blocks = []
        for record_pos, coalition in chunk:
            rows = np.asarray(background, dtype=float).copy()
            selected = _mask_positions(coalition, feature_count)
            if np.any(selected):
                rows[:, selected] = X_eval_mask[int(record_pos), selected]
            masked_blocks.append(rows)
        masked_batch = np.concatenate(masked_blocks, axis=0)
        prediction = np.asarray(predict_fn(masked_batch), dtype=float)
        expected_shape = (len(chunk) * background_rows, int(class_count))
        if prediction.shape != expected_shape:
            raise RuntimeError(
                "Coalition prediction shape mismatch: "
                f"expected={expected_shape}, observed={prediction.shape}."
            )
        prediction = prediction.reshape(
            len(chunk),
            background_rows,
            int(class_count),
        )
        means = prediction.mean(axis=1)
        for key, value in zip(chunk, means):
            values[key] = np.asarray(value, dtype=float)
        n_calls += 1
        n_rows += int(len(masked_batch))

    return values, {
        "unique_mask_states": int(len(ordered_states)),
        "background_rows_predicted": int(n_rows),
        "python_prediction_calls": int(n_calls),
    }


def _target_coordinate_shap_values(
    estimator,
    preprocessor,
    codec: RawFeatureCodec,
    X_eval_mask: np.ndarray,
    background: np.ndarray,
    stable_record_ids: np.ndarray,
    classes: np.ndarray,
    target_positions: Sequence[int],
    plan_version: str,
    phase_scope: str,
    outer_partition: int,
    inner_fold: int,
    full_path_max_evals: int,
    prediction_batch_rows: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Estimate requested coordinates while permuting all original features."""
    import shap

    target_positions = [int(position) for position in target_positions]
    feature_count = int(X_eval_mask.shape[1])
    class_count = int(len(classes))
    n_eval = int(len(X_eval_mask))
    if len(stable_record_ids) != n_eval:
        raise RuntimeError("Stable record identifiers do not match SHAP rows.")
    if len(set(target_positions)) != len(target_positions):
        raise ValueError("Target-coordinate SHAP received duplicate features.")
    if any(position < 0 or position >= feature_count for position in target_positions):
        raise IndexError("Target-coordinate SHAP feature position is out of range.")

    masker = shap.maskers.Independent(
        np.asarray(background, dtype=float),
        max_samples=int(len(background)),
    )
    predict_fn = _make_raw_probability_function(
        estimator,
        preprocessor,
        codec,
        np.asarray(classes, dtype=int),
    )
    terms: Dict[Tuple[int, int], List[Tuple[int, int, int, int]]] = {}
    state_keys: List[Tuple[int, int]] = []
    permutations_by_record: Dict[int, int] = {}
    varying_by_record: Dict[int, int] = {}
    requested_state_count = 0

    for record_pos, row in enumerate(np.asarray(X_eval_mask, dtype=float)):
        invariants = np.asarray(masker.invariants(row), dtype=bool)
        varying = np.flatnonzero(np.any(~invariants, axis=0)).astype(int)
        varying_set = set(int(value) for value in varying)
        n_varying = int(len(varying))
        varying_by_record[int(record_pos)] = n_varying
        if n_varying < 1:
            permutations_by_record[int(record_pos)] = 0
            continue
        n_permutations = max(
            1,
            int(full_path_max_evals) // (2 * n_varying + 1),
        )
        permutations_by_record[int(record_pos)] = int(n_permutations)
        stable_id = int(stable_record_ids[int(record_pos)])

        for cycle in range(int(n_permutations)):
            master = _keyed_master_permutation(
                plan_version=plan_version,
                phase_scope=phase_scope,
                outer_partition=outer_partition,
                inner_fold=inner_fold,
                stable_record_id=stable_id,
                permutation_cycle=cycle,
                feature_count=feature_count,
            )
            restricted = [
                int(value) for value in master if int(value) in varying_set
            ]
            order_position = {
                feature_pos: order for order, feature_pos in enumerate(restricted)
            }
            for target_pos in target_positions:
                if target_pos not in varying_set:
                    continue
                pivot = int(order_position[target_pos])
                predecessors = restricted[:pivot]
                successors = restricted[pivot + 1 :]
                s_mask = _coalition_mask(predecessors)
                sj_mask = int(s_mask | (1 << target_pos))
                r_mask = _coalition_mask(successors)
                rj_mask = int(r_mask | (1 << target_pos))
                term_key = (int(record_pos), int(target_pos))
                terms.setdefault(term_key, []).append(
                    (s_mask, sj_mask, r_mask, rj_mask)
                )
                for coalition in (s_mask, sj_mask, r_mask, rj_mask):
                    state_keys.append((int(record_pos), int(coalition)))
                requested_state_count += 4

    state_values, evaluation_diagnostics = _evaluate_coalition_states(
        state_keys=state_keys,
        X_eval_mask=np.asarray(X_eval_mask, dtype=float),
        background=np.asarray(background, dtype=float),
        predict_fn=predict_fn,
        class_count=class_count,
        prediction_batch_rows=int(prediction_batch_rows),
    )
    values = np.zeros(
        (n_eval, len(target_positions), class_count),
        dtype=float,
    )
    target_column = {
        int(position): column for column, position in enumerate(target_positions)
    }
    for (record_pos, target_pos), coordinate_terms in terms.items():
        contributions = []
        for s_mask, sj_mask, r_mask, rj_mask in coordinate_terms:
            delta_forward = (
                state_values[(record_pos, sj_mask)]
                - state_values[(record_pos, s_mask)]
            )
            delta_reverse = (
                state_values[(record_pos, rj_mask)]
                - state_values[(record_pos, r_mask)]
            )
            contributions.append(0.5 * (delta_forward + delta_reverse))
        values[
            int(record_pos),
            int(target_column[int(target_pos)]),
            :,
        ] = np.mean(np.asarray(contributions, dtype=float), axis=0)

    permutation_counts = np.asarray(
        list(permutations_by_record.values()),
        dtype=float,
    )
    varying_counts = np.asarray(
        list(varying_by_record.values()),
        dtype=float,
    )
    diagnostics: Dict[str, Any] = {
        "target_shap_contract_version": TARGET_SHAP_CONTRACT_VERSION,
        "permutation_plan_version": str(plan_version),
        "phase_scope": str(phase_scope),
        "full_path_max_evals_budget": int(full_path_max_evals),
        "target_mask_states_requested": int(requested_state_count),
        "target_unique_mask_states": int(
            evaluation_diagnostics["unique_mask_states"]
        ),
        "target_background_rows_predicted": int(
            evaluation_diagnostics["background_rows_predicted"]
        ),
        "target_python_prediction_calls": int(
            evaluation_diagnostics["python_prediction_calls"]
        ),
        "min_varying_features": int(np.min(varying_counts))
        if len(varying_counts)
        else 0,
        "mean_varying_features": float(np.mean(varying_counts))
        if len(varying_counts)
        else 0.0,
        "max_varying_features": int(np.max(varying_counts))
        if len(varying_counts)
        else 0,
        "min_permutations": int(np.min(permutation_counts))
        if len(permutation_counts)
        else 0,
        "mean_permutations": float(np.mean(permutation_counts))
        if len(permutation_counts)
        else 0.0,
        "max_permutations": int(np.max(permutation_counts))
        if len(permutation_counts)
        else 0,
    }
    return values, diagnostics


def _full_reference_shap_values(
    estimator,
    preprocessor,
    codec: RawFeatureCodec,
    X_eval_mask: np.ndarray,
    background: np.ndarray,
    stable_record_ids: np.ndarray,
    classes: np.ndarray,
    plan_version: str,
    phase_scope: str,
    outer_partition: int,
    inner_fold: int,
    full_path_max_evals: int,
    prediction_batch_rows: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Custom complete Permutation SHAP path used only for hard preflight."""
    import shap

    feature_count = int(X_eval_mask.shape[1])
    class_count = int(len(classes))
    n_eval = int(len(X_eval_mask))
    masker = shap.maskers.Independent(
        np.asarray(background, dtype=float),
        max_samples=int(len(background)),
    )
    predict_fn = _make_raw_probability_function(
        estimator,
        preprocessor,
        codec,
        np.asarray(classes, dtype=int),
    )
    state_keys: List[Tuple[int, int]] = []
    path_terms: Dict[
        Tuple[int, int],
        List[Tuple[int, int, int, int]],
    ] = {}
    requested_states = 0

    for record_pos, row in enumerate(np.asarray(X_eval_mask, dtype=float)):
        invariants = np.asarray(masker.invariants(row), dtype=bool)
        varying = np.flatnonzero(np.any(~invariants, axis=0)).astype(int)
        varying_set = set(int(value) for value in varying)
        n_varying = int(len(varying))
        if n_varying < 1:
            continue
        n_permutations = max(
            1,
            int(full_path_max_evals) // (2 * n_varying + 1),
        )
        stable_id = int(stable_record_ids[int(record_pos)])
        for cycle in range(int(n_permutations)):
            master = _keyed_master_permutation(
                plan_version=plan_version,
                phase_scope=phase_scope,
                outer_partition=outer_partition,
                inner_fold=inner_fold,
                stable_record_id=stable_id,
                permutation_cycle=cycle,
                feature_count=feature_count,
            )
            restricted = [
                int(value) for value in master if int(value) in varying_set
            ]
            prefix_masks = [0]
            current = 0
            for feature_pos in restricted:
                current |= 1 << int(feature_pos)
                prefix_masks.append(int(current))
            suffix_masks = [0] * (len(restricted) + 1)
            current = 0
            for reverse_pos in range(len(restricted) - 1, -1, -1):
                current |= 1 << int(restricted[reverse_pos])
                suffix_masks[reverse_pos] = int(current)
            for order, feature_pos in enumerate(restricted):
                path_terms.setdefault(
                    (int(record_pos), int(feature_pos)),
                    [],
                ).append(
                    (
                        int(prefix_masks[order]),
                        int(prefix_masks[order + 1]),
                        int(suffix_masks[order + 1]),
                        int(suffix_masks[order]),
                    )
                )
            for coalition in set(prefix_masks + suffix_masks):
                state_keys.append((int(record_pos), int(coalition)))
            requested_states += int(2 * n_varying + 1)

    state_values, evaluation_diagnostics = _evaluate_coalition_states(
        state_keys=state_keys,
        X_eval_mask=np.asarray(X_eval_mask, dtype=float),
        background=np.asarray(background, dtype=float),
        predict_fn=predict_fn,
        class_count=class_count,
        prediction_batch_rows=int(prediction_batch_rows),
    )
    values = np.zeros((n_eval, feature_count, class_count), dtype=float)
    for (record_pos, feature_pos), coordinate_terms in path_terms.items():
        contributions = []
        for s_mask, sj_mask, r_mask, rj_mask in coordinate_terms:
            delta_forward = (
                state_values[(record_pos, sj_mask)]
                - state_values[(record_pos, s_mask)]
            )
            delta_reverse = (
                state_values[(record_pos, rj_mask)]
                - state_values[(record_pos, r_mask)]
            )
            contributions.append(0.5 * (delta_forward + delta_reverse))
        values[int(record_pos), int(feature_pos), :] = np.mean(
            np.asarray(contributions, dtype=float),
            axis=0,
        )
    return values, {
        "full_reference_mask_states_requested": int(requested_states),
        "full_reference_unique_mask_states": int(
            evaluation_diagnostics["unique_mask_states"]
        ),
        "full_reference_background_rows_predicted": int(
            evaluation_diagnostics["background_rows_predicted"]
        ),
        "full_reference_python_prediction_calls": int(
            evaluation_diagnostics["python_prediction_calls"]
        ),
    }


def _shap_values_to_importance_table(
    values: np.ndarray,
    y_explained: np.ndarray,
    classes: np.ndarray,
    feature_names: Sequence[str],
    seed: int,
    model_name: str,
    n_background: int,
    execution_mode: str,
    diagnostics: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    values = np.asarray(values, dtype=float)
    y_explained = np.asarray(y_explained, dtype=int)
    classes = np.asarray(classes, dtype=int)
    expected_shape = (
        len(y_explained),
        len(feature_names),
        len(classes),
    )
    if values.shape != expected_shape:
        raise RuntimeError(
            "SHAP values do not match evaluation, feature, and class axes: "
            f"expected={expected_shape}, observed={values.shape}."
        )
    common_diagnostics = dict(diagnostics or {})
    rows: List[Dict[str, Any]] = []
    for class_pos, class_id in enumerate(classes):
        balanced_weights = class_balanced_evaluation_weights(
            y_explained,
            int(class_id),
        )
        focal_mask = y_explained == int(class_id)
        other_mask = ~focal_mask
        for feature_pos, feature in enumerate(feature_names):
            signed = values[:, int(feature_pos), int(class_pos)]
            balanced_abs = weighted_mean(np.abs(signed), balanced_weights)
            balanced_signed = weighted_mean(signed, balanced_weights)
            row: Dict[str, Any] = {
                "seed": int(seed),
                "model": str(model_name),
                "class_id": int(class_id),
                "feature": str(feature),
                "mean_abs_shap": balanced_abs,
                "mean_signed_shap": balanced_signed,
                "balanced_mean_abs_shap": balanced_abs,
                "balanced_mean_signed_shap": balanced_signed,
                "prevalence_mean_abs_shap": float(np.mean(np.abs(signed))),
                "prevalence_mean_signed_shap": float(np.mean(signed)),
                "focal_mean_abs_shap": float(np.mean(np.abs(signed[focal_mask])))
                if np.any(focal_mask)
                else np.nan,
                "other_mean_abs_shap": float(np.mean(np.abs(signed[other_mask])))
                if np.any(other_mask)
                else np.nan,
                "focal_mean_signed_shap": float(np.mean(signed[focal_mask]))
                if np.any(focal_mask)
                else np.nan,
                "other_mean_signed_shap": float(np.mean(signed[other_mask]))
                if np.any(other_mask)
                else np.nan,
                "n_focal": int(np.sum(focal_mask)),
                "n_other": int(np.sum(other_mask)),
                "evaluation_measure": "class_balanced_one_vs_rest",
                "n_eval": int(len(y_explained)),
                "n_background": int(n_background),
                "execution_mode": str(execution_mode),
            }
            row.update(common_diagnostics)
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["rank"] = out.groupby(["seed", "model", "class_id"])[
            "mean_abs_shap"
        ].rank(ascending=False, method="average")
    return out


def compute_target_coordinate_shap_importance(
    estimator,
    preprocessor,
    X_train_raw: pd.DataFrame,
    X_eval_raw: pd.DataFrame,
    y_eval: np.ndarray,
    schema: FeatureSchema,
    classes: Iterable[int],
    model_name: str,
    target_features: Sequence[str],
    phase_scope: str,
    outer_partition: int,
    inner_fold: int,
    plan_version: str = TARGET_SHAP_PERMUTATION_PLAN_VERSION,
) -> pd.DataFrame:
    """Compute target-coordinate SHAP for calibration on every evaluation row."""
    if not module_available("shap"):
        raise RuntimeError("SHAP is required for target-coordinate calibration.")
    if STABLE_RECORD_ID_COL not in X_eval_raw.columns:
        raise RuntimeError(
            f"Evaluation data are missing {STABLE_RECORD_ID_COL}."
        )
    requested = [str(feature) for feature in target_features]
    feature_to_position = {
        str(feature): position
        for position, feature in enumerate(schema.feature_cols)
    }
    unknown = [feature for feature in requested if feature not in feature_to_position]
    if unknown:
        raise KeyError(f"Unknown target-coordinate SHAP features: {unknown}")

    codec = RawFeatureCodec.fit(X_train_raw, schema)
    X_train_mask = codec.encode(X_train_raw.reset_index(drop=True))
    X_eval_mask = codec.encode(X_eval_raw.reset_index(drop=True))
    background_seed = _target_shap_background_seed(
        phase_scope=phase_scope,
        outer_partition=outer_partition,
        inner_fold=inner_fold,
    )
    background_indices = sample_indices(
        len(X_train_mask),
        int(SHAP_BACKGROUND_SIZE),
        background_seed,
    )
    background = X_train_mask[background_indices]
    classes_array = np.asarray(list(classes), dtype=int)
    stable_record_ids = (
        pd.to_numeric(
            X_eval_raw[STABLE_RECORD_ID_COL],
            errors="raise",
        )
        .astype(np.int64)
        .to_numpy()
    )
    full_path_budget = int(SHAP_PERMUTATION_CYCLES) * (
        2 * len(schema.feature_cols) + 1
    )
    values, diagnostics = _target_coordinate_shap_values(
        estimator=estimator,
        preprocessor=preprocessor,
        codec=codec,
        X_eval_mask=X_eval_mask,
        background=background,
        stable_record_ids=stable_record_ids,
        classes=classes_array,
        target_positions=[
            int(feature_to_position[feature]) for feature in requested
        ],
        plan_version=str(plan_version),
        phase_scope=str(phase_scope),
        outer_partition=int(outer_partition),
        inner_fold=int(inner_fold),
        full_path_max_evals=int(full_path_budget),
        prediction_batch_rows=int(TARGET_SHAP_PREDICTION_BATCH_ROWS),
    )
    diagnostics.update(
        {
            "background_size_requested": int(SHAP_BACKGROUND_SIZE),
            "background_index_digest": array_digest(
                np.asarray(background_indices, dtype=np.int64)
            ),
            "prediction_batch_rows": int(TARGET_SHAP_PREDICTION_BATCH_ROWS),
        }
    )
    return _shap_values_to_importance_table(
        values=values,
        y_explained=np.asarray(y_eval, dtype=int),
        classes=classes_array,
        feature_names=requested,
        seed=background_seed,
        model_name=model_name,
        n_background=len(background_indices),
        execution_mode="target_coordinate_permutation_shap",
        diagnostics=diagnostics,
    )


def run_target_shap_equivalence_audit(
    estimator,
    preprocessor,
    X_train_raw: pd.DataFrame,
    X_eval_raw: pd.DataFrame,
    schema: FeatureSchema,
    classes: Sequence[int],
    model_name: str,
    outer_partition: int,
    audit_root: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Verify custom full and target-coordinate paths before formal calibration."""
    audit_root = Path(audit_root)
    audit_root.mkdir(parents=True, exist_ok=True)
    safe_model = _safe_path_component(model_name, limit=28)
    hard_path = audit_root / f"{safe_model}_target_shap_equivalence.csv"
    sensitivity_path = audit_root / f"{safe_model}_plan_sensitivity.csv"
    marker_path = audit_root / f"{safe_model}_complete.json"
    audit_fingerprint = stable_json_digest(
        {
            "contract": TARGET_SHAP_CONTRACT_VERSION,
            "primary_plan": TARGET_SHAP_PERMUTATION_PLAN_VERSION,
            "alternative_plan": TARGET_SHAP_SENSITIVITY_PLAN_VERSION,
            "model": str(model_name),
            "features": list(schema.feature_cols),
            "background": int(SHAP_BACKGROUND_SIZE),
            "cycles": int(SHAP_PERMUTATION_CYCLES),
            "records": int(TARGET_SHAP_EQUIVALENCE_RECORDS),
            "batch_rows": int(TARGET_SHAP_PREDICTION_BATCH_ROWS),
            "script_sha256": file_sha256(str(Path(__file__).resolve())),
            "package_versions": {
                package: package_version(package)
                for package in [
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "shap",
                ]
            },
        }
    )
    if marker_path.exists() and hard_path.exists() and sensitivity_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("fingerprint") == audit_fingerprint:
            hard = pd.read_csv(hard_path)
            passed = hard["passed"].astype(str).str.lower().eq("true")
            if not hard.empty and bool(passed.all()):
                return hard, pd.read_csv(sensitivity_path)

    if STABLE_RECORD_ID_COL not in X_eval_raw.columns:
        raise RuntimeError(
            f"Equivalence audit requires {STABLE_RECORD_ID_COL}."
        )
    codec = RawFeatureCodec.fit(X_train_raw, schema)
    train_mask = codec.encode(X_train_raw.reset_index(drop=True))
    eval_indices = sample_indices(
        len(X_eval_raw),
        min(int(TARGET_SHAP_EQUIVALENCE_RECORDS), len(X_eval_raw)),
        stable_seed_from_key(
            "target_shap_equivalence_rows_v1",
            str(model_name),
        ),
    )
    eval_raw = X_eval_raw.iloc[eval_indices].reset_index(drop=True)
    eval_mask = codec.encode(eval_raw)
    stable_ids = (
        pd.to_numeric(eval_raw[STABLE_RECORD_ID_COL], errors="raise")
        .astype(np.int64)
        .to_numpy()
    )
    background_seed = _target_shap_background_seed(
        "equivalence_audit",
        int(outer_partition),
        -1,
    )
    background_indices = sample_indices(
        len(train_mask),
        int(SHAP_BACKGROUND_SIZE),
        background_seed,
    )
    background = train_mask[background_indices]
    class_array = np.asarray(classes, dtype=int)
    full_budget = int(SHAP_PERMUTATION_CYCLES) * (
        2 * len(schema.feature_cols) + 1
    )

    predict_fn = _make_raw_probability_function(
        estimator,
        preprocessor,
        codec,
        class_array,
    )
    probe = np.asarray(eval_mask, dtype=float)
    canonical = np.asarray(predict_fn(probe), dtype=float)
    midpoint = max(1, len(probe) // 2)
    production = np.concatenate(
        [
            np.asarray(predict_fn(probe[:midpoint]), dtype=float),
            np.asarray(predict_fn(probe[midpoint:]), dtype=float)
            if midpoint < len(probe)
            else np.zeros((0, len(class_array)), dtype=float),
        ],
        axis=0,
    )
    if not np.allclose(
        canonical,
        production,
        atol=float(TARGET_SHAP_EQUIVALENCE_ATOL),
        rtol=float(TARGET_SHAP_EQUIVALENCE_RTOL),
    ):
        raise RuntimeError(
            f"Prediction batching changes {model_name} probabilities; "
            "target-coordinate SHAP cannot use the production batching contract."
        )

    full_values, full_diagnostics = _full_reference_shap_values(
        estimator=estimator,
        preprocessor=preprocessor,
        codec=codec,
        X_eval_mask=eval_mask,
        background=background,
        stable_record_ids=stable_ids,
        classes=class_array,
        plan_version=TARGET_SHAP_PERMUTATION_PLAN_VERSION,
        phase_scope="equivalence_audit",
        outer_partition=int(outer_partition),
        inner_fold=-1,
        full_path_max_evals=full_budget,
        prediction_batch_rows=int(TARGET_SHAP_PREDICTION_BATCH_ROWS),
    )
    target_values, target_diagnostics = _target_coordinate_shap_values(
        estimator=estimator,
        preprocessor=preprocessor,
        codec=codec,
        X_eval_mask=eval_mask,
        background=background,
        stable_record_ids=stable_ids,
        classes=class_array,
        target_positions=list(range(len(schema.feature_cols))),
        plan_version=TARGET_SHAP_PERMUTATION_PLAN_VERSION,
        phase_scope="equivalence_audit",
        outer_partition=int(outer_partition),
        inner_fold=-1,
        full_path_max_evals=full_budget,
        prediction_batch_rows=int(TARGET_SHAP_PREDICTION_BATCH_ROWS),
    )
    passed_array = np.isclose(
        full_values,
        target_values,
        atol=float(TARGET_SHAP_EQUIVALENCE_ATOL),
        rtol=float(TARGET_SHAP_EQUIVALENCE_RTOL),
    )
    hard_rows = []
    epsilon = np.finfo(float).eps
    for row_pos, stable_id in enumerate(stable_ids):
        for feature_pos, feature in enumerate(schema.feature_cols):
            for class_pos, class_id in enumerate(class_array):
                phi_full = float(full_values[row_pos, feature_pos, class_pos])
                phi_target = float(target_values[row_pos, feature_pos, class_pos])
                absolute_error = abs(phi_full - phi_target)
                relative_error = absolute_error / max(abs(phi_full), epsilon)
                hard_rows.append(
                    {
                        "model": str(model_name),
                        "stable_record_id": int(stable_id),
                        "class_id": int(class_id),
                        "feature": str(feature),
                        "phi_full_reference": phi_full,
                        "phi_target": phi_target,
                        "abs_error": float(absolute_error),
                        "rel_error": float(relative_error),
                        "passed": bool(
                            passed_array[row_pos, feature_pos, class_pos]
                        ),
                        "prediction_batch_max_abs_delta": float(
                            np.max(np.abs(canonical - production))
                        ),
                        **full_diagnostics,
                        **target_diagnostics,
                    }
                )
    hard = pd.DataFrame(hard_rows)
    hard.to_csv(hard_path, index=False, encoding="utf-8-sig")
    if hard.empty or not bool(hard["passed"].astype(bool).all()):
        raise RuntimeError(
            f"Target-coordinate SHAP equivalence failed for model={model_name}. "
            f"See {hard_path}."
        )

    sensitivity_rows: List[Dict[str, Any]] = []
    if ENABLE_TARGET_SHAP_PLAN_SENSITIVITY_AUDIT:
        alternative_values, _ = _target_coordinate_shap_values(
            estimator=estimator,
            preprocessor=preprocessor,
            codec=codec,
            X_eval_mask=eval_mask,
            background=background,
            stable_record_ids=stable_ids,
            classes=class_array,
            target_positions=list(range(len(schema.feature_cols))),
            plan_version=TARGET_SHAP_SENSITIVITY_PLAN_VERSION,
            phase_scope="equivalence_audit",
            outer_partition=int(outer_partition),
            inner_fold=-1,
            full_path_max_evals=full_budget,
            prediction_batch_rows=int(TARGET_SHAP_PREDICTION_BATCH_ROWS),
        )
        for class_pos, class_id in enumerate(class_array):
            primary = np.mean(np.abs(target_values[:, :, class_pos]), axis=0)
            alternative = np.mean(
                np.abs(alternative_values[:, :, class_pos]),
                axis=0,
            )
            primary_series = pd.Series(primary)
            alternative_series = pd.Series(alternative)
            sensitivity_rows.append(
                {
                    "model": str(model_name),
                    "class_id": int(class_id),
                    "plan_primary": TARGET_SHAP_PERMUTATION_PLAN_VERSION,
                    "plan_alternative": TARGET_SHAP_SENSITIVITY_PLAN_VERSION,
                    "n_records": int(len(eval_raw)),
                    "spearman": float(
                        primary_series.corr(
                            alternative_series,
                            method="spearman",
                        )
                    ),
                    "kendall": float(
                        primary_series.corr(
                            alternative_series,
                            method="kendall",
                        )
                    ),
                    "normalized_l1": float(
                        np.sum(np.abs(primary - alternative))
                        / max(np.sum(np.abs(primary)), epsilon)
                    ),
                    "max_abs_difference": float(
                        np.max(np.abs(primary - alternative))
                    ),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(
        sensitivity_path,
        index=False,
        encoding="utf-8-sig",
    )
    atomic_json_dump(
        {
            "fingerprint": audit_fingerprint,
            "model": str(model_name),
            "hard_equivalence_passed": True,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        },
        marker_path,
    )
    return hard, sensitivity


def tabpfn_shap_batch_schedule(initial_size: int, minimum_size: int) -> List[int]:
    """Return a deterministic halving schedule for CUDA-memory retries."""
    initial = int(initial_size)
    minimum = int(minimum_size)
    if initial <= 0 or minimum <= 0:
        raise ValueError("TabPFN SHAP batch sizes must be positive integers.")
    if minimum > initial:
        minimum = initial
    sizes = []
    current = initial
    while current >= minimum:
        if current not in sizes:
            sizes.append(current)
        if current == minimum:
            break
        current = max(minimum, current // 2)
    return sizes


def is_retryable_cuda_failure(message: str) -> bool:
    text = str(message).lower()
    markers = [
        "cuda error",
        "illegal memory access",
        "out of memory",
        "cudaerror",
        "cublas",
        "cudnn",
    ]
    return any(marker in text for marker in markers)


def is_retryable_native_worker_exit(exit_code: Optional[int]) -> bool:
    """Recognize native CUDA/driver crashes without masking IPC failures."""
    if exit_code is None:
        return False
    code = int(exit_code)
    unsigned_code = code % (1 << 32)
    # 0xC0000005 is the Windows access-violation code observed when a native
    # CUDA extension terminates the spawned worker. Negative representation is
    # normalized above. Other nonzero exits remain non-retryable unless the
    # worker wrote an explicit CUDA error to its status record.
    return unsigned_code == 0xC0000005


class TabPFNExternalBatchMismatch(RuntimeError):
    """Raised when external row batching changes TabPFN probabilities."""


def validate_tabpfn_prediction_semantics(
    unbatched_first: np.ndarray,
    unbatched_second: np.ndarray,
    externally_batched: Optional[np.ndarray],
    tolerance: float,
) -> Tuple[float, float]:
    """Validate repeatability first, then optional external-batch equivalence."""
    first = np.asarray(unbatched_first, dtype=float)
    second = np.asarray(unbatched_second, dtype=float)
    if first.shape != second.shape:
        raise RuntimeError(
            "Repeated TabPFN predictions changed shape: "
            f"first={first.shape}, second={second.shape}."
        )
    repeatability_delta = float(np.max(np.abs(first - second)))
    if not np.allclose(first, second, atol=tolerance, rtol=tolerance):
        raise RuntimeError(
            "Repeated same-shape TabPFN predictions are not reproducible "
            "enough for model-agnostic SHAP under the configured FP32 mode. "
            f"max_abs_delta={repeatability_delta:.8g}, tolerance={tolerance:g}."
        )

    batch_equivalence_delta = np.nan
    if externally_batched is not None:
        batched = np.asarray(externally_batched, dtype=float)
        if first.shape != batched.shape:
            raise TabPFNExternalBatchMismatch(
                "External prediction batching changed the TabPFN probability "
                f"shape: unbatched={first.shape}, batched={batched.shape}."
            )
        batch_equivalence_delta = float(np.max(np.abs(first - batched)))
        if not np.allclose(first, batched, atol=tolerance, rtol=tolerance):
            raise TabPFNExternalBatchMismatch(
                "External prediction batching changes TabPFN probabilities. "
                "The worker must explain the unbatched prediction function "
                "and rely on TabPFN's internal memory-saving mode. "
                f"max_abs_delta={batch_equivalence_delta:.8g}, "
                f"tolerance={tolerance:g}."
            )
    return repeatability_delta, batch_equivalence_delta


def _tabpfn_shap_worker(payload: Dict[str, object]) -> None:
    """Fit and explain TabPFN inside a disposable CUDA process."""
    status_path = Path(str(payload["status_path"]))
    result_path = Path(str(payload["result_path"]))
    status: Dict[str, object]
    estimator = None
    status_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(
        {
            "status": "running",
            "pid": int(os.getpid()),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "device": str(payload.get("device", "unknown")),
            "prediction_mode": str(payload.get("prediction_mode", "unknown")),
            "predict_batch_size": int(payload.get("predict_batch_size", 0)),
        },
        status_path,
    )
    try:
        # These variables are read when torch first initializes CUDA in this
        # spawned process. They improve diagnostics and allocator behavior but
        # do not alter model predictions or the SHAP estimand.
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        global _TABPFN_DEVICE_OVERRIDE
        _TABPFN_DEVICE_OVERRIDE = str(payload["device"])
        seed = int(payload["seed"])
        set_global_seed(seed)

        X_train_raw = payload["X_train_raw"]
        X_eval_raw = payload["X_eval_raw"]
        y_train = np.asarray(payload["y_train"], dtype=int)
        y_eval = np.asarray(payload["y_eval"], dtype=int)
        schema_payload = payload["schema"]
        schema = FeatureSchema(
            feature_cols=list(schema_payload["feature_cols"]),
            numeric_cols=list(schema_payload["numeric_cols"]),
            categorical_cols=list(schema_payload["categorical_cols"]),
        )
        classes = np.asarray(payload["classes"], dtype=int)
        batch_size_value = int(payload["predict_batch_size"])
        batch_size = batch_size_value if batch_size_value > 0 else None
        prediction_mode = str(payload["prediction_mode"])

        encoded = fit_transform_split(
            X_train_raw,
            X_eval_raw,
            schema,
            model_name="tabpfn",
        )
        estimator = build_estimator(
            "tabpfn",
            seed=seed,
            n_jobs=1,
            categorical_feature_indices=encoded.categorical_feature_indices,
        )
        fit_estimator(estimator, encoded.X_train, y_train, sample_weight=None)

        # The parent reference is always an unbatched prediction. Reconstruct
        # that exact function before any optional external batching is tested.
        set_global_seed(seed)
        reconstructed = predict_proba_aligned(estimator, encoded.X_test, classes)
        reference = np.asarray(payload["reference_proba"], dtype=float)
        if reconstructed.shape != reference.shape:
            raise RuntimeError(
                "The isolated TabPFN reconstruction changed the probability "
                f"shape: reconstructed={reconstructed.shape}, reference={reference.shape}."
            )
        max_abs_delta = float(np.max(np.abs(reconstructed - reference)))
        tolerance = float(payload["reconstruction_tolerance"])
        if not np.allclose(
            reconstructed,
            reference,
            atol=tolerance,
            rtol=tolerance,
        ):
            raise RuntimeError(
                "The isolated TabPFN model did not reproduce the parent-model "
                f"probabilities within tolerance={tolerance:g}; "
                f"max_abs_delta={max_abs_delta:.8g}."
            )

        probe_rows = int(payload["batch_equivalence_probe_rows"])
        if probe_rows < 2 or encoded.X_test.shape[0] < 1:
            raise RuntimeError(
                "TabPFN batch-equivalence validation requires at least two "
                "probe rows and one evaluation record."
            )
        probe_indices = np.arange(probe_rows, dtype=int) % encoded.X_test.shape[0]
        probe = encoded.X_test[probe_indices]
        set_global_seed(seed)
        probe_unbatched_first = predict_proba_aligned(estimator, probe, classes)
        set_global_seed(seed)
        probe_unbatched_second = predict_proba_aligned(estimator, probe, classes)
        probe_batch_size = int(payload["semantic_probe_batch_size"])
        probe_batch_size = int(min(probe_batch_size, max(1, probe_rows // 2)))
        set_global_seed(seed)
        probe_batched = predict_proba_aligned_batched(
            estimator,
            probe,
            classes,
            batch_size=probe_batch_size,
        )
        repeatability_delta, batch_equivalence_delta = (
            validate_tabpfn_prediction_semantics(
                probe_unbatched_first,
                probe_unbatched_second,
                probe_batched,
                tolerance,
            )
        )

        set_global_seed(seed)
        result = compute_shap_importance(
            estimator=estimator,
            preprocessor=encoded.preprocessor,
            X_train_raw=X_train_raw,
            X_eval_raw=X_eval_raw,
            y_eval=y_eval,
            schema=schema,
            classes=classes,
            model_name="tabpfn",
            seed=seed,
            background_size=payload["background_size"],
            eval_size=payload["eval_size"],
            max_evals=payload["max_evals"],
            predict_batch_size=batch_size,
        )
        result["execution_mode"] = f"isolated_spawn_{str(payload['device']).lower()}"
        result["tabpfn_prediction_mode"] = prediction_mode
        result["tabpfn_inference_precision"] = "float32"
        result["tabpfn_internal_memory_saving"] = bool(TABPFN_INTERNAL_MEMORY_SAVING)
        result["reconstruction_max_abs_delta"] = max_abs_delta
        result["reconstruction_tolerance"] = tolerance
        result["repeatability_max_abs_delta"] = repeatability_delta
        result["batch_equivalence_max_abs_delta"] = batch_equivalence_delta
        result["batch_equivalence_probe_rows"] = probe_rows
        atomic_pickle_dump(result, result_path)
        status = {
            "status": "ok",
            "device": str(payload["device"]),
            "prediction_mode": prediction_mode,
            "predict_batch_size": batch_size_value,
            "reconstruction_max_abs_delta": max_abs_delta,
            "repeatability_max_abs_delta": repeatability_delta,
            "batch_equivalence_max_abs_delta": batch_equivalence_delta,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
    except BaseException as exc:
        status = {
            "status": "error",
            "device": str(payload.get("device", "unknown")),
            "prediction_mode": str(payload.get("prediction_mode", "unknown")),
            "predict_batch_size": int(payload.get("predict_batch_size", 0)),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        try:
            del estimator
        except Exception:
            pass
        try:
            clear_cuda_cache()
        except Exception:
            # A fatal CUDA context belongs only to this worker and is discarded
            # when the process exits. The status file remains the source of the
            # failure diagnostic.
            pass
        # Re-create the durable parent defensively. This also protects against
        # an external cleanup utility removing an idle directory during a
        # multi-hour SHAP call.
        status_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(status, status_path)


def _safe_path_component(value: object, limit: int = 96) -> str:
    text = str(value)
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    safe = safe.strip("_") or "call"
    return safe[: int(limit)]


def tabpfn_shap_cache_digest(
    X_train_raw: pd.DataFrame,
    y_train: np.ndarray,
    X_eval_raw: pd.DataFrame,
    y_eval: np.ndarray,
    schema: FeatureSchema,
    classes: Iterable[int],
    seed: int,
    background_size: Optional[int],
    eval_size: Optional[int],
    max_evals: Optional[int],
    reference_proba: np.ndarray,
) -> str:
    specification = {
        "format_version": int(CHECKPOINT_FORMAT_VERSION),
        "model": "tabpfn",
        "seed": int(seed),
        "background_size": background_size,
        "eval_size": eval_size,
        "max_evals": max_evals,
        "feature_cols": list(schema.feature_cols),
        "numeric_cols": list(schema.numeric_cols),
        "categorical_cols": list(schema.categorical_cols),
        "classes": [int(value) for value in classes],
        "shap_permutation_cycles": int(SHAP_PERMUTATION_CYCLES),
        "force_fp32": bool(TABPFN_FORCE_FP32_INFERENCE),
        "internal_memory_saving": bool(TABPFN_INTERNAL_MEMORY_SAVING),
        "reconstruction_tolerance": float(TABPFN_SHAP_RECONSTRUCTION_TOL),
        "checkpoint_sha256": file_sha256(TABPFN_LOCAL_CKPT),
        "X_train_digest": dataframe_digest(X_train_raw),
        "X_eval_digest": dataframe_digest(X_eval_raw),
        "y_train_digest": array_digest(np.asarray(y_train, dtype=int)),
        "y_eval_digest": array_digest(np.asarray(y_eval, dtype=int)),
        "reference_proba_digest": array_digest(
            np.asarray(reference_proba, dtype=float)
        ),
    }
    return stable_json_digest(specification)


def compute_tabpfn_shap_isolated(
    X_train_raw: pd.DataFrame,
    y_train: np.ndarray,
    X_eval_raw: pd.DataFrame,
    y_eval: np.ndarray,
    schema: FeatureSchema,
    classes: Iterable[int],
    seed: int,
    background_size: Optional[int],
    eval_size: Optional[int],
    max_evals: Optional[int],
    reference_proba: np.ndarray,
    work_root: Path,
    call_key: str,
) -> pd.DataFrame:
    """Run TabPFN permutation SHAP without exposing the parent CUDA context."""
    if not ISOLATE_TABPFN_SHAP:
        raise RuntimeError(
            "TabPFN SHAP isolation is disabled. Enabling it is required for the "
            "paper-facing CUDA workflow."
        )
    if not EXPLAIN_ALL_FEATURES:
        raise RuntimeError(
            "The isolated TabPFN path requires EXPLAIN_ALL_FEATURES=True so "
            "feature selection cannot depend on execution order."
        )

    classes_array = np.asarray(list(classes), dtype=int)
    work_root = Path(work_root)
    attempts_root = work_root / "attempts"
    cache_root = work_root / "cache"
    attempts_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    call_digest = tabpfn_shap_cache_digest(
        X_train_raw=X_train_raw,
        y_train=y_train,
        X_eval_raw=X_eval_raw,
        y_eval=y_eval,
        schema=schema,
        classes=classes_array,
        seed=seed,
        background_size=background_size,
        eval_size=eval_size,
        max_evals=max_evals,
        reference_proba=reference_proba,
    )
    safe_call_key = _safe_path_component(call_key)
    # The complete call key remains in the cache payload.  The filesystem name
    # uses only the content digest to remain comfortably below Windows path
    # limits even when OUTPUT_DIR is deeply nested.
    cache_path = cache_root / f"c_{call_digest}.pkl"
    if ENABLE_TABPFN_SHAP_CACHE and cache_path.exists():
        cached = pd.read_pickle(cache_path)
        if (
            not isinstance(cached, dict)
            or cached.get("digest") != call_digest
            or not isinstance(cached.get("result"), pd.DataFrame)
        ):
            raise RuntimeError(
                f"Invalid TabPFN SHAP cache artifact: {cache_path}. "
                "Delete this artifact before retrying."
            )
        cprint(
            f"[TABPFN SHAP CACHE HIT] key={safe_call_key} "
            f"background={background_size}"
        )
        return cached["result"].copy()

    clear_cuda_cache()
    batch_sizes = tabpfn_shap_batch_schedule(
        TABPFN_SHAP_INITIAL_PREDICT_BATCH_SIZE,
        TABPFN_SHAP_MIN_PREDICT_BATCH_SIZE,
    )
    devices = ["cuda"]
    if TABPFN_SHAP_ALLOW_CPU_FALLBACK:
        devices.append("cpu")
    failures = []
    context = mp.get_context("spawn")

    for device in devices:
        if device == "cuda":
            attempts: List[Tuple[str, Optional[int]]] = [
                ("externally_batched_after_equivalence_check", int(size))
                for size in batch_sizes
            ]
            attempts.extend(
                [("unbatched_internal_memory_saving", None)]
                * max(1, int(TABPFN_SHAP_UNBATCHED_RETRIES))
            )
        else:
            attempts = [("unbatched_internal_memory_saving", None)]
        for prediction_mode, batch_size in attempts:
            batch_label = "none" if batch_size is None else str(int(batch_size))
            semantic_probe_batch_size = (
                int(batch_size)
                if batch_size is not None
                else int(TABPFN_SHAP_MIN_PREDICT_BATCH_SIZE)
            )
            semantic_probe_rows = max(
                int(TABPFN_SHAP_BATCH_EQUIVALENCE_PROBE_ROWS),
                2 * semantic_probe_batch_size,
            )
            cprint(
                f"[TABPFN SHAP ISOLATED] seed={seed} device={device} "
                f"background={background_size} mode={prediction_mode} "
                f"predict_batch={batch_label} probe_rows={semantic_probe_rows}"
            )
            mode_code = (
                "bat"
                if prediction_mode == "externally_batched_after_equivalence_check"
                else "unb"
            )
            attempt_name = (
                f"a_{call_digest[:10]}_{device[:1]}_{mode_code}_"
                f"b{batch_label}_{uuid.uuid4().hex[:8]}"
            )
            attempt_path = attempts_root / attempt_name
            attempt_path.mkdir(parents=True, exist_ok=False)
            status_path = attempt_path / "status.json"
            result_path = attempt_path / "result.pkl"
            atomic_json_dump(
                {
                    "status": "queued",
                    "call_key": safe_call_key,
                    "call_digest": call_digest,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "device": device,
                    "prediction_mode": prediction_mode,
                    "predict_batch_size": (
                        int(batch_size) if batch_size is not None else 0
                    ),
                },
                status_path,
            )
            payload: Dict[str, object] = {
                "status_path": str(status_path),
                "result_path": str(result_path),
                "device": device,
                "prediction_mode": prediction_mode,
                "predict_batch_size": int(batch_size) if batch_size is not None else 0,
                "semantic_probe_batch_size": semantic_probe_batch_size,
                "reconstruction_tolerance": float(TABPFN_SHAP_RECONSTRUCTION_TOL),
                "batch_equivalence_probe_rows": semantic_probe_rows,
                "X_train_raw": X_train_raw.reset_index(drop=True),
                "X_eval_raw": X_eval_raw.reset_index(drop=True),
                "y_train": np.asarray(y_train, dtype=int),
                "y_eval": np.asarray(y_eval, dtype=int),
                "schema": {
                    "feature_cols": list(schema.feature_cols),
                    "numeric_cols": list(schema.numeric_cols),
                    "categorical_cols": list(schema.categorical_cols),
                },
                "classes": classes_array,
                "seed": int(seed),
                "background_size": background_size,
                "eval_size": eval_size,
                "max_evals": max_evals,
                "reference_proba": np.asarray(reference_proba, dtype=float),
            }
            process = context.Process(
                target=_tabpfn_shap_worker,
                args=(payload,),
                daemon=False,
            )
            process.start()
            process.join()
            exit_code = process.exitcode
            process.close()

            if not status_path.exists():
                failure = (
                    f"device={device}, mode={prediction_mode}, batch={batch_label}, "
                    f"exit_code={exit_code}, error=durable worker status is missing"
                )
                failures.append(failure)
                if not KEEP_FAILED_TABPFN_WORK_DIRS:
                    shutil.rmtree(attempt_path, ignore_errors=True)
                if device == "cuda" and is_retryable_native_worker_exit(exit_code):
                    cprint(f"[WARN] TabPFN native worker crash; retrying: {failure}")
                    continue
                raise RuntimeError(
                    "TabPFN SHAP IPC failed. This is not treated as a CUDA "
                    "memory error and will not trigger an expensive recomputation. "
                    + failure
                )

            status = json.loads(status_path.read_text(encoding="utf-8"))
            if (
                status.get("status") == "ok"
                and exit_code == 0
                and result_path.exists()
            ):
                result = pd.read_pickle(result_path)
                if not isinstance(result, pd.DataFrame):
                    raise RuntimeError(
                        f"TabPFN SHAP worker returned a non-DataFrame result: {result_path}."
                    )
                if ENABLE_TABPFN_SHAP_CACHE:
                    atomic_pickle_dump(
                        {
                            "digest": call_digest,
                            "call_key": safe_call_key,
                            "created_at": datetime.now().isoformat(timespec="seconds"),
                            "result": result,
                        },
                        cache_path,
                    )
                shutil.rmtree(attempt_path, ignore_errors=True)
                return result

            status_state = str(status.get("status", "unknown"))
            status_error = str(status.get("error", "unknown worker failure"))
            if status_state in {"queued", "running"} and is_retryable_native_worker_exit(
                exit_code
            ):
                status_error = "native worker access violation before final status write"
            failure = (
                f"device={device}, mode={prediction_mode}, "
                f"batch={batch_label}, exit_code={exit_code}, "
                f"status={status_state}, error={status_error}"
            )
            failures.append(failure)
            cprint(f"[WARN] TabPFN isolated SHAP attempt failed: {failure}")
            if not KEEP_FAILED_TABPFN_WORK_DIRS:
                shutil.rmtree(attempt_path, ignore_errors=True)
            if status.get("error_type") == "TabPFNExternalBatchMismatch":
                raise RuntimeError(
                    "TabPFN does not expose a sufficiently batch-invariant "
                    "pointwise probability function for model-agnostic SHAP "
                    "under the configured FP32 execution. The run is stopped "
                    "rather than changing the explained function. " + failure
                )
            if device == "cuda" and (
                is_retryable_native_worker_exit(exit_code)
                or is_retryable_cuda_failure(status_error)
            ):
                continue
            raise RuntimeError(
                "TabPFN isolated SHAP failed for a non-retryable reason. " + failure
            )

    raise RuntimeError(
        "TabPFN SHAP could not be completed in a clean worker process. The main "
        "CUDA context remains valid, but the paper-facing run is stopped rather "
        "than exporting incomplete evidence. Attempts: " + " | ".join(failures)
    )


def compute_shap_importance_safe(
    estimator,
    preprocessor,
    X_train_raw: pd.DataFrame,
    y_train: np.ndarray,
    X_eval_raw: pd.DataFrame,
    y_eval: np.ndarray,
    schema: FeatureSchema,
    classes: Iterable[int],
    model_name: str,
    seed: int,
    background_size: Optional[int],
    eval_size: Optional[int],
    max_evals: Optional[int],
    reference_proba: np.ndarray,
    work_root: Path,
    call_key: str,
) -> pd.DataFrame:
    if str(model_name).lower() == "tabpfn":
        return compute_tabpfn_shap_isolated(
            X_train_raw=X_train_raw,
            y_train=y_train,
            X_eval_raw=X_eval_raw,
            y_eval=y_eval,
            schema=schema,
            classes=classes,
            seed=seed,
            background_size=background_size,
            eval_size=eval_size,
            max_evals=max_evals,
            reference_proba=reference_proba,
            work_root=work_root,
            call_key=call_key,
        )
    return compute_shap_importance(
        estimator=estimator,
        preprocessor=preprocessor,
        X_train_raw=X_train_raw,
        X_eval_raw=X_eval_raw,
        y_eval=y_eval,
        schema=schema,
        classes=classes,
        model_name=model_name,
        seed=seed,
        background_size=background_size,
        eval_size=eval_size,
        max_evals=max_evals,
    )


def classwise_brier_loss(
    y_true: np.ndarray,
    probability: np.ndarray,
    class_id: int,
    evaluation_measure: str = "balanced",
) -> float:
    target = (np.asarray(y_true, dtype=int) == int(class_id)).astype(float)
    probability = np.asarray(probability, dtype=float)
    if len(target) != len(probability):
        raise ValueError("Classwise Brier inputs have different lengths.")
    squared_error = (target - probability) ** 2
    measure = str(evaluation_measure).lower()
    if measure == "balanced":
        weights = class_balanced_evaluation_weights(y_true, int(class_id))
        return weighted_mean(squared_error, weights)
    if measure in {"prevalence", "unweighted"}:
        return float(np.mean(squared_error))
    raise ValueError(
        f"Unknown classwise Brier evaluation measure: {evaluation_measure}"
    )


def _validate_pfi_base_probability(
    base_proba: Optional[np.ndarray],
    estimator,
    preprocessor,
    X_eval_raw: pd.DataFrame,
    classes: np.ndarray,
) -> np.ndarray:
    if base_proba is None:
        return predict_proba_aligned(
            estimator,
            transform_raw(preprocessor, X_eval_raw),
            classes,
        )
    base = np.asarray(base_proba, dtype=float)
    expected_shape = (len(X_eval_raw), len(classes))
    if base.shape != expected_shape:
        raise ValueError(
            "PFI base probabilities have shape "
            f"{base.shape}; expected {expected_shape}."
        )
    if not np.isfinite(base).all():
        raise ValueError("PFI base probabilities contain non-finite values.")
    return base


def _predict_proba_aligned_in_fixed_batches(
    estimator,
    X_encoded,
    classes: np.ndarray,
    max_rows: int,
) -> np.ndarray:
    n_rows = int(X_encoded.shape[0])
    if n_rows == 0:
        return np.zeros((0, len(classes)), dtype=float)
    batch_rows = max(1, int(max_rows))
    if n_rows <= batch_rows:
        return predict_proba_aligned(estimator, X_encoded, classes)
    parts = []
    for start in range(0, n_rows, batch_rows):
        stop = min(start + batch_rows, n_rows)
        parts.append(
            predict_proba_aligned(estimator, X_encoded[start:stop], classes)
        )
    return np.concatenate(parts, axis=0)


def _summarize_class_permutation_predictions(
    base: np.ndarray,
    permuted_probabilities: np.ndarray,
    y_eval: np.ndarray,
    classes: np.ndarray,
    feature: str,
    model_name: str,
    seed: int,
) -> List[Dict[str, object]]:
    y_eval = np.asarray(y_eval, dtype=int)
    permutations = np.asarray(permuted_probabilities, dtype=float)
    if permutations.ndim != 3:
        raise ValueError(
            "PFI permuted probabilities must have shape "
            "(repeats, records, classes)."
        )
    repeats, n_eval, n_classes = permutations.shape
    if (n_eval, n_classes) != base.shape:
        raise ValueError(
            "PFI permuted and base probability shapes are inconsistent."
        )

    base_losses_balanced = {
        int(class_id): classwise_brier_loss(
            y_eval,
            base[:, class_pos],
            int(class_id),
            evaluation_measure="balanced",
        )
        for class_pos, class_id in enumerate(classes)
    }
    base_losses_prevalence = {
        int(class_id): classwise_brier_loss(
            y_eval,
            base[:, class_pos],
            int(class_id),
            evaluation_measure="prevalence",
        )
        for class_pos, class_id in enumerate(classes)
    }
    base_group_primitives: Dict[int, Dict[str, object]] = {}
    for class_pos, class_id in enumerate(classes):
        cid = int(class_id)
        target = (y_eval == cid).astype(float)
        squared_error = (target - base[:, class_pos]) ** 2
        focal_mask = target == 1.0
        other_mask = ~focal_mask
        base_group_primitives[cid] = {
            "focal_mean": float(np.mean(squared_error[focal_mask])),
            "other_mean": float(np.mean(squared_error[other_mask])),
            "n_focal": int(np.sum(focal_mask)),
            "n_other": int(np.sum(other_mask)),
        }

    loss_increases = {int(class_id): [] for class_id in classes}
    prevalence_loss_increases = {int(class_id): [] for class_id in classes}
    probability_shifts = {int(class_id): [] for class_id in classes}
    permuted_focal_losses = {int(class_id): [] for class_id in classes}
    permuted_other_losses = {int(class_id): [] for class_id in classes}
    for repeat_index in range(int(repeats)):
        pp = permutations[repeat_index]
        for class_pos, class_id in enumerate(classes):
            cid = int(class_id)
            permuted_loss = classwise_brier_loss(
                y_eval,
                pp[:, class_pos],
                cid,
                evaluation_measure="balanced",
            )
            prevalence_permuted_loss = classwise_brier_loss(
                y_eval,
                pp[:, class_pos],
                cid,
                evaluation_measure="prevalence",
            )
            loss_increases[cid].append(
                permuted_loss - base_losses_balanced[cid]
            )
            prevalence_loss_increases[cid].append(
                prevalence_permuted_loss - base_losses_prevalence[cid]
            )
            probability_shifts[cid].append(
                weighted_mean(
                    np.abs(base[:, class_pos] - pp[:, class_pos]),
                    class_balanced_evaluation_weights(y_eval, cid),
                )
            )
            target = (y_eval == cid).astype(float)
            squared_error = (target - pp[:, class_pos]) ** 2
            focal_mask = target == 1.0
            other_mask = ~focal_mask
            permuted_focal_losses[cid].append(
                float(np.mean(squared_error[focal_mask]))
            )
            permuted_other_losses[cid].append(
                float(np.mean(squared_error[other_mask]))
            )

    rows: List[Dict[str, object]] = []
    for class_id in classes:
        cid = int(class_id)
        increases = np.asarray(loss_increases[cid], dtype=float)
        prevalence_increases = np.asarray(
            prevalence_loss_increases[cid], dtype=float
        )
        shifts = np.asarray(probability_shifts[cid], dtype=float)
        focal_permuted = np.asarray(
            permuted_focal_losses[cid], dtype=float
        )
        other_permuted = np.asarray(
            permuted_other_losses[cid], dtype=float
        )
        base_group = base_group_primitives[cid]
        mean_increase = float(np.mean(increases))
        prevalence_mean_increase = float(np.mean(prevalence_increases))
        rows.append(
            {
                "seed": int(seed),
                "model": str(model_name),
                "class_id": cid,
                "feature": str(feature),
                "baseline_class_brier": float(base_losses_balanced[cid]),
                "balanced_baseline_class_brier": float(
                    base_losses_balanced[cid]
                ),
                "prevalence_baseline_class_brier": float(
                    base_losses_prevalence[cid]
                ),
                "mean_brier_increase": mean_increase,
                "positive_brier_increase": float(max(mean_increase, 0.0)),
                "balanced_mean_brier_increase": mean_increase,
                "positive_balanced_brier_increase": float(
                    max(mean_increase, 0.0)
                ),
                "prevalence_mean_brier_increase": prevalence_mean_increase,
                "positive_prevalence_brier_increase": float(
                    max(prevalence_mean_increase, 0.0)
                ),
                "std_brier_increase": (
                    float(np.std(increases, ddof=1))
                    if len(increases) > 1
                    else 0.0
                ),
                "positive_repeat_share": float(np.mean(increases > 0.0)),
                "mean_abs_probability_shift": float(np.mean(shifts)),
                "std_abs_probability_shift": (
                    float(np.std(shifts, ddof=1))
                    if len(shifts) > 1
                    else 0.0
                ),
                "baseline_focal_brier_mean": float(
                    base_group["focal_mean"]
                ),
                "baseline_other_brier_mean": float(
                    base_group["other_mean"]
                ),
                "permuted_focal_brier_mean": float(
                    np.mean(focal_permuted)
                ),
                "permuted_other_brier_mean": float(
                    np.mean(other_permuted)
                ),
                "n_focal": int(base_group["n_focal"]),
                "n_other": int(base_group["n_other"]),
                "n_eval": int(n_eval),
                "n_repeats": int(repeats),
                "evaluation_measure": "class_balanced_one_vs_rest",
            }
        )
    return rows


def compute_class_permutation_importance_reference(
    estimator,
    preprocessor,
    X_eval_raw: pd.DataFrame,
    y_eval: np.ndarray,
    classes: Iterable[int],
    feature_cols: List[str],
    model_name: str,
    seed: int,
    repeats: int = PERMUTATION_REPEATS,
    base_proba: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Original repeat-by-repeat PFI path retained for equivalence audits."""
    classes = np.asarray(list(classes), dtype=int)
    X_eval_raw = X_eval_raw.reset_index(drop=True)
    base = _validate_pfi_base_probability(
        base_proba,
        estimator,
        preprocessor,
        X_eval_raw,
        classes,
    )
    rows: List[Dict[str, object]] = []
    for feature in feature_cols:
        feature_rs = np.random.RandomState(
            stable_seed_from_key("pfi", seed, model_name, feature)
        )
        predictions = []
        for _ in range(int(repeats)):
            Xp = X_eval_raw.copy()
            Xp[feature] = feature_rs.permutation(
                Xp[feature].to_numpy()
            )
            predictions.append(
                predict_proba_aligned(
                    estimator,
                    transform_raw(preprocessor, Xp),
                    classes,
                )
            )
        rows.extend(
            _summarize_class_permutation_predictions(
                base=base,
                permuted_probabilities=np.stack(predictions, axis=0),
                y_eval=y_eval,
                classes=classes,
                feature=str(feature),
                model_name=model_name,
                seed=int(seed),
            )
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["rank"] = out.groupby(["seed", "model", "class_id"])[
            "positive_brier_increase"
        ].rank(ascending=False, method="average")
    return out


def compute_class_permutation_importance(
    estimator,
    preprocessor,
    X_eval_raw: pd.DataFrame,
    y_eval: np.ndarray,
    classes: Iterable[int],
    feature_cols: List[str],
    model_name: str,
    seed: int,
    repeats: int = PERMUTATION_REPEATS,
    base_proba: Optional[np.ndarray] = None,
    use_batched_prediction: Optional[bool] = None,
) -> pd.DataFrame:
    """PFI with the original raw-feature permutation plan and batched inference."""
    use_batch = (
        bool(ENABLE_PFI_BATCH_PREDICTION)
        if use_batched_prediction is None
        else bool(use_batched_prediction)
    )
    if not use_batch:
        return compute_class_permutation_importance_reference(
            estimator=estimator,
            preprocessor=preprocessor,
            X_eval_raw=X_eval_raw,
            y_eval=y_eval,
            classes=classes,
            feature_cols=feature_cols,
            model_name=model_name,
            seed=seed,
            repeats=repeats,
            base_proba=base_proba,
        )

    classes = np.asarray(list(classes), dtype=int)
    X_eval_raw = X_eval_raw.reset_index(drop=True)
    base = _validate_pfi_base_probability(
        base_proba,
        estimator,
        preprocessor,
        X_eval_raw,
        classes,
    )
    n_eval = int(len(X_eval_raw))
    if n_eval == 0:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for feature in feature_cols:
        feature_rs = np.random.RandomState(
            stable_seed_from_key("pfi", seed, model_name, feature)
        )
        raw_batches = []
        for _ in range(int(repeats)):
            Xp = X_eval_raw.copy()
            Xp[feature] = feature_rs.permutation(
                Xp[feature].to_numpy()
            )
            raw_batches.append(Xp)
        stacked_raw = pd.concat(raw_batches, axis=0, ignore_index=True)
        stacked_encoded = transform_raw(preprocessor, stacked_raw)
        stacked_proba = _predict_proba_aligned_in_fixed_batches(
            estimator,
            stacked_encoded,
            classes,
            max_rows=int(PFI_BATCH_PREDICTION_ROWS),
        )
        expected_rows = int(repeats) * n_eval
        if stacked_proba.shape != (expected_rows, len(classes)):
            raise RuntimeError(
                "Batched PFI returned an unexpected probability shape: "
                f"{stacked_proba.shape}."
            )
        rows.extend(
            _summarize_class_permutation_predictions(
                base=base,
                permuted_probabilities=stacked_proba.reshape(
                    int(repeats), n_eval, len(classes)
                ),
                y_eval=y_eval,
                classes=classes,
                feature=str(feature),
                model_name=model_name,
                seed=int(seed),
            )
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["rank"] = out.groupby(["seed", "model", "class_id"])[
            "positive_brier_increase"
        ].rank(ascending=False, method="average")
    return out


def fit_loco_reduced_probability(
    y_train: np.ndarray,
    X_train_raw: pd.DataFrame,
    X_eval_raw: pd.DataFrame,
    schema: FeatureSchema,
    classes: Iterable[int],
    model_name: str,
    feature: str,
    seed: int,
    n_jobs: int = 1,
    sample_weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fit one reduced specification and return its evaluation probabilities."""
    if feature not in schema.feature_cols:
        raise ValueError(f"LOCO feature is absent from the schema: {feature}")
    classes = np.asarray(list(classes), dtype=int)
    reduced_features = [c for c in schema.feature_cols if c != feature]
    reduced_schema = FeatureSchema(
        feature_cols=reduced_features,
        numeric_cols=[c for c in schema.numeric_cols if c != feature],
        categorical_cols=[c for c in schema.categorical_cols if c != feature],
    )
    encoded = fit_transform_split(
        X_train_raw[reduced_features].reset_index(drop=True),
        X_eval_raw[reduced_features].reset_index(drop=True),
        reduced_schema,
        model_name=model_name,
    )
    estimator = build_estimator(
        model_name,
        seed=int(seed),
        n_jobs=n_jobs,
        categorical_feature_indices=encoded.categorical_feature_indices,
    )
    fit_estimator(
        estimator,
        encoded.X_train,
        np.asarray(y_train, dtype=int),
        sample_weight=sample_weight,
    )
    return predict_proba_aligned(estimator, encoded.X_test, classes)


def summarize_loco_probability_contrast(
    base_proba: np.ndarray,
    base_pred: np.ndarray,
    reduced_proba: np.ndarray,
    y_train: np.ndarray,
    y_eval: np.ndarray,
    classes: Iterable[int],
    model_name: str,
    feature: str,
    seed: int,
) -> List[Dict[str, object]]:
    """Apply the unchanged LOCO loss contrast to cached reduced probabilities."""
    classes = np.asarray(list(classes), dtype=int)
    y_eval = np.asarray(y_eval, dtype=int)
    base_proba = np.asarray(base_proba, dtype=float)
    reduced_proba = np.asarray(reduced_proba, dtype=float)
    expected_shape = (len(y_eval), len(classes))
    if base_proba.shape != expected_shape or reduced_proba.shape != expected_shape:
        raise ValueError(
            "LOCO probability shape mismatch: "
            f"base={base_proba.shape}, reduced={reduced_proba.shape}, "
            f"expected={expected_shape}."
        )
    base_metrics = evaluate_predictions(
        y_eval, np.asarray(base_pred, dtype=int), labels=classes
    )
    reduced_pred = classes[np.argmax(reduced_proba, axis=1)]
    reduced_metrics = evaluate_predictions(
        y_eval, reduced_pred, labels=classes
    )
    balacc_drop = float(
        base_metrics["balanced_accuracy"]
        - reduced_metrics["balanced_accuracy"]
    )
    macrof1_drop = float(
        base_metrics["macro_f1"] - reduced_metrics["macro_f1"]
    )

    rows: List[Dict[str, object]] = []
    for class_pos, class_id in enumerate(classes):
        cid = int(class_id)
        delta = base_proba[:, class_pos] - reduced_proba[:, class_pos]
        recall_key = f"recall_class_{cid}"
        base_class_brier = classwise_brier_loss(
            y_eval,
            base_proba[:, class_pos],
            cid,
            evaluation_measure="balanced",
        )
        reduced_class_brier = classwise_brier_loss(
            y_eval,
            reduced_proba[:, class_pos],
            cid,
            evaluation_measure="balanced",
        )
        prevalence_base_class_brier = classwise_brier_loss(
            y_eval,
            base_proba[:, class_pos],
            cid,
            evaluation_measure="prevalence",
        )
        prevalence_reduced_class_brier = classwise_brier_loss(
            y_eval,
            reduced_proba[:, class_pos],
            cid,
            evaluation_measure="prevalence",
        )
        binary_target = (y_eval == cid).astype(float)
        focal_mask = binary_target == 1.0
        other_mask = ~focal_mask
        full_squared_error = (
            binary_target - base_proba[:, class_pos]
        ) ** 2
        reduced_squared_error = (
            binary_target - reduced_proba[:, class_pos]
        ) ** 2
        brier_increase = float(
            reduced_class_brier - base_class_brier
        )
        prevalence_brier_increase = float(
            prevalence_reduced_class_brier
            - prevalence_base_class_brier
        )
        balanced_weights = class_balanced_evaluation_weights(y_eval, cid)
        rows.append(
            {
                "seed": int(seed),
                "model": str(model_name),
                "class_id": cid,
                "feature": str(feature),
                "baseline_class_brier": base_class_brier,
                "reduced_class_brier": reduced_class_brier,
                "brier_increase": brier_increase,
                "positive_brier_increase": float(
                    max(brier_increase, 0.0)
                ),
                "balanced_brier_increase": brier_increase,
                "positive_balanced_brier_increase": float(
                    max(brier_increase, 0.0)
                ),
                "prevalence_baseline_class_brier": (
                    prevalence_base_class_brier
                ),
                "prevalence_reduced_class_brier": (
                    prevalence_reduced_class_brier
                ),
                "prevalence_brier_increase": prevalence_brier_increase,
                "positive_prevalence_brier_increase": float(
                    max(prevalence_brier_increase, 0.0)
                ),
                "mean_probability_change": weighted_mean(
                    delta, balanced_weights
                ),
                "mean_abs_probability_change": weighted_mean(
                    np.abs(delta), balanced_weights
                ),
                "prevalence_mean_probability_change": float(
                    np.mean(delta)
                ),
                "prevalence_mean_abs_probability_change": float(
                    np.mean(np.abs(delta))
                ),
                "std_probability_change": float(np.std(delta)),
                "full_focal_brier_mean": float(
                    np.mean(full_squared_error[focal_mask])
                ),
                "full_other_brier_mean": float(
                    np.mean(full_squared_error[other_mask])
                ),
                "reduced_focal_brier_mean": float(
                    np.mean(reduced_squared_error[focal_mask])
                ),
                "reduced_other_brier_mean": float(
                    np.mean(reduced_squared_error[other_mask])
                ),
                "n_focal": int(np.sum(focal_mask)),
                "n_other": int(np.sum(other_mask)),
                "balanced_accuracy_drop": balacc_drop,
                "macro_f1_drop": macrof1_drop,
                "class_recall_drop": float(
                    base_metrics.get(recall_key, 0.0)
                    - reduced_metrics.get(recall_key, 0.0)
                ),
                "n_train": int(len(y_train)),
                "n_eval": int(len(y_eval)),
                "evaluation_measure": "class_balanced_one_vs_rest",
            }
        )
    return rows


def compute_loco_importance(
    base_proba: np.ndarray,
    base_pred: np.ndarray,
    y_train: np.ndarray,
    y_eval: np.ndarray,
    X_train_raw: pd.DataFrame,
    X_eval_raw: pd.DataFrame,
    schema: FeatureSchema,
    classes: Iterable[int],
    model_name: str,
    seed: int,
    n_jobs: int = 1,
    sample_weight: Optional[np.ndarray] = None,
    candidate_features: Optional[List[str]] = None,
    precomputed_reduced_proba: Optional[
        Mapping[str, np.ndarray]
    ] = None,
) -> pd.DataFrame:
    classes = np.asarray(list(classes), dtype=int)
    features = list(candidate_features or schema.feature_cols)
    cached = dict(precomputed_reduced_proba or {})
    rows: List[Dict[str, object]] = []
    for feature in features:
        if feature not in schema.feature_cols:
            continue
        reduced_proba = cached.get(str(feature))
        if reduced_proba is None:
            reduced_proba = fit_loco_reduced_probability(
                y_train=np.asarray(y_train, dtype=int),
                X_train_raw=X_train_raw,
                X_eval_raw=X_eval_raw,
                schema=schema,
                classes=classes,
                model_name=model_name,
                feature=str(feature),
                seed=int(seed),
                n_jobs=n_jobs,
                sample_weight=sample_weight,
            )
        rows.extend(
            summarize_loco_probability_contrast(
                base_proba=np.asarray(base_proba, dtype=float),
                base_pred=np.asarray(base_pred, dtype=int),
                reduced_proba=np.asarray(reduced_proba, dtype=float),
                y_train=np.asarray(y_train, dtype=int),
                y_eval=np.asarray(y_eval, dtype=int),
                classes=classes,
                model_name=model_name,
                feature=str(feature),
                seed=int(seed),
            )
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["rank"] = out.groupby(["seed", "model", "class_id"])[
            "positive_brier_increase"
        ].rank(ascending=False, method="average")
    return out


def compute_firm_importance(
    base_proba: np.ndarray,
    X_eval_raw: pd.DataFrame,
    y_eval: np.ndarray,
    schema: FeatureSchema,
    classes: Iterable[int],
    model_name: str,
    seed: int,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Estimate class-balanced and prevalence-weighted FIRM relevance."""
    classes = np.asarray(list(classes), dtype=int)
    probabilities = np.asarray(base_proba, dtype=float)
    y_eval = np.asarray(y_eval, dtype=int)
    X_eval = X_eval_raw.reset_index(drop=True).copy()
    features = list(feature_cols or schema.feature_cols)

    if probabilities.ndim != 2:
        raise ValueError(
            f"FIRM expects a two-dimensional probability matrix, got {probabilities.shape}."
        )
    if probabilities.shape[0] != len(X_eval):
        raise ValueError(
            "FIRM probabilities and holdout records have different lengths."
        )
    if probabilities.shape[1] != len(classes):
        raise ValueError("FIRM probability columns do not match the supplied classes.")
    if len(y_eval) != len(X_eval):
        raise ValueError("FIRM labels and holdout records have different lengths.")

    rows = []
    for feature in features:
        if feature not in X_eval.columns:
            raise ValueError(
                f"FIRM feature is absent from the evaluation table: {feature}"
            )

        raw_series = X_eval[feature]
        numeric_series = pd.to_numeric(raw_series, errors="coerce")
        use_empirical_levels = (
            feature not in set(FIRM_CONTINUOUS_FEATURES)
            and feature != NEGATIVE_CONTROL_FEATURE
        )

        for class_pos, class_id in enumerate(classes):
            class_scores = probabilities[:, class_pos]
            score_is_finite = np.isfinite(class_scores)

            balanced_row_weights = class_balanced_evaluation_weights(
                y_eval, int(class_id)
            )
            prevalence_row_weights = np.full(
                len(y_eval), 1.0 / float(len(y_eval)), dtype=float
            )

            def estimate(row_weights: np.ndarray) -> Dict[str, float]:
                if use_empirical_levels:
                    valid = raw_series.notna().to_numpy() & score_is_finite
                    frame = pd.DataFrame(
                        {
                            "feature_value": raw_series.loc[valid]
                            .astype(str)
                            .to_numpy(),
                            "class_score": class_scores[valid],
                            "row_weight": row_weights[valid],
                        }
                    )
                    if frame.empty:
                        return {
                            "variance": 0.0,
                            "mean": np.nan,
                            "slope": np.nan,
                            "n_valid": 0,
                            "n_values": 0,
                            "min_count": 0,
                            "max_count": 0,
                            "effective_values": 0.0,
                        }
                    frame["weighted_score"] = frame["row_weight"] * frame["class_score"]
                    grouped = frame.groupby("feature_value", as_index=False).agg(
                        group_weight=("row_weight", "sum"),
                        weighted_score=("weighted_score", "sum"),
                        n_value=("class_score", "size"),
                    )
                    total = float(grouped["group_weight"].sum())
                    if total <= SOURCE_WEIGHT_EPS:
                        group_weights = np.full(len(grouped), 1.0 / float(len(grouped)))
                    else:
                        group_weights = (
                            grouped["group_weight"].to_numpy(dtype=float) / total
                        )
                    conditional_means = grouped["weighted_score"].to_numpy(
                        dtype=float
                    ) / np.maximum(
                        grouped["group_weight"].to_numpy(dtype=float), SOURCE_WEIGHT_EPS
                    )
                    mean_score = float(np.sum(group_weights * conditional_means))
                    variance = float(
                        np.sum(group_weights * (conditional_means - mean_score) ** 2)
                    )
                    return {
                        "variance": variance,
                        "mean": mean_score,
                        "slope": np.nan,
                        "n_valid": int(valid.sum()),
                        "n_values": int(len(grouped)),
                        "min_count": int(grouped["n_value"].min()),
                        "max_count": int(grouped["n_value"].max()),
                        "effective_values": float(1.0 / np.sum(group_weights**2)),
                    }

                x_values = numeric_series.to_numpy(dtype=float)
                valid = np.isfinite(x_values) & score_is_finite
                x_valid = x_values[valid]
                score_valid = class_scores[valid]
                weights_valid = row_weights[valid]
                total = float(np.sum(weights_valid))
                if len(x_valid) == 0 or total <= SOURCE_WEIGHT_EPS:
                    return {
                        "variance": 0.0,
                        "mean": np.nan,
                        "slope": 0.0,
                        "n_valid": 0,
                        "n_values": 0,
                        "min_count": 0,
                        "max_count": 0,
                        "effective_values": 0.0,
                    }
                weights_valid = weights_valid / total
                x_mean = float(np.sum(weights_valid * x_valid))
                score_mean = float(np.sum(weights_valid * score_valid))
                x_centered = x_valid - x_mean
                score_centered = score_valid - score_mean
                denominator = float(np.sum(weights_valid * x_centered**2))
                slope = (
                    float(
                        np.sum(weights_valid * x_centered * score_centered)
                        / denominator
                    )
                    if denominator > SOURCE_WEIGHT_EPS
                    else 0.0
                )
                fitted = score_mean + slope * x_centered
                fitted_mean = float(np.sum(weights_valid * fitted))
                variance = float(np.sum(weights_valid * (fitted - fitted_mean) ** 2))
                n_values = int(pd.Series(x_valid).nunique())
                return {
                    "variance": variance,
                    "mean": fitted_mean,
                    "slope": slope,
                    "n_valid": int(len(x_valid)),
                    "n_values": n_values,
                    "min_count": 1,
                    "max_count": 1,
                    "effective_values": float(n_values),
                }

            balanced = estimate(balanced_row_weights)
            prevalence = estimate(prevalence_row_weights)
            estimation_basis = (
                "empirical_discrete_conditional_mean"
                if use_empirical_levels
                else "weighted_linear_conditional_mean_approximation"
            )

            rows.append(
                {
                    "seed": int(seed),
                    "model": str(model_name),
                    "class_id": int(class_id),
                    "feature": str(feature),
                    "firm_importance": float(np.sqrt(max(balanced["variance"], 0.0))),
                    "balanced_firm_importance": float(
                        np.sqrt(max(balanced["variance"], 0.0))
                    ),
                    "prevalence_firm_importance": float(
                        np.sqrt(max(prevalence["variance"], 0.0))
                    ),
                    "conditional_score_variance": float(max(balanced["variance"], 0.0)),
                    "prevalence_conditional_score_variance": float(
                        max(prevalence["variance"], 0.0)
                    ),
                    "mean_class_probability": balanced["mean"],
                    "prevalence_mean_class_probability": prevalence["mean"],
                    "linear_slope": balanced["slope"],
                    "estimation_basis": estimation_basis,
                    "n_eval": int(len(X_eval)),
                    "n_valid": int(balanced["n_valid"]),
                    "n_observed_values": int(balanced["n_values"]),
                    "min_value_count": int(balanced["min_count"]),
                    "max_value_count": int(balanced["max_count"]),
                    "effective_value_count": float(balanced["effective_values"]),
                    "evaluation_measure": "class_balanced_one_vs_rest",
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["rank"] = out.groupby(["seed", "model", "class_id"])[
            "firm_importance"
        ].rank(ascending=False, method="average")
    return out


def build_top_feature_table(
    shap_df: pd.DataFrame, perm_df: pd.DataFrame, top_k: int
) -> pd.DataFrame:
    if shap_df is not None and not shap_df.empty:
        cols = ["seed", "model", "class_id", "feature", "rank"]
        return shap_df.loc[shap_df["rank"] <= int(top_k), cols].copy()
    if perm_df is not None and not perm_df.empty:
        cols = ["seed", "model", "class_id", "feature", "rank"]
        return perm_df.loc[perm_df["rank"] <= int(top_k), cols].copy()
    return pd.DataFrame(columns=["seed", "model", "class_id", "feature", "rank"])


def selected_feature_pairs(
    feature_cols: List[str],
    classes: np.ndarray,
    top_features_by_class: Optional[pd.DataFrame],
    top_k: int,
) -> Dict[int, List[str]]:
    selected = {int(c): list(feature_cols[:top_k]) for c in classes}
    if top_features_by_class is None or top_features_by_class.empty:
        return selected
    for class_id, group in top_features_by_class.groupby("class_id"):
        feats = (
            group.sort_values("rank")["feature"]
            .astype(str)
            .drop_duplicates()
            .head(int(top_k))
            .tolist()
        )
        if feats:
            selected[int(class_id)] = feats
    return selected


def features_from_top_table(
    top_df: pd.DataFrame, feature_cols: List[str], top_k: int
) -> List[str]:
    if top_df is None or top_df.empty:
        return list(feature_cols[: int(top_k)])
    feats = (
        top_df.sort_values("rank")["feature"]
        .astype(str)
        .drop_duplicates()
        .head(int(top_k))
        .tolist()
    )
    return [f for f in feats if f in feature_cols] or list(feature_cols[: int(top_k)])


def ordered_values_for_pdp(
    X_train_raw: pd.DataFrame,
    feature: str,
    max_values: Optional[int],
    min_train_count: int,
) -> List[object]:
    train_series = X_train_raw[feature]
    values = list(pd.Series(train_series.dropna().unique()))
    vc = train_series.astype(str).value_counts()
    values = [v for v in values if int(vc.get(str(v), 0)) >= int(min_train_count)]
    if max_values is not None and len(values) > int(max_values):
        keep = set(vc.head(int(max_values)).index.astype(str))
        values = [v for v in values if str(v) in keep]
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    if len(values) > 0 and numeric.notna().all():
        return [
            v
            for _, v in sorted(
                zip(numeric.to_numpy(dtype=float), values), key=lambda item: item[0]
            )
        ]
    return sorted(values, key=lambda x: str(x))


def compute_response_profiles(
    estimator,
    preprocessor,
    X_train_raw: pd.DataFrame,
    X_eval_raw: pd.DataFrame,
    classes: Iterable[int],
    feature_cols: List[str],
    model_name: str,
    seed: int,
    top_features_by_class: Optional[pd.DataFrame],
    top_k: int = 6,
    max_values: Optional[int] = None,
    min_train_count: int = 1,
    return_pdp: bool = True,
    return_ice: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute PDP and centered ICE diagnostics from one prediction matrix.

    PDP is the observation-wise mean of the ICE matrix. Computing both outputs
    here prevents the same intervention predictions from being evaluated twice
    and guarantees that the two summaries use identical records and value grids.
    """
    classes = np.asarray(list(classes), dtype=int)
    X_eval_sub = X_eval_raw.reset_index(drop=True).copy()
    base = predict_proba_aligned(
        estimator, transform_raw(preprocessor, X_eval_sub), classes
    )
    selected = selected_feature_pairs(
        feature_cols, classes, top_features_by_class, top_k
    )
    pdp_rows = []
    ice_rows = []
    for class_pos, class_id in enumerate(classes):
        baseline = float(np.mean(base[:, class_pos]))
        for feature in selected.get(int(class_id), feature_cols[:top_k]):
            values = ordered_values_for_pdp(
                X_train_raw, feature, max_values, min_train_count
            )
            curve_values = []
            value_records = []
            for order, value in enumerate(values, 1):
                Xm = X_eval_sub.copy()
                Xm[feature] = value
                proba = predict_proba_aligned(
                    estimator, transform_raw(preprocessor, Xm), classes
                )
                vals = np.asarray(proba[:, class_pos], dtype=float)
                curve_values.append(vals)
                train_count = int(
                    (X_train_raw[feature].astype(str) == str(value)).sum()
                )
                eval_count = int((X_eval_raw[feature].astype(str) == str(value)).sum())
                train_share = float(train_count / max(len(X_train_raw), 1))
                value_records.append(
                    {
                        "feature_value": str(value),
                        "value_order": int(order),
                        "mean_pred_proba": float(np.mean(vals)),
                        "std_across_ice": float(np.std(vals)),
                        "q25_pred_proba": float(np.quantile(vals, 0.25)),
                        "q75_pred_proba": float(np.quantile(vals, 0.75)),
                        "n_train_value": train_count,
                        "n_eval_value": eval_count,
                        "train_support_share": train_share,
                        "is_training_supported": int(train_count > 0),
                    }
                )

            if not curve_values:
                continue
            curves = np.column_stack(curve_values)
            support_weights = np.asarray(
                [record["n_train_value"] for record in value_records], dtype=float
            )
            support_total = float(np.sum(support_weights))
            if support_total <= 0.0:
                raise RuntimeError(
                    f"No training support was available for response profile: {feature}."
                )
            support_weights /= support_total

            individual_ranges = np.ptp(curves, axis=1)
            individual_centers = curves @ support_weights
            centered_curves = curves - individual_centers[:, None]
            centered_mean_curve = np.mean(centered_curves, axis=0, keepdims=True)
            centered_residual = centered_curves - centered_mean_curve
            centered_dispersion = float(
                np.sqrt(
                    np.mean(
                        np.sum(centered_residual**2 * support_weights[None, :], axis=1)
                    )
                )
            )
            profile_metrics = {
                "mean_individual_range": float(np.mean(individual_ranges)),
                "median_individual_range": float(np.median(individual_ranges)),
                "std_individual_range": float(np.std(individual_ranges)),
                "centered_ice_rms_heterogeneity": centered_dispersion,
                "population_mean_curve_range": float(
                    np.max(np.mean(curves, axis=0)) - np.min(np.mean(curves, axis=0))
                ),
                "ice_centering_basis": "outer_training_value_frequency",
            }
            for value_pos, record in enumerate(value_records):
                vals = curves[:, value_pos]
                if return_pdp:
                    pdp_rows.append(
                        {
                            "seed": int(seed),
                            "model": str(model_name),
                            "class_id": int(class_id),
                            "feature": str(feature),
                            "feature_value": record["feature_value"],
                            "value_order": record["value_order"],
                            "mean_pred_proba": float(np.mean(vals)),
                            "median_pred_proba": float(np.median(vals)),
                            "std_pred_proba": float(np.std(vals)),
                            "baseline_mean_proba": baseline,
                            "delta_vs_baseline": float(np.mean(vals) - baseline),
                            "n_eval": int(len(X_eval_sub)),
                            "n_train_value": record["n_train_value"],
                            "n_eval_value": record["n_eval_value"],
                            "train_support_share": record["train_support_share"],
                            "is_training_supported": record["is_training_supported"],
                            "profile_prediction_basis": "shared_ice_matrix",
                        }
                    )
                if return_ice:
                    ice_rows.append(
                        {
                            "seed": int(seed),
                            "model": str(model_name),
                            "class_id": int(class_id),
                            "feature": str(feature),
                            "n_ice_eval": int(len(X_eval_sub)),
                            **record,
                            **profile_metrics,
                        }
                    )
    return pd.DataFrame(pdp_rows), pd.DataFrame(ice_rows)


def add_negative_control_feature(
    X_train_raw: pd.DataFrame,
    X_eval_raw: pd.DataFrame,
    schema: FeatureSchema,
    seed: int,
    feature_name: str = NEGATIVE_CONTROL_FEATURE,
) -> Tuple[pd.DataFrame, pd.DataFrame, FeatureSchema]:
    rs = np.random.RandomState(int(seed) + 9001)
    X_train_nc = X_train_raw.reset_index(drop=True).copy()
    X_eval_nc = X_eval_raw.reset_index(drop=True).copy()
    X_train_nc[feature_name] = rs.normal(loc=0.0, scale=1.0, size=len(X_train_nc))
    X_eval_nc[feature_name] = rs.normal(loc=0.0, scale=1.0, size=len(X_eval_nc))
    schema_nc = FeatureSchema(
        feature_cols=list(schema.feature_cols) + [feature_name],
        numeric_cols=list(schema.numeric_cols) + [feature_name],
        categorical_cols=list(schema.categorical_cols),
    )
    return X_train_nc, X_eval_nc, schema_nc


def compute_negative_control_check(
    y_train: np.ndarray,
    y_eval: np.ndarray,
    X_train_raw: pd.DataFrame,
    X_eval_raw: pd.DataFrame,
    schema: FeatureSchema,
    classes: Iterable[int],
    model_name: str,
    seed: int,
    sample_weight: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    classes = np.asarray(list(classes), dtype=int)
    X_train_nc, X_eval_nc, schema_nc = add_negative_control_feature(
        X_train_raw=X_train_raw,
        X_eval_raw=X_eval_raw,
        schema=schema,
        seed=seed,
        feature_name=NEGATIVE_CONTROL_FEATURE,
    )
    encoded_nc = fit_transform_split(
        X_train_nc, X_eval_nc, schema_nc, model_name=model_name
    )
    estimator_nc = build_estimator(
        model_name,
        seed=int(seed),
        n_jobs=MODEL_N_JOBS,
        categorical_feature_indices=encoded_nc.categorical_feature_indices,
    )
    fit_estimator(
        estimator_nc, encoded_nc.X_train, y_train, sample_weight=sample_weight
    )
    proba_nc = predict_proba_aligned(estimator_nc, encoded_nc.X_test, classes)
    pred_nc = classes[np.argmax(proba_nc, axis=1)]
    metrics_nc = evaluate_predictions(y_eval, pred_nc, labels=classes)

    perm_nc = compute_class_permutation_importance(
        estimator=estimator_nc,
        preprocessor=encoded_nc.preprocessor,
        X_eval_raw=X_eval_nc,
        y_eval=y_eval,
        classes=classes,
        feature_cols=[NEGATIVE_CONTROL_FEATURE],
        model_name=model_name,
        seed=int(seed) + 9209,
        repeats=NEGATIVE_CONTROL_PERMUTATION_REPEATS,
    )
    loco_nc = compute_loco_importance(
        base_proba=proba_nc,
        base_pred=pred_nc,
        y_train=y_train,
        y_eval=y_eval,
        X_train_raw=X_train_nc,
        X_eval_raw=X_eval_nc,
        schema=schema_nc,
        classes=classes,
        model_name=model_name,
        seed=int(seed),
        n_jobs=MODEL_N_JOBS,
        sample_weight=sample_weight,
        candidate_features=[NEGATIVE_CONTROL_FEATURE],
    )
    firm_nc = compute_firm_importance(
        base_proba=proba_nc,
        X_eval_raw=X_eval_nc,
        y_eval=y_eval,
        schema=schema_nc,
        classes=classes,
        model_name=model_name,
        seed=int(seed),
        feature_cols=[NEGATIVE_CONTROL_FEATURE],
    )
    require_full_evaluation(perm_nc, "n_eval", len(y_eval), "negative-control PFI")
    require_full_evaluation(loco_nc, "n_eval", len(y_eval), "negative-control LOCO")
    require_full_evaluation(firm_nc, "n_eval", len(y_eval), "negative-control FIRM")

    rows = []
    for class_id in classes:
        perm_row = perm_nc.loc[perm_nc["class_id"].astype(int) == int(class_id)]
        loco_row = loco_nc.loc[loco_nc["class_id"].astype(int) == int(class_id)]
        firm_row = firm_nc.loc[firm_nc["class_id"].astype(int) == int(class_id)]
        rows.append(
            {
                "seed": int(seed),
                "model": str(model_name),
                "class_id": int(class_id),
                "feature": NEGATIVE_CONTROL_FEATURE,
                "n_eval": int(len(y_eval)),
                "accuracy_with_control": float(metrics_nc["accuracy"]),
                "balanced_accuracy_with_control": float(
                    metrics_nc["balanced_accuracy"]
                ),
                "macro_f1_with_control": float(metrics_nc["macro_f1"]),
                "control_perm_brier_increase": float(
                    perm_row["mean_brier_increase"].iloc[0]
                )
                if not perm_row.empty
                else np.nan,
                "control_perm_positive_brier_increase": float(
                    perm_row["positive_brier_increase"].iloc[0]
                )
                if not perm_row.empty
                else np.nan,
                "control_loco_brier_increase": float(loco_row["brier_increase"].iloc[0])
                if not loco_row.empty
                else np.nan,
                "control_loco_positive_brier_increase": float(
                    loco_row["positive_brier_increase"].iloc[0]
                )
                if not loco_row.empty
                else np.nan,
                "control_loco_balanced_accuracy_drop": float(
                    loco_row["balanced_accuracy_drop"].iloc[0]
                )
                if not loco_row.empty
                else np.nan,
                "control_firm_importance": float(firm_row["firm_importance"].iloc[0])
                if not firm_row.empty
                else np.nan,
                "control_firm_estimation_basis": str(
                    firm_row["estimation_basis"].iloc[0]
                )
                if not firm_row.empty
                else "",
            }
        )
    return pd.DataFrame(rows)


def benchmark_negative_control(
    negative_control_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
    loco_df: pd.DataFrame,
    firm_df: pd.DataFrame,
) -> pd.DataFrame:
    """Place random-feature reliance within the substantive-feature distribution."""
    if negative_control_df is None or negative_control_df.empty:
        return pd.DataFrame()
    out = negative_control_df.copy()
    out["control_reference_basis"] = (
        "matching_original_fit_by_seed_model_class_and_source"
    )
    source_specs = [
        (
            "perm",
            permutation_df,
            "control_perm_positive_brier_increase",
            "positive_brier_increase",
        ),
        (
            "loco",
            loco_df,
            "control_loco_positive_brier_increase",
            "positive_brier_increase",
        ),
        (
            "firm",
            firm_df,
            "control_firm_importance",
            "firm_importance",
        ),
    ]
    for source, source_df, control_col, value_col in source_specs:
        if (
            source_df is None
            or source_df.empty
            or not {"seed", "model", "class_id", value_col}.issubset(source_df.columns)
        ):
            out[f"control_{source}_percentile_among_substantive_features"] = np.nan
            out[f"control_{source}_rank_among_all_features"] = np.nan
            out[f"control_{source}_reference_feature_count"] = 0
            continue
        percentiles = []
        ranks = []
        reference_counts = []
        for row in out.itertuples(index=False):
            reference = source_df.loc[
                (source_df["seed"].astype(int) == int(row.seed))
                & (source_df["model"].astype(str) == str(row.model))
                & (source_df["class_id"].astype(int) == int(row.class_id)),
                value_col,
            ]
            reference = pd.to_numeric(reference, errors="coerce").dropna().astype(float)
            control = finite_float(getattr(row, control_col), default=np.nan)
            if reference.empty or not np.isfinite(control):
                percentiles.append(np.nan)
                ranks.append(np.nan)
                reference_counts.append(int(len(reference)))
                continue
            combined = pd.Series(
                np.concatenate([reference.to_numpy(dtype=float), [control]])
            )
            rank = float(combined.rank(ascending=False, method="average").iloc[-1])
            percentile = float(np.mean(reference.to_numpy(dtype=float) <= control))
            percentiles.append(percentile)
            ranks.append(rank)
            reference_counts.append(int(len(reference)))
        out[f"control_{source}_percentile_among_substantive_features"] = percentiles
        out[f"control_{source}_rank_among_all_features"] = ranks
        out[f"control_{source}_reference_feature_count"] = reference_counts
    return out


# =============================================================================
# RELIABILITY AGGREGATION
# =============================================================================


def finite_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def spearman_rank_correlation(left: Iterable[float], right: Iterable[float]) -> float:
    """Spearman correlation without an optional SciPy runtime dependency."""
    pair = pd.DataFrame(
        {
            "left": pd.to_numeric(pd.Series(list(left)), errors="coerce"),
            "right": pd.to_numeric(pd.Series(list(right)), errors="coerce"),
        }
    ).dropna()
    if len(pair) < 2:
        return np.nan
    left_rank = pair["left"].rank(method="average").to_numpy(dtype=float)
    right_rank = pair["right"].rank(method="average").to_numpy(dtype=float)
    if (
        np.std(left_rank) <= SOURCE_WEIGHT_EPS
        or np.std(right_rank) <= SOURCE_WEIGHT_EPS
    ):
        return 1.0 if np.allclose(left_rank, right_rank) else 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def aggregate_pdp(pdp_df: pd.DataFrame) -> pd.DataFrame:
    if pdp_df is None or pdp_df.empty:
        return pd.DataFrame()
    rows = []
    for (class_id, feature, seed, model), group in pdp_df.groupby(
        ["class_id", "feature", "seed", "model"]
    ):
        rows.append(
            {
                "class_id": int(class_id),
                "feature": str(feature),
                "seed": int(seed),
                "model": str(model),
                "pdp_range": float(
                    group["mean_pred_proba"].max() - group["mean_pred_proba"].min()
                ),
                "pdp_min_train_count": int(group["n_train_value"].min()),
                "pdp_min_train_share": float(group["train_support_share"].min())
                if "train_support_share" in group
                else 0.0,
                "pdp_training_supported_share": float(
                    group["is_training_supported"].mean()
                )
                if "is_training_supported" in group
                else 0.0,
            }
        )
    run_df = pd.DataFrame(rows)
    if run_df.empty:
        return pd.DataFrame()
    seed_df = run_df.groupby(["class_id", "feature", "seed"], as_index=False).agg(
        pdp_range=("pdp_range", "mean"),
        pdp_min_train_count=("pdp_min_train_count", "min"),
        pdp_min_train_share=("pdp_min_train_share", "min"),
        pdp_training_supported_share=("pdp_training_supported_share", "mean"),
        n_models=("model", "nunique"),
    )
    return (
        seed_df.groupby(["class_id", "feature"], as_index=False)
        .agg(
            pdp_mean_range=("pdp_range", "mean"),
            pdp_std_range=("pdp_range", "std"),
            pdp_min_train_count=("pdp_min_train_count", "min"),
            pdp_min_train_share=("pdp_min_train_share", "min"),
            pdp_training_supported_share=("pdp_training_supported_share", "mean"),
            n_pdp_splits=("seed", "nunique"),
            max_pdp_models_per_split=("n_models", "max"),
        )
        .fillna(0.0)
    )


def aggregate_ice(ice_df: pd.DataFrame) -> pd.DataFrame:
    if ice_df is None or ice_df.empty:
        return pd.DataFrame()
    rows = []
    for (class_id, feature, seed, model), group in ice_df.groupby(
        ["class_id", "feature", "seed", "model"]
    ):
        rows.append(
            {
                "class_id": int(class_id),
                "feature": str(feature),
                "seed": int(seed),
                "model": str(model),
                "ice_mean_std": float(group["std_across_ice"].mean()),
                "ice_mean_iqr": float(
                    (group["q75_pred_proba"] - group["q25_pred_proba"]).mean()
                ),
                "ice_mean_individual_range_run": float(
                    group["mean_individual_range"].iloc[0]
                ),
                "ice_median_individual_range_run": float(
                    group["median_individual_range"].iloc[0]
                ),
                "ice_centered_dispersion_run": float(
                    group["centered_ice_rms_heterogeneity"].iloc[0]
                ),
                "ice_population_range_run": float(
                    group["population_mean_curve_range"].iloc[0]
                ),
                "ice_min_train_count": int(group["n_train_value"].min()),
                "ice_min_train_share": float(group["train_support_share"].min())
                if "train_support_share" in group
                else 0.0,
                "ice_training_supported_share": float(
                    group["is_training_supported"].mean()
                )
                if "is_training_supported" in group
                else 0.0,
            }
        )
    run_df = pd.DataFrame(rows)
    if run_df.empty:
        return pd.DataFrame()
    seed_df = run_df.groupby(["class_id", "feature", "seed"], as_index=False).agg(
        ice_mean_std=("ice_mean_std", "mean"),
        ice_mean_iqr=("ice_mean_iqr", "mean"),
        ice_mean_individual_range_run=("ice_mean_individual_range_run", "mean"),
        ice_median_individual_range_run=("ice_median_individual_range_run", "mean"),
        ice_centered_dispersion_run=("ice_centered_dispersion_run", "mean"),
        ice_population_range_run=("ice_population_range_run", "mean"),
        ice_min_train_count=("ice_min_train_count", "min"),
        ice_min_train_share=("ice_min_train_share", "min"),
        ice_training_supported_share=("ice_training_supported_share", "mean"),
        n_models=("model", "nunique"),
    )
    return (
        seed_df.groupby(["class_id", "feature"], as_index=False)
        .agg(
            ice_mean_std=("ice_mean_std", "mean"),
            ice_std_std=("ice_mean_std", "std"),
            ice_mean_iqr=("ice_mean_iqr", "mean"),
            ice_mean_individual_range=("ice_mean_individual_range_run", "mean"),
            ice_std_individual_range=("ice_mean_individual_range_run", "std"),
            ice_median_individual_range=("ice_median_individual_range_run", "mean"),
            ice_mean_centered_dispersion=("ice_centered_dispersion_run", "mean"),
            ice_std_centered_dispersion=("ice_centered_dispersion_run", "std"),
            ice_population_range=("ice_population_range_run", "mean"),
            ice_min_train_count=("ice_min_train_count", "min"),
            ice_min_train_share=("ice_min_train_share", "min"),
            ice_training_supported_share=("ice_training_supported_share", "mean"),
            n_ice_splits=("seed", "nunique"),
            max_ice_models_per_split=("n_models", "max"),
        )
        .fillna(0.0)
    )


def rank_to_signal(values: pd.Series, n_features: int) -> pd.Series:
    """Convert average ranks to a [0, 1] signal without a top-k cutoff."""
    denom = max(int(n_features) - 1, 1)
    ranks = pd.to_numeric(values, errors="coerce").fillna(float(n_features))
    return (1.0 - (ranks - 1.0) / float(denom)).clip(0.0, 1.0)


def summarize_negative_control(negative_control_df: pd.DataFrame) -> pd.DataFrame:
    if negative_control_df is None or negative_control_df.empty:
        return pd.DataFrame()
    return (
        negative_control_df.groupby(["model", "class_id"], as_index=False)
        .agg(
            n_runs=("seed", "count"),
            n_splits=("seed", "nunique"),
            mean_control_perm_brier_increase=("control_perm_brier_increase", "mean"),
            sd_control_perm_brier_increase=("control_perm_brier_increase", "std"),
            mean_control_loco_brier_increase=("control_loco_brier_increase", "mean"),
            sd_control_loco_brier_increase=("control_loco_brier_increase", "std"),
            mean_control_firm_importance=("control_firm_importance", "mean"),
            sd_control_firm_importance=("control_firm_importance", "std"),
            mean_control_loco_balanced_accuracy_drop=(
                "control_loco_balanced_accuracy_drop",
                "mean",
            ),
            mean_control_perm_percentile=(
                "control_perm_percentile_among_substantive_features",
                "mean",
            ),
            mean_control_loco_percentile=(
                "control_loco_percentile_among_substantive_features",
                "mean",
            ),
            mean_control_firm_percentile=(
                "control_firm_percentile_among_substantive_features",
                "mean",
            ),
            mean_control_perm_rank=("control_perm_rank_among_all_features", "mean"),
            mean_control_loco_rank=("control_loco_rank_among_all_features", "mean"),
            mean_control_firm_rank=("control_firm_rank_among_all_features", "mean"),
            mean_balanced_accuracy_with_control=(
                "balanced_accuracy_with_control",
                "mean",
            ),
        )
        .fillna(0.0)
        .assign(feature=NEGATIVE_CONTROL_FEATURE)
    )


FUSION_SOURCE_SPECS = {
    "shap": {"value": "balanced_mean_abs_shap", "label": "SHAP attribution"},
    "pfi": {
        "value": "positive_balanced_brier_increase",
        "label": "Permutation reliance",
    },
    "loco": {"value": "positive_balanced_brier_increase", "label": "LOCO reliance"},
    "firm": {
        "value": "balanced_firm_importance",
        "label": "FIRM observational relevance",
    },
}

PREVALENCE_FUSION_SOURCE_SPECS = {
    "shap": {"value": "prevalence_mean_abs_shap", "label": "SHAP attribution"},
    "pfi": {
        "value": "positive_prevalence_brier_increase",
        "label": "Permutation reliance",
    },
    "loco": {
        "value": "positive_prevalence_brier_increase",
        "label": "LOCO reliance",
    },
    "firm": {
        "value": "prevalence_firm_importance",
        "label": "FIRM observational relevance",
    },
}

# The frozen credal method uses signed source statistics where the source
# definition permits signed values. Positive clipping is intentionally absent.
CREDAL_SOURCE_SPECS = {
    "SHAP": {
        "table_key": "shap",
        "value": "balanced_mean_abs_shap",
        "direction": "upper_tail",
    },
    "PFI": {
        "table_key": "pfi",
        "value": "balanced_mean_brier_increase",
        "direction": "upper_tail",
    },
    "LOCO": {
        "table_key": "loco",
        "value": "balanced_brier_increase",
        "direction": "upper_tail",
    },
    "FIRM": {
        "table_key": "firm",
        "value": "balanced_firm_importance",
        "direction": "upper_tail",
    },
}


@dataclass(frozen=True)
class InnerFoldSpec:
    fold_id: int
    subtrain_idx: np.ndarray
    validation_idx: np.ndarray


def pairwise_rank_agreement(values: Iterable[float]) -> float:
    """Mean pairwise agreement for rank signals already scaled to [0, 1]."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return np.nan
    diffs = [
        abs(arr[i] - arr[j]) for i in range(len(arr)) for j in range(i + 1, len(arr))
    ]
    return float(np.clip(1.0 - np.mean(diffs), 0.0, 1.0))


def prepare_source_run_table(
    table: pd.DataFrame,
    source: str,
    value_col: str,
    n_features: int,
) -> pd.DataFrame:
    required = {"seed", "model", "class_id", "feature", value_col}
    if table is None or table.empty:
        return pd.DataFrame()
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{source} evidence is missing columns: {missing}")

    metadata_cols = [
        column
        for column in ["outer_repeat", "outer_fold", "split_seed"]
        if column in table.columns
    ]
    group_cols = ["seed", *metadata_cols, "model", "class_id", "feature"]
    out = table[group_cols + [value_col]].copy()
    out = (
        out.groupby(group_cols, as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "raw_importance"})
    )
    out["raw_importance"] = (
        pd.to_numeric(out["raw_importance"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    out["rank"] = out.groupby(["seed", "model", "class_id"])["raw_importance"].rank(
        ascending=False, method="average"
    )
    out["rank_signal"] = rank_to_signal(out["rank"], n_features=n_features)
    out.loc[out["raw_importance"] <= SOURCE_WEIGHT_EPS, "rank_signal"] = 0.0
    out["source"] = str(source)
    return out


# Archived helpers below reproduce the superseded feature-specific conflict
# discount for sensitivity checks only. The executable workflow calls the
# cross-fitted functions defined after this block.
def _legacy_summarize_source_run_table(
    run_df: pd.DataFrame, source: str
) -> pd.DataFrame:
    if run_df is None or run_df.empty:
        return pd.DataFrame()

    rows = []
    for (class_id, feature), group in run_df.groupby(["class_id", "feature"]):
        model_rows = group.groupby(["seed", "model"], as_index=False).agg(
            rank_signal=("rank_signal", "mean"),
            raw_importance=("raw_importance", "mean"),
            rank=("rank", "mean"),
        )
        seed_rows = model_rows.groupby("seed", as_index=False).agg(
            seed_rank_signal=("rank_signal", "mean"),
            seed_raw_importance=("raw_importance", "mean"),
            seed_mean_rank=("rank", "mean"),
            n_models=("model", "nunique"),
        )
        between_split = pairwise_rank_agreement(seed_rows["seed_rank_signal"])
        within_split_values = []
        for _, seed_group in model_rows.groupby("seed"):
            agreement = pairwise_rank_agreement(seed_group["rank_signal"])
            if np.isfinite(agreement):
                within_split_values.append(float(agreement))
        within_split = (
            float(np.mean(within_split_values)) if within_split_values else np.nan
        )

        available_components = [
            value for value in [between_split, within_split] if np.isfinite(value)
        ]
        if len(available_components) == 2:
            reliability = float(
                np.sqrt(available_components[0] * available_components[1])
            )
            reliability_basis = "between_split_and_within_split_model"
        elif len(available_components) == 1:
            reliability = float(available_components[0])
            reliability_basis = (
                "between_split_only"
                if np.isfinite(between_split)
                else "within_split_model_only"
            )
        else:
            # A single run cannot identify repeatability. Preserve that absence
            # explicitly instead of treating unavailable evidence as reliable.
            reliability = np.nan
            reliability_basis = "unavailable_not_estimable_single_run"

        rows.append(
            {
                "class_id": int(class_id),
                "feature": str(feature),
                f"{source}_rank_signal": float(seed_rows["seed_rank_signal"].mean()),
                f"{source}_mean_importance": float(
                    seed_rows["seed_raw_importance"].mean()
                ),
                f"{source}_std_importance_across_splits": (
                    float(seed_rows["seed_raw_importance"].std(ddof=1))
                    if len(seed_rows) > 1
                    else 0.0
                ),
                f"{source}_mean_rank": float(seed_rows["seed_mean_rank"].mean()),
                f"{source}_between_split_stability": between_split,
                f"{source}_within_split_model_agreement": within_split,
                f"{source}_source_reliability": float(np.clip(reliability, 0.0, 1.0)),
                f"{source}_reliability_basis": reliability_basis,
                f"n_{source}_splits": int(seed_rows["seed"].nunique()),
                f"n_{source}_runs": int(len(model_rows)),
                f"max_{source}_models_per_split": int(seed_rows["n_models"].max()),
            }
        )
    return pd.DataFrame(rows)


def _legacy_build_split_level_source_conflict(
    source_run_tables: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Average cross-source rank disagreement within outer splits."""
    merged = None
    active_sources = []
    for source, run_df in source_run_tables.items():
        if run_df is None or run_df.empty:
            continue
        seed_signal = run_df.groupby(
            ["seed", "class_id", "feature"], as_index=False
        ).agg(**{f"{source}_seed_rank_signal": ("rank_signal", "mean")})
        merged = (
            seed_signal
            if merged is None
            else merged.merge(
                seed_signal, on=["seed", "class_id", "feature"], how="inner"
            )
        )
        active_sources.append(source)
    if merged is None or len(active_sources) < 2:
        return pd.DataFrame()

    records = []
    for row in merged.itertuples(index=False):
        record = {
            "seed": int(row.seed),
            "class_id": int(row.class_id),
            "feature": str(row.feature),
        }
        for source in active_sources:
            signal = float(getattr(row, f"{source}_seed_rank_signal"))
            other_signals = [
                float(getattr(row, f"{other}_seed_rank_signal"))
                for other in active_sources
                if other != source
            ]
            record[f"{source}_split_conflict"] = float(
                np.mean([abs(signal - other_signal) for other_signal in other_signals])
            )
        records.append(record)
    split_df = pd.DataFrame(records)
    aggregations = {
        f"{source}_mean_split_conflict": (f"{source}_split_conflict", "mean")
        for source in active_sources
    }
    aggregations["n_conflict_splits"] = ("seed", "nunique")
    return split_df.groupby(["class_id", "feature"], as_index=False).agg(**aggregations)


def _legacy_add_rank_conflict_fusion(out: pd.DataFrame) -> pd.DataFrame:
    """Fuse the configured rank evidence without duplicate penalties."""
    df = out.copy()
    source_names = list(FUSION_SOURCE_SPECS)
    records = []
    for _, row in df.iterrows():
        active = []
        for source in source_names:
            signal = finite_float(
                row.get(f"{source}_rank_signal", np.nan), default=np.nan
            )
            reliability = finite_float(
                row.get(f"{source}_source_reliability", np.nan), default=np.nan
            )
            if np.isfinite(signal) and np.isfinite(reliability):
                active.append((source, signal, float(np.clip(reliability, 0.0, 1.0))))

        row_out: Dict[str, object] = {"n_active_sources": int(len(active))}
        conflicts: Dict[str, float] = {}
        for source, signal, _ in active:
            precomputed = finite_float(
                row.get(f"{source}_mean_split_conflict", np.nan), default=np.nan
            )
            if np.isfinite(precomputed):
                conflicts[source] = float(np.clip(precomputed, 0.0, 1.0))
            else:
                other_signals = [
                    other_signal for other, other_signal, _ in active if other != source
                ]
                conflicts[source] = (
                    float(
                        np.mean(
                            [
                                abs(signal - other_signal)
                                for other_signal in other_signals
                            ]
                        )
                    )
                    if other_signals
                    else np.nan
                )

        discounts = {}
        for source, _, reliability in active:
            conflict = conflicts[source]
            discount = (
                reliability
                if not np.isfinite(conflict)
                else reliability * (1.0 - conflict)
            )
            discounts[source] = float(np.clip(discount, 0.0, 1.0))
            row_out[f"{source}_source_conflict"] = conflict
            row_out[f"{source}_discount"] = discounts[source]

        discount_total = float(sum(discounts.values()))
        if discount_total > SOURCE_WEIGHT_EPS:
            adaptive_weights = {
                source: discounts[source] / discount_total for source, _, _ in active
            }
            adaptive_score = float(
                sum(adaptive_weights[source] * signal for source, signal, _ in active)
            )
        else:
            adaptive_weights = {source: np.nan for source, _, _ in active}
            adaptive_score = np.nan

        if active:
            equal_weight = 1.0 / float(len(active))
            equal_weights = {source: equal_weight for source, _, _ in active}
            equal_score = float(np.mean([signal for _, signal, _ in active]))
            mean_reliability = float(
                np.mean([reliability for _, _, reliability in active])
            )
            finite_conflicts = [
                value for value in conflicts.values() if np.isfinite(value)
            ]
            mean_conflict = (
                float(np.mean(finite_conflicts)) if finite_conflicts else np.nan
            )
            evidence_confidence = float(np.mean(list(discounts.values())))
        else:
            equal_weights = {}
            equal_score = np.nan
            mean_reliability = np.nan
            mean_conflict = np.nan
            evidence_confidence = np.nan

        for source in source_names:
            row_out[f"w_{source}"] = adaptive_weights.get(source, 0.0)
            row_out[f"w_equal_{source}"] = equal_weights.get(source, 0.0)
            row_out.setdefault(f"{source}_source_conflict", np.nan)
            row_out.setdefault(f"{source}_discount", 0.0)

        row_out["consensus_importance_adaptive"] = adaptive_score
        row_out["consensus_importance_equal"] = equal_score
        row_out["evidence_reliability"] = mean_reliability
        row_out["evidence_conflict_index"] = mean_conflict
        row_out["evidence_confidence"] = evidence_confidence
        records.append(row_out)

    fusion = pd.DataFrame(records, index=df.index)
    df = pd.concat([df, fusion], axis=1)
    # Reliability-conflict weighting is the prespecified primary analysis.
    # Equal weighting is retained only as a sensitivity analysis.
    df["consensus_importance"] = df["consensus_importance_adaptive"]
    return df


def _legacy_aggregate_reliability(
    shap_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    loco_df: pd.DataFrame,
    firm_df: pd.DataFrame,
    class_ids: Iterable[int],
    feature_cols: List[str],
    top_k: int = 6,
) -> pd.DataFrame:
    del top_k  # The full ranking, not a display cutoff, defines all evidence.
    base = pd.MultiIndex.from_product(
        [list(class_ids), feature_cols], names=["class_id", "feature"]
    ).to_frame(index=False)
    source_tables = {
        "shap": shap_df,
        "pfi": perm_df,
        "loco": loco_df,
        "firm": firm_df,
    }
    source_run_tables: Dict[str, pd.DataFrame] = {}
    out = base.copy()
    for source, table in source_tables.items():
        value_col = FUSION_SOURCE_SPECS[source]["value"]
        run_table = prepare_source_run_table(
            table=table,
            source=source,
            value_col=value_col,
            n_features=len(feature_cols),
        )
        source_run_tables[source] = run_table
        summary = _legacy_summarize_source_run_table(run_table, source=source)
        if summary.empty:
            raise RuntimeError(f"No {source} evidence was available for fusion.")
        out = out.merge(summary, on=["class_id", "feature"], how="left")

    conflict_summary = _legacy_build_split_level_source_conflict(source_run_tables)
    if conflict_summary.empty:
        raise RuntimeError(
            "Cross-source conflict could not be estimated within outer splits."
        )
    out = out.merge(conflict_summary, on=["class_id", "feature"], how="left")

    out = _legacy_add_rank_conflict_fusion(out)
    for scheme, score_col in [
        ("adaptive", "consensus_importance_adaptive"),
        ("equal", "consensus_importance_equal"),
    ]:
        out[f"class_rank_fused_{scheme}"] = out.groupby("class_id")[score_col].rank(
            ascending=False, method="average", na_option="bottom"
        )
    out["class_rank_fused"] = out.groupby("class_id")["consensus_importance"].rank(
        ascending=False, method="average", na_option="bottom"
    )
    out["class_rank"] = out["class_rank_fused"]
    out = out.sort_values(
        ["class_id", "class_rank_fused", "feature"], ascending=[True, True, True]
    ).reset_index(drop=True)
    return out


def normalized_spearman_footrule_agreement(
    left: pd.Series,
    right: pd.Series,
    n_features: int,
) -> float:
    """Return normalized Spearman-footrule agreement for complete rank vectors.

    The unnormalized footrule is ``sum_j |R_j - R'_j|``. Its maximum over two
    permutations of ``J`` items is ``floor(J**2 / 2)``. Average ranks are
    permitted for ties. Missing or incomplete vectors are not silently filled.
    """
    pair = pd.concat(
        [left.rename("left"), right.rename("right")], axis=1, join="inner"
    ).dropna()
    if int(n_features) < 2 or len(pair) != int(n_features):
        return np.nan
    footrule = float(
        np.sum(
            np.abs(
                pair["left"].to_numpy(dtype=float) - pair["right"].to_numpy(dtype=float)
            )
        )
    )
    maximum = float(np.floor((int(n_features) ** 2) / 2.0))
    if maximum <= 0.0:
        return np.nan
    return float(np.clip(1.0 - footrule / maximum, 0.0, 1.0))


def _mean_pairwise_vector_agreement(
    vectors: List[pd.Series],
    n_features: int,
) -> float:
    """Mean normalized footrule agreement over complete rank vectors."""
    agreements = []
    for left_pos, left in enumerate(vectors):
        for right in vectors[left_pos + 1 :]:
            agreement = normalized_spearman_footrule_agreement(
                left, right, n_features=n_features
            )
            if np.isfinite(agreement):
                agreements.append(float(agreement))
    return float(np.mean(agreements)) if agreements else np.nan


def _run_contains_rank_information(
    raw_values: pd.Series,
    n_features: int,
) -> bool:
    """Check whether one source run contains finite, distinguishable evidence."""
    values = pd.to_numeric(raw_values, errors="coerce").to_numpy(dtype=float)
    if len(values) != int(n_features) or not np.all(np.isfinite(values)):
        return False
    scale = max(1.0, float(np.max(np.abs(values))))
    tolerance = np.finfo(float).eps * max(8.0, float(n_features)) * scale
    return bool(float(np.max(values) - np.min(values)) > tolerance)


def estimate_source_class_reliability(
    reference_run_df: pd.DataFrame,
    feature_cols: List[str],
) -> Dict[str, float]:
    """Estimate class-by-source reliability from informative calibration runs.

    Reliability combines between-fold rank repeatability, within-fold model
    agreement, and the fraction of scheduled runs that contain usable ranking
    information. A stable all-zero source is therefore unavailable, not highly
    reliable.
    """
    if reference_run_df is None or reference_run_df.empty:
        return {
            "between_partition_repeatability": np.nan,
            "within_partition_model_agreement": np.nan,
            "source_reliability": np.nan,
            "information_availability": 0.0,
            "n_reference_partitions": 0,
            "n_informative_partitions": 0,
            "n_reference_runs": 0,
            "n_informative_runs": 0,
            "n_partitions_with_within_agreement": 0,
            "max_models_per_partition": 0,
            "reliability_basis": "not_estimable_no_runs",
        }

    expected = set(str(feature) for feature in feature_cols)
    n_features = int(len(feature_cols))
    run_records: List[Dict[str, object]] = []
    for (partition_id, model_name), group in reference_run_df.groupby(
        ["seed", "model"], dropna=False
    ):
        grouped = (
            group.groupby("feature", as_index=True)
            .agg(raw_importance=("raw_importance", "mean"), rank=("rank", "mean"))
            .reindex(feature_cols)
        )
        complete = set(grouped.dropna().index.astype(str)) == expected
        informative = bool(
            complete
            and _run_contains_rank_information(
                grouped["raw_importance"], n_features=n_features
            )
        )
        run_records.append(
            {
                "seed": int(partition_id),
                "model": str(model_name),
                "informative": informative,
                "rank_vector": grouped["rank"].astype(float),
            }
        )

    n_reference_runs = int(len(run_records))
    n_informative_runs = int(sum(bool(row["informative"]) for row in run_records))
    availability = (
        float(n_informative_runs / n_reference_runs) if n_reference_runs else 0.0
    )

    run_frame = pd.DataFrame(
        [
            {
                "seed": int(row["seed"]),
                "model": str(row["model"]),
                "informative": int(bool(row["informative"])),
            }
            for row in run_records
        ]
    )
    max_models_per_partition = (
        int(run_frame.groupby("seed")["model"].nunique().max())
        if not run_frame.empty
        else 0
    )

    partition_vectors = []
    informative_partition_ids: Set[int] = set()
    for partition_id in sorted(reference_run_df["seed"].astype(int).unique()):
        vectors = [
            row["rank_vector"]
            for row in run_records
            if int(row["seed"]) == int(partition_id) and bool(row["informative"])
        ]
        if vectors:
            partition_vectors.append(
                pd.concat(vectors, axis=1).mean(axis=1).reindex(feature_cols)
            )
            informative_partition_ids.add(int(partition_id))
    between = _mean_pairwise_vector_agreement(partition_vectors, n_features=n_features)

    within_values = []
    for partition_id in sorted(reference_run_df["seed"].astype(int).unique()):
        model_vectors = [
            row["rank_vector"]
            for row in run_records
            if int(row["seed"]) == int(partition_id) and bool(row["informative"])
        ]
        agreement = _mean_pairwise_vector_agreement(
            model_vectors, n_features=n_features
        )
        if np.isfinite(agreement):
            within_values.append(float(agreement))
    within = float(np.mean(within_values)) if within_values else np.nan

    if np.isfinite(between) and np.isfinite(within):
        reliability = float(
            availability * np.sqrt(max(between, 0.0) * max(within, 0.0))
        )
        basis = "geometric_mean_between_partition_and_within_partition"
    elif np.isfinite(between) and max_models_per_partition == 1:
        reliability = float(availability * max(between, 0.0))
        basis = "between_partition_only_single_admitted_model"
    else:
        reliability = np.nan
        basis = "not_estimable"

    return {
        "between_partition_repeatability": between,
        "within_partition_model_agreement": within,
        "source_reliability": (
            float(np.clip(reliability, 0.0, 1.0))
            if np.isfinite(reliability)
            else np.nan
        ),
        "information_availability": float(np.clip(availability, 0.0, 1.0)),
        "reliability_basis": basis,
        "n_reference_partitions": int(reference_run_df["seed"].nunique()),
        "n_informative_partitions": int(len(informative_partition_ids)),
        "n_reference_runs": n_reference_runs,
        "n_informative_runs": n_informative_runs,
        "n_partitions_with_within_agreement": int(len(within_values)),
        "max_models_per_partition": int(max_models_per_partition),
    }


def _normalize_source_reliabilities(
    current: List[Dict[str, object]],
) -> Tuple[np.ndarray, str, str, int]:
    """Normalize positive reliabilities without inventing an equal fallback."""
    reliabilities = np.asarray(
        [finite_float(row["source_reliability"], np.nan) for row in current],
        dtype=float,
    )
    valid = np.isfinite(reliabilities) & (reliabilities > SOURCE_WEIGHT_EPS)
    n_active = int(np.sum(valid))
    total = float(np.sum(reliabilities[valid])) if np.any(valid) else 0.0
    if total > SOURCE_WEIGHT_EPS:
        weights = np.zeros(len(current), dtype=float)
        weights[valid] = reliabilities[valid] / total
        status = "multi_source_consensus" if n_active >= 2 else "source_specific_only"
        return (
            weights,
            "normalized_positive_training_internal_reliability",
            status,
            n_active,
        )
    return (
        np.zeros(len(current), dtype=float),
        "no_automatic_fallback_reliability_not_estimable",
        "consensus_unavailable",
        0,
    )


def build_training_internal_source_weights(
    calibration_source_tables: Dict[str, pd.DataFrame],
    class_ids: Iterable[int],
    feature_cols: List[str],
    partitions: List[OuterPartition],
    source_specs: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    """Learn source weights from inner folds of each outer training partition.

    The calibration tables are generated without using the corresponding outer
    holdout. Consequently, target-holdout features and labels cannot influence
    either calibration-model fitting or source-reliability estimation.
    """
    active_sources = [
        source
        for source in source_specs
        if source in calibration_source_tables
        and calibration_source_tables[source] is not None
        and not calibration_source_tables[source].empty
    ]
    if len(active_sources) < 2:
        raise RuntimeError(
            "At least two training-internal explanation sources are required."
        )

    partition_lookup = {int(part.partition_id): part for part in partitions}
    rows = []
    for target_partition, part in partition_lookup.items():
        for class_id in class_ids:
            current: List[Dict[str, object]] = []
            for source in active_sources:
                raw = calibration_source_tables[source]
                if "target_outer_partition" not in raw.columns:
                    raise ValueError(
                        f"Training-internal {source} evidence lacks "
                        "target_outer_partition."
                    )
                subset = raw.loc[
                    pd.to_numeric(raw["target_outer_partition"], errors="coerce")
                    .fillna(-1)
                    .astype(int)
                    == int(target_partition)
                ].copy()
                run_table = prepare_source_run_table(
                    table=subset,
                    source=source,
                    value_col=source_specs[source]["value"],
                    n_features=len(feature_cols),
                )
                reference = run_table.loc[
                    run_table["class_id"].astype(int) == int(class_id)
                ].copy()
                stats = estimate_source_class_reliability(reference, feature_cols)
                if int(stats["n_reference_partitions"]) < 3:
                    raise RuntimeError(
                        "Training-internal source weighting requires at least "
                        f"three calibration folds; target={target_partition}, "
                        f"class={class_id}, source={source}."
                    )
                calibration_folds = sorted(
                    reference["seed"].astype(int).unique().tolist()
                )
                current.append(
                    {
                        "outer_partition": int(target_partition),
                        "outer_repeat": int(part.repeat_id),
                        "outer_fold": int(part.fold_id),
                        "class_id": int(class_id),
                        "source": str(source),
                        **stats,
                        "reference_rule": ("outer_training_stratified_inner_crossfit"),
                        "reference_partitions": ";".join(
                            f"inner_fold_{fold}" for fold in calibration_folds
                        ),
                    }
                )

            (
                weights,
                normalization_basis,
                consensus_status,
                n_active_sources,
            ) = _normalize_source_reliabilities(current)
            for row, weight in zip(current, weights):
                row["source_weight"] = float(weight)
                row["weight_normalization_basis"] = normalization_basis
                row["consensus_status"] = consensus_status
                row["n_active_sources"] = int(n_active_sources)
                rows.append(row)

    out = pd.DataFrame(rows)
    for _, group in out.groupby(["outer_partition", "class_id"]):
        expected_sum = 1.0 if int(group["n_active_sources"].iloc[0]) > 0 else 0.0
        observed_sum = float(group["source_weight"].sum())
        if not np.isclose(observed_sum, expected_sum, atol=1e-10):
            raise RuntimeError(
                "Training-internal source weights violate the declared availability state."
            )
    expected_targets = set(partition_lookup)
    observed_targets = set(out["outer_partition"].astype(int).unique())
    if observed_targets != expected_targets:
        raise RuntimeError(
            "Training-internal source weights do not cover every outer partition."
        )
    return out


def build_crossfitted_source_weights(
    source_run_tables: Dict[str, pd.DataFrame],
    class_ids: Iterable[int],
    feature_cols: List[str],
) -> pd.DataFrame:
    """Fallback leave-partition-out weighting for synthetic validation only.

    The empirical workflow uses ``build_training_internal_source_weights``.
    This fallback remains for known-truth simulations whose partition evidence
    is generated independently and has no underlying model-training overlap.
    """
    active_sources = [
        source
        for source, table in source_run_tables.items()
        if table is not None and not table.empty
    ]
    if len(active_sources) < 2:
        raise RuntimeError(
            "At least two complete explanation sources are required for fusion."
        )
    partition_ids = sorted(
        set.intersection(
            *[
                set(table["seed"].astype(int).unique().tolist())
                for table in source_run_tables.values()
                if table is not None and not table.empty
            ]
        )
    )
    if len(partition_ids) < 3:
        raise RuntimeError(
            "Cross-fitted source weighting requires at least three outer partitions."
        )

    rows = []
    for target_partition in partition_ids:
        repeat_values = []
        fold_values = []
        for table in source_run_tables.values():
            if table is None or table.empty:
                continue
            target_meta = table.loc[table["seed"].astype(int) == int(target_partition)]
            if target_meta.empty:
                continue
            if "outer_repeat" in target_meta.columns:
                repeat_values.extend(
                    pd.to_numeric(target_meta["outer_repeat"], errors="coerce")
                    .dropna()
                    .astype(int)
                    .unique()
                    .tolist()
                )
            if "outer_fold" in target_meta.columns:
                fold_values.extend(
                    pd.to_numeric(target_meta["outer_fold"], errors="coerce")
                    .dropna()
                    .astype(int)
                    .unique()
                    .tolist()
                )
        repeat_values = sorted(set(repeat_values))
        fold_values = sorted(set(fold_values))
        if len(repeat_values) > 1 or len(fold_values) > 1:
            raise RuntimeError(
                f"Partition {target_partition} has inconsistent repeat/fold metadata."
            )
        target_repeat = repeat_values[0] if repeat_values else None
        target_fold = fold_values[0] if fold_values else None

        for class_id in class_ids:
            current = []
            for source in active_sources:
                table = source_run_tables[source]
                reference_mask = (
                    table["seed"].astype(int) != int(target_partition)
                ) & (table["class_id"].astype(int) == int(class_id))
                if target_repeat is not None and "outer_repeat" in table.columns:
                    reference_mask &= pd.to_numeric(
                        table["outer_repeat"], errors="coerce"
                    ).fillna(-1).astype(int) == int(target_repeat)
                    reference_rule = "simulation_other_partitions_same_repeat"
                else:
                    reference_rule = "simulation_all_other_partitions"
                reference = table.loc[reference_mask].copy()
                stats = estimate_source_class_reliability(reference, feature_cols)
                current.append(
                    {
                        "outer_partition": int(target_partition),
                        "outer_repeat": (
                            int(target_repeat) if target_repeat is not None else np.nan
                        ),
                        "outer_fold": (
                            int(target_fold) if target_fold is not None else np.nan
                        ),
                        "class_id": int(class_id),
                        "source": str(source),
                        **stats,
                        "reference_rule": reference_rule,
                        "reference_partitions": ";".join(
                            str(v)
                            for v in sorted(
                                reference["seed"].astype(int).unique().tolist()
                            )
                        ),
                    }
                )

            (
                weights,
                normalization_basis,
                consensus_status,
                n_active_sources,
            ) = _normalize_source_reliabilities(current)
            normalization_basis = normalization_basis.replace(
                "training_internal", "simulation_crossfitted"
            )

            for row, weight in zip(current, weights):
                row["source_weight"] = float(weight)
                row["weight_normalization_basis"] = normalization_basis
                row["consensus_status"] = consensus_status
                row["n_active_sources"] = int(n_active_sources)
                rows.append(row)

    out = pd.DataFrame(rows)
    for _, group in out.groupby(["outer_partition", "class_id"]):
        expected_sum = 1.0 if int(group["n_active_sources"].iloc[0]) > 0 else 0.0
        observed_sum = float(group["source_weight"].sum())
        if not np.isclose(observed_sum, expected_sum, atol=1e-10):
            raise RuntimeError(
                "Cross-fitted source weights violate the declared availability state."
            )
    return out


def build_crossfitted_partition_consensus(
    source_run_tables: Dict[str, pd.DataFrame],
    source_weights: pd.DataFrame,
    class_ids: Iterable[int],
    feature_cols: List[str],
    top_k: int,
) -> pd.DataFrame:
    """Fuse each target partition with externally supplied source weights."""
    active_sources = [
        source
        for source, table in source_run_tables.items()
        if table is not None and not table.empty
    ]
    signal_tables = {}
    for source in active_sources:
        signal_tables[source] = (
            source_run_tables[source]
            .groupby(["seed", "class_id", "feature"], as_index=False)
            .agg(
                rank_signal=("rank_signal", "mean"),
                raw_importance=("raw_importance", "mean"),
                n_models=("model", "nunique"),
            )
        )

    rows = []
    partition_ids = sorted(source_weights["outer_partition"].astype(int).unique())
    for partition_id in partition_ids:
        for class_id in class_ids:
            weight_part = source_weights.loc[
                (source_weights["outer_partition"].astype(int) == int(partition_id))
                & (source_weights["class_id"].astype(int) == int(class_id))
            ].set_index("source")
            if set(weight_part.index.astype(str)) != set(active_sources):
                raise RuntimeError(
                    f"Incomplete source weights for partition={partition_id}, class={class_id}."
                )

            for feature in feature_cols:
                signals: Dict[str, float] = {}
                raw_values: Dict[str, float] = {}
                for source in active_sources:
                    source_row = signal_tables[source].loc[
                        (signal_tables[source]["seed"].astype(int) == int(partition_id))
                        & (
                            signal_tables[source]["class_id"].astype(int)
                            == int(class_id)
                        )
                        & (signal_tables[source]["feature"].astype(str) == str(feature))
                    ]
                    if len(source_row) != 1:
                        raise RuntimeError(
                            f"Missing {source} signal for partition={partition_id}, "
                            f"class={class_id}, feature={feature}."
                        )
                    signals[source] = float(source_row["rank_signal"].iloc[0])
                    raw_values[source] = float(source_row["raw_importance"].iloc[0])

                weights = {
                    source: finite_float(weight_part.loc[source, "source_weight"], 0.0)
                    for source in active_sources
                }
                reliabilities = {
                    source: finite_float(
                        weight_part.loc[source, "source_reliability"], np.nan
                    )
                    for source in active_sources
                }
                available_sources = [
                    source
                    for source in active_sources
                    if weights[source] > SOURCE_WEIGHT_EPS
                    and np.isfinite(reliabilities[source])
                    and np.isfinite(signals[source])
                ]
                n_available = int(len(available_sources))
                if n_available >= 2:
                    primary = float(
                        sum(
                            weights[source] * signals[source]
                            for source in available_sources
                        )
                    )
                    source_specific = np.nan
                    ambiguity = float(
                        sum(
                            weights[source] * abs(signals[source] - primary)
                            for source in available_sources
                        )
                    )
                    consensus_status = "multi_source_consensus"
                elif n_available == 1:
                    primary = np.nan
                    source_specific = float(signals[available_sources[0]])
                    ambiguity = np.nan
                    consensus_status = "source_specific_only"
                else:
                    primary = np.nan
                    source_specific = np.nan
                    ambiguity = np.nan
                    consensus_status = "consensus_unavailable"

                equal = (
                    float(np.mean([signals[source] for source in available_sources]))
                    if available_sources
                    else np.nan
                )
                median = (
                    float(np.median([signals[source] for source in available_sources]))
                    if available_sources
                    else np.nan
                )
                source_conflicts = {}
                for source in active_sources:
                    others = [other for other in available_sources if other != source]
                    source_conflicts[source] = (
                        float(
                            np.mean(
                                [
                                    abs(signals[source] - signals[other])
                                    for other in others
                                ]
                            )
                        )
                        if source in available_sources and others
                        else np.nan
                    )
                legacy_discounts = {}
                for source in active_sources:
                    conflict = source_conflicts[source]
                    reliability = reliabilities[source]
                    legacy_discounts[source] = (
                        float(reliability * (1.0 - conflict))
                        if source in available_sources
                        and np.isfinite(reliability)
                        and np.isfinite(conflict)
                        else 0.0
                    )
                legacy_total = float(sum(legacy_discounts.values()))
                legacy = (
                    float(
                        sum(
                            legacy_discounts[source] * signals[source]
                            for source in available_sources
                        )
                        / legacy_total
                    )
                    if legacy_total > SOURCE_WEIGHT_EPS
                    else equal
                )

                row = {
                    "outer_partition": int(partition_id),
                    "seed": int(partition_id),
                    "outer_repeat": int(weight_part["outer_repeat"].iloc[0])
                    if "outer_repeat" in weight_part.columns
                    and pd.notna(weight_part["outer_repeat"].iloc[0])
                    else np.nan,
                    "outer_fold": int(weight_part["outer_fold"].iloc[0])
                    if "outer_fold" in weight_part.columns
                    and pd.notna(weight_part["outer_fold"].iloc[0])
                    else np.nan,
                    "class_id": int(class_id),
                    "feature": str(feature),
                    "consensus_importance_crossfitted": primary,
                    "source_specific_importance": source_specific,
                    "consensus_status": consensus_status,
                    "consensus_importance_equal": equal,
                    "consensus_importance_median": median,
                    "consensus_importance_legacy_conflict_discount": legacy,
                    "evidence_ambiguity": ambiguity,
                    "source_signal_range": float(
                        max(signals[source] for source in available_sources)
                        - min(signals[source] for source in available_sources)
                    )
                    if len(available_sources) >= 2
                    else np.nan,
                    "evidence_support": float(
                        sum(
                            weights[source] * reliabilities[source]
                            for source in available_sources
                        )
                    )
                    if available_sources
                    else np.nan,
                    "n_active_sources": n_available,
                    "active_sources": "; ".join(available_sources),
                }
                for source in active_sources:
                    row[f"{source}_rank_signal"] = signals[source]
                    row[f"{source}_raw_importance"] = raw_values[source]
                    row[f"w_{source}"] = weights[source]
                    row[f"w_equal_{source}"] = 1.0 / float(len(active_sources))
                    row[f"{source}_source_reliability"] = reliabilities[source]
                    row[f"{source}_source_conflict"] = source_conflicts[source]
                    row[f"{source}_discount"] = legacy_discounts[source]
                    row[f"{source}_between_split_stability"] = float(
                        weight_part.loc[source, "between_partition_repeatability"]
                    )
                    row[f"{source}_within_split_model_agreement"] = float(
                        weight_part.loc[source, "within_partition_model_agreement"]
                    )
                    row[f"{source}_information_availability"] = float(
                        weight_part.loc[source, "information_availability"]
                    )
                    row[f"{source}_n_informative_runs"] = int(
                        weight_part.loc[source, "n_informative_runs"]
                    )
                rows.append(row)

    out = pd.DataFrame(rows)
    scheme_cols = {
        "crossfitted": "consensus_importance_crossfitted",
        "equal": "consensus_importance_equal",
        "median": "consensus_importance_median",
        "legacy": "consensus_importance_legacy_conflict_discount",
    }
    for scheme, score_col in scheme_cols.items():
        out[f"rank_{scheme}"] = out.groupby(["outer_partition", "class_id"])[
            score_col
        ].rank(ascending=False, method="average", na_option="keep")
        out[f"top_{int(top_k)}_{scheme}"] = (
            pd.to_numeric(out[f"rank_{scheme}"], errors="coerce") <= float(top_k)
        ).astype(int)
    return out


def build_pairwise_dominance_table(partition_df: pd.DataFrame) -> pd.DataFrame:
    """Estimate pairwise dominance and mark only unanimous strict relations."""
    rows = []
    for class_id, group in partition_df.groupby("class_id"):
        expected_n_partitions = int(group["outer_partition"].nunique())
        pivot = group.pivot(
            index="outer_partition",
            columns="feature",
            values="consensus_importance_crossfitted",
        )
        features = list(pivot.columns.astype(str))
        for left in features:
            for right in features:
                if left == right:
                    continue
                pair = pivot[[left, right]].dropna()
                difference = pair[left].to_numpy(dtype=float) - pair[right].to_numpy(
                    dtype=float
                )
                wins = difference > SOURCE_WEIGHT_EPS
                ties = np.abs(difference) <= SOURCE_WEIGHT_EPS
                rows.append(
                    {
                        "class_id": int(class_id),
                        "feature_left": str(left),
                        "feature_right": str(right),
                        "n_partitions": int(len(pair)),
                        "expected_n_partitions": expected_n_partitions,
                        "complete_partition_coverage": int(
                            len(pair) == expected_n_partitions
                        ),
                        "dominance_probability": float(
                            np.mean(wins.astype(float) + 0.5 * ties.astype(float))
                        )
                        if len(pair)
                        else np.nan,
                        "strict_win_frequency": float(np.mean(wins))
                        if len(pair)
                        else np.nan,
                        "tie_frequency": float(np.mean(ties)) if len(pair) else np.nan,
                        "unanimous_strict_dominance": int(
                            len(pair) == expected_n_partitions
                            and expected_n_partitions > 0
                            and np.all(wins)
                        ),
                        "mean_score_difference": float(np.mean(difference))
                        if len(pair)
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def build_partial_order_levels(
    pairwise_df: pd.DataFrame, class_ids: Iterable[int], feature_cols: List[str]
) -> pd.DataFrame:
    """Layer the unanimous dominance relation without imposing a cutoff."""
    rows = []
    for class_id in class_ids:
        class_pairs = pairwise_df.loc[
            (pairwise_df["class_id"].astype(int) == int(class_id))
            & (pairwise_df["unanimous_strict_dominance"].astype(int) == 1)
        ]
        dominates = {str(feature): set() for feature in feature_cols}
        dominated_by = {str(feature): set() for feature in feature_cols}
        for pair in class_pairs.itertuples(index=False):
            dominates[str(pair.feature_left)].add(str(pair.feature_right))
            dominated_by[str(pair.feature_right)].add(str(pair.feature_left))

        remaining = set(str(feature) for feature in feature_cols)
        level = 1
        assigned = {}
        while remaining:
            front = sorted(
                feature
                for feature in remaining
                if not (dominated_by[feature] & remaining)
            )
            if not front:
                raise RuntimeError("The unanimous dominance relation contains a cycle.")
            for feature in front:
                assigned[feature] = int(level)
            remaining.difference_update(front)
            level += 1

        for feature in feature_cols:
            rows.append(
                {
                    "class_id": int(class_id),
                    "feature": str(feature),
                    "partial_order_level": int(assigned[str(feature)]),
                    "n_unanimously_dominates": int(len(dominates[str(feature)])),
                    "n_unanimously_dominated_by": int(len(dominated_by[str(feature)])),
                }
            )
    return pd.DataFrame(rows)


def aggregate_partition_consensus(
    partition_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    class_ids: Iterable[int],
    feature_cols: List[str],
    top_k: int,
) -> pd.DataFrame:
    """Summarize consensus magnitude, dispersion, inclusion, and partial order."""
    rows = []
    sources = list(FUSION_SOURCE_SPECS)
    for (class_id, feature), group in partition_df.groupby(["class_id", "feature"]):
        score_series = pd.to_numeric(
            group["consensus_importance_crossfitted"], errors="coerce"
        )
        scores = score_series.dropna().to_numpy(dtype=float)
        status_counts = group["consensus_status"].astype(str).value_counts()
        consensus_mask = (
            group["consensus_status"].astype(str).eq("multi_source_consensus")
        )

        def _safe_group_mean(column: str) -> float:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            return float(values.mean()) if len(values) else np.nan

        row = {
            "class_id": int(class_id),
            "feature": str(feature),
            "n_outer_partitions": int(group["outer_partition"].nunique()),
            "n_consensus_partitions": int(
                status_counts.get("multi_source_consensus", 0)
            ),
            "n_source_specific_partitions": int(
                status_counts.get("source_specific_only", 0)
            ),
            "n_unavailable_partitions": int(
                status_counts.get("consensus_unavailable", 0)
            ),
            "consensus_partition_fraction": float(
                status_counts.get("multi_source_consensus", 0) / max(len(group), 1)
            ),
            "consensus_importance": float(np.mean(scores)) if len(scores) else np.nan,
            "consensus_importance_adaptive": float(np.mean(scores))
            if len(scores)
            else np.nan,
            "consensus_importance_crossfitted": float(np.mean(scores))
            if len(scores)
            else np.nan,
            "source_specific_importance": _safe_group_mean(
                "source_specific_importance"
            ),
            "consensus_importance_equal": _safe_group_mean(
                "consensus_importance_equal"
            ),
            "consensus_importance_median": _safe_group_mean(
                "consensus_importance_median"
            ),
            "consensus_importance_legacy_conflict_discount": _safe_group_mean(
                "consensus_importance_legacy_conflict_discount"
            ),
            "consensus_median": float(np.median(scores)) if len(scores) else np.nan,
            "consensus_std": float(np.std(scores, ddof=1))
            if len(scores) > 1
            else (0.0 if len(scores) == 1 else np.nan),
            "consensus_q25": float(np.quantile(scores, 0.25))
            if len(scores)
            else np.nan,
            "consensus_q75": float(np.quantile(scores, 0.75))
            if len(scores)
            else np.nan,
            "consensus_min": float(np.min(scores)) if len(scores) else np.nan,
            "consensus_max": float(np.max(scores)) if len(scores) else np.nan,
            "mean_partition_rank": _safe_group_mean("rank_crossfitted"),
            "min_partition_rank": float(
                pd.to_numeric(group["rank_crossfitted"], errors="coerce").min()
            ),
            "max_partition_rank": float(
                pd.to_numeric(group["rank_crossfitted"], errors="coerce").max()
            ),
            "top_k": int(top_k),
            "top_k_inclusion_frequency": _safe_group_mean(
                f"top_{int(top_k)}_crossfitted"
            ),
            "top_k_inclusion_frequency_all_partitions": _safe_group_mean(
                f"top_{int(top_k)}_crossfitted"
            ),
            "top_k_inclusion_frequency_given_consensus": (
                float(
                    pd.to_numeric(
                        group.loc[consensus_mask, f"top_{int(top_k)}_crossfitted"],
                        errors="coerce",
                    ).mean()
                )
                if bool(consensus_mask.any())
                else np.nan
            ),
            "evidence_ambiguity": _safe_group_mean("evidence_ambiguity"),
            "evidence_conflict_index": _safe_group_mean("evidence_ambiguity"),
            "source_signal_range": _safe_group_mean("source_signal_range"),
            "evidence_support": _safe_group_mean("evidence_support"),
            "evidence_reliability": _safe_group_mean("evidence_support"),
            # Retained for backward-compatible plots; this is an evidence-support
            # index and must not be interpreted as a probability of correctness.
            "evidence_confidence": _safe_group_mean("evidence_support"),
        }
        for source in sources:
            row[f"w_{source}"] = _safe_group_mean(f"w_{source}")
            row[f"w_equal_{source}"] = _safe_group_mean(f"w_equal_{source}")
            row[f"{source}_source_reliability"] = _safe_group_mean(
                f"{source}_source_reliability"
            )
            row[f"{source}_source_conflict"] = _safe_group_mean(
                f"{source}_source_conflict"
            )
            row[f"{source}_discount"] = _safe_group_mean(f"{source}_discount")
            row[f"{source}_rank_signal"] = _safe_group_mean(f"{source}_rank_signal")
            row[f"{source}_mean_importance"] = _safe_group_mean(
                f"{source}_raw_importance"
            )
            row[f"{source}_between_split_stability"] = _safe_group_mean(
                f"{source}_between_split_stability"
            )
            row[f"{source}_within_split_model_agreement"] = _safe_group_mean(
                f"{source}_within_split_model_agreement"
            )
            row[f"{source}_information_availability"] = _safe_group_mean(
                f"{source}_information_availability"
            )
        rows.append(row)

    out = pd.DataFrame(rows)
    partial = build_partial_order_levels(pairwise_df, class_ids, feature_cols)
    out = out.merge(partial, on=["class_id", "feature"], how="left")
    out["class_rank_fused"] = out.groupby("class_id")["consensus_importance"].rank(
        ascending=False, method="average"
    )
    out["class_rank_fused_adaptive"] = out["class_rank_fused"]
    out["class_rank_fused_equal"] = out.groupby("class_id")[
        "consensus_importance_equal"
    ].rank(ascending=False, method="average")
    out["class_rank"] = out["class_rank_fused"]
    return out.sort_values(
        ["class_id", "class_rank_fused", "feature"], ascending=[True, True, True]
    ).reset_index(drop=True)


def build_fusion_scheme_sensitivity(
    partition_df: pd.DataFrame, top_k: int
) -> pd.DataFrame:
    schemes = {
        "crossfitted_reliability": "consensus_importance_crossfitted",
        "equal_weight": "consensus_importance_equal",
        "median_rank": "consensus_importance_median",
        "legacy_conflict_discount": "consensus_importance_legacy_conflict_discount",
    }
    rows = []
    for class_id, class_group in partition_df.groupby("class_id"):
        aggregate = {
            scheme: class_group.groupby("feature")[column]
            .mean()
            .dropna()
            .sort_values(ascending=False)
            for scheme, column in schemes.items()
        }
        primary = aggregate["crossfitted_reliability"]
        primary_top = set(primary.head(int(top_k)).index.astype(str))
        primary_rank = primary.rank(ascending=False, method="average")
        for scheme, values in aggregate.items():
            comparison_available = bool(len(primary) and len(values))
            current_top = set(values.head(int(top_k)).index.astype(str))
            current_rank = values.rank(ascending=False, method="average")
            pair = pd.concat(
                [primary_rank, current_rank], axis=1, join="inner"
            ).dropna()
            rows.append(
                {
                    "class_id": int(class_id),
                    "class_name": TARGET_NAMES.get(int(class_id), str(class_id)),
                    "scheme": scheme,
                    "comparison_status": (
                        "available" if comparison_available else "unavailable"
                    ),
                    "top_k": int(top_k),
                    "top_features": "; ".join(
                        values.head(int(top_k)).index.astype(str)
                    ),
                    "top_k_jaccard_vs_primary": (
                        _jaccard(primary_top, current_top)
                        if comparison_available
                        else np.nan
                    ),
                    "rank_spearman_vs_primary": (
                        spearman_rank_correlation(pair.iloc[:, 0], pair.iloc[:, 1])
                        if len(pair) >= 2
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_crossfitted_source_ablation(
    partition_df: pd.DataFrame, top_k: int
) -> pd.DataFrame:
    sources = list(FUSION_SOURCE_SPECS)
    rows = []
    for class_id, class_group in partition_df.groupby("class_id"):
        primary = (
            class_group.groupby("feature")["consensus_importance_crossfitted"]
            .mean()
            .dropna()
            .sort_values(ascending=False)
        )
        primary_rank = primary.rank(ascending=False, method="average")
        primary_top = set(primary.head(int(top_k)).index.astype(str))
        for omitted in sources:
            retained = [source for source in sources if source != omitted]
            work = class_group.copy()
            retained_weight_sum = work[[f"w_{source}" for source in retained]].sum(
                axis=1
            )
            numerator = sum(
                work[f"w_{source}"] * work[f"{source}_rank_signal"]
                for source in retained
            )
            work["ablated_score"] = np.where(
                retained_weight_sum > SOURCE_WEIGHT_EPS,
                numerator / retained_weight_sum,
                np.nan,
            )
            ablated = (
                work.groupby("feature")["ablated_score"]
                .mean()
                .dropna()
                .sort_values(ascending=False)
            )
            ablated_rank = ablated.rank(ascending=False, method="average")
            pair = pd.concat(
                [primary_rank, ablated_rank], axis=1, join="inner"
            ).dropna()
            ablated_top = set(ablated.head(int(top_k)).index.astype(str))
            comparison_available = bool(len(primary) and len(ablated))
            rows.append(
                {
                    "class_id": int(class_id),
                    "class_name": TARGET_NAMES.get(int(class_id), str(class_id)),
                    "omitted_source": omitted,
                    "comparison_status": (
                        "available" if comparison_available else "unavailable"
                    ),
                    "retained_sources": "; ".join(retained),
                    "top_k": int(top_k),
                    "top_k_jaccard_vs_full": (
                        _jaccard(primary_top, ablated_top)
                        if comparison_available
                        else np.nan
                    ),
                    "rank_spearman_vs_full_consensus": (
                        spearman_rank_correlation(pair.iloc[:, 0], pair.iloc[:, 1])
                        if len(pair) >= 2
                        else np.nan
                    ),
                    "top_features_without_source": "; ".join(
                        ablated.head(int(top_k)).index.astype(str)
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_measure_sensitivity(
    balanced_summary: pd.DataFrame,
    prevalence_summary: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    rows = []
    for class_id in sorted(balanced_summary["class_id"].astype(int).unique()):
        balanced = (
            balanced_summary.loc[
                balanced_summary["class_id"].astype(int) == int(class_id)
            ]
            .set_index("feature")["consensus_importance"]
            .dropna()
        )
        prevalence = (
            prevalence_summary.loc[
                prevalence_summary["class_id"].astype(int) == int(class_id)
            ]
            .set_index("feature")["consensus_importance"]
            .dropna()
        )
        pair = pd.concat(
            [balanced.rename("balanced"), prevalence.rename("prevalence")], axis=1
        ).dropna()
        balanced_top = set(balanced.nlargest(int(top_k)).index.astype(str))
        prevalence_top = set(prevalence.nlargest(int(top_k)).index.astype(str))
        comparison_available = bool(len(balanced) and len(prevalence))
        rows.append(
            {
                "class_id": int(class_id),
                "class_name": TARGET_NAMES.get(int(class_id), str(class_id)),
                "top_k": int(top_k),
                "comparison_status": (
                    "available" if comparison_available else "unavailable"
                ),
                "balanced_top_features": "; ".join(
                    balanced.nlargest(int(top_k)).index.astype(str)
                ),
                "prevalence_top_features": "; ".join(
                    prevalence.nlargest(int(top_k)).index.astype(str)
                ),
                "top_k_jaccard": (
                    _jaccard(balanced_top, prevalence_top)
                    if comparison_available
                    else np.nan
                ),
                "rank_spearman": (
                    spearman_rank_correlation(pair["balanced"], pair["prevalence"])
                    if len(pair) >= 2
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def filter_evidence_to_partition_models(
    table: pd.DataFrame,
    model_by_partition: Dict[int, str],
    partition_col: str,
) -> pd.DataFrame:
    """Retain one prespecified model for every target outer partition."""
    if table is None or table.empty:
        return pd.DataFrame()
    if partition_col not in table.columns or "model" not in table.columns:
        raise ValueError(f"Evidence table must contain {partition_col!r} and 'model'.")
    partition_values = pd.to_numeric(table[partition_col], errors="coerce")
    expected_models = partition_values.map(
        lambda value: (
            model_by_partition.get(int(value), None) if pd.notna(value) else None
        )
    )
    mask = (
        table["model"].astype(str).to_numpy() == expected_models.astype(str).to_numpy()
    )
    out = table.loc[mask].copy()
    observed = set(
        pd.to_numeric(out[partition_col], errors="coerce").dropna().astype(int)
    )
    missing = sorted(set(int(v) for v in model_by_partition) - observed)
    if missing:
        raise RuntimeError(
            f"Best-single-model sensitivity lacks evidence for partitions: {missing}"
        )
    return out


def build_model_set_sensitivity(
    ensemble_summary: pd.DataFrame,
    single_summary: pd.DataFrame,
    model_by_partition: Dict[int, str],
    top_k: int,
) -> pd.DataFrame:
    """Compare the one-SE model set with the split-specific best single model."""
    rows = []
    for class_id in sorted(ensemble_summary["class_id"].astype(int).unique()):
        ensemble = ensemble_summary.loc[
            ensemble_summary["class_id"].astype(int) == int(class_id)
        ].set_index("feature")["consensus_importance"]
        single = single_summary.loc[
            single_summary["class_id"].astype(int) == int(class_id)
        ].set_index("feature")["consensus_importance"]
        pair = pd.concat(
            [ensemble.rename("one_se_set"), single.rename("best_single")], axis=1
        ).dropna()
        ensemble_order = ensemble.dropna().sort_values(ascending=False)
        single_order = single.dropna().sort_values(ascending=False)
        ensemble_top = set(ensemble_order.head(int(top_k)).index.astype(str))
        single_top = set(single_order.head(int(top_k)).index.astype(str))
        comparison_available = bool(len(ensemble_order) and len(single_order))
        rows.append(
            {
                "class_id": int(class_id),
                "class_name": TARGET_NAMES.get(int(class_id), str(class_id)),
                "top_k": int(top_k),
                "comparison_status": (
                    "available" if comparison_available else "unavailable"
                ),
                "one_se_model_set_top_features": "; ".join(
                    ensemble_order.head(int(top_k)).index.astype(str)
                ),
                "best_single_model_top_features": "; ".join(
                    single_order.head(int(top_k)).index.astype(str)
                ),
                "top_k_jaccard": (
                    _jaccard(ensemble_top, single_top)
                    if comparison_available
                    else np.nan
                ),
                "rank_spearman": (
                    spearman_rank_correlation(
                        pair["one_se_set"].rank(ascending=False, method="average"),
                        pair["best_single"].rank(ascending=False, method="average"),
                    )
                    if len(pair) >= 2
                    else np.nan
                ),
                "mean_absolute_score_difference": (
                    float(np.mean(np.abs(pair["one_se_set"] - pair["best_single"])))
                    if len(pair)
                    else np.nan
                ),
                "best_model_by_outer_partition": "; ".join(
                    f"{partition}:{model_by_partition[partition]}"
                    for partition in sorted(model_by_partition)
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_crossfitted_consensus(
    shap_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    loco_df: pd.DataFrame,
    firm_df: pd.DataFrame,
    class_ids: Iterable[int],
    feature_cols: List[str],
    top_k: int,
    source_specs: Optional[Dict[str, Dict[str, str]]] = None,
    source_weight_calibration_tables: Optional[Dict[str, pd.DataFrame]] = None,
    partitions: Optional[List[OuterPartition]] = None,
    allow_simulation_reference_weights: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Run source weighting, partition fusion, and uncertainty summaries."""
    specs = source_specs or FUSION_SOURCE_SPECS
    source_tables = {
        "shap": shap_df,
        "pfi": perm_df,
        "loco": loco_df,
        "firm": firm_df,
    }
    run_tables = {}
    for source, table in source_tables.items():
        run_table = prepare_source_run_table(
            table=table,
            source=source,
            value_col=specs[source]["value"],
            n_features=len(feature_cols),
        )
        if run_table.empty:
            raise RuntimeError(
                f"No complete {source} evidence was available for fusion."
            )
        run_tables[source] = run_table

    if source_weight_calibration_tables is not None:
        if partitions is None:
            raise ValueError(
                "Outer partitions are required with training-internal calibration."
            )
        weights = build_training_internal_source_weights(
            calibration_source_tables=source_weight_calibration_tables,
            class_ids=class_ids,
            feature_cols=feature_cols,
            partitions=partitions,
            source_specs=specs,
        )
    else:
        if not bool(allow_simulation_reference_weights):
            raise ValueError(
                "Training-internal source calibration is required for empirical "
                "fusion. Set allow_simulation_reference_weights=True only inside "
                "a known-truth simulation with no paper-facing data."
            )
        # The alternative path exists only for synthetic known-truth validation.
        # It is inaccessible by default and is never used by the empirical workflow.
        weights = build_crossfitted_source_weights(
            run_tables,
            class_ids,
            feature_cols,
        )
    partition = build_crossfitted_partition_consensus(
        run_tables, weights, class_ids, feature_cols, top_k=top_k
    )
    pairwise = build_pairwise_dominance_table(partition)
    summary = aggregate_partition_consensus(
        partition, pairwise, class_ids, feature_cols, top_k=top_k
    )
    scheme_sensitivity = build_fusion_scheme_sensitivity(partition, top_k=top_k)
    source_ablation = build_crossfitted_source_ablation(partition, top_k=top_k)
    return {
        "summary": summary,
        "partition": partition,
        "weights": weights,
        "pairwise": pairwise,
        "scheme_sensitivity": scheme_sensitivity,
        "source_ablation": source_ablation,
    }


def build_response_profile_summary(
    pdp_df: pd.DataFrame, ice_df: pd.DataFrame
) -> pd.DataFrame:
    """Summarize average response and centered ICE heterogeneity outside fusion."""
    pdp_summary = aggregate_pdp(pdp_df)
    ice_summary = aggregate_ice(ice_df)
    if pdp_summary.empty:
        return ice_summary
    if ice_summary.empty:
        return pdp_summary
    return pdp_summary.merge(ice_summary, on=["class_id", "feature"], how="outer")


# =============================================================================
# DATA-DRIVEN STRUCTURAL COMPATIBILITY AUDIT
# =============================================================================


PAG_ENDPOINT_LABELS = {
    -1: "tail",
    0: "null",
    1: "arrow",
    2: "circle",
    3: "star",
    4: "tail_and_arrow",
    5: "arrow_and_arrow",
}

PAG_PATTERN_ORDER = [
    "-->",
    "o->",
    "<->",
    "o-o",
    "<-o",
    "<--",
    "---",
    "--o",
    "o--",
    "other",
]


def _encode_structural_data(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: List[str],
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Encode nominal variables without imposing metric distances.

    G-square treats integer values as category labels. The numerical codes are
    therefore identifiers only; their order and spacing have no meaning.
    """
    frame = X[feature_cols].copy().reset_index(drop=True)
    frame[STRUCTURAL_TARGET_NODE] = np.asarray(y).astype(int)
    encoded_columns = []
    coding_rows = []
    for column in feature_cols + [STRUCTURAL_TARGET_NODE]:
        values = frame[column].astype("string").fillna("__STRUCTURAL_MISSING__")
        codes, levels = pd.factorize(values, sort=True)
        if np.any(codes < 0):
            raise RuntimeError(f"Structural coding failed for column {column!r}.")
        encoded_columns.append(codes.astype(np.int64))
        counts = pd.Series(codes).value_counts().sort_index()
        for code, level in enumerate(levels.astype(str).tolist()):
            coding_rows.append(
                {
                    "variable": str(column),
                    "category_code": int(code),
                    "category_value": str(level),
                    "count": int(counts.get(int(code), 0)),
                }
            )
    return np.column_stack(encoded_columns).astype(np.int64), pd.DataFrame(coding_rows)


def _structural_category_support_audit(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Describe marginal and outcome-pair support before graph discovery."""
    frame = X[feature_cols].copy().reset_index(drop=True)
    frame[STRUCTURAL_TARGET_NODE] = np.asarray(y).astype(int)
    rows = []
    target = frame[STRUCTURAL_TARGET_NODE].astype(str)
    for variable in feature_cols + [STRUCTURAL_TARGET_NODE]:
        values = frame[variable].astype("string").fillna("__STRUCTURAL_MISSING__")
        counts = values.value_counts(dropna=False)
        probabilities = counts.to_numpy(dtype=float) / max(float(counts.sum()), 1.0)
        entropy = float(
            -np.sum(
                probabilities[probabilities > 0]
                * np.log(probabilities[probabilities > 0])
            )
        )
        normalized_entropy = (
            float(entropy / np.log(len(counts))) if len(counts) > 1 else 0.0
        )
        row = {
            "variable": str(variable),
            "n_records": int(len(frame)),
            "n_levels": int(len(counts)),
            "minimum_level_count": int(counts.min()),
            "median_level_count": float(counts.median()),
            "maximum_level_count": int(counts.max()),
            "n_levels_below_5": int((counts < 5).sum()),
            "record_fraction_in_levels_below_5": float(
                counts.loc[counts < 5].sum() / max(counts.sum(), 1)
            ),
            "normalized_marginal_entropy": normalized_entropy,
            "outcome_pair_minimum_expected_count": np.nan,
            "outcome_pair_fraction_expected_below_5": np.nan,
        }
        if variable != STRUCTURAL_TARGET_NODE:
            table = pd.crosstab(values, target).to_numpy(dtype=float)
            total = float(table.sum())
            if total > 0:
                expected = table.sum(axis=1, keepdims=True) @ table.sum(
                    axis=0, keepdims=True
                )
                expected = expected / total
                row["outcome_pair_minimum_expected_count"] = float(expected.min())
                row["outcome_pair_fraction_expected_below_5"] = float(
                    np.mean(expected < 5.0)
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _pag_pattern(endpoint_left: int, endpoint_right: int) -> str:
    pair = (int(endpoint_left), int(endpoint_right))
    mapping = {
        (-1, 1): "-->",
        (2, 1): "o->",
        (1, 1): "<->",
        (2, 2): "o-o",
        (1, 2): "<-o",
        (1, -1): "<--",
        (-1, -1): "---",
        (-1, 2): "--o",
        (2, -1): "o--",
    }
    return mapping.get(pair, "other")


def _extract_pag_edges(
    graph,
    node_names: List[str],
    run_metadata: Dict[str, object],
) -> pd.DataFrame:
    matrix = np.asarray(graph.graph, dtype=int)
    if matrix.shape != (len(node_names), len(node_names)):
        raise RuntimeError(
            "FCI returned a graph matrix that does not match the supplied node names."
        )
    rows = []
    for left in range(len(node_names)):
        for right in range(left + 1, len(node_names)):
            endpoint_left = int(matrix[left, right])
            endpoint_right = int(matrix[right, left])
            if endpoint_left == 0 and endpoint_right == 0:
                continue
            row = dict(run_metadata)
            row.update(
                {
                    "node_left": str(node_names[left]),
                    "node_right": str(node_names[right]),
                    "endpoint_left_code": endpoint_left,
                    "endpoint_right_code": endpoint_right,
                    "endpoint_left": PAG_ENDPOINT_LABELS.get(
                        endpoint_left, f"code_{endpoint_left}"
                    ),
                    "endpoint_right": PAG_ENDPOINT_LABELS.get(
                        endpoint_right, f"code_{endpoint_right}"
                    ),
                    "pag_pattern_left_to_right": _pag_pattern(
                        endpoint_left, endpoint_right
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _target_rows_from_graph(
    graph,
    feature_cols: List[str],
    run_metadata: Dict[str, object],
) -> pd.DataFrame:
    node_names = feature_cols + [STRUCTURAL_TARGET_NODE]
    matrix = np.asarray(graph.graph, dtype=int)
    target_index = len(node_names) - 1
    rows = []
    for feature_index, feature in enumerate(feature_cols):
        endpoint_feature = int(matrix[feature_index, target_index])
        endpoint_outcome = int(matrix[target_index, feature_index])
        adjacent = int(not (endpoint_feature == 0 and endpoint_outcome == 0))
        row = dict(run_metadata)
        row.update(
            {
                "feature": str(feature),
                "adjacent_to_outcome": adjacent,
                "endpoint_at_feature_code": endpoint_feature,
                "endpoint_at_outcome_code": endpoint_outcome,
                "endpoint_at_feature": PAG_ENDPOINT_LABELS.get(
                    endpoint_feature, f"code_{endpoint_feature}"
                ),
                "endpoint_at_outcome": PAG_ENDPOINT_LABELS.get(
                    endpoint_outcome, f"code_{endpoint_outcome}"
                ),
                "outcome_edge_pattern": (
                    _pag_pattern(endpoint_feature, endpoint_outcome)
                    if adjacent
                    else "not_adjacent"
                ),
                "arrowhead_at_outcome": int(adjacent and endpoint_outcome == 1),
                "arrowhead_at_feature": int(adjacent and endpoint_feature == 1),
                "circle_at_outcome": int(adjacent and endpoint_outcome == 2),
                "circle_at_feature": int(adjacent and endpoint_feature == 2),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _run_structural_fci(
    encoded_data: np.ndarray,
    node_names: List[str],
    alpha: float,
    depth: int,
):
    """Run one FCI specification and fail closed on incomplete output."""
    from causallearn.search.ConstraintBased.FCI import fci

    stdout_buffer = io.StringIO()
    try:
        if STRUCTURAL_SHOW_FCI_PROGRESS:
            graph, edges = fci(
                encoded_data,
                independence_test_method=STRUCTURAL_CI_TEST,
                alpha=float(alpha),
                depth=int(depth),
                max_path_length=int(STRUCTURAL_MAX_PATH_LENGTH),
                verbose=False,
                show_progress=True,
                node_names=list(node_names),
            )
        else:
            import contextlib

            with contextlib.redirect_stdout(stdout_buffer):
                graph, edges = fci(
                    encoded_data,
                    independence_test_method=STRUCTURAL_CI_TEST,
                    alpha=float(alpha),
                    depth=int(depth),
                    max_path_length=int(STRUCTURAL_MAX_PATH_LENGTH),
                    verbose=False,
                    show_progress=False,
                    node_names=list(node_names),
                )
    except Exception as exc:
        raise RuntimeError(
            "FCI failed for the structural compatibility audit "
            f"(alpha={alpha}, depth={depth}, n={len(encoded_data)})."
        ) from exc
    if graph is None or not hasattr(graph, "graph"):
        raise RuntimeError("FCI did not return a valid PAG.")
    return graph, edges, stdout_buffer.getvalue()


def _structural_setting_label(alpha: float, depth: int) -> str:
    depth_label = "unlimited" if int(depth) < 0 else str(int(depth))
    return f"alpha={float(alpha):.2f}|depth={depth_label}"


def _normalized_pattern_entropy(patterns: pd.Series) -> float:
    counts = patterns.value_counts()
    if counts.empty:
        return np.nan
    probabilities = counts.to_numpy(dtype=float) / float(counts.sum())
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    return float(entropy / np.log(len(PAG_PATTERN_ORDER)))


def _aggregate_structural_evidence(
    observed_target_rows: pd.DataFrame,
    null_target_rows: pd.DataFrame,
    full_target_rows: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observed_outer = observed_target_rows.loc[
        observed_target_rows["scope"].astype(str) == "outer_training"
    ].copy()
    if observed_outer.empty:
        raise RuntimeError("No outer-training structural evidence was produced.")

    setting_summary = (
        observed_outer.groupby(
            ["feature", "alpha", "depth", "setting"], as_index=False
        )
        .agg(
            adjacency_frequency=("adjacent_to_outcome", "mean"),
            n_outer_partitions=("partition_id", "nunique"),
            n_runs=("adjacent_to_outcome", "size"),
        )
        .sort_values(["feature", "alpha", "depth"])
        .reset_index(drop=True)
    )

    partition_summary = (
        observed_outer.groupby(["feature", "partition_id"], as_index=False)
        .agg(
            partition_adjacency_frequency=("adjacent_to_outcome", "mean"),
            n_design_settings=("setting", "nunique"),
        )
        .reset_index(drop=True)
    )

    null_replicate_summary = pd.DataFrame()
    if null_target_rows is not None and not null_target_rows.empty:
        null_replicate_summary = (
            null_target_rows.groupby(["feature", "null_replicate"], as_index=False)
            .agg(
                null_adjacency_frequency=("adjacent_to_outcome", "mean"),
                n_outer_partitions=("partition_id", "nunique"),
                n_design_settings=("setting", "nunique"),
            )
            .reset_index(drop=True)
        )

    rows = []
    for feature in feature_cols:
        feature_runs = observed_outer.loc[
            observed_outer["feature"].astype(str) == str(feature)
        ]
        feature_partitions = partition_summary.loc[
            partition_summary["feature"].astype(str) == str(feature)
        ]
        adjacent_runs = feature_runs.loc[
            feature_runs["adjacent_to_outcome"].astype(int) == 1
        ]
        pattern_counts = adjacent_runs["outcome_edge_pattern"].value_counts()
        dominant_pattern = (
            str(pattern_counts.index[0]) if not pattern_counts.empty else "not_adjacent"
        )
        dominant_pattern_fraction = (
            float(pattern_counts.iloc[0] / pattern_counts.sum())
            if not pattern_counts.empty
            else np.nan
        )

        full_feature = full_target_rows.loc[
            full_target_rows["feature"].astype(str) == str(feature)
        ]
        null_feature = (
            null_replicate_summary.loc[
                null_replicate_summary["feature"].astype(str) == str(feature)
            ]
            if not null_replicate_summary.empty
            else pd.DataFrame()
        )
        null_values = (
            null_feature["null_adjacency_frequency"].to_numpy(dtype=float)
            if not null_feature.empty
            else np.asarray([], dtype=float)
        )

        matched_differences = []
        if null_target_rows is not None and not null_target_rows.empty:
            observed_by_partition = feature_partitions.set_index("partition_id")[
                "partition_adjacency_frequency"
            ]
            null_by_partition = (
                null_target_rows.loc[
                    null_target_rows["feature"].astype(str) == str(feature)
                ]
                .groupby("partition_id")["adjacent_to_outcome"]
                .mean()
            )
            common = observed_by_partition.index.intersection(null_by_partition.index)
            matched_differences = (
                observed_by_partition.loc[common] - null_by_partition.loc[common]
            ).to_numpy(dtype=float)

        observed_frequency = float(feature_runs["adjacent_to_outcome"].mean())
        null_mean = float(np.mean(null_values)) if len(null_values) else np.nan
        null_q95 = (
            float(np.quantile(null_values, 0.95)) if len(null_values) else np.nan
        )
        rows.append(
            {
                "feature": str(feature),
                "structural_adjacency_frequency": observed_frequency,
                "partition_adjacency_mean": float(
                    feature_partitions["partition_adjacency_frequency"].mean()
                ),
                "partition_adjacency_std": (
                    float(
                        feature_partitions["partition_adjacency_frequency"].std(ddof=1)
                    )
                    if len(feature_partitions) > 1
                    else 0.0
                ),
                "partition_adjacency_q25": float(
                    feature_partitions["partition_adjacency_frequency"].quantile(0.25)
                ),
                "partition_adjacency_q75": float(
                    feature_partitions["partition_adjacency_frequency"].quantile(0.75)
                ),
                "partition_presence_frequency": float(
                    np.mean(
                        feature_partitions["partition_adjacency_frequency"].to_numpy(
                            dtype=float
                        )
                        > 0.0
                    )
                ),
                "full_data_design_persistence": (
                    float(full_feature["adjacent_to_outcome"].mean())
                    if not full_feature.empty
                    else np.nan
                ),
                "null_adjacency_mean": null_mean,
                "null_adjacency_q50": (
                    float(np.quantile(null_values, 0.50))
                    if len(null_values)
                    else np.nan
                ),
                "null_adjacency_q95": null_q95,
                "null_adjacency_max": (
                    float(np.max(null_values)) if len(null_values) else np.nan
                ),
                "excess_adjacency_over_null": (
                    float(observed_frequency - null_mean)
                    if np.isfinite(null_mean)
                    else np.nan
                ),
                "observed_above_null_q95_descriptive": (
                    int(observed_frequency > null_q95)
                    if np.isfinite(null_q95)
                    else np.nan
                ),
                "descriptive_null_replicate_exceedance_fraction": (
                    float(np.mean(null_values >= observed_frequency))
                    if len(null_values)
                    else np.nan
                ),
                "matched_partition_difference_mean": (
                    float(np.mean(matched_differences))
                    if len(matched_differences)
                    else np.nan
                ),
                "matched_partition_positive_fraction_descriptive": (
                    float(np.mean(np.asarray(matched_differences) > 0.0))
                    if len(matched_differences)
                    else np.nan
                ),
                "arrowhead_at_outcome_frequency_given_adjacent": (
                    float(adjacent_runs["arrowhead_at_outcome"].mean())
                    if not adjacent_runs.empty
                    else np.nan
                ),
                "arrowhead_at_feature_frequency_given_adjacent": (
                    float(adjacent_runs["arrowhead_at_feature"].mean())
                    if not adjacent_runs.empty
                    else np.nan
                ),
                "dominant_endpoint_pattern": dominant_pattern,
                "dominant_endpoint_pattern_fraction": dominant_pattern_fraction,
                "endpoint_pattern_entropy_normalized": _normalized_pattern_entropy(
                    adjacent_runs["outcome_edge_pattern"]
                ),
                "n_observed_runs": int(len(feature_runs)),
                "n_adjacent_runs": int(
                    feature_runs["adjacent_to_outcome"].astype(int).sum()
                ),
                "n_outer_partitions": int(feature_runs["partition_id"].nunique()),
                "n_design_settings": int(feature_runs["setting"].nunique()),
                "n_null_replicates": int(len(null_values)),
                "direction_claim_allowed": 0,
                "causal_effect_estimated": 0,
                "ranking_modified_by_structure": 0,
            }
        )
    evidence = pd.DataFrame(rows).sort_values(
        [
            "excess_adjacency_over_null",
            "structural_adjacency_frequency",
            "feature",
        ],
        ascending=[False, False, True],
        na_position="last",
    )
    return evidence.reset_index(drop=True), setting_summary, null_replicate_summary


def _build_endpoint_pattern_summary(
    target_rows: pd.DataFrame,
) -> pd.DataFrame:
    adjacent = target_rows.loc[
        (target_rows["scope"].astype(str) == "outer_training")
        & (target_rows["adjacent_to_outcome"].astype(int) == 1)
    ].copy()
    if adjacent.empty:
        return pd.DataFrame(
            columns=["feature", "outcome_edge_pattern", "count", "fraction"]
        )
    counts = (
        adjacent.groupby(["feature", "outcome_edge_pattern"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    totals = counts.groupby("feature")["count"].transform("sum")
    counts["fraction"] = counts["count"] / totals
    return counts.sort_values(["feature", "fraction"], ascending=[True, False])


def _build_predictive_structural_matrix(
    reliability_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
) -> pd.DataFrame:
    matrix = reliability_df.merge(
        evidence_df,
        on="feature",
        how="left",
        validate="many_to_one",
        suffixes=("", "_structural"),
    )
    matrix["structural_scope"] = (
        "three-level injury outcome; not class-specific causal validation"
    )
    matrix["interpretation_boundary"] = (
        "Predictive relevance and local PAG adjacency are complementary, "
        "non-independent observational summaries. Neither is a causal effect."
    )
    matrix["ranking_modified_by_structure"] = 0
    return matrix


def _save_structural_compatibility_figures(
    evidence_df: pd.DataFrame,
    setting_summary: pd.DataFrame,
    endpoint_summary: pd.DataFrame,
    predictive_structural_matrix: pd.DataFrame,
    output_dir: Path,
) -> None:
    if not _has_matplotlib() or evidence_df.empty:
        return
    import matplotlib.pyplot as plt

    _setup_publication_style()
    figure_dir = Path(output_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    ordered = evidence_df.sort_values(
        ["structural_adjacency_frequency", "feature"],
        ascending=[True, True],
    ).reset_index(drop=True)
    y_pos = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(7.4, max(4.6, 0.36 * len(ordered) + 1.2)))
    null_q95 = ordered["null_adjacency_q95"].to_numpy(dtype=float)
    null_mean = ordered["null_adjacency_mean"].to_numpy(dtype=float)
    observed = ordered["structural_adjacency_frequency"].to_numpy(dtype=float)
    ax.hlines(
        y_pos,
        np.nan_to_num(null_mean, nan=0.0),
        observed,
        color=FIG_COLORS["light_gray"],
        linewidth=2.0,
        zorder=1,
    )
    ax.scatter(
        null_mean,
        y_pos,
        marker="s",
        s=38,
        facecolor="white",
        edgecolor=FIG_COLORS["gray"],
        linewidth=1.1,
        label="Permuted-outcome mean",
        zorder=3,
    )
    ax.scatter(
        observed,
        y_pos,
        marker="o",
        s=44,
        color=FIG_COLORS["blue"],
        edgecolor="white",
        linewidth=0.5,
        label="Observed outcome",
        zorder=4,
    )
    for idx, q95 in enumerate(null_q95):
        if np.isfinite(q95):
            ax.plot(
                [q95, q95],
                [idx - 0.15, idx + 0.15],
                color=FIG_COLORS["coral"],
                linewidth=1.0,
                zorder=2,
            )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_feature_label(value) for value in ordered["feature"]])
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("PAG adjacency frequency across partitions and settings")
    ax.legend(frameon=False, loc="lower right")
    _clean_axis(ax, grid_axis="x")
    _save_paper_figure(fig, figure_dir / "structural_adjacency_vs_null")
    plt.close(fig)

    if setting_summary is not None and not setting_summary.empty:
        settings = (
            setting_summary[["alpha", "depth", "setting"]]
            .drop_duplicates()
            .sort_values(["depth", "alpha"])
        )
        feature_order = evidence_df.sort_values(
            ["structural_adjacency_frequency", "feature"],
            ascending=[False, True],
        )["feature"].astype(str).tolist()
        matrix = (
            setting_summary.pivot(
                index="feature", columns="setting", values="adjacency_frequency"
            )
            .reindex(index=feature_order, columns=settings["setting"].tolist())
            .fillna(0.0)
        )
        fig, ax = plt.subplots(
            figsize=(8.2, max(4.8, 0.34 * len(feature_order) + 1.2))
        )
        image = ax.imshow(
            matrix.to_numpy(dtype=float),
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_yticklabels([_feature_label(value) for value in matrix.index])
        ax.set_xticks(np.arange(len(matrix.columns)))
        ax.set_xticklabels(
            [
                value.replace("|", "\n").replace("alpha=", "a=")
                for value in matrix.columns
            ],
            rotation=45,
            ha="right",
        )
        colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
        colorbar.set_label("Outer-partition adjacency frequency")
        _save_paper_figure(fig, figure_dir / "structural_multiverse_heatmap")
        plt.close(fig)

    if endpoint_summary is not None and not endpoint_summary.empty:
        feature_order = (
            evidence_df.loc[evidence_df["n_adjacent_runs"].astype(int) > 0]
            .sort_values(
                ["structural_adjacency_frequency", "feature"],
                ascending=[False, True],
            )["feature"]
            .astype(str)
            .tolist()
        )
        pivot = (
            endpoint_summary.pivot(
                index="feature", columns="outcome_edge_pattern", values="fraction"
            )
            .reindex(index=feature_order)
            .fillna(0.0)
        )
        pattern_columns = [
            value for value in PAG_PATTERN_ORDER if value in pivot.columns
        ]
        pivot = pivot.reindex(columns=pattern_columns, fill_value=0.0)
        colors = [
            FIG_COLORS["blue"],
            FIG_COLORS["cyan"],
            FIG_COLORS["coral"],
            FIG_COLORS["gold"],
            FIG_COLORS["orange"],
            FIG_COLORS["teal"],
            FIG_COLORS["gray"],
            "#7A5195",
            "#EF5675",
            FIG_COLORS["light_gray"],
        ]
        fig, ax = plt.subplots(
            figsize=(7.8, max(4.3, 0.34 * max(len(pivot), 1) + 1.2))
        )
        left = np.zeros(len(pivot), dtype=float)
        for color, pattern in zip(colors, pattern_columns):
            values = pivot[pattern].to_numpy(dtype=float)
            ax.barh(
                np.arange(len(pivot)),
                values,
                left=left,
                color=color,
                edgecolor="white",
                linewidth=0.3,
                label=pattern,
            )
            left += values
        ax.set_yticks(np.arange(len(pivot)))
        ax.set_yticklabels([_feature_label(value) for value in pivot.index])
        ax.invert_yaxis()
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Endpoint-pattern fraction conditional on adjacency")
        ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13))
        _clean_axis(ax, grid_axis="x")
        _save_paper_figure(fig, figure_dir / "structural_endpoint_patterns")
        plt.close(fig)

    if (
        predictive_structural_matrix is not None
        and not predictive_structural_matrix.empty
    ):
        predictive_value_col = (
            "lower_expected_percentile"
            if "lower_expected_percentile" in predictive_structural_matrix.columns
            else "consensus_importance"
        )
        predictive_order_col = (
            "partial_order_level"
            if "partial_order_level" in predictive_structural_matrix.columns
            else "class_rank_fused"
        )
        class_ids = sorted(
            predictive_structural_matrix["class_id"].dropna().astype(int).unique()
        )
        fig, axes = plt.subplots(
            1,
            len(class_ids),
            figsize=(5.0 * max(len(class_ids), 1), 4.6),
            squeeze=False,
            sharex=True,
        )
        for axis_index, class_id in enumerate(class_ids):
            ax = axes[0, axis_index]
            subset = predictive_structural_matrix.loc[
                predictive_structural_matrix["class_id"].astype(int) == int(class_id)
            ].copy()
            ax.scatter(
                subset["structural_adjacency_frequency"],
                subset[predictive_value_col],
                s=46,
                color=FIG_CLASS_COLORS.get(int(class_id), FIG_COLORS["blue"]),
                edgecolor="white",
                linewidth=0.5,
                alpha=0.9,
            )
            label_subset = subset.nsmallest(
                min(DISPLAY_TOP_K_FEATURES, len(subset)), predictive_order_col
            )
            for _, row in label_subset.iterrows():
                ax.annotate(
                    _feature_label(str(row["feature"]), max_len=16),
                    (
                        float(row["structural_adjacency_frequency"]),
                        float(row[predictive_value_col]),
                    ),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=7.2,
                )
            ax.set_title(_class_label(class_id))
            ax.set_xlabel("Structural adjacency frequency")
            if axis_index == 0:
                ax.set_ylabel(
                    "Lower expected relevance percentile"
                    if predictive_value_col == "lower_expected_percentile"
                    else "Cross-fitted predictive consensus"
                )
            ax.set_xlim(-0.03, 1.03)
            _clean_axis(ax, grid_axis="both")
        _save_paper_figure(fig, figure_dir / "predictive_structural_compatibility")
        plt.close(fig)

    network = evidence_df.loc[
        evidence_df["structural_adjacency_frequency"].astype(float) > 0.0
    ].copy()
    if not network.empty:
        network = network.sort_values(
            ["structural_adjacency_frequency", "feature"],
            ascending=[False, True],
        ).reset_index(drop=True)
        angles = np.linspace(0, 2 * np.pi, len(network), endpoint=False)
        fig, ax = plt.subplots(figsize=(7.2, 7.2))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.scatter(
            [0.0],
            [0.0],
            s=1000,
            color=FIG_COLORS["dark"],
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
        )
        ax.text(
            0.0,
            0.0,
            "Injury\noutcome",
            color="white",
            ha="center",
            va="center",
            fontsize=9,
            zorder=6,
        )
        for angle, (_, row) in zip(angles, network.iterrows()):
            x_coord, y_coord = np.cos(angle), np.sin(angle)
            frequency = float(row["structural_adjacency_frequency"])
            excess = float(row["excess_adjacency_over_null"])
            color = FIG_COLORS["blue"] if excess > 0 else FIG_COLORS["gray"]
            ax.plot(
                [0.0, x_coord],
                [0.0, y_coord],
                color=color,
                linewidth=0.7 + 4.0 * frequency,
                alpha=0.78,
                zorder=1,
            )
            ax.scatter(
                [x_coord],
                [y_coord],
                s=145,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
            alignment = "left" if x_coord >= 0 else "right"
            ax.text(
                x_coord + (0.05 if x_coord >= 0 else -0.05),
                y_coord,
                _feature_label(str(row["feature"]), max_len=15),
                ha=alignment,
                va="center",
                fontsize=8,
            )
        ax.set_xlim(-1.48, 1.48)
        ax.set_ylim(-1.35, 1.35)
        _save_paper_figure(fig, figure_dir / "structural_target_neighborhood")
        plt.close(fig)


def _structural_group_checkpoint_paths(
    checkpoint_root: Path,
    task_key: str,
) -> Tuple[Path, Path]:
    directory = Path(checkpoint_root)
    directory.mkdir(parents=True, exist_ok=True)
    safe_key = _safe_path_component(task_key, limit=80)
    digest = stable_json_digest(task_key)[:16]
    stem = f"{safe_key}_{digest}"
    return directory / f"{stem}.pkl", directory / f"{stem}.json"


def _structural_group_payload_complete(
    payload: Mapping[str, Any],
    feature_cols: Sequence[str],
    expected_null_replicates: int,
) -> bool:
    target = payload.get("target_rows", pd.DataFrame())
    null_target = payload.get("null_target_rows", pd.DataFrame())
    run_rows = payload.get("run_rows", pd.DataFrame())
    if not isinstance(target, pd.DataFrame) or target.empty:
        return False
    if set(target["feature"].astype(str)) != {
        str(feature) for feature in feature_cols
    }:
        return False
    if not isinstance(run_rows, pd.DataFrame) or run_rows.empty:
        return False
    if int(expected_null_replicates) > 0:
        if (
            not isinstance(null_target, pd.DataFrame)
            or null_target.empty
        ):
            return False
        observed_nulls = set(
            pd.to_numeric(
                null_target["null_replicate"], errors="coerce"
            )
            .dropna()
            .astype(int)
        )
        if observed_nulls != set(
            range(1, int(expected_null_replicates) + 1)
        ):
            return False
        expected_rows = int(expected_null_replicates) * len(
            feature_cols
        )
        if len(null_target) != expected_rows:
            return False
    return True


def _run_structural_group_task(
    task: Mapping[str, Any],
) -> Dict[str, Any]:
    """Run one partition-setting FCI group and its matched label nulls."""
    payload_path = Path(str(task["payload_path"]))
    marker_path = Path(str(task["marker_path"]))
    fingerprint = str(task["fingerprint"])
    feature_cols = [str(value) for value in task["feature_cols"]]
    expected_nulls = int(task["n_null_replicates"])
    try:
        if marker_path.exists():
            marker = json.loads(
                marker_path.read_text(encoding="utf-8")
            )
            if marker.get("fingerprint") != fingerprint:
                raise RuntimeError(
                    "Incompatible structural group checkpoint."
                )
            if not payload_path.exists():
                raise RuntimeError(
                    "Structural checkpoint marker lacks payload."
                )
            if file_sha256(str(payload_path)) != marker.get(
                "payload_sha256"
            ):
                raise RuntimeError(
                    "Structural group checkpoint checksum mismatch."
                )
            payload = pd.read_pickle(payload_path)
            if (
                isinstance(payload, Mapping)
                and _structural_group_payload_complete(
                    payload,
                    feature_cols=feature_cols,
                    expected_null_replicates=expected_nulls,
                )
            ):
                return {
                    "status": "checkpoint",
                    "task_key": str(task["task_key"]),
                    "payload": dict(payload),
                }

        encoded = np.asarray(task["encoded_data"])
        node_names = [str(value) for value in task["node_names"]]
        metadata = dict(task["metadata"])
        alpha = float(task["alpha"])
        depth = int(task["depth"])
        graph, edges, captured_stdout = _run_structural_fci(
            encoded,
            node_names=node_names,
            alpha=alpha,
            depth=depth,
        )
        target_rows = _target_rows_from_graph(
            graph, feature_cols, metadata
        )
        edge_rows = _extract_pag_edges(
            graph, node_names, metadata
        )
        run_rows = [
            {
                **metadata,
                "n_pag_edges": int(len(edges)),
                "target_degree": int(
                    target_rows["adjacent_to_outcome"]
                    .astype(int)
                    .sum()
                ),
                "captured_stdout_characters": int(
                    len(captured_stdout)
                ),
                "run_status": "complete",
            }
        ]

        null_parts: List[pd.DataFrame] = []
        if expected_nulls > 0:
            y_observed = np.asarray(
                task["observed_outcome"], dtype=int
            )
            partition_id = int(metadata["partition_id"])
            for null_replicate in range(
                1, expected_nulls + 1
            ):
                null_seed = stable_seed_from_key(
                    "structural_null",
                    STRUCTURAL_RANDOM_SEED,
                    partition_id,
                    int(null_replicate),
                )
                permuted_y = np.random.RandomState(
                    int(null_seed)
                ).permutation(y_observed)
                null_encoded = encoded.copy()
                null_encoded[:, -1] = np.asarray(
                    permuted_y, dtype=np.int64
                )
                null_metadata = dict(metadata)
                null_metadata.update(
                    {
                        "null_replicate": int(
                            null_replicate
                        ),
                        "outcome_status": (
                            "permuted_negative_control"
                        ),
                    }
                )
                null_graph, null_edges, null_stdout = (
                    _run_structural_fci(
                        null_encoded,
                        node_names=node_names,
                        alpha=alpha,
                        depth=depth,
                    )
                )
                null_rows = _target_rows_from_graph(
                    null_graph, feature_cols, null_metadata
                )
                null_parts.append(null_rows)
                run_rows.append(
                    {
                        **null_metadata,
                        "n_pag_edges": int(len(null_edges)),
                        "target_degree": int(
                            null_rows["adjacent_to_outcome"]
                            .astype(int)
                            .sum()
                        ),
                        "captured_stdout_characters": int(
                            len(null_stdout)
                        ),
                        "run_status": "complete",
                    }
                )
        payload = {
            "target_rows": target_rows,
            "edge_rows": edge_rows,
            "null_target_rows": concat_or_empty(null_parts),
            "run_rows": pd.DataFrame(run_rows),
            "task_key": str(task["task_key"]),
            "fingerprint": fingerprint,
        }
        if not _structural_group_payload_complete(
            payload,
            feature_cols=feature_cols,
            expected_null_replicates=expected_nulls,
        ):
            raise RuntimeError(
                "Structural group produced an incomplete payload."
            )
        if ENABLE_STRUCTURAL_TASK_CHECKPOINTS:
            atomic_pickle_dump(payload, payload_path)
            atomic_json_dump(
                {
                    "format_version": 1,
                    "contract_version": (
                        STRUCTURAL_TASK_CONTRACT_VERSION
                    ),
                    "fingerprint": fingerprint,
                    "payload_sha256": file_sha256(
                        str(payload_path)
                    ),
                    "completed_at": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                },
                marker_path,
            )
        return {
            "status": "completed",
            "task_key": str(task["task_key"]),
            "payload": payload,
        }
    except BaseException as exc:
        return {
            "status": "fatal_error",
            "task_key": str(task.get("task_key", "unknown")),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    finally:
        gc.collect()


def run_structural_compatibility_audit(
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    feature_cols: List[str],
    partitions: List[OuterPartition],
    reliability_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, object]:
    """Audit predictive explanations against data-derived PAG neighborhoods.

    The audit is intentionally non-causal in its reported estimand. FCI is used
    to summarize conditional-dependence structure under latent-variable-aware
    graphical assumptions. Its endpoint marks are exported, but neither those
    marks nor adjacency are presented as treatment effects or mechanisms.
    """
    audit_dir = Path(output_dir) / "structural_compatibility_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    node_names = list(feature_cols) + [STRUCTURAL_TARGET_NODE]
    alpha_grid = sorted({float(value) for value in STRUCTURAL_ALPHA_GRID})
    depth_grid = list(dict.fromkeys(int(value) for value in STRUCTURAL_DEPTH_GRID))
    if not alpha_grid or not depth_grid:
        raise ValueError("The structural multiverse must contain alpha and depth values.")
    if any(not (0.0 < value < 1.0) for value in alpha_grid):
        raise ValueError("Every structural alpha must lie strictly between zero and one.")
    if any(value == 0 or value < -1 for value in depth_grid):
        raise ValueError("Structural depths must be positive integers or -1.")

    n_null_replicates = int(
        STRUCTURAL_NULL_REPLICATES_SMOKE
        if RUN_PROFILE.lower() == "smoke"
        else STRUCTURAL_NULL_REPLICATES_FULL
    )
    if n_null_replicates < 1:
        raise ValueError("At least one structural negative-control replicate is required.")

    category_support = _structural_category_support_audit(
        X_all, y_all, feature_cols
    )
    full_encoded, full_coding = _encode_structural_data(
        X_all, y_all, feature_cols
    )

    observed_target_parts: List[pd.DataFrame] = []
    observed_edge_parts: List[pd.DataFrame] = []
    null_target_parts: List[pd.DataFrame] = []
    full_target_parts: List[pd.DataFrame] = []
    full_edge_parts: List[pd.DataFrame] = []
    run_parts: List[pd.DataFrame] = []
    structural_checkpoint_root = (
        audit_dir / "checkpoints" / "fci_groups"
    )
    tasks: List[Dict[str, Any]] = []

    cprint(
        "[STRUCTURAL AUDIT] "
        f"outer_partitions={len(partitions)} | "
        f"settings={len(alpha_grid) * len(depth_grid)} "
        f"| null_replicates={n_null_replicates} "
        f"| workers={int(STRUCTURAL_PARALLEL_WORKERS)}"
    )
    for partition in partitions:
        train_idx = np.asarray(partition.train_idx, dtype=int)
        X_train = X_all.iloc[train_idx].reset_index(drop=True)
        y_train = np.asarray(y_all)[train_idx].astype(int)
        encoded_train, _ = _encode_structural_data(
            X_train, y_train, feature_cols
        )
        for alpha in alpha_grid:
            for depth in depth_grid:
                setting = _structural_setting_label(alpha, depth)
                metadata = {
                    "scope": "outer_training",
                    "partition_id": int(partition.partition_id),
                    "outer_repeat": int(partition.repeat_id),
                    "outer_fold": int(partition.fold_id),
                    "split_seed": int(partition.split_seed),
                    "alpha": float(alpha),
                    "depth": int(depth),
                    "setting": setting,
                    "null_replicate": 0,
                    "outcome_status": "observed",
                    "n_records": int(len(encoded_train)),
                }
                task_key = (
                    f"outer_p{int(partition.partition_id):03d}_"
                    f"a{float(alpha):.4f}_d{int(depth)}"
                )
                payload_path, marker_path = (
                    _structural_group_checkpoint_paths(
                        structural_checkpoint_root, task_key
                    )
                )
                fingerprint = stable_json_digest(
                    {
                        "contract": (
                            STRUCTURAL_TASK_CONTRACT_VERSION
                        ),
                        "task_key": task_key,
                        "encoded_data": array_digest(
                            np.asarray(encoded_train)
                        ),
                        "observed_outcome": array_digest(
                            np.asarray(y_train, dtype=np.int64)
                        ),
                        "node_names": node_names,
                        "metadata": metadata,
                        "null_replicates": int(
                            n_null_replicates
                        ),
                        "structural_random_seed": int(
                            STRUCTURAL_RANDOM_SEED
                        ),
                        "null_seed_contract": (
                            "partition_replicate_label_permutation_v1"
                        ),
                        "ci_test": STRUCTURAL_CI_TEST,
                        "max_path_length": int(
                            STRUCTURAL_MAX_PATH_LENGTH
                        ),
                        "causal_learn_version": package_version(
                            "causal-learn"
                        ),
                    }
                )
                tasks.append(
                    {
                        "task_key": task_key,
                        "payload_path": str(payload_path),
                        "marker_path": str(marker_path),
                        "fingerprint": fingerprint,
                        "encoded_data": encoded_train,
                        "observed_outcome": y_train,
                        "feature_cols": list(feature_cols),
                        "node_names": node_names,
                        "metadata": metadata,
                        "alpha": float(alpha),
                        "depth": int(depth),
                        "n_null_replicates": int(
                            n_null_replicates
                        ),
                    }
                )

    for alpha in alpha_grid:
        for depth in depth_grid:
            metadata = {
                "scope": "full_data_descriptive",
                "partition_id": 0,
                "outer_repeat": 0,
                "outer_fold": 0,
                "split_seed": int(STRUCTURAL_RANDOM_SEED),
                "alpha": float(alpha),
                "depth": int(depth),
                "setting": _structural_setting_label(
                    alpha, depth
                ),
                "null_replicate": 0,
                "outcome_status": "observed",
                "n_records": int(len(full_encoded)),
            }
            task_key = (
                f"full_a{float(alpha):.4f}_d{int(depth)}"
            )
            payload_path, marker_path = (
                _structural_group_checkpoint_paths(
                    structural_checkpoint_root, task_key
                )
            )
            fingerprint = stable_json_digest(
                {
                    "contract": STRUCTURAL_TASK_CONTRACT_VERSION,
                    "task_key": task_key,
                    "encoded_data": array_digest(
                        np.asarray(full_encoded)
                    ),
                    "node_names": node_names,
                    "metadata": metadata,
                    "null_replicates": 0,
                    "structural_random_seed": int(
                        STRUCTURAL_RANDOM_SEED
                    ),
                    "null_seed_contract": (
                        "partition_replicate_label_permutation_v1"
                    ),
                    "ci_test": STRUCTURAL_CI_TEST,
                    "max_path_length": int(
                        STRUCTURAL_MAX_PATH_LENGTH
                    ),
                    "causal_learn_version": package_version(
                        "causal-learn"
                    ),
                }
            )
            tasks.append(
                {
                    "task_key": task_key,
                    "payload_path": str(payload_path),
                    "marker_path": str(marker_path),
                    "fingerprint": fingerprint,
                    "encoded_data": full_encoded,
                    "observed_outcome": np.asarray(
                        y_all, dtype=int
                    ),
                    "feature_cols": list(feature_cols),
                    "node_names": node_names,
                    "metadata": metadata,
                    "alpha": float(alpha),
                    "depth": int(depth),
                    "n_null_replicates": 0,
                }
            )

    max_workers = max(
        1, min(int(STRUCTURAL_PARALLEL_WORKERS), len(tasks))
    )
    max_pending = max(
        max_workers, int(STRUCTURAL_MAX_PENDING_TASKS)
    )
    task_iterator = iter(tasks)
    process_context = mp.get_context("spawn")
    completed_groups = 0
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=process_context,
    ) as executor:
        future_to_task = {}
        for _ in range(min(max_pending, len(tasks))):
            task = next(task_iterator, None)
            if task is None:
                break
            future_to_task[
                executor.submit(_run_structural_group_task, task)
            ] = task
        while future_to_task:
            done, _ = wait(
                future_to_task, return_when=FIRST_COMPLETED
            )
            for future in done:
                task = future_to_task.pop(future)
                result = future.result()
                if result.get("status") == "fatal_error":
                    raise RuntimeError(
                        "Parallel structural group failed: "
                        f"{result.get('task_key')}. "
                        f"{result.get('error')}\n"
                        f"{result.get('traceback', '')}"
                    )
                payload = result.get("payload")
                if not isinstance(payload, Mapping):
                    raise RuntimeError(
                        "Structural group returned no payload."
                    )
                scope = str(
                    payload["target_rows"]["scope"].iloc[0]
                )
                if scope == "outer_training":
                    observed_target_parts.append(
                        payload["target_rows"]
                    )
                    edge_rows = payload.get(
                        "edge_rows", pd.DataFrame()
                    )
                    if isinstance(edge_rows, pd.DataFrame) and not edge_rows.empty:
                        observed_edge_parts.append(edge_rows)
                    null_rows = payload.get(
                        "null_target_rows", pd.DataFrame()
                    )
                    if isinstance(null_rows, pd.DataFrame) and not null_rows.empty:
                        null_target_parts.append(null_rows)
                elif scope == "full_data_descriptive":
                    full_target_parts.append(
                        payload["target_rows"]
                    )
                    edge_rows = payload.get(
                        "edge_rows", pd.DataFrame()
                    )
                    if isinstance(edge_rows, pd.DataFrame) and not edge_rows.empty:
                        full_edge_parts.append(edge_rows)
                else:
                    raise RuntimeError(
                        f"Unknown structural scope: {scope}"
                    )
                run_parts.append(payload["run_rows"])
                completed_groups += 1
                cprint(
                    "[STRUCTURAL GROUP DONE] "
                    f"{result.get('task_key')} "
                    f"status={result.get('status')} "
                    f"progress={completed_groups}/{len(tasks)}"
                )
                next_task = next(task_iterator, None)
                if next_task is not None:
                    future_to_task[
                        executor.submit(
                            _run_structural_group_task,
                            next_task,
                        )
                    ] = next_task

    observed_target = pd.concat(observed_target_parts, ignore_index=True)
    null_target = pd.concat(null_target_parts, ignore_index=True)
    full_target = pd.concat(full_target_parts, ignore_index=True)
    observed_edges = (
        pd.concat(observed_edge_parts, ignore_index=True)
        if observed_edge_parts
        else pd.DataFrame()
    )
    full_edges = (
        pd.concat(full_edge_parts, ignore_index=True)
        if full_edge_parts
        else pd.DataFrame()
    )
    run_audit = pd.concat(run_parts, ignore_index=True)
    structural_sort_contract = {
        "observed_target": [
            "partition_id", "alpha", "depth", "feature"
        ],
        "null_target": [
            "partition_id", "alpha", "depth",
            "null_replicate", "feature"
        ],
        "full_target": ["alpha", "depth", "feature"],
        "observed_edges": [
            "partition_id", "alpha", "depth", "node_i", "node_j"
        ],
        "full_edges": ["alpha", "depth", "node_i", "node_j"],
        "run_audit": [
            "scope", "partition_id", "alpha", "depth",
            "null_replicate"
        ],
    }
    structural_frames = {
        "observed_target": observed_target,
        "null_target": null_target,
        "full_target": full_target,
        "observed_edges": observed_edges,
        "full_edges": full_edges,
        "run_audit": run_audit,
    }
    for frame_name, frame in structural_frames.items():
        if frame is None or frame.empty:
            continue
        sort_columns = [
            column
            for column in structural_sort_contract[frame_name]
            if column in frame.columns
        ]
        if sort_columns:
            structural_frames[frame_name] = frame.sort_values(
                sort_columns, kind="mergesort"
            ).reset_index(drop=True)
    observed_target = structural_frames["observed_target"]
    null_target = structural_frames["null_target"]
    full_target = structural_frames["full_target"]
    observed_edges = structural_frames["observed_edges"]
    full_edges = structural_frames["full_edges"]
    run_audit = structural_frames["run_audit"]

    evidence, setting_summary, null_replicate_summary = (
        _aggregate_structural_evidence(
            observed_target_rows=observed_target,
            null_target_rows=null_target,
            full_target_rows=full_target,
            feature_cols=feature_cols,
        )
    )
    endpoint_summary = _build_endpoint_pattern_summary(observed_target)
    predictive_structural = _build_predictive_structural_matrix(
        reliability_df=reliability_df,
        evidence_df=evidence,
    )

    observed_target.to_csv(
        audit_dir / "target_pag_evidence_outer_training_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    null_target.to_csv(
        audit_dir / "target_pag_evidence_permuted_outcome_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    full_target.to_csv(
        audit_dir / "target_pag_evidence_full_data_multiverse.csv",
        index=False,
        encoding="utf-8-sig",
    )
    observed_edges.to_csv(
        audit_dir / "pag_edges_outer_training_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    full_edges.to_csv(
        audit_dir / "pag_edges_full_data_multiverse.csv",
        index=False,
        encoding="utf-8-sig",
    )
    evidence.to_csv(
        audit_dir / "structural_compatibility_by_feature.csv",
        index=False,
        encoding="utf-8-sig",
    )
    setting_summary.to_csv(
        audit_dir / "structural_setting_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    null_replicate_summary.to_csv(
        audit_dir / "structural_null_replicate_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    endpoint_summary.to_csv(
        audit_dir / "structural_endpoint_pattern_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictive_structural.to_csv(
        audit_dir / "predictive_structural_compatibility_matrix.csv",
        index=False,
        encoding="utf-8-sig",
    )
    category_support.to_csv(
        audit_dir / "structural_category_support_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    full_coding.to_csv(
        audit_dir / "structural_category_codebook.csv",
        index=False,
        encoding="utf-8-sig",
    )
    run_audit.to_csv(
        audit_dir / "structural_run_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if OUTPUT_FIGURES:
        _save_structural_compatibility_figures(
            evidence_df=evidence,
            setting_summary=setting_summary,
            endpoint_summary=endpoint_summary,
            predictive_structural_matrix=predictive_structural,
            output_dir=audit_dir,
        )

    meta = {
        "enabled": True,
        "status": "complete",
        "module_name": "data-driven structural compatibility audit",
        "input_scope": (
            "observed records in the modeled NHTSA SGO ADS crash sample; "
            "no synthetic or oversampled records"
        ),
        "target_representation": (
            "single nominal three-level injury outcome; structural evidence is "
            "outcome-level and is not a class-specific causal effect"
        ),
        "algorithm": "Fast Causal Inference (FCI)",
        "graph_output": "partial ancestral graph (PAG)",
        "conditional_independence_test": (
            "G-square test for discrete variables as implemented by causal-learn"
        ),
        "alpha_grid": alpha_grid,
        "depth_grid": depth_grid,
        "max_path_length": int(STRUCTURAL_MAX_PATH_LENGTH),
        "outer_partitions": int(len(partitions)),
        "settings_per_partition": int(len(alpha_grid) * len(depth_grid)),
        "null_replicates": int(n_null_replicates),
        "parallel_workers": int(STRUCTURAL_PARALLEL_WORKERS),
        "task_grouping": (
            "one outer-partition or full-data design setting per task; "
            "matched outcome permutations remain sequential within task"
        ),
        "task_checkpoint_contract": (
            STRUCTURAL_TASK_CONTRACT_VERSION
        ),
        "negative_control": (
            "within-partition permutation of the three-level injury labels; "
            "class counts are preserved"
        ),
        "expert_edges_used": False,
        "temporal_edges_forced": False,
        "ranking_modified_by_structure": False,
        "causal_effect_estimated": False,
        "direction_claim_allowed": False,
        "formal_significance_claim_allowed": False,
        "graphical_assumptions": {
            "causal_markov": (
                "required for a causal interpretation of an FCI PAG; not "
                "established by this observational dataset"
            ),
            "faithfulness": (
                "required for a causal interpretation of an FCI PAG; not "
                "established by this observational dataset"
            ),
            "acyclicity": (
                "required by the graphical model used by FCI; not tested as an "
                "empirical claim"
            ),
            "incident_level_independence": (
                "modeled crashes are treated as independent incident-level "
                "records; overlapping outer partitions are explicitly not treated "
                "as independent replications"
            ),
        },
        "inferential_boundary": (
            "PAG adjacency is reported as local structural compatibility under "
            "the observed-variable conditional-independence model. Endpoint "
            "marks are sensitivity diagnostics, not verified causal directions. "
            "No ATE, CATE, intervention response, or mechanism is inferred."
        ),
        "negative_control_boundary": (
            "The permuted-outcome distribution is a descriptive finite-sample "
            "benchmark. Its quantiles and exceedance fractions are not reported "
            "as multiplicity-adjusted hypothesis tests or p-values."
        ),
        "dependence_boundary": (
            "Predictive explanations and the structural audit use the same "
            "observational sample and are not statistically independent."
        ),
        "overlap_boundary": (
            "Outer-training partitions overlap. Fold frequencies quantify "
            "cross-fitted stability and are not treated as independent "
            "replication probabilities or p-values."
        ),
        "sparsity_boundary": (
            "G-square tests are asymptotic and conditional tables may be sparse. "
            "The category-support table and depth sensitivity are exported so "
            "weakly supported edges are not hidden behind one graph setting."
        ),
        "outputs": {
            "feature_evidence": str(
                audit_dir / "structural_compatibility_by_feature.csv"
            ),
            "predictive_structural_matrix": str(
                audit_dir / "predictive_structural_compatibility_matrix.csv"
            ),
            "setting_sensitivity": str(
                audit_dir / "structural_setting_sensitivity.csv"
            ),
            "null_replicate_summary": str(
                audit_dir / "structural_null_replicate_summary.csv"
            ),
            "endpoint_patterns": str(
                audit_dir / "structural_endpoint_pattern_summary.csv"
            ),
            "category_support": str(
                audit_dir / "structural_category_support_audit.csv"
            ),
            "run_audit": str(audit_dir / "structural_run_audit.csv"),
            "figures": str(audit_dir / "figures"),
        },
    }
    safe_json_dump(meta, audit_dir / "structural_compatibility_meta.json")
    return meta


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return float(len(a & b) / len(union))


def _legacy_build_weight_sensitivity_table(
    reliability_df: pd.DataFrame, top_k: int = 6
) -> pd.DataFrame:
    if reliability_df is None or reliability_df.empty:
        return pd.DataFrame()
    schemes = [
        ("equal", "equal weights"),
        ("adaptive", "reliability-conflict adaptive weights"),
    ]
    rows = []
    for class_id, group in reliability_df.groupby("class_id"):
        feature_sets = {}
        feature_lists = {}
        rank_vectors = {}
        for scheme, _ in schemes:
            score_col = f"consensus_importance_{scheme}"
            if score_col not in group.columns:
                continue
            ordered = group.sort_values(
                [score_col, "feature"], ascending=[False, True]
            ).reset_index(drop=True)
            feats = ordered["feature"].astype(str).head(int(top_k)).tolist()
            feature_lists[scheme] = feats
            feature_sets[scheme] = set(feats)
            rank_vectors[scheme] = (
                ordered.set_index("feature")[score_col]
                .astype(float)
                .rank(ascending=False, method="average")
            )

        if not {"equal", "adaptive"}.issubset(feature_sets.keys()):
            continue

        all_intersection = feature_sets["equal"] & feature_sets["adaptive"]
        j_ea = _jaccard(feature_sets["equal"], feature_sets["adaptive"])

        corr_pairs = []
        for left, right in [("equal", "adaptive")]:
            joined = pd.concat(
                [rank_vectors[left], rank_vectors[right]], axis=1, join="inner"
            ).dropna()
            if joined.shape[0] >= 2:
                left_rank = joined.iloc[:, 0].rank(method="average")
                right_rank = joined.iloc[:, 1].rank(method="average")
                corr_pairs.append(float(left_rank.corr(right_rank, method="pearson")))
            else:
                corr_pairs.append(np.nan)

        rows.append(
            {
                "class_id": int(class_id),
                "class_name": TARGET_NAMES.get(int(class_id), str(class_id)),
                "top_k": int(top_k),
                "top_features_equal": "; ".join(feature_lists["equal"]),
                "top_features_adaptive": "; ".join(feature_lists["adaptive"]),
                "stable_features_across_all": "; ".join(sorted(all_intersection)),
                "n_stable_features_across_all": int(len(all_intersection)),
                "jaccard_equal_adaptive": j_ea,
                "min_pairwise_jaccard": float(j_ea),
                "spearman_equal_adaptive": corr_pairs[0],
            }
        )
    return pd.DataFrame(rows)


def build_source_dependence_table(partition_df: pd.DataFrame) -> pd.DataFrame:
    """Describe source dependence over all outer-partition feature vectors."""
    if partition_df is None or partition_df.empty:
        return pd.DataFrame()
    sources = list(FUSION_SOURCE_SPECS)
    rows = []
    for class_id, group in partition_df.groupby("class_id"):
        for left_pos, left in enumerate(sources):
            for right in sources[left_pos + 1 :]:
                pair = group[
                    [
                        "outer_partition",
                        "feature",
                        f"{left}_rank_signal",
                        f"{right}_rank_signal",
                    ]
                ].dropna()
                if len(pair) >= 2:
                    spearman = spearman_rank_correlation(
                        pair[f"{left}_rank_signal"], pair[f"{right}_rank_signal"]
                    )
                    mean_abs_disagreement = float(
                        np.mean(
                            np.abs(
                                pair[f"{left}_rank_signal"].to_numpy(dtype=float)
                                - pair[f"{right}_rank_signal"].to_numpy(dtype=float)
                            )
                        )
                    )
                else:
                    spearman = np.nan
                    mean_abs_disagreement = np.nan
                rows.append(
                    {
                        "class_id": int(class_id),
                        "class_name": TARGET_NAMES.get(int(class_id), str(class_id)),
                        "source_1": left,
                        "source_2": right,
                        "n_partition_feature_observations": int(len(pair)),
                        "n_outer_partitions": int(pair["outer_partition"].nunique()),
                        "n_features": int(pair["feature"].nunique()),
                        "spearman_rank_correlation": spearman,
                        "mean_absolute_rank_signal_disagreement": mean_abs_disagreement,
                    }
                )
    return pd.DataFrame(rows)


def _legacy_build_source_ablation_sensitivity_table(
    reliability_df: pd.DataFrame, top_k: int = 6
) -> pd.DataFrame:
    """Recompute the adaptive consensus after omitting each source in turn."""
    if reliability_df is None or reliability_df.empty:
        return pd.DataFrame()
    sources = list(FUSION_SOURCE_SPECS)
    rows = []
    for class_id, group in reliability_df.groupby("class_id"):
        group = group.copy().set_index("feature")
        full_scores = group["consensus_importance"].astype(float)
        full_ranks = full_scores.rank(ascending=False, method="average")
        full_top = set(
            full_scores.sort_values(ascending=False).head(int(top_k)).index.astype(str)
        )
        for omitted in sources:
            retained = [source for source in sources if source != omitted]
            ablated_scores = []
            for _, feature_row in group.iterrows():
                discounts = np.asarray(
                    [
                        finite_float(
                            feature_row.get(f"{source}_discount", np.nan), np.nan
                        )
                        for source in retained
                    ],
                    dtype=float,
                )
                signals = np.asarray(
                    [
                        finite_float(
                            feature_row.get(f"{source}_rank_signal", np.nan), np.nan
                        )
                        for source in retained
                    ],
                    dtype=float,
                )
                valid = (
                    np.isfinite(discounts) & np.isfinite(signals) & (discounts >= 0.0)
                )
                total = float(np.sum(discounts[valid])) if np.any(valid) else 0.0
                score = (
                    float(np.sum(discounts[valid] * signals[valid]) / total)
                    if total > SOURCE_WEIGHT_EPS
                    else np.nan
                )
                ablated_scores.append(score)
            ablated = pd.Series(ablated_scores, index=group.index, dtype=float)
            ablated_ranks = ablated.rank(
                ascending=False, method="average", na_option="bottom"
            )
            joined = pd.concat(
                [full_ranks.rename("full"), ablated_ranks.rename("ablated")], axis=1
            ).dropna()
            spearman = (
                spearman_rank_correlation(joined["full"], joined["ablated"])
                if len(joined) >= 2
                else np.nan
            )
            ablated_top = set(
                ablated.sort_values(ascending=False).head(int(top_k)).index.astype(str)
            )
            rows.append(
                {
                    "class_id": int(class_id),
                    "class_name": TARGET_NAMES.get(int(class_id), str(class_id)),
                    "omitted_source": omitted,
                    "retained_sources": "; ".join(retained),
                    "top_k": int(top_k),
                    "top_k_jaccard_vs_full": _jaccard(full_top, ablated_top),
                    "rank_spearman_vs_full_consensus": spearman,
                    "top_features_without_source": "; ".join(
                        ablated.sort_values(ascending=False)
                        .head(int(top_k))
                        .index.astype(str)
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_shap_background_sensitivity(
    shap_df: pd.DataFrame,
    reference_size: int,
    top_k: int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare SHAP rankings across configured training-background sizes."""
    if (
        shap_df is None
        or shap_df.empty
        or "background_size_requested" not in shap_df.columns
    ):
        return pd.DataFrame(), pd.DataFrame()

    run_rows = []
    group_cols = ["seed", "model", "class_id"]
    for keys, group in shap_df.groupby(group_cols):
        ref = group.loc[
            group["background_size_requested"].astype(int) == int(reference_size)
        ].copy()
        if ref.empty:
            continue
        ref = ref.sort_values("feature").drop_duplicates("feature").set_index("feature")
        ref_signal = ref["mean_abs_shap"].astype(float)
        ref_rank = ref_signal.rank(ascending=False, method="average")
        ref_top = set(
            ref_signal.sort_values(ascending=False).head(int(top_k)).index.astype(str)
        )

        for requested_size, alt in group.groupby("background_size_requested"):
            alt = (
                alt.sort_values("feature")
                .drop_duplicates("feature")
                .set_index("feature")
            )
            joined = pd.concat(
                [
                    ref_signal.rename("reference"),
                    alt["mean_abs_shap"].astype(float).rename("alternative"),
                ],
                axis=1,
                join="inner",
            ).dropna()
            if joined.empty:
                continue
            alt_rank = joined["alternative"].rank(ascending=False, method="average")
            aligned_ref_rank = ref_rank.reindex(joined.index)
            n_features = int(len(joined))
            rank_displacement = float(
                np.mean(np.abs(aligned_ref_rank.to_numpy() - alt_rank.to_numpy()))
                / max(n_features - 1, 1)
            )
            spearman = (
                float(aligned_ref_rank.corr(alt_rank, method="pearson"))
                if n_features >= 2
                else np.nan
            )
            alt_top = set(
                joined["alternative"]
                .sort_values(ascending=False)
                .head(int(top_k))
                .index.astype(str)
            )
            run_rows.append(
                {
                    "seed": int(keys[0]),
                    "model": str(keys[1]),
                    "class_id": int(keys[2]),
                    "reference_background_size": int(reference_size),
                    "background_size_requested": int(requested_size),
                    "n_background_actual": int(alt["n_background"].iloc[0]),
                    "n_features": n_features,
                    "rank_spearman_vs_reference": spearman,
                    "top_k": int(top_k),
                    "top_k_jaccard_vs_reference": _jaccard(ref_top, alt_top),
                    "normalized_mean_rank_displacement": rank_displacement,
                }
            )

    run_df = pd.DataFrame(run_rows)
    if run_df.empty:
        return run_df, pd.DataFrame()
    summary_df = (
        run_df.groupby(
            [
                "reference_background_size",
                "background_size_requested",
                "n_background_actual",
            ],
            as_index=False,
        )
        .agg(
            n_comparisons=("rank_spearman_vs_reference", "count"),
            mean_rank_spearman=("rank_spearman_vs_reference", "mean"),
            min_rank_spearman=("rank_spearman_vs_reference", "min"),
            mean_top_k_jaccard=("top_k_jaccard_vs_reference", "mean"),
            min_top_k_jaccard=("top_k_jaccard_vs_reference", "min"),
            mean_normalized_rank_displacement=(
                "normalized_mean_rank_displacement",
                "mean",
            ),
            max_normalized_rank_displacement=(
                "normalized_mean_rank_displacement",
                "max",
            ),
        )
        .sort_values("background_size_requested")
        .reset_index(drop=True)
    )
    return run_df, summary_df


# =============================================================================
# FIGURES
# =============================================================================


def save_evidence_confidence_heatmap(
    reliability_df: pd.DataFrame, output_path: Path, top_k: int = 6
) -> None:
    if not OUTPUT_FIGURES or reliability_df.empty or not module_available("matplotlib"):
        return
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    top_features = (
        reliability_df.sort_values(["class_id", "class_rank"])
        .groupby("class_id")
        .head(int(top_k))["feature"]
        .drop_duplicates()
        .tolist()
    )
    plot_df = reliability_df.loc[reliability_df["feature"].isin(top_features)].copy()
    classes = sorted(plot_df["class_id"].unique())
    features = top_features
    mat = np.full((len(classes), len(features)), np.nan, dtype=float)
    for i, class_id in enumerate(classes):
        sub = plot_df.loc[plot_df["class_id"] == class_id]
        value_col = (
            "evidence_support"
            if "evidence_support" in sub.columns
            else "evidence_confidence"
        )
        mapping = dict(zip(sub["feature"], sub[value_col]))
        for j, feature in enumerate(features):
            mat[i, j] = mapping.get(feature, np.nan)

    fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(features) + 3), 4.8))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=35, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([TARGET_NAMES.get(int(c), str(c)) for c in classes])
    ax.set_title("Cross-fitted evidence support")
    for i in range(len(classes)):
        for j in range(len(features)):
            if pd.notna(mat[i, j]):
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if mat[i, j] > 0.55 else "black",
                    fontsize=8,
                )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Evidence support index")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_top_evidence_confidence_dotplot(
    reliability_df: pd.DataFrame, output_path: Path, top_k: int = 6
) -> None:
    if not OUTPUT_FIGURES or reliability_df.empty or not module_available("matplotlib"):
        return
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    classes = sorted(reliability_df["class_id"].unique())
    fig, axes = plt.subplots(
        1, len(classes), figsize=(5.0 * len(classes), 5.0), sharex=True
    )
    if len(classes) == 1:
        axes = [axes]
    for ax, class_id in zip(axes, classes):
        value_col = (
            "evidence_support"
            if "evidence_support" in reliability_df.columns
            else "evidence_confidence"
        )
        sub = (
            reliability_df.loc[reliability_df["class_id"] == class_id]
            .sort_values(value_col, ascending=False)
            .head(int(top_k))
            .sort_values(value_col, ascending=True)
        )
        colors = [FIG_CLASS_COLORS.get(int(class_id), "#2a9d8f")] * len(sub)
        ax.scatter(
            sub[value_col],
            sub["feature"],
            s=70,
            c=colors,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_title(TARGET_NAMES.get(int(class_id), str(class_id)))
        ax.set_xlim(0, 1)
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        ax.set_xlabel("Evidence support index")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# PAPER-READY FIGURE SUITE
# =============================================================================


FIG_COLORS = {
    "teal": "#264653",
    "cyan": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "coral": "#E76F51",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "gray": "#8C8C8C",
    "light_gray": "#D9DEE2",
    "dark": "#2F3437",
}
FIG_CLASS_COLORS = {0: "#264653", 1: "#C44E52", 2: "#2A9D8F"}
FIG_SOURCE_COLORS = {
    "shap": "#0072B2",
    "pfi": "#E69F00",
    "loco": "#009E73",
    "firm": "#CC79A7",
}


def normalized_discounted_gain(
    scores: np.ndarray,
    relevance: np.ndarray,
    k: int,
) -> float:
    """Compute graded NDCG without introducing a relevance threshold."""
    scores = np.asarray(scores, dtype=float)
    relevance = np.asarray(relevance, dtype=float)
    k = int(max(1, min(int(k), len(scores))))

    def discounted_gain(order: np.ndarray) -> float:
        selected = relevance[order[:k]]
        gains = np.power(2.0, selected) - 1.0
        discounts = np.log2(np.arange(2, len(selected) + 2, dtype=float))
        return float(np.sum(gains / discounts))

    observed = discounted_gain(np.argsort(-scores, kind="mergesort"))
    ideal = discounted_gain(np.argsort(-relevance, kind="mergesort"))
    return float(observed / ideal) if ideal > SOURCE_WEIGHT_EPS else np.nan


def run_fusion_recovery_simulation(
    output_dir: Path,
    n_replicates: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Test aggregation recovery when the latent feature relevance is known.

    This diagnostic isolates the fusion rule. It generates complementary noisy
    source rankings with known active and null features, including a redundant
    pair and an interaction-sensitive setting. It does not validate causal
    interpretation or the fidelity of an individual explainer.
    """
    from sklearn.metrics import roc_auc_score

    feature_cols = [f"x{index + 1}" for index in range(12)]
    scenario_truth = {
        0: {
            "name": "complementary_sources",
            "truth": np.asarray([1.00, 0.70, 0.44, 0.25] + [0.0] * 8, dtype=float),
        },
        1: {
            "name": "unstable_source",
            "truth": np.asarray([1.00, 0.68, 0.40, 0.23] + [0.0] * 8, dtype=float),
        },
        2: {
            "name": "valid_dissent",
            "truth": np.asarray([1.00, 0.67, 0.31] + [0.0] * 9, dtype=float),
        },
    }
    schemes = {
        "crossfitted_reliability": "consensus_importance_crossfitted",
        "equal_weight": "consensus_importance_equal",
        "median_rank": "consensus_importance_median",
        "legacy_conflict_discount": "consensus_importance_legacy_conflict_discount",
    }
    source_base_noise = {"shap": 0.12, "pfi": 0.17, "loco": 0.19, "firm": 0.20}
    metric_rows = []

    for replicate in range(1, int(n_replicates) + 1):
        rng = np.random.RandomState(
            stable_seed_from_key("fusion_recovery", FUSION_SIMULATION_SEED, replicate)
        )
        source_tables = {source: [] for source in FUSION_SOURCE_SPECS}
        # The source-specific multiplier varies by replicate. Cross-fitted
        # reliability must infer the resulting stability difference from ranks.
        noise_multiplier = {
            source: float(np.exp(rng.normal(loc=0.0, scale=0.25)))
            for source in FUSION_SOURCE_SPECS
        }
        for partition_id in range(1, int(FUSION_SIMULATION_PARTITIONS) + 1):
            partition_shift = rng.normal(0.0, 0.045, size=len(feature_cols))
            for model_id in range(1, int(FUSION_SIMULATION_MODELS) + 1):
                model_shift = rng.normal(0.0, 0.035, size=len(feature_cols))
                for class_id, scenario in scenario_truth.items():
                    truth = scenario["truth"].copy()
                    for source in FUSION_SOURCE_SPECS:
                        expected = truth.copy()
                        noise_scale = (
                            source_base_noise[source] * noise_multiplier[source]
                        )

                        if class_id == 0:
                            # The four sources retain the same active set but
                            # attenuate different members of that set.
                            attenuation_index = {
                                "shap": 3,
                                "pfi": 1,
                                "loco": 2,
                                "firm": 0,
                            }[source]
                            expected[attenuation_index] *= 0.42

                        if class_id == 1 and source == "firm":
                            # One source is unbiased on average but unstable
                            # across folds and repeatedly elevates a different
                            # null proxy. Cross-fitting must identify its weak
                            # repeatability without consulting the target fold.
                            noise_scale *= 3.1
                            proxy_index = 4 + ((partition_id + model_id) % 8)
                            expected[proxy_index] = 0.40

                        if class_id == 2:
                            # SHAP carries a stable, valid signal for the third
                            # active feature. Other sources understate it, while
                            # FIRM assigns modest observational relevance to a
                            # null proxy. This setting tests whether conflict is
                            # reported rather than used to silence dissent.
                            if source == "shap":
                                expected[2] = truth[2]
                                noise_scale *= 0.70
                            elif source in {"pfi", "loco"}:
                                expected[2] *= 0.12
                            elif source == "firm":
                                expected[2] = 0.0
                                expected[3] = 0.29

                        noise = rng.normal(
                            0.0,
                            noise_scale,
                            size=len(feature_cols),
                        )
                        observed = np.maximum(
                            expected + partition_shift + model_shift + noise, 0.0
                        )
                        value_col = FUSION_SOURCE_SPECS[source]["value"]
                        for feature, value in zip(feature_cols, observed):
                            source_tables[source].append(
                                {
                                    "seed": int(partition_id),
                                    "outer_repeat": 1,
                                    "outer_fold": int(partition_id),
                                    "split_seed": int(
                                        stable_seed_from_key(
                                            "fusion_simulation_partition",
                                            replicate,
                                            partition_id,
                                        )
                                    ),
                                    "model": f"m{model_id}",
                                    "class_id": int(class_id),
                                    "feature": str(feature),
                                    value_col: float(value),
                                }
                            )

        source_frames = {
            source: pd.DataFrame(rows) for source, rows in source_tables.items()
        }
        fused = aggregate_crossfitted_consensus(
            shap_df=source_frames["shap"],
            perm_df=source_frames["pfi"],
            loco_df=source_frames["loco"],
            firm_df=source_frames["firm"],
            class_ids=sorted(scenario_truth),
            feature_cols=feature_cols,
            top_k=4,
            source_specs=FUSION_SOURCE_SPECS,
            allow_simulation_reference_weights=True,
        )
        partition = fused["partition"]
        for class_id, scenario in scenario_truth.items():
            truth = scenario["truth"]
            active = truth > 0.0
            n_active = int(np.sum(active))
            for scheme, score_col in schemes.items():
                scores = (
                    partition.loc[partition["class_id"].astype(int) == int(class_id)]
                    .groupby("feature")[score_col]
                    .mean()
                    .reindex(feature_cols)
                    .to_numpy(dtype=float)
                )
                top_idx = np.argsort(-scores)[:n_active]
                top_active = int(np.sum(active[top_idx]))
                rank_spearman = spearman_rank_correlation(scores, truth)
                active_null_margin = float(
                    np.min(scores[active]) - np.max(scores[~active])
                )
                metric_rows.append(
                    {
                        "replicate": int(replicate),
                        "class_id": int(class_id),
                        "scenario": str(scenario["name"]),
                        "scheme": str(scheme),
                        "n_active": n_active,
                        "top_k_recall": float(top_active / max(n_active, 1)),
                        "null_false_inclusion_rate": float(
                            (n_active - top_active) / max(n_active, 1)
                        ),
                        "active_null_auroc": float(
                            roc_auc_score(active.astype(int), scores)
                        ),
                        "spearman_to_latent_relevance": float(rank_spearman),
                        "ndcg_at_active_count": normalized_discounted_gain(
                            scores, truth, n_active
                        ),
                        "active_null_margin": active_null_margin,
                        "complete_active_null_separation": int(
                            active_null_margin > SOURCE_WEIGHT_EPS
                        ),
                    }
                )

    raw = pd.DataFrame(metric_rows)
    summary_rows = []
    metric_cols = [
        "top_k_recall",
        "null_false_inclusion_rate",
        "active_null_auroc",
        "spearman_to_latent_relevance",
        "ndcg_at_active_count",
        "active_null_margin",
        "complete_active_null_separation",
    ]
    for keys, group in raw.groupby(["class_id", "scenario", "scheme"]):
        row = {
            "class_id": int(keys[0]),
            "scenario": str(keys[1]),
            "scheme": str(keys[2]),
            "n_replicates": int(group["replicate"].nunique()),
        }
        for metric in metric_cols:
            values = group[metric].to_numpy(dtype=float)
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"std_{metric}"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
            row[f"q025_{metric}"] = float(np.quantile(values, 0.025))
            row[f"q975_{metric}"] = float(np.quantile(values, 0.975))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    tables_dir = Path(output_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(
        tables_dir / "fusion_recovery_simulation_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        tables_dir / "fusion_recovery_simulation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return raw, summary


def _has_matplotlib() -> bool:
    return OUTPUT_FIGURES and module_available("matplotlib")


def _setup_publication_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Cambria"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
        }
    )


def _save_paper_figure(fig, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.06
    )
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06)


def _clean_axis(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, alpha=0.18, linewidth=0.6)


def _feature_label(feature: str, max_len: int = 22) -> str:
    label = str(feature).replace("_Group", "").replace("_", " ")
    if len(label) <= max_len:
        return label
    parts = label.split()
    lines, cur = [], ""
    for part in parts:
        if len(cur) + len(part) + 1 <= max_len:
            cur = f"{cur} {part}".strip()
        else:
            if cur:
                lines.append(cur)
            cur = part
    if cur:
        lines.append(cur)
    return "\n".join(lines[:3])


def _class_label(class_id: int) -> str:
    return TARGET_NAMES.get(int(class_id), str(class_id))


def _mean_std_by_group(
    df: pd.DataFrame, group_cols: List[str], value_col: str
) -> pd.DataFrame:
    if df is None or df.empty or value_col not in df.columns:
        return pd.DataFrame()
    out = (
        df.groupby(group_cols, as_index=False)[value_col]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": f"mean_{value_col}", "std": f"std_{value_col}"})
    )
    out[f"std_{value_col}"] = out[f"std_{value_col}"].fillna(0.0)
    return out


def _select_top_features(
    reliability_df: pd.DataFrame, class_id: int, top_k: int = 6
) -> List[str]:
    if reliability_df is None or reliability_df.empty:
        return []
    sub = reliability_df.loc[
        reliability_df["class_id"].astype(int) == int(class_id)
    ].copy()
    if sub.empty:
        return []
    return (
        sub.sort_values(["consensus_importance", "feature"], ascending=[False, True])
        .head(int(top_k))["feature"]
        .astype(str)
        .tolist()
    )


def save_crossfitted_consensus_figures(
    summary_df: pd.DataFrame,
    partition_df: pd.DataFrame,
    source_weight_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    scheme_sensitivity_df: pd.DataFrame,
    measure_sensitivity_df: pd.DataFrame,
    figs_dir: Path,
    top_k: int = 6,
) -> None:
    if not _has_matplotlib() or summary_df is None or summary_df.empty:
        return
    import matplotlib.pyplot as plt

    _setup_publication_style()
    colors = {0: "#1F5A94", 1: "#C44E52", 2: "#2A8C8C"}
    sources = list(FUSION_SOURCE_SPECS)
    source_colors = {
        "shap": "#3B6FB6",
        "pfi": "#D17C2F",
        "loco": "#4E9A78",
        "firm": "#8A63A8",
    }
    figs_dir = Path(figs_dir) / "crossfitted_consensus"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Partition interval plot: empirical IQR and full observed range.
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1), sharex=True)
    for ax, class_id in zip(axes, [0, 1, 2]):
        sub = (
            summary_df.loc[summary_df["class_id"].astype(int) == int(class_id)]
            .sort_values("consensus_importance", ascending=False)
            .head(int(top_k))
            .sort_values("consensus_importance", ascending=True)
        )
        y = np.arange(len(sub))
        mean = sub["consensus_importance"].to_numpy(dtype=float)
        q25 = sub["consensus_q25"].to_numpy(dtype=float)
        q75 = sub["consensus_q75"].to_numpy(dtype=float)
        low = sub["consensus_min"].to_numpy(dtype=float)
        high = sub["consensus_max"].to_numpy(dtype=float)
        ax.hlines(y, low, high, color="#B8B8B8", linewidth=1.0, zorder=1)
        ax.hlines(
            y,
            q25,
            q75,
            color=colors[class_id],
            linewidth=3.0,
            zorder=2,
        )
        ax.scatter(
            mean,
            y,
            s=28,
            facecolor="white",
            edgecolor=colors[class_id],
            linewidth=1.4,
            zorder=3,
        )
        ax.set_yticks(y)
        ax.set_yticklabels([_feature_label(v) for v in sub["feature"].astype(str)])
        ax.set_title(_class_label(class_id), loc="left", fontweight="bold")
        ax.set_xlabel("Cross-fitted consensus score")
        _clean_axis(ax, grid_axis="x")
    axes[0].set_ylabel("Feature")
    fig.tight_layout()
    _save_paper_figure(fig, figs_dir / "fig_consensus_partition_intervals")
    plt.close(fig)

    # Stability and ambiguity panels use the same feature order.
    selected = []
    for class_id in [0, 1, 2]:
        selected.extend(_select_top_features(summary_df, class_id, top_k=top_k))
    selected = list(dict.fromkeys(selected))
    class_ids = [0, 1, 2]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, max(4.2, 0.35 * len(selected))))
    for ax, value_col, title, cmap, vmin, vmax in [
        (
            axes[0],
            "top_k_inclusion_frequency",
            "Top-k inclusion frequency",
            "YlGnBu",
            0.0,
            1.0,
        ),
        (axes[1], "evidence_ambiguity", "Cross-source ambiguity", "OrRd", 0.0, None),
    ]:
        matrix = (
            summary_df.pivot(index="feature", columns="class_id", values=value_col)
            .reindex(index=selected, columns=class_ids)
            .to_numpy(dtype=float)
        )
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(class_ids)))
        ax.set_xticklabels(
            [_class_label(v) for v in class_ids], rotation=20, ha="right"
        )
        ax.set_yticks(np.arange(len(selected)))
        ax.set_yticklabels([_feature_label(v) for v in selected])
        ax.set_title(title, loc="left", fontweight="bold")
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    fig.tight_layout()
    _save_paper_figure(fig, figs_dir / "fig_stability_and_ambiguity_heatmaps")
    plt.close(fig)

    # Source weights are class-specific and should not vary by feature.
    weight_summary = (
        source_weight_df.groupby(["class_id", "source"], as_index=False)
        .agg(mean_weight=("source_weight", "mean"), sd_weight=("source_weight", "std"))
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(3, dtype=float)
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(sources))
    for offset, source in zip(offsets, sources):
        sub = weight_summary.loc[weight_summary["source"] == source].set_index(
            "class_id"
        )
        means = [float(sub.loc[c, "mean_weight"]) for c in [0, 1, 2]]
        errors = [float(sub.loc[c, "sd_weight"]) for c in [0, 1, 2]]
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=errors,
            capsize=2.5,
            color=source_colors[source],
            edgecolor="white",
            linewidth=0.5,
            label=source.upper(),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_class_label(v) for v in [0, 1, 2]])
    ax.set_ylabel("Cross-fitted source weight")
    ax.set_ylim(0.0, max(0.55, ax.get_ylim()[1]))
    ax.legend(frameon=False, ncol=4, loc="upper center")
    _clean_axis(ax)
    fig.tight_layout()
    _save_paper_figure(fig, figs_dir / "fig_class_specific_source_weights")
    plt.close(fig)

    # Pairwise dominance probabilities reveal incomparability hidden by total ranks.
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    for ax, class_id in zip(axes, [0, 1, 2]):
        top_features = _select_top_features(summary_df, class_id, top_k=top_k)
        sub = pairwise_df.loc[pairwise_df["class_id"].astype(int) == int(class_id)]
        matrix = sub.pivot(
            index="feature_left",
            columns="feature_right",
            values="dominance_probability",
        ).reindex(index=top_features, columns=top_features)
        matrix_values = matrix.to_numpy(dtype=float, copy=True)
        np.fill_diagonal(matrix_values, 0.5)
        image = ax.imshow(matrix_values, cmap="RdBu_r", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(top_features)))
        ax.set_xticklabels(
            [_feature_label(v, 15) for v in top_features], rotation=50, ha="right"
        )
        ax.set_yticks(np.arange(len(top_features)))
        ax.set_yticklabels([_feature_label(v, 15) for v in top_features])
        ax.set_title(_class_label(class_id), loc="left", fontweight="bold")
    fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        fraction=0.02,
        pad=0.02,
        label="P(left > right)",
    )
    fig.subplots_adjust(left=0.08, right=0.94, bottom=0.24, top=0.90, wspace=0.36)
    _save_paper_figure(fig, figs_dir / "fig_pairwise_dominance_probability")
    plt.close(fig)

    if scheme_sensitivity_df is not None and not scheme_sensitivity_df.empty:
        schemes = [
            "crossfitted_reliability",
            "equal_weight",
            "median_rank",
            "legacy_conflict_discount",
        ]
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        x = np.arange(3, dtype=float)
        width = 0.19
        palette = ["#2A6F97", "#6C9F76", "#C28E4A", "#8A6FA8"]
        for pos, (scheme, color) in enumerate(zip(schemes, palette)):
            sub = scheme_sensitivity_df.loc[
                scheme_sensitivity_df["scheme"] == scheme
            ].set_index("class_id")
            values = [float(sub.loc[c, "top_k_jaccard_vs_primary"]) for c in [0, 1, 2]]
            ax.bar(
                x + (pos - 1.5) * width,
                values,
                width=width,
                color=color,
                label=scheme.replace("_", " "),
            )
        ax.set_xticks(x)
        ax.set_xticklabels([_class_label(v) for v in [0, 1, 2]])
        ax.set_ylabel("Top-k Jaccard versus primary")
        ax.set_ylim(0.0, 1.05)
        ax.legend(frameon=False, ncol=2, loc="lower left")
        _clean_axis(ax)
        fig.tight_layout()
        _save_paper_figure(fig, figs_dir / "fig_fusion_scheme_sensitivity")
        plt.close(fig)

    if measure_sensitivity_df is not None and not measure_sensitivity_df.empty:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        sub = measure_sensitivity_df.sort_values("class_id")
        x = np.arange(len(sub))
        ax.bar(
            x,
            sub["top_k_jaccard"],
            width=0.55,
            color=[colors[int(v)] for v in sub["class_id"]],
        )
        ax.set_xticks(x)
        ax.set_xticklabels(sub["class_name"].astype(str))
        ax.set_ylabel("Top-k Jaccard")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(
            "Class-balanced versus prevalence-weighted fusion",
            loc="left",
            fontweight="bold",
        )
        _clean_axis(ax)
        fig.tight_layout()
        _save_paper_figure(fig, figs_dir / "fig_class_balance_measure_sensitivity")
        plt.close(fig)


def save_fusion_recovery_simulation_figure(
    summary_df: pd.DataFrame, figs_dir: Path
) -> None:
    if not _has_matplotlib() or summary_df is None or summary_df.empty:
        return
    import matplotlib.pyplot as plt

    _setup_publication_style()
    schemes = [
        "crossfitted_reliability",
        "equal_weight",
        "median_rank",
        "legacy_conflict_discount",
    ]
    colors = ["#2A6F97", "#6C9F76", "#C28E4A", "#8A6FA8"]
    scenarios = (
        summary_df[["class_id", "scenario"]].drop_duplicates().sort_values("class_id")
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), sharex=True)
    for ax, metric, ylabel in [
        (axes[0], "mean_active_null_auroc", "Active-null AUROC"),
        (axes[1], "mean_ndcg_at_active_count", "NDCG at active count"),
        (axes[2], "mean_top_k_recall", "Top-k active-feature recall"),
    ]:
        x = np.arange(len(scenarios), dtype=float)
        width = 0.19
        for pos, (scheme, color) in enumerate(zip(schemes, colors)):
            sub = summary_df.loc[summary_df["scheme"] == scheme].set_index("class_id")
            values = [float(sub.loc[int(c), metric]) for c in scenarios["class_id"]]
            ax.bar(
                x + (pos - 1.5) * width,
                values,
                width=width,
                color=color,
                label=scheme.replace("_", " "),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [str(v).replace("_", " ") for v in scenarios["scenario"]],
            rotation=18,
            ha="right",
        )
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.05)
        _clean_axis(ax)
    axes[0].legend(frameon=False, ncol=2, loc="lower left")
    fig.tight_layout()
    _save_paper_figure(
        fig, Path(figs_dir) / "crossfitted_consensus" / "fig_fusion_recovery_simulation"
    )
    plt.close(fig)


def save_dataset_figures(
    dataset_df: pd.DataFrame, feature_support_df: pd.DataFrame, figs_dir: Path
) -> None:
    if not _has_matplotlib():
        return
    import matplotlib.pyplot as plt

    _setup_publication_style()
    if dataset_df is not None and not dataset_df.empty:
        df = dataset_df.sort_values("class_id").copy()
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        y = np.arange(len(df))
        colors = [
            FIG_CLASS_COLORS.get(int(c), FIG_COLORS["gray"]) for c in df["class_id"]
        ]
        ax.barh(
            y,
            df["share"].astype(float) * 100.0,
            color=colors,
            edgecolor="white",
            height=0.62,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(df["class_name"].astype(str))
        ax.set_xlabel("Share of analytical sample (%)")
        for yi, (_, row) in zip(y, df.iterrows()):
            ax.text(
                row["share"] * 100.0 + 0.6,
                yi,
                f"{int(row['n'])} ({row['share'] * 100:.2f}%)",
                va="center",
                fontsize=8.5,
            )
        ax.set_xlim(0, max(100.0, float(df["share"].max() * 100.0) + 8.0))
        ax.invert_yaxis()
        _clean_axis(ax, "x")
        _save_paper_figure(fig, figs_dir / "fig_dataset_class_distribution")
        plt.close(fig)

    if feature_support_df is not None and not feature_support_df.empty:
        support = feature_support_df.copy()
        summary = (
            support.groupby("feature", as_index=False)
            .agg(
                n_values=("value", "count"),
                min_share=("share", "min"),
                max_share=("share", "max"),
            )
            .sort_values(["n_values", "min_share"], ascending=[False, True])
        )
        fig, ax1 = plt.subplots(figsize=(7.2, 3.8))
        x = np.arange(len(summary))
        ax1.bar(
            x,
            summary["n_values"],
            color=FIG_COLORS["teal"],
            alpha=0.86,
            label="Number of values",
        )
        ax1.set_ylabel("Number of observed values")
        ax1.set_xticks(x)
        ax1.set_xticklabels(
            [_feature_label(f, 14) for f in summary["feature"]], rotation=35, ha="right"
        )
        ax2 = ax1.twinx()
        ax2.plot(
            x,
            summary["min_share"] * 100.0,
            color=FIG_COLORS["coral"],
            marker="o",
            linewidth=1.7,
            label="Minimum value support",
        )
        ax2.set_ylabel("Minimum training support (%)")
        _clean_axis(ax1, "y")
        ax2.spines["top"].set_visible(False)
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="upper right")
        _save_paper_figure(fig, figs_dir / "fig_feature_value_support_overview")
        plt.close(fig)


def save_prediction_figures(
    metrics_raw: pd.DataFrame,
    metrics_summary: pd.DataFrame,
    calibration_df: pd.DataFrame,
    figs_dir: Path,
) -> None:
    if not _has_matplotlib():
        return
    import matplotlib.pyplot as plt

    _setup_publication_style()
    if metrics_summary is not None and not metrics_summary.empty:
        df = metrics_summary.sort_values(
            "mean_balanced_accuracy", ascending=True
        ).copy()
        y = np.arange(len(df))
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        for metric, color, marker, label in [
            ("balanced_accuracy", FIG_COLORS["coral"], "o", "Balanced accuracy"),
            ("macro_f1", FIG_COLORS["teal"], "s", "Macro-F1"),
            ("accuracy", FIG_COLORS["gray"], "^", "Accuracy"),
        ]:
            mean_col = f"mean_{metric}"
            std_col = f"std_{metric}"
            if mean_col not in df.columns:
                continue
            ax.errorbar(
                df[mean_col],
                y,
                xerr=df[std_col] if std_col in df.columns else None,
                fmt=marker,
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=2.8,
                label=label,
                markersize=5,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(df["model_display"].astype(str))
        ax.set_xlabel("Metric value")
        ax.set_xlim(0, 1)
        _clean_axis(ax, "x")
        ax.legend(frameon=False, loc="lower right")
        _save_paper_figure(fig, figs_dir / "fig_prediction_model_benchmark")
        plt.close(fig)

        recall_cols = [
            c
            for c in [
                "mean_recall_class_0",
                "mean_recall_class_1",
                "mean_recall_class_2",
            ]
            if c in df.columns
        ]
        if recall_cols:
            ordered = metrics_summary.sort_values(
                "mean_balanced_accuracy", ascending=False
            ).copy()
            mat = ordered[recall_cols].to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(5.6, max(3.2, 0.36 * len(ordered) + 1.4)))
            im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="YlGnBu")
            ax.set_yticks(np.arange(len(ordered)))
            ax.set_yticklabels(ordered["model_display"].astype(str))
            ax.set_xticks(np.arange(len(recall_cols)))
            ax.set_xticklabels(
                [_class_label(int(c.rsplit("_", 1)[-1])) for c in recall_cols],
                rotation=25,
                ha="right",
            )
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(
                        j,
                        i,
                        f"{mat[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="white" if mat[i, j] > 0.55 else FIG_COLORS["dark"],
                    )
            cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            cbar.set_label("Recall")
            _save_paper_figure(fig, figs_dir / "fig_prediction_class_recall_heatmap")
            plt.close(fig)

        diag_cols = [
            c
            for c in ["mean_log_loss", "mean_multiclass_brier", "mean_ece"]
            if c in metrics_summary.columns
        ]
        if diag_cols:
            ordered = metrics_summary.sort_values(
                "mean_balanced_accuracy", ascending=False
            ).copy()
            fig, axes = plt.subplots(
                1, len(diag_cols), figsize=(3.25 * len(diag_cols), 3.4)
            )
            if len(diag_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, diag_cols):
                sub = ordered.sort_values(col, ascending=True)
                ax.barh(
                    np.arange(len(sub)), sub[col], color=FIG_COLORS["teal"], alpha=0.86
                )
                ax.set_yticks(np.arange(len(sub)))
                ax.set_yticklabels(
                    sub["model_display"].astype(str) if ax is axes[0] else []
                )
                ax.set_xlabel(col.replace("mean_", "").replace("_", " ").title())
                ax.invert_yaxis()
                _clean_axis(ax, "x")
            _save_paper_figure(fig, figs_dir / "fig_probability_diagnostics_by_model")
            plt.close(fig)

    if calibration_df is not None and not calibration_df.empty:
        cal = calibration_df.dropna(
            subset=["mean_predicted_probability", "observed_frequency"]
        ).copy()
        if not cal.empty:
            top = cal.loc[cal["curve_type"] == "top_label"].copy()
            if not top.empty:
                agg = top.groupby(["model", "bin_id"], as_index=False).agg(
                    mean_pred=("mean_predicted_probability", "mean"),
                    obs=("observed_frequency", "mean"),
                    n_bin=("n_bin", "sum"),
                )
                models = (
                    agg.groupby("model")["n_bin"]
                    .sum()
                    .sort_values(ascending=False)
                    .index.tolist()
                )
                fig, ax = plt.subplots(figsize=(5.2, 4.2))
                ax.plot(
                    [0, 1],
                    [0, 1],
                    color=FIG_COLORS["gray"],
                    linewidth=1.0,
                    linestyle="--",
                    label="Perfect calibration",
                )
                markers = ["o", "s", "^", "D", "v", "P"]
                for i, model in enumerate(models[:6]):
                    sub = agg.loc[agg["model"] == model].sort_values("mean_pred")
                    ax.plot(
                        sub["mean_pred"],
                        sub["obs"],
                        marker=markers[i % len(markers)],
                        linewidth=1.5,
                        label=MODEL_DISPLAY_NAMES.get(model, model),
                    )
                ax.set_xlabel("Mean predicted probability")
                ax.set_ylabel("Observed frequency")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                _clean_axis(ax, "both")
                ax.legend(frameon=False, loc="upper left")
                _save_paper_figure(fig, figs_dir / "fig_calibration_top_label")
                plt.close(fig)

            classwise = cal.loc[cal["curve_type"] == "classwise"].copy()
            if not classwise.empty:
                best_model = (
                    metrics_summary.iloc[0]["model"]
                    if metrics_summary is not None and not metrics_summary.empty
                    else classwise["model"].iloc[0]
                )
                sub0 = classwise.loc[classwise["model"].astype(str) == str(best_model)]
                agg = sub0.groupby(["class_id", "bin_id"], as_index=False).agg(
                    mean_pred=("mean_predicted_probability", "mean"),
                    obs=("observed_frequency", "mean"),
                    n_bin=("n_bin", "sum"),
                )
                fig, ax = plt.subplots(figsize=(5.2, 4.2))
                ax.plot(
                    [0, 1],
                    [0, 1],
                    color=FIG_COLORS["gray"],
                    linewidth=1.0,
                    linestyle="--",
                    label="Perfect calibration",
                )
                for class_id, group in agg.groupby("class_id"):
                    group = group.sort_values("mean_pred")
                    ax.plot(
                        group["mean_pred"],
                        group["obs"],
                        marker="o",
                        linewidth=1.5,
                        color=FIG_CLASS_COLORS.get(int(class_id), FIG_COLORS["gray"]),
                        label=_class_label(int(class_id)),
                    )
                ax.set_xlabel("Mean predicted probability")
                ax.set_ylabel("Observed frequency")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                _clean_axis(ax, "both")
                ax.legend(frameon=False, loc="upper left")
                _save_paper_figure(
                    fig, figs_dir / "fig_calibration_classwise_best_model"
                )
                plt.close(fig)


def save_prediction_distribution_figures(
    predictions_df: pd.DataFrame, figs_dir: Path
) -> None:
    if not _has_matplotlib() or predictions_df is None or predictions_df.empty:
        return
    import matplotlib.pyplot as plt

    _setup_publication_style()
    pred = predictions_df.copy()
    if {"y_true", "y_pred"}.issubset(pred.columns):
        labels = sorted(
            set(pred["y_true"].astype(int).unique()).union(
                set(pred["y_pred"].astype(int).unique())
            )
        )
        mat = np.zeros((len(labels), len(labels)), dtype=float)
        for i, true_cls in enumerate(labels):
            sub = pred.loc[pred["y_true"].astype(int) == int(true_cls)]
            denom = max(len(sub), 1)
            for j, pred_cls in enumerate(labels):
                mat[i, j] = float(
                    (sub["y_pred"].astype(int) == int(pred_cls)).sum() / denom
                )
        fig, ax = plt.subplots(figsize=(4.7, 4.0))
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels([_class_label(c) for c in labels], rotation=25, ha="right")
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels([_class_label(c) for c in labels])
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Observed class")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if mat[i, j] > 0.55 else FIG_COLORS["dark"],
                )
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("Row-normalized share")
        _save_paper_figure(fig, figs_dir / "fig_confusion_matrix_best_explained_model")
        plt.close(fig)

    proba_cols = [
        c
        for c in ["proba_class_0", "proba_class_1", "proba_class_2"]
        if c in pred.columns
    ]
    if proba_cols and "y_true" in pred.columns:
        fig, axes = plt.subplots(
            1, len(proba_cols), figsize=(4.0 * len(proba_cols), 3.7), sharey=True
        )
        if len(proba_cols) == 1:
            axes = [axes]
        for ax, proba_col in zip(axes, proba_cols):
            class_id = int(proba_col.rsplit("_", 1)[-1])
            data = [
                pred.loc[pred["y_true"].astype(int) == c, proba_col].dropna().to_numpy()
                for c in sorted(pred["y_true"].astype(int).unique())
            ]
            bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
            for patch, c in zip(
                bp["boxes"], sorted(pred["y_true"].astype(int).unique())
            ):
                patch.set_facecolor(FIG_CLASS_COLORS.get(int(c), FIG_COLORS["gray"]))
                patch.set_alpha(0.75)
            ax.set_xticks(np.arange(1, len(data) + 1))
            ax.set_xticklabels(
                [_class_label(c) for c in sorted(pred["y_true"].astype(int).unique())],
                rotation=25,
                ha="right",
            )
            ax.set_title(_class_label(class_id))
            ax.set_ylim(0, 1)
            ax.set_ylabel("Predicted probability" if ax is axes[0] else "")
            _clean_axis(ax, "y")
        _save_paper_figure(fig, figs_dir / "fig_predicted_probability_by_true_class")
        plt.close(fig)


def save_consensus_importance_panel(
    reliability_df: pd.DataFrame, figs_dir: Path, top_k: int = 6
) -> None:
    if not _has_matplotlib() or reliability_df is None or reliability_df.empty:
        return
    import matplotlib.pyplot as plt

    _setup_publication_style()
    classes = sorted(reliability_df["class_id"].astype(int).unique())
    fig, axes = plt.subplots(
        1, len(classes), figsize=(4.9 * len(classes), 4.8), sharex=True
    )
    if len(classes) == 1:
        axes = [axes]
    for ax, class_id in zip(axes, classes):
        sub = (
            reliability_df.loc[reliability_df["class_id"].astype(int) == int(class_id)]
            .sort_values("consensus_importance", ascending=False)
            .head(int(top_k))
            .sort_values("consensus_importance", ascending=True)
        )
        y = np.arange(len(sub))
        ax.hlines(
            y,
            0,
            sub["consensus_importance"],
            color=FIG_COLORS["light_gray"],
            linewidth=2.6,
        )
        ax.scatter(
            sub["consensus_importance"],
            y,
            s=48,
            color=FIG_CLASS_COLORS.get(int(class_id), FIG_COLORS["teal"]),
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.set_yticks(y)
        ax.set_yticklabels([_feature_label(f, 18) for f in sub["feature"]])
        ax.set_title(_class_label(class_id), loc="left", fontweight="bold")
        ax.set_xlabel("Reliability-adjusted consensus importance")
        ax.set_xlim(
            0, max(1e-6, float(reliability_df["consensus_importance"].max()) * 1.10)
        )
        _clean_axis(ax, "x")
    _save_paper_figure(fig, figs_dir / "fig_consensus_importance_by_class")
    plt.close(fig)


def save_reliability_conflict_scatter(
    reliability_df: pd.DataFrame, figs_dir: Path
) -> None:
    if not _has_matplotlib() or reliability_df is None or reliability_df.empty:
        return
    import matplotlib.pyplot as plt

    _setup_publication_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.5))
    for class_id, group in reliability_df.groupby("class_id"):
        support_col = (
            "evidence_support"
            if "evidence_support" in group.columns
            else "evidence_confidence"
        )
        ax.scatter(
            group["evidence_conflict_index"],
            group[support_col],
            s=28 + 180 * group["consensus_importance"].clip(0, 1),
            alpha=0.78,
            color=FIG_CLASS_COLORS.get(int(class_id), FIG_COLORS["gray"]),
            edgecolor="white",
            linewidth=0.5,
            label=_class_label(int(class_id)),
        )
        top = group.sort_values("consensus_importance", ascending=False).head(3)
        for _, row in top.iterrows():
            ax.annotate(
                _feature_label(row["feature"], 12),
                (row["evidence_conflict_index"], row[support_col]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.2,
            )
    ax.set_xlabel("Cross-source ambiguity")
    ax.set_ylabel("Evidence support index")
    ax.set_xlim(
        0, max(0.05, float(reliability_df["evidence_conflict_index"].max()) * 1.12)
    )
    ax.set_ylim(0, 1.02)
    _clean_axis(ax, "both")
    ax.legend(frameon=False, loc="lower left")
    _save_paper_figure(fig, figs_dir / "fig_reliability_conflict_map")
    plt.close(fig)


def _source_heatmap_matrix(
    reliability_df: pd.DataFrame, value_cols: List[str], top_features: List[str]
) -> Tuple[np.ndarray, List[str], List[str]]:
    sources = [c.replace("w_", "").split("_", 1)[0] for c in value_cols]
    rows = []
    labels = []
    for class_id in sorted(reliability_df["class_id"].astype(int).unique()):
        sub = reliability_df.loc[
            reliability_df["class_id"].astype(int) == int(class_id)
        ]
        mapping = sub.set_index("feature")
        for feature in top_features:
            if feature not in mapping.index:
                continue
            vals = [
                finite_float(mapping.loc[feature].get(col, np.nan), default=np.nan)
                for col in value_cols
            ]
            rows.append(vals)
            labels.append(f"{_class_label(class_id)} | {_feature_label(feature, 16)}")
    return np.asarray(rows, dtype=float), labels, sources


def save_source_diagnostic_heatmaps(
    reliability_df: pd.DataFrame, figs_dir: Path, top_k: int = 6
) -> None:
    if not _has_matplotlib() or reliability_df is None or reliability_df.empty:
        return
    import matplotlib.pyplot as plt

    _setup_publication_style()
    top_features = []
    for class_id in sorted(reliability_df["class_id"].astype(int).unique()):
        top_features.extend(_select_top_features(reliability_df, class_id, top_k=top_k))
    top_features = list(dict.fromkeys(top_features))

    sources = list(FUSION_SOURCE_SPECS)
    heatmap_specs = [
        (
            "fig_source_weight_heatmap",
            [f"w_{source}" for source in sources],
            "Cross-fitted source weight",
            "YlOrRd",
            0,
            1,
        ),
        (
            "fig_source_reliability_heatmap",
            [f"{source}_source_reliability" for source in sources],
            "Source repeatability",
            "YlGnBu",
            0,
            1,
        ),
        (
            "fig_source_conflict_heatmap",
            [f"{source}_source_conflict" for source in sources],
            "Source disagreement",
            "OrRd",
            0,
            1,
        ),
        (
            "fig_source_signal_heatmap",
            [f"{source}_rank_signal" for source in sources],
            "Normalized rank evidence",
            "PuBuGn",
            0,
            1,
        ),
    ]
    for name, cols, cbar_label, cmap, vmin, vmax in heatmap_specs:
        cols = [c for c in cols if c in reliability_df.columns]
        if not cols:
            continue
        mat, ylabels, xlabels = _source_heatmap_matrix(
            reliability_df, cols, top_features
        )
        if mat.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(5.8, max(3.8, 0.32 * len(ylabels) + 1.2)))
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(
            [x.upper() if x == "pfi" else x.upper() for x in xlabels], rotation=0
        )
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=7.6)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{mat[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6.8,
                        color="white" if mat[i, j] > 0.58 else FIG_COLORS["dark"],
                    )
        cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
        cbar.set_label(cbar_label)
        _save_paper_figure(fig, figs_dir / name)
        plt.close(fig)


def save_weight_sensitivity_figure(
    sensitivity_df: pd.DataFrame, figs_dir: Path
) -> None:
    if not _has_matplotlib() or sensitivity_df is None or sensitivity_df.empty:
        return
    import matplotlib.pyplot as plt

    _setup_publication_style()
    df = sensitivity_df.copy()
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar(
        x - 0.18,
        df["jaccard_equal_adaptive"],
        width=0.34,
        color=FIG_COLORS["teal"],
        label="Top-k Jaccard",
    )
    if "spearman_equal_adaptive" in df.columns:
        ax.bar(
            x + 0.18,
            df["spearman_equal_adaptive"],
            width=0.34,
            color=FIG_COLORS["coral"],
            label="Rank Spearman",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(df["class_name"].astype(str), rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Stability")
    _clean_axis(ax, "y")
    ax.legend(frameon=False, loc="lower right")
    _save_paper_figure(fig, figs_dir / "fig_weighting_sensitivity_stability")
    plt.close(fig)


def save_negative_control_figure(
    negative_control_summary: pd.DataFrame, figs_dir: Path
) -> None:
    if (
        not _has_matplotlib()
        or negative_control_summary is None
        or negative_control_summary.empty
    ):
        return
    import matplotlib.pyplot as plt

    _setup_publication_style()
    df = (
        negative_control_summary.groupby("class_id", as_index=False)
        .agg(
            pfi=("mean_control_perm_percentile", "mean"),
            loco=("mean_control_loco_percentile", "mean"),
            firm=("mean_control_firm_percentile", "mean"),
        )
        .sort_values("class_id")
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    x = np.arange(len(df))
    sources = ["pfi", "loco", "firm"]
    width = 0.22
    offsets = (np.arange(len(sources)) - (len(sources) - 1) / 2.0) * width
    for offset, source in zip(offsets, sources):
        ax.bar(
            x + offset,
            df[source],
            width=width,
            color=FIG_SOURCE_COLORS[source],
            label=source.upper(),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        df["class_id"].map(lambda z: _class_label(int(z))), rotation=15, ha="right"
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Control percentile among substantive features")
    _clean_axis(ax, "y")
    ax.legend(frameon=False, loc="upper left")
    _save_paper_figure(fig, figs_dir / "fig_negative_control_response")
    plt.close(fig)


def save_firm_importance_figure(
    firm_df: pd.DataFrame,
    figs_dir: Path,
    top_k: int = 6,
) -> None:
    if not _has_matplotlib() or firm_df is None or firm_df.empty:
        return
    import matplotlib.pyplot as plt

    _setup_publication_style()
    run_level = firm_df.groupby(["seed", "class_id", "feature"], as_index=False).agg(
        firm_importance=("firm_importance", "mean")
    )
    summary = (
        run_level.groupby(["class_id", "feature"], as_index=False)
        .agg(
            mean_firm=("firm_importance", "mean"),
            sd_firm=("firm_importance", "std"),
        )
        .fillna(0.0)
    )
    classes = sorted(summary["class_id"].astype(int).unique())
    fig, axes = plt.subplots(
        1, len(classes), figsize=(4.0 * len(classes), 4.0), squeeze=False
    )
    for ax, class_id in zip(axes.ravel(), classes):
        sub = (
            summary.loc[summary["class_id"].astype(int) == int(class_id)]
            .sort_values(["mean_firm", "feature"], ascending=[False, True])
            .head(int(top_k))
            .sort_values("mean_firm", ascending=True)
        )
        y = np.arange(len(sub))
        ax.errorbar(
            sub["mean_firm"],
            y,
            xerr=sub["sd_firm"],
            fmt="o",
            color=FIG_SOURCE_COLORS["firm"],
            ecolor=FIG_COLORS["gray"],
            capsize=2.5,
            linewidth=1.0,
        )
        ax.set_yticks(y)
        ax.set_yticklabels([_feature_label(v, 18) for v in sub["feature"]])
        ax.set_title(_class_label(int(class_id)))
        ax.set_xlabel("FIRM relevance (probability SD)")
        _clean_axis(ax, "x")
    _save_paper_figure(fig, figs_dir / "fig_firm_observational_relevance")
    plt.close(fig)


def save_pdp_profile_figures(
    pdp_df: pd.DataFrame, reliability_df: pd.DataFrame, figs_dir: Path, top_k: int = 6
) -> None:
    if (
        not _has_matplotlib()
        or pdp_df is None
        or pdp_df.empty
        or reliability_df is None
        or reliability_df.empty
    ):
        return
    import matplotlib.pyplot as plt

    _setup_publication_style()
    df = pdp_df.copy()
    for class_id in sorted(df["class_id"].astype(int).unique()):
        top_features = _select_top_features(reliability_df, class_id, top_k=top_k)
        top_features = [f for f in top_features if f in set(df["feature"].astype(str))]
        if not top_features:
            continue
        n = len(top_features)
        ncols = 2
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(7.2, 2.75 * nrows), squeeze=False
        )
        for ax, feature in zip(axes.flat, top_features):
            sub = df.loc[
                (df["class_id"].astype(int) == int(class_id))
                & (df["feature"].astype(str) == str(feature))
            ].copy()
            seed_profiles = (
                (
                    sub.groupby(["feature_value", "value_order"], as_index=False).agg(
                        mean_pred=("mean_pred_proba", "mean"),
                        support=("train_support_share", "mean"),
                    )
                )
                if "seed" not in sub.columns
                else (
                    sub.groupby(
                        ["seed", "feature_value", "value_order"], as_index=False
                    ).agg(
                        mean_pred=("mean_pred_proba", "mean"),
                        support=("train_support_share", "mean"),
                    )
                )
            )
            agg = (
                seed_profiles.groupby(["feature_value", "value_order"], as_index=False)
                .agg(
                    mean_pred=("mean_pred", "mean"),
                    std_pred=("mean_pred", "std"),
                    support=("support", "mean"),
                    n_splits=("seed", "nunique")
                    if "seed" in seed_profiles.columns
                    else ("mean_pred", "count"),
                )
                .sort_values("value_order")
            )
            x = np.arange(len(agg))
            ax.plot(
                x,
                agg["mean_pred"],
                color=FIG_CLASS_COLORS.get(int(class_id), FIG_COLORS["teal"]),
                marker="o",
                linewidth=1.6,
            )
            if "std_pred" in agg.columns:
                sd = agg["std_pred"].fillna(0.0).to_numpy(dtype=float)
                y = agg["mean_pred"].to_numpy(dtype=float)
                ax.fill_between(
                    x,
                    np.maximum(0, y - sd),
                    np.minimum(1, y + sd),
                    color=FIG_CLASS_COLORS.get(int(class_id), FIG_COLORS["teal"]),
                    alpha=0.13,
                    linewidth=0,
                )
            ax2 = ax.twinx()
            ax2.bar(
                x,
                agg["support"] * 100.0,
                color=FIG_COLORS["light_gray"],
                alpha=0.45,
                width=0.72,
            )
            ax2.set_ylim(0, max(1.0, float((agg["support"] * 100.0).max()) * 1.2))
            ax2.set_ylabel("Support (%)", fontsize=8)
            ax2.tick_params(axis="y", labelsize=7)
            ax.set_xticks(x)
            ax.set_xticklabels(
                agg["feature_value"].astype(str), rotation=30, ha="right"
            )
            ax.set_ylim(0, 1)
            ax.set_title(_feature_label(feature), loc="left", fontweight="bold")
            ax.set_ylabel("Predicted probability")
            _clean_axis(ax, "y")
            ax2.spines["top"].set_visible(False)
        for ax in axes.flat[len(top_features) :]:
            ax.axis("off")
        fig.suptitle(_class_label(int(class_id)), x=0.02, ha="left", fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        _save_paper_figure(fig, figs_dir / f"fig_pdp_profiles_class{int(class_id)}")
        plt.close(fig)


def save_ice_heterogeneity_figures(
    ice_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    figs_dir: Path,
    top_k: int = 6,
) -> None:
    if not _has_matplotlib():
        return
    import matplotlib.pyplot as plt

    _setup_publication_style()
    if ice_df is not None and not ice_df.empty:
        ice = aggregate_ice(ice_df)
        if not ice.empty:
            classes = sorted(ice["class_id"].astype(int).unique())
            fig, axes = plt.subplots(
                1, len(classes), figsize=(4.6 * len(classes), 4.2), sharex=True
            )
            if len(classes) == 1:
                axes = [axes]
            for ax, class_id in zip(axes, classes):
                sub = (
                    ice.loc[ice["class_id"].astype(int) == int(class_id)]
                    .sort_values("ice_mean_centered_dispersion", ascending=False)
                    .head(int(top_k))
                )
                sub = sub.sort_values("ice_mean_centered_dispersion", ascending=True)
                y = np.arange(len(sub))
                ax.barh(
                    y,
                    sub["ice_mean_centered_dispersion"],
                    color=FIG_CLASS_COLORS.get(int(class_id), FIG_COLORS["teal"]),
                    alpha=0.86,
                )
                ax.set_yticks(y)
                ax.set_yticklabels([_feature_label(f, 18) for f in sub["feature"]])
                ax.set_title(_class_label(int(class_id)), loc="left", fontweight="bold")
                ax.set_xlabel("Centered ICE dispersion")
                _clean_axis(ax, "x")
            _save_paper_figure(fig, figs_dir / "fig_ice_heterogeneity_by_class")
            plt.close(fig)


# =============================================================================
# CREDAL SHADOW CALIBRATION
# =============================================================================


def shadow_replicate_count(run_profile: str) -> int:
    return (
        int(SHADOW_REPLICATES_SMOKE)
        if str(run_profile).lower() == "smoke"
        else int(SHADOW_REPLICATES_FULL)
    )


def make_inner_fold_specs(
    y_outer_train: np.ndarray,
    part: OuterPartition,
) -> List[InnerFoldSpec]:
    from sklearn.model_selection import StratifiedKFold

    y_values = np.asarray(y_outer_train, dtype=int)
    min_class_count = int(pd.Series(y_values).value_counts().min())
    n_splits = int(min(SHADOW_INNER_CV_FOLDS, min_class_count))
    if n_splits < 2:
        raise RuntimeError(
            f"Insufficient class support for shadow cross-fitting at "
            f"outer partition {part.partition_id}."
        )
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=stable_seed_from_key("inner_cv_split", part.split_seed),
    )
    return [
        InnerFoldSpec(
            fold_id=int(fold_id),
            subtrain_idx=np.asarray(subtrain_idx, dtype=int),
            validation_idx=np.asarray(validation_idx, dtype=int),
        )
        for fold_id, (subtrain_idx, validation_idx) in enumerate(
            cv.split(np.zeros(len(y_values)), y_values), 1
        )
    ]


def replace_raw_feature_from_training_marginal(
    X_subtrain: pd.DataFrame,
    X_validation: pd.DataFrame,
    feature: str,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Replace one raw feature using only its inner-training empirical marginal.

    True missing values, when present, are sampled jointly with observed values
    because they are elements of the empirical feature state. The original
    row-specific missingness mask is not retained.
    """
    if feature not in X_subtrain.columns or feature not in X_validation.columns:
        raise KeyError(f"Shadow feature is absent: {feature}")
    support = X_subtrain[feature].to_numpy(copy=True)
    if len(support) == 0:
        raise ValueError("Cannot construct a shadow from an empty inner-training set.")
    rng = np.random.RandomState(int(seed))
    train_draw = support[
        rng.randint(0, len(support), size=len(X_subtrain))
    ]
    validation_draw = support[
        rng.randint(0, len(support), size=len(X_validation))
    ]
    shadow_train = X_subtrain.copy()
    shadow_validation = X_validation.copy()
    shadow_train.loc[:, feature] = train_draw
    shadow_validation.loc[:, feature] = validation_draw
    return shadow_train, shadow_validation


def _annotate_evidence(
    table: pd.DataFrame,
    part: OuterPartition,
    model_name: str,
    role: str,
    inner_fold: Optional[int] = None,
    shadow_feature: Optional[str] = None,
    shadow_replicate: Optional[int] = None,
) -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame()
    out = table.copy()
    out["outer_partition"] = int(part.partition_id)
    out["outer_repeat"] = int(part.repeat_id)
    out["outer_fold"] = int(part.fold_id)
    out["split_seed"] = int(part.split_seed)
    out["model"] = str(model_name)
    out["evidence_role"] = str(role)
    out["inner_fold"] = (
        int(inner_fold) if inner_fold is not None else np.nan
    )
    out["shadow_feature"] = (
        str(shadow_feature) if shadow_feature is not None else ""
    )
    out["shadow_replicate"] = (
        int(shadow_replicate) if shadow_replicate is not None else 0
    )
    return out


def _safe_weighted_group_mean(
    frame: pd.DataFrame,
    value_column: str,
    count_column: str,
) -> float:
    values = pd.to_numeric(frame[value_column], errors="coerce").to_numpy(dtype=float)
    counts = pd.to_numeric(frame[count_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(counts) & (counts > 0)
    if not np.any(valid):
        return np.nan
    return float(np.sum(values[valid] * counts[valid]) / np.sum(counts[valid]))


def aggregate_crossfitted_source_statistics(
    source: str,
    fold_tables: Sequence[pd.DataFrame],
    feature_cols: Sequence[str],
    class_order: Sequence[int],
    model_name: str,
    outer_partition: int,
) -> pd.DataFrame:
    """Pool source-specific OOF primitives according to the source estimand."""
    source_name = str(source).upper()
    table = concat_or_empty(list(fold_tables))
    if table.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for class_id in class_order:
        for feature in feature_cols:
            group = table.loc[
                (pd.to_numeric(table["class_id"], errors="coerce") == int(class_id))
                & (table["feature"].astype(str) == str(feature))
            ].copy()
            if group.empty:
                continue
            if source_name == "SHAP":
                focal = _safe_weighted_group_mean(
                    group, "focal_mean_abs_shap", "n_focal"
                )
                other = _safe_weighted_group_mean(
                    group, "other_mean_abs_shap", "n_other"
                )
                statistic = 0.5 * focal + 0.5 * other
            elif source_name == "PFI":
                baseline_focal = _safe_weighted_group_mean(
                    group, "baseline_focal_brier_mean", "n_focal"
                )
                baseline_other = _safe_weighted_group_mean(
                    group, "baseline_other_brier_mean", "n_other"
                )
                permuted_focal = _safe_weighted_group_mean(
                    group, "permuted_focal_brier_mean", "n_focal"
                )
                permuted_other = _safe_weighted_group_mean(
                    group, "permuted_other_brier_mean", "n_other"
                )
                statistic = 0.5 * (permuted_focal - baseline_focal) + 0.5 * (
                    permuted_other - baseline_other
                )
            elif source_name == "LOCO":
                full_focal = _safe_weighted_group_mean(
                    group, "full_focal_brier_mean", "n_focal"
                )
                full_other = _safe_weighted_group_mean(
                    group, "full_other_brier_mean", "n_other"
                )
                reduced_focal = _safe_weighted_group_mean(
                    group, "reduced_focal_brier_mean", "n_focal"
                )
                reduced_other = _safe_weighted_group_mean(
                    group, "reduced_other_brier_mean", "n_other"
                )
                statistic = 0.5 * (reduced_focal - full_focal) + 0.5 * (
                    reduced_other - full_other
                )
            else:
                raise ValueError(
                    f"Primitive pooling is not defined for source={source_name}."
                )
            rows.append(
                {
                    "outer_partition": int(outer_partition),
                    "model": str(model_name),
                    "source": source_name,
                    "class_id": int(class_id),
                    "feature": str(feature),
                    "raw_statistic": float(statistic),
                    "n_focal": int(
                        pd.to_numeric(group["n_focal"], errors="coerce").sum()
                    ),
                    "n_other": int(
                        pd.to_numeric(group["n_other"], errors="coerce").sum()
                    ),
                    "n_inner_folds": int(group["inner_fold"].nunique())
                    if "inner_fold" in group.columns
                    else int(len(group)),
                }
            )
    return pd.DataFrame(rows)


def aggregate_crossfitted_firm_statistics(
    oof_payloads: Sequence[Mapping[str, Any]],
    schema: FeatureSchema,
    feature_cols: Sequence[str],
    class_order: Sequence[int],
    model_name: str,
    outer_partition: int,
) -> pd.DataFrame:
    if not oof_payloads:
        return pd.DataFrame()
    X_oof = pd.concat(
        [payload["X_eval_raw"] for payload in oof_payloads],
        axis=0,
        ignore_index=True,
    )
    y_oof = np.concatenate(
        [np.asarray(payload["y_eval"], dtype=int) for payload in oof_payloads]
    )
    proba_oof = np.concatenate(
        [np.asarray(payload["proba"], dtype=float) for payload in oof_payloads],
        axis=0,
    )
    firm = compute_firm_importance(
        base_proba=proba_oof,
        X_eval_raw=X_oof,
        y_eval=y_oof,
        schema=schema,
        classes=class_order,
        model_name=model_name,
        seed=stable_seed_from_key("pooled_firm", outer_partition, model_name),
        feature_cols=list(feature_cols),
    )
    if firm.empty:
        return pd.DataFrame()
    out = firm[
        ["model", "class_id", "feature", "balanced_firm_importance", "n_eval"]
    ].copy()
    out["outer_partition"] = int(outer_partition)
    out["source"] = "FIRM"
    out["raw_statistic"] = pd.to_numeric(
        out["balanced_firm_importance"], errors="coerce"
    )
    out["n_inner_folds"] = int(len(oof_payloads))
    return out[
        [
            "outer_partition",
            "model",
            "source",
            "class_id",
            "feature",
            "raw_statistic",
            "n_eval",
            "n_inner_folds",
        ]
    ]


def standardize_outer_source_table(
    source: str,
    table: pd.DataFrame,
    part: OuterPartition,
    model_name: str,
) -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame()
    source_name = str(source).upper()
    if source_name not in CREDAL_SOURCE_SPECS:
        raise KeyError(f"Unknown credal source: {source_name}")
    value_column = str(CREDAL_SOURCE_SPECS[source_name]["value"])
    if value_column not in table.columns:
        raise KeyError(
            f"{source_name} output lacks its prespecified statistic: {value_column}"
        )
    out = table[["class_id", "feature", value_column]].copy()
    out = out.rename(columns={value_column: "raw_statistic"})
    out["outer_partition"] = int(part.partition_id)
    out["outer_repeat"] = int(part.repeat_id)
    out["outer_fold"] = int(part.fold_id)
    out["split_seed"] = int(part.split_seed)
    out["model"] = str(model_name)
    out["source"] = source_name
    out["evidence_role"] = "outer_holdout_actual"
    return out[
        [
            "outer_partition",
            "outer_repeat",
            "outer_fold",
            "split_seed",
            "model",
            "source",
            "class_id",
            "feature",
            "raw_statistic",
            "evidence_role",
        ]
    ]


def source_table_contract_status(
    table: pd.DataFrame,
    source: str,
    class_id: int,
    feature_cols: Sequence[str],
) -> Tuple[bool, str]:
    if table is None or table.empty:
        return False, "missing_table"
    group = table.loc[
        (table["source"].astype(str) == str(source))
        & (pd.to_numeric(table["class_id"], errors="coerce") == int(class_id))
    ].copy()
    expected = {str(feature) for feature in feature_cols}
    observed = set(group["feature"].astype(str))
    if observed != expected:
        return False, "incomplete_feature_set"
    if group["feature"].astype(str).duplicated().any():
        return False, "duplicate_feature_rows"
    values = pd.to_numeric(group["raw_statistic"], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        return False, "nonfinite_statistic"
    return True, "valid"


def fit_fixed_predictive_specification(
    X_train_raw: pd.DataFrame,
    X_eval_raw: pd.DataFrame,
    y_train: np.ndarray,
    schema: FeatureSchema,
    model_name: str,
    fit_seed: int,
    class_order: Sequence[int],
) -> Tuple[object, EncodedData, np.ndarray, np.ndarray]:
    """Refit one fixed algorithm/hyperparameter/preprocessing specification."""
    encoded = fit_transform_split(
        X_train_raw.reset_index(drop=True),
        X_eval_raw.reset_index(drop=True),
        schema,
        model_name=model_name,
    )
    estimator = build_estimator(
        model_name,
        seed=int(fit_seed),
        n_jobs=MODEL_N_JOBS,
        categorical_feature_indices=encoded.categorical_feature_indices,
    )
    sample_weight = (
        make_balanced_sample_weight(np.asarray(y_train, dtype=int))
        if USE_BALANCED_SAMPLE_WEIGHT
        else None
    )
    fit_estimator(
        estimator,
        encoded.X_train,
        np.asarray(y_train, dtype=int),
        sample_weight=sample_weight,
    )
    proba = predict_proba_aligned(
        estimator,
        encoded.X_test,
        np.asarray(class_order, dtype=int),
    )
    pred = np.asarray(class_order, dtype=int)[np.argmax(proba, axis=1)]
    return estimator, encoded, proba, pred


def ensure_pfi_batch_equivalence_audit(
    X_outer_train: pd.DataFrame,
    y_outer_train: np.ndarray,
    schema: FeatureSchema,
    part: OuterPartition,
    model_name: str,
    class_order: Sequence[int],
    audit_root: Path,
) -> pd.DataFrame:
    """Verify that batched PFI reproduces the original repeat loop."""
    audit_root = Path(audit_root)
    audit_root.mkdir(parents=True, exist_ok=True)
    safe_model = _safe_path_component(model_name, limit=20)
    csv_path = audit_root / (
        f"p{int(part.partition_id):03d}_m{safe_model}.csv"
    )
    marker_path = csv_path.with_suffix(".json")
    folds = make_inner_fold_specs(y_outer_train, part)
    if not folds:
        raise RuntimeError("PFI equivalence audit has no inner fold.")
    fold = folds[0]
    source_seed = stable_seed_from_key(
        "fixed_source_protocol",
        part.partition_id,
        fold.fold_id,
        model_name,
    )
    audit_fingerprint = stable_json_digest(
        {
            "contract": "raw_pfi_batch_equivalence_v1",
            "outer_partition": int(part.partition_id),
            "inner_fold": int(fold.fold_id),
            "model": str(model_name),
            "subtrain_index": array_digest(
                np.asarray(fold.subtrain_idx, dtype=np.int64)
            ),
            "validation_index": array_digest(
                np.asarray(fold.validation_idx, dtype=np.int64)
            ),
            "outer_train": dataframe_digest(
                X_outer_train.reset_index(drop=True)
            ),
            "labels": array_digest(
                np.asarray(y_outer_train, dtype=np.int64)
            ),
            "features": list(schema.feature_cols),
            "repeats": int(PERMUTATION_REPEATS),
            "source_seed": int(source_seed),
            "batch_rows": int(PFI_BATCH_PREDICTION_ROWS),
            "atol": float(PFI_EQUIVALENCE_ATOL),
            "rtol": float(PFI_EQUIVALENCE_RTOL),
            "packages": {
                package: package_version(package)
                for package in [
                    "numpy",
                    "pandas",
                    "scikit-learn",
                ]
            },
        }
    )
    if marker_path.exists() and csv_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("fingerprint") == audit_fingerprint
            and bool(marker.get("passed"))
        ):
            return pd.read_csv(csv_path)

    X_subtrain = X_outer_train.iloc[
        fold.subtrain_idx
    ].reset_index(drop=True)
    X_validation = X_outer_train.iloc[
        fold.validation_idx
    ].reset_index(drop=True)
    y_subtrain = np.asarray(y_outer_train, dtype=int)[
        fold.subtrain_idx
    ]
    y_validation = np.asarray(y_outer_train, dtype=int)[
        fold.validation_idx
    ]
    fit_seed = stable_seed_from_key(
        "fixed_spec_inner_fit",
        part.partition_id,
        fold.fold_id,
        model_name,
    )
    estimator, encoded, proba, _ = fit_fixed_predictive_specification(
        X_train_raw=X_subtrain,
        X_eval_raw=X_validation,
        y_train=y_subtrain,
        schema=schema,
        model_name=model_name,
        fit_seed=fit_seed,
        class_order=class_order,
    )

    selected_features: List[str] = []
    if schema.numeric_cols:
        selected_features.append(str(schema.numeric_cols[0]))
    if schema.categorical_cols:
        selected_features.append(str(schema.categorical_cols[0]))
    for feature in schema.feature_cols:
        if len(selected_features) >= int(PFI_EQUIVALENCE_FEATURES):
            break
        if str(feature) not in selected_features:
            selected_features.append(str(feature))
    selected_features = selected_features[
        : max(1, int(PFI_EQUIVALENCE_FEATURES))
    ]

    reference = compute_class_permutation_importance_reference(
        estimator=estimator,
        preprocessor=encoded.preprocessor,
        X_eval_raw=X_validation,
        y_eval=y_validation,
        classes=class_order,
        feature_cols=selected_features,
        model_name=model_name,
        seed=int(source_seed),
        repeats=int(PERMUTATION_REPEATS),
        base_proba=proba,
    )
    batched = compute_class_permutation_importance(
        estimator=estimator,
        preprocessor=encoded.preprocessor,
        X_eval_raw=X_validation,
        y_eval=y_validation,
        classes=class_order,
        feature_cols=selected_features,
        model_name=model_name,
        seed=int(source_seed),
        repeats=int(PERMUTATION_REPEATS),
        base_proba=proba,
        use_batched_prediction=True,
    )
    keys = ["model", "class_id", "feature"]
    reference = reference.sort_values(keys).reset_index(drop=True)
    batched = batched.sort_values(keys).reset_index(drop=True)
    if reference[keys].astype(str).to_dict("records") != batched[
        keys
    ].astype(str).to_dict("records"):
        raise RuntimeError(
            f"PFI batch audit key mismatch for {model_name}."
        )

    excluded_numeric = {"seed", "class_id", "n_focal", "n_other", "n_eval"}
    numeric_columns = [
        column
        for column in reference.columns
        if column in batched.columns
        and column not in excluded_numeric
        and pd.api.types.is_numeric_dtype(reference[column])
        and pd.api.types.is_numeric_dtype(batched[column])
    ]
    audit_rows = []
    all_passed = True
    for column in numeric_columns:
        left = reference[column].to_numpy(dtype=float)
        right = batched[column].to_numpy(dtype=float)
        difference = np.abs(left - right)
        scale = np.maximum(np.abs(left), np.finfo(float).eps)
        passed = bool(
            np.allclose(
                left,
                right,
                atol=float(PFI_EQUIVALENCE_ATOL),
                rtol=float(PFI_EQUIVALENCE_RTOL),
                equal_nan=True,
            )
        )
        all_passed = all_passed and passed
        audit_rows.append(
            {
                "outer_partition": int(part.partition_id),
                "inner_fold": int(fold.fold_id),
                "model": str(model_name),
                "column": str(column),
                "n_values": int(len(left)),
                "max_abs_error": float(np.nanmax(difference)),
                "mean_abs_error": float(np.nanmean(difference)),
                "max_relative_error": float(
                    np.nanmax(difference / scale)
                ),
                "atol": float(PFI_EQUIVALENCE_ATOL),
                "rtol": float(PFI_EQUIVALENCE_RTOL),
                "passed": int(passed),
            }
        )
    audit = pd.DataFrame(audit_rows)
    atomic_dataframe_to_csv(audit, csv_path)
    atomic_json_dump(
        {
            "fingerprint": audit_fingerprint,
            "passed": bool(all_passed),
            "model": str(model_name),
            "outer_partition": int(part.partition_id),
            "inner_fold": int(fold.fold_id),
            "features_checked": selected_features,
            "completed_at": datetime.now().isoformat(
                timespec="seconds"
            ),
        },
        marker_path,
    )
    del estimator
    if SHADOW_WORKER_GC_AT_TASK_END:
        gc.collect()
    if not all_passed:
        raise RuntimeError(
            "Batched PFI did not reproduce the original PFI implementation "
            f"for outer={part.partition_id}, model={model_name}. "
            f"See {csv_path}."
        )
    return audit


def compute_fitted_source_tables(
    estimator,
    encoded: EncodedData,
    proba: np.ndarray,
    y_train: np.ndarray,
    y_eval: np.ndarray,
    X_train_raw: pd.DataFrame,
    X_eval_raw: pd.DataFrame,
    schema: FeatureSchema,
    class_order: Sequence[int],
    model_name: str,
    source_seed: int,
    feature_cols: Sequence[str],
    worker_root: Path,
    call_key: str,
    include_firm_table: bool,
    shap_phase_scope: str,
    outer_partition: int,
    inner_fold: int,
    loco_reduced_proba_by_feature: Optional[
        Mapping[str, np.ndarray]
    ] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Apply the fixed four-source execution contract to one fitted model."""
    tables: Dict[str, pd.DataFrame] = {}
    errors: Dict[str, str] = {}
    requested_features = [str(feature) for feature in feature_cols]

    try:
        shap_table = compute_target_coordinate_shap_importance(
            estimator=estimator,
            preprocessor=encoded.preprocessor,
            X_train_raw=X_train_raw.reset_index(drop=True),
            X_eval_raw=X_eval_raw.reset_index(drop=True),
            y_eval=np.asarray(y_eval, dtype=int),
            schema=schema,
            classes=class_order,
            model_name=model_name,
            target_features=requested_features,
            phase_scope=str(shap_phase_scope),
            outer_partition=int(outer_partition),
            inner_fold=int(inner_fold),
        )
        require_full_evaluation(shap_table, "n_eval", len(X_eval_raw), "SHAP")
        tables["SHAP"] = shap_table.loc[
            shap_table["feature"].astype(str).isin(requested_features)
        ].copy()
    except Exception as exc:
        errors["SHAP"] = f"{type(exc).__name__}: {exc}"

    try:
        pfi_table = compute_class_permutation_importance(
            estimator=estimator,
            preprocessor=encoded.preprocessor,
            X_eval_raw=X_eval_raw.reset_index(drop=True),
            y_eval=np.asarray(y_eval, dtype=int),
            classes=class_order,
            feature_cols=requested_features,
            model_name=model_name,
            seed=int(source_seed),
            repeats=int(PERMUTATION_REPEATS),
            base_proba=np.asarray(proba, dtype=float),
        )
        require_full_evaluation(pfi_table, "n_eval", len(X_eval_raw), "PFI")
        tables["PFI"] = pfi_table
    except Exception as exc:
        errors["PFI"] = f"{type(exc).__name__}: {exc}"

    try:
        loco_table = compute_loco_importance(
            base_proba=np.asarray(proba, dtype=float),
            base_pred=np.asarray(class_order, dtype=int)[
                np.argmax(np.asarray(proba, dtype=float), axis=1)
            ],
            y_train=np.asarray(y_train, dtype=int),
            y_eval=np.asarray(y_eval, dtype=int),
            X_train_raw=X_train_raw.reset_index(drop=True),
            X_eval_raw=X_eval_raw.reset_index(drop=True),
            schema=schema,
            classes=class_order,
            model_name=model_name,
            seed=int(source_seed),
            n_jobs=MODEL_N_JOBS,
            sample_weight=(
                make_balanced_sample_weight(np.asarray(y_train, dtype=int))
                if USE_BALANCED_SAMPLE_WEIGHT
                else None
            ),
            candidate_features=requested_features,
            precomputed_reduced_proba=loco_reduced_proba_by_feature,
        )
        require_full_evaluation(loco_table, "n_eval", len(X_eval_raw), "LOCO")
        tables["LOCO"] = loco_table
    except Exception as exc:
        errors["LOCO"] = f"{type(exc).__name__}: {exc}"

    if include_firm_table:
        try:
            firm_table = compute_firm_importance(
                base_proba=np.asarray(proba, dtype=float),
                X_eval_raw=X_eval_raw.reset_index(drop=True),
                y_eval=np.asarray(y_eval, dtype=int),
                schema=schema,
                classes=class_order,
                model_name=model_name,
                seed=int(source_seed),
                feature_cols=requested_features,
            )
            require_full_evaluation(firm_table, "n_eval", len(X_eval_raw), "FIRM")
            tables["FIRM"] = firm_table
        except Exception as exc:
            errors["FIRM"] = f"{type(exc).__name__}: {exc}"

    return tables, errors


def credal_unit_fingerprint(
    kind: str,
    part: OuterPartition,
    model_name: str,
    feature: Optional[str],
    replicate: Optional[int],
    data_digest: str,
    method_digest: str,
) -> str:
    return stable_json_digest(
        {
            "kind": str(kind),
            "partition": int(part.partition_id),
            "repeat": int(part.repeat_id),
            "fold": int(part.fold_id),
            "train_index": array_digest(np.asarray(part.train_idx, dtype=np.int64)),
            "holdout_index": array_digest(np.asarray(part.test_idx, dtype=np.int64)),
            "model": str(model_name),
            "feature": None if feature is None else str(feature),
            "replicate": None if replicate is None else int(replicate),
            "data_digest": str(data_digest),
            "method_digest": str(method_digest),
        }
    )


def compatible_legacy_unit_fingerprints(
    kind: str,
    part: OuterPartition,
    model_name: str,
    feature: Optional[str],
    replicate: Optional[int],
    data_digest: str,
    legacy_method_candidates: Sequence[Mapping[str, str]],
) -> Dict[str, Dict[str, str]]:
    """Map every recognized legacy unit fingerprint to its script provenance."""
    out: Dict[str, Dict[str, str]] = {}
    for candidate in legacy_method_candidates:
        method_digest = str(candidate.get("method_digest", "")).strip()
        if not method_digest:
            continue
        fingerprint = credal_unit_fingerprint(
            kind=kind,
            part=part,
            model_name=model_name,
            feature=feature,
            replicate=replicate,
            data_digest=data_digest,
            method_digest=method_digest,
        )
        out[fingerprint] = {
            "script_sha256": str(
                candidate.get("script_sha256", "unknown")
            ),
            "script_path": str(
                candidate.get("script_path", "unknown")
            ),
            "discovery": str(
                candidate.get("discovery", "unknown")
            ),
            "legacy_method_digest": method_digest,
        }
    return out


def credal_unit_paths(
    checkpoint_root: Path,
    kind: str,
    part: OuterPartition,
    model_name: str,
    feature: Optional[str],
    replicate: Optional[int],
) -> Tuple[Path, Path]:
    key_digest = stable_json_digest(
        {
            "kind": str(kind),
            "partition": int(part.partition_id),
            "model": str(model_name),
            "feature": feature,
            "replicate": replicate,
        }
    )[:16]
    stem = (
        f"p{int(part.partition_id):03d}_"
        f"m{_safe_path_component(model_name, limit=18)}_{key_digest}"
    )
    directory = Path(checkpoint_root) / _safe_path_component(kind, limit=30)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{stem}.pkl", directory / f"{stem}.json"


def load_credal_unit_checkpoint(
    checkpoint_root: Path,
    kind: str,
    part: OuterPartition,
    model_name: str,
    feature: Optional[str],
    replicate: Optional[int],
    expected_fingerprint: str,
    compatible_legacy_fingerprints: Optional[Any] = None,
) -> Optional[dict]:
    payload_path, marker_path = credal_unit_paths(
        checkpoint_root, kind, part, model_name, feature, replicate
    )
    if not marker_path.exists():
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    observed_fingerprint = str(marker.get("fingerprint", ""))
    legacy_metadata_by_fingerprint: Dict[str, Dict[str, object]] = {}
    if isinstance(compatible_legacy_fingerprints, Mapping):
        for fingerprint, metadata in compatible_legacy_fingerprints.items():
            legacy_metadata_by_fingerprint[str(fingerprint)] = (
                dict(metadata)
                if isinstance(metadata, Mapping)
                else {"source": str(metadata)}
            )
    else:
        for fingerprint in compatible_legacy_fingerprints or []:
            legacy_metadata_by_fingerprint[str(fingerprint)] = {}
    compatible = set(legacy_metadata_by_fingerprint)
    if (
        observed_fingerprint != str(expected_fingerprint)
        and observed_fingerprint not in compatible
    ):
        policy = str(INCOMPATIBLE_CHECKPOINT_POLICY).strip().lower()
        if policy == "recompute":
            rejection_dir = (
                Path(checkpoint_root)
                / "_incompatible_checkpoint_audit"
                / _safe_path_component(kind, limit=30)
            )
            rejection_dir.mkdir(parents=True, exist_ok=True)
            rejection_path = rejection_dir / marker_path.name
            atomic_json_dump(
                {
                    "status": "rejected_and_recompute",
                    "kind": str(kind),
                    "outer_partition": int(part.partition_id),
                    "model": str(model_name),
                    "feature": (
                        None if feature is None else str(feature)
                    ),
                    "replicate": (
                        None if replicate is None else int(replicate)
                    ),
                    "observed_fingerprint": observed_fingerprint,
                    "expected_fingerprint": str(expected_fingerprint),
                    "recognized_legacy_fingerprints": sorted(compatible),
                    "original_marker_path": str(marker_path),
                    "original_payload_path": str(payload_path),
                    "rejected_at": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                },
                rejection_path,
            )
            cprint(
                "[CHECKPOINT RECOMPUTE] unrecognized fingerprint | "
                f"kind={kind} outer={int(part.partition_id)} "
                f"model={model_name} feature={feature} "
                f"replicate={replicate}"
            )
            return None
        raise RuntimeError(
            f"Incompatible {kind} checkpoint for outer={part.partition_id}, "
            f"model={model_name}, feature={feature}, replicate={replicate}."
        )
    if not payload_path.exists():
        raise RuntimeError(f"Checkpoint marker lacks payload: {payload_path}")
    if file_sha256(str(payload_path)) != marker.get("payload_sha256"):
        raise RuntimeError(f"Checkpoint checksum mismatch: {payload_path}")
    payload = pd.read_pickle(payload_path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid checkpoint payload: {payload_path}")
    payload["_checkpoint_origin_fingerprint"] = observed_fingerprint
    payload["_checkpoint_requires_migration"] = bool(
        observed_fingerprint != str(expected_fingerprint)
    )
    payload["_checkpoint_legacy_metadata"] = dict(
        legacy_metadata_by_fingerprint.get(observed_fingerprint, {})
    )
    return payload


def save_credal_unit_checkpoint(
    checkpoint_root: Path,
    kind: str,
    part: OuterPartition,
    model_name: str,
    feature: Optional[str],
    replicate: Optional[int],
    fingerprint: str,
    payload: dict,
    marker_metadata: Optional[Mapping[str, object]] = None,
) -> None:
    payload_path, marker_path = credal_unit_paths(
        checkpoint_root, kind, part, model_name, feature, replicate
    )
    serializable = dict(payload)
    serializable["fingerprint"] = str(fingerprint)
    serializable["saved_at"] = datetime.now().isoformat(timespec="seconds")
    atomic_pickle_dump(serializable, payload_path)
    marker = {
        "format_version": int(SHADOW_CHECKPOINT_FORMAT_VERSION),
        "fingerprint": str(fingerprint),
        "payload_sha256": file_sha256(str(payload_path)),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    if marker_metadata:
        marker.update(dict(marker_metadata))
    atomic_json_dump(marker, marker_path)


def migrate_compatible_credal_checkpoint(
    checkpoint_root: Path,
    kind: str,
    part: OuterPartition,
    model_name: str,
    feature: Optional[str],
    replicate: Optional[int],
    expected_fingerprint: str,
    payload: Mapping[str, Any],
    compatibility_audit: str,
) -> dict:
    """Rewrite a validated known-version checkpoint under the new contract."""
    migrated = dict(payload)
    origin = str(
        migrated.pop("_checkpoint_origin_fingerprint", "unknown")
    )
    legacy_metadata = migrated.pop(
        "_checkpoint_legacy_metadata", {}
    )
    if not isinstance(legacy_metadata, Mapping):
        legacy_metadata = {}
    migrated.pop("_checkpoint_requires_migration", None)
    save_credal_unit_checkpoint(
        checkpoint_root=checkpoint_root,
        kind=kind,
        part=part,
        model_name=model_name,
        feature=feature,
        replicate=replicate,
        fingerprint=str(expected_fingerprint),
        payload=migrated,
        marker_metadata={
            "migrated_from_fingerprint": origin,
            "legacy_script_sha256": (
                str(
                    legacy_metadata.get(
                        "script_sha256",
                        LEGACY_TARGET_SHAP_V1_SCRIPT_SHA256,
                    )
                )
            ),
            "legacy_script_path": legacy_metadata.get("script_path"),
            "compatibility_audit_passed": str(compatibility_audit),
            "migration_timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
        },
    )
    cprint(
        "[CHECKPOINT MIGRATED] "
        f"kind={kind} outer={int(part.partition_id)} "
        f"model={model_name} feature={feature} replicate={replicate} "
        f"legacy_script={legacy_metadata.get('script_path', 'unknown')}"
    )
    migrated["_checkpoint_origin_fingerprint"] = str(
        expected_fingerprint
    )
    migrated["_checkpoint_requires_migration"] = False
    return migrated


def _credal_statistical_contract_payload(
    schema: FeatureSchema,
) -> Dict[str, object]:
    return {
        "contract_version": str(STATISTICAL_CONTRACT_VERSION),
        "predictive_specification_contract": (
            PREDICTIVE_SPECIFICATION_CONTRACT_VERSION
        ),
        "preprocessing_contract": PREPROCESSING_CONTRACT_VERSION,
        "source_statistic_contract": SOURCE_STATISTIC_CONTRACT_VERSION,
        "randomization_contract": RANDOMIZATION_CONTRACT_VERSION,
        "method": "credal_specification_robust_multi_explainer_inference",
        "features": list(schema.feature_cols),
        "numeric_features": list(schema.numeric_cols),
        "categorical_features": list(schema.categorical_cols),
        "candidate_models": list(CANDIDATE_MODELS),
        "one_se_model_admission": bool(
            USE_ONE_SE_ADMISSIBLE_MODEL_SET
        ),
        "sources": list(CREDAL_SOURCE_NAMES),
        "source_specs": CREDAL_SOURCE_SPECS,
        "shadow_null": SHADOW_NULL_MODE,
        "shadow_replicates": int(shadow_replicate_count(RUN_PROFILE)),
        "shadow_inner_folds": int(SHADOW_INNER_CV_FOLDS),
        "shap_background_size": int(SHAP_BACKGROUND_SIZE),
        "shap_cycles": int(SHAP_PERMUTATION_CYCLES),
        "target_shap_contract": TARGET_SHAP_CONTRACT_VERSION,
        "target_shap_primary_plan": (
            TARGET_SHAP_PERMUTATION_PLAN_VERSION
        ),
        "target_shap_sensitivity_plan": (
            TARGET_SHAP_SENSITIVITY_PLAN_VERSION
        ),
        "outer_folds": int(
            OUTER_N_SPLITS_SMOKE
            if str(RUN_PROFILE).lower() == "smoke"
            else OUTER_N_SPLITS_FULL
        ),
        "outer_repeats": int(
            OUTER_N_REPEATS_SMOKE
            if str(RUN_PROFILE).lower() == "smoke"
            else OUTER_N_REPEATS_FULL
        ),
        "pfi_repeats": int(PERMUTATION_REPEATS),
        "firm_continuous_features": list(FIRM_CONTINUOUS_FEATURES),
        "balanced_sample_weight": bool(USE_BALANCED_SAMPLE_WEIGHT),
        "numerical_tolerance": float(SHADOW_NUMERICAL_TOLERANCE),
        "package_versions": {
            package: package_version(package)
            for package in [
                "numpy",
                "pandas",
                "scikit-learn",
                "shap",
            ]
        },
    }


def credal_method_digest(schema: FeatureSchema) -> str:
    """Hash only choices that can change the resulting statistics."""
    return stable_json_digest(_credal_statistical_contract_payload(schema))


def legacy_credal_method_digest(
    schema: FeatureSchema,
    script_sha256: Optional[str] = None,
) -> str:
    """Reconstruct a target-SHAP-v1 digest for controlled migration."""
    legacy_script_sha256 = str(
        script_sha256 or LEGACY_TARGET_SHAP_V1_SCRIPT_SHA256
    ).strip().lower()
    if len(legacy_script_sha256) != 64:
        raise ValueError(
            "A legacy workflow script SHA256 must contain 64 hexadecimal "
            "characters."
        )
    return stable_json_digest(
        {
            "method": "credal_specification_robust_multi_explainer_inference",
            "features": list(schema.feature_cols),
            "sources": list(CREDAL_SOURCE_NAMES),
            "source_specs": CREDAL_SOURCE_SPECS,
            "shadow_null": SHADOW_NULL_MODE,
            "shadow_replicates": int(shadow_replicate_count(RUN_PROFILE)),
            "shadow_parallel_workers": 4,
            "shadow_inner_folds": int(SHADOW_INNER_CV_FOLDS),
            "shap_background_size": int(SHAP_BACKGROUND_SIZE),
            "shap_cycles": int(SHAP_PERMUTATION_CYCLES),
            "target_shap_contract": TARGET_SHAP_CONTRACT_VERSION,
            "target_shap_primary_plan": (
                TARGET_SHAP_PERMUTATION_PLAN_VERSION
            ),
            "target_shap_sensitivity_plan": (
                TARGET_SHAP_SENSITIVITY_PLAN_VERSION
            ),
            "target_shap_prediction_batch_rows": 32768,
            "target_shap_equivalence_records": 12,
            "outer_folds": int(
                OUTER_N_SPLITS_SMOKE
                if str(RUN_PROFILE).lower() == "smoke"
                else OUTER_N_SPLITS_FULL
            ),
            "outer_repeats": int(
                OUTER_N_REPEATS_SMOKE
                if str(RUN_PROFILE).lower() == "smoke"
                else OUTER_N_REPEATS_FULL
            ),
            "pfi_repeats": int(PERMUTATION_REPEATS),
            "balanced_sample_weight": bool(USE_BALANCED_SAMPLE_WEIGHT),
            "numerical_tolerance": float(SHADOW_NUMERICAL_TOLERANCE),
            "workflow_script_sha256": legacy_script_sha256,
            "script_format_version": 2,
            "package_versions": {
                package: package_version(package)
                for package in [
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "shap",
                ]
            },
        }
    )


def discover_legacy_credal_method_candidates(
    schema: FeatureSchema,
) -> List[Dict[str, str]]:
    """Find byte variants of the frozen target-SHAP-v1 workflow.

    The legacy fingerprint included the raw script SHA, so changing LF to CRLF
    changed every checkpoint fingerprint even when the executed Python program
    was semantically identical. Discovery is restricted to scripts that expose
    all frozen legacy contract tokens. A candidate hash is useful only if the
    reconstructed unit fingerprint exactly matches the stored marker.
    """
    candidates_by_sha: Dict[str, Dict[str, str]] = {
        str(LEGACY_TARGET_SHAP_V1_SCRIPT_SHA256).lower(): {
            "script_sha256": str(
                LEGACY_TARGET_SHAP_V1_SCRIPT_SHA256
            ).lower(),
            "script_path": "bundled_known_legacy_hash",
            "discovery": "bundled_known_hash",
        }
    }
    candidate_paths: Set[Path] = set()
    for value in LEGACY_RESUME_SCRIPT_PATHS:
        if str(value).strip():
            candidate_paths.add(
                Path(str(value)).expanduser().resolve()
            )
    if AUTO_DISCOVER_LEGACY_RESUME_SCRIPTS:
        candidate_paths.update(
            path.resolve()
            for path in Path(__file__).resolve().parent.glob("*.py")
        )

    required_tokens = (
        "SHADOW_CHECKPOINT_FORMAT_VERSION=2",
        "SHADOW_PARALLEL_WORKERS=4",
        "SHADOW_REPLICATES_FULL=49",
        'TARGET_SHAP_CONTRACT_VERSION="target_coordinate_permutation_shap_v1"',
        'TARGET_SHAP_PERMUTATION_PLAN_VERSION="common_plan_v1"',
        "TARGET_SHAP_PREDICTION_BATCH_ROWS=32768",
        "TARGET_SHAP_EQUIVALENCE_RECORDS=12",
    )
    current_script = Path(__file__).resolve()
    for path in sorted(candidate_paths, key=lambda value: str(value).lower()):
        if path == current_script or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        compact = "".join(text.split())
        if not all(token in compact for token in required_tokens):
            continue
        script_sha256 = file_sha256(str(path)).lower()
        candidates_by_sha.setdefault(
            script_sha256,
            {
                "script_sha256": script_sha256,
                "script_path": str(path),
                "discovery": "validated_legacy_script",
            },
        )

    candidates: List[Dict[str, str]] = []
    for script_sha256, metadata in sorted(candidates_by_sha.items()):
        row = dict(metadata)
        row["method_digest"] = legacy_credal_method_digest(
            schema, script_sha256=script_sha256
        )
        candidates.append(row)
    return candidates


def credal_execution_digest() -> str:
    """Record scheduling choices without making statistics incompatible."""
    return stable_json_digest(
        {
            "execution_contract_version": EXECUTION_CONTRACT_VERSION,
            "workflow_script_sha256": file_sha256(
                str(Path(__file__).resolve())
            ),
            "shadow_parallel_workers": int(SHADOW_PARALLEL_WORKERS),
            "shadow_max_pending_tasks": int(
                SHADOW_MAX_PENDING_TASKS
            ),
            "pfi_batch_prediction": bool(
                ENABLE_PFI_BATCH_PREDICTION
            ),
            "pfi_batch_prediction_rows": int(
                PFI_BATCH_PREDICTION_ROWS
            ),
            "loco_cache": bool(
                ENABLE_LOCO_REDUCED_PROBABILITY_CACHE
            ),
            "loco_cache_workers": int(
                LOCO_CACHE_PARALLEL_WORKERS
            ),
            "target_shap_prediction_batch_rows": int(
                TARGET_SHAP_PREDICTION_BATCH_ROWS
            ),
            "structural_parallel_workers": int(
                STRUCTURAL_PARALLEL_WORKERS
            ),
            "gc_at_shadow_task_end": bool(
                SHADOW_WORKER_GC_AT_TASK_END
            ),
        }
    )


def run_original_crossfit_unit(
    X_outer_train: pd.DataFrame,
    y_outer_train: np.ndarray,
    schema: FeatureSchema,
    part: OuterPartition,
    model_name: str,
    class_order: Sequence[int],
    worker_root: Path,
    loco_reduced_cache: Optional[
        Mapping[str, np.ndarray]
    ] = None,
) -> Dict[str, Any]:
    """Construct training-only cross-fitted source statistics for one model."""
    folds = make_inner_fold_specs(y_outer_train, part)
    fold_source_tables: Dict[str, List[pd.DataFrame]] = {
        "SHAP": [],
        "PFI": [],
        "LOCO": [],
    }
    firm_payloads: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for fold in folds:
        X_subtrain = X_outer_train.iloc[fold.subtrain_idx].reset_index(drop=True)
        X_validation = X_outer_train.iloc[fold.validation_idx].reset_index(drop=True)
        y_subtrain = np.asarray(y_outer_train, dtype=int)[fold.subtrain_idx]
        y_validation = np.asarray(y_outer_train, dtype=int)[fold.validation_idx]
        fit_seed = stable_seed_from_key(
            "fixed_spec_inner_fit",
            part.partition_id,
            fold.fold_id,
            model_name,
        )
        source_seed = stable_seed_from_key(
            "fixed_source_protocol",
            part.partition_id,
            fold.fold_id,
            model_name,
        )
        try:
            estimator, encoded, proba, _ = fit_fixed_predictive_specification(
                X_train_raw=X_subtrain,
                X_eval_raw=X_validation,
                y_train=y_subtrain,
                schema=schema,
                model_name=model_name,
                fit_seed=fit_seed,
                class_order=class_order,
            )
        except Exception as exc:
            errors.append(
                {
                    "inner_fold": int(fold.fold_id),
                    "source": "ALL",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        source_tables, source_errors = compute_fitted_source_tables(
            estimator=estimator,
            encoded=encoded,
            proba=proba,
            y_train=y_subtrain,
            y_eval=y_validation,
            X_train_raw=X_subtrain,
            X_eval_raw=X_validation,
            schema=schema,
            class_order=class_order,
            model_name=model_name,
            source_seed=source_seed,
            feature_cols=schema.feature_cols,
            worker_root=worker_root,
            call_key=(
                f"orig_o{int(part.partition_id):03d}_"
                f"i{int(fold.fold_id):02d}_m{model_name}"
            ),
            include_firm_table=False,
            shap_phase_scope="inner_calibration",
            outer_partition=int(part.partition_id),
            inner_fold=int(fold.fold_id),
            loco_reduced_proba_by_feature={
                str(feature): np.asarray(
                    loco_reduced_cache[
                        loco_reduced_cache_key(
                            str(feature), int(fold.fold_id)
                        )
                    ],
                    dtype=float,
                )
                for feature in schema.feature_cols
                if loco_reduced_cache is not None
                and loco_reduced_cache_key(
                    str(feature), int(fold.fold_id)
                )
                in loco_reduced_cache
            },
        )
        for source in ["SHAP", "PFI", "LOCO"]:
            table = source_tables.get(source, pd.DataFrame())
            if not table.empty:
                fold_source_tables[source].append(
                    _annotate_evidence(
                        table=table,
                        part=part,
                        model_name=model_name,
                        role="outer_training_original_crossfit",
                        inner_fold=fold.fold_id,
                    )
                )
        for source, message in source_errors.items():
            errors.append(
                {
                    "inner_fold": int(fold.fold_id),
                    "source": str(source),
                    "error": str(message),
                }
            )
        firm_payloads.append(
            {
                "X_eval_raw": X_validation,
                "y_eval": y_validation,
                "proba": proba,
                "inner_fold": int(fold.fold_id),
            }
        )

    aggregate_parts = []
    for source in ["SHAP", "PFI", "LOCO"]:
        pooled = aggregate_crossfitted_source_statistics(
            source=source,
            fold_tables=fold_source_tables[source],
            feature_cols=schema.feature_cols,
            class_order=class_order,
            model_name=model_name,
            outer_partition=part.partition_id,
        )
        if not pooled.empty:
            aggregate_parts.append(pooled)
    try:
        firm_pooled = aggregate_crossfitted_firm_statistics(
            oof_payloads=firm_payloads,
            schema=schema,
            feature_cols=schema.feature_cols,
            class_order=class_order,
            model_name=model_name,
            outer_partition=part.partition_id,
        )
        if not firm_pooled.empty:
            aggregate_parts.append(firm_pooled)
    except Exception as exc:
        errors.append(
            {
                "inner_fold": 0,
                "source": "FIRM",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    statistics = concat_or_empty(aggregate_parts)
    if not statistics.empty:
        statistics["outer_repeat"] = int(part.repeat_id)
        statistics["outer_fold"] = int(part.fold_id)
        statistics["split_seed"] = int(part.split_seed)
        statistics["evidence_role"] = "outer_training_original_crossfit"
        statistics["shadow_replicate"] = 0
    result = {
        "statistics": statistics,
        "errors": pd.DataFrame(errors),
        "n_inner_folds_expected": int(len(folds)),
    }
    if SHADOW_WORKER_GC_AT_TASK_END:
        gc.collect()
    return result


def loco_reduced_cache_key(feature: str, inner_fold: int) -> str:
    return f"fold={int(inner_fold):02d}|feature={str(feature)}"


def _loco_cache_payload_complete(
    payload: Mapping[str, Any],
    feature: str,
    inner_fold: int,
    expected_n_eval: int,
    n_classes: int,
) -> bool:
    if str(payload.get("feature", "")) != str(feature):
        return False
    if int(payload.get("inner_fold", -1)) != int(inner_fold):
        return False
    probability = np.asarray(
        payload.get("reduced_proba", np.asarray([])), dtype=float
    )
    if probability.shape != (int(expected_n_eval), int(n_classes)):
        return False
    if not np.isfinite(probability).all():
        return False
    if not bool(payload.get("reduced_data_equivalence_passed", False)):
        return False
    return True


_LOCO_CACHE_WORKER_CONTEXT: Optional[Dict[str, Any]] = None


def _initialize_loco_cache_worker(context: Dict[str, Any]) -> None:
    global _LOCO_CACHE_WORKER_CONTEXT
    _LOCO_CACHE_WORKER_CONTEXT = context
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _run_loco_cache_worker_task(
    task: Tuple[str, int, int, str],
) -> Dict[str, Any]:
    """Fit and checkpoint one reusable inner-fold reduced LOCO model."""
    if _LOCO_CACHE_WORKER_CONTEXT is None:
        raise RuntimeError("LOCO cache worker context was not initialized.")
    feature, feature_index, inner_fold, fingerprint = task
    context = _LOCO_CACHE_WORKER_CONTEXT
    part: OuterPartition = context["part"]
    model_name = str(context["model_name"])
    checkpoint_root = Path(context["checkpoint_root"])
    try:
        folds = make_inner_fold_specs(
            context["y_outer_train"], part
        )
        fold_matches = [
            fold for fold in folds
            if int(fold.fold_id) == int(inner_fold)
        ]
        if len(fold_matches) != 1:
            raise RuntimeError(
                f"LOCO cache fold lookup failed: {inner_fold}."
            )
        fold = fold_matches[0]
        checkpoint = load_credal_unit_checkpoint(
            checkpoint_root=checkpoint_root,
            kind="loco_reduced_probability",
            part=part,
            model_name=model_name,
            feature=str(feature),
            replicate=int(inner_fold),
            expected_fingerprint=str(fingerprint),
        )
        if checkpoint is not None and _loco_cache_payload_complete(
            checkpoint,
            feature=str(feature),
            inner_fold=int(inner_fold),
            expected_n_eval=len(fold.validation_idx),
            n_classes=len(context["class_order"]),
        ):
            return {
                "status": "checkpoint",
                "feature": str(feature),
                "feature_index": int(feature_index),
                "inner_fold": int(inner_fold),
                "payload": checkpoint,
            }

        X_outer_train = context["X_outer_train"]
        y_outer_train = np.asarray(
            context["y_outer_train"], dtype=int
        )
        schema: FeatureSchema = context["schema"]
        X_subtrain = X_outer_train.iloc[
            fold.subtrain_idx
        ].reset_index(drop=True)
        X_validation = X_outer_train.iloc[
            fold.validation_idx
        ].reset_index(drop=True)
        y_subtrain = y_outer_train[fold.subtrain_idx]
        reduced_features = [
            column
            for column in schema.feature_cols
            if str(column) != str(feature)
        ]

        # A deterministic shadow realization verifies the algebraic reuse
        # condition at the actual raw-data boundary used by the workflow.
        audit_shadow_seed = stable_seed_from_key(
            "raw_marginal_shadow",
            part.partition_id,
            fold.fold_id,
            feature,
            1,
        )
        X_shadow_train, X_shadow_validation = (
            replace_raw_feature_from_training_marginal(
                X_subtrain=X_subtrain,
                X_validation=X_validation,
                feature=str(feature),
                seed=int(audit_shadow_seed),
            )
        )
        original_train_digest = dataframe_digest(
            X_subtrain[reduced_features]
        )
        shadow_train_digest = dataframe_digest(
            X_shadow_train[reduced_features]
        )
        original_validation_digest = dataframe_digest(
            X_validation[reduced_features]
        )
        shadow_validation_digest = dataframe_digest(
            X_shadow_validation[reduced_features]
        )
        reduced_data_equal = bool(
            original_train_digest == shadow_train_digest
            and original_validation_digest
            == shadow_validation_digest
        )
        if not reduced_data_equal:
            raise RuntimeError(
                "LOCO cache reuse condition failed after deleting the "
                f"shadowed feature {feature}."
            )

        source_seed = stable_seed_from_key(
            "fixed_source_protocol",
            part.partition_id,
            fold.fold_id,
            model_name,
        )
        sample_weight = (
            make_balanced_sample_weight(y_subtrain)
            if USE_BALANCED_SAMPLE_WEIGHT
            else None
        )
        reduced_proba = fit_loco_reduced_probability(
            y_train=y_subtrain,
            X_train_raw=X_subtrain,
            X_eval_raw=X_validation,
            schema=schema,
            classes=context["class_order"],
            model_name=model_name,
            feature=str(feature),
            seed=int(source_seed),
            n_jobs=MODEL_N_JOBS,
            sample_weight=sample_weight,
        )
        payload = {
            "feature": str(feature),
            "feature_index": int(feature_index),
            "inner_fold": int(inner_fold),
            "source_seed": int(source_seed),
            "n_train": int(len(y_subtrain)),
            "n_eval": int(len(fold.validation_idx)),
            "validation_index_sha256": array_digest(
                np.asarray(fold.validation_idx, dtype=np.int64)
            ),
            "reduced_feature_order": reduced_features,
            "reduced_feature_order_sha256": stable_json_digest(
                reduced_features
            ),
            "reduced_proba": np.asarray(
                reduced_proba, dtype=float
            ),
            "reduced_proba_sha256": array_digest(
                np.asarray(reduced_proba, dtype=float)
            ),
            "original_reduced_train_sha256": (
                original_train_digest
            ),
            "shadow_reduced_train_sha256": shadow_train_digest,
            "original_reduced_validation_sha256": (
                original_validation_digest
            ),
            "shadow_reduced_validation_sha256": (
                shadow_validation_digest
            ),
            "reduced_data_equivalence_passed": reduced_data_equal,
            "cache_contract_version": (
                LOCO_CACHE_CONTRACT_VERSION
            ),
        }
        if not _loco_cache_payload_complete(
            payload,
            feature=str(feature),
            inner_fold=int(inner_fold),
            expected_n_eval=len(fold.validation_idx),
            n_classes=len(context["class_order"]),
        ):
            raise RuntimeError("LOCO cache payload failed validation.")
        save_credal_unit_checkpoint(
            checkpoint_root=checkpoint_root,
            kind="loco_reduced_probability",
            part=part,
            model_name=model_name,
            feature=str(feature),
            replicate=int(inner_fold),
            fingerprint=str(fingerprint),
            payload=payload,
            marker_metadata={
                "cache_contract_version": (
                    LOCO_CACHE_CONTRACT_VERSION
                ),
                "reduced_data_equivalence_passed": True,
            },
        )
        return {
            "status": "completed",
            "feature": str(feature),
            "feature_index": int(feature_index),
            "inner_fold": int(inner_fold),
            "payload": payload,
        }
    except BaseException as exc:
        return {
            "status": "fatal_error",
            "feature": str(feature),
            "feature_index": int(feature_index),
            "inner_fold": int(inner_fold),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    finally:
        if SHADOW_WORKER_GC_AT_TASK_END:
            gc.collect()


def prepare_loco_reduced_probability_cache(
    X_outer_train: pd.DataFrame,
    y_outer_train: np.ndarray,
    schema: FeatureSchema,
    part: OuterPartition,
    model_name: str,
    class_order: Sequence[int],
    checkpoint_root: Path,
    data_digest: str,
    method_digest: str,
    features_to_prepare: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Build only the reduced-probability entries required by pending units."""
    if not ENABLE_LOCO_REDUCED_PROBABILITY_CACHE:
        return {}
    folds = make_inner_fold_specs(y_outer_train, part)
    requested_features = list(
        dict.fromkeys(
            str(feature)
            for feature in (
                schema.feature_cols
                if features_to_prepare is None
                else features_to_prepare
            )
        )
    )
    unknown_features = sorted(
        set(requested_features).difference(schema.feature_cols)
    )
    if unknown_features:
        raise ValueError(
            "LOCO cache requested features outside the schema: "
            f"{unknown_features}."
        )
    if not requested_features:
        return {}
    cache: Dict[str, np.ndarray] = {}
    audit_rows: List[Dict[str, object]] = []
    pending_tasks: List[Tuple[str, int, int, str]] = []
    cache_method_digest = stable_json_digest(
        {
            "statistical_method_digest": str(method_digest),
            "loco_cache_contract": LOCO_CACHE_CONTRACT_VERSION,
            "model": str(model_name),
            "sample_weight": bool(USE_BALANCED_SAMPLE_WEIGHT),
            "model_n_jobs": int(MODEL_N_JOBS),
        }
    )
    fold_by_id = {int(fold.fold_id): fold for fold in folds}
    feature_index_lookup = {
        str(feature): int(index)
        for index, feature in enumerate(schema.feature_cols)
    }
    for feature in requested_features:
        feature_index = feature_index_lookup[str(feature)]
        for fold in folds:
            fingerprint = credal_unit_fingerprint(
                kind="loco_reduced_probability",
                part=part,
                model_name=model_name,
                feature=str(feature),
                replicate=int(fold.fold_id),
                data_digest=data_digest,
                method_digest=cache_method_digest,
            )
            checkpoint = load_credal_unit_checkpoint(
                checkpoint_root=checkpoint_root,
                kind="loco_reduced_probability",
                part=part,
                model_name=model_name,
                feature=str(feature),
                replicate=int(fold.fold_id),
                expected_fingerprint=fingerprint,
            )
            if checkpoint is not None:
                if not _loco_cache_payload_complete(
                    checkpoint,
                    feature=str(feature),
                    inner_fold=int(fold.fold_id),
                    expected_n_eval=len(fold.validation_idx),
                    n_classes=len(class_order),
                ):
                    raise RuntimeError(
                        "Invalid LOCO reduced-probability checkpoint: "
                        f"outer={part.partition_id}, model={model_name}, "
                        f"feature={feature}, fold={fold.fold_id}."
                    )
                key = loco_reduced_cache_key(
                    str(feature), int(fold.fold_id)
                )
                cache[key] = np.asarray(
                    checkpoint["reduced_proba"], dtype=float
                )
                audit_rows.append(
                    {
                        "outer_partition": int(part.partition_id),
                        "model": str(model_name),
                        "feature": str(feature),
                        "inner_fold": int(fold.fold_id),
                        "status": "checkpoint",
                        "n_eval": int(len(fold.validation_idx)),
                        "source_seed": int(
                            checkpoint["source_seed"]
                        ),
                        "reduced_proba_sha256": str(
                            checkpoint["reduced_proba_sha256"]
                        ),
                        "reduced_data_equivalence_passed": 1,
                    }
                )
            else:
                pending_tasks.append(
                    (
                        str(feature),
                        int(feature_index),
                        int(fold.fold_id),
                        str(fingerprint),
                    )
                )
    cprint(
        "[LOCO CACHE] "
        f"outer={int(part.partition_id)} model={model_name} "
        f"features={len(requested_features)} "
        f"total={len(requested_features) * len(folds)} "
        f"restored={len(cache)} pending={len(pending_tasks)} "
        f"workers={int(LOCO_CACHE_PARALLEL_WORKERS)}"
    )

    if pending_tasks:
        context: Dict[str, Any] = {
            "X_outer_train": X_outer_train,
            "y_outer_train": np.asarray(y_outer_train, dtype=int),
            "schema": schema,
            "part": part,
            "model_name": str(model_name),
            "class_order": list(class_order),
            "checkpoint_root": str(checkpoint_root),
        }
        max_workers = max(
            1,
            min(
                int(LOCO_CACHE_PARALLEL_WORKERS),
                len(pending_tasks),
            ),
        )
        max_pending = max(
            max_workers, int(LOCO_CACHE_MAX_PENDING_TASKS)
        )
        task_iterator = iter(pending_tasks)
        process_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=process_context,
            initializer=_initialize_loco_cache_worker,
            initargs=(context,),
        ) as executor:
            future_to_task = {}
            for _ in range(min(max_pending, len(pending_tasks))):
                task = next(task_iterator, None)
                if task is None:
                    break
                future_to_task[
                    executor.submit(_run_loco_cache_worker_task, task)
                ] = task
            completed = 0
            while future_to_task:
                done, _ = wait(
                    future_to_task, return_when=FIRST_COMPLETED
                )
                for future in done:
                    task = future_to_task.pop(future)
                    result = future.result()
                    if result.get("status") == "fatal_error":
                        raise RuntimeError(
                            "LOCO cache worker failed: "
                            f"outer={part.partition_id}, "
                            f"model={model_name}, feature={task[0]}, "
                            f"fold={task[2]}. {result.get('error')}\n"
                            f"{result.get('traceback', '')}"
                        )
                    payload = result.get("payload")
                    if not isinstance(payload, Mapping):
                        raise RuntimeError(
                            "LOCO cache worker returned no payload."
                        )
                    feature = str(result["feature"])
                    inner_fold = int(result["inner_fold"])
                    fold = fold_by_id[inner_fold]
                    if not _loco_cache_payload_complete(
                        payload,
                        feature=feature,
                        inner_fold=inner_fold,
                        expected_n_eval=len(fold.validation_idx),
                        n_classes=len(class_order),
                    ):
                        raise RuntimeError(
                            "LOCO cache worker returned an incomplete "
                            f"payload for {feature}, fold {inner_fold}."
                        )
                    key = loco_reduced_cache_key(
                        feature, inner_fold
                    )
                    cache[key] = np.asarray(
                        payload["reduced_proba"], dtype=float
                    )
                    audit_rows.append(
                        {
                            "outer_partition": int(
                                part.partition_id
                            ),
                            "model": str(model_name),
                            "feature": feature,
                            "inner_fold": inner_fold,
                            "status": str(result.get("status")),
                            "n_eval": int(len(fold.validation_idx)),
                            "source_seed": int(
                                payload["source_seed"]
                            ),
                            "reduced_proba_sha256": str(
                                payload[
                                    "reduced_proba_sha256"
                                ]
                            ),
                            "reduced_data_equivalence_passed": 1,
                        }
                    )
                    completed += 1
                    cprint(
                        "[LOCO CACHE DONE] "
                        f"outer={part.partition_id} model={model_name} "
                        f"feature={feature} fold={inner_fold} "
                        f"progress={len(cache)}/"
                        f"{len(requested_features) * len(folds)}"
                    )
                    next_task = next(task_iterator, None)
                    if next_task is not None:
                        future_to_task[
                            executor.submit(
                                _run_loco_cache_worker_task,
                                next_task,
                            )
                        ] = next_task

    expected_keys = {
        loco_reduced_cache_key(
            str(feature), int(fold.fold_id)
        )
        for feature in requested_features
        for fold in folds
    }
    if set(cache) != expected_keys:
        missing = sorted(expected_keys.difference(cache))
        raise RuntimeError(
            "LOCO reduced-probability cache is incomplete. "
            f"Missing keys: {missing[:10]}"
        )
    audit_dir = Path(checkpoint_root).parent / "loco_cache_audit"
    audit_path = audit_dir / (
        f"p{int(part.partition_id):03d}_"
        f"m{_safe_path_component(model_name, limit=20)}.csv"
    )
    audit = pd.DataFrame(audit_rows).sort_values(
        ["feature", "inner_fold"], kind="mergesort"
    )
    atomic_dataframe_to_csv(audit, audit_path)
    return cache


def run_shadow_crossfit_unit(
    X_outer_train: pd.DataFrame,
    y_outer_train: np.ndarray,
    schema: FeatureSchema,
    part: OuterPartition,
    model_name: str,
    class_order: Sequence[int],
    feature: str,
    feature_index: int,
    shadow_replicate: int,
    worker_root: Path,
    loco_reduced_cache: Optional[
        Mapping[str, np.ndarray]
    ] = None,
) -> Dict[str, Any]:
    """Produce one statistic per source/class for one shadow realization."""
    folds = make_inner_fold_specs(y_outer_train, part)
    fold_source_tables: Dict[str, List[pd.DataFrame]] = {
        "SHAP": [],
        "PFI": [],
        "LOCO": [],
    }
    firm_payloads: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for fold in folds:
        X_subtrain = X_outer_train.iloc[fold.subtrain_idx].reset_index(drop=True)
        X_validation = X_outer_train.iloc[fold.validation_idx].reset_index(drop=True)
        y_subtrain = np.asarray(y_outer_train, dtype=int)[fold.subtrain_idx]
        y_validation = np.asarray(y_outer_train, dtype=int)[fold.validation_idx]
        shadow_seed = stable_seed_from_key(
            "raw_marginal_shadow",
            part.partition_id,
            fold.fold_id,
            feature,
            shadow_replicate,
        )
        X_shadow_train, X_shadow_validation = (
            replace_raw_feature_from_training_marginal(
                X_subtrain=X_subtrain,
                X_validation=X_validation,
                feature=feature,
                seed=shadow_seed,
            )
        )
        # The fit seed deliberately excludes feature and shadow replicate.
        # Model stochasticity is paired across the original and shadow fits.
        fit_seed = stable_seed_from_key(
            "fixed_spec_inner_fit",
            part.partition_id,
            fold.fold_id,
            model_name,
        )
        source_seed = stable_seed_from_key(
            "fixed_source_protocol",
            part.partition_id,
            fold.fold_id,
            model_name,
        )
        try:
            estimator, encoded, proba, _ = fit_fixed_predictive_specification(
                X_train_raw=X_shadow_train,
                X_eval_raw=X_shadow_validation,
                y_train=y_subtrain,
                schema=schema,
                model_name=model_name,
                fit_seed=fit_seed,
                class_order=class_order,
            )
        except Exception as exc:
            errors.append(
                {
                    "inner_fold": int(fold.fold_id),
                    "source": "ALL",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        source_tables, source_errors = compute_fitted_source_tables(
            estimator=estimator,
            encoded=encoded,
            proba=proba,
            y_train=y_subtrain,
            y_eval=y_validation,
            X_train_raw=X_shadow_train,
            X_eval_raw=X_shadow_validation,
            schema=schema,
            class_order=class_order,
            model_name=model_name,
            source_seed=source_seed,
            feature_cols=[feature],
            worker_root=worker_root,
            call_key=(
                f"sh_o{int(part.partition_id):03d}_"
                f"i{int(fold.fold_id):02d}_m{model_name}_"
                f"j{int(feature_index):02d}_b{int(shadow_replicate):03d}"
            ),
            include_firm_table=False,
            shap_phase_scope="inner_calibration",
            outer_partition=int(part.partition_id),
            inner_fold=int(fold.fold_id),
            loco_reduced_proba_by_feature={
                str(feature): np.asarray(
                    loco_reduced_cache[
                        loco_reduced_cache_key(
                            str(feature), int(fold.fold_id)
                        )
                    ],
                    dtype=float,
                )
            }
            if loco_reduced_cache is not None
            and loco_reduced_cache_key(
                str(feature), int(fold.fold_id)
            )
            in loco_reduced_cache
            else None,
        )
        for source in ["SHAP", "PFI", "LOCO"]:
            table = source_tables.get(source, pd.DataFrame())
            if not table.empty:
                fold_source_tables[source].append(
                    _annotate_evidence(
                        table=table,
                        part=part,
                        model_name=model_name,
                        role="shadow_null_inner_crossfit",
                        inner_fold=fold.fold_id,
                        shadow_feature=feature,
                        shadow_replicate=shadow_replicate,
                    )
                )
        for source, message in source_errors.items():
            errors.append(
                {
                    "inner_fold": int(fold.fold_id),
                    "source": str(source),
                    "error": str(message),
                }
            )
        firm_payloads.append(
            {
                "X_eval_raw": X_shadow_validation,
                "y_eval": y_validation,
                "proba": proba,
                "inner_fold": int(fold.fold_id),
            }
        )
    aggregate_parts = []
    for source in ["SHAP", "PFI", "LOCO"]:
        pooled = aggregate_crossfitted_source_statistics(
            source=source,
            fold_tables=fold_source_tables[source],
            feature_cols=[feature],
            class_order=class_order,
            model_name=model_name,
            outer_partition=part.partition_id,
        )
        if not pooled.empty:
            aggregate_parts.append(pooled)
    try:
        firm_pooled = aggregate_crossfitted_firm_statistics(
            oof_payloads=firm_payloads,
            schema=schema,
            feature_cols=[feature],
            class_order=class_order,
            model_name=model_name,
            outer_partition=part.partition_id,
        )
        if not firm_pooled.empty:
            aggregate_parts.append(firm_pooled)
    except Exception as exc:
        errors.append(
            {
                "inner_fold": 0,
                "source": "FIRM",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    statistics = concat_or_empty(aggregate_parts)
    if not statistics.empty:
        statistics["outer_repeat"] = int(part.repeat_id)
        statistics["outer_fold"] = int(part.fold_id)
        statistics["split_seed"] = int(part.split_seed)
        statistics["evidence_role"] = "shadow_null_inner_crossfit"
        statistics["shadow_feature"] = str(feature)
        statistics["shadow_replicate"] = int(shadow_replicate)
    result = {
        "statistics": statistics,
        "errors": pd.DataFrame(errors),
        "n_inner_folds_expected": int(len(folds)),
    }
    if SHADOW_WORKER_GC_AT_TASK_END:
        gc.collect()
    return result


def class_balanced_one_vs_rest_brier(
    y_true: np.ndarray,
    class_probability: np.ndarray,
    class_id: int,
) -> float:
    """Give the focal class and its complement equal evaluation mass."""
    labels = np.asarray(y_true, dtype=int)
    probability = np.asarray(class_probability, dtype=float)
    target = (labels == int(class_id)).astype(float)
    focal = labels == int(class_id)
    other = ~focal
    if not np.any(focal) or not np.any(other):
        return np.nan
    squared_error = (target - probability) ** 2
    return float(
        0.5 * np.mean(squared_error[focal])
        + 0.5 * np.mean(squared_error[other])
    )


def run_outer_actual_unit(
    X_outer_train: pd.DataFrame,
    X_outer_holdout: pd.DataFrame,
    y_outer_train: np.ndarray,
    y_outer_holdout: np.ndarray,
    schema: FeatureSchema,
    part: OuterPartition,
    model_name: str,
    class_order: Sequence[int],
    worker_root: Path,
) -> Dict[str, Any]:
    """Fit one admissible specification and explain the untouched holdout."""
    fit_seed = stable_seed_from_key(
        "fixed_spec_outer_fit", part.partition_id, model_name
    )
    source_seed = stable_seed_from_key(
        "fixed_source_protocol_outer", part.partition_id, model_name
    )
    errors: List[Dict[str, Any]] = []
    try:
        estimator, encoded, proba, pred = fit_fixed_predictive_specification(
            X_train_raw=X_outer_train,
            X_eval_raw=X_outer_holdout,
            y_train=y_outer_train,
            schema=schema,
            model_name=model_name,
            fit_seed=fit_seed,
            class_order=class_order,
        )
    except Exception as exc:
        return {
            "statistics": pd.DataFrame(),
            "shap_full_display": pd.DataFrame(),
            "shap_display_target_concordance": pd.DataFrame(),
            "metrics": pd.DataFrame(),
            "predictions": pd.DataFrame(),
            "calibration": pd.DataFrame(),
            "pdp": pd.DataFrame(),
            "ice": pd.DataFrame(),
            "errors": pd.DataFrame(
                [
                    {
                        "outer_partition": int(part.partition_id),
                        "model": str(model_name),
                        "source": "ALL",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ]
            ),
        }

    if ENABLE_TARGET_SHAP_EQUIVALENCE_AUDIT:
        run_target_shap_equivalence_audit(
            estimator=estimator,
            preprocessor=encoded.preprocessor,
            X_train_raw=X_outer_train,
            X_eval_raw=X_outer_holdout,
            schema=schema,
            classes=class_order,
            model_name=model_name,
            outer_partition=int(part.partition_id),
            audit_root=Path(worker_root).parent / "target_shap_audit",
        )

    source_tables, source_errors = compute_fitted_source_tables(
        estimator=estimator,
        encoded=encoded,
        proba=proba,
        y_train=y_outer_train,
        y_eval=y_outer_holdout,
        X_train_raw=X_outer_train,
        X_eval_raw=X_outer_holdout,
        schema=schema,
        class_order=class_order,
        model_name=model_name,
        source_seed=source_seed,
        feature_cols=schema.feature_cols,
        worker_root=worker_root,
        call_key=f"outer_o{int(part.partition_id):03d}_m{model_name}",
        include_firm_table=True,
        shap_phase_scope="outer_actual",
        outer_partition=int(part.partition_id),
        inner_fold=-1,
    )
    shap_full_display = pd.DataFrame()
    shap_concordance = pd.DataFrame()
    try:
        display_seed = _target_shap_background_seed(
            "outer_actual",
            int(part.partition_id),
            -1,
        )
        shap_full_display = compute_shap_importance_safe(
            estimator=estimator,
            preprocessor=encoded.preprocessor,
            X_train_raw=X_outer_train.reset_index(drop=True),
            y_train=np.asarray(y_outer_train, dtype=int),
            X_eval_raw=X_outer_holdout.reset_index(drop=True),
            y_eval=np.asarray(y_outer_holdout, dtype=int),
            schema=schema,
            classes=class_order,
            model_name=model_name,
            seed=int(display_seed),
            background_size=int(SHAP_BACKGROUND_SIZE),
            eval_size=None,
            max_evals=(
                int(SHAP_PERMUTATION_CYCLES)
                * (2 * len(schema.feature_cols) + 1)
            ),
            reference_proba=np.asarray(proba, dtype=float),
            work_root=worker_root,
            call_key=(
                f"full_display_o{int(part.partition_id):03d}_m{model_name}"
            ),
        )
        if not shap_full_display.empty:
            shap_full_display = _annotate_evidence(
                table=shap_full_display,
                part=part,
                model_name=model_name,
                role="shap_full_display",
            )
        target_table = source_tables.get("SHAP", pd.DataFrame())
        if not shap_full_display.empty and not target_table.empty:
            merged = shap_full_display[
                [
                    "class_id",
                    "feature",
                    "balanced_mean_abs_shap",
                    "balanced_mean_signed_shap",
                ]
            ].merge(
                target_table[
                    [
                        "class_id",
                        "feature",
                        "balanced_mean_abs_shap",
                        "balanced_mean_signed_shap",
                    ]
                ],
                on=["class_id", "feature"],
                how="inner",
                suffixes=("_official_full", "_target"),
                validate="one_to_one",
            )
            concordance_rows = []
            epsilon = np.finfo(float).eps
            for class_id, group in merged.groupby("class_id", sort=True):
                official = group["balanced_mean_abs_shap_official_full"].astype(
                    float
                )
                target = group["balanced_mean_abs_shap_target"].astype(float)
                concordance_rows.append(
                    {
                        "outer_partition": int(part.partition_id),
                        "outer_repeat": int(part.repeat_id),
                        "outer_fold": int(part.fold_id),
                        "model": str(model_name),
                        "class_id": int(class_id),
                        "n_features": int(len(group)),
                        "spearman": float(
                            official.corr(target, method="spearman")
                        ),
                        "kendall": float(
                            official.corr(target, method="kendall")
                        ),
                        "normalized_l1": float(
                            np.sum(np.abs(official.to_numpy() - target.to_numpy()))
                            / max(np.sum(np.abs(official.to_numpy())), epsilon)
                        ),
                        "max_abs_difference": float(
                            np.max(
                                np.abs(
                                    official.to_numpy() - target.to_numpy()
                                )
                            )
                        ),
                        "diagnostic_role": (
                            "descriptive_official_full_vs_target_coordinate"
                        ),
                    }
                )
            shap_concordance = pd.DataFrame(concordance_rows)
    except Exception as exc:
        errors.append(
            {
                "outer_partition": int(part.partition_id),
                "model": str(model_name),
                "source": "SHAP_FULL_DISPLAY",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    statistic_parts = []
    for source in CREDAL_SOURCE_NAMES:
        table = source_tables.get(str(source), pd.DataFrame())
        if not table.empty:
            try:
                statistic_parts.append(
                    standardize_outer_source_table(
                        source=source,
                        table=table,
                        part=part,
                        model_name=model_name,
                    )
                )
            except Exception as exc:
                source_errors[str(source)] = f"{type(exc).__name__}: {exc}"
    for source, message in source_errors.items():
        errors.append(
            {
                "outer_partition": int(part.partition_id),
                "model": str(model_name),
                "source": str(source),
                "error": str(message),
            }
        )

    metrics = evaluate_predictions(
        np.asarray(y_outer_holdout, dtype=int),
        np.asarray(pred, dtype=int),
        labels=class_order,
    )
    metrics.update(
        compute_probability_diagnostics(
            y_true=np.asarray(y_outer_holdout, dtype=int),
            proba=np.asarray(proba, dtype=float),
            labels=class_order,
            n_bins=CALIBRATION_N_BINS,
        )
    )
    metrics_row = {
        "outer_partition": int(part.partition_id),
        "outer_repeat": int(part.repeat_id),
        "outer_fold": int(part.fold_id),
        "split_seed": int(part.split_seed),
        "model": str(model_name),
        "model_display": MODEL_DISPLAY_NAMES.get(str(model_name), str(model_name)),
        "n_train": int(len(y_outer_train)),
        "n_holdout": int(len(y_outer_holdout)),
        **metrics,
    }

    prediction_rows = []
    for row_id, true_label in enumerate(np.asarray(y_outer_holdout, dtype=int)):
        row = {
            "outer_partition": int(part.partition_id),
            "outer_repeat": int(part.repeat_id),
            "outer_fold": int(part.fold_id),
            "model": str(model_name),
            "holdout_position": int(row_id),
            "stable_record_id": int(
                X_outer_holdout.iloc[int(row_id)][STABLE_RECORD_ID_COL]
            ),
            "y_true": int(true_label),
            "y_pred": int(pred[row_id]),
        }
        for class_pos, class_id in enumerate(class_order):
            row[f"proba_class_{int(class_id)}"] = float(proba[row_id, class_pos])
        prediction_rows.append(row)

    calibration = compute_calibration_curve_rows(
        y_true=np.asarray(y_outer_holdout, dtype=int),
        proba=np.asarray(proba, dtype=float),
        labels=class_order,
        model_name=model_name,
        seed=int(part.partition_id),
        n_bins=CALIBRATION_N_BINS,
    )
    if not calibration.empty:
        calibration["outer_partition"] = int(part.partition_id)
        calibration["outer_repeat"] = int(part.repeat_id)
        calibration["outer_fold"] = int(part.fold_id)

    pdp = pd.DataFrame()
    ice = pd.DataFrame()
    if ENABLE_PDP or ENABLE_ICE:
        try:
            pdp, ice = compute_response_profiles(
                estimator=estimator,
                preprocessor=encoded.preprocessor,
                X_train_raw=X_outer_train,
                X_eval_raw=X_outer_holdout,
                classes=class_order,
                feature_cols=list(schema.feature_cols),
                model_name=model_name,
                seed=int(part.partition_id),
                top_features_by_class=None,
                top_k=len(schema.feature_cols),
                max_values=PDP_MAX_VALUES,
                min_train_count=PDP_MIN_TRAIN_COUNT,
                return_pdp=bool(ENABLE_PDP),
                return_ice=bool(ENABLE_ICE),
            )
            for frame in [pdp, ice]:
                if frame is not None and not frame.empty:
                    frame["outer_partition"] = int(part.partition_id)
                    frame["outer_repeat"] = int(part.repeat_id)
                    frame["outer_fold"] = int(part.fold_id)
        except Exception as exc:
            errors.append(
                {
                    "outer_partition": int(part.partition_id),
                    "model": str(model_name),
                    "source": "PDP_ICE",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    clear_cuda_cache()
    gc.collect()
    return {
        "statistics": concat_or_empty(statistic_parts),
        "shap_full_display": shap_full_display,
        "shap_display_target_concordance": shap_concordance,
        "metrics": pd.DataFrame([metrics_row]),
        "predictions": pd.DataFrame(prediction_rows),
        "calibration": calibration,
        "pdp": pdp,
        "ice": ice,
        "errors": pd.DataFrame(errors),
    }


def credal_core_payload_complete(
    payload: Mapping[str, Any],
    feature_cols: Sequence[str],
    class_order: Sequence[int],
    expected_inner_folds: Optional[int] = None,
) -> bool:
    statistics = payload.get("statistics", pd.DataFrame())
    if not isinstance(statistics, pd.DataFrame) or statistics.empty:
        return False
    for source in CREDAL_SOURCE_NAMES:
        for class_id in class_order:
            valid, _ = source_table_contract_status(
                statistics,
                source=source,
                class_id=int(class_id),
                feature_cols=feature_cols,
            )
            if not valid:
                return False
            if expected_inner_folds is not None:
                group = statistics.loc[
                    (statistics["source"].astype(str) == str(source))
                    & (
                        pd.to_numeric(statistics["class_id"], errors="coerce")
                        == int(class_id)
                    )
                ]
                folds = (
                    pd.to_numeric(group["n_inner_folds"], errors="coerce")
                    .dropna()
                    .astype(int)
                    .unique()
                    if "n_inner_folds" in group.columns
                    else np.asarray([], dtype=int)
                )
                if (
                    len(folds) != 1
                    or int(folds[0]) != int(expected_inner_folds)
                ):
                    return False
    return True


_SHADOW_WORKER_CONTEXT: Optional[Dict[str, Any]] = None


def _initialize_shadow_worker(context: Dict[str, Any]) -> None:
    """Install one immutable outer-partition/model context per worker process."""
    global _SHADOW_WORKER_CONTEXT
    _SHADOW_WORKER_CONTEXT = context
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _run_shadow_worker_task(
    task: Tuple[str, int, int, str],
) -> Dict[str, Any]:
    """Run and checkpoint one feature-replicate unit in a spawned process."""
    if _SHADOW_WORKER_CONTEXT is None:
        raise RuntimeError("Shadow worker context was not initialized.")
    feature, feature_index, shadow_replicate, fingerprint = task
    context = _SHADOW_WORKER_CONTEXT
    part: OuterPartition = context["part"]
    model_name = str(context["model_name"])
    checkpoint_root = Path(context["checkpoint_root"])
    try:
        checkpoint = (
            load_credal_unit_checkpoint(
                checkpoint_root,
                "shadow_crossfit",
                part,
                model_name,
                str(feature),
                int(shadow_replicate),
                str(fingerprint),
            )
            if ENABLE_SHADOW_UNIT_CHECKPOINTS
            else None
        )
        if checkpoint is not None:
            return {
                "status": "checkpoint",
                "feature": str(feature),
                "shadow_replicate": int(shadow_replicate),
                "payload": checkpoint,
            }
        payload = run_shadow_crossfit_unit(
            X_outer_train=context["X_outer_train"],
            y_outer_train=context["y_outer_train"],
            schema=context["schema"],
            part=part,
            model_name=model_name,
            class_order=context["class_order"],
            feature=str(feature),
            feature_index=int(feature_index),
            shadow_replicate=int(shadow_replicate),
            worker_root=Path(context["worker_root"]),
            loco_reduced_cache=context.get(
                "loco_reduced_cache"
            ),
        )
        complete = credal_core_payload_complete(
            payload,
            feature_cols=[str(feature)],
            class_order=context["class_order"],
            expected_inner_folds=int(
                payload.get("n_inner_folds_expected", 0)
            ),
        )
        if ENABLE_SHADOW_UNIT_CHECKPOINTS and complete:
            save_credal_unit_checkpoint(
                checkpoint_root,
                "shadow_crossfit",
                part,
                model_name,
                str(feature),
                int(shadow_replicate),
                str(fingerprint),
                payload,
            )
        return {
            "status": "completed" if complete else "incomplete",
            "feature": str(feature),
            "shadow_replicate": int(shadow_replicate),
            "payload": payload,
        }
    except BaseException as exc:
        return {
            "status": "fatal_error",
            "feature": str(feature),
            "shadow_replicate": int(shadow_replicate),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _append_shadow_payload(
    payload: Mapping[str, Any],
    feature: str,
    shadow_replicate: int,
    partition_id: int,
    model_name: str,
    buckets: Dict[str, List[pd.DataFrame]],
) -> None:
    shadow_frame = payload.get("statistics", pd.DataFrame())
    if isinstance(shadow_frame, pd.DataFrame) and not shadow_frame.empty:
        buckets["shadow_statistics"].append(shadow_frame)
    shadow_errors = payload.get("errors", pd.DataFrame())
    if isinstance(shadow_errors, pd.DataFrame) and not shadow_errors.empty:
        annotated_errors = shadow_errors.copy()
        annotated_errors["outer_partition"] = int(partition_id)
        annotated_errors["model"] = str(model_name)
        annotated_errors["unit"] = "shadow_crossfit"
        annotated_errors["shadow_feature"] = str(feature)
        annotated_errors["shadow_replicate"] = int(shadow_replicate)
        buckets["errors"].append(annotated_errors)


def _execute_parallel_shadow_units(
    X_outer_train: pd.DataFrame,
    y_outer_train: np.ndarray,
    schema: FeatureSchema,
    part: OuterPartition,
    model_name: str,
    class_order: Sequence[int],
    checkpoint_root: Path,
    worker_root: Path,
    data_digest: str,
    method_digest: str,
    legacy_method_candidates: Sequence[Mapping[str, str]],
    pfi_audit_passed: bool,
    n_shadow: int,
    buckets: Dict[str, List[pd.DataFrame]],
) -> None:
    """Execute missing shadow units with bounded process parallelism."""
    pending_tasks: List[Tuple[str, int, int, str]] = []
    total_units = int(len(schema.feature_cols) * int(n_shadow))
    restored_units = 0
    for feature_index, feature in enumerate(schema.feature_cols):
        for shadow_replicate in range(1, int(n_shadow) + 1):
            fingerprint = credal_unit_fingerprint(
                "shadow_crossfit",
                part,
                model_name,
                feature,
                shadow_replicate,
                data_digest,
                method_digest,
            )
            legacy_fingerprints = compatible_legacy_unit_fingerprints(
                kind="shadow_crossfit",
                part=part,
                model_name=model_name,
                feature=feature,
                replicate=shadow_replicate,
                data_digest=data_digest,
                legacy_method_candidates=legacy_method_candidates,
            )
            checkpoint = (
                load_credal_unit_checkpoint(
                    checkpoint_root,
                    "shadow_crossfit",
                    part,
                    model_name,
                    feature,
                    shadow_replicate,
                    fingerprint,
                    compatible_legacy_fingerprints=legacy_fingerprints,
                )
                if ENABLE_SHADOW_UNIT_CHECKPOINTS
                else None
            )
            if checkpoint is not None:
                complete = credal_core_payload_complete(
                    checkpoint,
                    feature_cols=[str(feature)],
                    class_order=class_order,
                    expected_inner_folds=int(
                        SHADOW_INNER_CV_FOLDS
                    ),
                )
                if not complete:
                    raise RuntimeError(
                        "Restored shadow checkpoint failed the current "
                        "four-source payload contract: "
                        f"outer={part.partition_id}, model={model_name}, "
                        f"feature={feature}, replicate={shadow_replicate}."
                    )
                if bool(
                    checkpoint.get(
                        "_checkpoint_requires_migration", False
                    )
                ):
                    if not bool(pfi_audit_passed):
                        raise RuntimeError(
                            "Legacy shadow checkpoint migration requires "
                            "a passed PFI execution-equivalence audit."
                        )
                    checkpoint = migrate_compatible_credal_checkpoint(
                        checkpoint_root=checkpoint_root,
                        kind="shadow_crossfit",
                        part=part,
                        model_name=model_name,
                        feature=str(feature),
                        replicate=int(shadow_replicate),
                        expected_fingerprint=fingerprint,
                        payload=checkpoint,
                        compatibility_audit=(
                            "payload_complete; pfi_reference_vs_batch_"
                            "equivalent; loco_cache_reuses_identical_"
                            "reduced_raw_tables"
                        ),
                    )
                _append_shadow_payload(
                    checkpoint,
                    feature,
                    shadow_replicate,
                    int(part.partition_id),
                    model_name,
                    buckets,
                )
                restored_units += 1
            else:
                pending_tasks.append(
                    (
                        str(feature),
                        int(feature_index),
                        int(shadow_replicate),
                        str(fingerprint),
                    )
                )
    cprint(
        "[CREDAL SHADOW QUEUE] "
        f"outer={int(part.partition_id)} model={model_name} "
        f"total={total_units} restored={restored_units} "
        f"pending={len(pending_tasks)} workers={int(SHADOW_PARALLEL_WORKERS)}"
    )
    if not pending_tasks:
        return

    loco_reduced_cache = prepare_loco_reduced_probability_cache(
        X_outer_train=X_outer_train,
        y_outer_train=y_outer_train,
        schema=schema,
        part=part,
        model_name=model_name,
        class_order=class_order,
        checkpoint_root=checkpoint_root,
        data_digest=data_digest,
        method_digest=method_digest,
        features_to_prepare=sorted(
            {str(task[0]) for task in pending_tasks}
        ),
    )
    context: Dict[str, Any] = {
        "X_outer_train": X_outer_train,
        "y_outer_train": np.asarray(y_outer_train, dtype=int),
        "schema": schema,
        "part": part,
        "model_name": str(model_name),
        "class_order": list(class_order),
        "checkpoint_root": str(checkpoint_root),
        "worker_root": str(worker_root),
        "loco_reduced_cache": loco_reduced_cache,
    }
    max_workers = max(1, min(int(SHADOW_PARALLEL_WORKERS), len(pending_tasks)))
    max_pending = max(max_workers, int(SHADOW_MAX_PENDING_TASKS))
    task_iterator = iter(pending_tasks)
    completed = int(restored_units)
    completed_this_run = 0
    queue_started_at = time.time()
    process_context = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=process_context,
        initializer=_initialize_shadow_worker,
        initargs=(context,),
    ) as executor:
        future_to_task = {}
        for _ in range(min(max_pending, len(pending_tasks))):
            task = next(task_iterator, None)
            if task is None:
                break
            future_to_task[executor.submit(_run_shadow_worker_task, task)] = task

        while future_to_task:
            done, _ = wait(
                future_to_task,
                return_when=FIRST_COMPLETED,
            )
            for future in done:
                task = future_to_task.pop(future)
                result = future.result()
                feature = str(result.get("feature", task[0]))
                replicate = int(result.get("shadow_replicate", task[2]))
                if result.get("status") == "fatal_error":
                    raise RuntimeError(
                        "Parallel shadow unit failed: "
                        f"outer={int(part.partition_id)}, model={model_name}, "
                        f"feature={feature}, replicate={replicate}. "
                        f"{result.get('error')}\n{result.get('traceback', '')}"
                    )
                payload = result.get("payload")
                if not isinstance(payload, Mapping):
                    raise RuntimeError(
                        "Parallel shadow unit returned no valid payload: "
                        f"feature={feature}, replicate={replicate}."
                    )
                _append_shadow_payload(
                    payload,
                    feature,
                    replicate,
                    int(part.partition_id),
                    model_name,
                    buckets,
                )
                completed += 1
                completed_this_run += 1
                elapsed_seconds = max(time.time() - queue_started_at, 0.0)
                remaining_this_run = max(
                    len(pending_tasks) - completed_this_run,
                    0,
                )
                eta_seconds = (
                    elapsed_seconds
                    * remaining_this_run
                    / max(completed_this_run, 1)
                )
                cprint(
                    "[CREDAL SHADOW DONE] "
                    f"outer={int(part.partition_id)} model={model_name} "
                    f"feature={feature} replicate={replicate}/{n_shadow} "
                    f"progress={completed}/{total_units} "
                    f"elapsed={elapsed_seconds / 3600.0:.2f}h "
                    f"queue_eta={eta_seconds / 3600.0:.2f}h"
                )
                next_task = next(task_iterator, None)
                if next_task is not None:
                    future_to_task[
                        executor.submit(_run_shadow_worker_task, next_task)
                    ] = next_task


def execute_credal_evidence_contract(
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    schema: FeatureSchema,
    partitions: Sequence[OuterPartition],
    admissible_models_by_partition: Mapping[int, Sequence[str]],
    output_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """Run and checkpoint actual, original-crossfit, and shadow evidence units."""
    checkpoint_root = Path(output_dir) / "checkpoints" / "credal_shadow_units"
    worker_root = Path(output_dir) / "checkpoints" / "shap_runtime"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    worker_root.mkdir(parents=True, exist_ok=True)
    class_order = [0, 1, 2]
    n_shadow = int(shadow_replicate_count(RUN_PROFILE))
    data_digest = stable_json_digest(
        {
            "X": dataframe_digest(X_all.reset_index(drop=True)),
            "y": array_digest(np.asarray(y_all, dtype=np.int64)),
        }
    )
    method_digest = credal_method_digest(schema)
    legacy_method_candidates = (
        discover_legacy_credal_method_candidates(schema)
    )
    cprint(
        "[LEGACY CHECKPOINT CANDIDATES] "
        f"count={len(legacy_method_candidates)}"
    )
    for candidate in legacy_method_candidates:
        cprint(
            "  "
            f"{candidate['discovery']} | "
            f"{candidate['script_sha256']} | "
            f"{candidate['script_path']}"
        )
    execution_digest = credal_execution_digest()
    atomic_json_dump(
        {
            "statistical_contract_version": (
                STATISTICAL_CONTRACT_VERSION
            ),
            "statistical_method_digest": method_digest,
            "legacy_target_shap_v1_method_digest": (
                [
                    {
                        "method_digest": str(
                            candidate["method_digest"]
                        ),
                        "script_sha256": str(
                            candidate["script_sha256"]
                        ),
                        "script_path": str(
                            candidate["script_path"]
                        ),
                        "discovery": str(
                            candidate["discovery"]
                        ),
                    }
                    for candidate in legacy_method_candidates
                ]
            ),
            "execution_contract_version": (
                EXECUTION_CONTRACT_VERSION
            ),
            "execution_digest": execution_digest,
            "workflow_script_sha256": file_sha256(
                str(Path(__file__).resolve())
            ),
            "recorded_at": datetime.now().isoformat(
                timespec="seconds"
            ),
        },
        Path(output_dir)
        / "checkpoints"
        / "credal_contract_fingerprints.json",
    )

    buckets: Dict[str, List[pd.DataFrame]] = {
        "outer_statistics": [],
        "training_statistics": [],
        "shadow_statistics": [],
        "shap_full_display": [],
        "shap_display_target_concordance": [],
        "outer_metrics": [],
        "outer_predictions": [],
        "outer_calibration": [],
        "pdp": [],
        "ice": [],
        "errors": [],
    }
    pfi_audit_passed_by_model: Dict[str, bool] = {}

    for part in partitions:
        partition_id = int(part.partition_id)
        models = [
            str(model)
            for model in admissible_models_by_partition.get(partition_id, [])
        ]
        if not models:
            raise RuntimeError(
                f"No admissible model was supplied for outer partition {partition_id}."
            )
        X_outer_train = X_all.iloc[part.train_idx].reset_index(drop=True)
        X_outer_holdout = X_all.iloc[part.test_idx].reset_index(drop=True)
        y_outer_train = np.asarray(y_all, dtype=int)[part.train_idx]
        y_outer_holdout = np.asarray(y_all, dtype=int)[part.test_idx]

        for model_name in models:
            if model_name not in pfi_audit_passed_by_model:
                pfi_audit = ensure_pfi_batch_equivalence_audit(
                    X_outer_train=X_outer_train,
                    y_outer_train=y_outer_train,
                    schema=schema,
                    part=part,
                    model_name=model_name,
                    class_order=class_order,
                    audit_root=(
                        Path(output_dir)
                        / "checkpoints"
                        / "pfi_batch_equivalence"
                    ),
                )
                pfi_audit_passed_by_model[model_name] = bool(
                    not pfi_audit.empty
                    and pd.to_numeric(
                        pfi_audit["passed"], errors="coerce"
                    )
                    .fillna(0)
                    .astype(int)
                    .eq(1)
                    .all()
                )
            pfi_audit_passed = bool(
                pfi_audit_passed_by_model[model_name]
            )
            if not pfi_audit_passed:
                raise RuntimeError(
                    "PFI execution-equivalence audit did not pass for "
                    f"outer={partition_id}, model={model_name}."
                )
            cprint(
                f"[CREDAL ACTUAL] outer={partition_id} model={model_name}"
            )
            actual_fp = credal_unit_fingerprint(
                "outer_actual",
                part,
                model_name,
                None,
                None,
                data_digest,
                method_digest,
            )
            actual_legacy_fingerprints = (
                compatible_legacy_unit_fingerprints(
                    kind="outer_actual",
                    part=part,
                    model_name=model_name,
                    feature=None,
                    replicate=None,
                    data_digest=data_digest,
                    legacy_method_candidates=(
                        legacy_method_candidates
                    ),
                )
            )
            actual = (
                load_credal_unit_checkpoint(
                    checkpoint_root,
                    "outer_actual",
                    part,
                    model_name,
                    None,
                    None,
                    actual_fp,
                    compatible_legacy_fingerprints=(
                        actual_legacy_fingerprints
                    ),
                )
                if ENABLE_SHADOW_UNIT_CHECKPOINTS
                else None
            )
            if actual is not None:
                if not credal_core_payload_complete(
                    actual,
                    feature_cols=schema.feature_cols,
                    class_order=class_order,
                ):
                    raise RuntimeError(
                        "Restored outer-actual checkpoint failed the "
                        f"current payload contract: outer={partition_id}, "
                        f"model={model_name}."
                    )
                if bool(
                    actual.get(
                        "_checkpoint_requires_migration", False
                    )
                ):
                    actual = migrate_compatible_credal_checkpoint(
                        checkpoint_root=checkpoint_root,
                        kind="outer_actual",
                        part=part,
                        model_name=model_name,
                        feature=None,
                        replicate=None,
                        expected_fingerprint=actual_fp,
                        payload=actual,
                        compatibility_audit=(
                            "payload_complete; pfi_reference_vs_batch_"
                            "equivalent; statistical_contract_unchanged"
                        ),
                    )
            if actual is None:
                actual = run_outer_actual_unit(
                    X_outer_train=X_outer_train,
                    X_outer_holdout=X_outer_holdout,
                    y_outer_train=y_outer_train,
                    y_outer_holdout=y_outer_holdout,
                    schema=schema,
                    part=part,
                    model_name=model_name,
                    class_order=class_order,
                    worker_root=worker_root,
                )
                if ENABLE_SHADOW_UNIT_CHECKPOINTS and credal_core_payload_complete(
                    actual,
                    feature_cols=schema.feature_cols,
                    class_order=class_order,
                ):
                    save_credal_unit_checkpoint(
                        checkpoint_root,
                        "outer_actual",
                        part,
                        model_name,
                        None,
                        None,
                        actual_fp,
                        actual,
                    )
            for key, bucket in [
                ("statistics", "outer_statistics"),
                ("shap_full_display", "shap_full_display"),
                (
                    "shap_display_target_concordance",
                    "shap_display_target_concordance",
                ),
                ("metrics", "outer_metrics"),
                ("predictions", "outer_predictions"),
                ("calibration", "outer_calibration"),
                ("pdp", "pdp"),
                ("ice", "ice"),
                ("errors", "errors"),
            ]:
                frame = actual.get(key, pd.DataFrame())
                if isinstance(frame, pd.DataFrame) and not frame.empty:
                    buckets[bucket].append(frame)

            cprint(
                f"[CREDAL TRAINING REFERENCE] outer={partition_id} model={model_name}"
            )
            training_fp = credal_unit_fingerprint(
                "training_original_crossfit",
                part,
                model_name,
                None,
                None,
                data_digest,
                method_digest,
            )
            training_legacy_fingerprints = (
                compatible_legacy_unit_fingerprints(
                    kind="training_original_crossfit",
                    part=part,
                    model_name=model_name,
                    feature=None,
                    replicate=None,
                    data_digest=data_digest,
                    legacy_method_candidates=(
                        legacy_method_candidates
                    ),
                )
            )
            training = (
                load_credal_unit_checkpoint(
                    checkpoint_root,
                    "training_original_crossfit",
                    part,
                    model_name,
                    None,
                    None,
                    training_fp,
                    compatible_legacy_fingerprints=(
                        training_legacy_fingerprints
                    ),
                )
                if ENABLE_SHADOW_UNIT_CHECKPOINTS
                else None
            )
            if training is not None:
                if not credal_core_payload_complete(
                    training,
                    feature_cols=schema.feature_cols,
                    class_order=class_order,
                    expected_inner_folds=int(
                        SHADOW_INNER_CV_FOLDS
                    ),
                ):
                    raise RuntimeError(
                        "Restored training-reference checkpoint failed "
                        f"the current payload contract: outer={partition_id}, "
                        f"model={model_name}."
                    )
                if bool(
                    training.get(
                        "_checkpoint_requires_migration", False
                    )
                ):
                    training = migrate_compatible_credal_checkpoint(
                        checkpoint_root=checkpoint_root,
                        kind="training_original_crossfit",
                        part=part,
                        model_name=model_name,
                        feature=None,
                        replicate=None,
                        expected_fingerprint=training_fp,
                        payload=training,
                        compatibility_audit=(
                            "payload_complete; pfi_reference_vs_batch_"
                            "equivalent; statistical_contract_unchanged"
                        ),
                    )
            if training is None:
                training_loco_cache = (
                    prepare_loco_reduced_probability_cache(
                        X_outer_train=X_outer_train,
                        y_outer_train=y_outer_train,
                        schema=schema,
                        part=part,
                        model_name=model_name,
                        class_order=class_order,
                        checkpoint_root=checkpoint_root,
                        data_digest=data_digest,
                        method_digest=method_digest,
                    )
                    if ENABLE_LOCO_REDUCED_PROBABILITY_CACHE
                    else {}
                )
                training = run_original_crossfit_unit(
                    X_outer_train=X_outer_train,
                    y_outer_train=y_outer_train,
                    schema=schema,
                    part=part,
                    model_name=model_name,
                    class_order=class_order,
                    worker_root=worker_root,
                    loco_reduced_cache=training_loco_cache,
                )
                if ENABLE_SHADOW_UNIT_CHECKPOINTS and credal_core_payload_complete(
                    training,
                    feature_cols=schema.feature_cols,
                    class_order=class_order,
                    expected_inner_folds=int(
                        training.get("n_inner_folds_expected", 0)
                    ),
                ):
                    save_credal_unit_checkpoint(
                        checkpoint_root,
                        "training_original_crossfit",
                        part,
                        model_name,
                        None,
                        None,
                        training_fp,
                        training,
                    )
            training_frame = training.get("statistics", pd.DataFrame())
            if isinstance(training_frame, pd.DataFrame) and not training_frame.empty:
                buckets["training_statistics"].append(training_frame)
            training_errors = training.get("errors", pd.DataFrame())
            if isinstance(training_errors, pd.DataFrame) and not training_errors.empty:
                annotated_errors = training_errors.copy()
                annotated_errors["outer_partition"] = partition_id
                annotated_errors["model"] = model_name
                annotated_errors["unit"] = "training_original_crossfit"
                buckets["errors"].append(annotated_errors)

            _execute_parallel_shadow_units(
                X_outer_train=X_outer_train,
                y_outer_train=y_outer_train,
                schema=schema,
                part=part,
                model_name=model_name,
                class_order=class_order,
                checkpoint_root=checkpoint_root,
                worker_root=worker_root,
                data_digest=data_digest,
                method_digest=method_digest,
                legacy_method_candidates=legacy_method_candidates,
                pfi_audit_passed=pfi_audit_passed,
                n_shadow=n_shadow,
                buckets=buckets,
            )

    outputs = {
        key: concat_or_empty(frames)
        for key, frames in buckets.items()
    }
    preferred_sort_columns = [
        "outer_partition",
        "outer_repeat",
        "outer_fold",
        "model",
        "source",
        "class_id",
        "feature",
        "shadow_replicate",
        "inner_fold",
        "holdout_position",
    ]
    for key, frame in outputs.items():
        if frame is None or frame.empty:
            continue
        sort_columns = [
            column for column in preferred_sort_columns if column in frame.columns
        ]
        if sort_columns:
            outputs[key] = frame.sort_values(
                sort_columns,
                kind="mergesort",
            ).reset_index(drop=True)
    return outputs


def _inner_fold_contract_status(
    table: pd.DataFrame,
    expected_folds: int,
) -> Tuple[bool, str]:
    if table is None or table.empty or "n_inner_folds" not in table.columns:
        return False, "missing_inner_fold_count"
    counts = (
        pd.to_numeric(table["n_inner_folds"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    if len(counts) != 1 or int(counts[0]) != int(expected_folds):
        return False, "incomplete_inner_crossfit"
    return True, "valid"


def build_credal_common_support(
    outer_statistics: pd.DataFrame,
    training_statistics: pd.DataFrame,
    shadow_statistics: pd.DataFrame,
    partitions: Sequence[OuterPartition],
    admissible_models_by_partition: Mapping[int, Sequence[str]],
    feature_cols: Sequence[str],
    class_order: Sequence[int],
    n_shadow_replicates: int,
    expected_inner_folds_by_partition: Mapping[int, int],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple[int, int], List[str]]]:
    """Apply the common-support contract before relevance calibration."""
    validity_rows = []
    coverage_rows = []
    common_support: Dict[Tuple[int, int], List[str]] = {}

    for part in partitions:
        partition_id = int(part.partition_id)
        expected_folds_contract = int(
            expected_inner_folds_by_partition[partition_id]
        )
        models = [
            str(model)
            for model in admissible_models_by_partition.get(partition_id, [])
        ]
        for class_id in class_order:
            valid_models = []
            for model_name in models:
                all_sources_valid = True
                for source in CREDAL_SOURCE_NAMES:
                    outer_group = outer_statistics.loc[
                        (
                            pd.to_numeric(
                                outer_statistics["outer_partition"], errors="coerce"
                            )
                            == partition_id
                        )
                        & (outer_statistics["model"].astype(str) == model_name)
                    ].copy()
                    outer_valid, outer_reason = source_table_contract_status(
                        outer_group,
                        source=source,
                        class_id=int(class_id),
                        feature_cols=feature_cols,
                    )

                    training_group = training_statistics.loc[
                        (
                            pd.to_numeric(
                                training_statistics["outer_partition"],
                                errors="coerce",
                            )
                            == partition_id
                        )
                        & (training_statistics["model"].astype(str) == model_name)
                    ].copy()
                    training_valid, training_reason = source_table_contract_status(
                        training_group,
                        source=source,
                        class_id=int(class_id),
                        feature_cols=feature_cols,
                    )
                    expected_folds_values = (
                        pd.to_numeric(
                            training_group.loc[
                                (
                                    training_group["source"].astype(str)
                                    == str(source)
                                )
                                & (
                                    pd.to_numeric(
                                        training_group["class_id"], errors="coerce"
                                    )
                                    == int(class_id)
                                ),
                                "n_inner_folds",
                            ],
                            errors="coerce",
                        )
                        .dropna()
                        .astype(int)
                        .unique()
                        if "n_inner_folds" in training_group.columns
                        else np.asarray([], dtype=int)
                    )
                    if training_valid:
                        if (
                            len(expected_folds_values) != 1
                            or int(expected_folds_values[0]) < 2
                        ):
                            training_valid = False
                            training_reason = "invalid_inner_crossfit_count"
                    expected_folds = (
                        int(expected_folds_values[0])
                        if len(expected_folds_values) == 1
                        else 0
                    )
                    if (
                        training_valid
                        and expected_folds != expected_folds_contract
                    ):
                        training_valid = False
                        training_reason = "incomplete_inner_crossfit"

                    shadow_source = shadow_statistics.loc[
                        (
                            pd.to_numeric(
                                shadow_statistics["outer_partition"], errors="coerce"
                            )
                            == partition_id
                        )
                        & (shadow_statistics["model"].astype(str) == model_name)
                        & (shadow_statistics["source"].astype(str) == str(source))
                        & (
                            pd.to_numeric(
                                shadow_statistics["class_id"], errors="coerce"
                            )
                            == int(class_id)
                        )
                    ].copy()
                    shadow_valid = True
                    shadow_reason = "valid"
                    for feature in feature_cols:
                        feature_null = shadow_source.loc[
                            shadow_source["feature"].astype(str) == str(feature)
                        ].copy()
                        replicates = (
                            pd.to_numeric(
                                feature_null["shadow_replicate"], errors="coerce"
                            )
                            .dropna()
                            .astype(int)
                        )
                        if (
                            len(feature_null) != int(n_shadow_replicates)
                            or replicates.nunique() != int(n_shadow_replicates)
                            or replicates.duplicated().any()
                        ):
                            shadow_valid = False
                            shadow_reason = f"incomplete_shadow_replicates:{feature}"
                            break
                        values = pd.to_numeric(
                            feature_null["raw_statistic"], errors="coerce"
                        ).to_numpy(dtype=float)
                        if not np.isfinite(values).all():
                            shadow_valid = False
                            shadow_reason = f"nonfinite_shadow_statistic:{feature}"
                            break
                        if expected_folds_contract > 0:
                            fold_counts = (
                                pd.to_numeric(
                                    feature_null["n_inner_folds"], errors="coerce"
                                )
                                .dropna()
                                .astype(int)
                                .unique()
                                if "n_inner_folds" in feature_null.columns
                                else np.asarray([], dtype=int)
                            )
                            if (
                                len(fold_counts) != 1
                                or int(fold_counts[0])
                                != int(expected_folds_contract)
                            ):
                                shadow_valid = False
                                shadow_reason = (
                                    f"incomplete_shadow_inner_crossfit:{feature}"
                                )
                                break

                    source_valid = bool(
                        outer_valid and training_valid and shadow_valid
                    )
                    all_sources_valid = all_sources_valid and source_valid
                    validity_rows.append(
                        {
                            "outer_partition": partition_id,
                            "outer_repeat": int(part.repeat_id),
                            "outer_fold": int(part.fold_id),
                            "class_id": int(class_id),
                            "model": model_name,
                            "source": str(source),
                            "outer_actual_valid": int(outer_valid),
                            "outer_actual_reason": outer_reason,
                            "training_original_valid": int(training_valid),
                            "training_original_reason": training_reason,
                            "shadow_null_valid": int(shadow_valid),
                            "shadow_null_reason": shadow_reason,
                            "source_specification_valid": int(source_valid),
                            "expected_inner_folds": int(expected_folds),
                            "required_inner_folds": int(
                                expected_folds_contract
                            ),
                            "expected_shadow_replicates": int(
                                n_shadow_replicates
                            ),
                        }
                    )
                if all_sources_valid:
                    valid_models.append(model_name)

            common_support[(partition_id, int(class_id))] = sorted(valid_models)
            coverage_rows.append(
                {
                    "outer_partition": partition_id,
                    "outer_repeat": int(part.repeat_id),
                    "outer_fold": int(part.fold_id),
                    "class_id": int(class_id),
                    "n_admissible_models": int(len(models)),
                    "n_common_support_models": int(len(valid_models)),
                    "coverage": float(len(valid_models) / max(len(models), 1)),
                    "admissible_models": "|".join(sorted(models)),
                    "common_support_models": "|".join(sorted(valid_models)),
                    "common_support_nonempty": int(bool(valid_models)),
                }
            )

    return (
        pd.DataFrame(validity_rows),
        pd.DataFrame(coverage_rows),
        common_support,
    )


def calibrate_shadow_relevance_percentiles(
    observed_statistics: pd.DataFrame,
    shadow_statistics: pd.DataFrame,
    common_support: Mapping[Tuple[int, int], Sequence[str]],
    role: str,
    n_shadow_replicates: int,
) -> pd.DataFrame:
    """Map signed source statistics to finite replacement-shadow percentiles."""
    rows = []
    for _, observed in observed_statistics.iterrows():
        partition_id = int(observed["outer_partition"])
        class_id = int(observed["class_id"])
        model_name = str(observed["model"])
        source = str(observed["source"])
        feature = str(observed["feature"])
        if model_name not in set(common_support.get((partition_id, class_id), [])):
            continue
        null = shadow_statistics.loc[
            (
                pd.to_numeric(
                    shadow_statistics["outer_partition"], errors="coerce"
                )
                == partition_id
            )
            & (shadow_statistics["model"].astype(str) == model_name)
            & (shadow_statistics["source"].astype(str) == source)
            & (
                pd.to_numeric(shadow_statistics["class_id"], errors="coerce")
                == class_id
            )
            & (shadow_statistics["feature"].astype(str) == feature)
        ].copy()
        null_values = pd.to_numeric(
            null["raw_statistic"], errors="coerce"
        ).to_numpy(dtype=float)
        if (
            len(null_values) != int(n_shadow_replicates)
            or not np.isfinite(null_values).all()
        ):
            raise RuntimeError(
                "Shadow calibration received an incomplete null after common-support "
                f"validation: outer={partition_id}, class={class_id}, "
                f"model={model_name}, source={source}, feature={feature}."
            )
        observed_value = float(observed["raw_statistic"])
        exceedance_count = int(np.sum(null_values >= observed_value))
        p_tilde = float(
            (1 + exceedance_count) / (int(n_shadow_replicates) + 1)
        )
        rows.append(
            {
                "outer_partition": partition_id,
                "outer_repeat": int(observed.get("outer_repeat", 0)),
                "outer_fold": int(observed.get("outer_fold", 0)),
                "split_seed": int(observed.get("split_seed", 0)),
                "model": model_name,
                "source": source,
                "class_id": class_id,
                "feature": feature,
                "observed_raw_statistic": observed_value,
                "shadow_exceedance_count": exceedance_count,
                "n_shadow_replicates": int(n_shadow_replicates),
                "shadow_tail_fraction_finite": p_tilde,
                "relevance_percentile": float(1.0 - p_tilde),
                "shadow_mean": float(np.mean(null_values)),
                "shadow_std": float(np.std(null_values, ddof=1))
                if len(null_values) > 1
                else 0.0,
                "shadow_q05": float(np.quantile(null_values, 0.05)),
                "shadow_median": float(np.median(null_values)),
                "shadow_q95": float(np.quantile(null_values, 0.95)),
                "calibration_role": str(role),
                "scale_boundary": (
                    "operational replacement-shadow relevance percentile; "
                    "not a p-value, probability, effect size, or causal effect"
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_relevance_lookup(
    calibrated: pd.DataFrame,
) -> Dict[Tuple[int, str, str, int, str], float]:
    required = [
        "outer_partition",
        "model",
        "source",
        "class_id",
        "feature",
        "relevance_percentile",
    ]
    missing = [column for column in required if column not in calibrated.columns]
    if missing:
        raise ValueError(f"Calibrated evidence lacks columns: {missing}")
    keys = required[:-1]
    if calibrated.duplicated(keys).any():
        duplicate = calibrated.loc[calibrated.duplicated(keys, keep=False), keys]
        raise RuntimeError(
            "Calibrated evidence contains duplicate specification rows: "
            f"{duplicate.head(10).to_dict('records')}"
        )
    return {
        (
            int(row.outer_partition),
            str(row.model),
            str(row.source),
            int(row.class_id),
            str(row.feature),
        ): float(row.relevance_percentile)
        for row in calibrated.itertuples(index=False)
    }


def _topological_levels_from_strict_edges(
    features: Sequence[str],
    strict_edges: Set[Tuple[str, str]],
) -> Dict[str, int]:
    remaining = {str(feature) for feature in features}
    levels: Dict[str, int] = {}
    level = 1
    while remaining:
        incoming = {
            target
            for source, target in strict_edges
            if source in remaining and target in remaining
        }
        maxima = sorted(remaining - incoming)
        if not maxima:
            raise RuntimeError(
                "The strict credal preference relation contains a directed cycle."
            )
        for feature in maxima:
            levels[feature] = int(level)
        remaining -= set(maxima)
        level += 1
    return levels


def _has_directed_path(
    source: str,
    target: str,
    edges: Set[Tuple[str, str]],
) -> bool:
    adjacency: Dict[str, List[str]] = {}
    for left, right in edges:
        adjacency.setdefault(str(left), []).append(str(right))
    stack = list(adjacency.get(str(source), []))
    visited = set()
    while stack:
        node = stack.pop()
        if node == str(target):
            return True
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adjacency.get(node, []))
    return False


def transitive_reduction_edges(
    strict_edges: Set[Tuple[str, str]],
) -> Set[Tuple[str, str]]:
    reduced = set(strict_edges)
    for edge in sorted(strict_edges):
        candidate = set(reduced)
        candidate.discard(edge)
        if _has_directed_path(edge[0], edge[1], candidate):
            reduced.discard(edge)
    return reduced


def compute_credal_partial_order(
    calibrated: pd.DataFrame,
    partitions: Sequence[OuterPartition],
    common_support: Mapping[Tuple[int, int], Sequence[str]],
    feature_cols: Sequence[str],
    class_order: Sequence[int],
    sources: Optional[Sequence[str]] = None,
    repeat_subset: Optional[Sequence[int]] = None,
    analysis_label: str = "primary",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute lower expectations and strict preferences over joint specifications."""
    source_set = [str(source) for source in (sources or CREDAL_SOURCE_NAMES)]
    repeat_set = (
        sorted({int(part.repeat_id) for part in partitions})
        if repeat_subset is None
        else sorted({int(value) for value in repeat_subset})
    )
    selected_parts = [
        part for part in partitions if int(part.repeat_id) in set(repeat_set)
    ]
    lookup = _build_relevance_lookup(calibrated)
    part_by_repeat: Dict[int, List[OuterPartition]] = {}
    for part in selected_parts:
        part_by_repeat.setdefault(int(part.repeat_id), []).append(part)

    feature_rows = []
    pair_rows = []
    hasse_rows = []
    for class_id in class_order:
        feature_bounds: Dict[str, Tuple[float, float, str, str]] = {}
        for feature in feature_cols:
            lower_candidates = []
            upper_candidates = []
            for repeat_id, repeat_parts in sorted(part_by_repeat.items()):
                total_holdout = float(sum(len(part.test_idx) for part in repeat_parts))
                for source in source_set:
                    lower_value = 0.0
                    upper_value = 0.0
                    for part in repeat_parts:
                        models = [
                            str(model)
                            for model in common_support.get(
                                (int(part.partition_id), int(class_id)), []
                            )
                        ]
                        if not models:
                            raise RuntimeError(
                                "Empty common support reached credal inference: "
                                f"outer={part.partition_id}, class={class_id}."
                            )
                        values = []
                        for model in models:
                            key = (
                                int(part.partition_id),
                                model,
                                source,
                                int(class_id),
                                str(feature),
                            )
                            if key not in lookup:
                                raise RuntimeError(
                                    f"Missing calibrated specification row: {key}"
                                )
                            values.append(float(lookup[key]))
                        fold_weight = float(len(part.test_idx) / total_holdout)
                        lower_value += fold_weight * float(np.min(values))
                        upper_value += fold_weight * float(np.max(values))
                    lower_candidates.append((lower_value, repeat_id, source))
                    upper_candidates.append((upper_value, repeat_id, source))
            lower_record = min(lower_candidates, key=lambda item: item[0])
            upper_record = max(upper_candidates, key=lambda item: item[0])
            feature_bounds[str(feature)] = (
                float(lower_record[0]),
                float(upper_record[0]),
                f"repeat={lower_record[1]}|source={lower_record[2]}",
                f"repeat={upper_record[1]}|source={upper_record[2]}",
            )

        strict_edges: Set[Tuple[str, str]] = set()
        for feature_j in feature_cols:
            for feature_k in feature_cols:
                if str(feature_j) == str(feature_k):
                    continue
                lower_candidates = []
                upper_candidates = []
                for repeat_id, repeat_parts in sorted(part_by_repeat.items()):
                    total_holdout = float(
                        sum(len(part.test_idx) for part in repeat_parts)
                    )
                    for source in source_set:
                        fold_lower = 0.0
                        fold_upper = 0.0
                        for part in repeat_parts:
                            models = [
                                str(model)
                                for model in common_support.get(
                                    (int(part.partition_id), int(class_id)), []
                                )
                            ]
                            differences = []
                            for model in models:
                                key_j = (
                                    int(part.partition_id),
                                    model,
                                    source,
                                    int(class_id),
                                    str(feature_j),
                                )
                                key_k = (
                                    int(part.partition_id),
                                    model,
                                    source,
                                    int(class_id),
                                    str(feature_k),
                                )
                                if key_j not in lookup or key_k not in lookup:
                                    raise RuntimeError(
                                        "Pairwise credal comparison lacks a "
                                        f"specification row: {key_j}, {key_k}"
                                    )
                                differences.append(
                                    float(lookup[key_j] - lookup[key_k])
                                )
                            fold_weight = float(
                                len(part.test_idx) / total_holdout
                            )
                            fold_lower += fold_weight * float(
                                np.min(differences)
                            )
                            fold_upper += fold_weight * float(
                                np.max(differences)
                            )
                        lower_candidates.append(
                            (fold_lower, repeat_id, source)
                        )
                        upper_candidates.append(
                            (fold_upper, repeat_id, source)
                        )
                lower_record = min(lower_candidates, key=lambda item: item[0])
                upper_record = max(upper_candidates, key=lambda item: item[0])
                strict = bool(
                    float(lower_record[0]) > float(SHADOW_NUMERICAL_TOLERANCE)
                )
                if strict:
                    strict_edges.add((str(feature_j), str(feature_k)))
                pair_rows.append(
                    {
                        "analysis": str(analysis_label),
                        "class_id": int(class_id),
                        "feature_j": str(feature_j),
                        "feature_k": str(feature_k),
                        "lower_difference": float(lower_record[0]),
                        "upper_difference": float(upper_record[0]),
                        "strict_preference_j_over_k": int(strict),
                        "lower_attaining_repeat": int(lower_record[1]),
                        "lower_attaining_source": str(lower_record[2]),
                        "upper_attaining_repeat": int(upper_record[1]),
                        "upper_attaining_source": str(upper_record[2]),
                    }
                )

        for left, right in strict_edges:
            if (right, left) in strict_edges:
                raise RuntimeError(
                    f"Mutual strict preference detected for {left} and {right}."
                )
        levels = _topological_levels_from_strict_edges(
            feature_cols, strict_edges
        )
        maximal = {
            feature
            for feature in map(str, feature_cols)
            if not any(target == feature for _, target in strict_edges)
        }
        hasse = transitive_reduction_edges(strict_edges)
        for feature in feature_cols:
            lower, upper, lower_spec, upper_spec = feature_bounds[str(feature)]
            feature_rows.append(
                {
                    "analysis": str(analysis_label),
                    "class_id": int(class_id),
                    "feature": str(feature),
                    "lower_expected_percentile": lower,
                    "upper_expected_percentile": upper,
                    "specification_imprecision_width": float(upper - lower),
                    "partial_order_level": int(levels[str(feature)]),
                    "in_maximal_set": int(str(feature) in maximal),
                    "n_strictly_dominates": int(
                        sum(left == str(feature) for left, _ in strict_edges)
                    ),
                    "n_strictly_dominated_by": int(
                        sum(right == str(feature) for _, right in strict_edges)
                    ),
                    "lower_attaining_specification": lower_spec,
                    "upper_attaining_specification": upper_spec,
                    "n_sources": int(len(source_set)),
                    "n_repeats": int(len(repeat_set)),
                }
            )
        pair_lookup = {
            (str(row["feature_j"]), str(row["feature_k"])): row
            for row in pair_rows
            if int(row["class_id"]) == int(class_id)
            and str(row["analysis"]) == str(analysis_label)
        }
        for left, right in sorted(hasse):
            pair = pair_lookup[(left, right)]
            hasse_rows.append(
                {
                    "analysis": str(analysis_label),
                    "class_id": int(class_id),
                    "source_feature": left,
                    "target_feature": right,
                    "source_level": int(levels[left]),
                    "target_level": int(levels[right]),
                    "lower_difference": float(pair["lower_difference"]),
                    "upper_difference": float(pair["upper_difference"]),
                }
            )
    return (
        pd.DataFrame(feature_rows),
        pd.DataFrame(pair_rows),
        pd.DataFrame(hasse_rows),
    )


def compute_local_training_partial_orders(
    calibrated_training: pd.DataFrame,
    common_support: Mapping[Tuple[int, int], Sequence[str]],
    partitions: Sequence[OuterPartition],
    feature_cols: Sequence[str],
    class_order: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build one training-only credal order for each outer partition and class."""
    lookup = _build_relevance_lookup(calibrated_training)
    summary_rows = []
    edge_rows = []
    for part in partitions:
        partition_id = int(part.partition_id)
        for class_id in class_order:
            models = [
                str(model)
                for model in common_support.get((partition_id, int(class_id)), [])
            ]
            if not models:
                raise RuntimeError(
                    "Training order cannot be built from empty common support: "
                    f"outer={partition_id}, class={class_id}."
                )
            strict_edges: Set[Tuple[str, str]] = set()
            feature_bounds = {}
            for feature in feature_cols:
                values = [
                    float(
                        lookup[
                            (
                                partition_id,
                                model,
                                source,
                                int(class_id),
                                str(feature),
                            )
                        ]
                    )
                    for source in CREDAL_SOURCE_NAMES
                    for model in models
                ]
                feature_bounds[str(feature)] = (
                    float(np.min(values)),
                    float(np.max(values)),
                )
            for feature_j in feature_cols:
                for feature_k in feature_cols:
                    if str(feature_j) == str(feature_k):
                        continue
                    source_lower = []
                    source_upper = []
                    for source in CREDAL_SOURCE_NAMES:
                        differences = [
                            float(
                                lookup[
                                    (
                                        partition_id,
                                        model,
                                        source,
                                        int(class_id),
                                        str(feature_j),
                                    )
                                ]
                                - lookup[
                                    (
                                        partition_id,
                                        model,
                                        source,
                                        int(class_id),
                                        str(feature_k),
                                    )
                                ]
                            )
                            for model in models
                        ]
                        source_lower.append(float(np.min(differences)))
                        source_upper.append(float(np.max(differences)))
                    lower = float(np.min(source_lower))
                    upper = float(np.max(source_upper))
                    if lower > float(SHADOW_NUMERICAL_TOLERANCE):
                        strict_edges.add((str(feature_j), str(feature_k)))
                    edge_rows.append(
                        {
                            "outer_partition": partition_id,
                            "outer_repeat": int(part.repeat_id),
                            "outer_fold": int(part.fold_id),
                            "class_id": int(class_id),
                            "feature_j": str(feature_j),
                            "feature_k": str(feature_k),
                            "lower_difference": lower,
                            "upper_difference": upper,
                            "strict_preference_j_over_k": int(
                                lower > float(SHADOW_NUMERICAL_TOLERANCE)
                            ),
                        }
                    )
            levels = _topological_levels_from_strict_edges(
                feature_cols, strict_edges
            )
            maximal = {
                feature
                for feature in map(str, feature_cols)
                if not any(target == feature for _, target in strict_edges)
            }
            for feature in feature_cols:
                lower, upper = feature_bounds[str(feature)]
                summary_rows.append(
                    {
                        "outer_partition": partition_id,
                        "outer_repeat": int(part.repeat_id),
                        "outer_fold": int(part.fold_id),
                        "class_id": int(class_id),
                        "feature": str(feature),
                        "lower_expected_percentile": lower,
                        "upper_expected_percentile": upper,
                        "specification_imprecision_width": float(upper - lower),
                        "partial_order_level": int(levels[str(feature)]),
                        "in_maximal_set": int(str(feature) in maximal),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(edge_rows)


def compute_credal_sensitivity_diagnostics(
    calibrated_outer: pd.DataFrame,
    partitions: Sequence[OuterPartition],
    common_support: Mapping[Tuple[int, int], Sequence[str]],
    feature_cols: Sequence[str],
    class_order: Sequence[int],
    primary_summary: pd.DataFrame,
    primary_hasse: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assess source deletion and repeat-specific preference reproducibility."""
    source_rows = []
    source_feature_parts = []
    if ENABLE_LEAVE_ONE_EXPLAINER_OUT_SENSITIVITY:
        for omitted in CREDAL_SOURCE_NAMES:
            retained = [
                source for source in CREDAL_SOURCE_NAMES if source != omitted
            ]
            summary, _, hasse = compute_credal_partial_order(
                calibrated=calibrated_outer,
                partitions=partitions,
                common_support=common_support,
                feature_cols=feature_cols,
                class_order=class_order,
                sources=retained,
                analysis_label=f"leave_out_{omitted}",
            )
            source_feature_parts.append(summary)
            for class_id in class_order:
                full_maximal = set(
                    primary_summary.loc[
                        (
                            pd.to_numeric(
                                primary_summary["class_id"], errors="coerce"
                            )
                            == int(class_id)
                        )
                        & (primary_summary["in_maximal_set"].astype(int) == 1),
                        "feature",
                    ].astype(str)
                )
                reduced_maximal = set(
                    summary.loc[
                        (
                            pd.to_numeric(summary["class_id"], errors="coerce")
                            == int(class_id)
                        )
                        & (summary["in_maximal_set"].astype(int) == 1),
                        "feature",
                    ].astype(str)
                )
                full_edges = {
                    (str(row.source_feature), str(row.target_feature))
                    for row in primary_hasse.loc[
                        pd.to_numeric(primary_hasse["class_id"], errors="coerce")
                        == int(class_id)
                    ].itertuples(index=False)
                }
                reduced_edges = {
                    (str(row.source_feature), str(row.target_feature))
                    for row in hasse.loc[
                        pd.to_numeric(hasse["class_id"], errors="coerce")
                        == int(class_id)
                    ].itertuples(index=False)
                }
                source_rows.append(
                    {
                        "omitted_source": str(omitted),
                        "class_id": int(class_id),
                        "maximal_set_jaccard": float(
                            len(full_maximal & reduced_maximal)
                            / max(len(full_maximal | reduced_maximal), 1)
                        ),
                        "hasse_edge_jaccard": float(
                            len(full_edges & reduced_edges)
                            / max(len(full_edges | reduced_edges), 1)
                        ),
                        "n_primary_maximal": int(len(full_maximal)),
                        "n_reduced_maximal": int(len(reduced_maximal)),
                        "n_primary_hasse_edges": int(len(full_edges)),
                        "n_reduced_hasse_edges": int(len(reduced_edges)),
                    }
                )

    repeat_edge_parts = []
    repeat_feature_parts = []
    if ENABLE_REPEAT_CONSISTENCY_DIAGNOSTIC:
        repeat_ids = sorted({int(part.repeat_id) for part in partitions})
        for repeat_id in repeat_ids:
            summary, _, hasse = compute_credal_partial_order(
                calibrated=calibrated_outer,
                partitions=partitions,
                common_support=common_support,
                feature_cols=feature_cols,
                class_order=class_order,
                repeat_subset=[repeat_id],
                analysis_label=f"repeat_{repeat_id}",
            )
            repeat_feature_parts.append(summary)
            hasse["outer_repeat"] = int(repeat_id)
            repeat_edge_parts.append(hasse)

    repeat_edges = concat_or_empty(repeat_edge_parts)
    repeat_consistency_rows = []
    if not repeat_edges.empty:
        n_repeats = int(repeat_edges["outer_repeat"].nunique())
        for class_id in class_order:
            class_edges = repeat_edges.loc[
                pd.to_numeric(repeat_edges["class_id"], errors="coerce")
                == int(class_id)
            ]
            all_possible = sorted(
                {
                    (str(left), str(right))
                    for left in feature_cols
                    for right in feature_cols
                    if str(left) != str(right)
                }
            )
            for left, right in all_possible:
                present = class_edges.loc[
                    (class_edges["source_feature"].astype(str) == left)
                    & (class_edges["target_feature"].astype(str) == right),
                    "outer_repeat",
                ].nunique()
                repeat_consistency_rows.append(
                    {
                        "class_id": int(class_id),
                        "source_feature": left,
                        "target_feature": right,
                        "n_repeats_with_hasse_edge": int(present),
                        "n_repeats": n_repeats,
                        "repeat_edge_frequency": float(present / max(n_repeats, 1)),
                    }
                )
    return (
        pd.DataFrame(source_rows),
        concat_or_empty(source_feature_parts),
        pd.DataFrame(repeat_consistency_rows),
    )


def _reduced_feature_schema(
    schema: FeatureSchema,
    retained_features: Sequence[str],
) -> FeatureSchema:
    retained = [str(feature) for feature in retained_features]
    if not retained:
        raise ValueError("A predictive specification cannot be fitted with zero features.")
    return FeatureSchema(
        feature_cols=retained,
        numeric_cols=[feature for feature in schema.numeric_cols if feature in retained],
        categorical_cols=[
            feature for feature in schema.categorical_cols if feature in retained
        ],
    )


def run_training_order_deletion_validation(
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    schema: FeatureSchema,
    partitions: Sequence[OuterPartition],
    common_support: Mapping[Tuple[int, int], Sequence[str]],
    training_order_summary: pd.DataFrame,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate training-only order layers on outer holdouts without retuning."""
    if not ENABLE_TRAINING_ONLY_UTILITY_VALIDATION:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    class_order = [0, 1, 2]
    raw_rows = []
    error_rows = []
    checkpoint_root = Path(output_dir) / "checkpoints" / "credal_shadow_units"
    data_digest = stable_json_digest(
        {
            "X": dataframe_digest(X_all.reset_index(drop=True)),
            "y": array_digest(np.asarray(y_all, dtype=np.int64)),
        }
    )
    method_digest = credal_method_digest(schema)

    for part in partitions:
        partition_id = int(part.partition_id)
        X_train_full = X_all.iloc[part.train_idx].reset_index(drop=True)
        X_holdout_full = X_all.iloc[part.test_idx].reset_index(drop=True)
        y_train = np.asarray(y_all, dtype=int)[part.train_idx]
        y_holdout = np.asarray(y_all, dtype=int)[part.test_idx]
        part_models = sorted(
            {
                str(model)
                for class_id in class_order
                for model in common_support.get((partition_id, class_id), [])
            }
        )
        for model_name in part_models:
            fingerprint = credal_unit_fingerprint(
                "training_order_deletion",
                part,
                model_name,
                None,
                None,
                data_digest,
                method_digest,
            )
            cached = (
                load_credal_unit_checkpoint(
                    checkpoint_root,
                    "training_order_deletion",
                    part,
                    model_name,
                    None,
                    None,
                    fingerprint,
                )
                if ENABLE_SHADOW_UNIT_CHECKPOINTS
                else None
            )
            if cached is not None:
                cached_rows = cached.get("rows", pd.DataFrame())
                cached_errors = cached.get("errors", pd.DataFrame())
                if isinstance(cached_rows, pd.DataFrame) and not cached_rows.empty:
                    raw_rows.append(cached_rows)
                if (
                    isinstance(cached_errors, pd.DataFrame)
                    and not cached_errors.empty
                ):
                    error_rows.append(cached_errors)
                continue

            cprint(
                f"[ORDER VALIDATION] outer={partition_id} model={model_name}"
            )
            fit_cache: Dict[Tuple[str, ...], Dict[str, Any]] = {}

            def evaluate_feature_set(retained: Sequence[str]) -> Dict[str, Any]:
                key = tuple(str(feature) for feature in retained)
                if key in fit_cache:
                    return fit_cache[key]
                reduced_schema = _reduced_feature_schema(schema, key)
                fit_seed = stable_seed_from_key(
                    "utility_refit", partition_id, model_name
                )
                estimator, _, proba, pred = fit_fixed_predictive_specification(
                    X_train_raw=X_train_full.loc[:, list(key)],
                    X_eval_raw=X_holdout_full.loc[:, list(key)],
                    y_train=y_train,
                    schema=reduced_schema,
                    model_name=model_name,
                    fit_seed=fit_seed,
                    class_order=class_order,
                )
                metrics = evaluate_predictions(
                    y_holdout, pred, labels=class_order
                )
                class_brier = {
                    int(class_id): class_balanced_one_vs_rest_brier(
                        y_holdout,
                        proba[:, class_pos],
                        int(class_id),
                    )
                    for class_pos, class_id in enumerate(class_order)
                }
                result = {
                    "metrics": metrics,
                    "class_brier": class_brier,
                }
                fit_cache[key] = result
                del estimator
                clear_cuda_cache()
                gc.collect()
                return result

            model_rows = []
            model_errors = []
            for class_id in class_order:
                if model_name not in set(
                    common_support.get((partition_id, int(class_id)), [])
                ):
                    continue
                order = training_order_summary.loc[
                    (
                        pd.to_numeric(
                            training_order_summary["outer_partition"],
                            errors="coerce",
                        )
                        == partition_id
                    )
                    & (
                        pd.to_numeric(
                            training_order_summary["class_id"], errors="coerce"
                        )
                        == int(class_id)
                    )
                ].copy()
                if set(order["feature"].astype(str)) != set(schema.feature_cols):
                    raise RuntimeError(
                        "Training-only partial order lacks the complete feature set: "
                        f"outer={partition_id}, class={class_id}."
                    )
                level_groups = {
                    int(level): sorted(group["feature"].astype(str).tolist())
                    for level, group in order.groupby("partial_order_level")
                }
                baseline = evaluate_feature_set(schema.feature_cols)
                baseline_brier = float(baseline["class_brier"][int(class_id)])

                for direction, ordered_levels in [
                    ("top_down", sorted(level_groups)),
                    ("bottom_up", sorted(level_groups, reverse=True)),
                ]:
                    removed: Set[str] = set()
                    model_rows.append(
                        {
                            "outer_partition": partition_id,
                            "outer_repeat": int(part.repeat_id),
                            "outer_fold": int(part.fold_id),
                            "model": model_name,
                            "class_id": int(class_id),
                            "deletion_direction": direction,
                            "deletion_step": 0,
                            "removed_through_level": 0,
                            "n_removed_features": 0,
                            "removed_feature_fraction": 0.0,
                            "retained_features": "|".join(schema.feature_cols),
                            "removed_features": "",
                            "class_balanced_brier": baseline_brier,
                            "delta_class_balanced_brier": 0.0,
                            **baseline["metrics"],
                        }
                    )
                    for step, level in enumerate(ordered_levels, 1):
                        removed.update(level_groups[int(level)])
                        retained = [
                            feature
                            for feature in schema.feature_cols
                            if feature not in removed
                        ]
                        if not retained:
                            break
                        try:
                            result = evaluate_feature_set(retained)
                            brier = float(
                                result["class_brier"][int(class_id)]
                            )
                            model_rows.append(
                                {
                                    "outer_partition": partition_id,
                                    "outer_repeat": int(part.repeat_id),
                                    "outer_fold": int(part.fold_id),
                                    "model": model_name,
                                    "class_id": int(class_id),
                                    "deletion_direction": direction,
                                    "deletion_step": int(step),
                                    "removed_through_level": int(level),
                                    "n_removed_features": int(len(removed)),
                                    "removed_feature_fraction": float(
                                        len(removed) / len(schema.feature_cols)
                                    ),
                                    "retained_features": "|".join(retained),
                                    "removed_features": "|".join(
                                        sorted(removed)
                                    ),
                                    "class_balanced_brier": brier,
                                    "delta_class_balanced_brier": float(
                                        brier - baseline_brier
                                    ),
                                    **result["metrics"],
                                }
                            )
                        except Exception as exc:
                            model_errors.append(
                                {
                                    "outer_partition": partition_id,
                                    "model": model_name,
                                    "class_id": int(class_id),
                                    "deletion_direction": direction,
                                    "deletion_step": int(step),
                                    "removed_features": "|".join(
                                        sorted(removed)
                                    ),
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
            model_rows_df = pd.DataFrame(model_rows)
            model_errors_df = pd.DataFrame(model_errors)
            if not model_rows_df.empty:
                raw_rows.append(model_rows_df)
            if not model_errors_df.empty:
                error_rows.append(model_errors_df)
            if (
                ENABLE_SHADOW_UNIT_CHECKPOINTS
                and not model_rows_df.empty
                and model_errors_df.empty
            ):
                save_credal_unit_checkpoint(
                    checkpoint_root,
                    "training_order_deletion",
                    part,
                    model_name,
                    None,
                    None,
                    fingerprint,
                    {"rows": model_rows_df, "errors": model_errors_df},
                )

    raw = concat_or_empty(raw_rows)
    errors = concat_or_empty(error_rows)
    if raw.empty:
        return raw, pd.DataFrame(), errors

    auc_rows = []
    for keys, group in raw.groupby(
        [
            "outer_partition",
            "outer_repeat",
            "outer_fold",
            "model",
            "class_id",
            "deletion_direction",
        ]
    ):
        ordered = group.sort_values(
            ["removed_feature_fraction", "deletion_step"]
        ).drop_duplicates("removed_feature_fraction", keep="last")
        x_values = ordered["removed_feature_fraction"].to_numpy(dtype=float)
        y_values = ordered["delta_class_balanced_brier"].to_numpy(dtype=float)
        max_fraction = float(np.max(x_values))
        if len(x_values) < 2 or max_fraction <= 0:
            normalized_auc = np.nan
        else:
            trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            normalized_auc = float(
                trapezoid(y_values, x_values) / max_fraction
            )
        auc_rows.append(
            {
                "outer_partition": int(keys[0]),
                "outer_repeat": int(keys[1]),
                "outer_fold": int(keys[2]),
                "model": str(keys[3]),
                "class_id": int(keys[4]),
                "deletion_direction": str(keys[5]),
                "normalized_observed_range_auc_delta_brier": normalized_auc,
                "max_removed_feature_fraction": max_fraction,
                "n_refitted_feature_sets": int(len(ordered)),
                "n_holdout": int(
                    len(
                        next(
                            part.test_idx
                            for part in partitions
                            if int(part.partition_id) == int(keys[0])
                        )
                    )
                ),
            }
        )
    fold_auc = pd.DataFrame(auc_rows)

    repeat_rows = []
    expected_outer_folds_by_repeat = {
        int(repeat_id): int(
            len(
                {
                    int(part.fold_id)
                    for part in partitions
                    if int(part.repeat_id) == int(repeat_id)
                }
            )
        )
        for repeat_id in {int(part.repeat_id) for part in partitions}
    }
    for (repeat_id, class_id, direction), repeat_group in fold_auc.groupby(
        ["outer_repeat", "class_id", "deletion_direction"]
    ):
        fold_values = []
        for fold_id, fold_group in repeat_group.groupby("outer_fold"):
            values = pd.to_numeric(
                fold_group["normalized_observed_range_auc_delta_brier"],
                errors="coerce",
            ).dropna()
            if values.empty:
                continue
            weight = float(fold_group["n_holdout"].iloc[0])
            fold_values.append(
                {
                    "fold": int(fold_id),
                    "weight": weight,
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )
        expected_fold_count = int(
            expected_outer_folds_by_repeat.get(int(repeat_id), 0)
        )
        if len(fold_values) != expected_fold_count:
            raise RuntimeError(
                "Training-order utility validation is incomplete within an "
                f"outer repeat: repeat={repeat_id}, class={class_id}, "
                f"direction={direction}, observed_folds={len(fold_values)}, "
                f"expected_folds={expected_fold_count}."
            )
        total_weight = float(sum(item["weight"] for item in fold_values))
        repeat_rows.append(
            {
                "outer_repeat": int(repeat_id),
                "class_id": int(class_id),
                "deletion_direction": str(direction),
                "lower_model_path_auc_delta_brier": float(
                    sum(
                        item["weight"] * item["minimum"]
                        for item in fold_values
                    )
                    / total_weight
                ),
                "upper_model_path_auc_delta_brier": float(
                    sum(
                        item["weight"] * item["maximum"]
                        for item in fold_values
                    )
                    / total_weight
                ),
                "n_outer_folds": int(len(fold_values)),
                "aggregation_rule": (
                    "holdout-size weighted aggregation across folds within "
                    "repeat; model-path extrema taken within each fold"
                ),
            }
        )
    return raw, pd.DataFrame(repeat_rows), errors


def save_credal_interval_figures(
    summary: pd.DataFrame,
    figure_dir: Path,
) -> None:
    if not OUTPUT_FIGURES or not _has_matplotlib() or summary.empty:
        return
    import matplotlib.pyplot as plt

    _setup_publication_style()
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for class_id, group in summary.groupby("class_id"):
        ordered = group.sort_values(
            [
                "partial_order_level",
                "lower_expected_percentile",
                "feature",
            ],
            ascending=[True, False, True],
        ).reset_index(drop=True)
        y_pos = np.arange(len(ordered))
        lower = ordered["lower_expected_percentile"].to_numpy(dtype=float)
        upper = ordered["upper_expected_percentile"].to_numpy(dtype=float)
        maximal = ordered["in_maximal_set"].astype(int).to_numpy() == 1
        fig, ax = plt.subplots(
            figsize=(7.4, max(4.8, 0.38 * len(ordered) + 1.2))
        )
        ax.hlines(
            y_pos,
            lower,
            upper,
            color=FIG_COLORS["light_gray"],
            linewidth=3.0,
            zorder=1,
        )
        ax.scatter(
            upper,
            y_pos,
            marker="|",
            s=72,
            color=FIG_COLORS["gray"],
            zorder=2,
        )
        ax.scatter(
            lower[~maximal],
            y_pos[~maximal],
            s=43,
            facecolor="white",
            edgecolor=FIG_CLASS_COLORS.get(int(class_id), FIG_COLORS["blue"]),
            linewidth=1.2,
            zorder=3,
            label="Nonmaximal",
        )
        ax.scatter(
            lower[maximal],
            y_pos[maximal],
            s=55,
            color=FIG_CLASS_COLORS.get(int(class_id), FIG_COLORS["blue"]),
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
            label="Maximal set",
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [
                f"L{int(level)}  {_feature_label(feature)}"
                for level, feature in zip(
                    ordered["partial_order_level"], ordered["feature"]
                )
            ]
        )
        ax.invert_yaxis()
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("Replacement-shadow relevance percentile")
        ax.set_title(_class_label(class_id))
        ax.legend(frameon=False, loc="lower right")
        _clean_axis(ax, grid_axis="x")
        _save_paper_figure(
            fig,
            figure_dir
            / f"credal_relevance_intervals_class_{int(class_id)}",
        )
        plt.close(fig)


def save_credal_hasse_figures(
    summary: pd.DataFrame,
    hasse: pd.DataFrame,
    figure_dir: Path,
) -> None:
    if not OUTPUT_FIGURES or not _has_matplotlib() or summary.empty:
        return
    import matplotlib.pyplot as plt

    _setup_publication_style()
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for class_id, group in summary.groupby("class_id"):
        class_hasse = hasse.loc[
            pd.to_numeric(hasse["class_id"], errors="coerce") == int(class_id)
        ].copy() if not hasse.empty else pd.DataFrame()
        positions: Dict[str, Tuple[float, float]] = {}
        level_groups = {
            int(level): sorted(level_group["feature"].astype(str).tolist())
            for level, level_group in group.groupby("partial_order_level")
        }
        max_width = max(len(features) for features in level_groups.values())
        for level, features in sorted(level_groups.items()):
            offsets = np.linspace(
                -(len(features) - 1) / 2.0,
                (len(features) - 1) / 2.0,
                len(features),
            )
            for offset, feature in zip(offsets, features):
                positions[feature] = (float(offset), float(-level))

        fig, ax = plt.subplots(
            figsize=(
                max(7.2, 1.35 * max_width + 2.0),
                max(4.8, 1.15 * len(level_groups) + 1.5),
            )
        )
        if not class_hasse.empty:
            for edge in class_hasse.itertuples(index=False):
                x0, y0 = positions[str(edge.source_feature)]
                x1, y1 = positions[str(edge.target_feature)]
                ax.annotate(
                    "",
                    xy=(x1, y1 + 0.16),
                    xytext=(x0, y0 - 0.16),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": FIG_COLORS["gray"],
                        "linewidth": 1.0,
                        "shrinkA": 10,
                        "shrinkB": 10,
                    },
                    zorder=1,
                )
        maximal = set(
            group.loc[group["in_maximal_set"].astype(int) == 1, "feature"].astype(str)
        )
        for feature, (x_coord, y_coord) in positions.items():
            is_maximal = feature in maximal
            ax.text(
                x_coord,
                y_coord,
                _feature_label(feature, max_len=22),
                ha="center",
                va="center",
                fontsize=8.0,
                color="white" if is_maximal else FIG_COLORS["dark"],
                bbox={
                    "boxstyle": "round,pad=0.34,rounding_size=0.06",
                    "facecolor": (
                        FIG_CLASS_COLORS.get(
                            int(class_id), FIG_COLORS["blue"]
                        )
                        if is_maximal
                        else "white"
                    ),
                    "edgecolor": FIG_CLASS_COLORS.get(
                        int(class_id), FIG_COLORS["blue"]
                    ),
                    "linewidth": 1.2,
                },
                zorder=4,
            )
        ax.set_title(f"{_class_label(class_id)}: strict credal Hasse diagram")
        ax.set_xlim(
            -max(1.2, max_width / 2.0 + 0.6),
            max(1.2, max_width / 2.0 + 0.6),
        )
        ax.set_ylim(-max(level_groups) - 0.7, -0.3)
        ax.axis("off")
        _save_paper_figure(
            fig, figure_dir / f"credal_hasse_class_{int(class_id)}"
        )
        plt.close(fig)


def save_credal_diagnostic_figures(
    coverage: pd.DataFrame,
    calibrated: pd.DataFrame,
    source_sensitivity: pd.DataFrame,
    deletion_raw: pd.DataFrame,
    figure_dir: Path,
) -> None:
    if not OUTPUT_FIGURES or not _has_matplotlib():
        return
    import matplotlib.pyplot as plt

    _setup_publication_style()
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if coverage is not None and not coverage.empty:
        pivot = coverage.pivot(
            index="outer_partition", columns="class_id", values="coverage"
        ).sort_index()
        fig, ax = plt.subplots(figsize=(6.6, 5.2))
        image = ax.imshow(
            pivot.to_numpy(dtype=float),
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            cmap="YlGnBu",
        )
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([_class_label(value) for value in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([str(value) for value in pivot.index])
        ax.set_xlabel("Injury class")
        ax.set_ylabel("Outer partition")
        colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
        colorbar.set_label("Common-support coverage")
        _save_paper_figure(fig, figure_dir / "common_support_coverage")
        plt.close(fig)

    if calibrated is not None and not calibrated.empty:
        class_ids = sorted(calibrated["class_id"].astype(int).unique())
        fig, axes = plt.subplots(
            1,
            len(class_ids),
            figsize=(4.7 * max(len(class_ids), 1), 4.6),
            squeeze=False,
            sharey=True,
        )
        for axis_index, class_id in enumerate(class_ids):
            ax = axes[0, axis_index]
            class_frame = calibrated.loc[
                calibrated["class_id"].astype(int) == int(class_id)
            ]
            data = [
                class_frame.loc[
                    class_frame["source"].astype(str) == source,
                    "relevance_percentile",
                ].to_numpy(dtype=float)
                for source in CREDAL_SOURCE_NAMES
            ]
            try:
                plot = ax.boxplot(
                    data,
                    tick_labels=list(CREDAL_SOURCE_NAMES),
                    patch_artist=True,
                    showfliers=False,
                )
            except TypeError:
                plot = ax.boxplot(
                    data,
                    labels=list(CREDAL_SOURCE_NAMES),
                    patch_artist=True,
                    showfliers=False,
                )
            for patch in plot["boxes"]:
                patch.set_facecolor(
                    FIG_CLASS_COLORS.get(int(class_id), FIG_COLORS["blue"])
                )
                patch.set_alpha(0.55)
            ax.set_title(_class_label(class_id))
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlabel("Explanation source")
            if axis_index == 0:
                ax.set_ylabel("Replacement-shadow relevance percentile")
            _clean_axis(ax, grid_axis="y")
        _save_paper_figure(fig, figure_dir / "relevance_percentiles_by_source")
        plt.close(fig)

    if source_sensitivity is not None and not source_sensitivity.empty:
        pivot = source_sensitivity.pivot(
            index="omitted_source",
            columns="class_id",
            values="maximal_set_jaccard",
        ).reindex(index=list(CREDAL_SOURCE_NAMES))
        fig, ax = plt.subplots(figsize=(6.8, 3.8))
        image = ax.imshow(
            pivot.to_numpy(dtype=float),
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            cmap="Blues",
        )
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([_class_label(value) for value in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Injury class")
        ax.set_ylabel("Omitted source")
        colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
        colorbar.set_label("Maximal-set Jaccard similarity")
        _save_paper_figure(fig, figure_dir / "leave_one_source_out_sensitivity")
        plt.close(fig)

    if deletion_raw is not None and not deletion_raw.empty:
        colors = {
            "top_down": FIG_COLORS["coral"],
            "bottom_up": FIG_COLORS["blue"],
        }
        class_ids = sorted(deletion_raw["class_id"].astype(int).unique())
        fig, axes = plt.subplots(
            1,
            len(class_ids),
            figsize=(4.8 * max(len(class_ids), 1), 4.5),
            squeeze=False,
            sharey=True,
        )
        for axis_index, class_id in enumerate(class_ids):
            ax = axes[0, axis_index]
            class_frame = deletion_raw.loc[
                deletion_raw["class_id"].astype(int) == int(class_id)
            ]
            for direction in ["top_down", "bottom_up"]:
                direction_frame = class_frame.loc[
                    class_frame["deletion_direction"].astype(str) == direction
                ]
                summary = (
                    direction_frame.groupby(
                        "removed_feature_fraction", as_index=False
                    )["delta_class_balanced_brier"]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                    .sort_values("removed_feature_fraction")
                )
                if summary.empty:
                    continue
                x_values = summary["removed_feature_fraction"].to_numpy(
                    dtype=float
                )
                y_values = summary["mean"].to_numpy(dtype=float)
                ax.plot(
                    x_values,
                    y_values,
                    marker="o",
                    linewidth=1.5,
                    color=colors[direction],
                    label=direction.replace("_", " "),
                )
            ax.axhline(0.0, color=FIG_COLORS["gray"], linewidth=0.8)
            ax.set_title(_class_label(class_id))
            ax.set_xlabel("Fraction of features removed by whole order layers")
            if axis_index == 0:
                ax.set_ylabel("Change in class-balanced Brier loss")
            _clean_axis(ax, grid_axis="both")
        axes[0, -1].legend(frameon=False, loc="best")
        _save_paper_figure(fig, figure_dir / "training_order_deletion_validation")
        plt.close(fig)


def run_credal_shadow_workflow(
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    schema: FeatureSchema,
    partitions: Sequence[OuterPartition],
    admissible_models_by_partition: Mapping[int, Sequence[str]],
    output_dir: Path,
) -> Dict[str, Any]:
    """Execute the frozen credal method and export reviewer-auditable artifacts."""
    tables_dir = Path(output_dir) / "tables"
    figures_dir = Path(output_dir) / "figures" / "credal_shadow"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    class_order = [0, 1, 2]
    n_shadow = int(shadow_replicate_count(RUN_PROFILE))

    cprint("\n" + "=" * 110)
    cprint("STEP 2: CREDAL SHADOW-CALIBRATED EXPLANATION EVIDENCE")
    cprint("=" * 110)
    evidence = execute_credal_evidence_contract(
        X_all=X_all,
        y_all=y_all,
        schema=schema,
        partitions=partitions,
        admissible_models_by_partition=admissible_models_by_partition,
        output_dir=output_dir,
    )
    raw_paths = {
        "outer_statistics": "outer_holdout_source_statistics_raw.csv",
        "training_statistics": "training_crossfit_source_statistics_raw.csv",
        "shadow_statistics": "replacement_shadow_statistics_raw.csv",
        "shap_full_display": "shap_full_display_statistics.csv",
        "shap_display_target_concordance": (
            "shap_official_display_target_concordance.csv"
        ),
        "outer_metrics": "admissible_model_holdout_metrics_raw.csv",
        "outer_predictions": "admissible_model_holdout_predictions_raw.csv",
        "outer_calibration": "admissible_model_calibration_curves_raw.csv",
        "pdp": "descriptive_pdp_profiles_raw.csv",
        "ice": "descriptive_centered_ice_raw.csv",
        "errors": "credal_execution_errors.csv",
    }
    for key, filename in raw_paths.items():
        frame = evidence.get(key, pd.DataFrame())
        frame.to_csv(tables_dir / filename, index=False, encoding="utf-8-sig")
    target_audit_root = (
        Path(output_dir) / "checkpoints" / "target_shap_audit"
    )
    hard_audits = [
        pd.read_csv(path)
        for path in sorted(
            target_audit_root.glob("*_target_shap_equivalence.csv")
        )
    ]
    plan_audits = [
        pd.read_csv(path)
        for path in sorted(target_audit_root.glob("*_plan_sensitivity.csv"))
    ]
    concat_or_empty(hard_audits).to_csv(
        tables_dir / "shap_target_equivalence_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    concat_or_empty(plan_audits).to_csv(
        tables_dir / "shap_target_common_plan_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for required_key in [
        "outer_statistics",
        "training_statistics",
        "shadow_statistics",
    ]:
        if evidence[required_key].empty:
            raise RuntimeError(
                f"The primary credal workflow produced no {required_key}."
            )

    validity, coverage, common_support = build_credal_common_support(
        outer_statistics=evidence["outer_statistics"],
        training_statistics=evidence["training_statistics"],
        shadow_statistics=evidence["shadow_statistics"],
        partitions=partitions,
        admissible_models_by_partition=admissible_models_by_partition,
        feature_cols=schema.feature_cols,
        class_order=class_order,
        n_shadow_replicates=n_shadow,
        expected_inner_folds_by_partition={
            int(part.partition_id): int(
                len(
                    make_inner_fold_specs(
                        np.asarray(y_all, dtype=int)[part.train_idx],
                        part,
                    )
                )
            )
            for part in partitions
        },
    )
    validity.to_csv(
        tables_dir / "credal_specification_validity_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    coverage.to_csv(
        tables_dir / "credal_common_support_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    empty_support = coverage.loc[
        coverage["common_support_nonempty"].astype(int) != 1
    ]
    if not empty_support.empty and FAIL_IF_ANY_PRIMARY_COMMON_SUPPORT_EMPTY:
        raise RuntimeError(
            "Primary credal inference stopped because at least one outer "
            "partition/class has empty common support. The failure audit was "
            "written to credal_specification_validity_audit.csv. Empty cells: "
            f"{empty_support[['outer_partition', 'class_id']].to_dict('records')}"
        )

    calibrated_outer = calibrate_shadow_relevance_percentiles(
        observed_statistics=evidence["outer_statistics"],
        shadow_statistics=evidence["shadow_statistics"],
        common_support=common_support,
        role="outer_holdout_inference",
        n_shadow_replicates=n_shadow,
    )
    calibrated_training = calibrate_shadow_relevance_percentiles(
        observed_statistics=evidence["training_statistics"],
        shadow_statistics=evidence["shadow_statistics"],
        common_support=common_support,
        role="outer_training_order_construction",
        n_shadow_replicates=n_shadow,
    )
    expected_calibrated_rows = int(
        sum(
            len(common_support[(int(part.partition_id), int(class_id))])
            for part in partitions
            for class_id in class_order
        )
        * len(CREDAL_SOURCE_NAMES)
        * len(schema.feature_cols)
    )
    for label, frame in [
        ("outer-holdout", calibrated_outer),
        ("training-crossfit", calibrated_training),
    ]:
        if len(frame) != expected_calibrated_rows:
            raise RuntimeError(
                f"{label} calibration coverage mismatch: expected "
                f"{expected_calibrated_rows}, observed {len(frame)}."
            )
    calibrated_outer.to_csv(
        tables_dir / "outer_holdout_shadow_calibrated_relevance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    calibrated_training.to_csv(
        tables_dir / "training_crossfit_shadow_calibrated_relevance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    primary_summary, primary_pairs, primary_hasse = compute_credal_partial_order(
        calibrated=calibrated_outer,
        partitions=partitions,
        common_support=common_support,
        feature_cols=schema.feature_cols,
        class_order=class_order,
        analysis_label="primary_four_source_common_support",
    )
    primary_summary.to_csv(
        tables_dir / "credal_feature_intervals_and_partial_order.csv",
        index=False,
        encoding="utf-8-sig",
    )
    primary_pairs.to_csv(
        tables_dir / "credal_pairwise_lower_expectations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    primary_hasse.to_csv(
        tables_dir / "credal_hasse_edges.csv",
        index=False,
        encoding="utf-8-sig",
    )

    training_summary, training_pairs = compute_local_training_partial_orders(
        calibrated_training=calibrated_training,
        common_support=common_support,
        partitions=partitions,
        feature_cols=schema.feature_cols,
        class_order=class_order,
    )
    training_summary.to_csv(
        tables_dir / "training_only_partial_orders_by_outer_partition.csv",
        index=False,
        encoding="utf-8-sig",
    )
    training_pairs.to_csv(
        tables_dir / "training_only_pairwise_preferences.csv",
        index=False,
        encoding="utf-8-sig",
    )

    source_sensitivity, source_feature_sensitivity, repeat_consistency = (
        compute_credal_sensitivity_diagnostics(
            calibrated_outer=calibrated_outer,
            partitions=partitions,
            common_support=common_support,
            feature_cols=schema.feature_cols,
            class_order=class_order,
            primary_summary=primary_summary,
            primary_hasse=primary_hasse,
        )
    )
    source_sensitivity.to_csv(
        tables_dir / "leave_one_explainer_out_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    source_feature_sensitivity.to_csv(
        tables_dir / "leave_one_explainer_out_feature_intervals.csv",
        index=False,
        encoding="utf-8-sig",
    )
    repeat_consistency.to_csv(
        tables_dir / "repeat_specific_hasse_edge_consistency.csv",
        index=False,
        encoding="utf-8-sig",
    )

    deletion_raw, deletion_repeat, deletion_errors = (
        run_training_order_deletion_validation(
            X_all=X_all,
            y_all=y_all,
            schema=schema,
            partitions=partitions,
            common_support=common_support,
            training_order_summary=training_summary,
            output_dir=output_dir,
        )
    )
    deletion_raw.to_csv(
        tables_dir / "training_order_deletion_validation_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    deletion_repeat.to_csv(
        tables_dir / "training_order_deletion_validation_by_repeat.csv",
        index=False,
        encoding="utf-8-sig",
    )
    deletion_errors.to_csv(
        tables_dir / "training_order_deletion_validation_errors.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if (
        FAIL_IF_UTILITY_VALIDATION_INCOMPLETE
        and deletion_errors is not None
        and not deletion_errors.empty
    ):
        raise RuntimeError(
            "Training-only deletion validation failed for one or more fixed "
            "model paths. See training_order_deletion_validation_errors.csv; "
            "partial utility evidence is not accepted."
        )

    structural_meta: Dict[str, Any] = {
        "enabled": False,
        "status": "disabled_by_config",
    }
    if ENABLE_STRUCTURAL_COMPATIBILITY_AUDIT:
        structural_meta = run_structural_compatibility_audit(
            X_all=X_all,
            y_all=y_all,
            feature_cols=list(schema.feature_cols),
            partitions=list(partitions),
            reliability_df=primary_summary,
            output_dir=output_dir,
        )

    save_credal_interval_figures(primary_summary, figures_dir)
    save_credal_hasse_figures(primary_summary, primary_hasse, figures_dir)
    save_credal_diagnostic_figures(
        coverage=coverage,
        calibrated=calibrated_outer,
        source_sensitivity=source_sensitivity,
        deletion_raw=deletion_raw,
        figure_dir=figures_dir,
    )
    if OUTPUT_FIGURES:
        holdout_metric_summary = (
            summarize_metrics(
                evidence["outer_metrics"],
                group_cols=["model", "model_display"],
            )
            if not evidence["outer_metrics"].empty
            else pd.DataFrame()
        )
        save_prediction_figures(
            metrics_raw=evidence["outer_metrics"],
            metrics_summary=holdout_metric_summary,
            calibration_df=evidence["outer_calibration"],
            figs_dir=Path(output_dir) / "figures" / "admissible_models",
        )
        display_summary = primary_summary.copy()
        display_summary["consensus_importance"] = display_summary[
            "lower_expected_percentile"
        ]
        display_summary["class_rank_fused"] = display_summary[
            "partial_order_level"
        ]
        if not evidence["pdp"].empty:
            save_pdp_profile_figures(
                evidence["pdp"],
                display_summary,
                Path(output_dir) / "figures",
                top_k=DISPLAY_TOP_K_FEATURES,
            )
        if not evidence["ice"].empty:
            save_ice_heterogeneity_figures(
                evidence["ice"],
                display_summary,
                Path(output_dir) / "figures",
                top_k=DISPLAY_TOP_K_FEATURES,
            )

    return {
        "primary_summary": primary_summary,
        "primary_pairs": primary_pairs,
        "primary_hasse": primary_hasse,
        "common_support_coverage": coverage,
        "validity_audit": validity,
        "outer_calibrated": calibrated_outer,
        "training_calibrated": calibrated_training,
        "deletion_repeat": deletion_repeat,
        "structural_compatibility_audit": structural_meta,
    }


# =============================================================================
# WORKFLOW
# =============================================================================


def concat_or_empty(frames: List[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


def build_explanation_coverage_audit(
    holdout_metrics: pd.DataFrame,
    source_specs: Dict[str, Tuple[pd.DataFrame, str, Optional[List[str]]]],
    expected_classes: List[int],
) -> pd.DataFrame:
    """Verify row, class, and feature coverage for every explanation run."""
    if holdout_metrics is None or holdout_metrics.empty:
        raise RuntimeError(
            "Holdout metrics are required for the explanation coverage audit."
        )

    expected = (
        holdout_metrics[["seed", "model", "n_holdout"]]
        .drop_duplicates()
        .sort_values(["seed", "model"])
        .reset_index(drop=True)
    )
    rows = []
    expected_class_set = {int(v) for v in expected_classes}
    for source_name, (table, count_column, expected_features) in source_specs.items():
        expected_feature_set = (
            {str(v) for v in expected_features}
            if expected_features is not None
            else None
        )
        for record in expected.itertuples(index=False):
            if table is None or table.empty:
                subset = pd.DataFrame()
            else:
                subset = table.loc[
                    (table["seed"].astype(int) == int(record.seed))
                    & (table["model"].astype(str) == str(record.model))
                ]
            if subset.empty or count_column not in subset.columns:
                observed_counts: List[int] = []
            else:
                observed_counts = sorted(
                    pd.to_numeric(subset[count_column], errors="coerce")
                    .dropna()
                    .astype(int)
                    .unique()
                    .tolist()
                )
            full = observed_counts == [int(record.n_holdout)]
            observed_class_set = (
                set(
                    pd.to_numeric(subset["class_id"], errors="coerce")
                    .dropna()
                    .astype(int)
                )
                if (not subset.empty and "class_id" in subset.columns)
                else set()
            )
            observed_feature_set = (
                set(subset["feature"].astype(str))
                if (not subset.empty and "feature" in subset.columns)
                else set()
            )
            class_coverage = observed_class_set == expected_class_set
            feature_coverage = (
                True
                if expected_feature_set is None
                else observed_feature_set == expected_feature_set
            )
            rows.append(
                {
                    "seed": int(record.seed),
                    "model": str(record.model),
                    "source": str(source_name),
                    "expected_holdout_rows": int(record.n_holdout),
                    "observed_evaluation_rows": (
                        int(observed_counts[0]) if len(observed_counts) == 1 else np.nan
                    ),
                    "n_output_records": int(len(subset)),
                    "source_available": int(not subset.empty),
                    "full_holdout_confirmed": int(full),
                    "expected_class_count": int(len(expected_class_set)),
                    "observed_class_count": int(len(observed_class_set)),
                    "all_classes_confirmed": int(class_coverage),
                    "expected_feature_count": (
                        int(len(expected_feature_set))
                        if expected_feature_set is not None
                        else np.nan
                    ),
                    "observed_feature_count": int(len(observed_feature_set)),
                    "all_expected_features_confirmed": int(feature_coverage),
                    "coverage_confirmed": int(
                        full and class_coverage and feature_coverage
                    ),
                    "observed_count_values": ",".join(str(v) for v in observed_counts),
                }
            )

    audit = pd.DataFrame(rows)
    failed = audit.loc[audit["coverage_confirmed"].astype(int) != 1]
    if not failed.empty:
        details = failed[
            [
                "seed",
                "model",
                "source",
                "observed_count_values",
                "observed_class_count",
                "observed_feature_count",
            ]
        ].to_dict("records")
        raise RuntimeError(
            "Explanation coverage audit failed. Every enabled source must be available, "
            "use the complete outer holdout partition, and cover its expected classes "
            f"and features. Failures: {details}"
        )
    return audit


def select_best_model(metrics_summary: pd.DataFrame) -> str:
    if metrics_summary.empty:
        raise RuntimeError("No model metrics were produced.")
    row = metrics_summary.iloc[0]
    return str(row["model"])


def select_explanation_models_by_seed(
    metrics_by_split: pd.DataFrame,
    seeds: List[int],
    max_models: Optional[int] = None,
) -> Dict[int, List[str]]:
    """Return the split-specific one-standard-error model sets."""
    if metrics_by_split is None or metrics_by_split.empty:
        raise RuntimeError(
            "Inner cross-validation summaries are required for model selection."
        )
    selected: Dict[int, List[str]] = {}
    for seed in seeds:
        group = metrics_by_split.loc[
            metrics_by_split["seed"].astype(int) == int(seed)
        ].copy()
        if group.empty:
            raise RuntimeError(
                f"No inner cross-validation summary was produced for seed={seed}."
            )
        if "selected_by_one_se" not in group.columns:
            raise RuntimeError(
                "The inner cross-validation table lacks one-SE selection flags."
            )
        group = group.loc[group["selected_by_one_se"].astype(int) == 1].copy()
        if group.empty:
            raise RuntimeError(f"The one-SE rule selected no model for seed={seed}.")
        group = group.sort_values(
            ["mean_balanced_accuracy", "mean_macro_f1", "mean_accuracy", "model"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        if max_models is not None:
            group = group.head(int(max_models))
        selected[int(seed)] = group["model"].astype(str).drop_duplicates().tolist()
    return selected


def summarize_inner_cv_by_split(metrics_raw: pd.DataFrame) -> pd.DataFrame:
    if metrics_raw is None or metrics_raw.empty:
        return pd.DataFrame()
    excluded = {
        "seed",
        "inner_fold",
        "model",
        "model_display",
        "n_subtrain",
        "n_validation",
        "evaluation_partition",
        "outer_repeat",
        "outer_fold",
        "split_seed",
    }
    metric_cols = [
        col
        for col in metrics_raw.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(metrics_raw[col])
    ]
    rows = []
    for (seed, model, display), group in metrics_raw.groupby(
        ["seed", "model", "model_display"], dropna=False
    ):
        row = {
            "seed": int(seed),
            "model": str(model),
            "model_display": str(display),
            "n_folds": int(group["inner_fold"].nunique()),
            "outer_repeat": int(group["outer_repeat"].iloc[0])
            if "outer_repeat" in group
            else np.nan,
            "outer_fold": int(group["outer_fold"].iloc[0])
            if "outer_fold" in group
            else np.nan,
            "split_seed": int(group["split_seed"].iloc[0])
            if "split_seed" in group
            else np.nan,
        }
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"mean_{metric}"] = float(values.mean()) if len(values) else np.nan
            row[f"std_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"se_{metric}"] = (
                float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
        rows.append(row)

    out = pd.DataFrame(rows)
    selected_parts = []
    for seed, group in out.groupby("seed"):
        group = group.copy()
        best = group.sort_values(
            ["mean_balanced_accuracy", "mean_macro_f1", "mean_accuracy", "model"],
            ascending=[False, False, False, True],
        ).iloc[0]
        cutoff = float(best["mean_balanced_accuracy"] - best["se_balanced_accuracy"])
        group["one_se_reference_model"] = str(best["model"])
        group["one_se_cutoff_balanced_accuracy"] = cutoff
        group["selected_by_one_se"] = (
            group["mean_balanced_accuracy"].astype(float) >= cutoff - 1e-12
        ).astype(int)
        selected_parts.append(group)
    return (
        pd.concat(selected_parts, axis=0, ignore_index=True)
        .sort_values(
            ["seed", "selected_by_one_se", "mean_balanced_accuracy", "model"],
            ascending=[True, False, False, True],
        )
        .reset_index(drop=True)
    )


def summarize_model_selection_across_splits(
    metrics_by_split: pd.DataFrame,
) -> pd.DataFrame:
    if metrics_by_split is None or metrics_by_split.empty:
        return pd.DataFrame()
    mean_metric_cols = [
        col
        for col in metrics_by_split.columns
        if col.startswith("mean_")
        and pd.api.types.is_numeric_dtype(metrics_by_split[col])
    ]
    rows = []
    for (model, display), group in metrics_by_split.groupby(["model", "model_display"]):
        row = {
            "model": str(model),
            "model_display": str(display),
            "n_rep": int(group["seed"].nunique()),
            "one_se_selection_frequency": float(group["selected_by_one_se"].mean()),
        }
        for mean_col in mean_metric_cols:
            metric = mean_col.replace("mean_", "", 1)
            values = pd.to_numeric(group[mean_col], errors="coerce").dropna()
            row[f"mean_{metric}"] = float(values.mean()) if len(values) else np.nan
            row[f"std_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    return (
        out.sort_values(
            ["mean_balanced_accuracy", "mean_macro_f1", "mean_accuracy", "model"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
        .assign(rank_balanced_accuracy=lambda frame: np.arange(1, len(frame) + 1))
    )


def run_prediction_benchmark(
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    schema: FeatureSchema,
    partitions: List[OuterPartition],
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    from sklearn.model_selection import StratifiedKFold

    model_names = available_model_names(CANDIDATE_MODELS)
    if not model_names:
        raise RuntimeError("No candidate model is available.")

    metric_rows = []
    calibration_curve_rows = []
    class_order = [0, 1, 2]

    cprint("\n" + "=" * 110)
    cprint("STEP 1: INNER CROSS-VALIDATION MODEL SELECTION")
    cprint("=" * 110)
    cprint(f"models used: {model_names}")

    for part in partitions:
        seed = int(part.partition_id)
        X_outer_train = X_all.iloc[part.train_idx].reset_index(drop=True)
        y_outer_train = np.asarray(y_all, dtype=int)[part.train_idx]
        min_class_count = int(pd.Series(y_outer_train).value_counts().min())
        n_splits = int(min(INNER_CV_FOLDS, min_class_count))
        if n_splits < 2:
            raise RuntimeError(
                f"Insufficient class support for inner cross-validation at seed={seed}."
            )
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=stable_seed_from_key("inner_cv_split", part.split_seed),
        )

        for fold, (train_idx, validation_idx) in enumerate(
            cv.split(X_outer_train, y_outer_train), 1
        ):
            X_subtrain = X_outer_train.iloc[train_idx].reset_index(drop=True)
            X_validation = X_outer_train.iloc[validation_idx].reset_index(drop=True)
            y_subtrain = np.asarray(y_outer_train)[train_idx]
            y_validation = np.asarray(y_outer_train)[validation_idx]
            sample_weight = (
                make_balanced_sample_weight(y_subtrain)
                if USE_BALANCED_SAMPLE_WEIGHT
                else None
            )

            for model_name in model_names:
                cprint(
                    f"[BENCHMARK] seed={seed} fold={fold}/{n_splits} model={model_name}"
                )
                encoded = fit_transform_split(
                    X_subtrain, X_validation, schema, model_name=model_name
                )
                estimator = build_estimator(
                    model_name,
                    seed=stable_seed_from_key("inner_cv", seed, fold, model_name),
                    n_jobs=MODEL_N_JOBS,
                    categorical_feature_indices=encoded.categorical_feature_indices,
                )
                fit_estimator(
                    estimator, encoded.X_train, y_subtrain, sample_weight=sample_weight
                )
                proba = predict_proba_aligned(estimator, encoded.X_test, class_order)
                y_pred = np.asarray(class_order, dtype=int)[np.argmax(proba, axis=1)]
                metrics = evaluate_predictions(y_validation, y_pred, labels=class_order)
                if ENABLE_CALIBRATION_DIAGNOSTICS:
                    metrics.update(
                        compute_probability_diagnostics(
                            y_true=y_validation,
                            proba=proba,
                            labels=class_order,
                            n_bins=CALIBRATION_N_BINS,
                        )
                    )
                    calibration_part = compute_calibration_curve_rows(
                        y_true=y_validation,
                        proba=proba,
                        labels=class_order,
                        model_name=model_name,
                        seed=int(seed),
                        n_bins=CALIBRATION_N_BINS,
                    )
                    calibration_part["inner_fold"] = int(fold)
                    calibration_curve_rows.append(calibration_part)
                metric_rows.append(
                    {
                        "seed": int(seed),
                        "outer_repeat": int(part.repeat_id),
                        "outer_fold": int(part.fold_id),
                        "split_seed": int(part.split_seed),
                        "inner_fold": int(fold),
                        "model": model_name,
                        "model_display": MODEL_DISPLAY_NAMES[model_name],
                        "preprocessing_mode": preprocessing_mode(model_name),
                        "imbalance_handling": imbalance_handling_mode(model_name),
                        "n_subtrain": int(len(y_subtrain)),
                        "n_validation": int(len(y_validation)),
                        "evaluation_partition": "inner_cross_validation",
                        **metrics,
                    }
                )
                gc.collect()

    metrics_raw = pd.DataFrame(metric_rows)
    calibration_curves = concat_or_empty(calibration_curve_rows)
    metrics_by_split = summarize_inner_cv_by_split(metrics_raw)
    metrics_summary = summarize_model_selection_across_splits(metrics_by_split)
    metrics_raw.to_csv(
        output_dir / "tables" / "model_selection_metrics_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics_by_split.to_csv(
        output_dir / "tables" / "model_selection_metrics_by_outer_split.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics_summary.to_csv(
        output_dir / "tables" / "model_selection_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    calibration_curves.to_csv(
        output_dir / "tables" / "model_selection_calibration_curves_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if OUTPUT_FIGURES:
        save_prediction_figures(
            metrics_raw=metrics_raw,
            metrics_summary=metrics_summary,
            calibration_df=calibration_curves,
            figs_dir=output_dir / "figures" / "inner_selection",
        )
    best_model = select_best_model(metrics_summary)
    cprint(
        f"[INNER-CV SUMMARY LEADER] {best_model} | "
        f"{MODEL_DISPLAY_NAMES.get(best_model, best_model)}"
    )
    return metrics_raw, metrics_by_split, metrics_summary, best_model


def _annotate_training_calibration_evidence(
    table: pd.DataFrame,
    part: OuterPartition,
    calibration_fold: int,
    calibration_run_seed: int,
) -> pd.DataFrame:
    """Attach the target partition and inner-fold provenance to evidence."""
    if table is None or table.empty:
        return pd.DataFrame()
    out = table.copy()
    # In reliability estimation, ``seed`` identifies an independent inner
    # validation fold. The actual RNG seed remains available separately.
    out["seed"] = int(calibration_fold)
    out["calibration_run_seed"] = int(calibration_run_seed)
    out["target_outer_partition"] = int(part.partition_id)
    out["outer_repeat"] = int(part.repeat_id)
    out["outer_fold"] = int(part.fold_id)
    out["split_seed"] = int(part.split_seed)
    out["calibration_fold"] = int(calibration_fold)
    out["calibration_role"] = "outer_training_inner_validation"
    return out


def compute_training_internal_source_calibration(
    X_outer_train: pd.DataFrame,
    y_outer_train: np.ndarray,
    schema: FeatureSchema,
    models_for_partition: List[str],
    part: OuterPartition,
    class_order: List[int],
    worker_root: Path,
) -> Dict[str, pd.DataFrame]:
    """Compute source-reliability evidence without the outer holdout.

    Every calibration model is fitted on an inner-subtraining fold and explained
    on the corresponding inner-validation fold. The outer holdout is absent from
    preprocessing, model fitting, perturbation, loss evaluation, and FIRM.
    """
    from sklearn.model_selection import StratifiedKFold

    min_class_count = int(pd.Series(y_outer_train).value_counts().min())
    n_splits = int(min(SHADOW_INNER_CV_FOLDS, min_class_count))
    if n_splits < 3:
        raise RuntimeError(
            "At least three inner folds are required to estimate source "
            f"repeatability for outer partition {part.partition_id}."
        )
    split_seed = stable_seed_from_key("inner_cv_split", part.split_seed)
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=split_seed,
    )
    source_parts: Dict[str, List[pd.DataFrame]] = {
        "shap": [],
        "pfi": [],
        "loco": [],
        "firm": [],
    }

    for calibration_fold, (subtrain_idx, validation_idx) in enumerate(
        cv.split(X_outer_train, y_outer_train), 1
    ):
        X_subtrain = X_outer_train.iloc[subtrain_idx].reset_index(drop=True)
        X_validation = X_outer_train.iloc[validation_idx].reset_index(drop=True)
        y_subtrain = np.asarray(y_outer_train, dtype=int)[subtrain_idx]
        y_validation = np.asarray(y_outer_train, dtype=int)[validation_idx]
        sample_weight = (
            make_balanced_sample_weight(y_subtrain)
            if USE_BALANCED_SAMPLE_WEIGHT
            else None
        )

        for model_name in models_for_partition:
            run_seed = stable_seed_from_key(
                "source_weight_calibration",
                part.partition_id,
                calibration_fold,
                model_name,
            )
            cprint(
                f"[SOURCE CALIBRATION] outer={part.partition_id} "
                f"inner={calibration_fold}/{n_splits} model={model_name}"
            )
            encoded = fit_transform_split(
                X_subtrain,
                X_validation,
                schema,
                model_name=model_name,
            )
            estimator = build_estimator(
                model_name,
                seed=run_seed,
                n_jobs=MODEL_N_JOBS,
                categorical_feature_indices=encoded.categorical_feature_indices,
            )
            fit_estimator(
                estimator,
                encoded.X_train,
                y_subtrain,
                sample_weight=sample_weight,
            )
            proba = predict_proba_aligned(
                estimator,
                encoded.X_test,
                class_order,
            )
            y_pred = np.asarray(class_order, dtype=int)[np.argmax(proba, axis=1)]

            if ENABLE_SHAP:
                shap_df = compute_shap_importance_safe(
                    estimator=estimator,
                    preprocessor=encoded.preprocessor,
                    X_train_raw=X_subtrain,
                    y_train=y_subtrain,
                    X_eval_raw=X_validation,
                    y_eval=y_validation,
                    schema=schema,
                    classes=class_order,
                    model_name=model_name,
                    seed=run_seed,
                    background_size=int(SHAP_BACKGROUND_SIZE),
                    eval_size=None,
                    max_evals=(
                        int(SHAP_PERMUTATION_CYCLES)
                        * (2 * len(schema.feature_cols) + 1)
                    ),
                    reference_proba=proba,
                    work_root=worker_root,
                    call_key=(
                        f"outer_{int(part.partition_id):03d}_inner_"
                        f"{int(calibration_fold):02d}_{model_name}_"
                        f"seed_{int(run_seed)}_bg_{int(SHAP_BACKGROUND_SIZE)}"
                    ),
                )
                require_full_evaluation(
                    shap_df,
                    "n_eval",
                    len(X_validation),
                    "training-internal SHAP",
                )
                source_parts["shap"].append(
                    _annotate_training_calibration_evidence(
                        shap_df,
                        part,
                        calibration_fold,
                        run_seed,
                    )
                )

            if ENABLE_PFI:
                pfi_df = compute_class_permutation_importance(
                    estimator=estimator,
                    preprocessor=encoded.preprocessor,
                    X_eval_raw=X_validation,
                    y_eval=y_validation,
                    classes=class_order,
                    feature_cols=schema.feature_cols,
                    model_name=model_name,
                    seed=run_seed,
                    repeats=PERMUTATION_REPEATS,
                )
                require_full_evaluation(
                    pfi_df,
                    "n_eval",
                    len(X_validation),
                    "training-internal PFI",
                )
                source_parts["pfi"].append(
                    _annotate_training_calibration_evidence(
                        pfi_df,
                        part,
                        calibration_fold,
                        run_seed,
                    )
                )

            if ENABLE_FIRM:
                firm_df = compute_firm_importance(
                    base_proba=proba,
                    X_eval_raw=X_validation,
                    y_eval=y_validation,
                    schema=schema,
                    classes=class_order,
                    model_name=model_name,
                    seed=run_seed,
                    feature_cols=list(schema.feature_cols),
                )
                require_full_evaluation(
                    firm_df,
                    "n_eval",
                    len(X_validation),
                    "training-internal FIRM",
                )
                source_parts["firm"].append(
                    _annotate_training_calibration_evidence(
                        firm_df,
                        part,
                        calibration_fold,
                        run_seed,
                    )
                )

            if ENABLE_LOCO:
                loco_df = compute_loco_importance(
                    base_proba=proba,
                    base_pred=y_pred,
                    y_train=y_subtrain,
                    y_eval=y_validation,
                    X_train_raw=X_subtrain,
                    X_eval_raw=X_validation,
                    schema=schema,
                    classes=class_order,
                    model_name=model_name,
                    seed=run_seed,
                    n_jobs=MODEL_N_JOBS,
                    sample_weight=sample_weight,
                    candidate_features=list(schema.feature_cols),
                )
                require_full_evaluation(
                    loco_df,
                    "n_eval",
                    len(X_validation),
                    "training-internal LOCO",
                )
                source_parts["loco"].append(
                    _annotate_training_calibration_evidence(
                        loco_df,
                        part,
                        calibration_fold,
                        run_seed,
                    )
                )
            gc.collect()

    out = {
        source: concat_or_empty(parts)
        for source, parts in source_parts.items()
        if parts
    }
    missing_sources = [source for source in FUSION_SOURCE_SPECS if source not in out]
    if missing_sources:
        raise RuntimeError(
            "Training-internal source calibration is incomplete: "
            + ", ".join(missing_sources)
        )
    return out


EXPLANATION_CHECKPOINT_LISTS = (
    "shap_tables",
    "perm_tables",
    "loco_tables",
    "firm_tables",
    "pdp_tables",
    "ice_tables",
    "pred_rows",
    "shap_background_sensitivity_tables",
    "negative_control_tables",
    "holdout_metric_rows",
    "holdout_calibration_rows",
)


def explanation_method_digest(schema: FeatureSchema) -> str:
    tabpfn_enabled = "tabpfn" in [name.lower() for name in CANDIDATE_MODELS]
    fingerprint_packages = ["numpy", "pandas", "scikit-learn", "shap"]
    if tabpfn_enabled:
        fingerprint_packages.extend(["tabpfn", "torch"])
    specification = {
        "checkpoint_format_version": int(CHECKPOINT_FORMAT_VERSION),
        "run_profile": str(RUN_PROFILE).lower(),
        "feature_cols": list(schema.feature_cols),
        "numeric_cols": list(schema.numeric_cols),
        "categorical_cols": list(schema.categorical_cols),
        "use_balanced_sample_weight": bool(USE_BALANCED_SAMPLE_WEIGHT),
        "inner_cv_folds": int(INNER_CV_FOLDS),
        "source_weight_cv_folds": int(SHADOW_INNER_CV_FOLDS),
        "enable_shap": bool(ENABLE_SHAP),
        "enable_pfi": bool(ENABLE_PFI),
        "enable_loco": bool(ENABLE_LOCO),
        "enable_firm": bool(ENABLE_FIRM),
        "enable_pdp": bool(ENABLE_PDP),
        "enable_ice": bool(ENABLE_ICE),
        "enable_negative_control": bool(ENABLE_NEGATIVE_CONTROL),
        "shap_background_size": int(SHAP_BACKGROUND_SIZE),
        "shap_background_sensitivity": bool(ENABLE_SHAP_BACKGROUND_SENSITIVITY),
        "shap_background_sensitivity_sizes": [
            int(value) for value in SHAP_BACKGROUND_SENSITIVITY_SIZES
        ],
        "shap_eval_size": SHAP_EVAL_SIZE,
        "shap_permutation_cycles": int(SHAP_PERMUTATION_CYCLES),
        "permutation_repeats": int(PERMUTATION_REPEATS),
        "negative_control_repeats": int(NEGATIVE_CONTROL_PERMUTATION_REPEATS),
        "explain_all_features": bool(EXPLAIN_ALL_FEATURES),
        "pdp_max_values": PDP_MAX_VALUES,
        "pdp_min_train_count": int(PDP_MIN_TRAIN_COUNT),
        "firm_continuous_features": list(FIRM_CONTINUOUS_FEATURES),
        "tabpfn_enabled": bool(tabpfn_enabled),
        "tabpfn_checkpoint_sha256": (
            file_sha256(TABPFN_LOCAL_CKPT) if tabpfn_enabled else None
        ),
        "tabpfn_force_fp32": (
            bool(TABPFN_FORCE_FP32_INFERENCE) if tabpfn_enabled else None
        ),
        "tabpfn_internal_memory_saving": (
            bool(TABPFN_INTERNAL_MEMORY_SAVING) if tabpfn_enabled else None
        ),
        "tabpfn_reconstruction_tolerance": (
            float(TABPFN_SHAP_RECONSTRUCTION_TOL) if tabpfn_enabled else None
        ),
        "package_versions": {
            name: package_version(name)
            for name in fingerprint_packages
        },
    }
    return stable_json_digest(specification)


def explanation_partition_fingerprint(
    data_digest: str,
    method_digest: str,
    part: OuterPartition,
    models_for_partition: List[str],
) -> str:
    return stable_json_digest(
        {
            "data_digest": data_digest,
            "method_digest": method_digest,
            "partition_id": int(part.partition_id),
            "repeat_id": int(part.repeat_id),
            "fold_id": int(part.fold_id),
            "split_seed": int(part.split_seed),
            "train_index_digest": array_digest(
                np.asarray(part.train_idx, dtype=np.int64)
            ),
            "test_index_digest": array_digest(np.asarray(part.test_idx, dtype=np.int64)),
            "models": [str(model) for model in models_for_partition],
        }
    )


def explanation_checkpoint_paths(
    checkpoint_dir: Path, partition_id: int
) -> Tuple[Path, Path]:
    stem = f"outer_{int(partition_id):03d}"
    return checkpoint_dir / f"{stem}.pkl", checkpoint_dir / f"{stem}.complete.json"


def load_explanation_partition_checkpoint(
    checkpoint_dir: Path,
    partition_id: int,
    expected_fingerprint: str,
) -> Optional[dict]:
    payload_path, marker_path = explanation_checkpoint_paths(
        checkpoint_dir, partition_id
    )
    if not marker_path.exists():
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("fingerprint") != expected_fingerprint:
        raise RuntimeError(
            f"Checkpoint fingerprint mismatch for outer partition {partition_id}. "
            "Use a new output directory after changing data, software, models, "
            "or explanation settings."
        )
    if not payload_path.exists():
        raise RuntimeError(
            f"Completed checkpoint marker exists without payload: {payload_path}."
        )
    observed_sha = file_sha256(str(payload_path))
    if observed_sha != marker.get("payload_sha256"):
        raise RuntimeError(
            f"Checkpoint checksum mismatch for outer partition {partition_id}: "
            f"{payload_path}."
        )
    payload = pd.read_pickle(payload_path)
    if not isinstance(payload, dict) or payload.get("fingerprint") != expected_fingerprint:
        raise RuntimeError(f"Invalid checkpoint payload: {payload_path}.")
    return payload


def save_explanation_partition_checkpoint(
    checkpoint_dir: Path,
    partition_id: int,
    fingerprint: str,
    accumulators: Dict[str, list],
    start_lengths: Dict[str, int],
    source_calibration_parts: Dict[str, List[pd.DataFrame]],
    source_start_lengths: Dict[str, int],
) -> None:
    payload_path, marker_path = explanation_checkpoint_paths(
        checkpoint_dir, partition_id
    )
    payload = {
        "format_version": int(CHECKPOINT_FORMAT_VERSION),
        "partition_id": int(partition_id),
        "fingerprint": fingerprint,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "accumulators": {
            name: list(accumulators[name][start_lengths[name] :])
            for name in EXPLANATION_CHECKPOINT_LISTS
        },
        "source_calibration_parts": {
            source: list(parts[source_start_lengths[source] :])
            for source, parts in source_calibration_parts.items()
        },
    }
    atomic_pickle_dump(payload, payload_path)
    atomic_json_dump(
        {
            "format_version": int(CHECKPOINT_FORMAT_VERSION),
            "partition_id": int(partition_id),
            "fingerprint": fingerprint,
            "payload_sha256": file_sha256(str(payload_path)),
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        },
        marker_path,
    )


def run_explanation_workflow(
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    schema: FeatureSchema,
    partitions: List[OuterPartition],
    explanation_models_by_seed: Dict[int, List[str]],
    output_dir: Path,
) -> Dict[str, object]:
    class_order = [0, 1, 2]
    worker_root = output_dir / "checkpoints" / "tabpfn_shap"
    worker_root.mkdir(parents=True, exist_ok=True)
    (
        shap_tables,
        perm_tables,
        loco_tables,
        firm_tables,
        pdp_tables,
        ice_tables,
        pred_rows,
    ) = ([], [], [], [], [], [], [])
    shap_background_sensitivity_tables = []
    negative_control_tables = []
    holdout_metric_rows, holdout_calibration_rows = [], []
    source_calibration_parts: Dict[str, List[pd.DataFrame]] = {
        source: [] for source in FUSION_SOURCE_SPECS
    }
    accumulators: Dict[str, list] = {
        "shap_tables": shap_tables,
        "perm_tables": perm_tables,
        "loco_tables": loco_tables,
        "firm_tables": firm_tables,
        "pdp_tables": pdp_tables,
        "ice_tables": ice_tables,
        "pred_rows": pred_rows,
        "shap_background_sensitivity_tables": shap_background_sensitivity_tables,
        "negative_control_tables": negative_control_tables,
        "holdout_metric_rows": holdout_metric_rows,
        "holdout_calibration_rows": holdout_calibration_rows,
    }
    checkpoint_dir = output_dir / "checkpoints" / "outer_partitions"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_digest = stable_json_digest(
        {
            "X": dataframe_digest(X_all),
            "y": array_digest(np.asarray(y_all, dtype=int)),
        }
    )
    method_digest = explanation_method_digest(schema)

    cprint("\n" + "=" * 110)
    cprint("STEP 2: RELIABILITY-AWARE EXPLANATION EVIDENCE")
    cprint("=" * 110)
    cprint(f"explanation models by outer split: {explanation_models_by_seed}")

    for part in partitions:
        seed = int(part.partition_id)
        X_train = X_all.iloc[part.train_idx].reset_index(drop=True)
        X_test = X_all.iloc[part.test_idx].reset_index(drop=True)
        y_train = np.asarray(y_all, dtype=int)[part.train_idx]
        y_test = np.asarray(y_all, dtype=int)[part.test_idx]
        sample_weight = (
            make_balanced_sample_weight(y_train) if USE_BALANCED_SAMPLE_WEIGHT else None
        )

        models_for_seed = explanation_models_by_seed.get(int(seed), [])
        if not models_for_seed:
            raise RuntimeError(f"No explanation model was selected for seed={seed}.")

        checkpoint_fingerprint = explanation_partition_fingerprint(
            data_digest=data_digest,
            method_digest=method_digest,
            part=part,
            models_for_partition=models_for_seed,
        )
        if ENABLE_OUTER_PARTITION_CHECKPOINTS:
            checkpoint_payload = load_explanation_partition_checkpoint(
                checkpoint_dir=checkpoint_dir,
                partition_id=seed,
                expected_fingerprint=checkpoint_fingerprint,
            )
            if checkpoint_payload is not None:
                checkpoint_accumulators = checkpoint_payload.get("accumulators", {})
                missing_accumulators = [
                    name
                    for name in EXPLANATION_CHECKPOINT_LISTS
                    if name not in checkpoint_accumulators
                ]
                if missing_accumulators:
                    raise RuntimeError(
                        f"Outer partition {seed} checkpoint is missing accumulators: "
                        + ", ".join(missing_accumulators)
                    )
                for name in EXPLANATION_CHECKPOINT_LISTS:
                    accumulators[name].extend(checkpoint_accumulators[name])
                checkpoint_sources = checkpoint_payload.get(
                    "source_calibration_parts", {}
                )
                for source in source_calibration_parts:
                    if source not in checkpoint_sources:
                        raise RuntimeError(
                            f"Outer partition {seed} checkpoint is missing source: {source}."
                        )
                    source_calibration_parts[source].extend(
                        checkpoint_sources[source]
                    )
                cprint(f"[CHECKPOINT HIT] outer={seed}; completed evidence restored.")
                continue

        start_lengths = {
            name: len(values) for name, values in accumulators.items()
        }
        source_start_lengths = {
            source: len(values) for source, values in source_calibration_parts.items()
        }

        calibration_evidence = compute_training_internal_source_calibration(
            X_outer_train=X_train,
            y_outer_train=y_train,
            schema=schema,
            models_for_partition=models_for_seed,
            part=part,
            class_order=class_order,
            worker_root=worker_root,
        )
        for source, table in calibration_evidence.items():
            source_calibration_parts[source].append(table)

        for model_name in models_for_seed:
            cprint(f"[EXPLAIN] seed={seed} model={model_name}")
            encoded = fit_transform_split(
                X_train, X_test, schema, model_name=model_name
            )
            estimator = build_estimator(
                model_name,
                seed=int(seed),
                n_jobs=MODEL_N_JOBS,
                categorical_feature_indices=encoded.categorical_feature_indices,
            )
            fit_estimator(
                estimator, encoded.X_train, y_train, sample_weight=sample_weight
            )
            proba = predict_proba_aligned(estimator, encoded.X_test, class_order)
            y_pred = np.asarray(class_order, dtype=int)[np.argmax(proba, axis=1)]

            holdout_metrics = evaluate_predictions(y_test, y_pred, labels=class_order)
            if ENABLE_CALIBRATION_DIAGNOSTICS:
                holdout_metrics.update(
                    compute_probability_diagnostics(
                        y_true=y_test,
                        proba=proba,
                        labels=class_order,
                        n_bins=CALIBRATION_N_BINS,
                    )
                )
                holdout_calibration_rows.append(
                    compute_calibration_curve_rows(
                        y_true=y_test,
                        proba=proba,
                        labels=class_order,
                        model_name=model_name,
                        seed=int(seed),
                        n_bins=CALIBRATION_N_BINS,
                    )
                )
            holdout_metric_rows.append(
                {
                    "seed": int(seed),
                    "outer_repeat": int(part.repeat_id),
                    "outer_fold": int(part.fold_id),
                    "split_seed": int(part.split_seed),
                    "model": model_name,
                    "model_display": MODEL_DISPLAY_NAMES[model_name],
                    "preprocessing_mode": preprocessing_mode(model_name),
                    "imbalance_handling": imbalance_handling_mode(model_name),
                    "n_train": int(len(y_train)),
                    "n_holdout": int(len(y_test)),
                    "evaluation_partition": "outer_holdout",
                    **holdout_metrics,
                }
            )

            pred_df = X_test.reset_index(drop=True).copy()
            pred_df["seed"] = int(seed)
            pred_df["outer_repeat"] = int(part.repeat_id)
            pred_df["outer_fold"] = int(part.fold_id)
            pred_df["split_seed"] = int(part.split_seed)
            pred_df["model"] = model_name
            pred_df["y_true"] = y_test
            pred_df["y_pred"] = y_pred
            for pos, class_id in enumerate(class_order):
                pred_df[f"proba_class_{int(class_id)}"] = proba[:, pos]
            pred_rows.append(pred_df)

            if ENABLE_NEGATIVE_CONTROL:
                try:
                    nc_df = compute_negative_control_check(
                        y_train=y_train,
                        y_eval=y_test,
                        X_train_raw=X_train.reset_index(drop=True),
                        X_eval_raw=X_test.reset_index(drop=True),
                        schema=schema,
                        classes=class_order,
                        model_name=model_name,
                        seed=int(seed),
                        sample_weight=sample_weight,
                    )
                    if not nc_df.empty:
                        negative_control_tables.append(nc_df)
                except Exception as exc:
                    cprint(
                        f"[WARN] negative control failed for seed={seed} model={model_name}: {exc}"
                    )

            shap_df = pd.DataFrame()
            if ENABLE_SHAP:
                if ENABLE_SHAP_BACKGROUND_SENSITIVITY and RUN_PROFILE.lower() == "full":
                    background_sizes = list(
                        dict.fromkeys(
                            [int(SHAP_BACKGROUND_SIZE)]
                            + [int(v) for v in SHAP_BACKGROUND_SENSITIVITY_SIZES]
                        )
                    )
                else:
                    background_sizes = [int(SHAP_BACKGROUND_SIZE)]

                for background_size in background_sizes:
                    try:
                        current_shap = compute_shap_importance_safe(
                            estimator=estimator,
                            preprocessor=encoded.preprocessor,
                            X_train_raw=X_train.reset_index(drop=True),
                            y_train=y_train,
                            X_eval_raw=X_test.reset_index(drop=True),
                            y_eval=y_test,
                            schema=schema,
                            classes=class_order,
                            model_name=model_name,
                            seed=int(seed),
                            background_size=background_size,
                            eval_size=SHAP_EVAL_SIZE,
                            max_evals=(
                                int(SHAP_PERMUTATION_CYCLES)
                                * (2 * len(schema.feature_cols) + 1)
                            ),
                            reference_proba=proba,
                            work_root=worker_root,
                            call_key=(
                                f"outer_{int(seed):03d}_holdout_{model_name}_"
                                f"bg_{int(background_size)}"
                            ),
                        )
                        require_full_evaluation(
                            current_shap, "n_eval", len(X_test), "SHAP"
                        )
                        if (
                            ENABLE_SHAP_BACKGROUND_SENSITIVITY
                            and RUN_PROFILE.lower() == "full"
                            and not current_shap.empty
                        ):
                            shap_background_sensitivity_tables.append(current_shap)
                        if int(background_size) == int(SHAP_BACKGROUND_SIZE):
                            shap_df = current_shap
                    except Exception as exc:
                        cprint(
                            f"[WARN] SHAP failed for seed={seed} model={model_name} "
                            f"background={background_size}: {exc}"
                        )
                        if RUN_PROFILE.lower() == "full":
                            raise RuntimeError(
                                "A full paper run cannot continue with missing "
                                f"SHAP evidence for seed={seed}, model={model_name}, "
                                f"background={background_size}."
                            ) from exc
            if not shap_df.empty:
                shap_tables.append(shap_df)

            if ENABLE_PFI:
                perm_df = compute_class_permutation_importance(
                    estimator=estimator,
                    preprocessor=encoded.preprocessor,
                    X_eval_raw=X_test.reset_index(drop=True),
                    y_eval=y_test,
                    classes=class_order,
                    feature_cols=schema.feature_cols,
                    model_name=model_name,
                    seed=int(seed),
                    repeats=PERMUTATION_REPEATS,
                )
                require_full_evaluation(perm_df, "n_eval", len(X_test), "PFI")
                perm_tables.append(perm_df)
            else:
                perm_df = pd.DataFrame()

            if ENABLE_FIRM:
                firm_df = compute_firm_importance(
                    base_proba=proba,
                    X_eval_raw=X_test.reset_index(drop=True),
                    y_eval=y_test,
                    schema=schema,
                    classes=class_order,
                    model_name=model_name,
                    seed=int(seed),
                    feature_cols=list(schema.feature_cols),
                )
                require_full_evaluation(firm_df, "n_eval", len(X_test), "FIRM")
                if not firm_df.empty:
                    firm_tables.append(firm_df)
            else:
                firm_df = pd.DataFrame()

            top_df = build_top_feature_table(
                shap_df, perm_df, top_k=DISPLAY_TOP_K_FEATURES
            )

            if ENABLE_LOCO:
                loco_features = (
                    list(schema.feature_cols)
                    if EXPLAIN_ALL_FEATURES
                    else features_from_top_table(
                        top_df, schema.feature_cols, DISPLAY_TOP_K_FEATURES
                    )
                )
                loco_df = compute_loco_importance(
                    base_proba=proba,
                    base_pred=y_pred,
                    y_train=y_train,
                    y_eval=y_test,
                    X_train_raw=X_train.reset_index(drop=True),
                    X_eval_raw=X_test.reset_index(drop=True),
                    schema=schema,
                    classes=class_order,
                    model_name=model_name,
                    seed=int(seed),
                    n_jobs=MODEL_N_JOBS,
                    sample_weight=sample_weight,
                    candidate_features=loco_features,
                )
                require_full_evaluation(loco_df, "n_eval", len(X_test), "LOCO")
                if not loco_df.empty:
                    loco_tables.append(loco_df)

            if ENABLE_PDP or ENABLE_ICE:
                pdp_df, ice_df = compute_response_profiles(
                    estimator=estimator,
                    preprocessor=encoded.preprocessor,
                    X_train_raw=X_train.reset_index(drop=True),
                    X_eval_raw=X_test.reset_index(drop=True),
                    classes=class_order,
                    feature_cols=schema.feature_cols,
                    model_name=model_name,
                    seed=int(seed),
                    top_features_by_class=None if EXPLAIN_ALL_FEATURES else top_df,
                    top_k=len(schema.feature_cols)
                    if EXPLAIN_ALL_FEATURES
                    else DISPLAY_TOP_K_FEATURES,
                    max_values=PDP_MAX_VALUES,
                    min_train_count=PDP_MIN_TRAIN_COUNT,
                    return_pdp=bool(ENABLE_PDP),
                    return_ice=bool(ENABLE_ICE),
                )
                if ENABLE_PDP:
                    require_full_evaluation(pdp_df, "n_eval", len(X_test), "PDP")
                    pdp_tables.append(pdp_df)
                if ENABLE_ICE:
                    require_full_evaluation(ice_df, "n_ice_eval", len(X_test), "ICE")
                    if not ice_df.empty:
                        ice_tables.append(ice_df)
            else:
                pdp_df = pd.DataFrame()
                ice_df = pd.DataFrame()

            gc.collect()

        if ENABLE_OUTER_PARTITION_CHECKPOINTS:
            save_explanation_partition_checkpoint(
                checkpoint_dir=checkpoint_dir,
                partition_id=seed,
                fingerprint=checkpoint_fingerprint,
                accumulators=accumulators,
                start_lengths=start_lengths,
                source_calibration_parts=source_calibration_parts,
                source_start_lengths=source_start_lengths,
            )
            cprint(f"[CHECKPOINT SAVED] outer={seed}")

    shap_all = attach_outer_partition_metadata(concat_or_empty(shap_tables), partitions)
    shap_background_sensitivity_all = attach_outer_partition_metadata(
        concat_or_empty(shap_background_sensitivity_tables), partitions
    )
    perm_all = attach_outer_partition_metadata(concat_or_empty(perm_tables), partitions)
    loco_all = attach_outer_partition_metadata(concat_or_empty(loco_tables), partitions)
    firm_all = attach_outer_partition_metadata(concat_or_empty(firm_tables), partitions)
    pdp_all = attach_outer_partition_metadata(concat_or_empty(pdp_tables), partitions)
    ice_all = attach_outer_partition_metadata(concat_or_empty(ice_tables), partitions)
    predictions_all = concat_or_empty(pred_rows)
    holdout_metrics_raw = pd.DataFrame(holdout_metric_rows)
    holdout_metrics_summary = (
        summarize_metrics(holdout_metrics_raw, ["model", "model_display"])
        if not holdout_metrics_raw.empty
        else pd.DataFrame()
    )
    holdout_calibration_all = concat_or_empty(holdout_calibration_rows)
    negative_control_all = concat_or_empty(negative_control_tables)
    negative_control_all = benchmark_negative_control(
        negative_control_all,
        permutation_df=perm_all,
        loco_df=loco_all,
        firm_df=firm_all,
    )
    negative_control_summary = summarize_negative_control(negative_control_all)
    source_calibration_all = {
        source: concat_or_empty(parts)
        for source, parts in source_calibration_parts.items()
    }
    coverage_specs: Dict[str, Tuple[pd.DataFrame, str, Optional[List[str]]]] = {}
    if ENABLE_SHAP:
        coverage_specs["SHAP"] = (shap_all, "n_eval", list(schema.feature_cols))
        if ENABLE_SHAP_BACKGROUND_SENSITIVITY and RUN_PROFILE.lower() == "full":
            for background_size in sorted(
                set(int(v) for v in SHAP_BACKGROUND_SENSITIVITY_SIZES)
            ):
                if int(background_size) == int(SHAP_BACKGROUND_SIZE):
                    continue
                sensitivity_subset = shap_background_sensitivity_all.loc[
                    pd.to_numeric(
                        shap_background_sensitivity_all.get(
                            "background_size_requested", pd.Series(dtype=float)
                        ),
                        errors="coerce",
                    )
                    == int(background_size)
                ].copy()
                coverage_specs[f"SHAP_background_{int(background_size)}"] = (
                    sensitivity_subset,
                    "n_eval",
                    list(schema.feature_cols),
                )
    if ENABLE_PFI:
        coverage_specs["PFI"] = (perm_all, "n_eval", list(schema.feature_cols))
    if ENABLE_LOCO:
        coverage_specs["LOCO"] = (loco_all, "n_eval", list(schema.feature_cols))
    if ENABLE_FIRM:
        coverage_specs["FIRM"] = (firm_all, "n_eval", list(schema.feature_cols))
    if ENABLE_PDP:
        coverage_specs["PDP"] = (pdp_all, "n_eval", list(schema.feature_cols))
    if ENABLE_ICE:
        coverage_specs["ICE"] = (ice_all, "n_ice_eval", list(schema.feature_cols))
    if ENABLE_NEGATIVE_CONTROL:
        coverage_specs["negative_control"] = (
            negative_control_all,
            "n_eval",
            [NEGATIVE_CONTROL_FEATURE],
        )
    coverage_audit = build_explanation_coverage_audit(
        holdout_metrics=holdout_metrics_raw,
        source_specs=coverage_specs,
        expected_classes=class_order,
    )
    if ENABLE_SHAP_BACKGROUND_SENSITIVITY and RUN_PROFILE.lower() == "full":
        shap_background_sensitivity_runs, shap_background_sensitivity_summary = (
            summarize_shap_background_sensitivity(
                shap_background_sensitivity_all,
                reference_size=int(SHAP_BACKGROUND_SIZE),
                top_k=DISPLAY_TOP_K_FEATURES,
            )
        )
    else:
        shap_background_sensitivity_runs = pd.DataFrame()
        shap_background_sensitivity_summary = pd.DataFrame()

    balanced_fusion = aggregate_crossfitted_consensus(
        shap_df=shap_all,
        perm_df=perm_all,
        loco_df=loco_all,
        firm_df=firm_all,
        class_ids=class_order,
        feature_cols=schema.feature_cols,
        top_k=DISPLAY_TOP_K_FEATURES,
        source_specs=FUSION_SOURCE_SPECS,
        source_weight_calibration_tables=source_calibration_all,
        partitions=partitions,
    )
    prevalence_fusion = aggregate_crossfitted_consensus(
        shap_df=shap_all,
        perm_df=perm_all,
        loco_df=loco_all,
        firm_df=firm_all,
        class_ids=class_order,
        feature_cols=schema.feature_cols,
        top_k=DISPLAY_TOP_K_FEATURES,
        source_specs=PREVALENCE_FUSION_SOURCE_SPECS,
        source_weight_calibration_tables=source_calibration_all,
        partitions=partitions,
    )
    reliability_df = balanced_fusion["summary"]
    partition_consensus_df = balanced_fusion["partition"]
    source_weight_df = balanced_fusion["weights"]
    pairwise_dominance_df = balanced_fusion["pairwise"]
    response_profile_summary = build_response_profile_summary(pdp_all, ice_all)
    sensitivity_df = (
        balanced_fusion["scheme_sensitivity"]
        if ENABLE_WEIGHT_SENSITIVITY_TABLE
        else pd.DataFrame()
    )
    measure_sensitivity_df = build_measure_sensitivity(
        reliability_df,
        prevalence_fusion["summary"],
        top_k=WEIGHT_SENSITIVITY_TOP_K,
    )
    source_dependence_df = build_source_dependence_table(partition_consensus_df)
    source_ablation_df = balanced_fusion["source_ablation"]

    best_model_by_partition = {
        int(partition_id): str(models[0])
        for partition_id, models in explanation_models_by_seed.items()
        if models
    }
    best_single_holdout = {
        "shap": filter_evidence_to_partition_models(
            shap_all, best_model_by_partition, "seed"
        ),
        "pfi": filter_evidence_to_partition_models(
            perm_all, best_model_by_partition, "seed"
        ),
        "loco": filter_evidence_to_partition_models(
            loco_all, best_model_by_partition, "seed"
        ),
        "firm": filter_evidence_to_partition_models(
            firm_all, best_model_by_partition, "seed"
        ),
    }
    best_single_calibration = {
        source: filter_evidence_to_partition_models(
            table, best_model_by_partition, "target_outer_partition"
        )
        for source, table in source_calibration_all.items()
    }
    best_single_fusion = aggregate_crossfitted_consensus(
        shap_df=best_single_holdout["shap"],
        perm_df=best_single_holdout["pfi"],
        loco_df=best_single_holdout["loco"],
        firm_df=best_single_holdout["firm"],
        class_ids=class_order,
        feature_cols=schema.feature_cols,
        top_k=DISPLAY_TOP_K_FEATURES,
        source_specs=FUSION_SOURCE_SPECS,
        source_weight_calibration_tables=best_single_calibration,
        partitions=partitions,
    )
    model_set_sensitivity_df = build_model_set_sensitivity(
        ensemble_summary=reliability_df,
        single_summary=best_single_fusion["summary"],
        model_by_partition=best_model_by_partition,
        top_k=DISPLAY_TOP_K_FEATURES,
    )

    tables_dir = output_dir / "tables"
    shap_all.to_csv(
        tables_dir / "shap_importance_raw.csv", index=False, encoding="utf-8-sig"
    )
    shap_background_sensitivity_all.to_csv(
        tables_dir / "shap_background_sensitivity_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    shap_background_sensitivity_runs.to_csv(
        tables_dir / "shap_background_sensitivity_by_run.csv",
        index=False,
        encoding="utf-8-sig",
    )
    shap_background_sensitivity_summary.to_csv(
        tables_dir / "shap_background_sensitivity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    perm_all.to_csv(
        tables_dir / "permutation_reliance_raw.csv", index=False, encoding="utf-8-sig"
    )
    loco_all.to_csv(
        tables_dir / "loco_reliance_raw.csv", index=False, encoding="utf-8-sig"
    )
    firm_all.to_csv(
        tables_dir / "firm_observational_relevance_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pdp_all.to_csv(
        tables_dir / "support_aware_pdp_raw.csv", index=False, encoding="utf-8-sig"
    )
    ice_all.to_csv(
        tables_dir / "ice_heterogeneity_raw.csv", index=False, encoding="utf-8-sig"
    )
    response_profile_summary.to_csv(
        tables_dir / "response_profile_summary.csv", index=False, encoding="utf-8-sig"
    )
    predictions_all.to_csv(
        tables_dir / "test_predictions_raw.csv", index=False, encoding="utf-8-sig"
    )
    holdout_metrics_raw.to_csv(
        tables_dir / "explanation_model_holdout_metrics_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    holdout_metrics_summary.to_csv(
        tables_dir / "explanation_model_holdout_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    holdout_calibration_all.to_csv(
        tables_dir / "explanation_model_holdout_calibration_curves_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    coverage_audit.to_csv(
        tables_dir / "explanation_coverage_audit.csv", index=False, encoding="utf-8-sig"
    )
    negative_control_all.to_csv(
        tables_dir / "negative_control_raw.csv", index=False, encoding="utf-8-sig"
    )
    negative_control_summary.to_csv(
        tables_dir / "negative_control_summary.csv", index=False, encoding="utf-8-sig"
    )
    for source, table in source_calibration_all.items():
        table.to_csv(
            tables_dir / f"source_weight_calibration_{source}_raw.csv",
            index=False,
            encoding="utf-8-sig",
        )
    reliability_df.to_csv(
        tables_dir / "explanation_reliability_map.csv",
        index=False,
        encoding="utf-8-sig",
    )
    partition_consensus_df.to_csv(
        tables_dir / "crossfitted_consensus_by_outer_partition.csv",
        index=False,
        encoding="utf-8-sig",
    )
    source_weight_df.to_csv(
        tables_dir / "crossfitted_source_weights.csv", index=False, encoding="utf-8-sig"
    )
    pairwise_dominance_df.to_csv(
        tables_dir / "pairwise_feature_dominance.csv", index=False, encoding="utf-8-sig"
    )
    prevalence_fusion["summary"].to_csv(
        tables_dir / "prevalence_weighted_consensus_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    measure_sensitivity_df.to_csv(
        tables_dir / "class_balance_measure_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sensitivity_df.to_csv(
        tables_dir / "weight_sensitivity_top_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    source_dependence_df.to_csv(
        tables_dir / "source_dependence_diagnostic.csv",
        index=False,
        encoding="utf-8-sig",
    )
    source_ablation_df.to_csv(
        tables_dir / "source_ablation_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_single_fusion["summary"].to_csv(
        tables_dir / "best_single_model_consensus_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_single_fusion["partition"].to_csv(
        tables_dir / "best_single_model_consensus_by_outer_partition.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_single_fusion["weights"].to_csv(
        tables_dir / "best_single_model_source_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )
    model_set_sensitivity_df.to_csv(
        tables_dir / "model_set_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fusion_cols = [
        "class_id",
        "feature",
        "class_rank_fused",
        "consensus_importance",
        "consensus_median",
        "consensus_std",
        "consensus_q25",
        "consensus_q75",
        "consensus_min",
        "consensus_max",
        "n_consensus_partitions",
        "n_source_specific_partitions",
        "n_unavailable_partitions",
        "consensus_partition_fraction",
        "top_k_inclusion_frequency",
        "top_k_inclusion_frequency_all_partitions",
        "top_k_inclusion_frequency_given_consensus",
        "partial_order_level",
        "n_unanimously_dominates",
        "n_unanimously_dominated_by",
        "evidence_ambiguity",
        "source_signal_range",
        "evidence_support",
    ]
    for source in FUSION_SOURCE_SPECS:
        fusion_cols.extend(
            [
                f"w_{source}",
                f"{source}_source_reliability",
                f"{source}_rank_signal",
                f"{source}_mean_importance",
                f"{source}_between_split_stability",
                f"{source}_within_split_model_agreement",
                f"{source}_information_availability",
            ]
        )
    keep_fusion_cols = [c for c in fusion_cols if c in reliability_df.columns]
    reliability_df[keep_fusion_cols].sort_values(
        ["class_id", "class_rank_fused"], ascending=[True, True]
    ).to_csv(
        tables_dir / "crossfitted_consensus_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if ENABLE_STRUCTURAL_COMPATIBILITY_AUDIT:
        structural_audit_meta = run_structural_compatibility_audit(
            X_all=X_all,
            y_all=y_all,
            partitions=partitions,
            reliability_df=reliability_df,
            feature_cols=list(schema.feature_cols),
            output_dir=output_dir,
        )
    else:
        structural_audit_meta = {
            "enabled": False,
            "status": "disabled_by_config",
        }

    if OUTPUT_FIGURES:
        figs_dir = output_dir / "figures"
        save_prediction_figures(
            metrics_raw=holdout_metrics_raw,
            metrics_summary=holdout_metrics_summary,
            calibration_df=holdout_calibration_all,
            figs_dir=figs_dir / "outer_holdout",
        )
        save_evidence_confidence_heatmap(
            reliability_df,
            figs_dir / "explanation_evidence_support_heatmap.png",
            top_k=DISPLAY_TOP_K_FEATURES,
        )
        save_top_evidence_confidence_dotplot(
            reliability_df,
            figs_dir / "top_feature_evidence_support_dotplot.png",
            top_k=DISPLAY_TOP_K_FEATURES,
        )
        save_prediction_distribution_figures(predictions_all, figs_dir)
        save_consensus_importance_panel(
            reliability_df, figs_dir, top_k=DISPLAY_TOP_K_FEATURES
        )
        save_reliability_conflict_scatter(reliability_df, figs_dir)
        save_source_diagnostic_heatmaps(
            reliability_df, figs_dir, top_k=DISPLAY_TOP_K_FEATURES
        )
        save_negative_control_figure(negative_control_summary, figs_dir)
        save_firm_importance_figure(firm_all, figs_dir, top_k=DISPLAY_TOP_K_FEATURES)
        save_pdp_profile_figures(
            pdp_all, reliability_df, figs_dir, top_k=DISPLAY_TOP_K_FEATURES
        )
        save_ice_heterogeneity_figures(
            ice_all, reliability_df, figs_dir, top_k=DISPLAY_TOP_K_FEATURES
        )
        save_crossfitted_consensus_figures(
            summary_df=reliability_df,
            partition_df=partition_consensus_df,
            source_weight_df=source_weight_df,
            pairwise_df=pairwise_dominance_df,
            scheme_sensitivity_df=sensitivity_df,
            measure_sensitivity_df=measure_sensitivity_df,
            figs_dir=figs_dir,
            top_k=DISPLAY_TOP_K_FEATURES,
        )

    return {
        "reliability_summary": reliability_df,
        "model_set_sensitivity": model_set_sensitivity_df,
        "structural_compatibility_audit": structural_audit_meta,
    }


def run_internal_method_checks() -> None:
    """Exercise deterministic method rules before any expensive model fitting."""
    features = ["a", "b", "c", "d"]
    forward = pd.Series([1.0, 2.0, 3.0, 4.0], index=features)
    reverse = pd.Series([4.0, 3.0, 2.0, 1.0], index=features)
    if not np.isclose(normalized_spearman_footrule_agreement(forward, forward, 4), 1.0):
        raise RuntimeError("Internal check failed: identical footrule ranks.")
    if not np.isclose(normalized_spearman_footrule_agreement(forward, reverse, 4), 0.0):
        raise RuntimeError("Internal check failed: reversed footrule ranks.")
    if _run_contains_rank_information(pd.Series([0.0] * 4), 4):
        raise RuntimeError(
            "Internal check failed: all-zero evidence marked informative."
        )
    if not _run_contains_rank_information(pd.Series([0.0, 0.0, 0.1, 0.0]), 4):
        raise RuntimeError(
            "Internal check failed: varying evidence marked uninformative."
        )

    empty_weights, _, empty_status, empty_n = _normalize_source_reliabilities(
        [{"source_reliability": np.nan}, {"source_reliability": 0.0}]
    )
    if (
        empty_status != "consensus_unavailable"
        or empty_n != 0
        or not np.allclose(empty_weights, 0.0)
    ):
        raise RuntimeError("Internal check failed: unavailable-source fallback.")
    single_weights, _, single_status, single_n = _normalize_source_reliabilities(
        [{"source_reliability": 0.8}, {"source_reliability": np.nan}]
    )
    if (
        single_status != "source_specific_only"
        or single_n != 1
        or not np.allclose(single_weights, [1.0, 0.0])
    ):
        raise RuntimeError("Internal check failed: one-source status.")

    expected_patterns = {
        (-1, 1): "-->",
        (2, 1): "o->",
        (1, 1): "<->",
        (2, 2): "o-o",
        (1, 2): "<-o",
        (1, -1): "<--",
    }
    for endpoints, expected in expected_patterns.items():
        if _pag_pattern(*endpoints) != expected:
            raise RuntimeError(
                "Internal check failed: PAG endpoint-pattern decoding."
            )
    synthetic_X = pd.DataFrame(
        {
            "A": ["x", "x", "y", "y"],
            "B": [0, 1, 0, 1],
        }
    )
    synthetic_y = np.asarray([0, 1, 1, 0], dtype=int)
    encoded, codebook = _encode_structural_data(
        synthetic_X, synthetic_y, ["A", "B"]
    )
    if encoded.shape != (4, 3) or codebook.empty:
        raise RuntimeError("Internal check failed: structural nominal encoding.")
    if _normalized_pattern_entropy(pd.Series(["o-o", "o-o"])) != 0.0:
        raise RuntimeError("Internal check failed: endpoint entropy.")

    shadow_train = pd.DataFrame(
        {"feature": ["A", "B", "A", np.nan], "other": [1, 2, 3, 4]}
    )
    shadow_eval = pd.DataFrame(
        {"feature": ["outside", "outside"], "other": [5, 6]}
    )
    replaced_train_a, replaced_eval_a = replace_raw_feature_from_training_marginal(
        shadow_train, shadow_eval, "feature", seed=123
    )
    replaced_train_b, replaced_eval_b = replace_raw_feature_from_training_marginal(
        shadow_train, shadow_eval, "feature", seed=123
    )
    if not replaced_train_a.equals(replaced_train_b) or not replaced_eval_a.equals(
        replaced_eval_b
    ):
        raise RuntimeError("Internal check failed: deterministic shadow replacement.")
    allowed_shadow_states = set(shadow_train["feature"].astype(str))
    if not set(replaced_eval_a["feature"].astype(str)).issubset(
        allowed_shadow_states
    ):
        raise RuntimeError("Internal check failed: shadow leaked evaluation support.")

    synthetic_parts = [
        OuterPartition(
            partition_id=partition_id,
            repeat_id=(partition_id - 1) // 2 + 1,
            fold_id=(partition_id - 1) % 2 + 1,
            split_seed=partition_id,
            train_idx=np.asarray([0, 1, 2, 3], dtype=int),
            test_idx=np.asarray([4, 5], dtype=int),
        )
        for partition_id in range(1, 5)
    ]
    calibrated_rows = []
    for part in synthetic_parts:
        for source in CREDAL_SOURCE_NAMES:
            for feature, value in [("a", 0.8), ("b", 0.5), ("c", 0.2)]:
                calibrated_rows.append(
                    {
                        "outer_partition": int(part.partition_id),
                        "model": "q",
                        "source": source,
                        "class_id": 0,
                        "feature": feature,
                        "relevance_percentile": value,
                    }
                )
    synthetic_common_support = {
        (int(part.partition_id), 0): ["q"] for part in synthetic_parts
    }
    synthetic_summary, _, synthetic_hasse = compute_credal_partial_order(
        calibrated=pd.DataFrame(calibrated_rows),
        partitions=synthetic_parts,
        common_support=synthetic_common_support,
        feature_cols=["a", "b", "c"],
        class_order=[0],
    )
    synthetic_levels = synthetic_summary.set_index("feature")[
        "partial_order_level"
    ].astype(int).to_dict()
    if synthetic_levels != {"a": 1, "b": 2, "c": 3}:
        raise RuntimeError("Internal check failed: credal partial-order levels.")
    synthetic_hasse_set = {
        (str(row.source_feature), str(row.target_feature))
        for row in synthetic_hasse.itertuples(index=False)
    }
    if synthetic_hasse_set != {("a", "b"), ("b", "c")}:
        raise RuntimeError("Internal check failed: Hasse transitive reduction.")

    observed = pd.DataFrame(
        [
            {
                "outer_partition": 1,
                "outer_repeat": 1,
                "outer_fold": 1,
                "split_seed": 1,
                "model": "q",
                "source": "SHAP",
                "class_id": 0,
                "feature": "a",
                "raw_statistic": 2.0,
            }
        ]
    )
    null = pd.DataFrame(
        [
            {
                "outer_partition": 1,
                "model": "q",
                "source": "SHAP",
                "class_id": 0,
                "feature": "a",
                "shadow_replicate": replicate,
                "raw_statistic": value,
            }
            for replicate, value in enumerate([0.0, 1.0, 3.0], 1)
        ]
    )
    calibrated_check = calibrate_shadow_relevance_percentiles(
        observed_statistics=observed,
        shadow_statistics=null,
        common_support={(1, 0): ["q"]},
        role="internal_check",
        n_shadow_replicates=3,
    )
    if not np.isclose(
        float(calibrated_check["relevance_percentile"].iloc[0]), 0.5
    ):
        raise RuntimeError("Internal check failed: finite shadow calibration.")
    cprint("[SELF-CHECK] deterministic method checks passed.")


def validate_or_write_resume_table(
    frame: pd.DataFrame,
    path: Path,
    resume: bool,
) -> None:
    if resume and path.exists():
        existing = pd.read_csv(path, encoding="utf-8-sig")
        try:
            pd.testing.assert_frame_equal(
                existing.reset_index(drop=True),
                frame.reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
        except AssertionError as exc:
            raise RuntimeError(
                f"Resume validation failed for existing table: {path}. "
                "The data or partition definition differs from the interrupted run."
            ) from exc
        return
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def load_prediction_benchmark_tables(
    output_dir: Path,
    expected_partition_ids: List[int],
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]]:
    tables_dir = output_dir / "tables"
    paths = {
        "raw": tables_dir / "model_selection_metrics_raw.csv",
        "by_split": tables_dir / "model_selection_metrics_by_outer_split.csv",
        "summary": tables_dir / "model_selection_metrics_summary.csv",
    }
    exists = {name: path.exists() for name, path in paths.items()}
    if not any(exists.values()):
        return None
    if not all(exists.values()):
        missing = [name for name, present in exists.items() if not present]
        raise RuntimeError(
            "Cannot resume from an incomplete Step 1 table set. Missing: "
            + ", ".join(missing)
        )
    metrics_raw = pd.read_csv(paths["raw"], encoding="utf-8-sig")
    metrics_by_split = pd.read_csv(paths["by_split"], encoding="utf-8-sig")
    metrics_summary = pd.read_csv(paths["summary"], encoding="utf-8-sig")
    observed_partitions = sorted(
        pd.to_numeric(metrics_by_split["seed"], errors="raise").astype(int).unique()
    )
    if observed_partitions != sorted(int(value) for value in expected_partition_ids):
        raise RuntimeError(
            "Existing Step 1 tables use different outer partitions: "
            f"observed={observed_partitions}, expected={sorted(expected_partition_ids)}."
        )
    available = set(available_model_names(CANDIDATE_MODELS))
    observed_models = set(metrics_by_split["model"].astype(str).unique())
    if observed_models != available:
        raise RuntimeError(
            "Existing Step 1 tables use a different candidate-model set: "
            f"observed={sorted(observed_models)}, expected={sorted(available)}."
        )
    best_model = select_best_model(metrics_summary)
    cprint(
        f"[RESUME] loaded completed Step 1 tables | leader={best_model} | "
        f"outer_partitions={len(observed_partitions)}"
    )
    return metrics_raw, metrics_by_split, metrics_summary, best_model


def legacy_weighted_main() -> None:
    t0 = time.time()
    if RUN_INTERNAL_METHOD_CHECKS:
        run_internal_method_checks()
    validate_runtime_requirements()
    if RUN_PROFILE.lower() == "smoke" and ENABLE_SHAP:
        cprint("[INFO] RUN_PROFILE='smoke': SHAP remains controlled by ENABLE_SHAP.")

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    resume_mode = RESUME_RUN_DIR is not None
    if resume_mode:
        output_dir = Path(str(RESUME_RUN_DIR)).expanduser().resolve()
        if not output_dir.exists():
            raise FileNotFoundError(f"RESUME_RUN_DIR does not exist: {output_dir}")
    else:
        output_dir = Path(BASE_OUTPUT_DIR) / f"ads_reliable_xai_{RUN_PROFILE}_{run_tag}"
    tables_dir = output_dir / "tables"
    figs_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    cprint("\n" + "#" * 110)
    cprint("ADS RELIABLE EXPLANATION AND STRUCTURAL COMPATIBILITY WORKFLOW")
    cprint("#" * 110)
    cprint(f"EXCEL_PATH = {EXCEL_PATH}")
    cprint(f"OUTPUT_DIR = {output_dir}")
    cprint(f"RUN_PROFILE = {RUN_PROFILE}")
    cprint(f"RESUME_MODE = {resume_mode}")
    if "tabpfn" in [m.lower() for m in CANDIDATE_MODELS]:
        cprint(f"TABPFN_LOCAL_CKPT = {TABPFN_LOCAL_CKPT}")
        cprint(f"FORCE_TABPFN_CUDA = {FORCE_TABPFN_CUDA}")
        cprint(f"ISOLATE_TABPFN_SHAP = {ISOLATE_TABPFN_SHAP}")

    df = load_ads_table(EXCEL_PATH, use_hour_group=True)
    if EXPECTED_N_SAMPLES is not None and int(len(df)) != int(EXPECTED_N_SAMPLES):
        raise RuntimeError(
            f"Loaded n_samples={len(df)}, but EXPECTED_N_SAMPLES={EXPECTED_N_SAMPLES}. "
            "Check whether the Excel file and cleaning rules match the manuscript dataset."
        )
    schema = get_feature_schema(df)
    X_all, y_all = split_xy(df, schema)
    partitions = make_outer_partitions(y_all, RUN_PROFILE)
    partition_ids = [int(part.partition_id) for part in partitions]
    cprint(
        f"OUTER PARTITIONS = {len(partitions)} "
        f"({max(part.repeat_id for part in partitions)} repeats x "
        f"{max(part.fold_id for part in partitions)} folds)"
    )

    dataset_summary = summarize_dataset(df)
    feature_support = category_support_table(df, schema)
    partition_manifest = outer_partition_manifest(partitions, y_all)
    validate_or_write_resume_table(
        dataset_summary,
        tables_dir / "dataset_class_distribution.csv",
        resume=resume_mode,
    )
    validate_or_write_resume_table(
        feature_support,
        tables_dir / "feature_value_support.csv",
        resume=resume_mode,
    )
    validate_or_write_resume_table(
        partition_manifest,
        tables_dir / "outer_partition_manifest.csv",
        resume=resume_mode,
    )
    if OUTPUT_FIGURES:
        save_dataset_figures(dataset_summary, feature_support, figs_dir)

    resumed_benchmark = (
        load_prediction_benchmark_tables(output_dir, partition_ids)
        if resume_mode
        else None
    )
    if resumed_benchmark is None:
        metrics_raw, metrics_by_split, metrics_summary, best_model = run_prediction_benchmark(
            X_all=X_all,
            y_all=y_all,
            schema=schema,
            partitions=partitions,
            output_dir=output_dir,
        )
    else:
        metrics_raw, metrics_by_split, metrics_summary, best_model = resumed_benchmark

    if RUN_PROFILE.lower() == "smoke" or EXPLAIN_ONLY_BEST_MODEL:
        explanation_models_by_seed = select_explanation_models_by_seed(
            metrics_by_split=metrics_by_split,
            seeds=partition_ids,
            max_models=1,
        )
    elif USE_ONE_SE_ADMISSIBLE_MODEL_SET:
        explanation_models_by_seed = select_explanation_models_by_seed(
            metrics_by_split=metrics_by_split,
            seeds=partition_ids,
            max_models=None,
        )
    else:
        available_models = available_model_names(CANDIDATE_MODELS)
        explanation_models_by_seed = {
            int(seed): list(available_models) for seed in partition_ids
        }
    explanation_model_rows = []
    for seed in partition_ids:
        for rank, model_name in enumerate(explanation_models_by_seed[int(seed)], 1):
            selected_metric = metrics_by_split.loc[
                (metrics_by_split["seed"].astype(int) == int(seed))
                & (metrics_by_split["model"].astype(str) == str(model_name))
            ]
            if selected_metric.empty:
                raise RuntimeError(
                    f"Missing inner cross-validation record for seed={seed}, model={model_name}."
                )
            selected_metric = selected_metric.iloc[0]
            explanation_model_rows.append(
                {
                    "seed": int(seed),
                    "one_se_set_order": int(rank),
                    "explanation_model": str(model_name),
                    "model_display": MODEL_DISPLAY_NAMES.get(
                        str(model_name), str(model_name)
                    ),
                    "inner_cv_mean_balanced_accuracy": float(
                        selected_metric["mean_balanced_accuracy"]
                    ),
                    "inner_cv_se_balanced_accuracy": float(
                        selected_metric["se_balanced_accuracy"]
                    ),
                    "one_se_cutoff_balanced_accuracy": float(
                        selected_metric["one_se_cutoff_balanced_accuracy"]
                    ),
                    "inner_cv_mean_macro_f1": float(selected_metric["mean_macro_f1"]),
                    "inner_cv_mean_accuracy": float(selected_metric["mean_accuracy"]),
                }
            )
    validate_or_write_resume_table(
        pd.DataFrame(explanation_model_rows),
        tables_dir / "explanation_model_set_by_outer_split.csv",
        resume=resume_mode,
    )

    explanation_outputs = run_explanation_workflow(
        X_all=X_all,
        y_all=y_all,
        schema=schema,
        partitions=partitions,
        explanation_models_by_seed=explanation_models_by_seed,
        output_dir=output_dir,
    )

    if ENABLE_FUSION_RECOVERY_SIMULATION:
        n_simulation_replicates = (
            FUSION_SIMULATION_REPLICATES_SMOKE
            if RUN_PROFILE.lower() == "smoke"
            else FUSION_SIMULATION_REPLICATES_FULL
        )
        cprint(
            f"[SIMULATION] known-relevance fusion recovery | "
            f"replicates={n_simulation_replicates}"
        )
        _, simulation_summary = run_fusion_recovery_simulation(
            output_dir=output_dir,
            n_replicates=int(n_simulation_replicates),
        )
        if OUTPUT_FIGURES:
            save_fusion_recovery_simulation_figure(simulation_summary, figs_dir)

    meta = {
        "workflow": (
            "ads_reliable_xai_crossfitted_consensus_with_"
            "data_driven_structural_compatibility_audit"
        ),
        "excel_path": EXCEL_PATH,
        "excel_sha256": file_sha256(EXCEL_PATH),
        "workflow_script": str(Path(__file__).resolve()),
        "workflow_script_sha256": file_sha256(str(Path(__file__).resolve())),
        "output_dir": str(output_dir),
        "run_profile": RUN_PROFILE,
        "execution_recovery": {
            "resume_mode": bool(resume_mode),
            "resume_run_dir": str(output_dir) if resume_mode else None,
            "outer_partition_checkpoints": bool(
                ENABLE_OUTER_PARTITION_CHECKPOINTS
            ),
            "tabpfn_shap_call_cache": (
                bool(ENABLE_TABPFN_SHAP_CACHE)
                if "tabpfn" in [name.lower() for name in CANDIDATE_MODELS]
                else False
            ),
            "durable_tabpfn_worker_root": (
                str(output_dir / "checkpoints" / "tabpfn_shap")
                if "tabpfn" in [name.lower() for name in CANDIDATE_MODELS]
                else None
            ),
            "checkpoint_format_version": int(CHECKPOINT_FORMAT_VERSION),
        },
        "n_samples": int(len(df)),
        "expected_n_samples": None
        if EXPECTED_N_SAMPLES is None
        else int(EXPECTED_N_SAMPLES),
        "data_cleaning": {
            "n_rows_in_workbook": int(df.attrs.get("n_rows_in_workbook", len(df))),
            "exclude_unknown_vehicle_age": bool(
                df.attrs.get("exclude_unknown_vehicle_age", EXCLUDE_UNKNOWN_VEHICLE_AGE)
            ),
            "n_removed_unknown_vehicle_age": int(
                df.attrs.get("n_removed_unknown_vehicle_age", 0)
            ),
            "n_rows_analyzed": int(len(df)),
        },
        "feature_columns": schema.feature_cols,
        "numeric_columns": schema.numeric_cols,
        "categorical_columns": schema.categorical_cols,
        "candidate_models": CANDIDATE_MODELS,
        "model_specific_preprocessing": {
            model: preprocessing_mode(model)
            for model in available_model_names(CANDIDATE_MODELS)
        },
        "model_specific_imbalance_handling": {
            model: imbalance_handling_mode(model)
            for model in available_model_names(CANDIDATE_MODELS)
        },
        "inner_cross_validation_summary_leader": best_model,
        "explanation_models_by_outer_split": {
            str(seed): models for seed, models in explanation_models_by_seed.items()
        },
        "outer_partition_ids": partition_ids,
        "data_partitioning": {
            "outer_scheme": "stratified K-fold cross-fitting",
            "outer_n_splits": int(max(part.fold_id for part in partitions)),
            "outer_n_repeats": int(max(part.repeat_id for part in partitions)),
            "outer_random_seed": int(OUTER_RANDOM_SEED),
            "inner_cross_validation_folds_requested": int(INNER_CV_FOLDS),
            "model_selection_partition": "stratified cross-validation within outer training data only",
            "model_set_rule": "split-specific one-standard-error rule on mean Balanced Accuracy",
            "explanation_evaluation_partition": "complete outer holdout",
        },
        "tabpfn_local_checkpoint": (
            TABPFN_LOCAL_CKPT
            if "tabpfn" in [name.lower() for name in CANDIDATE_MODELS]
            else None
        ),
        "force_tabpfn_cuda": (
            bool(FORCE_TABPFN_CUDA)
            if "tabpfn" in [name.lower() for name in CANDIDATE_MODELS]
            else False
        ),
        "tabpfn_shap_execution": {
            "enabled": (
                "tabpfn" in [name.lower() for name in CANDIDATE_MODELS]
            ),
            "isolated_spawn_process": (
                bool(ISOLATE_TABPFN_SHAP)
                if "tabpfn" in [name.lower() for name in CANDIDATE_MODELS]
                else False
            ),
            "inference_precision": (
                "float32" if TABPFN_FORCE_FP32_INFERENCE else "auto"
            ),
            "internal_memory_saving": bool(TABPFN_INTERNAL_MEMORY_SAVING),
            "external_batching_rule": (
                "permitted only when same-shape repeatability and unbatched-versus-"
                "batched probability equivalence both pass the prespecified tolerance"
            ),
            "external_initial_prediction_batch_size": int(
                TABPFN_SHAP_INITIAL_PREDICT_BATCH_SIZE
            ),
            "external_minimum_prediction_batch_size": int(
                TABPFN_SHAP_MIN_PREDICT_BATCH_SIZE
            ),
            "unbatched_internal_memory_retries": int(TABPFN_SHAP_UNBATCHED_RETRIES),
            "reconstruction_tolerance": float(TABPFN_SHAP_RECONSTRUCTION_TOL),
            "batch_equivalence_probe_rows": int(
                TABPFN_SHAP_BATCH_EQUIVALENCE_PROBE_ROWS
            ),
            "cpu_fallback_allowed": bool(TABPFN_SHAP_ALLOW_CPU_FALLBACK),
        },
        "environment": collect_environment_metadata(),
        "diagnostic_settings": {
            "enable_calibration_diagnostics": bool(ENABLE_CALIBRATION_DIAGNOSTICS),
            "calibration_n_bins": int(CALIBRATION_N_BINS),
            "use_one_se_admissible_model_set": bool(USE_ONE_SE_ADMISSIBLE_MODEL_SET),
            "admissible_model_max_models": None,
            "model_set_selection": (
                "within each outer split, one-standard-error selection from inner cross-validation only"
            ),
            "enable_negative_control": bool(ENABLE_NEGATIVE_CONTROL),
            "negative_control_feature": NEGATIVE_CONTROL_FEATURE,
            "enable_firm": bool(ENABLE_FIRM),
            "shap_background_size": int(SHAP_BACKGROUND_SIZE),
            "enable_shap_background_sensitivity": bool(
                ENABLE_SHAP_BACKGROUND_SENSITIVITY
            ),
            "shap_background_sensitivity_sizes": [
                int(v) for v in SHAP_BACKGROUND_SENSITIVITY_SIZES
            ],
            "shap_permutation_cycles_per_observation": int(SHAP_PERMUTATION_CYCLES),
            "shap_evaluation_rows": "complete holdout partition"
            if SHAP_EVAL_SIZE is None
            else int(SHAP_EVAL_SIZE),
            "ice_evaluation_rows": "complete holdout partition",
            "pfi_evaluation_rows": "complete holdout partition for every permutation",
            "pfi_permutations_per_feature_and_run": int(PERMUTATION_REPEATS),
            "loco_evaluation_rows": "complete holdout partition",
            "firm_evaluation_rows": "complete holdout partition",
            "firm_discrete_estimator": "class-balanced support-weighted SD of empirical conditional mean class probabilities",
            "firm_continuous_estimator": "class-balanced weighted linear conditional-mean approximation; used for the Gaussian negative control",
            "firm_continuous_features": list(FIRM_CONTINUOUS_FEATURES),
            "pdp_evaluation_rows": "complete holdout partition",
            "fusion_sources": [
                "SHAP",
                "classwise Brier-loss PFI",
                "classwise Brier-loss LOCO",
                "class-specific FIRM observational relevance",
            ],
            "firm_role": "observed-distribution association between a feature and fitted class probability; not a causal effect or direct-reliance measure",
            "pdp_role": "average class-probability response derived from the shared ICE prediction matrix outside fusion",
            "ice_role": "training-support-weighted RMS heterogeneity of centered individual response curves outside fusion",
            "response_profile_prediction_basis": "PDP and ICE are derived from one shared intervention prediction matrix",
            "response_profile_aggregation": "admitted models are averaged within each outer split before equal-weight aggregation across splits",
            "enable_weight_sensitivity_table": bool(ENABLE_WEIGHT_SENSITIVITY_TABLE),
            "weight_sensitivity_top_k": int(WEIGHT_SENSITIVITY_TOP_K),
            "explain_all_features": bool(EXPLAIN_ALL_FEATURES),
            "display_top_k_features": int(DISPLAY_TOP_K_FEATURES),
            "source_weight_calibration_folds": int(SHADOW_INNER_CV_FOLDS),
            "source_weight_calibration_partition": (
                "stratified inner validation folds drawn only from each outer "
                "training partition"
            ),
            "main_source_weighting": (
                "outer-training-internal cross-fitted class-by-source reliability "
                "weighting; weights are constant across features"
            ),
            "weight_sensitivity_schemes": [
                "cross-fitted reliability weighting",
                "equal source weighting",
                "median normalized rank",
                "legacy feature-specific reliability-conflict discount",
            ],
            "source_dependence_diagnostic": "pairwise Spearman correlation and absolute disagreement among complete source rank signals",
            "source_ablation_diagnostic": "leave-one-source-out comparison against the full cross-fitted consensus",
            "source_weight_rule": (
                "for each outer partition and class, source reliability is learned "
                "from inner-validation explanations computed wholly within the "
                "outer training data; the outer holdout never enters source-weight "
                "estimation, and the same source weights apply to every feature"
            ),
            "rank_agreement_rule": (
                "normalized Spearman footrule over complete feature-rank vectors; "
                "identical rankings equal one and exact reverse rankings equal zero"
            ),
            "uninformative_source_rule": (
                "runs without finite machine-distinguishable feature evidence are "
                "excluded from agreement estimation and reduce information availability"
            ),
            "consensus_availability_rule": (
                "at least two sources with positive estimable reliability are required "
                "for multi-source consensus; one source is reported as source-specific, "
                "and no automatic equal-weight fallback is used"
            ),
            "conflict_rule": "weighted absolute source deviation from the held-out consensus is reported as ambiguity and does not enter the primary source weights",
            "repeatability_rule": "rank-vector repeatability across the reference folds is combined geometrically with within-fold model agreement when both are estimable",
            "class_balance_rule": "for each one-vs-rest class, the focal class and its complement each receive one half of the empirical evaluation mass across SHAP, PFI, LOCO, and FIRM",
            "uncertainty_summary": "empirical outer-partition IQR and range, top-k inclusion frequency, pairwise dominance probability, and unanimous partial-order levels",
            "enable_fusion_recovery_simulation": bool(
                ENABLE_FUSION_RECOVERY_SIMULATION
            ),
            "fusion_recovery_simulation_replicates": int(
                FUSION_SIMULATION_REPLICATES_SMOKE
                if RUN_PROFILE.lower() == "smoke"
                else FUSION_SIMULATION_REPLICATES_FULL
            ),
            "fusion_recovery_simulation_scope": "known-relevance validation of the aggregation rule; not causal validation of individual explainers",
            "support_rule": "PDP and ICE values are restricted to training-observed support; centered ICE summaries use training value frequencies; neither output enters fusion",
            "model_set_sensitivity": (
                "the split-specific one-standard-error model set is compared with the "
                "split-specific inner-CV leader using the same outer evidence records"
            ),
            "structural_compatibility_audit": explanation_outputs.get(
                "structural_compatibility_audit", {}
            ),
        },
        "structural_compatibility_audit": explanation_outputs.get(
            "structural_compatibility_audit", {}
        ),
        "elapsed_min": float((time.time() - t0) / 60.0),
        "main_outputs": {
            "model_selection_metrics_summary": str(
                tables_dir / "model_selection_metrics_summary.csv"
            ),
            "model_selection_metrics_by_outer_split": str(
                tables_dir / "model_selection_metrics_by_outer_split.csv"
            ),
            "model_selection_calibration_curves": str(
                tables_dir / "model_selection_calibration_curves_raw.csv"
            ),
            "explanation_model_holdout_metrics_summary": str(
                tables_dir / "explanation_model_holdout_metrics_summary.csv"
            ),
            "explanation_model_holdout_calibration_curves": str(
                tables_dir / "explanation_model_holdout_calibration_curves_raw.csv"
            ),
            "explanation_coverage_audit": str(
                tables_dir / "explanation_coverage_audit.csv"
            ),
            "explanation_model_set": str(
                tables_dir / "explanation_model_set_by_outer_split.csv"
            ),
            "explanation_reliability_map": str(
                tables_dir / "explanation_reliability_map.csv"
            ),
            "shap_background_sensitivity": str(
                tables_dir / "shap_background_sensitivity_summary.csv"
            ),
            "crossfitted_consensus_ranking": str(
                tables_dir / "crossfitted_consensus_ranking.csv"
            ),
            "crossfitted_consensus_by_outer_partition": str(
                tables_dir / "crossfitted_consensus_by_outer_partition.csv"
            ),
            "crossfitted_source_weights": str(
                tables_dir / "crossfitted_source_weights.csv"
            ),
            "source_weight_calibration_shap": str(
                tables_dir / "source_weight_calibration_shap_raw.csv"
            ),
            "source_weight_calibration_pfi": str(
                tables_dir / "source_weight_calibration_pfi_raw.csv"
            ),
            "source_weight_calibration_loco": str(
                tables_dir / "source_weight_calibration_loco_raw.csv"
            ),
            "source_weight_calibration_firm": str(
                tables_dir / "source_weight_calibration_firm_raw.csv"
            ),
            "pairwise_feature_dominance": str(
                tables_dir / "pairwise_feature_dominance.csv"
            ),
            "class_balance_measure_sensitivity": str(
                tables_dir / "class_balance_measure_sensitivity.csv"
            ),
            "fusion_recovery_simulation_summary": str(
                tables_dir / "fusion_recovery_simulation_summary.csv"
            ),
            "weight_sensitivity_top_features": str(
                tables_dir / "weight_sensitivity_top_features.csv"
            ),
            "source_dependence_diagnostic": str(
                tables_dir / "source_dependence_diagnostic.csv"
            ),
            "source_ablation_sensitivity": str(
                tables_dir / "source_ablation_sensitivity.csv"
            ),
            "model_set_sensitivity": str(tables_dir / "model_set_sensitivity.csv"),
            "best_single_model_consensus": str(
                tables_dir / "best_single_model_consensus_ranking.csv"
            ),
            "loco_reliance": str(tables_dir / "loco_reliance_raw.csv"),
            "firm_observational_relevance": str(
                tables_dir / "firm_observational_relevance_raw.csv"
            ),
            "support_aware_pdp": str(tables_dir / "support_aware_pdp_raw.csv"),
            "ice_heterogeneity": str(tables_dir / "ice_heterogeneity_raw.csv"),
            "response_profile_summary": str(
                tables_dir / "response_profile_summary.csv"
            ),
            "negative_control_summary": str(
                tables_dir / "negative_control_summary.csv"
            ),
            "structural_compatibility_audit_dir": str(
                output_dir / "structural_compatibility_audit"
            ),
            "structural_compatibility_by_feature": str(
                output_dir
                / "structural_compatibility_audit"
                / "structural_compatibility_by_feature.csv"
            ),
            "predictive_structural_compatibility_matrix": str(
                output_dir
                / "structural_compatibility_audit"
                / "predictive_structural_compatibility_matrix.csv"
            ),
            "structural_setting_sensitivity": str(
                output_dir
                / "structural_compatibility_audit"
                / "structural_setting_sensitivity.csv"
            ),
            "structural_negative_control": str(
                output_dir
                / "structural_compatibility_audit"
                / "structural_null_replicate_summary.csv"
            ),
            "structural_category_support_audit": str(
                output_dir
                / "structural_compatibility_audit"
                / "structural_category_support_audit.csv"
            ),
            "figures_dir": str(figs_dir),
            "figure_export_formats": ["png", "pdf"],
            "figure_suite_note": (
                "The figures directory contains prediction, calibration, "
                "cross-fitted consensus intervals, top-k stability, ambiguity, "
                "source weights, pairwise dominance, fusion recovery, FIRM, "
                "source diagnostics, sensitivity, negative-control, PDP, and "
                "centered-ICE figures. The structural_compatibility_audit "
                "directory contains the PAG multiverse, endpoint, null-control, "
                "and predictive-structural figures."
            ),
        },
    }
    safe_json_dump(meta, output_dir / "run_meta.json")

    cprint("\n" + "=" * 110)
    cprint(f"DONE. elapsed = {(time.time() - t0) / 60.0:.2f} min")
    cprint(f"INNER-CV SUMMARY LEADER = {best_model}")
    cprint(f"OUTPUT_DIR = {output_dir}")
    cprint("=" * 110)


def parse_distributed_partition_ids(
    available_partition_ids: Sequence[int],
) -> List[int]:
    """Validate the execution mode and return the worker's assigned partitions."""
    mode = str(DISTRIBUTED_EXECUTION_MODE).strip().lower()
    if mode not in {"single", "worker", "aggregate"}:
        raise ValueError(
            "ADS_XAI_DISTRIBUTED_MODE must be one of: "
            "single, worker, aggregate."
        )
    available = [int(value) for value in available_partition_ids]
    if mode != "worker":
        if DISTRIBUTED_PARTITIONS_TEXT:
            raise ValueError(
                "ADS_XAI_DISTRIBUTED_PARTITIONS is accepted only in worker "
                "mode. Leave it empty for single or aggregate mode."
            )
        return available
    if not DISTRIBUTED_WORKER_ID:
        raise ValueError(
            "ADS_XAI_DISTRIBUTED_WORKER_ID must be nonempty in worker mode."
        )
    tokens = [
        token.strip()
        for token in str(DISTRIBUTED_PARTITIONS_TEXT).split(",")
        if token.strip()
    ]
    if not tokens:
        raise ValueError(
            "Worker mode requires ADS_XAI_DISTRIBUTED_PARTITIONS, for "
            "example 1,2,3."
        )
    try:
        assigned = list(dict.fromkeys(int(token) for token in tokens))
    except ValueError as exc:
        raise ValueError(
            "ADS_XAI_DISTRIBUTED_PARTITIONS must be a comma-separated list "
            "of integer outer-partition IDs."
        ) from exc
    unknown = sorted(set(assigned).difference(available))
    if unknown:
        raise ValueError(
            "Distributed worker requested unavailable outer partitions: "
            f"{unknown}; available={available}."
        )
    return assigned


def run_distributed_checkpoint_worker(
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    schema: FeatureSchema,
    partitions: Sequence[OuterPartition],
    admissible_models_by_partition: Mapping[int, Sequence[str]],
    output_dir: Path,
    dataset_identity: Mapping[str, object],
) -> Dict[str, object]:
    """Compute a disjoint checkpoint shard without running global inference."""
    worker_id = _safe_path_component(
        str(DISTRIBUTED_WORKER_ID), limit=40
    )
    worker_dir = (
        Path(output_dir) / "distributed_workers" / worker_id
    )
    worker_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    evidence = execute_credal_evidence_contract(
        X_all=X_all,
        y_all=y_all,
        schema=schema,
        partitions=partitions,
        admissible_models_by_partition=admissible_models_by_partition,
        output_dir=output_dir,
    )
    for key in [
        "outer_statistics",
        "training_statistics",
        "shadow_statistics",
    ]:
        frame = evidence.get(key, pd.DataFrame())
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise RuntimeError(
                "Distributed worker produced no required evidence for "
                f"{key}."
            )

    validity, coverage, common_support = build_credal_common_support(
        outer_statistics=evidence["outer_statistics"],
        training_statistics=evidence["training_statistics"],
        shadow_statistics=evidence["shadow_statistics"],
        partitions=partitions,
        admissible_models_by_partition=admissible_models_by_partition,
        feature_cols=schema.feature_cols,
        class_order=[0, 1, 2],
        n_shadow_replicates=int(
            shadow_replicate_count(RUN_PROFILE)
        ),
        expected_inner_folds_by_partition={
            int(part.partition_id): int(SHADOW_INNER_CV_FOLDS)
            for part in partitions
        },
    )
    empty_support = [
        {
            "outer_partition": int(partition_id),
            "class_id": int(class_id),
        }
        for (partition_id, class_id), models in common_support.items()
        if not models
    ]
    if empty_support:
        raise RuntimeError(
            "Distributed checkpoint shard is incomplete because at least "
            f"one class has empty common support: {empty_support}."
        )
    atomic_dataframe_to_csv(
        validity,
        worker_dir / "worker_source_validity.csv",
    )
    atomic_dataframe_to_csv(
        coverage,
        worker_dir / "worker_common_support_coverage.csv",
    )
    errors = evidence.get("errors", pd.DataFrame())
    if isinstance(errors, pd.DataFrame) and not errors.empty:
        atomic_dataframe_to_csv(
            errors,
            worker_dir / "worker_execution_messages.csv",
        )

    assigned_ids = [
        int(part.partition_id) for part in partitions
    ]
    evidence_rows = {
        key: int(len(value))
        for key, value in evidence.items()
        if isinstance(value, pd.DataFrame)
    }
    manifest: Dict[str, object] = {
        "status": "complete",
        "worker_id": str(DISTRIBUTED_WORKER_ID),
        "execution_mode": "worker",
        "assigned_outer_partitions": assigned_ids,
        "admissible_models_by_outer_partition": {
            str(partition_id): list(
                admissible_models_by_partition[int(partition_id)]
            )
            for partition_id in assigned_ids
        },
        "dataset_identity": dict(dataset_identity),
        "statistical_contract_version": (
            STATISTICAL_CONTRACT_VERSION
        ),
        "statistical_method_digest": credal_method_digest(schema),
        "execution_contract_version": EXECUTION_CONTRACT_VERSION,
        "execution_digest": credal_execution_digest(),
        "shadow_replicates": int(
            shadow_replicate_count(RUN_PROFILE)
        ),
        "evidence_row_counts": evidence_rows,
        "common_support_groups": int(len(common_support)),
        "started_at": started_at,
        "completed_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "elapsed_hours": float((time.time() - t0) / 3600.0),
        "script_sha256": file_sha256(
            str(Path(__file__).resolve())
        ),
    }
    atomic_json_dump(
        manifest,
        worker_dir / "worker_manifest.json",
    )
    cprint("\n" + "=" * 110)
    cprint(
        "DISTRIBUTED WORKER COMPLETE | "
        f"worker={DISTRIBUTED_WORKER_ID} | "
        f"outer_partitions={assigned_ids}"
    )
    cprint(
        "No final ranking or figures were produced on this worker. "
        "Merge its checkpoint shard into the aggregation computer."
    )
    cprint(f"WORKER OUTPUT = {output_dir}")
    cprint("=" * 110)
    return manifest


def main() -> None:
    t0 = time.time()
    if RUN_INTERNAL_METHOD_CHECKS:
        run_internal_method_checks()
    validate_runtime_requirements()

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    resume_mode = RESUME_RUN_DIR is not None
    if resume_mode:
        output_dir = Path(str(RESUME_RUN_DIR)).expanduser().resolve()
        if not output_dir.exists():
            if DISTRIBUTED_EXECUTION_MODE == "worker":
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(
                    f"RESUME_RUN_DIR does not exist: {output_dir}"
                )
    else:
        output_dir = (
            Path(BASE_OUTPUT_DIR)
            / f"ads_credal_shadow_{RUN_PROFILE}_{run_tag}"
        )
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cprint("\n" + "#" * 110)
    cprint("ADS CREDAL SPECIFICATION-ROBUST MULTI-EXPLAINER INFERENCE")
    cprint("#" * 110)
    cprint(f"EXCEL_PATH = {EXCEL_PATH}")
    cprint(f"OUTPUT_DIR = {output_dir}")
    cprint(f"RUN_PROFILE = {RUN_PROFILE}")
    cprint(f"RESUME_MODE = {resume_mode}")
    cprint(
        "DISTRIBUTED_EXECUTION = "
        f"{DISTRIBUTED_EXECUTION_MODE} | "
        f"worker_id={DISTRIBUTED_WORKER_ID}"
    )
    cprint(f"SHADOW_REPLICATES = {shadow_replicate_count(RUN_PROFILE)}")
    cprint(f"SHADOW_PARALLEL_WORKERS = {SHADOW_PARALLEL_WORKERS}")
    cprint(
        "TARGET_SHAP = "
        f"{TARGET_SHAP_CONTRACT_VERSION} | "
        f"plan={TARGET_SHAP_PERMUTATION_PLAN_VERSION}"
    )
    if "tabpfn" in [model.lower() for model in CANDIDATE_MODELS]:
        cprint(f"TABPFN_LOCAL_CKPT = {TABPFN_LOCAL_CKPT}")
        cprint(f"FORCE_TABPFN_CUDA = {FORCE_TABPFN_CUDA}")
        cprint(f"ISOLATE_TABPFN_SHAP = {ISOLATE_TABPFN_SHAP}")

    data = load_ads_table(EXCEL_PATH, use_hour_group=True)
    if EXPECTED_N_SAMPLES is not None and len(data) != int(EXPECTED_N_SAMPLES):
        raise RuntimeError(
            f"Loaded n_samples={len(data)}, but EXPECTED_N_SAMPLES="
            f"{EXPECTED_N_SAMPLES}. Check the workbook and cleaning rules."
        )
    schema = get_feature_schema(data)
    X_all, y_all = split_xy(data, schema)
    ordered_dataset_sha256 = dataframe_digest(
        data[
            [STABLE_RECORD_ID_COL]
            + list(schema.feature_cols)
            + [TARGET_COL]
        ]
    )
    feature_order_sha256 = stable_json_digest(list(schema.feature_cols))
    stable_id_hashes = stable_record_hash_table(data, schema)
    partitions = make_outer_partitions(y_all, RUN_PROFILE)
    partition_ids = [int(part.partition_id) for part in partitions]
    assigned_partition_ids = parse_distributed_partition_ids(
        partition_ids
    )
    cprint(
        f"OUTER PARTITIONS = {len(partitions)} "
        f"({max(part.repeat_id for part in partitions)} repeats x "
        f"{max(part.fold_id for part in partitions)} folds)"
    )
    if DISTRIBUTED_EXECUTION_MODE == "worker":
        cprint(
            "ASSIGNED OUTER PARTITIONS = "
            f"{assigned_partition_ids}"
        )

    dataset_summary = summarize_dataset(data)
    feature_support = category_support_table(data, schema)
    partition_manifest = outer_partition_manifest(partitions, y_all)
    validate_or_write_resume_table(
        dataset_summary,
        tables_dir / "dataset_class_distribution.csv",
        resume=resume_mode,
    )
    validate_or_write_resume_table(
        feature_support,
        tables_dir / "feature_value_support.csv",
        resume=resume_mode,
    )
    validate_or_write_resume_table(
        partition_manifest,
        tables_dir / "outer_partition_manifest.csv",
        resume=resume_mode,
    )
    validate_or_write_resume_table(
        stable_id_hashes,
        tables_dir / "stable_record_id_to_source_row_hash.csv",
        resume=resume_mode,
    )
    dataset_identity = {
        "excel_sha256": file_sha256(EXCEL_PATH),
        "ordered_dataset_sha256": ordered_dataset_sha256,
        "feature_order_sha256": feature_order_sha256,
        "stable_id_to_source_row_hash_digest": dataframe_digest(
            stable_id_hashes
        ),
        "n_records": int(len(data)),
    }
    identity_path = tables_dir / "dataset_identity.json"
    if resume_mode and identity_path.exists():
        prior_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if prior_identity != dataset_identity:
            raise RuntimeError(
                "Resume dataset identity does not match the current cleaned "
                "ordered analysis table."
            )
    else:
        atomic_json_dump(dataset_identity, identity_path)
    if OUTPUT_FIGURES:
        save_dataset_figures(dataset_summary, feature_support, figures_dir)

    resumed_benchmark = (
        load_prediction_benchmark_tables(output_dir, partition_ids)
        if resume_mode
        else None
    )
    if resumed_benchmark is None:
        (
            metrics_raw,
            metrics_by_split,
            metrics_summary,
            benchmark_leader,
        ) = run_prediction_benchmark(
            X_all=X_all,
            y_all=y_all,
            schema=schema,
            partitions=partitions,
            output_dir=output_dir,
        )
    else:
        (
            metrics_raw,
            metrics_by_split,
            metrics_summary,
            benchmark_leader,
        ) = resumed_benchmark

    if RUN_PROFILE.lower() == "smoke" or EXPLAIN_ONLY_BEST_MODEL:
        admissible_models = select_explanation_models_by_seed(
            metrics_by_split=metrics_by_split,
            seeds=partition_ids,
            max_models=1,
        )
    elif USE_ONE_SE_ADMISSIBLE_MODEL_SET:
        admissible_models = select_explanation_models_by_seed(
            metrics_by_split=metrics_by_split,
            seeds=partition_ids,
            max_models=None,
        )
    else:
        available_models = available_model_names(CANDIDATE_MODELS)
        admissible_models = {
            partition_id: list(available_models)
            for partition_id in partition_ids
        }

    model_set_rows = []
    for partition_id in partition_ids:
        partition_row = next(
            part for part in partitions if int(part.partition_id) == partition_id
        )
        for order, model_name in enumerate(admissible_models[partition_id], 1):
            selected = metrics_by_split.loc[
                (
                    pd.to_numeric(metrics_by_split["seed"], errors="coerce")
                    == partition_id
                )
                & (metrics_by_split["model"].astype(str) == str(model_name))
            ]
            if selected.empty:
                raise RuntimeError(
                    "Missing model-selection evidence for "
                    f"outer={partition_id}, model={model_name}."
                )
            row = selected.iloc[0]
            model_set_rows.append(
                {
                    "outer_partition": partition_id,
                    "outer_repeat": int(partition_row.repeat_id),
                    "outer_fold": int(partition_row.fold_id),
                    "one_se_set_order": int(order),
                    "model": str(model_name),
                    "model_display": MODEL_DISPLAY_NAMES.get(
                        str(model_name), str(model_name)
                    ),
                    "inner_cv_mean_balanced_accuracy": float(
                        row["mean_balanced_accuracy"]
                    ),
                    "inner_cv_se_balanced_accuracy": float(
                        row["se_balanced_accuracy"]
                    ),
                    "one_se_cutoff_balanced_accuracy": float(
                        row["one_se_cutoff_balanced_accuracy"]
                    ),
                }
            )
    model_set_table = pd.DataFrame(model_set_rows)
    validate_or_write_resume_table(
        model_set_table,
        tables_dir / "admissible_model_set_by_outer_partition.csv",
        resume=resume_mode,
    )

    if DISTRIBUTED_EXECUTION_MODE == "worker":
        assigned_set = set(assigned_partition_ids)
        worker_partitions = [
            part
            for part in partitions
            if int(part.partition_id) in assigned_set
        ]
        worker_admissible_models = {
            int(partition_id): list(
                admissible_models[int(partition_id)]
            )
            for partition_id in assigned_partition_ids
        }
        run_distributed_checkpoint_worker(
            X_all=X_all,
            y_all=y_all,
            schema=schema,
            partitions=worker_partitions,
            admissible_models_by_partition=(
                worker_admissible_models
            ),
            output_dir=output_dir,
            dataset_identity=dataset_identity,
        )
        return

    credal_outputs = run_credal_shadow_workflow(
        X_all=X_all,
        y_all=y_all,
        schema=schema,
        partitions=partitions,
        admissible_models_by_partition=admissible_models,
        output_dir=output_dir,
    )

    meta = {
        "workflow": "credal_specification_robust_multi_explainer_inference",
        "method_contract": {
            "predictive_model_admission": (
                "split-specific one-standard-error set from outer-training-only "
                "inner cross-validation using Balanced Accuracy"
            ),
            "sources": list(CREDAL_SOURCE_NAMES),
            "source_statistics": CREDAL_SOURCE_SPECS,
            "source_weighting": "none",
            "shadow_null": SHADOW_NULL_MODE,
            "shadow_replicates": int(shadow_replicate_count(RUN_PROFILE)),
            "shadow_construction": (
                "raw-feature replacement from the empirical marginal estimated "
                "on each inner-training fold; preprocessing and model parameters "
                "are refitted under a fixed predictive specification"
            ),
            "shadow_fold_counting": (
                "one complete inner cross-fitting run contributes one null statistic"
            ),
            "finite_calibration": (
                "p_tilde=(1+count(I_shadow>=I_observed))/(B+1); "
                "relevance_percentile=1-p_tilde"
            ),
            "finite_calibration_resolution": float(
                1.0 / (int(shadow_replicate_count(RUN_PROFILE)) + 1)
            ),
            "shap_calibration_estimator": (
                "target-coordinate Permutation SHAP with dynamic record-level "
                "permutation counts under the complete-path reference budget"
            ),
            "shap_display_estimator": (
                "official full-vector permutation SHAP; descriptive display "
                "only and never mixed with target-coordinate shadow calibration"
            ),
            "shap_permutation_plan": (
                "model-independent common_plan_v1 keyed by phase, outer "
                "partition, inner fold, stable record ID, and cycle"
            ),
            "relevance_scale_boundary": (
                "operational replacement-shadow percentile; not a p-value, "
                "probability, effect size, intervention effect, or causal effect"
            ),
            "common_support_rule": (
                "intersection of model specifications valid for all four sources, "
                "all classes, all original features, and every shadow replicate"
            ),
            "strict_preference_rule": (
                "feature j strictly precedes k only when the lower expected "
                "holdout-size-weighted relevance difference is positive across "
                "every source, outer fold, and admissible model path"
            ),
            "primary_output": (
                "partial order, maximal set, Hasse reduction, and lower/upper "
                "expected relevance percentiles"
            ),
            "pdp_ice_role": (
                "descriptive probability-response profiles outside ranking inference"
            ),
            "utility_validation": (
                "training-only order layers are deleted cumulatively and each "
                "fixed model specification is refitted; utility is evaluated only "
                "on the corresponding outer holdout"
            ),
            "strict_completion_policy": (
                "empty primary common support or any incomplete fixed-path "
                "utility validation stops the run; partial evidence is not reported "
                "as a completed analysis"
            ),
            "structural_audit_role": (
                "independent observational FCI/PAG compatibility diagnostic; "
                "does not alter relevance ordering and supports no causal claim"
            ),
        },
        "data": {
            "excel_path": EXCEL_PATH,
            "excel_sha256": file_sha256(EXCEL_PATH),
            "n_samples": int(len(data)),
            "expected_n_samples": (
                None
                if EXPECTED_N_SAMPLES is None
                else int(EXPECTED_N_SAMPLES)
            ),
            "feature_columns": list(schema.feature_cols),
            "numeric_columns": list(schema.numeric_cols),
            "categorical_columns": list(schema.categorical_cols),
            "ordered_dataset_sha256": ordered_dataset_sha256,
            "feature_order_sha256": feature_order_sha256,
            "stable_id_to_source_row_hash_digest": dataframe_digest(
                stable_id_hashes
            ),
        },
        "partitioning": {
            "outer_scheme": "stratified K-fold cross-fitting",
            "outer_partitions": int(len(partitions)),
            "outer_repeats": int(max(part.repeat_id for part in partitions)),
            "outer_folds_per_repeat": int(
                max(part.fold_id for part in partitions)
            ),
            "inner_folds": int(INNER_CV_FOLDS),
            "outer_random_seed": int(OUTER_RANDOM_SEED),
        },
        "candidate_models": list(CANDIDATE_MODELS),
        "available_models": available_model_names(CANDIDATE_MODELS),
        "benchmark_summary_leader": str(benchmark_leader),
        "admissible_models_by_outer_partition": {
            str(key): list(value) for key, value in admissible_models.items()
        },
        "execution": {
            "run_profile": RUN_PROFILE,
            "resume_mode": bool(resume_mode),
            "resume_run_dir": str(output_dir) if resume_mode else None,
            "checkpoint_format_version": int(SHADOW_CHECKPOINT_FORMAT_VERSION),
            "statistical_contract_version": (
                STATISTICAL_CONTRACT_VERSION
            ),
            "statistical_method_digest": credal_method_digest(schema),
            "execution_contract_version": EXECUTION_CONTRACT_VERSION,
            "execution_digest": credal_execution_digest(),
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": file_sha256(str(Path(__file__).resolve())),
            "tabpfn_enabled": (
                "tabpfn" in [name.lower() for name in CANDIDATE_MODELS]
            ),
            "tabpfn_checkpoint": (
                TABPFN_LOCAL_CKPT
                if "tabpfn" in [name.lower() for name in CANDIDATE_MODELS]
                else None
            ),
            "tabpfn_checkpoint_sha256": (
                file_sha256(TABPFN_LOCAL_CKPT)
                if "tabpfn" in [name.lower() for name in CANDIDATE_MODELS]
                else None
            ),
            "force_tabpfn_cuda": (
                bool(FORCE_TABPFN_CUDA)
                if "tabpfn" in [name.lower() for name in CANDIDATE_MODELS]
                else False
            ),
            "environment": collect_environment_metadata(),
        },
        "structural_compatibility_audit": credal_outputs.get(
            "structural_compatibility_audit", {}
        ),
        "elapsed_min": float((time.time() - t0) / 60.0),
        "main_outputs": {
            "admissible_model_set": str(
                tables_dir / "admissible_model_set_by_outer_partition.csv"
            ),
            "common_support_coverage": str(
                tables_dir / "credal_common_support_coverage.csv"
            ),
            "calibrated_relevance": str(
                tables_dir / "outer_holdout_shadow_calibrated_relevance.csv"
            ),
            "credal_partial_order": str(
                tables_dir / "credal_feature_intervals_and_partial_order.csv"
            ),
            "hasse_edges": str(tables_dir / "credal_hasse_edges.csv"),
            "training_only_orders": str(
                tables_dir
                / "training_only_partial_orders_by_outer_partition.csv"
            ),
            "deletion_validation": str(
                tables_dir / "training_order_deletion_validation_by_repeat.csv"
            ),
            "figures": str(figures_dir),
            "output_manifest": str(output_dir / "output_manifest.csv"),
        },
    }
    safe_json_dump(meta, output_dir / "run_meta.json")
    manifest_path = write_output_manifest(output_dir)

    cprint("\n" + "=" * 110)
    cprint(f"DONE. elapsed = {(time.time() - t0) / 60.0:.2f} min")
    cprint(f"INNER-CV SUMMARY LEADER = {benchmark_leader}")
    cprint(f"OUTPUT_DIR = {output_dir}")
    cprint(f"OUTPUT MANIFEST = {manifest_path}")
    cprint("=" * 110)


if __name__ == "__main__":
    mp.freeze_support()
    main()
