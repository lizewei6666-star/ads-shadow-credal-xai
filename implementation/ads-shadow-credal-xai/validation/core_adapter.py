from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import Future, ProcessPoolExecutor as NativeProcessPoolExecutor
import importlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .common import SOURCES


class InlineExecutor:
    """Run unchanged core worker functions locally, including their initializer."""

    def __init__(self, *args, initializer=None, initargs=(), **kwargs):
        if initializer is not None:
            initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, function, *args, **kwargs):
        future = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future


def _initialize_validated_worker(settings, original_initializer, original_initargs):
    """Install and verify the parent contract before a spawned worker runs."""
    core = get_core()
    for key, value in settings.items():
        setattr(core, key, value)
    observed = {key: getattr(core, key) for key in settings}
    if observed != settings:
        raise RuntimeError(f"Spawned worker contract mismatch: {observed!r} != {settings!r}")
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[name] = "1"
    if original_initializer is not None:
        original_initializer(*original_initargs)


class ValidatedProcessPoolFactory:
    """Create spawn-safe pools whose workers inherit the configured core contract."""

    def __init__(self, worker_settings):
        self.worker_settings = dict(worker_settings)

    def __call__(self, *args, initializer=None, initargs=(), **kwargs):
        return NativeProcessPoolExecutor(
            *args,
            initializer=_initialize_validated_worker,
            initargs=(self.worker_settings, initializer, tuple(initargs)),
            **kwargs,
        )


def worker_contract_snapshot(keys):
    """Return configured values from a spawned process for integration tests."""
    core = get_core()
    return {key: getattr(core, key) for key in keys}


def get_core():
    return importlib.import_module("analysis.run_predictive_credal")


def preflight(cfg):
    imports = ["numpy", "pandas", "scipy", "sklearn", "shap"]
    imports += [dict(xgb="xgboost", lgbm="lightgbm", catboost="catboost")[m]
                for m in cfg["models"] if m in {"xgb", "lgbm", "catboost"}]
    for name in imports:
        try:
            importlib.import_module(name)
        except Exception as exc:
            raise RuntimeError(f"Required package {name} failed to import; the learner pool will not be silently reduced") from exc


@contextmanager
def configured_core(cfg, split_seed, shadow_jobs=1):
    if type(shadow_jobs) is not int or not 1 <= shadow_jobs <= 4:
        raise ValueError("shadow_jobs must be an integer from 1 to 4")
    core = get_core()
    s = cfg["settings"]
    worker_settings = {
        "RUN_PROFILE": "full", "CANDIDATE_MODELS": list(cfg["models"]),
        "OUTER_N_SPLITS_FULL": s["outer_folds"], "OUTER_N_REPEATS_FULL": 1,
        "OUTER_RANDOM_SEED": int(split_seed), "INNER_CV_FOLDS": s["inner_folds"],
        "SHADOW_INNER_CV_FOLDS": s["inner_folds"], "SHADOW_REPLICATES_FULL": s["shadows"],
        "SHAP_BACKGROUND_SIZE": s["shap_background"], "SHAP_PERMUTATION_CYCLES": s["shap_cycles"],
        "PERMUTATION_REPEATS": s["pfi_repeats"], "MODEL_N_JOBS": 1,
        "SHADOW_PARALLEL_WORKERS": shadow_jobs, "LOCO_CACHE_PARALLEL_WORKERS": shadow_jobs,
        "OUTPUT_FIGURES": False, "ENABLE_PDP": False, "ENABLE_ICE": False,
        "ENABLE_STRUCTURAL_COMPATIBILITY_AUDIT": False,
        "ENABLE_TRAINING_ONLY_UTILITY_VALIDATION": False,
        "AUTO_DISCOVER_LEGACY_RESUME_SCRIPTS": False,
        "ENABLE_TARGET_SHAP_EQUIVALENCE_AUDIT": cfg["profile"] in {"formal", "targeted"},
        "ENABLE_TARGET_SHAP_PLAN_SENSITIVITY_AUDIT": cfg["profile"] in {"formal", "targeted"},
    }
    changes = dict(worker_settings)
    changes["ProcessPoolExecutor"] = (
        InlineExecutor if shadow_jobs == 1 else ValidatedProcessPoolFactory(worker_settings)
    )
    previous = {key: getattr(core, key) for key in changes}
    try:
        for key, value in changes.items():
            setattr(core, key, value)
        yield core
    finally:
        for key, value in previous.items():
            setattr(core, key, value)


def schema_for(core, x):
    features = [c for c in x.columns if c != core.STABLE_RECORD_ID_COL]
    return core.FeatureSchema(features, [], features)


def validate_partitions(y, parts, inner_folds):
    if set(np.unique(y)) != {0, 1, 2}:
        raise ValueError("A generated dataset is missing an outcome class")
    seen = []
    for part in parts:
        if set(part.train_idx) & set(part.test_idx):
            raise ValueError("Training/holdout overlap")
        if np.bincount(y[part.train_idx], minlength=3).min() < inner_folds:
            raise ValueError("Insufficient training class counts; inner folds must not be silently reduced")
        if np.bincount(y[part.test_idx], minlength=3).min() < 1:
            raise ValueError("Missing holdout class")
        seen.extend(part.test_idx.tolist())
    if sorted(seen) != list(range(len(y))):
        raise ValueError("Outer holdouts must partition the complete dataset")


def collect_evidence(core, x, y, schema, parts, admitted, out: Path):
    evidence = core.execute_credal_evidence_contract(x, y, schema, parts, admitted, out)
    tables = out / "tables"
    for name, frame in evidence.items():
        frame.to_csv(tables / f"evidence_{name}.csv", index=False)
    validity, coverage, support = core.build_credal_common_support(
        evidence["outer_statistics"], evidence["training_statistics"], evidence["shadow_statistics"],
        parts, admitted, schema.feature_cols, [0, 1, 2], core.SHADOW_REPLICATES_FULL,
        {p.partition_id: core.INNER_CV_FOLDS for p in parts})
    validity.to_csv(tables / "validity.csv", index=False)
    coverage.to_csv(tables / "support_coverage.csv", index=False)
    if any(not v for v in support.values()):
        raise RuntimeError("Empty common support; see tables/validity.csv. This is not an empty order.")
    z = core.calibrate_shadow_relevance_percentiles(evidence["outer_statistics"], evidence["shadow_statistics"],
                                                  support, "outer_holdout_inference", core.SHADOW_REPLICATES_FULL)
    z_train = core.calibrate_shadow_relevance_percentiles(evidence["training_statistics"], evidence["shadow_statistics"],
                                                        support, "outer_training_order_construction", core.SHADOW_REPLICATES_FULL)
    expected = sum(len(v) for v in support.values()) * len(SOURCES) * len(schema.feature_cols)
    if len(z) != expected or len(z_train) != expected:
        raise RuntimeError("Calibration coverage mismatch")
    z.to_csv(tables / "calibrated_outer.csv", index=False)
    z_train.to_csv(tables / "calibrated_training.csv", index=False)
    return evidence, z, z_train, support
