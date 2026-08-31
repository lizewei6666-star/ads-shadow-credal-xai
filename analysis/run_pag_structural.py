# -*- coding: utf-8 -*-
"""PAG-based structural interpretation for the completed predictive analysis.

The program evaluates outcome adjacency, two-step skeleton proximity, and
recurrent bridge configurations across the prespecified FCI multiverse. Label
permutations provide the structural null reference. It reads the completed
predictive run but does not recompute or modify its credal results.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import importlib.metadata as importlib_metadata
import io
import json
import multiprocessing as mp
import os
import pickle
import tempfile
import time
import traceback
import uuid
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fci_pag_extension_mpl")
)

import numpy as np
import pandas as pd


# =============================================================================
# USER CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Command-line arguments override every path and the worker count.
EXCEL_PATH = str(PROJECT_ROOT / "data" / "ads_crashes.xlsx")
SOURCE_RUN_DIR = str(PROJECT_ROOT / "outputs" / "predictive_run")

# A new child directory is used. Existing paper results are never overwritten.
OUTPUT_SUBDIR = "pag_multiscale_extension_v1"

# Six workers is conservative on the PC1 environment used for the main run.
# Each worker uses one numerical thread because the environment variables above
# are fixed before NumPy and causal-learn are imported.
PARALLEL_WORKERS = 6
MAX_PENDING_TASKS = 12

# These values reproduce the completed structural audit exactly. They are also
# checked against run_meta.json before computation begins.
EXPECTED_N_SAMPLES = 1469
EXCLUDE_UNKNOWN_VEHICLE_AGE = True
EXPECTED_CAUSAL_LEARN_VERSION = "0.1.4.8"
STRUCTURAL_RANDOM_SEED = 20260723
OUTER_RANDOM_SEED = 42
STRUCTURAL_CI_TEST = "gsq"
STRUCTURAL_MAX_PATH_LENGTH = -1
NULL_REPLICATES = 99

CONTRACT_VERSION = "pag_multiscale_null_full_edges_v1"
CHECKPOINT_FORMAT_VERSION = 1
TARGET_COL = "Injury_3Class"
TARGET_NODE = "Injury_3Class"
STABLE_RECORD_ID_COL = "_stable_record_id"

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

# Columns required before Incident_Hour is converted to Incident_Hour_Group.
RAW_FEATURE_COLUMNS = [
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

PAG_ENDPOINT_LABELS = {
    -1: "tail",
    0: "none",
    1: "arrow",
    2: "circle",
    3: "star",
    4: "tail_and_arrow",
    5: "arrow_and_arrow",
}

RUN_KEY_COLUMNS = [
    "scope",
    "partition_id",
    "outer_repeat",
    "outer_fold",
    "split_seed",
    "alpha",
    "depth",
    "setting",
]

EDGE_COLUMNS = RUN_KEY_COLUMNS + [
    "null_replicate",
    "outcome_status",
    "n_records",
    "node_left",
    "node_right",
    "endpoint_left_code",
    "endpoint_right_code",
    "endpoint_left",
    "endpoint_right",
    "pag_pattern_left_to_right",
]

METRIC_COLUMNS = RUN_KEY_COLUMNS + [
    "null_replicate",
    "outcome_status",
    "n_records",
    "feature",
    "shortest_skeleton_distance",
    "direct_adjacency",
    "exact_two_step",
    "within_two_steps",
    "connected_to_outcome",
]

BRIDGE_COLUMNS = RUN_KEY_COLUMNS + [
    "null_replicate",
    "outcome_status",
    "n_records",
    "feature",
    "bridge_node",
]

RUN_AUDIT_COLUMNS = RUN_KEY_COLUMNS + [
    "null_replicate",
    "outcome_status",
    "n_records",
    "n_pag_edges",
    "target_degree",
    "fraction_features_within_two_steps",
    "captured_stdout_characters",
    "elapsed_seconds",
    "run_status",
]


# =============================================================================
# REPRODUCIBILITY UTILITIES
# =============================================================================


def cprint(message: str = "") -> None:
    print(message, flush=True)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def stable_json_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_seed_from_key(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(stable_json_digest(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def dataframe_digest(frame: pd.DataFrame) -> str:
    normalized = frame.reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(stable_json_digest(list(normalized.columns)).encode("ascii"))
    digest.update(
        stable_json_digest([str(dtype) for dtype in normalized.dtypes]).encode(
            "ascii"
        )
    )
    values_hash = pd.util.hash_pandas_object(
        normalized, index=True, categorize=True
    ).to_numpy(dtype=np.uint64, copy=False)
    digest.update(values_hash.tobytes())
    return digest.hexdigest()


def stable_record_hash_table(
    data: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Build the same version-robust row identity table as the source run."""
    value_columns = list(feature_columns) + [TARGET_COL]
    rows: List[Dict[str, object]] = []
    for _, row in data[[STABLE_RECORD_ID_COL] + value_columns].iterrows():
        payload = {column: row[column] for column in value_columns}
        rows.append(
            {
                STABLE_RECORD_ID_COL: int(row[STABLE_RECORD_ID_COL]),
                "source_row_sha256": stable_json_digest(payload),
            }
        )
    return pd.DataFrame(rows)


def file_sha256(path_like: str | Path, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path_like)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        temp.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_csv_dump(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        compression = "gzip" if path.suffix.lower() == ".gz" else None
        frame.to_csv(
            temp,
            index=False,
            encoding="utf-8-sig" if compression is None else "utf-8",
            compression=compression,
        )
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_pickle_gzip_dump(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with gzip.open(temp, "wb", compresslevel=5) as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def load_pickle_gzip(path: Path) -> object:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def package_version(distribution_name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def write_output_manifest(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    path = output_dir / "output_manifest.csv"
    rows: List[Dict[str, object]] = []
    for item in sorted(output_dir.rglob("*")):
        if not item.is_file() or item == path:
            continue
        relative = item.relative_to(output_dir)
        if relative.parts and relative.parts[0].lower() == "checkpoints":
            continue
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": int(item.stat().st_size),
                "sha256": file_sha256(item),
            }
        )
    atomic_csv_dump(pd.DataFrame(rows), path)
    return path


# =============================================================================
# DATA AND PARTITIONS
# =============================================================================


@dataclass(frozen=True)
class OuterPartition:
    partition_id: int
    repeat_id: int
    fold_id: int
    split_seed: int
    train_idx: np.ndarray
    test_idx: np.ndarray


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


def load_ads_table(excel_path: Path) -> pd.DataFrame:
    if not excel_path.exists():
        raise FileNotFoundError(f"ADS workbook not found: {excel_path}")
    frame = pd.read_excel(excel_path, sheet_name=0)
    required = RAW_FEATURE_COLUMNS + [TARGET_COL]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Workbook is missing required columns: {missing}")
    frame = frame[required].copy()

    if EXCLUDE_UNKNOWN_VEHICLE_AGE:
        unknown_age = (
            frame["VehicleAge_Group"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("unknown")
        )
        frame = frame.loc[~unknown_age].copy()

    frame = frame.loc[frame[TARGET_COL].notna()].copy()
    if frame[TARGET_COL].dtype.kind not in "iu":
        unknown_targets = sorted(
            set(frame[TARGET_COL].astype(str)) - set(TARGET_MAP)
        )
        if unknown_targets:
            raise ValueError(f"Unknown target labels: {unknown_targets}")
        frame[TARGET_COL] = frame[TARGET_COL].map(TARGET_MAP)
    frame[TARGET_COL] = frame[TARGET_COL].astype(int)

    for column in ["Incident_Month", "Is_Weekend", "Incident_Hour"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(
        subset=["Incident_Month", "Is_Weekend", "Incident_Hour"]
    ).copy()
    frame["Incident_Month"] = frame["Incident_Month"].astype(int)
    frame["Is_Weekend"] = frame["Is_Weekend"].astype(int)
    frame["Incident_Hour"] = frame["Incident_Hour"].astype(int)

    frame["Incident_Hour_Group"] = frame["Incident_Hour"].apply(
        map_hour_to_group
    )
    frame = frame.drop(columns=["Incident_Hour"])
    frame["Incident_Month"] = frame["Incident_Month"].astype(str)

    feature_columns = [column for column in frame.columns if column != TARGET_COL]
    for column in feature_columns:
        if column not in ["Incident_Month", "Is_Weekend", "Incident_Hour"]:
            frame[column] = (
                frame[column]
                .astype(str)
                .str.strip()
                .replace({"Unknown": "Other_Unknown", "Other": "Other_Unknown"})
            )

    output = frame.reset_index(drop=True)
    output.insert(
        0,
        STABLE_RECORD_ID_COL,
        np.arange(len(output), dtype=np.int64),
    )
    return output


def make_outer_partitions(
    y: np.ndarray,
    n_splits: int,
    n_repeats: int,
    random_seed: int,
) -> List[OuterPartition]:
    from sklearn.model_selection import RepeatedStratifiedKFold

    splitter = RepeatedStratifiedKFold(
        n_splits=int(n_splits),
        n_repeats=int(n_repeats),
        random_state=int(random_seed),
    )
    dummy = np.zeros((len(y), 1), dtype=float)
    partitions: List[OuterPartition] = []
    for zero_id, (train_idx, test_idx) in enumerate(splitter.split(dummy, y)):
        repeat_id = zero_id // int(n_splits) + 1
        fold_id = zero_id % int(n_splits) + 1
        partitions.append(
            OuterPartition(
                partition_id=zero_id + 1,
                repeat_id=repeat_id,
                fold_id=fold_id,
                split_seed=stable_seed_from_key(
                    "outer_partition",
                    int(random_seed),
                    repeat_id,
                    fold_id,
                ),
                train_idx=np.asarray(train_idx, dtype=int),
                test_idx=np.asarray(test_idx, dtype=int),
            )
        )
    return partitions


def partition_manifest(
    partitions: Sequence[OuterPartition], y: np.ndarray
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for partition in partitions:
        row: Dict[str, object] = {
            "outer_partition": int(partition.partition_id),
            "outer_repeat": int(partition.repeat_id),
            "outer_fold": int(partition.fold_id),
            "split_seed": int(partition.split_seed),
            "n_train": int(len(partition.train_idx)),
            "n_holdout": int(len(partition.test_idx)),
        }
        for class_id in sorted(np.unique(y)):
            row[f"n_train_class_{int(class_id)}"] = int(
                np.sum(y[partition.train_idx] == int(class_id))
            )
            row[f"n_holdout_class_{int(class_id)}"] = int(
                np.sum(y[partition.test_idx] == int(class_id))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def verify_partition_manifest(
    observed: pd.DataFrame, expected_path: Path
) -> None:
    if not expected_path.exists():
        raise FileNotFoundError(f"Partition manifest not found: {expected_path}")
    expected = pd.read_csv(expected_path)
    if list(observed.columns) != list(expected.columns):
        raise RuntimeError("Reconstructed partition-manifest columns do not match.")
    for column in observed.columns:
        left = observed[column].astype(str).to_numpy()
        right = expected[column].astype(str).to_numpy()
        if not np.array_equal(left, right):
            raise RuntimeError(
                f"Reconstructed outer partitions differ in column {column!r}."
            )


def encode_structural_data(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_columns: Sequence[str],
) -> np.ndarray:
    frame = X[list(feature_columns)].copy().reset_index(drop=True)
    frame[TARGET_NODE] = np.asarray(y, dtype=int)
    encoded_columns: List[np.ndarray] = []
    for column in list(feature_columns) + [TARGET_NODE]:
        values = frame[column].astype("string").fillna("__STRUCTURAL_MISSING__")
        codes, _ = pd.factorize(values, sort=True)
        if np.any(codes < 0):
            raise RuntimeError(f"Structural encoding failed for {column!r}.")
        encoded_columns.append(codes.astype(np.int64))
    return np.column_stack(encoded_columns).astype(np.int64)


# =============================================================================
# FCI AND PAG EXTRACTION
# =============================================================================


def pag_pattern(endpoint_left: int, endpoint_right: int) -> str:
    return {
        (-1, 1): "-->",
        (2, 1): "o->",
        (1, 1): "<->",
        (2, 2): "o-o",
        (1, 2): "<-o",
        (1, -1): "<--",
        (-1, -1): "---",
        (-1, 2): "--o",
        (2, -1): "o--",
    }.get((int(endpoint_left), int(endpoint_right)), "other")


def run_fci(
    encoded_data: np.ndarray,
    node_names: Sequence[str],
    alpha: float,
    depth: int,
    ci_test: str,
    max_path_length: int,
) -> Tuple[Any, Sequence[Any], str]:
    from causallearn.search.ConstraintBased.FCI import fci

    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            graph, edges = fci(
                encoded_data,
                independence_test_method=str(ci_test),
                alpha=float(alpha),
                depth=int(depth),
                max_path_length=int(max_path_length),
                verbose=False,
                show_progress=False,
                node_names=list(node_names),
            )
    except Exception as exc:
        raise RuntimeError(
            "FCI failed for "
            f"alpha={alpha}, depth={depth}, n={len(encoded_data)}."
        ) from exc
    if graph is None or not hasattr(graph, "graph"):
        raise RuntimeError("FCI did not return a valid PAG.")
    return graph, edges, stdout.getvalue()


def extract_pag_edges(
    graph: Any,
    node_names: Sequence[str],
    metadata: Mapping[str, object],
) -> pd.DataFrame:
    matrix = np.asarray(graph.graph, dtype=int)
    if matrix.shape != (len(node_names), len(node_names)):
        raise RuntimeError("PAG matrix shape does not match node names.")
    rows: List[Dict[str, object]] = []
    for left in range(len(node_names)):
        for right in range(left + 1, len(node_names)):
            endpoint_left = int(matrix[left, right])
            endpoint_right = int(matrix[right, left])
            if endpoint_left == 0 and endpoint_right == 0:
                continue
            row = dict(metadata)
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
                    "pag_pattern_left_to_right": pag_pattern(
                        endpoint_left, endpoint_right
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=EDGE_COLUMNS)


def undirected_adjacency_from_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=int)
    adjacency = np.logical_or(matrix != 0, matrix.T != 0)
    np.fill_diagonal(adjacency, False)
    return adjacency


def shortest_distances(adjacency: np.ndarray, source: int) -> np.ndarray:
    n_nodes = int(adjacency.shape[0])
    distances = np.full(n_nodes, np.inf, dtype=float)
    distances[int(source)] = 0.0
    queue: deque[int] = deque([int(source)])
    while queue:
        current = queue.popleft()
        for neighbor in np.flatnonzero(adjacency[current]):
            neighbor = int(neighbor)
            if not np.isfinite(distances[neighbor]):
                distances[neighbor] = distances[current] + 1.0
                queue.append(neighbor)
    return distances


def graph_multiscale_rows(
    graph: Any,
    feature_columns: Sequence[str],
    metadata: Mapping[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame, int, float]:
    matrix = np.asarray(graph.graph, dtype=int)
    adjacency = undirected_adjacency_from_matrix(matrix)
    target_index = len(feature_columns)
    distances = shortest_distances(adjacency, target_index)

    metric_rows: List[Dict[str, object]] = []
    bridge_rows: List[Dict[str, object]] = []
    for feature_index, feature in enumerate(feature_columns):
        distance = float(distances[feature_index])
        finite_distance: object = int(distance) if np.isfinite(distance) else np.nan
        metric_rows.append(
            {
                **metadata,
                "feature": str(feature),
                "shortest_skeleton_distance": finite_distance,
                "direct_adjacency": int(distance == 1.0),
                "exact_two_step": int(distance == 2.0),
                "within_two_steps": int(distance <= 2.0),
                "connected_to_outcome": int(np.isfinite(distance)),
            }
        )
        if distance == 2.0:
            common_neighbors = np.flatnonzero(
                np.logical_and(
                    adjacency[feature_index], adjacency[target_index]
                )
            )
            for bridge_index in common_neighbors:
                bridge_index = int(bridge_index)
                if bridge_index in {feature_index, target_index}:
                    continue
                bridge_rows.append(
                    {
                        **metadata,
                        "feature": str(feature),
                        "bridge_node": str(feature_columns[bridge_index]),
                    }
                )

    target_degree = int(np.sum(adjacency[target_index]))
    fraction_within_two = float(
        np.mean([row["within_two_steps"] for row in metric_rows])
    )
    return (
        pd.DataFrame(metric_rows, columns=METRIC_COLUMNS),
        pd.DataFrame(bridge_rows, columns=BRIDGE_COLUMNS),
        target_degree,
        fraction_within_two,
    )


def graph_rows_from_edge_table(
    edges: pd.DataFrame,
    node_names: Sequence[str],
    feature_columns: Sequence[str],
    metadata: Mapping[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame, int, float]:
    node_to_index = {str(node): index for index, node in enumerate(node_names)}
    adjacency = np.zeros((len(node_names), len(node_names)), dtype=bool)
    for row in edges.itertuples(index=False):
        left = node_to_index[str(row.node_left)]
        right = node_to_index[str(row.node_right)]
        adjacency[left, right] = True
        adjacency[right, left] = True

    class MinimalGraph:
        pass

    graph = MinimalGraph()
    # Any nonzero symmetric entries are sufficient because only the skeleton is
    # used in this extension.
    graph.graph = adjacency.astype(int)
    return graph_multiscale_rows(graph, feature_columns, metadata)


# =============================================================================
# CHECKPOINTED NULL TASKS
# =============================================================================


def structural_setting_label(alpha: float, depth: int) -> str:
    depth_label = "unlimited" if int(depth) < 0 else str(int(depth))
    return f"alpha={float(alpha):.2f}|depth={depth_label}"


def checkpoint_paths(root: Path, task_key: str) -> Tuple[Path, Path]:
    return root / f"{task_key}.pkl.gz", root / f"{task_key}.json"


def checkpoint_is_valid(
    payload_path: Path,
    marker_path: Path,
    fingerprint: str,
    expected_null_replicates: int,
    expected_feature_count: int,
) -> bool:
    try:
        if not payload_path.exists() or not marker_path.exists():
            return False
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("fingerprint") != fingerprint:
            return False
        if marker.get("payload_sha256") != file_sha256(payload_path):
            return False
        payload = load_pickle_gzip(payload_path)
        if not isinstance(payload, Mapping):
            return False
        metrics = payload.get("metrics")
        run_audit = payload.get("run_audit")
        if not isinstance(metrics, pd.DataFrame) or not isinstance(
            run_audit, pd.DataFrame
        ):
            return False
        expected_metric_rows = int(expected_null_replicates) * int(
            expected_feature_count
        )
        if len(metrics) != expected_metric_rows:
            return False
        observed_replicates = set(
            pd.to_numeric(metrics["null_replicate"], errors="coerce")
            .dropna()
            .astype(int)
        )
        if observed_replicates != set(range(1, expected_null_replicates + 1)):
            return False
        if len(run_audit) != int(expected_null_replicates):
            return False
        return True
    except Exception:
        return False


def run_null_group_task(task: Mapping[str, object]) -> Dict[str, object]:
    payload_path = Path(str(task["payload_path"]))
    marker_path = Path(str(task["marker_path"]))
    fingerprint = str(task["fingerprint"])
    n_null = int(task["n_null_replicates"])
    feature_columns = [str(value) for value in task["feature_columns"]]
    if checkpoint_is_valid(
        payload_path,
        marker_path,
        fingerprint,
        n_null,
        len(feature_columns),
    ):
        return {
            "status": "checkpoint",
            "task_key": str(task["task_key"]),
            "payload_path": str(payload_path),
        }

    encoded = np.asarray(task["encoded_data"], dtype=np.int64)
    observed_y = np.asarray(task["observed_outcome"], dtype=np.int64)
    node_names = [str(value) for value in task["node_names"]]
    base_metadata = dict(task["metadata"])
    alpha = float(task["alpha"])
    depth = int(task["depth"])
    partition_id = int(base_metadata["partition_id"])

    edge_parts: List[pd.DataFrame] = []
    metric_parts: List[pd.DataFrame] = []
    bridge_parts: List[pd.DataFrame] = []
    run_rows: List[Dict[str, object]] = []

    for null_replicate in range(1, n_null + 1):
        started = time.time()
        null_seed = stable_seed_from_key(
            "structural_null",
            int(task["structural_random_seed"]),
            partition_id,
            int(null_replicate),
        )
        permuted_y = np.random.RandomState(int(null_seed)).permutation(
            observed_y
        )
        null_encoded = encoded.copy()
        null_encoded[:, -1] = np.asarray(permuted_y, dtype=np.int64)
        metadata = {
            **base_metadata,
            "null_replicate": int(null_replicate),
            "outcome_status": "permuted_negative_control",
        }

        graph, edges, captured_stdout = run_fci(
            null_encoded,
            node_names=node_names,
            alpha=alpha,
            depth=depth,
            ci_test=str(task["ci_test"]),
            max_path_length=int(task["max_path_length"]),
        )
        edge_frame = extract_pag_edges(graph, node_names, metadata)
        metrics, bridges, target_degree, fraction_within_two = (
            graph_multiscale_rows(graph, feature_columns, metadata)
        )
        if not edge_frame.empty:
            edge_parts.append(edge_frame)
        metric_parts.append(metrics)
        if not bridges.empty:
            bridge_parts.append(bridges)
        run_rows.append(
            {
                **metadata,
                "n_pag_edges": int(len(edges)),
                "target_degree": int(target_degree),
                "fraction_features_within_two_steps": float(
                    fraction_within_two
                ),
                "captured_stdout_characters": int(len(captured_stdout)),
                "elapsed_seconds": float(time.time() - started),
                "run_status": "complete",
            }
        )

    payload = {
        "contract_version": CONTRACT_VERSION,
        "fingerprint": fingerprint,
        "task_key": str(task["task_key"]),
        "edges": (
            pd.concat(edge_parts, ignore_index=True)
            if edge_parts
            else pd.DataFrame(columns=EDGE_COLUMNS)
        ),
        "metrics": pd.concat(metric_parts, ignore_index=True),
        "bridges": (
            pd.concat(bridge_parts, ignore_index=True)
            if bridge_parts
            else pd.DataFrame(columns=BRIDGE_COLUMNS)
        ),
        "run_audit": pd.DataFrame(run_rows, columns=RUN_AUDIT_COLUMNS),
    }
    atomic_pickle_gzip_dump(payload, payload_path)
    atomic_json_dump(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "contract_version": CONTRACT_VERSION,
            "fingerprint": fingerprint,
            "payload_sha256": file_sha256(payload_path),
            "task_key": str(task["task_key"]),
            "n_null_replicates": n_null,
            "n_metric_rows": int(len(payload["metrics"])),
            "n_edge_rows": int(len(payload["edges"])),
            "n_bridge_rows": int(len(payload["bridges"])),
        },
        marker_path,
    )
    return {
        "status": "computed",
        "task_key": str(task["task_key"]),
        "payload_path": str(payload_path),
    }


def execute_tasks(
    tasks: Sequence[Mapping[str, object]],
    workers: int,
    max_pending: int,
) -> List[Path]:
    if not tasks:
        return []
    workers = max(1, int(workers))
    max_pending = max(workers, int(max_pending))
    completed_paths: List[Path] = []

    if workers == 1:
        for index, task in enumerate(tasks, start=1):
            result = run_null_group_task(task)
            completed_paths.append(Path(str(result["payload_path"])))
            cprint(
                f"[FCI NULL GROUP {index}/{len(tasks)}] "
                f"{result['task_key']} | {result['status']}"
            )
        return completed_paths

    context = mp.get_context("spawn")
    task_iterator = iter(tasks)
    pending: Dict[Any, Mapping[str, object]] = {}
    finished = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
    ) as executor:
        while len(pending) < min(max_pending, len(tasks)):
            try:
                task = next(task_iterator)
            except StopIteration:
                break
            pending[executor.submit(run_null_group_task, task)] = task

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"FCI null task failed: {task['task_key']}"
                    ) from exc
                finished += 1
                completed_paths.append(Path(str(result["payload_path"])))
                cprint(
                    f"[FCI NULL GROUP {finished}/{len(tasks)}] "
                    f"{result['task_key']} | {result['status']}"
                )
                try:
                    next_task = next(task_iterator)
                except StopIteration:
                    continue
                pending[executor.submit(run_null_group_task, next_task)] = next_task
    return completed_paths


# =============================================================================
# OBSERVED PAG RECONSTRUCTION AND SUMMARIES
# =============================================================================


def observed_multiscale_tables(
    structural_dir: Path,
    feature_columns: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    edge_path = structural_dir / "pag_edges_outer_training_raw.csv"
    run_path = structural_dir / "structural_run_audit.csv"
    if not edge_path.exists() or not run_path.exists():
        raise FileNotFoundError(
            "The completed structural audit is missing observed PAG edges or "
            "the structural run audit."
        )
    edges = pd.read_csv(edge_path)
    runs = pd.read_csv(run_path)
    runs = runs.loc[
        (runs["scope"].astype(str) == "outer_training")
        & (runs["outcome_status"].astype(str) == "observed")
    ].copy()
    runs = runs.sort_values(["partition_id", "alpha", "depth"]).reset_index(
        drop=True
    )
    if len(runs) != 45:
        raise RuntimeError(f"Expected 45 observed FCI specifications; found {len(runs)}.")

    node_names = list(feature_columns) + [TARGET_NODE]
    metric_parts: List[pd.DataFrame] = []
    bridge_parts: List[pd.DataFrame] = []
    reconstructed_runs: List[Dict[str, object]] = []
    for run in runs.itertuples(index=False):
        metadata = {
            column: getattr(run, column)
            for column in RUN_KEY_COLUMNS
        }
        metadata.update(
            {
                "null_replicate": 0,
                "outcome_status": "observed",
                "n_records": int(run.n_records),
            }
        )
        selected_edges = edges.loc[
            (pd.to_numeric(edges["partition_id"]) == int(run.partition_id))
            & (edges["setting"].astype(str) == str(run.setting))
        ].copy()
        metrics, bridges, target_degree, fraction_within_two = (
            graph_rows_from_edge_table(
                selected_edges,
                node_names=node_names,
                feature_columns=feature_columns,
                metadata=metadata,
            )
        )
        if int(target_degree) != int(run.target_degree):
            raise RuntimeError(
                "Observed PAG reconstruction disagrees with structural_run_audit "
                f"for partition={run.partition_id}, setting={run.setting}."
            )
        metric_parts.append(metrics)
        if not bridges.empty:
            bridge_parts.append(bridges)
        reconstructed_runs.append(
            {
                **metadata,
                "n_pag_edges": int(len(selected_edges)),
                "target_degree": int(target_degree),
                "fraction_features_within_two_steps": float(
                    fraction_within_two
                ),
                "captured_stdout_characters": np.nan,
                "elapsed_seconds": np.nan,
                "run_status": "reconstructed_from_completed_observed_pag",
            }
        )
    return (
        pd.concat(metric_parts, ignore_index=True),
        (
            pd.concat(bridge_parts, ignore_index=True)
            if bridge_parts
            else pd.DataFrame(columns=BRIDGE_COLUMNS)
        ),
        pd.DataFrame(reconstructed_runs, columns=RUN_AUDIT_COLUMNS),
    )


def quantile_95(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.quantile(array, 0.95, method="linear"))


def summarize_feature_metrics(
    observed_metrics: pd.DataFrame,
    null_metrics: pd.DataFrame,
    n_null_replicates: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_names = ["direct_adjacency", "exact_two_step", "within_two_steps"]
    observed_overall = (
        observed_metrics.groupby("feature", as_index=False)[metric_names]
        .mean()
        .rename(
            columns={
                "direct_adjacency": "observed_direct_frequency",
                "exact_two_step": "observed_exact_two_step_frequency",
                "within_two_steps": "observed_within_two_steps_frequency",
            }
        )
    )
    observed_by_partition = (
        observed_metrics.groupby(
            ["partition_id", "outer_repeat", "outer_fold", "feature"],
            as_index=False,
        )[metric_names]
        .mean()
        .rename(
            columns={
                "direct_adjacency": "within_partition_direct_setting_frequency",
                "exact_two_step": "within_partition_exact_two_step_setting_frequency",
                "within_two_steps": "within_partition_two_step_setting_frequency",
            }
        )
    )
    observed_by_setting = (
        observed_metrics.groupby(
            ["alpha", "depth", "setting", "feature"], as_index=False
        )[metric_names]
        .mean()
        .rename(
            columns={
                "direct_adjacency": "across_partition_direct_frequency",
                "exact_two_step": "across_partition_exact_two_step_frequency",
                "within_two_steps": "across_partition_two_step_frequency",
            }
        )
    )

    partition_summary_rows: List[Dict[str, object]] = []
    for feature, group in observed_by_partition.groupby("feature", sort=False):
        row: Dict[str, object] = {"feature": str(feature)}
        for short_name, column in [
            ("direct", "within_partition_direct_setting_frequency"),
            ("exact_two_step", "within_partition_exact_two_step_setting_frequency"),
            ("within_two_steps", "within_partition_two_step_setting_frequency"),
        ]:
            values = group[column].to_numpy(dtype=float)
            row[f"partition_{short_name}_mean"] = float(np.mean(values))
            row[f"partition_{short_name}_std"] = float(
                np.std(values, ddof=1) if len(values) > 1 else 0.0
            )
            row[f"partition_{short_name}_q25"] = float(np.quantile(values, 0.25))
            row[f"partition_{short_name}_q75"] = float(np.quantile(values, 0.75))
            row[f"partition_{short_name}_presence_fraction"] = float(
                np.mean(values > 0.0)
            )
        partition_summary_rows.append(row)
    partition_summary = pd.DataFrame(partition_summary_rows)

    null_by_replicate = (
        null_metrics.groupby(["null_replicate", "feature"], as_index=False)[
            metric_names
        ]
        .mean()
        .rename(
            columns={
                "direct_adjacency": "null_direct_frequency",
                "exact_two_step": "null_exact_two_step_frequency",
                "within_two_steps": "null_within_two_steps_frequency",
            }
        )
    )
    replicate_counts = null_by_replicate.groupby("feature")[
        "null_replicate"
    ].nunique()
    if set(replicate_counts.astype(int)) != {int(n_null_replicates)}:
        raise RuntimeError("Null feature summaries do not contain every replicate.")

    summary = observed_overall.merge(partition_summary, on="feature", how="left")
    for observed_column, null_column, prefix in [
        ("observed_direct_frequency", "null_direct_frequency", "direct"),
        (
            "observed_exact_two_step_frequency",
            "null_exact_two_step_frequency",
            "exact_two_step",
        ),
        (
            "observed_within_two_steps_frequency",
            "null_within_two_steps_frequency",
            "within_two_steps",
        ),
    ]:
        rows: List[Dict[str, object]] = []
        for feature, group in null_by_replicate.groupby("feature", sort=False):
            observed_value = float(
                summary.loc[summary["feature"] == feature, observed_column].iloc[0]
            )
            values = group[null_column].to_numpy(dtype=float)
            upper_tail = float(
                (1 + np.sum(values >= observed_value)) / (len(values) + 1)
            )
            rows.append(
                {
                    "feature": str(feature),
                    f"null_{prefix}_mean": float(np.mean(values)),
                    f"null_{prefix}_median": float(np.median(values)),
                    f"null_{prefix}_q95": quantile_95(values),
                    f"null_{prefix}_max": float(np.max(values)),
                    f"excess_{prefix}_over_null_median": float(
                        observed_value - np.median(values)
                    ),
                    f"observed_{prefix}_above_null_q95_descriptive": int(
                        observed_value > quantile_95(values)
                    ),
                    f"null_{prefix}_upper_tail_fraction_descriptive": upper_tail,
                    f"null_{prefix}_percentile_descriptive": float(1.0 - upper_tail),
                }
            )
        summary = summary.merge(pd.DataFrame(rows), on="feature", how="left")
    summary["n_observed_designs"] = int(len(observed_metrics) / summary.shape[0])
    summary["n_outer_partitions"] = int(
        observed_metrics["partition_id"].nunique()
    )
    summary["n_settings_per_partition"] = int(
        observed_metrics["setting"].nunique()
    )
    summary["n_null_replicates"] = int(n_null_replicates)
    summary["formal_significance_claim_allowed"] = False
    return summary, observed_by_partition, observed_by_setting, null_by_replicate


def summarize_target_connectivity(
    observed_run_audit: pd.DataFrame,
    null_run_audit: pd.DataFrame,
    n_null_replicates: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    observed_degree = observed_run_audit["target_degree"].to_numpy(dtype=float)
    null_replicates = (
        null_run_audit.groupby("null_replicate", as_index=False)
        .agg(
            mean_target_degree=("target_degree", "mean"),
            fraction_designs_with_any_neighbor=(
                "target_degree",
                lambda values: float(np.mean(np.asarray(values, dtype=float) > 0.0)),
            ),
            mean_fraction_features_within_two_steps=(
                "fraction_features_within_two_steps",
                "mean",
            ),
        )
        .sort_values("null_replicate")
        .reset_index(drop=True)
    )
    if len(null_replicates) != int(n_null_replicates):
        raise RuntimeError("Global target null summary is missing replicates.")

    observed_values = {
        "mean_target_degree": float(np.mean(observed_degree)),
        "fraction_designs_with_any_neighbor": float(
            np.mean(observed_degree > 0.0)
        ),
        "mean_fraction_features_within_two_steps": float(
            observed_run_audit["fraction_features_within_two_steps"].mean()
        ),
    }
    rows: List[Dict[str, object]] = []
    for metric, observed_value in observed_values.items():
        values = null_replicates[metric].to_numpy(dtype=float)
        upper_tail = float(
            (1 + np.sum(values >= observed_value)) / (len(values) + 1)
        )
        rows.append(
            {
                "metric": metric,
                "observed_value": observed_value,
                "null_mean": float(np.mean(values)),
                "null_median": float(np.median(values)),
                "null_q95": quantile_95(values),
                "null_max": float(np.max(values)),
                "observed_minus_null_median": float(
                    observed_value - np.median(values)
                ),
                "observed_above_null_q95_descriptive": int(
                    observed_value > quantile_95(values)
                ),
                "null_upper_tail_fraction_descriptive": upper_tail,
                "null_percentile_descriptive": float(1.0 - upper_tail),
                "n_observed_designs": int(len(observed_run_audit)),
                "n_null_replicates": int(n_null_replicates),
                "formal_significance_claim_allowed": False,
            }
        )
    return pd.DataFrame(rows), null_replicates


def summarize_bridges(
    observed_bridges: pd.DataFrame,
    null_bridges: pd.DataFrame,
    observed_run_count: int,
    n_null_replicates: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if observed_bridges.empty:
        return pd.DataFrame(), pd.DataFrame()
    pair_columns = ["feature", "bridge_node"]
    observed = (
        observed_bridges.drop_duplicates(
            ["partition_id", "setting"] + pair_columns
        )
        .groupby(pair_columns, as_index=False)
        .size()
        .rename(columns={"size": "observed_design_count"})
    )
    observed["observed_bridge_frequency"] = (
        observed["observed_design_count"] / float(observed_run_count)
    )
    fold_presence = (
        observed_bridges.drop_duplicates(
            ["partition_id"] + pair_columns
        )
        .groupby(pair_columns, as_index=False)["partition_id"]
        .nunique()
        .rename(columns={"partition_id": "n_outer_partitions_present"})
    )
    observed = observed.merge(fold_presence, on=pair_columns, how="left")
    n_partitions = int(observed_bridges["partition_id"].nunique())
    observed["outer_partition_presence_fraction"] = (
        observed["n_outer_partitions_present"] / float(n_partitions)
    )
    observed["n_observed_designs"] = int(observed_run_count)

    null_unique = null_bridges.drop_duplicates(
        ["null_replicate", "partition_id", "setting"] + pair_columns
    )
    null_counts = (
        null_unique.groupby(["null_replicate"] + pair_columns, as_index=False)
        .size()
        .rename(columns={"size": "null_design_count"})
    )
    grid = pd.MultiIndex.from_product(
        [
            range(1, int(n_null_replicates) + 1),
            list(
                observed[pair_columns]
                .itertuples(index=False, name=None)
            ),
        ],
        names=["null_replicate", "pair"],
    ).to_frame(index=False)
    grid[["feature", "bridge_node"]] = pd.DataFrame(
        grid.pop("pair").tolist(), index=grid.index
    )
    null_complete = grid.merge(
        null_counts,
        on=["null_replicate"] + pair_columns,
        how="left",
    )
    null_complete["null_design_count"] = (
        null_complete["null_design_count"].fillna(0).astype(int)
    )
    null_complete["null_bridge_frequency"] = (
        null_complete["null_design_count"] / float(observed_run_count)
    )

    summary_rows: List[Dict[str, object]] = []
    for pair in observed[pair_columns].itertuples(index=False, name=None):
        feature, bridge = pair
        observed_value = float(
            observed.loc[
                (observed["feature"] == feature)
                & (observed["bridge_node"] == bridge),
                "observed_bridge_frequency",
            ].iloc[0]
        )
        values = null_complete.loc[
            (null_complete["feature"] == feature)
            & (null_complete["bridge_node"] == bridge),
            "null_bridge_frequency",
        ].to_numpy(dtype=float)
        upper_tail = float(
            (1 + np.sum(values >= observed_value)) / (len(values) + 1)
        )
        summary_rows.append(
            {
                "feature": str(feature),
                "bridge_node": str(bridge),
                "null_bridge_mean": float(np.mean(values)),
                "null_bridge_median": float(np.median(values)),
                "null_bridge_q95": quantile_95(values),
                "null_bridge_max": float(np.max(values)),
                "excess_bridge_over_null_median": float(
                    observed_value - np.median(values)
                ),
                "observed_bridge_above_null_q95_descriptive": int(
                    observed_value > quantile_95(values)
                ),
                "null_bridge_upper_tail_fraction_descriptive": upper_tail,
                "null_bridge_percentile_descriptive": float(1.0 - upper_tail),
            }
        )
    observed = observed.merge(
        pd.DataFrame(summary_rows), on=pair_columns, how="left"
    )
    observed["formal_mediation_claim_allowed"] = False
    observed["formal_causal_path_claim_allowed"] = False
    observed = observed.sort_values(
        ["observed_bridge_frequency", "feature", "bridge_node"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return observed, null_complete


def build_cross_view_matrix(
    source_run_dir: Path,
    structural_summary: pd.DataFrame,
) -> pd.DataFrame:
    path = source_run_dir / "tables" / "credal_feature_intervals_and_partial_order.csv"
    if not path.exists():
        raise FileNotFoundError(f"Credal feature table not found: {path}")
    credal = pd.read_csv(path)
    output = credal.merge(structural_summary, on="feature", how="left")
    output["class_name"] = output["class_id"].map(TARGET_NAMES)

    def describe(row: pd.Series) -> str:
        maximal = int(row.get("in_maximal_set", 0)) == 1
        direct = int(
            row.get("observed_direct_above_null_q95_descriptive", 0)
        ) == 1
        within_two = int(
            row.get("observed_within_two_steps_above_null_q95_descriptive", 0)
        ) == 1
        if maximal and direct:
            return "predictive_structural_direct_correspondence"
        if maximal and within_two:
            return "predictive_prominence_with_structural_proximity"
        if maximal:
            return "predictive_channel_prominence_only_in_current_design"
        if direct or within_two:
            return "structural_proximity_without_credal_maximality"
        return "neither_channel_prominent_in_current_design"

    output["descriptive_cross_view_pattern"] = output.apply(describe, axis=1)
    output["cross_view_pattern_is_ranking_rule"] = False
    output["cross_view_pattern_is_validation_claim"] = False
    output["structural_result_is_class_specific"] = False
    output["channels_are_statistically_independent"] = False
    return output


# =============================================================================
# FIGURES
# =============================================================================


def configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_target_connectivity_figure(
    summary: pd.DataFrame,
    null_replicates: pd.DataFrame,
    figure_dir: Path,
) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    row = summary.loc[summary["metric"] == "mean_target_degree"].iloc[0]
    observed = float(row["observed_value"])
    n_designs = int(row["n_observed_designs"])
    null_values = null_replicates["mean_target_degree"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    bins = np.linspace(0.0, max(observed, float(np.max(null_values))) + 0.03, 14)
    ax.hist(
        null_values,
        bins=bins,
        color="#B7BEC8",
        edgecolor="white",
        linewidth=0.8,
        label="Permuted-outcome reference",
    )
    ax.axvline(
        observed,
        color="#B54745",
        linewidth=2.0,
        label=f"Observed mean degree = {observed:.3f}",
    )
    ax.set_xlabel(
        f"Mean injury-node degree across {n_designs} design specifications"
    )
    ax.set_ylabel("Permutation replicates")
    ax.legend(frameon=False)
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "global_outcome_node_null_comparison.png", dpi=400)
    plt.close(fig)


def save_multiscale_feature_figure(
    feature_summary: pd.DataFrame,
    figure_dir: Path,
) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    order = feature_summary.sort_values(
        ["observed_within_two_steps_frequency", "observed_direct_frequency"],
        ascending=[True, True],
    )["feature"].tolist()
    frame = feature_summary.set_index("feature").loc[order].reset_index()
    n_designs = int(frame["n_observed_designs"].iloc[0])
    y = np.arange(len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.4), sharey=True)
    specifications = [
        (
            axes[0],
            "observed_direct_frequency",
            "null_direct_median",
            "null_direct_q95",
            "Direct outcome adjacency",
            "#21867A",
        ),
        (
            axes[1],
            "observed_within_two_steps_frequency",
            "null_within_two_steps_median",
            "null_within_two_steps_q95",
            "Within-two-step skeleton proximity",
            "#21867A",
        ),
    ]
    for ax, observed_col, median_col, q95_col, title, color in specifications:
        ax.hlines(
            y,
            frame[median_col],
            frame[q95_col],
            color="#A7ADB5",
            linewidth=2.0,
            label="Null median to 95th percentile",
        )
        ax.scatter(
            frame[median_col],
            y,
            marker="x",
            color="#666666",
            s=28,
            label="Null median",
            zorder=3,
        )
        ax.scatter(
            frame[observed_col],
            y,
            facecolor="white",
            edgecolor=color,
            linewidth=1.8,
            s=48,
            label="Observed",
            zorder=4,
        )
        ax.set_title(title, loc="left")
        ax.set_xlabel(f"Frequency across {n_designs} design specifications")
        ax.grid(axis="x", color="#E3E5E8", linewidth=0.8)
        ax.set_xlim(left=-0.01)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(frame["feature"])
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "multiscale_pag_feature_proximity.png", dpi=400)
    plt.close(fig)


def save_bridge_figure(bridge_summary: pd.DataFrame, figure_dir: Path) -> None:
    if bridge_summary.empty:
        return
    configure_matplotlib()
    import matplotlib.pyplot as plt

    frame = bridge_summary.head(12).copy()
    n_designs = int(frame["n_observed_designs"].iloc[0])
    frame["label"] = frame["feature"] + "  |  " + frame["bridge_node"]
    frame = frame.sort_values("observed_bridge_frequency", ascending=True)
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.hlines(
        y,
        frame["null_bridge_median"],
        frame["null_bridge_q95"],
        color="#A7ADB5",
        linewidth=2.0,
    )
    ax.scatter(
        frame["observed_bridge_frequency"],
        y,
        facecolor="white",
        edgecolor="#8E4A7A",
        linewidth=1.8,
        s=50,
        zorder=3,
        label="Observed",
    )
    ax.scatter(
        frame["null_bridge_q95"],
        y,
        marker="|",
        color="#666666",
        s=80,
        label="Null 95th percentile",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(frame["label"])
    ax.set_xlabel(f"Frequency across {n_designs} design specifications")
    ax.set_title("Recurrent exact-two-step PAG skeleton bridges", loc="left")
    ax.grid(axis="x", color="#E3E5E8", linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "recurrent_two_step_skeleton_bridges.png", dpi=400)
    plt.close(fig)


def save_cross_view_figure(
    cross_view: pd.DataFrame,
    feature_summary: pd.DataFrame,
    figure_dir: Path,
) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    feature_order = feature_summary.sort_values(
        "observed_within_two_steps_frequency", ascending=False
    )["feature"].tolist()
    credal = (
        cross_view.pivot_table(
            index="feature",
            columns="class_name",
            values="lower_expected_percentile",
            aggfunc="first",
        )
        .reindex(feature_order)
        .reindex(columns=["No injury", "Minor injury", "Moderate-plus injury"])
    )
    structural = (
        feature_summary.set_index("feature")
        .reindex(feature_order)[
            ["observed_direct_frequency", "observed_within_two_steps_frequency"]
        ]
        .rename(
            columns={
                "observed_direct_frequency": "Direct",
                "observed_within_two_steps_frequency": "Within 2 steps",
            }
        )
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.8, 5.8),
        gridspec_kw={"width_ratios": [3.0, 2.0]},
        sharey=True,
    )
    image1 = axes[0].imshow(
        credal.to_numpy(dtype=float),
        aspect="auto",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
    )
    image2 = axes[1].imshow(
        structural.to_numpy(dtype=float),
        aspect="auto",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=max(0.3, float(np.nanmax(structural.to_numpy(dtype=float)))),
    )
    axes[0].set_title("Credal lower predictive relevance bound", loc="left")
    axes[1].set_title("PAG structural frequency", loc="left")
    for ax, columns in [(axes[0], credal.columns), (axes[1], structural.columns)]:
        ax.set_xticks(np.arange(len(columns)))
        ax.set_xticklabels(columns, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(feature_order)))
        ax.set_yticklabels(feature_order)
        ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(feature_order), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)
    colorbar1 = fig.colorbar(image1, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar1.set_label("Lower bound")
    colorbar2 = fig.colorbar(image2, ax=axes[1], fraction=0.069, pad=0.04)
    colorbar2.set_label("Frequency")
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "predictive_structural_cross_view.png", dpi=400)
    plt.close(fig)


# =============================================================================
# SELF-CHECKS AND MAIN WORKFLOW
# =============================================================================


def run_deterministic_self_checks() -> None:
    # A--B--Y must be a two-step relation with B as the only bridge.
    feature_columns = ["A", "B", "C"]
    matrix = np.zeros((4, 4), dtype=int)
    matrix[0, 1] = matrix[1, 0] = 1
    matrix[1, 3] = matrix[3, 1] = 1
    matrix[2, 3] = matrix[3, 2] = 1

    class Graph:
        pass

    graph = Graph()
    graph.graph = matrix
    metadata = {
        "scope": "self_check",
        "partition_id": 1,
        "outer_repeat": 1,
        "outer_fold": 1,
        "split_seed": 1,
        "alpha": 0.05,
        "depth": 1,
        "setting": "self_check",
        "null_replicate": 0,
        "outcome_status": "observed",
        "n_records": 10,
    }
    metrics, bridges, degree, _ = graph_multiscale_rows(
        graph, feature_columns, metadata
    )
    row_a = metrics.loc[metrics["feature"] == "A"].iloc[0]
    row_b = metrics.loc[metrics["feature"] == "B"].iloc[0]
    row_c = metrics.loc[metrics["feature"] == "C"].iloc[0]
    assert int(row_a["exact_two_step"]) == 1
    assert int(row_a["direct_adjacency"]) == 0
    assert int(row_b["direct_adjacency"]) == 1
    assert int(row_c["direct_adjacency"]) == 1
    assert int(degree) == 2
    assert len(bridges) == 1
    assert bridges.iloc[0]["feature"] == "A"
    assert bridges.iloc[0]["bridge_node"] == "B"

    seed_a = stable_seed_from_key("structural_null", 20260723, 1, 1)
    seed_b = stable_seed_from_key("structural_null", 20260723, 1, 1)
    assert seed_a == seed_b
    cprint("[SELF-CHECK] graph distance, bridge, and seed checks passed.")


def load_and_verify_analysis(
    excel_path: Path,
    source_run_dir: Path,
) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    List[str],
    List[OuterPartition],
    Dict[str, object],
    Dict[str, object],
]:
    run_meta_path = source_run_dir / "run_meta.json"
    identity_path = source_run_dir / "tables" / "dataset_identity.json"
    if not run_meta_path.exists() or not identity_path.exists():
        raise FileNotFoundError(
            "SOURCE_RUN_DIR must contain run_meta.json and tables/dataset_identity.json."
        )
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    structural_meta = dict(run_meta.get("structural_compatibility_audit", {}))
    if structural_meta.get("status") != "complete":
        raise RuntimeError("The source structural audit is not marked complete.")

    observed_excel_hash = file_sha256(excel_path)
    expected_excel_hash = str(identity["excel_sha256"])
    if observed_excel_hash != expected_excel_hash:
        raise RuntimeError(
            "Workbook SHA-256 differs from the completed run.\n"
            f"Expected: {expected_excel_hash}\nObserved: {observed_excel_hash}"
        )

    data = load_ads_table(excel_path)
    if len(data) != int(identity["n_records"]):
        raise RuntimeError(
            f"Expected {identity['n_records']} cleaned records; observed {len(data)}."
        )
    if len(data) != int(EXPECTED_N_SAMPLES):
        raise RuntimeError(
            f"This extension is frozen for {EXPECTED_N_SAMPLES} records."
        )
    feature_columns = [str(value) for value in run_meta["data"]["feature_columns"]]
    ordered_hash = dataframe_digest(
        data[[STABLE_RECORD_ID_COL] + feature_columns + [TARGET_COL]]
    )
    expected_row_hash_path = (
        source_run_dir / "tables" / "stable_record_id_to_source_row_hash.csv"
    )
    if not expected_row_hash_path.exists():
        raise FileNotFoundError(
            "Stable-record identity table not found: "
            f"{expected_row_hash_path}"
        )
    observed_row_hashes = stable_record_hash_table(data, feature_columns)
    expected_row_hashes = pd.read_csv(expected_row_hash_path)
    expected_row_hashes = expected_row_hashes[
        [STABLE_RECORD_ID_COL, "source_row_sha256"]
    ].copy()
    expected_row_hashes[STABLE_RECORD_ID_COL] = expected_row_hashes[
        STABLE_RECORD_ID_COL
    ].astype(np.int64)
    expected_row_hashes["source_row_sha256"] = expected_row_hashes[
        "source_row_sha256"
    ].astype(str)
    if not observed_row_hashes.equals(expected_row_hashes):
        raise RuntimeError(
            "The cleaned rows or their order do not match the completed run."
        )
    if ordered_hash != str(identity["ordered_dataset_sha256"]):
        cprint(
            "[VERIFY] The legacy whole-table digest differs because the active "
            "Pandas version represents string dtypes differently. Exact workbook "
            "SHA-256 and all 1,469 canonical row hashes match the source run."
        )
    if stable_json_digest(feature_columns) != str(identity["feature_order_sha256"]):
        raise RuntimeError("Feature order differs from the completed run.")

    y = data[TARGET_COL].astype(int).to_numpy()
    partitioning = dict(run_meta["partitioning"])
    partitions = make_outer_partitions(
        y,
        n_splits=int(partitioning["outer_folds_per_repeat"]),
        n_repeats=int(partitioning["outer_repeats"]),
        random_seed=int(partitioning["outer_random_seed"]),
    )
    manifest = partition_manifest(partitions, y)
    verify_partition_manifest(
        manifest,
        source_run_dir / "tables" / "outer_partition_manifest.csv",
    )

    observed_causal_version = package_version("causal-learn")
    source_causal_version = str(
        run_meta["execution"]["environment"]["package_versions"]["causal-learn"]
    )
    if observed_causal_version != source_causal_version:
        raise RuntimeError(
            "causal-learn version differs from the completed run. "
            f"Expected {source_causal_version}; observed {observed_causal_version}."
        )
    if observed_causal_version != EXPECTED_CAUSAL_LEARN_VERSION:
        raise RuntimeError(
            "The frozen extension expects causal-learn "
            f"{EXPECTED_CAUSAL_LEARN_VERSION}; observed {observed_causal_version}."
        )
    return data, y, feature_columns, partitions, run_meta, structural_meta


def create_tasks(
    data: pd.DataFrame,
    y: np.ndarray,
    feature_columns: Sequence[str],
    partitions: Sequence[OuterPartition],
    structural_meta: Mapping[str, object],
    checkpoint_root: Path,
    n_null_replicates: int,
    smoke: bool,
) -> List[Dict[str, object]]:
    alpha_grid = [float(value) for value in structural_meta["alpha_grid"]]
    depth_grid = [int(value) for value in structural_meta["depth_grid"]]
    if smoke:
        alpha_grid = [0.05]
        depth_grid = [1]
        partitions = list(partitions)[:1]
    node_names = list(feature_columns) + [TARGET_NODE]
    X = data[list(feature_columns) + [STABLE_RECORD_ID_COL]].copy()
    tasks: List[Dict[str, object]] = []
    causal_version = package_version("causal-learn")
    for partition in partitions:
        train_idx = np.asarray(partition.train_idx, dtype=int)
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = np.asarray(y)[train_idx].astype(int)
        encoded_train = encode_structural_data(
            X_train, y_train, feature_columns
        )
        for alpha in alpha_grid:
            for depth in depth_grid:
                setting = structural_setting_label(alpha, depth)
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
                    f"outer_p{partition.partition_id:03d}_"
                    f"a{alpha:.4f}_d{depth}"
                )
                payload_path, marker_path = checkpoint_paths(
                    checkpoint_root, task_key
                )
                fingerprint = stable_json_digest(
                    {
                        "contract": CONTRACT_VERSION,
                        "task_key": task_key,
                        "encoded_data": array_digest(encoded_train),
                        "observed_outcome": array_digest(y_train),
                        "node_names": node_names,
                        "metadata": metadata,
                        "null_replicates": int(n_null_replicates),
                        "structural_random_seed": STRUCTURAL_RANDOM_SEED,
                        "null_seed_contract": (
                            "partition_replicate_label_permutation_v1"
                        ),
                        "ci_test": str(structural_meta["conditional_independence_test"]),
                        "alpha": alpha,
                        "depth": depth,
                        "max_path_length": int(structural_meta["max_path_length"]),
                        "causal_learn_version": causal_version,
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
                        "feature_columns": list(feature_columns),
                        "node_names": node_names,
                        "metadata": metadata,
                        "alpha": float(alpha),
                        "depth": int(depth),
                        "n_null_replicates": int(n_null_replicates),
                        "structural_random_seed": int(STRUCTURAL_RANDOM_SEED),
                        "ci_test": STRUCTURAL_CI_TEST,
                        "max_path_length": int(STRUCTURAL_MAX_PATH_LENGTH),
                    }
                )
    return tasks


def combine_payloads(
    payload_paths: Sequence[Path],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    edge_parts: List[pd.DataFrame] = []
    metric_parts: List[pd.DataFrame] = []
    bridge_parts: List[pd.DataFrame] = []
    run_parts: List[pd.DataFrame] = []
    for path in payload_paths:
        payload = load_pickle_gzip(path)
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"Invalid structural checkpoint payload: {path}")
        edges = payload["edges"]
        metrics = payload["metrics"]
        bridges = payload["bridges"]
        runs = payload["run_audit"]
        if not edges.empty:
            edge_parts.append(edges)
        metric_parts.append(metrics)
        if not bridges.empty:
            bridge_parts.append(bridges)
        run_parts.append(runs)
    return (
        pd.concat(edge_parts, ignore_index=True)
        if edge_parts
        else pd.DataFrame(columns=EDGE_COLUMNS),
        pd.concat(metric_parts, ignore_index=True),
        pd.concat(bridge_parts, ignore_index=True)
        if bridge_parts
        else pd.DataFrame(columns=BRIDGE_COLUMNS),
        pd.concat(run_parts, ignore_index=True),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun only matched FCI label-null graphs and produce multiscale "
            "PAG structural interpretation outputs."
        )
    )
    parser.add_argument("--excel", default=EXCEL_PATH)
    parser.add_argument("--source-run", default=SOURCE_RUN_DIR)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--workers", type=int, default=PARALLEL_WORKERS)
    parser.add_argument("--max-pending", type=int, default=MAX_PENDING_TASKS)
    parser.add_argument(
        "--null-replicates", type=int, default=NULL_REPLICATES
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one partition, one setting, and two null replicates.",
    )
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Do not fit FCI; aggregate already completed checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    started = time.time()
    args = parse_args()
    run_deterministic_self_checks()

    excel_path = Path(args.excel).resolve()
    source_run_dir = Path(args.source_run).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else source_run_dir
        / "structural_compatibility_audit"
        / (OUTPUT_SUBDIR + ("_smoke" if args.smoke else ""))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output_dir / "checkpoints" / "null_fci_groups"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"

    n_null_replicates = 2 if args.smoke else int(args.null_replicates)
    if n_null_replicates < 1:
        raise ValueError("At least one null replicate is required.")

    cprint("\n" + "#" * 100)
    cprint("FCI/PAG MULTISCALE STRUCTURAL INTERPRETATION EXTENSION")
    cprint("#" * 100)
    cprint(f"EXCEL_PATH = {excel_path}")
    cprint(f"SOURCE_RUN_DIR = {source_run_dir}")
    cprint(f"OUTPUT_DIR = {output_dir}")
    cprint(f"NULL_REPLICATES = {n_null_replicates}")
    cprint(f"PARALLEL_WORKERS = {int(args.workers)}")
    cprint(f"SMOKE = {bool(args.smoke)}")

    data, y, feature_columns, partitions, run_meta, structural_meta = (
        load_and_verify_analysis(excel_path, source_run_dir)
    )
    expected_settings = int(structural_meta["settings_per_partition"])
    if not args.smoke:
        if int(structural_meta["null_replicates"]) != n_null_replicates:
            raise RuntimeError(
                "The requested null-replicate count differs from the completed "
                "structural audit."
            )
        if int(structural_meta["outer_partitions"]) != len(partitions):
            raise RuntimeError("Outer-partition count differs from the source run.")
        if expected_settings != 9:
            raise RuntimeError("This frozen extension expects nine FCI settings.")

    tasks = create_tasks(
        data=data,
        y=y,
        feature_columns=feature_columns,
        partitions=partitions,
        structural_meta=structural_meta,
        checkpoint_root=checkpoint_root,
        n_null_replicates=n_null_replicates,
        smoke=bool(args.smoke),
    )
    cprint(f"FCI task groups = {len(tasks)}")
    cprint(
        "Expected FCI fits = "
        f"{len(tasks) * n_null_replicates} "
        "(prediction and explainer models are not run)"
    )

    if args.postprocess_only:
        payload_paths = []
        for task in tasks:
            payload_path = Path(str(task["payload_path"]))
            marker_path = Path(str(task["marker_path"]))
            if not checkpoint_is_valid(
                payload_path,
                marker_path,
                str(task["fingerprint"]),
                n_null_replicates,
                len(feature_columns),
            ):
                raise RuntimeError(
                    f"Incomplete checkpoint for {task['task_key']}."
                )
            payload_paths.append(payload_path)
    else:
        payload_paths = execute_tasks(
            tasks,
            workers=int(args.workers),
            max_pending=int(args.max_pending),
        )

    null_edges, null_metrics, null_bridges, null_run_audit = combine_payloads(
        payload_paths
    )
    expected_metric_rows = (
        len(tasks) * n_null_replicates * len(feature_columns)
    )
    if len(null_metrics) != expected_metric_rows:
        raise RuntimeError(
            f"Expected {expected_metric_rows} null metric rows; found {len(null_metrics)}."
        )

    structural_dir = source_run_dir / "structural_compatibility_audit"
    observed_metrics, observed_bridges, observed_run_audit = (
        observed_multiscale_tables(structural_dir, feature_columns)
    )
    if args.smoke:
        allowed_partitions = sorted(
            null_metrics["partition_id"].astype(int).unique().tolist()
        )
        allowed_settings = sorted(null_metrics["setting"].astype(str).unique())
        observed_metrics = observed_metrics.loc[
            observed_metrics["partition_id"].astype(int).isin(allowed_partitions)
            & observed_metrics["setting"].astype(str).isin(allowed_settings)
        ].copy()
        observed_bridges = observed_bridges.loc[
            observed_bridges["partition_id"].astype(int).isin(allowed_partitions)
            & observed_bridges["setting"].astype(str).isin(allowed_settings)
        ].copy()
        observed_run_audit = observed_run_audit.loc[
            observed_run_audit["partition_id"].astype(int).isin(allowed_partitions)
            & observed_run_audit["setting"].astype(str).isin(allowed_settings)
        ].copy()

    feature_summary, by_partition, by_setting, null_by_replicate = (
        summarize_feature_metrics(
            observed_metrics,
            null_metrics,
            n_null_replicates=n_null_replicates,
        )
    )
    target_summary, target_null_by_replicate = summarize_target_connectivity(
        observed_run_audit,
        null_run_audit,
        n_null_replicates=n_null_replicates,
    )
    bridge_summary, bridge_null_by_replicate = summarize_bridges(
        observed_bridges,
        null_bridges,
        observed_run_count=int(len(observed_run_audit)),
        n_null_replicates=n_null_replicates,
    )
    cross_view = build_cross_view_matrix(source_run_dir, feature_summary)

    atomic_csv_dump(
        null_edges,
        output_dir / "pag_edges_permuted_outcome_full_raw.csv.gz",
    )
    atomic_csv_dump(
        observed_metrics,
        output_dir / "observed_pag_multiscale_metrics_raw.csv",
    )
    atomic_csv_dump(
        null_metrics,
        output_dir / "permuted_pag_multiscale_metrics_raw.csv.gz",
    )
    atomic_csv_dump(
        observed_bridges,
        output_dir / "observed_exact_two_step_bridges_raw.csv",
    )
    atomic_csv_dump(
        null_bridges,
        output_dir / "permuted_exact_two_step_bridges_raw.csv.gz",
    )
    atomic_csv_dump(
        null_run_audit,
        output_dir / "permuted_pag_run_audit.csv",
    )
    atomic_csv_dump(
        feature_summary,
        output_dir / "pag_multiscale_structural_compatibility_by_feature.csv",
    )
    atomic_csv_dump(
        by_partition,
        output_dir / "pag_multiscale_by_outer_partition.csv",
    )
    atomic_csv_dump(
        by_setting,
        output_dir / "pag_multiscale_by_fci_setting.csv",
    )
    atomic_csv_dump(
        null_by_replicate,
        output_dir / "pag_multiscale_null_summary_by_replicate.csv",
    )
    atomic_csv_dump(
        target_summary,
        output_dir / "global_outcome_node_structural_signal.csv",
    )
    atomic_csv_dump(
        target_null_by_replicate,
        output_dir / "global_outcome_node_null_replicates.csv",
    )
    atomic_csv_dump(
        bridge_summary,
        output_dir / "recurrent_two_step_skeleton_bridges.csv",
    )
    atomic_csv_dump(
        bridge_null_by_replicate,
        output_dir / "bridge_null_summary_by_replicate.csv",
    )
    atomic_csv_dump(
        cross_view,
        output_dir / "credal_pag_cross_view_matrix.csv",
    )

    try:
        save_target_connectivity_figure(target_summary, target_null_by_replicate, figure_dir)
        save_multiscale_feature_figure(feature_summary, figure_dir)
        save_bridge_figure(bridge_summary, figure_dir)
        save_cross_view_figure(cross_view, feature_summary, figure_dir)
    except Exception as exc:
        atomic_json_dump(
            {"figure_error": str(exc), "traceback": traceback.format_exc()},
            output_dir / "figure_generation_error.json",
        )
        raise

    metadata = {
        "status": "complete",
        "module_name": "multiscale PAG structural compatibility extension",
        "contract_version": CONTRACT_VERSION,
        "source_run_dir": str(source_run_dir),
        "source_run_meta_sha256": file_sha256(source_run_dir / "run_meta.json"),
        "excel_path": str(excel_path),
        "excel_sha256": file_sha256(excel_path),
        "ordered_dataset_sha256": run_meta["data"]["ordered_dataset_sha256"],
        "n_samples": int(len(data)),
        "feature_columns": list(feature_columns),
        "outer_partitions": int(len(partitions) if not args.smoke else 1),
        "fci_settings": int(len(tasks) / (1 if args.smoke else len(partitions))),
        "null_replicates": int(n_null_replicates),
        "expected_fci_fits": int(len(tasks) * n_null_replicates),
        "parallel_workers": int(args.workers),
        "ci_test": STRUCTURAL_CI_TEST,
        "alpha_grid": structural_meta["alpha_grid"],
        "depth_grid": structural_meta["depth_grid"],
        "max_path_length": STRUCTURAL_MAX_PATH_LENGTH,
        "causal_learn_version": package_version("causal-learn"),
        "negative_control": (
            "Within-partition permutation of the nominal three-level injury "
            "outcome using the exact seed contract from the completed run."
        ),
        "distance_definition": (
            "Shortest path in the undirected skeleton of each PAG; endpoint "
            "marks are ignored for distance and bridge calculations."
        ),
        "bridge_definition": (
            "A feature and the injury node have shortest skeleton distance two, "
            "and the reported bridge is adjacent to both in that PAG."
        ),
        "cross_view_role": (
            "Credal predictive relevance and PAG structural compatibility are "
            "reported side by side. Neither channel validates, rejects, weights, "
            "or reranks the other."
        ),
        "inferential_boundaries": [
            "PAG proximity is not a causal effect, causal direction, mechanism, mediator, or intervention response.",
            "The 45 observed designs are five overlapping outer-training folds crossed with nine FCI settings; they are not independent replications.",
            "Permutation quantiles and upper-tail fractions are descriptive finite-sample benchmarks, not multiplicity-adjusted p-values.",
            "Credal results are class-specific; the FCI target is one nominal three-level outcome node.",
            "The two channels use the same observational sample and are not statistically independent.",
        ],
        "elapsed_minutes": float((time.time() - started) / 60.0),
    }
    atomic_json_dump(metadata, output_dir / "pag_multiscale_extension_meta.json")
    manifest_path = write_output_manifest(output_dir)
    cprint("\n" + "=" * 100)
    cprint("FCI/PAG multiscale extension completed.")
    cprint(f"Output directory: {output_dir}")
    cprint(f"Output manifest: {manifest_path}")
    cprint(f"Elapsed minutes: {metadata['elapsed_minutes']:.2f}")
    cprint("=" * 100)


if __name__ == "__main__":
    mp.freeze_support()
    main()
