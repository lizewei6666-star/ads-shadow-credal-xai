"""Predictive learners and native-scale SHAP, PFI, LOCO, and FIRM statistics."""

from __future__ import annotations
import gc
import hashlib
import importlib.util
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd

EXCLUDE_UNKNOWN_VEHICLE_AGE = True
RUN_PROFILE = os.environ.get("ADS_XAI_RUN_PROFILE", "full").strip().lower()
CANDIDATE_MODELS = ["lr", "rf", "et", "hgb", "xgb", "lgbm", "catboost"]
INNER_CV_FOLDS = 5
MODEL_N_JOBS = 1
USE_BALANCED_SAMPLE_WEIGHT = True
OUTPUT_FIGURES = False
ENABLE_PFI_BATCH_PREDICTION = True
PFI_BATCH_PREDICTION_ROWS = int(os.environ.get("ADS_XAI_PFI_BATCH_PREDICTION_ROWS", "8192"))
FIRM_CONTINUOUS_FEATURES: List[str] = []
ENABLE_CALIBRATION_DIAGNOSTICS = False
NEGATIVE_CONTROL_FEATURE = "Random_Noise_Control"
SOURCE_WEIGHT_EPS = 1e-12
SHAP_BACKGROUND_SIZE = 120
SHAP_PERMUTATION_CYCLES = 5
TARGET_SHAP_CONTRACT_VERSION = "target_coordinate_permutation_shap_v1"
TARGET_SHAP_PERMUTATION_PLAN_VERSION = "common_plan_v1"
TARGET_SHAP_PREDICTION_BATCH_ROWS = int(
    os.environ.get("ADS_XAI_TARGET_SHAP_PREDICTION_BATCH_ROWS", "8192")
)
ENABLE_TARGET_SHAP_EQUIVALENCE_AUDIT = False
ENABLE_TARGET_SHAP_PLAN_SENSITIVITY_AUDIT = False
PERMUTATION_REPEATS = 30
STABLE_RECORD_ID_COL = "_stable_record_id"
TARGET_COL = "Injury_3Class"
TARGET_MAP = {
    "No_Injury": 0,
    "Minor_Injury": 1,
    "ModeratePlus_Injury": 2,
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
            feature: tuple(sorted(X_train[feature].astype(str).dropna().unique().tolist()))
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
                out[:, pos] = pd.to_numeric(X[feature], errors="coerce").to_numpy(dtype=float)
                continue
            mapping = {value: idx for idx, value in enumerate(self.categorical_levels[feature])}
            out[:, pos] = X[feature].astype(str).map(mapping).fillna(-1).to_numpy(dtype=float)
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


def cprint(msg: str = "") -> None:
    print(msg, flush=True)


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


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


def stable_json_digest(obj: object) -> str:
    payload = json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    h = hashlib.sha256()
    h.update(str(array.dtype).encode("ascii"))
    h.update(stable_json_digest(list(array.shape)).encode("ascii"))
    h.update(array.tobytes())
    return h.hexdigest()


def sample_indices(n: int, max_n: Optional[int], seed: int) -> np.ndarray:
    idx = np.arange(n)
    if max_n is None or n <= int(max_n):
        return idx
    rs = np.random.RandomState(int(seed))
    # A permutation prefix gives reproducible, nested background subsets.
    out = rs.permutation(idx)[: int(max_n)]
    return np.sort(out)


def require_full_evaluation(
    table: pd.DataFrame,
    count_column: str,
    expected_n: int,
    source_name: str,
) -> None:
    """Require each source to cover every evaluation record."""
    if table is None or table.empty:
        return
    if count_column not in table.columns:
        raise RuntimeError(f"{source_name} output is missing {count_column}.")
    observed = set(pd.to_numeric(table[count_column], errors="coerce").dropna().astype(int))
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

    # Categorical coding avoids imposing a linear January-to-December effect.
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
    feature_cols = [c for c in df.columns if c not in {TARGET_COL, STABLE_RECORD_ID_COL}]
    numeric_cols = [c for c in ["Is_Weekend", "Incident_Hour"] if c in feature_cols]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return FeatureSchema(
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )


def split_xy(df: pd.DataFrame, schema: FeatureSchema) -> Tuple[pd.DataFrame, np.ndarray]:
    columns = list(schema.feature_cols)
    if STABLE_RECORD_ID_COL in df.columns:
        columns.append(STABLE_RECORD_ID_COL)
    X = df[columns].copy()
    y = df[TARGET_COL].astype(int).to_numpy()
    return X, y


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
    if schema.numeric_cols:
        transformers.append(("num", Pipeline([("scaler", StandardScaler())]), schema.numeric_cols))
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
                c for c in schema.categorical_cols if stripped == c or stripped.startswith(c + "_")
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
        raise RuntimeError(f"Failed to map encoded columns back to original features: {empty}")
    return groups


def transform_raw(preprocessor, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(preprocessor.transform(X), dtype=float)


MODEL_DISPLAY_NAMES = {
    "lr": "Logistic regression",
    "rf": "Random forest",
    "et": "Extra trees",
    "hgb": "Histogram gradient boosting",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "catboost": "CatBoost",
}


MODEL_DEPENDENCIES = {"xgb": "xgboost", "lgbm": "lightgbm", "catboost": "catboost"}


def preprocessing_mode(model_name: str) -> str:
    return "training-fitted standardization and one-hot encoding"


def imbalance_handling_mode(model_name: str) -> str:
    if not USE_BALANCED_SAMPLE_WEIGHT:
        return "none"
    return "balanced training-sample weights"


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
            n_estimators=600, min_samples_leaf=2, random_state=seed, n_jobs=n_jobs
        )
    if name == "et":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_estimators=800, min_samples_leaf=2, random_state=seed, n_jobs=n_jobs
        )
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=350, max_depth=None, random_state=seed
        )
    if name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
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
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
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
    if sample_weight is None:
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


def predict_proba_aligned(estimator, X: np.ndarray, classes_expected: Iterable[int]) -> np.ndarray:
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
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
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
            restricted = [int(value) for value in master if int(value) in varying_set]
            order_position = {feature_pos: order for order, feature_pos in enumerate(restricted)}
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
                terms.setdefault(term_key, []).append((s_mask, sj_mask, r_mask, rj_mask))
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
    target_column = {int(position): column for column, position in enumerate(target_positions)}
    for (record_pos, target_pos), coordinate_terms in terms.items():
        contributions = []
        for s_mask, sj_mask, r_mask, rj_mask in coordinate_terms:
            delta_forward = state_values[(record_pos, sj_mask)] - state_values[(record_pos, s_mask)]
            delta_reverse = state_values[(record_pos, rj_mask)] - state_values[(record_pos, r_mask)]
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
        "target_unique_mask_states": int(evaluation_diagnostics["unique_mask_states"]),
        "target_background_rows_predicted": int(
            evaluation_diagnostics["background_rows_predicted"]
        ),
        "target_python_prediction_calls": int(evaluation_diagnostics["python_prediction_calls"]),
        "min_varying_features": int(np.min(varying_counts)) if len(varying_counts) else 0,
        "mean_varying_features": float(np.mean(varying_counts)) if len(varying_counts) else 0.0,
        "max_varying_features": int(np.max(varying_counts)) if len(varying_counts) else 0,
        "min_permutations": int(np.min(permutation_counts)) if len(permutation_counts) else 0,
        "mean_permutations": float(np.mean(permutation_counts)) if len(permutation_counts) else 0.0,
        "max_permutations": int(np.max(permutation_counts)) if len(permutation_counts) else 0,
    }
    return values, diagnostics


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
                "focal_mean_abs_shap": (
                    float(np.mean(np.abs(signed[focal_mask]))) if np.any(focal_mask) else np.nan
                ),
                "other_mean_abs_shap": (
                    float(np.mean(np.abs(signed[other_mask]))) if np.any(other_mask) else np.nan
                ),
                "focal_mean_signed_shap": (
                    float(np.mean(signed[focal_mask])) if np.any(focal_mask) else np.nan
                ),
                "other_mean_signed_shap": (
                    float(np.mean(signed[other_mask])) if np.any(other_mask) else np.nan
                ),
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
        out["rank"] = out.groupby(["seed", "model", "class_id"])["mean_abs_shap"].rank(
            ascending=False, method="average"
        )
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
        raise RuntimeError(f"Evaluation data are missing {STABLE_RECORD_ID_COL}.")
    requested = [str(feature) for feature in target_features]
    feature_to_position = {
        str(feature): position for position, feature in enumerate(schema.feature_cols)
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
    full_path_budget = int(SHAP_PERMUTATION_CYCLES) * (2 * len(schema.feature_cols) + 1)
    values, diagnostics = _target_coordinate_shap_values(
        estimator=estimator,
        preprocessor=preprocessor,
        codec=codec,
        X_eval_mask=X_eval_mask,
        background=background,
        stable_record_ids=stable_record_ids,
        classes=classes_array,
        target_positions=[int(feature_to_position[feature]) for feature in requested],
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
            "background_index_digest": array_digest(np.asarray(background_indices, dtype=np.int64)),
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
    raise ValueError(f"Unknown classwise Brier evaluation measure: {evaluation_measure}")


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
            "PFI base probabilities have shape " f"{base.shape}; expected {expected_shape}."
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
        parts.append(predict_proba_aligned(estimator, X_encoded[start:stop], classes))
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
            "PFI permuted probabilities must have shape " "(repeats, records, classes)."
        )
    repeats, n_eval, n_classes = permutations.shape
    if (n_eval, n_classes) != base.shape:
        raise ValueError("PFI permuted and base probability shapes are inconsistent.")

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
            loss_increases[cid].append(permuted_loss - base_losses_balanced[cid])
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
            permuted_focal_losses[cid].append(float(np.mean(squared_error[focal_mask])))
            permuted_other_losses[cid].append(float(np.mean(squared_error[other_mask])))

    rows: List[Dict[str, object]] = []
    for class_id in classes:
        cid = int(class_id)
        increases = np.asarray(loss_increases[cid], dtype=float)
        prevalence_increases = np.asarray(prevalence_loss_increases[cid], dtype=float)
        shifts = np.asarray(probability_shifts[cid], dtype=float)
        focal_permuted = np.asarray(permuted_focal_losses[cid], dtype=float)
        other_permuted = np.asarray(permuted_other_losses[cid], dtype=float)
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
                "balanced_baseline_class_brier": float(base_losses_balanced[cid]),
                "prevalence_baseline_class_brier": float(base_losses_prevalence[cid]),
                "mean_brier_increase": mean_increase,
                "positive_brier_increase": float(max(mean_increase, 0.0)),
                "balanced_mean_brier_increase": mean_increase,
                "positive_balanced_brier_increase": float(max(mean_increase, 0.0)),
                "prevalence_mean_brier_increase": prevalence_mean_increase,
                "positive_prevalence_brier_increase": float(max(prevalence_mean_increase, 0.0)),
                "std_brier_increase": (
                    float(np.std(increases, ddof=1)) if len(increases) > 1 else 0.0
                ),
                "positive_repeat_share": float(np.mean(increases > 0.0)),
                "mean_abs_probability_shift": float(np.mean(shifts)),
                "std_abs_probability_shift": (
                    float(np.std(shifts, ddof=1)) if len(shifts) > 1 else 0.0
                ),
                "baseline_focal_brier_mean": float(base_group["focal_mean"]),
                "baseline_other_brier_mean": float(base_group["other_mean"]),
                "permuted_focal_brier_mean": float(np.mean(focal_permuted)),
                "permuted_other_brier_mean": float(np.mean(other_permuted)),
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
        feature_rs = np.random.RandomState(stable_seed_from_key("pfi", seed, model_name, feature))
        predictions = []
        for _ in range(int(repeats)):
            Xp = X_eval_raw.copy()
            Xp[feature] = feature_rs.permutation(Xp[feature].to_numpy())
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
        out["rank"] = out.groupby(["seed", "model", "class_id"])["positive_brier_increase"].rank(
            ascending=False, method="average"
        )
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
        feature_rs = np.random.RandomState(stable_seed_from_key("pfi", seed, model_name, feature))
        raw_batches = []
        for _ in range(int(repeats)):
            Xp = X_eval_raw.copy()
            Xp[feature] = feature_rs.permutation(Xp[feature].to_numpy())
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
                "Batched PFI returned an unexpected probability shape: " f"{stacked_proba.shape}."
            )
        rows.extend(
            _summarize_class_permutation_predictions(
                base=base,
                permuted_probabilities=stacked_proba.reshape(int(repeats), n_eval, len(classes)),
                y_eval=y_eval,
                classes=classes,
                feature=str(feature),
                model_name=model_name,
                seed=int(seed),
            )
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["rank"] = out.groupby(["seed", "model", "class_id"])["positive_brier_increase"].rank(
            ascending=False, method="average"
        )
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
    base_metrics = evaluate_predictions(y_eval, np.asarray(base_pred, dtype=int), labels=classes)
    reduced_pred = classes[np.argmax(reduced_proba, axis=1)]
    reduced_metrics = evaluate_predictions(y_eval, reduced_pred, labels=classes)
    balacc_drop = float(base_metrics["balanced_accuracy"] - reduced_metrics["balanced_accuracy"])
    macrof1_drop = float(base_metrics["macro_f1"] - reduced_metrics["macro_f1"])

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
        full_squared_error = (binary_target - base_proba[:, class_pos]) ** 2
        reduced_squared_error = (binary_target - reduced_proba[:, class_pos]) ** 2
        brier_increase = float(reduced_class_brier - base_class_brier)
        prevalence_brier_increase = float(
            prevalence_reduced_class_brier - prevalence_base_class_brier
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
                "positive_brier_increase": float(max(brier_increase, 0.0)),
                "balanced_brier_increase": brier_increase,
                "positive_balanced_brier_increase": float(max(brier_increase, 0.0)),
                "prevalence_baseline_class_brier": (prevalence_base_class_brier),
                "prevalence_reduced_class_brier": (prevalence_reduced_class_brier),
                "prevalence_brier_increase": prevalence_brier_increase,
                "positive_prevalence_brier_increase": float(max(prevalence_brier_increase, 0.0)),
                "mean_probability_change": weighted_mean(delta, balanced_weights),
                "mean_abs_probability_change": weighted_mean(np.abs(delta), balanced_weights),
                "prevalence_mean_probability_change": float(np.mean(delta)),
                "prevalence_mean_abs_probability_change": float(np.mean(np.abs(delta))),
                "std_probability_change": float(np.std(delta)),
                "full_focal_brier_mean": float(np.mean(full_squared_error[focal_mask])),
                "full_other_brier_mean": float(np.mean(full_squared_error[other_mask])),
                "reduced_focal_brier_mean": float(np.mean(reduced_squared_error[focal_mask])),
                "reduced_other_brier_mean": float(np.mean(reduced_squared_error[other_mask])),
                "n_focal": int(np.sum(focal_mask)),
                "n_other": int(np.sum(other_mask)),
                "balanced_accuracy_drop": balacc_drop,
                "macro_f1_drop": macrof1_drop,
                "class_recall_drop": float(
                    base_metrics.get(recall_key, 0.0) - reduced_metrics.get(recall_key, 0.0)
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
    precomputed_reduced_proba: Optional[Mapping[str, np.ndarray]] = None,
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
        out["rank"] = out.groupby(["seed", "model", "class_id"])["positive_brier_increase"].rank(
            ascending=False, method="average"
        )
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
        raise ValueError("FIRM probabilities and holdout records have different lengths.")
    if probabilities.shape[1] != len(classes):
        raise ValueError("FIRM probability columns do not match the supplied classes.")
    if len(y_eval) != len(X_eval):
        raise ValueError("FIRM labels and holdout records have different lengths.")

    rows = []
    for feature in features:
        if feature not in X_eval.columns:
            raise ValueError(f"FIRM feature is absent from the evaluation table: {feature}")

        raw_series = X_eval[feature]
        numeric_series = pd.to_numeric(raw_series, errors="coerce")
        use_empirical_levels = (
            feature not in set(FIRM_CONTINUOUS_FEATURES) and feature != NEGATIVE_CONTROL_FEATURE
        )

        for class_pos, class_id in enumerate(classes):
            class_scores = probabilities[:, class_pos]
            score_is_finite = np.isfinite(class_scores)

            balanced_row_weights = class_balanced_evaluation_weights(y_eval, int(class_id))
            prevalence_row_weights = np.full(len(y_eval), 1.0 / float(len(y_eval)), dtype=float)

            def estimate(row_weights: np.ndarray) -> Dict[str, float]:
                if use_empirical_levels:
                    valid = raw_series.notna().to_numpy() & score_is_finite
                    frame = pd.DataFrame(
                        {
                            "feature_value": raw_series.loc[valid].astype(str).to_numpy(),
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
                        group_weights = grouped["group_weight"].to_numpy(dtype=float) / total
                    conditional_means = grouped["weighted_score"].to_numpy(
                        dtype=float
                    ) / np.maximum(grouped["group_weight"].to_numpy(dtype=float), SOURCE_WEIGHT_EPS)
                    mean_score = float(np.sum(group_weights * conditional_means))
                    variance = float(np.sum(group_weights * (conditional_means - mean_score) ** 2))
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
                    float(np.sum(weights_valid * x_centered * score_centered) / denominator)
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
                    "balanced_firm_importance": float(np.sqrt(max(balanced["variance"], 0.0))),
                    "prevalence_firm_importance": float(np.sqrt(max(prevalence["variance"], 0.0))),
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
        out["rank"] = out.groupby(["seed", "model", "class_id"])["firm_importance"].rank(
            ascending=False, method="average"
        )
    return out


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
        raise KeyError(f"Unknown explanation source: {source_name}")
    value_column = str(CREDAL_SOURCE_SPECS[source_name]["value"])
    if value_column not in table.columns:
        raise KeyError(f"{source_name} output lacks its prespecified statistic: {value_column}")
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
    loco_reduced_proba_by_feature: Optional[Mapping[str, np.ndarray]] = None,
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
    return float(0.5 * np.mean(squared_error[focal]) + 0.5 * np.mean(squared_error[other]))


def concat_or_empty(frames: List[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


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
        raise RuntimeError("Inner cross-validation summaries are required for model selection.")
    selected: Dict[int, List[str]] = {}
    for seed in seeds:
        group = metrics_by_split.loc[metrics_by_split["seed"].astype(int) == int(seed)].copy()
        if group.empty:
            raise RuntimeError(f"No inner cross-validation summary was produced for seed={seed}.")
        if "selected_by_one_se" not in group.columns:
            raise RuntimeError("The inner cross-validation table lacks one-SE selection flags.")
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
            "outer_repeat": (
                int(group["outer_repeat"].iloc[0]) if "outer_repeat" in group else np.nan
            ),
            "outer_fold": int(group["outer_fold"].iloc[0]) if "outer_fold" in group else np.nan,
            "split_seed": int(group["split_seed"].iloc[0]) if "split_seed" in group else np.nan,
        }
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"mean_{metric}"] = float(values.mean()) if len(values) else np.nan
            row[f"std_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"se_{metric}"] = (
                float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
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
        if col.startswith("mean_") and pd.api.types.is_numeric_dtype(metrics_by_split[col])
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
                make_balanced_sample_weight(y_subtrain) if USE_BALANCED_SAMPLE_WEIGHT else None
            )
            for model_name in model_names:
                cprint(f"[BENCHMARK] seed={seed} fold={fold}/{n_splits} model={model_name}")
                encoded = fit_transform_split(
                    X_subtrain, X_validation, schema, model_name=model_name
                )
                estimator = build_estimator(
                    model_name,
                    seed=stable_seed_from_key("inner_cv", seed, fold, model_name),
                    n_jobs=MODEL_N_JOBS,
                    categorical_feature_indices=encoded.categorical_feature_indices,
                )
                fit_estimator(estimator, encoded.X_train, y_subtrain, sample_weight=sample_weight)
                proba = predict_proba_aligned(estimator, encoded.X_test, class_order)
                y_pred = np.asarray(class_order, dtype=int)[np.argmax(proba, axis=1)]
                metrics = evaluate_predictions(y_validation, y_pred, labels=class_order)
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
        output_dir / "tables" / "model_selection_metrics_raw.csv", index=False, encoding="utf-8-sig"
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
    best_model = select_best_model(metrics_summary)
    cprint(
        f"[INNER-CV SUMMARY LEADER] {best_model} | {MODEL_DISPLAY_NAMES.get(best_model, best_model)}"
    )
    return (metrics_raw, metrics_by_split, metrics_summary, best_model)
