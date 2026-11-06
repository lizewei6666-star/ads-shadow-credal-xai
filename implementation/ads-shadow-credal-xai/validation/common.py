from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["lr", "rf", "et", "hgb", "xgb", "lgbm", "catboost"]
SOURCES = ["SHAP", "PFI", "LOCO", "FIRM"]


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_for(*parts) -> int:
    return int(digest(list(parts))[:8], 16) % (2**31 - 1)


def replace_file(source, destination):
    """Allow brief Windows file-scanner locks without abandoning atomic writes."""
    for attempt in range(6):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 5:
                raise
            time.sleep(0.05 * 2**attempt)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        replace_file(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def environment() -> dict:
    versions = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "shap", "xgboost", "lightgbm", "catboost"]:
        try:
            module = importlib.import_module("sklearn" if name == "scikit-learn" else name)
            versions[name] = str(module.__version__)
        except (ImportError, OSError, AttributeError):
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions,
            "thread_environment": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]}}


@contextmanager
def exclusive_lock(path: Path):
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode())
    finally:
        os.close(fd)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def source_identity() -> dict:
    paths = [ROOT / "analysis/run_predictive_credal.py"]
    paths += sorted((ROOT / "validation").glob("*.py"))
    return {str(p.relative_to(ROOT)).replace("\\", "/"): file_hash(p) for p in paths}


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    required = {"profile", "phase", "seed", "replicates", "n_features", "models", "settings", "bootstrap", "cells"}
    if set(cfg) != required:
        raise ValueError(f"Configuration keys must be {sorted(required)}")
    phases = {"formal": {"pilot", "confirmatory"}, "smoke": {"smoke"}, "targeted": {"illustrative"}}
    if cfg["profile"] not in phases or cfg["phase"] not in phases[cfg["profile"]]:
        raise ValueError("Invalid profile or study phase")
    if cfg["profile"] == "smoke" and cfg["phase"] != "smoke":
        raise ValueError("Reduced settings must be labeled smoke")
    for field in ["seed", "replicates", "n_features"]:
        if type(cfg[field]) is not int or cfg[field] < (0 if field == "seed" else 1):
            raise ValueError(f"Invalid {field}")
    if cfg["n_features"] < 4:
        raise ValueError("Generators require at least four predictors")
    if not cfg["models"] or len(set(cfg["models"])) != len(cfg["models"]) or set(cfg["models"]) - set(MODELS):
        raise ValueError("Invalid model pool")
    fixed = {"outer_folds": 5, "inner_folds": 5, "shadows": 49, "shap_background": 120, "shap_cycles": 5, "pfi_repeats": 30}
    if set(cfg["settings"]) != set(fixed):
        raise ValueError("Unexpected computational settings")
    if any(type(v) is not int or v < 1 for v in cfg["settings"].values()):
        raise ValueError("Settings must be positive integers")
    if min(cfg["settings"]["outer_folds"], cfg["settings"]["inner_folds"]) < 2:
        raise ValueError("At least two folds are required")
    if cfg["profile"] == "formal" and (cfg["settings"] != fixed or cfg["models"] != MODELS or cfg["n_features"] != 14):
        raise ValueError("Formal profile must retain the published settings, model pool and 14 features")
    if cfg["profile"] == "targeted" and (cfg["settings"] != fixed or cfg["models"] != MODELS or cfg["n_features"] != 10):
        raise ValueError("Targeted profile requires 10 features, all seven learners and unchanged core settings")
    boot = cfg["bootstrap"]
    if set(boot) != {"replicates", "confidence_levels"} or type(boot["replicates"]) is not int:
        raise ValueError("Invalid bootstrap settings")
    levels = boot["confidence_levels"]
    disabled = boot["replicates"] == 0 and levels == [] and cfg["profile"] != "formal"
    if not disabled and (boot["replicates"] < 2 or not levels or len(set(levels)) != len(levels) or any(not 0 < x < 1 for x in levels) or 0.95 not in levels):
        raise ValueError("Bootstrap confidence levels must include 0.95")
    if cfg["profile"] == "formal" and boot["replicates"] < 99:
        raise ValueError("Formal bootstrap requires at least 99 draws")
    ids = []
    for cell in cfg["cells"]:
        if set(cell) != {"id", "scenario", "prior", "n"}:
            raise ValueError("Invalid experiment cell keys")
        if cell["scenario"] not in {"null", "signals", "symmetric", "interaction"} or cell["prior"] not in {"ads", "balanced"}:
            raise ValueError("Invalid generator")
        if not isinstance(cell["id"], str) or not cell["id"] or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_" for ch in cell["id"]):
            raise ValueError("Cell IDs must be lowercase letters, digits or underscores")
        if type(cell["n"]) is not int or cell["n"] < 12:
            raise ValueError("Invalid sample size")
        ids.append(cell["id"])
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Cell IDs must be unique and nonempty")
    if cfg["profile"] == "targeted":
        expected = {"signals_ads": "signals", "symmetric_ads": "symmetric", "null_ads": "null"}
        if {c["id"]: c["scenario"] for c in cfg["cells"]} != expected or any(c["n"] != 1500 or c["prior"] != "ads" for c in cfg["cells"]):
            raise ValueError("Targeted profile requires the three specified ADS-prior scenarios with 1,500 observations each")
    return cfg


def analysis_config(cfg: dict) -> dict:
    """Dataset repetition is a scheduling choice, not an analytical setting."""
    return {k: v for k, v in cfg.items() if k != "replicates" or cfg["profile"] != "targeted"}


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def finite_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite evidence")
    return arr
