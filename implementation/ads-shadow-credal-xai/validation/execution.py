from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time

import pandas as pd

from .common import analysis_config, digest, file_hash, replace_file, write_json


def run_identity(cfg, env, sources):
    identity = {"config": analysis_config(cfg), "environment": env, "sources": sources}
    return digest(identity)


def prepare_manifest(root, cfg, env, sources, requested_replicates=None):
    """Extend a targeted run without changing existing dataset identities."""
    cfg = dict(cfg)
    if requested_replicates is not None:
        if cfg["profile"] != "targeted" or type(requested_replicates) is not int or requested_replicates < 1:
            raise ValueError("--replicates is a positive total allowed only for the targeted profile")
        cfg["replicates"] = requested_replicates
    path = root / "run_manifest.json"
    fingerprint = run_identity(cfg, env, sources)
    old = None
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["fingerprint"] != fingerprint:
            raise ValueError("Configuration, source, or environment changed. Use a new output directory.")
        previous = old["config"]["replicates"]
        if cfg["profile"] == "targeted":
            if requested_replicates is not None and requested_replicates < previous:
                raise ValueError("Cannot reduce the existing replicate plan")
            cfg["replicates"] = max(previous, cfg["replicates"])
    elif any(p.name != "run_session.lock" for p in root.iterdir()):
        raise ValueError("Output directory is not empty and has no validation manifest")
    manifest = {"config": cfg, "environment": env, "sources": sources, "fingerprint": fingerprint}
    history = [] if old is None else old.get("replicate_plan_history", [])
    if old is None or old["config"]["replicates"] != cfg["replicates"]:
        history = history + [{"replicates": cfg["replicates"], "unix_time": time.time()}]
    if cfg["profile"] == "targeted":
        manifest["replicate_plan_history"] = history
    write_json(path, manifest)
    return cfg, fingerprint


def cached_selection(core, x, y, schema, parts, out, identity):
    manifest_path = out / "selection_cache.json"
    selection_path = out / "selection.pkl"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved["identity"] != identity:
            raise ValueError("Model-selection cache belongs to another dataset or run")
        for name, checksum in saved["artifacts"].items():
            path = out / name
            if not path.is_file() or file_hash(path) != checksum:
                raise ValueError(f"Model-selection cache failed verification: {name}")
        # This is a locally produced, checksum-verified checkpoint, like the core caches.
        return pd.read_pickle(selection_path), saved["seconds"], True
    started = time.perf_counter()
    _, selection, _, _ = core.run_prediction_benchmark(x, y, schema, parts, out)
    seconds = time.perf_counter() - started
    fd, name = tempfile.mkstemp(dir=out, suffix=".tmp")
    os.close(fd)
    temporary = Path(name)
    try:
        selection.to_pickle(temporary)
        replace_file(temporary, selection_path)
    finally:
        temporary.unlink(missing_ok=True)
    paths = sorted((out / "tables").glob("model_selection_*.csv")) + [selection_path]
    write_json(manifest_path, {"identity": identity, "seconds": seconds,
               "artifacts": {p.relative_to(out).as_posix(): file_hash(p) for p in paths}})
    return selection, seconds, False


def workload(cfg, admitted):
    rows = []
    units_per_entry = cfg["n_features"] * cfg["settings"]["shadows"]
    for fold, models in admitted.items():
        rows.append({"outer_partition": int(fold), "models": models,
                     "shadow_units": len(models) * units_per_entry})
    units = sum(row["shadow_units"] for row in rows)
    return {"n_features": cfg["n_features"], "folds": rows,
            "n_admitted_model_fold_entries": sum(len(row["models"]) for row in rows),
            "shadow_units": units, "shadow_full_fit_count": units * cfg["settings"]["inner_folds"],
            "bootstrap_comparator_enabled": cfg["bootstrap"]["replicates"] > 0,
            "scope": "Shadow refits only; excludes selection, real-data explanations and LOCO refits. Counts are not a runtime estimate."}
