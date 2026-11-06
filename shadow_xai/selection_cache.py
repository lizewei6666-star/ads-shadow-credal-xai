"""Model-admission checkpoints with content-hash verification."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import pandas as pd


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    write_json(
        manifest_path,
        {
            "identity": identity,
            "seconds": seconds,
            "artifacts": {p.relative_to(out).as_posix(): file_hash(p) for p in paths},
        },
    )
    return selection, seconds, False
