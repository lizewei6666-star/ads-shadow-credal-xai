"""Check the code-only release or the Git index before committing."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "provenance/release_files.json"
DATA_SUFFIXES = {
    ".csv", ".tsv", ".xlsx", ".xls", ".npz", ".npy", ".pkl", ".pickle",
    ".joblib", ".parquet", ".feather", ".arrow", ".h5", ".hdf5", ".sqlite",
    ".db", ".pt", ".pth", ".safetensors", ".log", ".zip",
}


def check_names(names):
    for name in names:
        path = Path(name)
        if path.suffix.lower() in DATA_SUFFIXES or set(path.parts) & {"data", "results", "outputs"}:
            raise ValueError(f"Data or output file is not permitted: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged", action="store_true", help="Check Git index contents rather than working files")
    args = parser.parse_args()
    expected = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    allowed = set(expected) | {MANIFEST}
    check_names(allowed)
    if args.staged:
        top = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"], text=True, encoding="utf-8").strip()
        if Path(top).resolve() != ROOT:
            raise ValueError("Run --staged in the release repository, not a parent repository")
        names = set(subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z"]).decode("utf-8").split("\0")) - {""}

        def read_file(name):
            return subprocess.check_output(["git", "-C", str(ROOT), "show", f":{name}"])
    else:
        ignored = {"outputs", ".git", ".venv", "__pycache__", ".pytest_cache"}
        names = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
                 if p.is_file() and not set(p.relative_to(ROOT).parts) & ignored}

        def read_file(name):
            return (ROOT / name).read_bytes()

    check_names(names)
    if names != allowed:
        raise ValueError(f"File list differs from release: extra={sorted(names-allowed)}, missing={sorted(allowed-names)}")
    if json.loads(read_file(MANIFEST)) != expected:
        raise ValueError("Staged release manifest differs from the checked manifest")
    for name, checksum in expected.items():
        raw = read_file(name)
        if hashlib.sha256(raw).hexdigest() != checksum:
            raise ValueError(f"File checksum differs: {name}")
        text = raw.decode("utf-8-sig")
        if re.search(r"[A-Za-z]:[/\\]Users[/\\]", text):
            raise ValueError(f"Personal absolute path: {name}")
        if re.search(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})", text):
            raise ValueError(f"Possible access token: {name}")
    original = json.loads(read_file("provenance/numerical_sources.json"))
    for name, checksum in original.items():
        if hashlib.sha256(read_file("implementation/"+name)).hexdigest() != checksum:
            raise ValueError(f"Numerical implementation changed: {name}")
    print(f"PASS: {len(names)} code/configuration files; no datasets or computed results")


if __name__ == "__main__":
    main()
