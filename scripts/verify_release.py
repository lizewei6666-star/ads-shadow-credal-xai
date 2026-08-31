"""Run deterministic checks on the publication repository."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import py_compile
import re
import sys
from pathlib import Path
from typing import Any

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
CHECKSUM_FILE = ROOT / "provenance" / "SHA256SUMS.txt"
DATA_FILE = ROOT / "data" / "ads_crashes.xlsx"
EXPECTED_DATA_SHA256 = (
    "5d70b42898355269607df61a7ca8b2ac3255193d5cbd846fef9398a7c577c205"
)
EXPECTED_COLUMNS = [
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
    "Injury_3Class",
]

REQUIRED_RESULT_ROWS = {
    "results/predictive/model_selection_metrics_summary.csv": 7,
    "results/predictive/admissible_model_holdout_metrics_raw.csv": 12,
    "results/predictive/outer_holdout_shadow_calibrated_relevance.csv": 2016,
    "results/predictive/outer_partition_manifest.csv": 5,
    "results/predictive/credal_common_support_coverage.csv": 15,
    "results/predictive/credal_feature_intervals_and_partial_order.csv": 42,
    "results/predictive/credal_hasse_edges.csv": 53,
    "results/pag/global_outcome_node_structural_signal.csv": 3,
    "results/pag/pag_multiscale_structural_compatibility_by_feature.csv": 14,
    "results/pag/recurrent_two_step_skeleton_bridges.csv": 14,
    "results/pag/credal_pag_cross_view_matrix.csv": 42,
}

EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".matplotlib",
    ".matplotlib_cache",
    "generated",
    "outputs",
}
EXCLUDED_RELEASE_FILES = {
    "data/ads_crashes.xlsx",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == CHECKSUM_FILE:
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if relative.as_posix() in EXCLUDED_RELEASE_FILES:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def write_checksums() -> None:
    lines = [
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in release_files()
    ]
    CHECKSUM_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKSUM_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_checksums() -> list[str]:
    errors: list[str] = []
    if not CHECKSUM_FILE.is_file():
        return ["Missing provenance/SHA256SUMS.txt"]
    expected: dict[str, str] = {}
    for line in CHECKSUM_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        digest, relative = line.split(maxsplit=1)
        expected[relative.strip()] = digest.lower()

    observed_files = {
        path.relative_to(ROOT).as_posix(): path for path in release_files()
    }
    if set(expected) != set(observed_files):
        missing = sorted(set(expected).difference(observed_files))
        extra = sorted(set(observed_files).difference(expected))
        if missing:
            errors.append(f"Files missing from repository: {missing}")
        if extra:
            errors.append(f"Files missing from checksum manifest: {extra}")
    for relative, path in observed_files.items():
        if relative in expected and sha256(path) != expected[relative]:
            errors.append(f"Checksum mismatch: {relative}")
    return errors


def literal_constants(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            continue
    return values


def check_contract() -> list[str]:
    errors: list[str] = []
    predictive = literal_constants(ROOT / "analysis" / "run_predictive_credal.py")
    pag = literal_constants(ROOT / "analysis" / "run_pag_structural.py")
    expected_predictive = {
        "EXPECTED_N_SAMPLES": 1469,
        "OUTER_N_SPLITS_FULL": 5,
        "OUTER_N_REPEATS_FULL": 1,
        "INNER_CV_FOLDS": 5,
        "SHADOW_REPLICATES_FULL": 49,
        "SHAP_BACKGROUND_SIZE": 120,
        "CANDIDATE_MODELS": ["lr", "rf", "et", "hgb", "xgb", "lgbm", "catboost"],
        "TARGET_SHAP_CONTRACT_VERSION": "target_coordinate_permutation_shap_v1",
        "TARGET_SHAP_PERMUTATION_PLAN_VERSION": "common_plan_v1",
    }
    expected_pag = {
        "EXPECTED_N_SAMPLES": 1469,
        "EXPECTED_CAUSAL_LEARN_VERSION": "0.1.4.8",
        "STRUCTURAL_RANDOM_SEED": 20260723,
        "OUTER_RANDOM_SEED": 42,
        "STRUCTURAL_CI_TEST": "gsq",
        "NULL_REPLICATES": 99,
    }
    for name, expected in expected_predictive.items():
        if predictive.get(name) != expected:
            errors.append(
                f"Predictive contract mismatch for {name}: "
                f"{predictive.get(name)!r} != {expected!r}"
            )
    for name, expected in expected_pag.items():
        if pag.get(name) != expected:
            errors.append(
                f"PAG contract mismatch for {name}: "
                f"{pag.get(name)!r} != {expected!r}"
            )
    return errors


def check_workbook(required: bool = False) -> list[str]:
    errors: list[str] = []
    if not DATA_FILE.is_file():
        if required:
            errors.append(
                "Missing analytical workbook: data/ads_crashes.xlsx. "
                "See data/README.md for access instructions."
            )
        return errors
    if sha256(DATA_FILE) != EXPECTED_DATA_SHA256:
        errors.append("The analytical workbook SHA-256 does not match the formal run.")
        return errors
    workbook = load_workbook(DATA_FILE, read_only=True, data_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    header = [cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1))]
    if header != EXPECTED_COLUMNS:
        errors.append(f"Unexpected workbook columns: {header}")
    if sheet.max_row - 1 != 1469:
        errors.append(f"Unexpected analytical row count: {sheet.max_row - 1}")
    if sheet.max_column != 15:
        errors.append(f"Unexpected workbook column count: {sheet.max_column}")
    workbook.close()
    return errors


def check_result_tables() -> list[str]:
    errors: list[str] = []
    for relative, expected_rows in REQUIRED_RESULT_ROWS.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"Missing result table: {relative}")
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = sum(1 for _ in csv.reader(handle)) - 1
        if rows != expected_rows:
            errors.append(
                f"Unexpected row count for {relative}: {rows} != {expected_rows}"
            )
    return errors


def check_python_syntax() -> list[str]:
    errors: list[str] = []
    for path in sorted(ROOT.rglob("*.py")):
        if any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts):
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(str(exc))
    return errors


def check_private_paths() -> list[str]:
    errors: list[str] = []
    pattern = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+", re.IGNORECASE)
    suffixes = {".py", ".md", ".ps1", ".yml", ".yaml", ".cff"}
    for path in release_files():
        if path.suffix.lower() not in suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            errors.append(f"Personal absolute path found in {path.relative_to(ROOT)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-checksums",
        action="store_true",
        help="Regenerate provenance/SHA256SUMS.txt after an intentional edit.",
    )
    parser.add_argument(
        "--require-data",
        action="store_true",
        help="Fail when data/ads_crashes.xlsx is unavailable or differs from the formal run.",
    )
    args = parser.parse_args()
    if args.write_checksums:
        write_checksums()
        print(f"Wrote {CHECKSUM_FILE.relative_to(ROOT)}")
        return 0

    checks = [
        ("Python syntax", check_python_syntax),
        ("method contract", check_contract),
        ("analytical workbook", lambda: check_workbook(required=args.require_data)),
        ("aggregate tables", check_result_tables),
        ("portable paths", check_private_paths),
        ("release checksums", verify_checksums),
    ]
    failures: list[str] = []
    for label, check in checks:
        errors = check()
        print(f"{label}: {'passed' if not errors else 'failed'}")
        failures.extend(errors)
    if failures:
        for error in failures:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Release verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
