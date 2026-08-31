"""Verify the package versions used by the formal analysis."""

from __future__ import annotations

import importlib.metadata as metadata
import platform
import sys


EXPECTED = {
    "numpy": "2.2.6",
    "pandas": "2.3.3",
    "scipy": "1.15.3",
    "scikit-learn": "1.7.2",
    "shap": "0.49.1",
    "openpyxl": "3.1.5",
    "xgboost": "3.2.0",
    "lightgbm": "4.6.0",
    "catboost": "1.2.10",
    "matplotlib": "3.10.9",
    "causal-learn": "0.1.4.8",
}


def main() -> int:
    observed_python = ".".join(platform.python_version_tuple()[:2])
    passed = observed_python == "3.10"
    print(f"Python: expected=3.10, observed={observed_python}, passed={passed}")

    for package, expected in EXPECTED.items():
        try:
            observed = metadata.version(package)
        except metadata.PackageNotFoundError:
            observed = "not installed"
        item_passed = observed == expected
        passed = passed and item_passed
        print(
            f"{package}: expected={expected}, observed={observed}, "
            f"passed={item_passed}"
        )

    if not passed:
        print("Environment verification failed.", file=sys.stderr)
        return 1
    print("Environment verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

