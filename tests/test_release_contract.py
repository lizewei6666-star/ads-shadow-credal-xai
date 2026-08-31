from __future__ import annotations

from scripts.verify_release import (
    check_contract,
    check_private_paths,
    check_result_tables,
    check_workbook,
)


def test_method_contract() -> None:
    assert check_contract() == []


def test_analytical_workbook() -> None:
    """Validate the formal workbook when it has been supplied locally."""
    assert check_workbook() == []


def test_aggregate_tables() -> None:
    assert check_result_tables() == []


def test_no_personal_absolute_paths() -> None:
    assert check_private_paths() == []
