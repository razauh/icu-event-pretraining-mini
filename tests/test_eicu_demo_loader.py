"""Tests for safe loading of synthetic eICU demo CSV tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from icu_pretrain.data.eicu_demo import MVP_TABLES, load_tables, table_path


EXPECTED_MVP_TABLES = [
    "patient",
    "apachePatientResult",
    "diagnosis",
    "lab",
    "medication",
    "infusionDrug",
    "treatment",
    "vitalPeriodic",
    "vitalAperiodic",
]


def write_table(raw_dir: Path, table_name: str, **columns: list[object]) -> None:
    pd.DataFrame(columns).to_csv(table_path(raw_dir, table_name), index=False)


def test_mvp_table_names_match_the_plan() -> None:
    assert MVP_TABLES == EXPECTED_MVP_TABLES


def test_table_path_resolves_csv_under_raw_directory(tmp_path: Path) -> None:
    assert table_path(tmp_path, "patient") == tmp_path / "patient.csv"


def test_load_tables_reads_all_synthetic_mvp_tables(tmp_path: Path) -> None:
    for table_name in MVP_TABLES:
        write_table(tmp_path, table_name, patientunitstayid=[101])

    tables = load_tables(tmp_path)

    assert list(tables) == MVP_TABLES
    assert all(frame["patientunitstayid"].tolist() == [101] for frame in tables.values())


def test_load_tables_reads_only_the_requested_subset(tmp_path: Path) -> None:
    write_table(tmp_path, "patient", patientunitstayid=[101])

    tables = load_tables(tmp_path, tables=["patient"])

    assert list(tables) == ["patient"]


def test_load_tables_rejects_missing_table_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="diagnosis.csv"):
        load_tables(tmp_path, tables=["diagnosis"])


def test_load_tables_rejects_missing_patientunitstayid(tmp_path: Path) -> None:
    write_table(tmp_path, "lab", labname=["creatinine"])

    with pytest.raises(ValueError, match="lab.*patientunitstayid"):
        load_tables(tmp_path, tables=["lab"])


def test_load_tables_rejects_csv_without_columns(tmp_path: Path) -> None:
    table_path(tmp_path, "medication").touch()

    with pytest.raises(ValueError, match="medication.*empty"):
        load_tables(tmp_path, tables=["medication"])


def test_load_tables_accepts_header_only_table(tmp_path: Path) -> None:
    write_table(tmp_path, "treatment", patientunitstayid=[])

    tables = load_tables(tmp_path, tables=["treatment"])

    assert tables["treatment"].empty
    assert list(tables["treatment"].columns) == ["patientunitstayid"]


def test_load_tables_preserves_extra_columns_and_duplicate_stays(tmp_path: Path) -> None:
    write_table(
        tmp_path,
        "diagnosis",
        patientunitstayid=[101, 101],
        diagnosisstring=["sepsis", "respiratory failure"],
    )

    tables = load_tables(tmp_path, tables=["diagnosis"])

    assert tables["diagnosis"]["patientunitstayid"].tolist() == [101, 101]
    assert tables["diagnosis"]["diagnosisstring"].tolist() == [
        "sepsis",
        "respiratory failure",
    ]


def test_load_tables_rejects_incorrect_table_capitalization(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown eICU demo table.*InfusionDrug"):
        load_tables(tmp_path, tables=["InfusionDrug"])
