"""Tests for safe loading of synthetic eICU demo CSV tables."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest

from icu_pretrain.data.eicu_demo import (
    MVP_TABLES,
    load_tables,
    table_path,
    discover_raw_dir,
    load_table_chunks,
    load_table_full,
)

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

TABLE_REQUIRED_COLS = {
    "patient": {
        "patientunitstayid": [101],
        "uniquepid": ["p1"],
        "hospitalid": ["h1"],
        "hospitaldischargestatus": ["Alive"],
        "unitdischargeoffset": [1440],
    },
    "apachePatientResult": {
        "patientunitstayid": [101],
    },
    "diagnosis": {
        "patientunitstayid": [101],
        "diagnosisstring": ["sepsis"],
        "diagnosisoffset": [100],
    },
    "lab": {
        "patientunitstayid": [101],
        "labname": ["creatinine"],
        "labresult": [1.2],
        "labresultoffset": [120],
    },
    "medication": {
        "patientunitstayid": [101],
        "drugname": ["aspirin"],
        "drughiclseqno": [1234],
        "drugstartoffset": [60],
    },
    "infusionDrug": {
        "patientunitstayid": [101],
        "drugname": ["insulin"],
        "infusionoffset": [90],
    },
    "treatment": {
        "patientunitstayid": [101],
        "treatmentstring": ["ventilation"],
        "treatmentoffset": [150],
    },
    "vitalPeriodic": {
        "patientunitstayid": [101],
        "observationoffset": [180],
    },
    "vitalAperiodic": {
        "patientunitstayid": [101],
        "observationoffset": [180],
    },
}


def write_table(raw_dir: Path, table_name: str, **columns: list[object]) -> None:
    pd.DataFrame(columns).to_csv(table_path(raw_dir, table_name), index=False)


def write_valid_table(raw_dir: Path, table_name: str, **overrides: list[object]) -> None:
    data = dict(TABLE_REQUIRED_COLS.get(table_name, {"patientunitstayid": [101]}))
    data.update(overrides)
    pd.DataFrame(data).to_csv(table_path(raw_dir, table_name), index=False)


def test_mvp_table_names_match_the_plan() -> None:
    assert MVP_TABLES == EXPECTED_MVP_TABLES


def test_table_path_resolves_csv_under_raw_directory(tmp_path: Path) -> None:
    assert table_path(tmp_path, "patient") == tmp_path / "patient.csv.gz"


def test_load_tables_reads_all_synthetic_mvp_tables(tmp_path: Path) -> None:
    for table_name in MVP_TABLES:
        write_valid_table(tmp_path, table_name)

    tables = load_tables(tmp_path)

    assert list(tables) == MVP_TABLES
    assert all(frame["patientunitstayid"].tolist() == [101] for frame in tables.values())


def test_load_tables_reads_only_the_requested_subset(tmp_path: Path) -> None:
    write_valid_table(tmp_path, "patient")

    tables = load_tables(tmp_path, tables=["patient"])

    assert list(tables) == ["patient"]


def test_load_tables_rejects_missing_table_file(tmp_path: Path) -> None:
    write_valid_table(tmp_path, "patient")
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
    write_valid_table(tmp_path, "treatment", patientunitstayid=[], treatmentstring=[], treatmentoffset=[])

    tables = load_tables(tmp_path, tables=["treatment"])

    assert tables["treatment"].empty
    assert set(tables["treatment"].columns) == {"patientunitstayid", "treatmentstring", "treatmentoffset"}


def test_load_tables_preserves_extra_columns_and_duplicate_stays(tmp_path: Path) -> None:
    write_valid_table(
        tmp_path,
        "diagnosis",
        patientunitstayid=[101, 101],
        diagnosisstring=["sepsis", "respiratory failure"],
        diagnosisoffset=[100, 200],
        extra_col=["foo", "bar"],
    )

    tables = load_tables(tmp_path, tables=["diagnosis"])

    assert tables["diagnosis"]["patientunitstayid"].tolist() == [101, 101]
    assert tables["diagnosis"]["diagnosisstring"].tolist() == [
        "sepsis",
        "respiratory failure",
    ]
    assert tables["diagnosis"]["extra_col"].tolist() == ["foo", "bar"]


def test_load_tables_rejects_incorrect_table_capitalization(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown eICU demo table.*InfusionDrug"):
        load_tables(tmp_path, tables=["InfusionDrug"])


def test_load_tables_does_not_require_apachePatientResult(tmp_path: Path) -> None:
    for table_name in MVP_TABLES:
        if table_name != "apachePatientResult":
            write_valid_table(tmp_path, table_name)
    tables = load_tables(tmp_path)
    assert "apachePatientResult" not in tables
    assert len(tables) == len(MVP_TABLES) - 1


def test_discover_raw_dir_exact(tmp_path: Path) -> None:
    write_valid_table(tmp_path, "patient")
    assert discover_raw_dir(tmp_path) == tmp_path


def test_discover_raw_dir_nested(tmp_path: Path) -> None:
    nested = tmp_path / "subdir" / "physionet.org" / "files" / "eicu-crd-demo" / "2.0.1"
    nested.mkdir(parents=True)
    write_valid_table(nested, "patient")
    assert discover_raw_dir(tmp_path) == nested


def test_discover_raw_dir_ambiguous(tmp_path: Path) -> None:
    nested1 = tmp_path / "dir1" / "eicu-crd-demo" / "2.0.1"
    nested2 = tmp_path / "dir2" / "eicu-crd-demo" / "2.0.1"
    nested1.mkdir(parents=True)
    nested2.mkdir(parents=True)
    with pytest.raises(ValueError, match="ambiguous raw directory"):
        discover_raw_dir(tmp_path)


def test_discover_raw_dir_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="could not discover"):
        discover_raw_dir(tmp_path)


def test_load_table_chunks_yields_correct_chunks(tmp_path: Path) -> None:
    data = {
        "patientunitstayid": [101, 102, 103],
        "diagnosisstring": ["sepsis", "shock", "failure"],
        "diagnosisoffset": [100, 200, 300],
    }
    pd.DataFrame(data).to_csv(table_path(tmp_path, "diagnosis"), index=False)
    chunks = list(load_table_chunks(tmp_path, "diagnosis", chunksize=2))
    assert len(chunks) == 2
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 1
    assert chunks[0]["patientunitstayid"].tolist() == [101, 102]
    assert chunks[1]["patientunitstayid"].tolist() == [103]


def test_load_table_chunks_only_requested_columns(tmp_path: Path) -> None:
    write_valid_table(tmp_path, "diagnosis")
    chunks = list(load_table_chunks(tmp_path, "diagnosis", chunksize=10, columns=["patientunitstayid"]))
    assert list(chunks[0].columns) == ["patientunitstayid"]
