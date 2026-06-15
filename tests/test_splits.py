from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from icu_pretrain.data.eicu_event_builder import write_split_metadata
from icu_pretrain.data.splits import assign_patient_splits


def test_assign_patient_splits_groups_by_uniquepid() -> None:
    outcomes = pd.DataFrame({
        "patientunitstayid": [1, 2, 3, 4, 5, 6, 7],
        "mortality": [0, 0, 0, 0, 1, 1, 1],
    })
    patient_df = pd.DataFrame({
        "patientunitstayid": [1, 2, 3, 4, 5, 6, 7],
        "uniquepid": ["P1", "P1", "P2", "P3", "P4", "P5", "P6"],
        "hospitalid": [1, 1, 1, 1, 1, 1, 1],
    })

    records = assign_patient_splits(outcomes, patient_df, seed=42)

    p1_splits = {r.split_name for r in records if r.uniquepid == "P1"}
    assert len(p1_splits) == 1

    stay_ids = [r.patientunitstayid for r in records]
    assert sorted(stay_ids) == [1, 2, 3, 4, 5, 6, 7]


def test_assign_patient_splits_requires_both_outcome_classes_per_split() -> None:
    outcomes = pd.DataFrame({
        "patientunitstayid": [1, 2, 3, 4, 5],
        "mortality": [0, 0, 1, 1, 1],
    })
    patient_df = pd.DataFrame({
        "patientunitstayid": [1, 2, 3, 4, 5],
        "uniquepid": ["P1", "P2", "P3", "P4", "P5"],
        "hospitalid": [1, 1, 1, 1, 1],
    })

    with pytest.raises(ValueError, match="cohort size is insufficient"):
        assign_patient_splits(outcomes, patient_df, seed=42)


def test_assign_patient_splits_is_reproducible() -> None:
    outcomes = pd.DataFrame({
        "patientunitstayid": [1, 2, 3, 4, 5, 6],
        "mortality": [0, 0, 0, 1, 1, 1],
    })
    patient_df = pd.DataFrame({
        "patientunitstayid": [1, 2, 3, 4, 5, 6],
        "uniquepid": ["P1", "P2", "P3", "P4", "P5", "P6"],
        "hospitalid": [1, 1, 1, 1, 1, 1],
    })

    records_1 = assign_patient_splits(outcomes, patient_df, seed=42)
    records_2 = assign_patient_splits(outcomes, patient_df, seed=42)

    assert [r.split_name for r in records_1] == [r.split_name for r in records_2]


def test_assign_patient_splits_persists_under_processed_root(tmp_path: Path) -> None:
    outcomes = pd.DataFrame({
        "patientunitstayid": [1, 2, 3, 4, 5, 6],
        "mortality": [0, 0, 0, 1, 1, 1],
    })
    patient_df = pd.DataFrame({
        "patientunitstayid": [1, 2, 3, 4, 5, 6],
        "uniquepid": ["P1", "P2", "P3", "P4", "P5", "P6"],
        "hospitalid": [1, 1, 1, 1, 1, 1],
    })

    records = assign_patient_splits(outcomes, patient_df, seed=42)

    processed_root = tmp_path / "processed"
    processed_root.mkdir()

    valid_path = processed_root / "splits.json"
    write_split_metadata(valid_path, records, processed_root=processed_root)
    assert valid_path.exists()

    invalid_path = tmp_path / "splits.json"
    with pytest.raises(ValueError, match="split metadata must be written under processed_root"):
        write_split_metadata(invalid_path, records, processed_root=processed_root)
