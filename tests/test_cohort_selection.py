from __future__ import annotations

import math
import pandas as pd
import pytest

from icu_pretrain.data.eicu_event_builder import extract_outcomes


def test_extract_outcomes_maps_alive_and_expired() -> None:
    tables = {
        "patient": pd.DataFrame({
            "patientunitstayid": [1, 2, 3],
            "uniquepid": ["P1", "P2", "P3"],
            "hospitalid": [10, 20, 30],
            "hospitaldischargestatus": ["Alive", "EXPIRED", "alive"],
            "unitdischargeoffset": [1440, 1500, 2000],
        })
    }
    outcomes = extract_outcomes(tables)
    assert outcomes.to_dict("records") == [
        {"patientunitstayid": 1, "mortality": 0},
        {"patientunitstayid": 2, "mortality": 1},
        {"patientunitstayid": 3, "mortality": 0},
    ]
    summary = outcomes.attrs["cohort_summary"]
    assert summary.total_stays == 3
    assert summary.eligible_stays == 3
    assert summary.class_counts == {"Alive": 2, "Expired": 1}
    assert summary.exclusion_counts == {
        "missing_id": 0,
        "unrecognised_label": 0,
        "short_stay": 0,
        "conflicting_stay": 0,
    }


def test_extract_outcomes_excludes_unrecognised_labels() -> None:
    tables = {
        "patient": pd.DataFrame({
            "patientunitstayid": [1, 2, 3],
            "uniquepid": ["P1", "P2", "P3"],
            "hospitalid": [10, 20, 30],
            "hospitaldischargestatus": ["Alive", "Unknown", None],
            "unitdischargeoffset": [1440, 1440, 1440],
        })
    }
    outcomes = extract_outcomes(tables)
    assert outcomes.to_dict("records") == [
        {"patientunitstayid": 1, "mortality": 0},
    ]
    summary = outcomes.attrs["cohort_summary"]
    assert summary.total_stays == 3
    assert summary.eligible_stays == 1
    assert summary.exclusion_counts["unrecognised_label"] == 2


def test_extract_outcomes_excludes_missing_ids() -> None:
    tables = {
        "patient": pd.DataFrame({
            "patientunitstayid": [1, 2, 3, None],
            "uniquepid": ["P1", None, "P3", "P4"],
            "hospitalid": [10, 20, None, 40],
            "hospitaldischargestatus": ["Alive", "Alive", "Alive", "Alive"],
            "unitdischargeoffset": [1440, 1440, 1440, 1440],
        })
    }
    outcomes = extract_outcomes(tables)
    assert outcomes.to_dict("records") == [
        {"patientunitstayid": 1, "mortality": 0},
    ]
    summary = outcomes.attrs["cohort_summary"]
    assert summary.total_stays == 4
    assert summary.eligible_stays == 1
    assert summary.exclusion_counts["missing_id"] == 3


def test_extract_outcomes_excludes_short_stays() -> None:
    tables = {
        "patient": pd.DataFrame({
            "patientunitstayid": [1, 2, 3, 4],
            "uniquepid": ["P1", "P2", "P3", "P4"],
            "hospitalid": [10, 20, 30, 40],
            "hospitaldischargestatus": ["Alive", "Alive", "Alive", "Alive"],
            "unitdischargeoffset": [1440, 1439, None, "not a number"],
        })
    }
    outcomes = extract_outcomes(tables)
    assert outcomes.to_dict("records") == [
        {"patientunitstayid": 1, "mortality": 0},
    ]
    summary = outcomes.attrs["cohort_summary"]
    assert summary.total_stays == 4
    assert summary.eligible_stays == 1
    assert summary.exclusion_counts["short_stay"] == 3


def test_extract_outcomes_rejects_conflicting_rows() -> None:
    tables = {
        "patient": pd.DataFrame({
            "patientunitstayid": [1, 1, 2, 2],
            "uniquepid": ["P1", "P1", "P2", "P2_diff"],
            "hospitalid": [10, 10, 20, 20],
            "hospitaldischargestatus": ["Alive", "Expired", "Alive", "Alive"],
            "unitdischargeoffset": [1440, 1440, 1440, 1440],
        })
    }
    outcomes = extract_outcomes(tables)
    assert outcomes.empty
    summary = outcomes.attrs["cohort_summary"]
    assert summary.total_stays == 2
    assert summary.eligible_stays == 0
    assert summary.exclusion_counts["conflicting_stay"] == 2


def test_extract_outcomes_accepts_duplicate_identical_rows() -> None:
    tables = {
        "patient": pd.DataFrame({
            "patientunitstayid": [1, 1],
            "uniquepid": ["P1", "P1"],
            "hospitalid": [10, 10],
            "hospitaldischargestatus": ["Alive", "Alive"],
            "unitdischargeoffset": [1440, 1440],
        })
    }
    outcomes = extract_outcomes(tables)
    assert outcomes.to_dict("records") == [
        {"patientunitstayid": 1, "mortality": 0},
    ]
    summary = outcomes.attrs["cohort_summary"]
    assert summary.total_stays == 1
    assert summary.eligible_stays == 1
    assert summary.exclusion_counts["conflicting_stay"] == 0


def test_extract_outcomes_no_fallback() -> None:
    tables = {
        "patient": pd.DataFrame({
            "patientunitstayid": [1],
            "uniquepid": ["P1"],
            "hospitalid": [10],
            "hospitaldischargestatus": [None],
            "unitdischargestatus": ["Alive"],
            "unitdischargeoffset": [1440],
        }),
        "apachePatientResult": pd.DataFrame({
            "patientunitstayid": [1],
            "actualhospitalmortality": ["Alive"],
        }),
    }
    outcomes = extract_outcomes(tables)
    assert outcomes.empty
    summary = outcomes.attrs["cohort_summary"]
    assert summary.eligible_stays == 0
    assert summary.exclusion_counts["unrecognised_label"] == 1
