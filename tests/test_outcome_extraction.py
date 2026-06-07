"""Tests for conservative mortality outcome extraction."""

from __future__ import annotations

import pandas as pd
import pytest

from icu_pretrain.data.eicu_event_builder import extract_outcomes


def test_extract_outcomes_uses_hospital_discharge_status() -> None:
    tables = {
        "patient": pd.DataFrame(
            {
                "patientunitstayid": [101, 102],
                "hospitaldischargestatus": ["Alive", "EXPIRED"],
            }
        )
    }

    outcomes = extract_outcomes(tables)

    assert outcomes.to_dict("records") == [
        {"patientunitstayid": 101, "mortality": 0},
        {"patientunitstayid": 102, "mortality": 1},
    ]
    assert outcomes["mortality"].dtype.kind in {"i", "u"}


def test_extract_outcomes_falls_back_to_apache_hospital_mortality() -> None:
    tables = {
        "patient": pd.DataFrame({"patientunitstayid": [101, 102]}),
        "apachePatientResult": pd.DataFrame(
            {
                "patientunitstayid": [101, 102],
                "actualhospitalmortality": [0, 1],
            }
        ),
    }

    outcomes = extract_outcomes(tables)

    assert outcomes["mortality"].tolist() == [0, 1]


def test_extract_outcomes_prefers_hospital_label_over_icu_fallback() -> None:
    tables = {
        "patient": pd.DataFrame(
            {
                "patientunitstayid": [101],
                "hospitaldischargestatus": ["Alive"],
                "unitdischargestatus": ["Expired"],
            }
        )
    }

    outcomes = extract_outcomes(tables)

    assert outcomes.to_dict("records") == [
        {"patientunitstayid": 101, "mortality": 0}
    ]


def test_extract_outcomes_uses_unambiguous_icu_status_as_last_resort() -> None:
    tables = {
        "patient": pd.DataFrame(
            {
                "patientunitstayid": [101, 102],
                "unitdischargestatus": ["alive", "dead"],
            }
        )
    }

    outcomes = extract_outcomes(tables)

    assert outcomes["mortality"].tolist() == [0, 1]


def test_extract_outcomes_drops_missing_and_ambiguous_labels() -> None:
    tables = {
        "patient": pd.DataFrame(
            {
                "patientunitstayid": [101, 102, 103],
                "hospitaldischargestatus": [None, "Unknown", 2],
            }
        )
    }

    outcomes = extract_outcomes(tables)

    assert outcomes.empty
    assert list(outcomes.columns) == ["patientunitstayid", "mortality"]
    assert outcomes.attrs["outcome_stats"] == {
        "candidate_stays": 3,
        "labelled_stays": 0,
        "unavailable_labels": 3,
        "conflicting_labels": 0,
    }


def test_extract_outcomes_drops_conflicting_duplicate_rows() -> None:
    tables = {
        "patient": pd.DataFrame(
            {
                "patientunitstayid": [101, 101, 102, 102],
                "hospitaldischargestatus": ["Alive", "Expired", "Alive", "alive"],
            }
        )
    }

    outcomes = extract_outcomes(tables)

    assert outcomes.to_dict("records") == [
        {"patientunitstayid": 102, "mortality": 0}
    ]
    assert outcomes.attrs["outcome_stats"]["conflicting_labels"] == 1


def test_extract_outcomes_uses_another_table_when_primary_label_is_missing() -> None:
    tables = {
        "patient": pd.DataFrame(
            {
                "patientunitstayid": [101],
                "hospitaldischargestatus": [None],
            }
        ),
        "apachePatientResult": pd.DataFrame(
            {
                "patientunitstayid": [101],
                "actualhospitalmortality": ["expired"],
            }
        ),
    }

    outcomes = extract_outcomes(tables)

    assert outcomes.to_dict("records") == [
        {"patientunitstayid": 101, "mortality": 1}
    ]


def test_extract_outcomes_requires_a_supported_label_field() -> None:
    tables = {"patient": pd.DataFrame({"patientunitstayid": [101]})}

    with pytest.raises(ValueError, match="mortality or discharge-status label field"):
        extract_outcomes(tables)


def test_extract_outcomes_requires_patientunitstayid() -> None:
    tables = {
        "patient": pd.DataFrame({"hospitaldischargestatus": ["Alive"]})
    }

    with pytest.raises(ValueError, match="patient.*patientunitstayid"):
        extract_outcomes(tables)
