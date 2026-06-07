"""Tests for deterministic lab and vital numeric discretization."""

from __future__ import annotations

import pandas as pd

from icu_pretrain.data.eicu_event_builder import (
    apply_numeric_bin,
    extract_numeric_events,
    fit_numeric_bins,
)


def test_fits_and_applies_lab_and_vital_quantile_bins() -> None:
    tables = {
        "lab": pd.DataFrame(
            {
                "patientunitstayid": [101, 102, 103, 104],
                "labname": ["Creatinine"] * 4,
                "labresult": [1.0, 2.0, 3.0, 4.0],
                "labresultoffset": [40, 30, 20, 10],
            }
        ),
        "vitalPeriodic": pd.DataFrame(
            {
                "patientunitstayid": [101, 102, 103, 104],
                "observationoffset": [5, 5, 5, 5],
                "heartrate": [60, 70, 80, 90],
            }
        ),
    }

    thresholds = fit_numeric_bins(tables)
    events, stats = extract_numeric_events(tables, thresholds)

    assert events[101] == [
        ("VITAL::HEARTRATE::Q1", 5),
        ("LAB::CREATININE::Q1", 40),
    ]
    assert events[104] == [
        ("VITAL::HEARTRATE::Q4", 5),
        ("LAB::CREATININE::Q4", 10),
    ]
    assert stats["candidate_values"] == 8
    assert stats["emitted_events"] == 8


def test_numeric_strings_negative_values_and_outside_ranges_are_supported() -> None:
    tables = {
        "lab": pd.DataFrame(
            {
                "patientunitstayid": [101, 102, 103, 104],
                "labname": ["Base excess"] * 4,
                "labresult": ["-4", "-2", "0", "5.2"],
            }
        )
    }

    thresholds = fit_numeric_bins(tables)

    assert apply_numeric_bin("-100", thresholds["LAB::BASE_EXCESS"]) == "Q1"
    assert apply_numeric_bin("100", thresholds["LAB::BASE_EXCESS"]) == "Q4"


def test_small_and_all_equal_samples_do_not_crash() -> None:
    tables = {
        "vitalAperiodic": pd.DataFrame(
            {
                "patientunitstayid": [101, 102],
                "observationoffset": [None, None],
                "noninvasivemean": [75, 75],
            }
        )
    }

    thresholds = fit_numeric_bins(tables)
    events, _ = extract_numeric_events(tables, thresholds)

    assert thresholds["VITAL::NONINVASIVEMEAN"] == (75.0, 75.0, 75.0)
    assert events == {
        101: [("VITAL::NONINVASIVEMEAN::Q1", None)],
        102: [("VITAL::NONINVASIVEMEAN::Q1", None)],
    }


def test_invalid_values_missing_names_and_unfitted_measurements_are_counted() -> None:
    tables = {
        "lab": pd.DataFrame(
            {
                "patientunitstayid": [101, 101, 101, 101],
                "labname": ["Creatinine", "Creatinine", "Creatinine", None],
                "labresult": ["5.2", "invalid", float("inf"), 7.0],
                "labresultoffset": [10, 20, 25, 30],
            }
        ),
        "vitalPeriodic": pd.DataFrame(
            {
                "patientunitstayid": [101],
                "observationoffset": [40],
                "heartrate": [80],
            }
        ),
    }
    thresholds = {"LAB::CREATININE": (4.0, 5.0, 6.0)}

    events, stats = extract_numeric_events(tables, thresholds)

    assert events == {101: [("LAB::CREATININE::Q3", 10)]}
    assert stats == {
        "candidate_values": 5,
        "emitted_events": 1,
        "skipped_missing_measurement": 1,
        "skipped_nonnumeric_value": 2,
        "skipped_unfitted_measurement": 1,
    }


def test_vital_metadata_columns_are_not_treated_as_measurements() -> None:
    tables = {
        "vitalPeriodic": pd.DataFrame(
            {
                "patientunitstayid": [101],
                "vitalperiodicid": [999],
                "observationoffset": [15],
                "heartrate": [80],
            }
        )
    }

    thresholds = fit_numeric_bins(tables)

    assert thresholds == {"VITAL::HEARTRATE": (80.0, 80.0, 80.0)}
