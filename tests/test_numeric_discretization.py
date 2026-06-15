from __future__ import annotations

import pandas as pd
import pytest

from icu_pretrain.data.eicu_event_builder import (
    apply_numeric_bin,
    extract_numeric_events,
    fit_numeric_bins,
)


def test_fits_and_applies_lab_and_vital_quantile_bins() -> None:
    stay_ids = list(range(100, 150))
    tables = {
        "lab": pd.DataFrame(
            {
                "patientunitstayid": stay_ids,
                "labname": ["Creatinine"] * 50,
                "labresult": [float(i) for i in range(1, 51)],
                "labresultoffset": [10] * 50,
            }
        ),
        "vitalPeriodic": pd.DataFrame(
            {
                "patientunitstayid": stay_ids,
                "observationoffset": [5] * 50,
                "heartrate": [float(i) for i in range(60, 110)],
            }
        ),
    }

    thresholds = fit_numeric_bins(tables)
    events, stats = extract_numeric_events(tables, thresholds)

    assert len(thresholds["LAB::CREATININE"]) == 3
    assert len(thresholds["VITAL::HEARTRATE"]) == 3
    assert events[100] == [
        ("LAB::CREATININE::Q1", 10),
        ("VITAL::HEARTRATE::Q1", 60),
    ]
    assert events[149] == [
        ("LAB::CREATININE::Q4", 10),
        ("VITAL::HEARTRATE::Q4", 60),
    ]
    assert stats["candidate_values"] == 100
    assert stats["emitted_events"] == 100


def test_numeric_strings_negative_values_and_outside_ranges_are_supported() -> None:
    stay_ids = list(range(100, 150))
    results = [str(i - 25) for i in range(50)]
    tables = {
        "lab": pd.DataFrame(
            {
                "patientunitstayid": stay_ids,
                "labname": ["Base excess"] * 50,
                "labresult": results,
                "labresultoffset": [10] * 50,
            }
        )
    }

    thresholds = fit_numeric_bins(tables)

    assert apply_numeric_bin("-100", thresholds["LAB::BASE_EXCESS"]) == "Q1"
    assert apply_numeric_bin("100", thresholds["LAB::BASE_EXCESS"]) == "Q4"


def test_small_and_all_equal_samples_do_not_crash() -> None:
    stay_ids = list(range(100, 150))
    tables = {
        "vitalAperiodic": pd.DataFrame(
            {
                "patientunitstayid": stay_ids,
                "observationoffset": [0] * 50,
                "noninvasivemean": [75.0] * 50,
            }
        )
    }

    thresholds = fit_numeric_bins(tables)
    events, _ = extract_numeric_events(tables, thresholds)

    assert thresholds["VITAL::NONINVASIVEMEAN"] == (75.0, 75.0, 75.0)
    assert events[100] == [("VITAL::NONINVASIVEMEAN::Q1", 60)]


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
    stay_ids = list(range(100, 150))
    tables = {
        "vitalPeriodic": pd.DataFrame(
            {
                "patientunitstayid": stay_ids,
                "vitalperiodicid": list(range(50)),
                "observationoffset": [15] * 50,
                "heartrate": [80.0] * 50,
            }
        )
    }

    thresholds = fit_numeric_bins(tables)

    assert list(thresholds.keys()) == ["VITAL::HEARTRATE"]


def test_train_validation_isolation() -> None:
    train_stays = set(range(100, 150))
    val_stays = set(range(150, 160))
    stay_ids = list(train_stays) + list(val_stays)

    tables = {
        "lab": pd.DataFrame(
            {
                "patientunitstayid": stay_ids,
                "labname": ["Creatinine"] * 60,
                "labresult": [float(i) for i in range(1, 61)],
                "labresultoffset": [10] * 60,
            }
        )
    }

    thresholds_1 = fit_numeric_bins(tables, train_stay_ids=train_stays)

    tables["lab"].loc[tables["lab"]["patientunitstayid"].isin(val_stays), "labresult"] = 999.0

    thresholds_2 = fit_numeric_bins(tables, train_stay_ids=train_stays)

    assert thresholds_1 == thresholds_2


def test_vital_aggregation_by_median_and_bucket_mapping() -> None:
    tables = {
        "vitalPeriodic": pd.DataFrame(
            {
                "patientunitstayid": [101, 101, 101],
                "observationoffset": [15, 30, 45],
                "heartrate": [60.0, 90.0, 70.0],
            }
        )
    }

    thresholds = {"VITAL::HEARTRATE": (50.0, 75.0, 85.0)}
    events, _ = extract_numeric_events(tables, thresholds)

    assert events[101] == [("VITAL::HEARTRATE::Q2", 60)]


def test_lab_deduplication_keeps_last_source_order() -> None:
    tables = {
        "lab": pd.DataFrame(
            {
                "patientunitstayid": [101, 101],
                "labname": ["Creatinine", "Creatinine"],
                "labresult": [2.0, 5.5],
                "labresultoffset": [10, 10],
            }
        )
    }

    thresholds = {"LAB::CREATININE": (1.0, 3.0, 5.0)}
    events, _ = extract_numeric_events(tables, thresholds)

    assert events[101] == [("LAB::CREATININE::Q4", 10)]
