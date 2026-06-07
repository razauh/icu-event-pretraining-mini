"""Synthetic fixtures for data-safe tests.

All identifiers and values in this file are invented for unit tests. Do not
replace them with raw or processed ICU records.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest


@pytest.fixture
def synthetic_patient_table() -> pd.DataFrame:
    """Return a small invented patient table with split and grouping columns."""
    return pd.DataFrame(
        {
            "patientunitstayid": [1001, 1002, 1003, 1004],
            "patienthealthsystemstayid": [2001, 2002, 2003, 2004],
            "uniquepid": [
                "syn-patient-a",
                "syn-patient-b",
                "syn-patient-c",
                "syn-patient-d",
            ],
            "hospitalid": [10, 10, 20, 20],
            "wardid": [101, 102, 201, 202],
            "unitadmitoffset": [0, 0, 0, 0],
            "unitdischargeoffset": [1800, 2100, 2400, 3000],
            "hospitaldischargestatus": ["Alive", "Expired", "Alive", "Alive"],
            "age": [43, 67, 58, 74],
            "gender": ["Female", "Male", "Other", "Unknown"],
        }
    )


@pytest.fixture
def synthetic_event_tables(
    synthetic_patient_table: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return invented first-24-hour ICU event source tables."""
    stay_ids = synthetic_patient_table["patientunitstayid"].tolist()
    return {
        "patient": synthetic_patient_table.copy(),
        "diagnosis": pd.DataFrame(
            {
                "patientunitstayid": stay_ids,
                "diagnosisoffset": [15, 45, 90, 120],
                "diagnosisstring": [
                    "Synthetic respiratory concern",
                    "Synthetic metabolic concern",
                    "Synthetic infection concern",
                    "Synthetic monitoring concern",
                ],
            }
        ),
        "lab": pd.DataFrame(
            {
                "patientunitstayid": [
                    stay_ids[0],
                    stay_ids[1],
                    stay_ids[2],
                    stay_ids[3],
                ],
                "labresultoffset": [30, 60, 90, 120],
                "labname": ["Sodium", "Creatinine", "Lactate", "Hemoglobin"],
                "labresult": [140.0, 1.2, 2.4, 11.8],
            }
        ),
        "medication": pd.DataFrame(
            {
                "patientunitstayid": [stay_ids[0], stay_ids[1]],
                "drugstartoffset": [75, 135],
                "drugname": ["Synthetic antibiotic", "Synthetic sedative"],
            }
        ),
        "infusionDrug": pd.DataFrame(
            {
                "patientunitstayid": [stay_ids[2]],
                "infusionoffset": [180],
                "drugname": ["Synthetic infusion"],
            }
        ),
        "treatment": pd.DataFrame(
            {
                "patientunitstayid": [stay_ids[3]],
                "treatmentoffset": [240],
                "treatmentstring": ["Synthetic respiratory support"],
            }
        ),
        "vitalPeriodic": pd.DataFrame(
            {
                "patientunitstayid": stay_ids,
                "observationoffset": [0, 60, 120, 180],
                "heartrate": [72, 88, 95, 81],
                "systemicsystolic": [118, 126, 104, 132],
            }
        ),
        "vitalAperiodic": pd.DataFrame(
            {
                "patientunitstayid": [stay_ids[0], stay_ids[2]],
                "observationoffset": [25, 155],
                "noninvasivemean": [76, 69],
            }
        ),
    }


@pytest.fixture
def make_synthetic_event_tables(
    synthetic_event_tables: dict[str, pd.DataFrame],
) -> Callable[[], dict[str, pd.DataFrame]]:
    """Return a factory that gives each test isolated synthetic tables."""

    def factory() -> dict[str, pd.DataFrame]:
        return {name: frame.copy() for name, frame in synthetic_event_tables.items()}

    return factory
