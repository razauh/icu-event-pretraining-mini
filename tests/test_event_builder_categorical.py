from __future__ import annotations

import pandas as pd

from icu_pretrain.data.eicu_event_builder import extract_categorical_events


def test_extracts_static_and_clinical_categorical_events() -> None:
    tables = {
        "patient": pd.DataFrame(
            {
                "patientunitstayid": [101],
                "age": [65],
                "gender": ["Female"],
                "unitadmitsource": [" Emergency Department "],
                "unittype": ["Med-Surg ICU"],
            }
        ),
        "diagnosis": pd.DataFrame(
            {
                "patientunitstayid": [101],
                "diagnosisoffset": [20],
                "diagnosisstring": [" Respiratory   failure "],
            }
        ),
        "medication": pd.DataFrame(
            {
                "patientunitstayid": [101],
                "drugstartoffset": [10],
                "drugname": ["Norepinephrine 4 mg/250 mL"],
            }
        ),
        "infusionDrug": pd.DataFrame(
            {
                "patientunitstayid": [101],
                "infusionoffset": [30],
                "drugname": ["Propofol (mg/hr)"],
            }
        ),
        "treatment": pd.DataFrame(
            {
                "patientunitstayid": [101],
                "treatmentoffset": [40],
                "treatmentstring": [
                    "pulmonary|ventilation and oxygenation|mechanical ventilation"
                ],
            }
        ),
    }

    events = extract_categorical_events(tables, representation="timegap_static")

    assert events[101] == [
        ("STATIC::AGE_BIN::60_79", None),
        ("STATIC::GENDER::F", None),
        ("STATIC::UNIT_ADMIT_SOURCE::EMERGENCY_DEPARTMENT", None),
        ("STATIC::UNIT_TYPE::MED_SURG_ICU", None),
        ("MED::NOREPINEPHRINE_4_MG_250_ML", 10),
        ("DX::RESPIRATORY_FAILURE", 20),
        ("INFUSION::PROPOFOL_MG_HR", 30),
        (
            "TREATMENT::PULMONARY_VENTILATION_AND_OXYGENATION_MECHANICAL_VENTILATION",
            40,
        ),
    ]


def test_basic_and_timegap_exclude_static_and_time_gap_tokens() -> None:
    tables = {
        "patient": pd.DataFrame(
            {"patientunitstayid": [101], "age": [55], "gender": ["Male"]}
        ),
        "diagnosis": pd.DataFrame(
            {
                "patientunitstayid": [101],
                "diagnosisoffset": [5],
                "diagnosisstring": ["Sepsis"],
            }
        ),
    }

    assert extract_categorical_events(tables, representation="basic") == {
        101: [("DX::SEPSIS", 5)]
    }
    assert extract_categorical_events(tables, representation="timegap") == {
        101: [("DX::SEPSIS", 5)]
    }


def test_age_masking_and_unknown_static_values_are_deterministic() -> None:
    patient = pd.DataFrame(
        {
            "patientunitstayid": [101, 102],
            "age": ["> 89", "unknown"],
            "gender": ["Other", None],
            "unitadmitsource": ["Unknown", None],
        }
    )

    events = extract_categorical_events(
        {"patient": patient}, representation="timegap_static"
    )

    assert events[101] == [
        ("STATIC::AGE_BIN::80_PLUS", None),
        ("STATIC::GENDER::UNKNOWN", None),
        ("STATIC::UNIT_ADMIT_SOURCE::UNKNOWN", None),
        ("STATIC::UNIT_TYPE::UNKNOWN", None),
    ]
    assert events[102] == [
        ("STATIC::AGE_BIN::UNKNOWN", None),
        ("STATIC::GENDER::UNKNOWN", None),
        ("STATIC::UNIT_ADMIT_SOURCE::UNKNOWN", None),
        ("STATIC::UNIT_TYPE::UNKNOWN", None),
    ]


def test_nulls_duplicates_equal_offsets_and_missing_offsets_are_handled() -> None:
    tables = {
        "diagnosis": pd.DataFrame(
            {
                "patientunitstayid": [101, 101, 101, 101],
                "diagnosisoffset": [10, 10, 10, None],
                "diagnosisstring": ["Sepsis", "Sepsis", "Acidosis", None],
            }
        ),
        "medication": pd.DataFrame(
            {
                "patientunitstayid": [101, 101],
                "drugstartoffset": [None, 20],
                "drugname": ["Vancomycin", "  Aspirin 81 mg  "],
            }
        ),
    }

    events = extract_categorical_events(tables, representation="basic")

    assert events[101] == [
        ("DX::ACIDOSIS", 10),
        ("DX::SEPSIS", 10),
        ("MED::ASPIRIN_81_MG", 20),
    ]


def test_missing_optional_tables_and_columns_do_not_crash() -> None:
    tables = {
        "patient": pd.DataFrame({"patientunitstayid": [101], "age": [45]}),
        "medication": pd.DataFrame(
            {"patientunitstayid": [101], "unrelated_column": ["ignored"]}
        ),
    }

    assert extract_categorical_events(
        tables, representation="timegap_static"
    ) == {
        101: [
            ("STATIC::AGE_BIN::40_59", None),
            ("STATIC::GENDER::UNKNOWN", None),
            ("STATIC::UNIT_ADMIT_SOURCE::UNKNOWN", None),
            ("STATIC::UNIT_TYPE::UNKNOWN", None),
        ]
    }


def test_medication_hicl_fallback() -> None:
    tables = {
        "medication": pd.DataFrame(
            {
                "patientunitstayid": [101, 101],
                "drugstartoffset": [15, 30],
                "drugname": [None, "unknown"],
                "drughiclseqno": [1234, 5678],
            }
        )
    }
    events = extract_categorical_events(tables, representation="basic")
    assert events[101] == [
        ("MED::HICL::1234", 15),
        ("MED::HICL::5678", 30),
    ]


def test_leakage_prohibited_tokens() -> None:
    tables = {
        "diagnosis": pd.DataFrame(
            {
                "patientunitstayid": [101],
                "diagnosisoffset": [5],
                "diagnosisstring": ["Sepsis with patientunitstayid in it"],
            }
        ),
        "treatment": pd.DataFrame(
            {
                "patientunitstayid": [101],
                "treatmentoffset": [10],
                "treatmentstring": ["ActiveUponDischarge"],
            }
        ),
    }
    events = extract_categorical_events(tables, representation="basic")
    assert 101 not in events
