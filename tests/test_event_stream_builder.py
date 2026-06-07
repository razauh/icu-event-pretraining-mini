"""Tests for complete patient-level ICU event stream construction."""

from __future__ import annotations

import pandas as pd
import pytest

from icu_pretrain.constants import EVENT_FAMILIES
from icu_pretrain.data.eicu_event_builder import build_event_streams


def _tables() -> dict[str, pd.DataFrame]:
    return {
        "patient": pd.DataFrame(
            {
                "patientunitstayid": [101, 102, 103],
                "age": [65, 50, 40],
                "gender": ["Female", "Male", "Female"],
                "hospitaldischargestatus": ["Alive", "Expired", None],
            }
        ),
        "diagnosis": pd.DataFrame(
            {
                "patientunitstayid": [101, 101, 102, 103],
                "diagnosisoffset": [60, 60, None, 10],
                "diagnosisstring": ["Sepsis", "Acidosis", "Shock", "Asthma"],
            }
        ),
        "medication": pd.DataFrame(
            {
                "patientunitstayid": [101],
                "drugstartoffset": [0],
                "drugname": ["Norepinephrine"],
            }
        ),
        "lab": pd.DataFrame(
            {
                "patientunitstayid": [101, 102],
                "labname": ["Creatinine", "Creatinine"],
                "labresult": [1.0, 3.0],
                "labresultoffset": [240, 30],
            }
        ),
    }


def test_builds_chronological_timegap_static_streams_and_filters_outcomes() -> None:
    streams, outcomes, stats = build_event_streams(
        _tables(), representation="timegap_static", min_events_per_stay=2
    )

    assert [stream.patientunitstayid for stream in streams] == [101, 102]
    assert streams[0].events == [
        "STATIC::AGE_BIN::60_70",
        "STATIC::GENDER::F",
        "MED::NOREPINEPHRINE",
        "TIME_GAP::15M_1H",
        "DX::ACIDOSIS",
        "DX::SEPSIS",
        "TIME_GAP::1H_3H",
        "LAB::CREATININE::Q1",
    ]
    assert streams[0].event_times == [None, None, 0, 60, 60, 60, 240, 240]
    assert streams[1].events[-1] == "DX::SHOCK"
    assert streams[1].event_times[-1] is None
    assert outcomes.to_dict("records") == [
        {"patientunitstayid": 101, "mortality": 0},
        {"patientunitstayid": 102, "mortality": 1},
    ]
    assert stats.total_stays == 3
    assert stats.kept_stays == 2
    assert stats.skipped_stays == 1
    assert stats.min_sequence_length == 4
    assert stats.max_sequence_length == 8
    assert stats.median_sequence_length == 6.0
    assert set(stats.token_family_counts) == set(EVENT_FAMILIES)
    assert stats.token_family_counts["TIME_GAP"] == 2


def test_representation_controls_static_and_time_gap_tokens() -> None:
    basic, _, _ = build_event_streams(
        _tables(), representation="basic", min_events_per_stay=1
    )
    timegap, _, _ = build_event_streams(
        _tables(), representation="timegap", min_events_per_stay=1
    )

    assert all(
        not token.startswith(("STATIC::", "TIME_GAP::"))
        for stream in basic
        for token in stream.events
    )
    assert all(
        not token.startswith("STATIC::")
        for stream in timegap
        for token in stream.events
    )
    assert any(
        token.startswith("TIME_GAP::")
        for stream in timegap
        for token in stream.events
    )


def test_equal_negative_and_missing_offsets_do_not_create_artificial_gaps() -> None:
    tables = {
        "patient": pd.DataFrame(
            {
                "patientunitstayid": [101],
                "hospitaldischargestatus": ["Alive"],
            }
        ),
        "diagnosis": pd.DataFrame(
            {
                "patientunitstayid": [101, 101, 101],
                "diagnosisoffset": [-10, -10, None],
                "diagnosisstring": ["B", "A", "C"],
            }
        ),
    }

    streams, _, _ = build_event_streams(
        tables, representation="timegap", min_events_per_stay=1
    )

    assert streams[0].events == ["DX::A", "DX::B", "DX::C"]
    assert streams[0].event_times == [-10, -10, None]


def test_all_missing_offsets_use_token_order_without_time_gaps() -> None:
    tables = {
        "patient": pd.DataFrame(
            {
                "patientunitstayid": [101],
                "hospitaldischargestatus": ["Alive"],
            }
        ),
        "diagnosis": pd.DataFrame(
            {
                "patientunitstayid": [101, 101],
                "diagnosisstring": ["Zeta", "Alpha"],
            }
        ),
    }

    streams, _, _ = build_event_streams(
        tables, representation="timegap", min_events_per_stay=1
    )

    assert streams[0].events == ["DX::ALPHA", "DX::ZETA"]
    assert streams[0].event_times == [None, None]


def test_gap_bucket_boundaries_are_stable() -> None:
    offsets = [0, 15, 31, 91, 152, 332, 513, 873, 1234, 1954, 2675, 4115, 5556]
    tables = {
        "patient": pd.DataFrame(
            {
                "patientunitstayid": [101],
                "hospitaldischargestatus": ["Alive"],
            }
        ),
        "diagnosis": pd.DataFrame(
            {
                "patientunitstayid": [101] * len(offsets),
                "diagnosisoffset": offsets,
                "diagnosisstring": [f"Event {index}" for index in range(len(offsets))],
            }
        ),
    }

    streams, _, _ = build_event_streams(
        tables, representation="timegap", min_events_per_stay=1
    )

    gaps = [token for token in streams[0].events if token.startswith("TIME_GAP::")]
    assert gaps == [
        "TIME_GAP::LE_15M",
        "TIME_GAP::15M_1H",
        "TIME_GAP::15M_1H",
        "TIME_GAP::1H_3H",
        "TIME_GAP::1H_3H",
        "TIME_GAP::3H_6H",
        "TIME_GAP::3H_6H",
        "TIME_GAP::6H_12H",
        "TIME_GAP::6H_12H",
        "TIME_GAP::12H_24H",
        "TIME_GAP::12H_24H",
        "TIME_GAP::GT_24H",
    ]


def test_minimum_length_filter_and_empty_static_context_are_supported() -> None:
    tables = {
        "patient": pd.DataFrame(
            {
                "patientunitstayid": [101, 102],
                "hospitaldischargestatus": ["Alive", "Expired"],
            }
        ),
        "diagnosis": pd.DataFrame(
            {
                "patientunitstayid": [101, 102, 102],
                "diagnosisoffset": [1, 1, 2],
                "diagnosisstring": ["A", "A", "B"],
            }
        ),
    }

    streams, outcomes, stats = build_event_streams(
        tables, representation="timegap_static", min_events_per_stay=2
    )

    assert [stream.patientunitstayid for stream in streams] == [102]
    assert outcomes["patientunitstayid"].tolist() == [102]
    assert stats.kept_stays == 1
    assert stats.skipped_stays == 1


def test_builder_does_not_truncate_long_sequences() -> None:
    event_count = 150
    tables = {
        "patient": pd.DataFrame(
            {
                "patientunitstayid": [101],
                "hospitaldischargestatus": ["Alive"],
            }
        ),
        "diagnosis": pd.DataFrame(
            {
                "patientunitstayid": [101] * event_count,
                "diagnosisoffset": list(range(event_count)),
                "diagnosisstring": [f"Event {index}" for index in range(event_count)],
            }
        ),
    }

    streams, _, stats = build_event_streams(
        tables, representation="basic", min_events_per_stay=1
    )

    assert len(streams[0].events) == event_count
    assert stats.max_sequence_length == event_count


@pytest.mark.parametrize("minimum", [0, -1, 1.5, True])
def test_invalid_minimum_event_count_is_rejected(minimum: object) -> None:
    with pytest.raises(ValueError, match="min_events_per_stay"):
        build_event_streams(
            _tables(), representation="basic", min_events_per_stay=minimum
        )


def test_invalid_representation_is_rejected() -> None:
    with pytest.raises(ValueError, match="representation"):
        build_event_streams(
            _tables(), representation="raw", min_events_per_stay=1
        )
