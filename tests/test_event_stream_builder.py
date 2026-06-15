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
                "uniquepid": ["P101", "P102", "P103"],
                "hospitalid": [1, 1, 1],
                "age": [65, 50, 40],
                "gender": ["Female", "Male", "Female"],
                "hospitaldischargestatus": ["Alive", "Expired", None],
                "unitdischargeoffset": [1440, 1440, 1440],
            }
        ),
        "diagnosis": pd.DataFrame(
            {
                "patientunitstayid": [
                    101, 101, 101, 101, 101,
                    102, 102, 102, 102, 102
                ],
                "diagnosisoffset": [
                    60, 60, 120, 180, 240,
                    10, 20, 30, 40, 50
                ],
                "diagnosisstring": [
                    "Sepsis", "Acidosis", "Shock", "Asthma", "Arrhythmia",
                    "Shock", "Sepsis", "Acidosis", "Asthma", "Arrhythmia"
                ],
            }
        )
    }


def test_builds_chronological_timegap_static_streams_and_filters_outcomes() -> None:
    streams, outcomes, stats = build_event_streams(
        _tables(), representation="timegap_static", min_events_per_stay=5
    )

    assert [stream.patientunitstayid for stream in streams] == [101, 102]
    assert streams[0].events == [
        "STATIC::AGE_BIN::60_79",
        "STATIC::GENDER::F",
        "STATIC::UNIT_ADMIT_SOURCE::UNKNOWN",
        "STATIC::UNIT_TYPE::UNKNOWN",
        "DX::ACIDOSIS",
        "DX::SEPSIS",
        "TIME_GAP::16_60M",
        "DX::SHOCK",
        "TIME_GAP::16_60M",
        "DX::ASTHMA",
        "TIME_GAP::16_60M",
        "DX::ARRHYTHMIA",
    ]
    assert streams[0].event_times == [None, None, None, None, 60, 60, 120, 120, 180, 180, 240, 240]
    assert outcomes.to_dict("records") == [
        {"patientunitstayid": 101, "mortality": 0},
        {"patientunitstayid": 102, "mortality": 1},
    ]
    assert stats.total_stays == 3
    assert stats.kept_stays == 2
    assert stats.skipped_stays == 1
    assert stats.min_sequence_length == 12
    assert stats.max_sequence_length == 13
    assert stats.median_sequence_length == 12.5
    assert set(stats.token_family_counts) == set(EVENT_FAMILIES)
    assert stats.token_family_counts["TIME_GAP"] == 7
    assert "original_counts" in streams[0].metadata
    assert "retained_counts" in streams[0].metadata


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
                "uniquepid": ["P101"],
                "hospitalid": [1],
                "hospitaldischargestatus": ["Alive"],
                "unitdischargeoffset": [1440],
            }
        ),
        "diagnosis": pd.DataFrame(
            {
                "patientunitstayid": [101, 101, 101, 101, 101, 101],
                "diagnosisoffset": [10, 10, 10, 20, 20, 30],
                "diagnosisstring": ["A", "B", "C", "D", "E", "F"],
            }
        ),
    }

    streams, _, _ = build_event_streams(
        tables, representation="timegap", min_events_per_stay=1
    )

    assert streams[0].events == [
        "DX::A",
        "DX::B",
        "DX::C",
        "TIME_GAP::0_15M",
        "DX::D",
        "DX::E",
        "TIME_GAP::0_15M",
        "DX::F",
    ]


def test_gap_bucket_boundaries_are_stable() -> None:
    offsets = [0, 15, 31, 92, 273, 634, 1000]
    tables = {
        "patient": pd.DataFrame(
            {
                "patientunitstayid": [101],
                "uniquepid": ["P101"],
                "hospitalid": [1],
                "hospitaldischargestatus": ["Alive"],
                "unitdischargeoffset": [1440],
            }
        ),
        "diagnosis": pd.DataFrame(
            {
                "patientunitstayid": [101] * len(offsets),
                "diagnosisoffset": offsets,
                "diagnosisstring": [f"Event{index}" for index in range(len(offsets))],
            }
        ),
    }

    streams, _, _ = build_event_streams(
        tables, representation="timegap", min_events_per_stay=1
    )

    gaps = [token for token in streams[0].events if token.startswith("TIME_GAP::")]
    assert gaps == [
        "TIME_GAP::0_15M",
        "TIME_GAP::16_60M",
        "TIME_GAP::61_180M",
        "TIME_GAP::181_360M",
        "TIME_GAP::GT_360M",
        "TIME_GAP::GT_360M",
    ]


def test_minimum_length_filter_and_empty_static_context_are_supported() -> None:
    tables = {
        "patient": pd.DataFrame(
            {
                "patientunitstayid": [101, 102],
                "uniquepid": ["P101", "P102"],
                "hospitalid": [1, 1],
                "hospitaldischargestatus": ["Alive", "Expired"],
                "unitdischargeoffset": [1440, 1440],
            }
        ),
        "diagnosis": pd.DataFrame(
            {
                "patientunitstayid": [101, 102, 102, 102, 102, 102],
                "diagnosisoffset": [10, 10, 20, 30, 40, 50],
                "diagnosisstring": ["A", "A", "B", "C", "D", "E"],
            }
        ),
    }

    streams, outcomes, stats = build_event_streams(
        tables, representation="timegap_static", min_events_per_stay=5
    )

    assert [stream.patientunitstayid for stream in streams] == [102]
    assert outcomes["patientunitstayid"].tolist() == [102]
    assert stats.kept_stays == 1
    assert stats.skipped_stays == 1


def test_sequence_capping_limit_and_sampling() -> None:
    event_count = 300
    tables = {
        "patient": pd.DataFrame(
            {
                "patientunitstayid": [101],
                "uniquepid": ["P101"],
                "hospitalid": [1],
                "hospitaldischargestatus": ["Alive"],
                "unitdischargeoffset": [1440],
            }
        ),
        "diagnosis": pd.DataFrame(
            {
                "patientunitstayid": [101] * event_count,
                "diagnosisoffset": list(range(event_count)),
                "diagnosisstring": [f"Event{index}" for index in range(event_count)],
            }
        ),
    }

    streams, _, stats = build_event_streams(
        tables, representation="basic", min_events_per_stay=1
    )

    assert len(streams[0].events) == 256
    assert streams[0].events[0] == "DX::EVENT0"
    assert streams[0].events[-1] == f"DX::EVENT{event_count - 1}"


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
