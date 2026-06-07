"""Tests for ICU event-stream data and serialization contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from icu_pretrain.constants import EVENT_FAMILIES
from icu_pretrain.data.eicu_event_builder import (
    EventStats,
    EventStream,
    OutcomeRecord,
    read_event_streams_jsonl,
    validate_event_stats,
    validate_event_stream,
    validate_outcome_record,
    write_event_streams_jsonl,
)


def make_stream(**overrides: object) -> EventStream:
    values = {
        "patientunitstayid": "stay-1",
        "events": ["STATIC::GENDER::F", "DX::SEPSIS", "LAB::CREATININE::Q2"],
        "event_times": [None, 10.0, 10.0],
        "representation": "timegap_static",
        "metadata": {"source": "synthetic"},
    }
    values.update(overrides)
    return EventStream(**values)


def test_event_families_match_the_public_contract() -> None:
    assert EVENT_FAMILIES == [
        "STATIC",
        "DX",
        "LAB",
        "VITAL",
        "MED",
        "INFUSION",
        "TREATMENT",
        "TIME_GAP",
    ]


def test_event_stream_round_trips_through_jsonl(tmp_path: Path) -> None:
    stream = make_stream()
    path = tmp_path / "event_streams.jsonl"

    write_event_streams_jsonl(path, [stream])

    assert read_event_streams_jsonl(path) == [stream]


def test_event_stream_accepts_duplicate_timestamps() -> None:
    assert validate_event_stream(make_stream()) == make_stream()


@pytest.mark.parametrize("patientunitstayid", ["", "   ", None])
def test_event_stream_rejects_empty_patient_id(patientunitstayid: object) -> None:
    with pytest.raises(ValueError, match="patientunitstayid"):
        validate_event_stream(make_stream(patientunitstayid=patientunitstayid))


def test_event_stream_rejects_too_few_events_after_filtering() -> None:
    with pytest.raises(ValueError, match="min_events_per_stay"):
        validate_event_stream(
            make_stream(events=["STATIC::GENDER::F"], event_times=[None]),
            min_events_per_stay=2,
        )


@pytest.mark.parametrize(
    "events",
    [
        ["UNKNOWN::VALUE"],
        ["DX::SEPSIS", 42],
        ["DX::"],
    ],
)
def test_event_stream_rejects_invalid_event_tokens(events: list[object]) -> None:
    with pytest.raises(ValueError, match="events"):
        validate_event_stream(make_stream(events=events, event_times=None))


def test_event_stream_rejects_unsorted_timestamps() -> None:
    with pytest.raises(ValueError, match="event_times"):
        validate_event_stream(make_stream(event_times=[None, 20, 10]))


def test_event_stream_rejects_mismatched_timestamp_count() -> None:
    with pytest.raises(ValueError, match="event_times"):
        validate_event_stream(make_stream(event_times=[1, 2]))


def test_outcome_record_requires_binary_mortality() -> None:
    assert validate_outcome_record(OutcomeRecord("stay-1", 1)) == OutcomeRecord(
        "stay-1", 1
    )

    with pytest.raises(ValueError, match="mortality"):
        validate_outcome_record(OutcomeRecord("stay-1", None))

    with pytest.raises(ValueError, match="mortality"):
        validate_outcome_record(OutcomeRecord("stay-1", 2))


def test_event_stats_contract_contains_required_summary_fields() -> None:
    stats = EventStats(
        total_stays=3,
        kept_stays=2,
        skipped_stays=1,
        min_sequence_length=2,
        max_sequence_length=4,
        median_sequence_length=3.0,
        token_family_counts={family: 0 for family in EVENT_FAMILIES},
    )

    assert validate_event_stats(stats) == stats

    invalid = EventStats(
        total_stays=3,
        kept_stays=2,
        skipped_stays=0,
        min_sequence_length=2,
        max_sequence_length=4,
        median_sequence_length=3.0,
        token_family_counts={family: 0 for family in EVENT_FAMILIES},
    )
    with pytest.raises(ValueError, match="total_stays"):
        validate_event_stats(invalid)
