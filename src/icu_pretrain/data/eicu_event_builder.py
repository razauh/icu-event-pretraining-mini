"""Build chronological ICU event streams from eICU demo tables."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from icu_pretrain.constants import EVENT_FAMILIES, EVENT_REPRESENTATIONS


PatientStayId = int | str
EventTime = int | float | None


@dataclass(slots=True)
class EventStream:
    """Validated patient-level chronological event sequence."""

    patientunitstayid: PatientStayId
    events: list[str]
    representation: str
    event_times: list[EventTime] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OutcomeRecord:
    """Binary mortality-style outcome associated with one ICU stay."""

    patientunitstayid: PatientStayId
    mortality: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EventStats:
    """Aggregate, non-patient-level event-stream summary."""

    total_stays: int
    kept_stays: int
    skipped_stays: int
    min_sequence_length: int
    max_sequence_length: int
    median_sequence_length: float
    token_family_counts: dict[str, int]


def _validate_patient_id(patientunitstayid: Any) -> None:
    if isinstance(patientunitstayid, bool):
        raise ValueError("patientunitstayid must be a non-empty string or positive integer")
    if isinstance(patientunitstayid, int):
        if patientunitstayid <= 0:
            raise ValueError("patientunitstayid must be a positive integer")
        return
    if isinstance(patientunitstayid, str) and patientunitstayid.strip():
        return
    raise ValueError("patientunitstayid must be a non-empty string or positive integer")


def _validate_event_token(token: Any, index: int) -> None:
    if not isinstance(token, str):
        raise ValueError(f"events[{index}] must be a string")
    family, separator, suffix = token.partition("::")
    if not separator or not suffix:
        raise ValueError(f"events[{index}] must use FAMILY::VALUE format")
    if family not in EVENT_FAMILIES:
        raise ValueError(f"events[{index}] has unknown token family: {family}")


def validate_event_stream(
    stream: EventStream, *, min_events_per_stay: int = 1
) -> EventStream:
    """Validate an event stream without modifying it."""
    if not isinstance(stream, EventStream):
        raise ValueError("event stream must be an EventStream")
    _validate_patient_id(stream.patientunitstayid)

    if not isinstance(min_events_per_stay, int) or min_events_per_stay < 1:
        raise ValueError("min_events_per_stay must be a positive integer")
    if not isinstance(stream.events, list) or len(stream.events) < min_events_per_stay:
        raise ValueError(
            f"events must contain at least min_events_per_stay={min_events_per_stay} entries"
        )
    for index, token in enumerate(stream.events):
        _validate_event_token(token, index)

    if stream.representation not in EVENT_REPRESENTATIONS:
        choices = ", ".join(EVENT_REPRESENTATIONS)
        raise ValueError(f"representation must be one of: {choices}")
    if not isinstance(stream.metadata, dict):
        raise ValueError("metadata must be a mapping")

    if stream.event_times is not None:
        if not isinstance(stream.event_times, list):
            raise ValueError("event_times must be a list or null")
        if len(stream.event_times) != len(stream.events):
            raise ValueError("event_times must contain one entry per event")

        known_times: list[int | float] = []
        for index, event_time in enumerate(stream.event_times):
            if event_time is None:
                continue
            if isinstance(event_time, bool) or not isinstance(event_time, (int, float)):
                raise ValueError(f"event_times[{index}] must be numeric or null")
            known_times.append(event_time)
        if any(current < previous for previous, current in zip(known_times, known_times[1:])):
            raise ValueError("event_times must be sorted in nondecreasing order")

    return stream


def validate_outcome_record(record: OutcomeRecord) -> OutcomeRecord:
    """Validate a binary mortality outcome without modifying it."""
    if not isinstance(record, OutcomeRecord):
        raise ValueError("outcome record must be an OutcomeRecord")
    _validate_patient_id(record.patientunitstayid)
    if isinstance(record.mortality, bool) or record.mortality not in {0, 1}:
        raise ValueError("mortality must be an integer binary label: 0 or 1")
    if not isinstance(record.metadata, dict):
        raise ValueError("metadata must be a mapping")
    return record


def validate_event_stats(stats: EventStats) -> EventStats:
    """Validate aggregate event statistics without modifying them."""
    if not isinstance(stats, EventStats):
        raise ValueError("event stats must be an EventStats")

    count_fields = {
        "total_stays": stats.total_stays,
        "kept_stays": stats.kept_stays,
        "skipped_stays": stats.skipped_stays,
        "min_sequence_length": stats.min_sequence_length,
        "max_sequence_length": stats.max_sequence_length,
    }
    for name, value in count_fields.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if stats.total_stays != stats.kept_stays + stats.skipped_stays:
        raise ValueError("total_stays must equal kept_stays plus skipped_stays")
    if stats.min_sequence_length > stats.max_sequence_length:
        raise ValueError("min_sequence_length must not exceed max_sequence_length")
    if (
        isinstance(stats.median_sequence_length, bool)
        or not isinstance(stats.median_sequence_length, (int, float))
        or stats.median_sequence_length < 0
    ):
        raise ValueError("median_sequence_length must be non-negative")

    if not isinstance(stats.token_family_counts, dict):
        raise ValueError("token_family_counts must be a mapping")
    if set(stats.token_family_counts) != set(EVENT_FAMILIES):
        raise ValueError("token_family_counts must contain every supported event family")
    for family, count in stats.token_family_counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"token_family_counts.{family} must be a non-negative integer")
    return stats


def event_stream_from_dict(record: Mapping[str, Any]) -> EventStream:
    """Create and validate an event stream from a serialized mapping."""
    if not isinstance(record, Mapping):
        raise ValueError("serialized event stream must be a mapping")
    required = {"patientunitstayid", "events", "representation"}
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(f"serialized event stream is missing: {', '.join(missing)}")

    stream = EventStream(
        patientunitstayid=record["patientunitstayid"],
        events=record["events"],
        representation=record["representation"],
        event_times=record.get("event_times"),
        metadata=record.get("metadata", {}),
    )
    return validate_event_stream(stream)


def write_event_streams_jsonl(
    path: str | Path, streams: Iterable[EventStream]
) -> None:
    """Write validated event streams as UTF-8 JSON Lines."""
    output_path = Path(path)
    with output_path.open("w", encoding="utf-8") as handle:
        for stream in streams:
            validate_event_stream(stream)
            handle.write(json.dumps(asdict(stream), sort_keys=True))
            handle.write("\n")


def read_event_streams_jsonl(path: str | Path) -> list[EventStream]:
    """Read and validate event streams from UTF-8 JSON Lines."""
    streams: list[EventStream] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                streams.append(event_stream_from_dict(record))
            except (json.JSONDecodeError, ValueError, TypeError) as error:
                raise ValueError(
                    f"Invalid event stream at JSONL line {line_number}: {error}"
                ) from error
    return streams


def build_event_streams() -> None:
    """Placeholder for the event stream builder implementation."""
    raise NotImplementedError("Event stream construction is not implemented yet.")
