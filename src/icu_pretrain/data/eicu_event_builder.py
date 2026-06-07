"""Build chronological ICU event streams from eICU demo tables."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import pandas as pd

from icu_pretrain.constants import EVENT_FAMILIES, EVENT_REPRESENTATIONS


PatientStayId = int | str
EventTime = int | float | None


OUTCOME_FIELDS = (
    ("patient", "hospitaldischargestatus"),
    ("apachePatientResult", "actualhospitalmortality"),
    ("patient", "unitdischargestatus"),
    ("apachePatientResult", "actualicumortality"),
)

_NEGATIVE_OUTCOME_LABELS = frozenset({"0", "alive", "survived"})
_POSITIVE_OUTCOME_LABELS = frozenset({"1", "dead", "died", "expired"})

_CATEGORICAL_EVENT_FIELDS = (
    ("diagnosis", "DX", "diagnosisstring", "diagnosisoffset"),
    ("medication", "MED", "drugname", "drugstartoffset"),
    ("infusionDrug", "INFUSION", "drugname", "infusionoffset"),
    ("treatment", "TREATMENT", "treatmentstring", "treatmentoffset"),
)
_UNKNOWN_TEXT_VALUES = frozenset(
    {"", "n/a", "na", "none", "not available", "not recorded", "other", "unknown"}
)
_TOKEN_SEPARATOR_PATTERN = re.compile(r"[^A-Z0-9]+")


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


def normalize_token_text(value: Any, *, preserve_unknown: bool = False) -> str | None:
    """Normalize a categorical value for use as an event-token suffix."""
    if pd.isna(value):
        return None
    normalized = str(value).strip().casefold()
    if normalized in _UNKNOWN_TEXT_VALUES:
        return "UNKNOWN" if preserve_unknown else None
    token = _TOKEN_SEPARATOR_PATTERN.sub("_", normalized.upper()).strip("_")
    return token or None


def age_bin(value: Any) -> str | None:
    """Map an eICU age value to a compact ten-year bin."""
    if pd.isna(value) or isinstance(value, bool):
        return None
    normalized = str(value).strip().casefold()
    if normalized.startswith(">") and "89" in normalized:
        return "90_PLUS"
    try:
        age = float(normalized)
    except (TypeError, ValueError):
        return None
    if age < 0:
        return None
    if age >= 90:
        return "90_PLUS"
    lower = int(age // 10) * 10
    return f"{lower}_{lower + 10}"


def _gender_token(value: Any) -> str | None:
    normalized = normalize_token_text(value, preserve_unknown=True)
    if normalized in {"F", "FEMALE"}:
        return "F"
    if normalized in {"M", "MALE"}:
        return "M"
    return "UNKNOWN" if normalized is not None else None


def _event_time(value: Any) -> EventTime:
    if pd.isna(value) or isinstance(value, bool):
        return None
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    number = float(numeric)
    return int(number) if number.is_integer() else number


def _require_stay_column(frame: pd.DataFrame, table_name: str) -> None:
    if "patientunitstayid" not in frame.columns:
        raise ValueError(
            f"eICU demo table '{table_name}' is missing required column: "
            "patientunitstayid"
        )


def _extract_static_events(
    tables: Mapping[str, pd.DataFrame],
) -> dict[PatientStayId, list[tuple[str, EventTime]]]:
    patient = tables.get("patient")
    if patient is None:
        return {}
    _require_stay_column(patient, "patient")

    field_builders = (
        ("age", "AGE_BIN", age_bin),
        ("gender", "GENDER", _gender_token),
        (
            "hospitaladmitsource",
            "ADMISSION_SOURCE",
            lambda value: normalize_token_text(value, preserve_unknown=True),
        ),
        (
            "unittype",
            "UNIT_TYPE",
            lambda value: normalize_token_text(value, preserve_unknown=True),
        ),
    )
    events: dict[PatientStayId, list[tuple[str, EventTime]]] = {}
    seen: dict[PatientStayId, set[str]] = {}
    for _, row in patient.iterrows():
        stay_id = row["patientunitstayid"]
        if pd.isna(stay_id):
            continue
        for column, token_name, builder in field_builders:
            if column not in patient.columns:
                continue
            suffix = builder(row[column])
            if suffix is None:
                continue
            token = f"STATIC::{token_name}::{suffix}"
            stay_seen = seen.setdefault(stay_id, set())
            if token not in stay_seen:
                events.setdefault(stay_id, []).append((token, None))
                stay_seen.add(token)
    return events


def _extract_clinical_categorical_events(
    tables: Mapping[str, pd.DataFrame],
) -> dict[PatientStayId, list[tuple[str, EventTime]]]:
    grouped: dict[PatientStayId, list[tuple[str, EventTime, str]]] = {}
    seen: dict[PatientStayId, set[tuple[str, EventTime]]] = {}

    for table_name, family, value_column, offset_column in _CATEGORICAL_EVENT_FIELDS:
        frame = tables.get(table_name)
        if frame is None:
            continue
        _require_stay_column(frame, table_name)
        if value_column not in frame.columns:
            continue

        for _, row in frame.iterrows():
            stay_id = row["patientunitstayid"]
            if pd.isna(stay_id):
                continue
            suffix = normalize_token_text(row[value_column])
            if suffix is None:
                continue
            event_time = (
                _event_time(row[offset_column])
                if offset_column in frame.columns
                else None
            )
            token = f"{family}::{suffix}"
            event_key = (token, event_time)
            stay_seen = seen.setdefault(stay_id, set())
            if event_key in stay_seen:
                continue
            grouped.setdefault(stay_id, []).append((token, event_time, table_name))
            stay_seen.add(event_key)

    result: dict[PatientStayId, list[tuple[str, EventTime]]] = {}
    for stay_id, events in grouped.items():
        events.sort(
            key=lambda event: (
                event[1] is None,
                event[1] if event[1] is not None else 0,
                event[0],
                event[2],
            )
        )
        result[stay_id] = [(token, event_time) for token, event_time, _ in events]
    return result


def extract_categorical_events(
    tables: Mapping[str, pd.DataFrame], *, representation: str
) -> dict[PatientStayId, list[tuple[str, EventTime]]]:
    """Extract deterministic non-numeric events grouped by ICU stay.

    ``basic`` and ``timegap`` contain clinical categorical events only. Static
    context is prepended for ``timegap_static``. Time-gap tokens are added by
    the full event-stream builder in a later pipeline stage.
    """
    if not isinstance(tables, Mapping):
        raise ValueError("tables must be a mapping of eICU table names to DataFrames")
    if representation not in EVENT_REPRESENTATIONS:
        choices = ", ".join(EVENT_REPRESENTATIONS)
        raise ValueError(f"representation must be one of: {choices}")

    clinical = _extract_clinical_categorical_events(tables)
    static = (
        _extract_static_events(tables)
        if representation == "timegap_static"
        else {}
    )
    stay_ids = sorted(set(clinical) | set(static), key=lambda value: str(value))
    return {
        stay_id: [*static.get(stay_id, []), *clinical.get(stay_id, [])]
        for stay_id in stay_ids
        if static.get(stay_id) or clinical.get(stay_id)
    }


def _normalize_outcome_label(value: Any) -> int | None:
    """Return a binary outcome only for explicit, unambiguous values."""
    if pd.isna(value) or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if value == 0:
            return 0
        if value == 1:
            return 1
        return None

    normalized = str(value).strip().casefold()
    if normalized in _NEGATIVE_OUTCOME_LABELS:
        return 0
    if normalized in _POSITIVE_OUTCOME_LABELS:
        return 1
    return None


def _labels_by_stay(
    frame: pd.DataFrame, table_name: str, field: str
) -> tuple[dict[Any, int], set[Any]]:
    if "patientunitstayid" not in frame.columns:
        raise ValueError(
            f"eICU demo table '{table_name}' is missing required column: "
            "patientunitstayid"
        )

    labels: dict[Any, int] = {}
    conflicts: set[Any] = set()
    for stay_id, group in frame.groupby("patientunitstayid", sort=False, dropna=False):
        if pd.isna(stay_id):
            continue
        normalized = {
            label
            for label in (_normalize_outcome_label(value) for value in group[field])
            if label is not None
        }
        if len(normalized) == 1:
            labels[stay_id] = normalized.pop()
        elif len(normalized) > 1:
            conflicts.add(stay_id)
    return labels, conflicts


def extract_outcomes(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Extract one conservative mortality-style binary label per ICU stay.

    Hospital-level fields take precedence over ICU-level fallback fields. Stays
    with unavailable or conflicting labels are omitted rather than treated as
    survivors. Aggregate exclusion counts are stored in ``DataFrame.attrs``.
    """
    if not isinstance(tables, Mapping):
        raise ValueError("tables must be a mapping of eICU table names to DataFrames")

    available_fields = [
        (table_name, field)
        for table_name, field in OUTCOME_FIELDS
        if table_name in tables and field in tables[table_name].columns
    ]
    if not available_fields:
        raise ValueError(
            "patient or apachePatientResult must contain a supported mortality "
            "or discharge-status label field"
        )

    candidate_stays: set[Any] = set()
    for table_name in {table_name for table_name, _ in available_fields}:
        frame = tables[table_name]
        if "patientunitstayid" not in frame.columns:
            raise ValueError(
                f"eICU demo table '{table_name}' is missing required column: "
                "patientunitstayid"
            )
        candidate_stays.update(frame["patientunitstayid"].dropna().tolist())

    selected: dict[Any, int] = {}
    conflicting_stays: set[Any] = set()
    for table_name, field in available_fields:
        labels, conflicts = _labels_by_stay(tables[table_name], table_name, field)
        unresolved_conflicts = conflicts.difference(selected)
        conflicting_stays.update(unresolved_conflicts)
        for stay_id, label in labels.items():
            if stay_id not in selected and stay_id not in conflicting_stays:
                selected[stay_id] = label

    records = [
        {"patientunitstayid": stay_id, "mortality": int(label)}
        for stay_id, label in selected.items()
    ]
    outcomes = pd.DataFrame.from_records(
        records, columns=["patientunitstayid", "mortality"]
    )
    if not outcomes.empty:
        outcomes = outcomes.sort_values("patientunitstayid", kind="stable").reset_index(
            drop=True
        )
        outcomes["mortality"] = outcomes["mortality"].astype(int)

    outcomes.attrs["outcome_stats"] = {
        "candidate_stays": len(candidate_stays),
        "labelled_stays": len(outcomes),
        "unavailable_labels": len(candidate_stays - set(selected) - conflicting_stays),
        "conflicting_labels": len(conflicting_stays),
    }
    return outcomes


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
