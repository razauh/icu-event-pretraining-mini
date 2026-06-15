"""Build chronological ICU event streams from eICU demo tables."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import pandas as pd

from icu_pretrain.constants import (
    ARTIFACT_HASH_KEYS,
    CONTRACT_SCHEMA_VERSION,
    EVENT_FAMILIES,
    EVENT_REPRESENTATIONS,
    MANIFEST_STATUSES,
    RUN_STATUSES,
    SPLIT_NAMES,
)


PatientStayId = int | str
EventTime = int | float | None




_CATEGORICAL_EVENT_FIELDS = (
    ("diagnosis", "DX", "diagnosisstring", "diagnosisoffset"),
    ("medication", "MED", "drugname", "drugstartoffset"),
    ("infusionDrug", "INFUSION", "drugname", "infusionoffset"),
    ("treatment", "TREATMENT", "treatmentstring", "treatmentoffset"),
)
_LAB_EVENT_FIELDS = ("labname", "labresult", "labresultoffset")
_VITAL_TABLES = ("vitalPeriodic", "vitalAperiodic")
_NUMERIC_METADATA_COLUMNS = frozenset({"patientunitstayid", "observationoffset"})
_UNKNOWN_TEXT_VALUES = frozenset(
    {"", "n/a", "na", "none", "not available", "not recorded", "other", "unknown"}
)
_TOKEN_SEPARATOR_PATTERN = re.compile(r"[^A-Z0-9]+")
_MODEL_TOKEN_PROHIBITED_FRAGMENTS = (
    "PATIENTUNITSTAYID",
    "PATIENTHEALTHSYSTEMSTAYID",
    "UNIQUEPID",
    "HOSPITALID",
    "HOSPITAL_ID",
    "WARDID",
    "WARD_ID",
)


@dataclass(slots=True)
class EventStream:
    """Validated patient-level chronological event sequence."""

    patientunitstayid: PatientStayId
    events: list[str]
    representation: str
    event_times: list[EventTime] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    split_name: str = "train"


@dataclass(slots=True)
class OutcomeRecord:
    """Binary mortality-style outcome associated with one ICU stay."""

    patientunitstayid: PatientStayId
    mortality: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ICUStayRecord:
    """Identifiers and outcome metadata retained outside model inputs."""

    patientunitstayid: PatientStayId
    uniquepid: str
    hospitalid: int | str
    mortality: int


@dataclass(slots=True)
class EligibilityDecision:
    """Eligibility result for one ICU stay."""

    patientunitstayid: PatientStayId
    eligible: bool
    exclusion_reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SplitRecord:
    """Local patient-grouped split metadata."""

    patientunitstayid: PatientStayId
    uniquepid: str
    hospitalid: int | str
    split_name: str


@dataclass(slots=True)
class FittedPreprocessing:
    """Training-only fitted preprocessing metadata."""

    artifact_hash: str
    fit_split: str = "train"
    numeric_bins: dict[str, list[float]] = field(default_factory=dict)
    category_maps: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(slots=True)
class CohortSummary:
    """Aggregate eligibility and outcome counts."""

    total_stays: int
    eligible_stays: int
    exclusion_counts: dict[str, int]
    class_counts: dict[str, int]


@dataclass(slots=True)
class StageManifest:
    """Restartable preprocessing-stage manifest."""

    stage_name: str
    status: str
    config_hash: str
    input_hashes: dict[str, str]
    upstream_hashes: dict[str, str]
    shard_count: int
    completed_shards: list[int]
    aggregate_counts: dict[str, int]
    skipped_counts: dict[str, int]
    started_at: str
    updated_at: str
    completed_at: str | None = None
    failure: dict[str, str] | None = None
    schema_version: int = CONTRACT_SCHEMA_VERSION


@dataclass(slots=True)
class RunState:
    """Atomic summary for one local run."""

    run_id: str
    status: str
    updated_at: str
    artifact_hashes: dict[str, str]
    last_checkpoint: str | None = None


@dataclass(slots=True)
class CheckpointContract:
    """Metadata required to resume training without restarting an epoch."""

    run_id: str
    model_state: dict[str, Any]
    prediction_head_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any]
    gradient_state: dict[str, Any]
    accumulation_step: int
    epoch: int
    next_batch_cursor: int
    global_batch: int
    optimizer_step: int
    best_metric: float | None
    best_epoch: int | None
    early_stopping_state: dict[str, Any]
    threshold_state: dict[str, Any]
    rng_state: dict[str, Any]
    sampler_state: dict[str, Any]
    artifact_hashes: dict[str, str]
    training_history: dict[str, Any]
    creation_reason: str
    schema_version: int = CONTRACT_SCHEMA_VERSION


@dataclass(slots=True)
class PublicAggregateSummary:
    """Validated public-safe aggregate result payload."""

    values: dict[str, Any]


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


def _validate_group_id(value: Any, name: str) -> None:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{name} must be a non-empty string or positive integer")
    if isinstance(value, int) and value > 0:
        return
    if isinstance(value, str) and value.strip():
        return
    raise ValueError(f"{name} must be a non-empty string or positive integer")


def _validate_nonempty_text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _validate_hashes(hashes: Any, name: str, *, exact_artifacts: bool = False) -> None:
    if not isinstance(hashes, dict):
        raise ValueError(f"{name} must be a mapping")
    if exact_artifacts and set(hashes) != set(ARTIFACT_HASH_KEYS):
        raise ValueError(f"{name} must contain every artifact compatibility hash")
    for key, value in hashes.items():
        _validate_nonempty_text(key, f"{name} key")
        _validate_nonempty_text(value, f"{name}.{key}")


def _validate_count_mapping(counts: Any, name: str) -> None:
    if not isinstance(counts, dict):
        raise ValueError(f"{name} must be a mapping")
    for key, value in counts.items():
        _validate_nonempty_text(key, f"{name} key")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name}.{key} must be a non-negative integer")


def _validate_timestamp(value: Any, name: str) -> None:
    _validate_nonempty_text(value, name)
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from error


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

    if not isinstance(min_events_per_stay, int) or min_events_per_stay < 0:
        raise ValueError("min_events_per_stay must be a non-negative integer")
    if not isinstance(stream.events, list) or len(stream.events) < min_events_per_stay:
        raise ValueError(
            f"events must contain at least min_events_per_stay={min_events_per_stay} entries"
        )
    for index, token in enumerate(stream.events):
        _validate_event_token(token, index)

    if stream.representation not in EVENT_REPRESENTATIONS:
        choices = ", ".join(EVENT_REPRESENTATIONS)
        raise ValueError(f"representation must be one of: {choices}")
    if stream.split_name not in SPLIT_NAMES:
        choices = ", ".join(SPLIT_NAMES)
        raise ValueError(f"split_name must be one of: {choices}")
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
            if not math.isfinite(event_time):
                raise ValueError(f"event_times[{index}] must be finite")
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


def validate_outcomes_for_eligible_stays(
    records: Iterable[OutcomeRecord], eligible_stay_ids: Iterable[PatientStayId]
) -> list[OutcomeRecord]:
    """Require exactly one binary outcome for every eligible ICU stay."""
    eligible_ids = list(eligible_stay_ids)
    for stay_id in eligible_ids:
        _validate_patient_id(stay_id)
    if len(set(eligible_ids)) != len(eligible_ids):
        raise ValueError("eligible_stay_ids must not contain duplicates")

    validated = [validate_outcome_record(record) for record in records]
    outcome_ids = [record.patientunitstayid for record in validated]
    if len(set(outcome_ids)) != len(outcome_ids):
        raise ValueError("outcomes must contain exactly one label per eligible stay")
    if set(outcome_ids) != set(eligible_ids):
        raise ValueError("outcomes must contain exactly one label per eligible stay")
    return validated


def validate_icu_stay_record(record: ICUStayRecord) -> ICUStayRecord:
    if not isinstance(record, ICUStayRecord):
        raise ValueError("ICU stay record must be an ICUStayRecord")
    _validate_patient_id(record.patientunitstayid)
    _validate_group_id(record.uniquepid, "uniquepid")
    _validate_group_id(record.hospitalid, "hospitalid")
    if isinstance(record.mortality, bool) or record.mortality not in {0, 1}:
        raise ValueError("mortality must be an integer binary label: 0 or 1")
    return record


def validate_eligibility_decision(
    decision: EligibilityDecision,
) -> EligibilityDecision:
    if not isinstance(decision, EligibilityDecision):
        raise ValueError("eligibility decision must be an EligibilityDecision")
    _validate_patient_id(decision.patientunitstayid)
    if not isinstance(decision.eligible, bool):
        raise ValueError("eligible must be boolean")
    if not isinstance(decision.exclusion_reasons, list) or any(
        not isinstance(reason, str) or not reason.strip()
        for reason in decision.exclusion_reasons
    ):
        raise ValueError("exclusion_reasons must contain non-empty strings")
    if decision.eligible and decision.exclusion_reasons:
        raise ValueError("eligible decisions cannot contain exclusion_reasons")
    if not decision.eligible and not decision.exclusion_reasons:
        raise ValueError("ineligible decisions require exclusion_reasons")
    return decision


def validate_split_record(record: SplitRecord) -> SplitRecord:
    if not isinstance(record, SplitRecord):
        raise ValueError("split record must be a SplitRecord")
    _validate_patient_id(record.patientunitstayid)
    _validate_group_id(record.uniquepid, "uniquepid")
    _validate_group_id(record.hospitalid, "hospitalid")
    if record.split_name not in SPLIT_NAMES:
        raise ValueError(f"split_name must be one of: {', '.join(SPLIT_NAMES)}")
    return record


def validate_fitted_preprocessing(
    fitted: FittedPreprocessing,
) -> FittedPreprocessing:
    if not isinstance(fitted, FittedPreprocessing):
        raise ValueError("fitted preprocessing must be FittedPreprocessing")
    _validate_nonempty_text(fitted.artifact_hash, "artifact_hash")
    if fitted.fit_split != "train":
        raise ValueError("fit_split must be train")
    if not isinstance(fitted.numeric_bins, dict) or not isinstance(
        fitted.category_maps, dict
    ):
        raise ValueError("fitted preprocessing metadata must use mappings")
    for name, boundaries in fitted.numeric_bins.items():
        _validate_nonempty_text(name, "numeric_bins key")
        if not isinstance(boundaries, list) or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in boundaries
        ):
            raise ValueError(f"numeric_bins.{name} must contain finite numbers")
    for name, category_map in fitted.category_maps.items():
        _validate_nonempty_text(name, "category_maps key")
        if not isinstance(category_map, dict) or any(
            not isinstance(source, str)
            or not isinstance(target, str)
            or not source.strip()
            or not target.strip()
            for source, target in category_map.items()
        ):
            raise ValueError(
                f"category_maps.{name} must map non-empty strings to non-empty strings"
            )
    return fitted


def validate_cohort_summary(summary: CohortSummary) -> CohortSummary:
    if not isinstance(summary, CohortSummary):
        raise ValueError("cohort summary must be a CohortSummary")
    for name, value in (
        ("total_stays", summary.total_stays),
        ("eligible_stays", summary.eligible_stays),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if summary.eligible_stays > summary.total_stays:
        raise ValueError("eligible_stays cannot exceed total_stays")
    _validate_count_mapping(summary.exclusion_counts, "exclusion_counts")
    _validate_count_mapping(summary.class_counts, "class_counts")
    if set(summary.class_counts) != {"Alive", "Expired"}:
        raise ValueError("class_counts must contain Alive and Expired")
    if sum(summary.class_counts.values()) != summary.eligible_stays:
        raise ValueError("class_counts must sum to eligible_stays")
    return summary


def validate_stage_manifest(manifest: StageManifest) -> StageManifest:
    if not isinstance(manifest, StageManifest):
        raise ValueError("stage manifest must be a StageManifest")
    _validate_nonempty_text(manifest.stage_name, "stage_name")
    if manifest.schema_version != CONTRACT_SCHEMA_VERSION:
        raise ValueError("schema_version is incompatible")
    if manifest.status not in MANIFEST_STATUSES:
        raise ValueError(f"status must be one of: {', '.join(MANIFEST_STATUSES)}")
    _validate_nonempty_text(manifest.config_hash, "config_hash")
    _validate_hashes(manifest.input_hashes, "input_hashes")
    _validate_hashes(manifest.upstream_hashes, "upstream_hashes")
    if (
        isinstance(manifest.shard_count, bool)
        or not isinstance(manifest.shard_count, int)
        or manifest.shard_count < 0
    ):
        raise ValueError("shard_count must be a non-negative integer")
    if not isinstance(manifest.completed_shards, list) or any(
        isinstance(shard, bool)
        or not isinstance(shard, int)
        or shard < 0
        or shard >= manifest.shard_count
        for shard in manifest.completed_shards
    ):
        raise ValueError("completed_shards contains an invalid shard ID")
    if len(set(manifest.completed_shards)) != len(manifest.completed_shards):
        raise ValueError("completed_shards must not contain duplicates")
    _validate_count_mapping(manifest.aggregate_counts, "aggregate_counts")
    _validate_count_mapping(manifest.skipped_counts, "skipped_counts")
    for name, value in (
        ("started_at", manifest.started_at),
        ("updated_at", manifest.updated_at),
    ):
        _validate_timestamp(value, name)
    if manifest.status == "complete":
        _validate_timestamp(manifest.completed_at, "completed_at")
        if len(manifest.completed_shards) != manifest.shard_count:
            raise ValueError("complete manifest must contain every completed shard")
    if manifest.status == "failed":
        if not isinstance(manifest.failure, dict) or set(manifest.failure) != {
            "type",
            "message",
        }:
            raise ValueError("failed manifest requires failure type and message")
        _validate_nonempty_text(manifest.failure["type"], "failure.type")
        _validate_nonempty_text(manifest.failure["message"], "failure.message")
        validate_public_aggregate(manifest.failure)
    elif manifest.failure is not None:
        raise ValueError("failure metadata is only valid for failed manifests")
    return manifest


def validate_run_state(state: RunState) -> RunState:
    if not isinstance(state, RunState):
        raise ValueError("run state must be a RunState")
    _validate_nonempty_text(state.run_id, "run_id")
    if state.status not in RUN_STATUSES:
        raise ValueError(f"status must be one of: {', '.join(RUN_STATUSES)}")
    _validate_timestamp(state.updated_at, "updated_at")
    _validate_hashes(state.artifact_hashes, "artifact_hashes", exact_artifacts=True)
    if state.last_checkpoint is not None:
        _validate_nonempty_text(state.last_checkpoint, "last_checkpoint")
    return state


def validate_checkpoint_contract(
    checkpoint: CheckpointContract,
    *, expected_artifact_hashes: Mapping[str, str] | None = None,
) -> CheckpointContract:
    if not isinstance(checkpoint, CheckpointContract):
        raise ValueError("checkpoint must be a CheckpointContract")
    if checkpoint.schema_version != CONTRACT_SCHEMA_VERSION:
        raise ValueError("schema_version is incompatible")
    _validate_nonempty_text(checkpoint.run_id, "run_id")
    for name in (
        "model_state",
        "prediction_head_state",
        "optimizer_state",
        "scheduler_state",
        "gradient_state",
        "early_stopping_state",
        "threshold_state",
        "rng_state",
        "sampler_state",
        "training_history",
    ):
        if not isinstance(getattr(checkpoint, name), dict):
            raise ValueError(f"{name} must be a mapping")
    for name in (
        "accumulation_step",
        "epoch",
        "next_batch_cursor",
        "global_batch",
        "optimizer_step",
    ):
        value = getattr(checkpoint, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if checkpoint.best_metric is not None and (
        isinstance(checkpoint.best_metric, bool)
        or not isinstance(checkpoint.best_metric, (int, float))
        or not math.isfinite(checkpoint.best_metric)
    ):
        raise ValueError("best_metric must be finite or null")
    if checkpoint.best_epoch is not None and (
        isinstance(checkpoint.best_epoch, bool)
        or not isinstance(checkpoint.best_epoch, int)
        or checkpoint.best_epoch < 0
    ):
        raise ValueError("best_epoch must be a non-negative integer or null")
    _validate_hashes(
        checkpoint.artifact_hashes, "artifact_hashes", exact_artifacts=True
    )
    if not {"python", "numpy", "torch"}.issubset(checkpoint.rng_state):
        raise ValueError("rng_state must contain Python, NumPy, and PyTorch state")
    if not {"permutation", "generator_state", "cursor"}.issubset(
        checkpoint.sampler_state
    ):
        raise ValueError(
            "sampler_state must contain permutation, generator_state, and cursor"
        )
    _validate_nonempty_text(checkpoint.creation_reason, "creation_reason")
    if expected_artifact_hashes is not None and dict(expected_artifact_hashes) != (
        checkpoint.artifact_hashes
    ):
        raise ValueError("checkpoint artifact hashes are incompatible")
    return checkpoint


_PUBLIC_PROHIBITED_KEYS = frozenset(
    {
        "patientunitstayid",
        "patienthealthsystemstayid",
        "uniquepid",
        "hospitalid",
        "wardid",
        "events",
        "event_times",
        "event_stream",
        "event_streams",
        "tokens",
        "patient_records",
        "stay_records",
    }
)


def validate_public_aggregate(value: Any, *, path: str = "summary") -> Any:
    """Reject patient-level fields from recursively nested public output."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            if key.casefold() in _PUBLIC_PROHIBITED_KEYS:
                raise ValueError(f"{path}.{key} is patient-level and not public-safe")
            validate_public_aggregate(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            validate_public_aggregate(item, path=f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must contain finite values")
    return value


def validate_public_summary(summary: PublicAggregateSummary) -> PublicAggregateSummary:
    if not isinstance(summary, PublicAggregateSummary):
        raise ValueError("public summary must be a PublicAggregateSummary")
    if not isinstance(summary.values, dict):
        raise ValueError("public summary values must be a mapping")
    validate_public_aggregate(summary.values)
    return summary


def serialize_model_input(stream: EventStream) -> dict[str, Any]:
    """Serialize model features without local patient or grouping identifiers."""
    validate_event_stream(stream, min_events_per_stay=0)
    validate_public_aggregate(stream.metadata, path="metadata")
    for index, token in enumerate(stream.events):
        components = token.upper().split("::")
        if any(fragment in components for fragment in _MODEL_TOKEN_PROHIBITED_FRAGMENTS):
            raise ValueError(
                f"events[{index}] contains a patient or grouping identifier"
            )
    return {
        "events": list(stream.events),
        "event_times": None if stream.event_times is None else list(stream.event_times),
        "representation": stream.representation,
        "split_name": stream.split_name,
        "metadata": dict(stream.metadata),
    }


def write_split_metadata(
    path: str | Path,
    records: Iterable[SplitRecord],
    *,
    processed_root: str | Path,
) -> None:
    """Write local split metadata only below the configured processed root."""
    output_path = Path(path).resolve()
    root = Path(processed_root).resolve()
    try:
        output_path.relative_to(root)
    except ValueError as error:
        raise ValueError("split metadata must be written under processed_root") from error
    serialized = []
    for record in records:
        validate_split_record(record)
        serialized.append(asdict(record))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(serialized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def stage_manifest_from_dict(record: Mapping[str, Any]) -> StageManifest:
    """Create and validate a stage manifest from decoded JSON."""
    if not isinstance(record, Mapping):
        raise ValueError("serialized stage manifest must be a mapping")
    try:
        manifest = StageManifest(**record)
    except TypeError as error:
        raise ValueError(f"invalid stage manifest fields: {error}") from error
    return validate_stage_manifest(manifest)


def write_stage_manifest_json(path: str | Path, manifest: StageManifest) -> None:
    """Atomically write a validated stage manifest."""
    validate_stage_manifest(manifest)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def read_stage_manifest_json(path: str | Path) -> StageManifest:
    """Read and validate a stage manifest JSON file."""
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid stage manifest JSON: {error}") from error
    return stage_manifest_from_dict(record)


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


def age_bin(value: Any) -> str:
    if pd.isna(value) or isinstance(value, bool):
        return "UNKNOWN"
    normalized = str(value).strip().casefold()
    if ">" in normalized and "89" in normalized:
        return "80_PLUS"
    try:
        age = float(normalized)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if age < 0:
        return "UNKNOWN"
    if age <= 17:
        return "0_17"
    if age <= 39:
        return "18_39"
    if age <= 59:
        return "40_59"
    if age <= 79:
        return "60_79"
    return "80_PLUS"


def _gender_token(value: Any) -> str:
    normalized = normalize_token_text(value, preserve_unknown=True)
    if normalized in {"F", "FEMALE"}:
        return "F"
    if normalized in {"M", "MALE"}:
        return "M"
    return "UNKNOWN"


def _unit_admit_source_token(value: Any) -> str:
    normalized = normalize_token_text(value, preserve_unknown=True)
    if normalized is None or normalized == "UNKNOWN":
        return "UNKNOWN"
    return normalized


def _unit_type_token(value: Any) -> str:
    normalized = normalize_token_text(value, preserve_unknown=True)
    if normalized is None or normalized == "UNKNOWN":
        return "UNKNOWN"
    return normalized


def _is_prohibited_token(token: str) -> bool:
    upper = token.upper()
    prohibited = (
        "PATIENTUNITSTAYID",
        "PATIENTHEALTHSYSTEMSTAYID",
        "UNIQUEPID",
        "HOSPITALID",
        "WARDID",
        "ETHNICITY",
        "ACTIVEUPONDISCHARGE",
        "APACHE",
        "DISCHARGE",
    )
    return any(frag in upper for frag in prohibited)


def _event_time(value: Any) -> EventTime:
    if pd.isna(value) or isinstance(value, bool):
        return None
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    number = float(numeric)
    return int(number) if number.is_integer() else number


def _numeric_value(value: Any) -> float | None:
    if pd.isna(value) or isinstance(value, bool):
        return None
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    number = float(numeric)
    return number if math.isfinite(number) else None


def apply_numeric_bin(
    value: Any, thresholds: tuple[float, float, float]
) -> str | None:
    """Apply fitted quartile thresholds to one numeric value."""
    numeric = _numeric_value(value)
    if numeric is None:
        return None
    if len(thresholds) != 3:
        raise ValueError("thresholds must contain the 25th, 50th, and 75th percentiles")
    if numeric <= thresholds[0]:
        return "Q1"
    if numeric <= thresholds[1]:
        return "Q2"
    if numeric <= thresholds[2]:
        return "Q3"
    return "Q4"


def _iter_vital_measurements(
    tables: Mapping[str, pd.DataFrame]
) -> Iterable[tuple[PatientStayId, str | None, Any, EventTime]]:
    vital_obs = {}
    valid_vitals = {
        "temperature",
        "sao2",
        "heartrate",
        "respiration",
        "noninvasivesystolic",
        "noninvasivediastolic",
        "noninvasivemean",
    }
    skipped_raw_vitals = []

    for table_name in _VITAL_TABLES:
        frame = tables.get(table_name)
        if frame is None:
            continue
        _require_stay_column(frame, table_name)
        cols = [c for c in frame.columns if c in valid_vitals]
        for _, row in frame.iterrows():
            stay_id = row["patientunitstayid"]
            if pd.isna(stay_id):
                continue
            offset_val = row.get("observationoffset")
            event_time = _event_time(offset_val)
            if event_time is None or event_time < 0 or event_time > 1440:
                continue
            bucket_idx = min(23, int(event_time // 60))
            for col in cols:
                val = row[col]
                num_val = _numeric_value(val)
                if num_val is None:
                    skipped_raw_vitals.append((stay_id, f"VITAL::{normalize_token_text(col)}", val, event_time))
                    continue
                var_name = normalize_token_text(col)
                if var_name is None:
                    continue
                key = (stay_id, var_name, bucket_idx)
                vital_obs.setdefault(key, []).append(num_val)

    for (stay_id, var_name, bucket_idx), vals in vital_obs.items():
        med_val = float(pd.Series(vals, dtype=float).median())
        final_minute = (bucket_idx + 1) * 60
        yield stay_id, f"VITAL::{var_name}", med_val, final_minute

    for stay_id, measurement, val, event_time in skipped_raw_vitals:
        yield stay_id, measurement, val, event_time


def _iter_numeric_measurements(
    tables: Mapping[str, pd.DataFrame],
) -> Iterable[tuple[PatientStayId, str | None, Any, EventTime]]:
    lab = tables.get("lab")
    if lab is not None:
        _require_stay_column(lab, "lab")
        name_column, value_column, offset_column = _LAB_EVENT_FIELDS
        if value_column in lab.columns:
            last_indices = {}
            for idx, row in lab.iterrows():
                stay_id = row["patientunitstayid"]
                if pd.isna(stay_id):
                    continue
                measurement = normalize_token_text(row[name_column]) if name_column in lab.columns else None
                event_time = _event_time(row[offset_column]) if offset_column in lab.columns else None
                if event_time is not None and (event_time < 0 or event_time > 1440):
                    continue
                key = (stay_id, measurement, event_time)
                last_indices[key] = idx

            for idx, row in lab.iterrows():
                stay_id = row["patientunitstayid"]
                if pd.isna(stay_id):
                    continue
                measurement = normalize_token_text(row[name_column]) if name_column in lab.columns else None
                event_time = _event_time(row[offset_column]) if offset_column in lab.columns else None
                if event_time is not None and (event_time < 0 or event_time > 1440):
                    continue
                key = (stay_id, measurement, event_time)
                if last_indices[key] != idx:
                    continue
                yield stay_id, (f"LAB::{measurement}" if measurement is not None else None), row[value_column], event_time

    yield from _iter_vital_measurements(tables)


def compute_quantiles_externally(
    values: Iterable[float], temp_dir: Path
) -> tuple[float, float, float]:
    chunk = []
    run_paths = []
    total_count = 0
    chunk_size = 10000

    for val in values:
        chunk.append(val)
        total_count += 1
        if len(chunk) >= chunk_size:
            chunk.sort()
            run_path = temp_dir / f"run_{len(run_paths)}.txt"
            with open(run_path, "w") as f:
                for v in chunk:
                    f.write(f"{v}\n")
            run_paths.append(run_path)
            chunk = []

    if chunk:
        chunk.sort()
        run_path = temp_dir / f"run_{len(run_paths)}.txt"
        with open(run_path, "w") as f:
            for v in chunk:
                f.write(f"{v}\n")
        run_paths.append(run_path)

    if total_count < 50:
        raise ValueError("At least 50 observations are required")

    files = [open(p, "r") for p in run_paths]

    def gen_file_values(f):
        for line in f:
            yield float(line.strip())

    import heapq
    import math

    merged_iter = heapq.merge(*(gen_file_values(f) for f in files))

    targets = [
        0.25 * (total_count - 1),
        0.50 * (total_count - 1),
        0.75 * (total_count - 1),
    ]
    target_indices = []
    for t in targets:
        target_indices.append(int(math.floor(t)))
        target_indices.append(int(math.ceil(t)))
    unique_target_indices = sorted(list(set(target_indices)))

    saved_values = {}
    current_idx = 0
    target_idx_set = set(unique_target_indices)

    for val in merged_iter:
        if current_idx in target_idx_set:
            saved_values[current_idx] = val
        current_idx += 1

    for f in files:
        f.close()
    for p in run_paths:
        p.unlink()

    results = []
    for t in targets:
        low_idx = int(math.floor(t))
        high_idx = int(math.ceil(t))
        low_val = saved_values[low_idx]
        high_val = saved_values[high_idx]
        if low_idx == high_idx:
            results.append(low_val)
        else:
            results.append(low_val + (high_val - low_val) * (t - low_idx))

    return float(results[0]), float(results[1]), float(results[2])


def fit_numeric_bins(
    tables: Mapping[str, pd.DataFrame],
    train_stay_ids: set[PatientStayId] | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Fit per-measurement quartile thresholds from lab and vital rows."""
    if not isinstance(tables, Mapping):
        raise ValueError("tables must be a mapping of eICU table names to DataFrames")

    values_by_measurement: dict[str, list[float]] = {}
    for stay_id, measurement, value, _ in _iter_numeric_measurements(tables):
        if train_stay_ids is not None and stay_id not in train_stay_ids:
            continue
        numeric = _numeric_value(value)
        if measurement is None or numeric is None:
            continue
        values_by_measurement.setdefault(measurement, []).append(numeric)

    thresholds: dict[str, tuple[float, float, float]] = {}
    
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for measurement in sorted(values_by_measurement):
            vals = values_by_measurement[measurement]
            if len(vals) < 50:
                continue
            try:
                q = compute_quantiles_externally(vals, tmp_path)
                thresholds[measurement] = q
            except ValueError:
                continue
                
    return thresholds


def extract_numeric_events(
    tables: Mapping[str, pd.DataFrame],
    thresholds: Mapping[str, tuple[float, float, float]],
) -> tuple[
    dict[PatientStayId, list[tuple[str, EventTime]]],
    dict[str, int],
]:
    """Discretize lab and vital rows into deterministic patient-level events."""
    if not isinstance(tables, Mapping):
        raise ValueError("tables must be a mapping of eICU table names to DataFrames")
    if not isinstance(thresholds, Mapping):
        raise ValueError("thresholds must be a measurement-to-quantiles mapping")

    grouped: dict[PatientStayId, list[tuple[str, EventTime]]] = {}
    seen: dict[PatientStayId, set[tuple[str, EventTime]]] = {}
    stats = {
        "candidate_values": 0,
        "emitted_events": 0,
        "skipped_missing_measurement": 0,
        "skipped_nonnumeric_value": 0,
        "skipped_unfitted_measurement": 0,
    }

    for stay_id, measurement, value, event_time in _iter_numeric_measurements(tables):
        stats["candidate_values"] += 1
        if measurement is None:
            stats["skipped_missing_measurement"] += 1
            continue
        numeric = _numeric_value(value)
        if numeric is None:
            stats["skipped_nonnumeric_value"] += 1
            continue
        measurement_thresholds = thresholds.get(measurement)
        if measurement_thresholds is None:
            stats["skipped_unfitted_measurement"] += 1
            continue
        label = apply_numeric_bin(numeric, measurement_thresholds)
        token = f"{measurement}::{label}"
        event = (token, event_time)
        stay_seen = seen.setdefault(stay_id, set())
        if event in stay_seen:
            continue
        grouped.setdefault(stay_id, []).append(event)
        stay_seen.add(event)
        stats["emitted_events"] += 1

    for events in grouped.values():
        events.sort(
            key=lambda event: (
                event[1] is None,
                event[1] if event[1] is not None else 0,
                event[0],
            )
        )
    return dict(sorted(grouped.items(), key=lambda item: str(item[0]))), stats


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
        ("unitadmitsource", "UNIT_ADMIT_SOURCE", _unit_admit_source_token),
        ("unittype", "UNIT_TYPE", _unit_type_token),
    )
    events: dict[PatientStayId, list[tuple[str, EventTime]]] = {}
    seen: dict[PatientStayId, set[str]] = {}
    for _, row in patient.iterrows():
        stay_id = row["patientunitstayid"]
        if pd.isna(stay_id):
            continue
        for column, token_name, builder in field_builders:
            if column in patient.columns:
                suffix = builder(row[column])
            else:
                suffix = "UNKNOWN"
            if suffix is None:
                suffix = "UNKNOWN"
            token = f"STATIC::{token_name}::{suffix}"
            if _is_prohibited_token(token):
                continue
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

        for _, row in frame.iterrows():
            stay_id = row["patientunitstayid"]
            if pd.isna(stay_id):
                continue

            event_time = None
            if offset_column in frame.columns:
                event_time = _event_time(row[offset_column])

            if event_time is None or event_time < 0 or event_time > 1440:
                continue

            token = None
            if table_name == "diagnosis":
                if value_column in frame.columns:
                    suffix = normalize_token_text(row[value_column])
                    if suffix is not None and suffix != "UNKNOWN":
                        token = f"DX::{suffix}"
            elif table_name == "medication":
                suffix = None
                if value_column in frame.columns:
                    suffix = normalize_token_text(row[value_column])
                if suffix is None or suffix == "UNKNOWN":
                    hicl = row.get("drughiclseqno")
                    if pd.notna(hicl) and str(hicl).strip() != "":
                        hicl_suffix = normalize_token_text(hicl)
                        if hicl_suffix is not None:
                            token = f"MED::HICL::{hicl_suffix}"
                else:
                    token = f"MED::{suffix}"
            elif table_name == "infusionDrug":
                if value_column in frame.columns:
                    suffix = normalize_token_text(row[value_column])
                    if suffix is not None and suffix != "UNKNOWN":
                        token = f"INFUSION::{suffix}"
            elif table_name == "treatment":
                if value_column in frame.columns:
                    suffix = normalize_token_text(row[value_column])
                    if suffix is not None and suffix != "UNKNOWN":
                        token = f"TREATMENT::{suffix}"

            if token is None or _is_prohibited_token(token):
                continue

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


def extract_outcomes(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    if not isinstance(tables, Mapping):
        raise ValueError("tables must be a mapping of eICU table names to DataFrames")
    if "patient" not in tables:
        raise ValueError("patient table is required for outcome extraction")
    patient_df = tables["patient"]
    required_cols = {"patientunitstayid", "uniquepid", "hospitalid", "hospitaldischargestatus", "unitdischargeoffset"}
    missing_cols = required_cols.difference(patient_df.columns)
    if missing_cols:
        raise ValueError(f"patient table is missing required columns: {', '.join(sorted(missing_cols))}")

    total_stays = 0
    eligible_stays = 0
    exclusion_counts = {
        "missing_id": 0,
        "unrecognised_label": 0,
        "short_stay": 0,
        "conflicting_stay": 0,
    }
    class_counts = {
        "Alive": 0,
        "Expired": 0,
    }

    nan_count = patient_df["patientunitstayid"].isna().sum()
    if nan_count > 0:
        exclusion_counts["missing_id"] += nan_count

    eligible_records = []
    valid_groups = patient_df.dropna(subset=["patientunitstayid"])
    total_stays = len(valid_groups["patientunitstayid"].unique()) + nan_count

    for stay_id, group in valid_groups.groupby("patientunitstayid", sort=False):
        unique_pids = {val for val in group["uniquepid"].dropna() if str(val).strip()}
        unique_hids = {val for val in group["hospitalid"].dropna() if str(val).strip()}

        if not unique_pids or not unique_hids:
            exclusion_counts["missing_id"] += 1
            continue

        if len(unique_pids) > 1 or len(unique_hids) > 1:
            exclusion_counts["conflicting_stay"] += 1
            continue

        resolved_labels = set()
        for val in group["hospitaldischargestatus"]:
            if pd.isna(val):
                continue
            s_val = str(val).strip().casefold()
            if s_val == "alive":
                resolved_labels.add(0)
            elif s_val == "expired":
                resolved_labels.add(1)
            else:
                resolved_labels.add(-1)

        if not resolved_labels:
            exclusion_counts["unrecognised_label"] += 1
            continue

        if len(resolved_labels) > 1:
            exclusion_counts["conflicting_stay"] += 1
            continue

        label = resolved_labels.pop()
        if label == -1:
            exclusion_counts["unrecognised_label"] += 1
            continue

        offsets = []
        invalid_offset = False
        for val in group["unitdischargeoffset"]:
            if pd.isna(val):
                invalid_offset = True
                continue
            try:
                f_val = float(val)
                if math.isnan(f_val) or not math.isfinite(f_val):
                    invalid_offset = True
                else:
                    offsets.append(f_val)
            except (ValueError, TypeError):
                invalid_offset = True

        if invalid_offset or not offsets:
            exclusion_counts["short_stay"] += 1
            continue

        if len(set(offsets)) > 1:
            exclusion_counts["conflicting_stay"] += 1
            continue

        offset = offsets[0]
        if offset < 1440.0:
            exclusion_counts["short_stay"] += 1
            continue

        eligible_records.append({
            "patientunitstayid": stay_id,
            "mortality": label,
        })
        eligible_stays += 1
        if label == 0:
            class_counts["Alive"] += 1
        else:
            class_counts["Expired"] += 1

    outcomes = pd.DataFrame.from_records(
        eligible_records, columns=["patientunitstayid", "mortality"]
    )
    if not outcomes.empty:
        outcomes = outcomes.sort_values("patientunitstayid", kind="stable").reset_index(
            drop=True
        )
        outcomes["mortality"] = outcomes["mortality"].astype(int)

    summary = CohortSummary(
        total_stays=int(total_stays),
        eligible_stays=int(eligible_stays),
        exclusion_counts={k: int(v) for k, v in exclusion_counts.items()},
        class_counts={k: int(v) for k, v in class_counts.items()},
    )
    validate_cohort_summary(summary)
    outcomes.attrs["cohort_summary"] = summary

    outcomes.attrs["outcome_stats"] = {
        "candidate_stays": int(total_stays),
        "labelled_stays": int(eligible_stays),
        "unavailable_labels": int(exclusion_counts["unrecognised_label"] + exclusion_counts["missing_id"] + exclusion_counts["short_stay"]),
        "conflicting_labels": int(exclusion_counts["conflicting_stay"]),
    }
    return outcomes



def event_stream_from_dict(record: Mapping[str, Any]) -> EventStream:
    """Create and validate an event stream from a serialized mapping."""
    if not isinstance(record, Mapping):
        raise ValueError("serialized event stream must be a mapping")
    required = {"patientunitstayid", "events", "representation", "split_name"}
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(f"serialized event stream is missing: {', '.join(missing)}")

    stream = EventStream(
        patientunitstayid=record["patientunitstayid"],
        events=record["events"],
        representation=record["representation"],
        event_times=record.get("event_times"),
        metadata=record.get("metadata", {}),
        split_name=record["split_name"],
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


def _time_gap_token(gap_minutes: float) -> str | None:
    """Return a compact token for a positive interval measured in minutes."""
    if gap_minutes <= 0:
        return None
    if gap_minutes <= 15:
        bucket = "LE_15M"
    elif gap_minutes <= 60:
        bucket = "15M_1H"
    elif gap_minutes <= 180:
        bucket = "1H_3H"
    elif gap_minutes <= 360:
        bucket = "3H_6H"
    elif gap_minutes <= 720:
        bucket = "6H_12H"
    elif gap_minutes <= 1440:
        bucket = "12H_24H"
    else:
        bucket = "GT_24H"
    return f"TIME_GAP::{bucket}"


def _insert_time_gap_events(
    events: list[tuple[str, EventTime]],
) -> list[tuple[str, EventTime]]:
    """Insert gaps between consecutive timed clinical events."""
    with_gaps: list[tuple[str, EventTime]] = []
    previous_time: int | float | None = None
    for token, event_time in events:
        if event_time is not None and previous_time is not None:
            gap_token = _time_gap_token(event_time - previous_time)
            if gap_token is not None:
                with_gaps.append((gap_token, event_time))
        with_gaps.append((token, event_time))
        if event_time is not None:
            previous_time = event_time
    return with_gaps


def _all_stay_ids(tables: Mapping[str, pd.DataFrame]) -> set[PatientStayId]:
    stay_ids: set[PatientStayId] = set()
    for frame in tables.values():
        if isinstance(frame, pd.DataFrame) and "patientunitstayid" in frame.columns:
            stay_ids.update(frame["patientunitstayid"].dropna().tolist())
    return stay_ids


def build_event_streams(
    tables: Mapping[str, pd.DataFrame],
    representation: str,
    min_events_per_stay: int,
) -> tuple[list[EventStream], pd.DataFrame, EventStats]:
    """Build validated event streams, aligned outcomes, and aggregate stats."""
    if not isinstance(tables, Mapping):
        raise ValueError("tables must be a mapping of eICU table names to DataFrames")
    if representation not in EVENT_REPRESENTATIONS:
        choices = ", ".join(EVENT_REPRESENTATIONS)
        raise ValueError(f"representation must be one of: {choices}")
    if (
        isinstance(min_events_per_stay, bool)
        or not isinstance(min_events_per_stay, int)
        or min_events_per_stay < 1
    ):
        raise ValueError("min_events_per_stay must be a positive integer")

    outcomes = extract_outcomes(tables)
    outcome_by_stay = dict(
        zip(outcomes["patientunitstayid"], outcomes["mortality"], strict=True)
    )
    categorical = extract_categorical_events(tables, representation=representation)
    thresholds = fit_numeric_bins(tables)
    numeric, _ = extract_numeric_events(tables, thresholds)
    all_stay_ids = _all_stay_ids(tables) | set(categorical) | set(numeric)

    streams: list[EventStream] = []
    kept_outcomes: list[dict[str, Any]] = []
    family_counts = {family: 0 for family in EVENT_FAMILIES}
    sequence_lengths: list[int] = []

    for stay_id in sorted(all_stay_ids, key=lambda value: str(value)):
        if stay_id not in outcome_by_stay:
            continue

        static_events = [
            event
            for event in categorical.get(stay_id, [])
            if event[0].startswith("STATIC::")
        ]
        clinical_events = [
            event
            for event in categorical.get(stay_id, [])
            if not event[0].startswith("STATIC::")
        ]
        clinical_events.extend(numeric.get(stay_id, []))
        clinical_events = list(dict.fromkeys(clinical_events))
        clinical_events.sort(
            key=lambda event: (
                event[1] is None,
                event[1] if event[1] is not None else 0,
                event[0],
            )
        )
        if representation in {"timegap", "timegap_static"}:
            clinical_events = _insert_time_gap_events(clinical_events)

        combined = [*static_events, *clinical_events]
        if len(combined) < min_events_per_stay:
            continue

        stream = EventStream(
            patientunitstayid=stay_id,
            events=[token for token, _ in combined],
            event_times=[event_time for _, event_time in combined],
            representation=representation,
        )
        validate_event_stream(stream, min_events_per_stay=min_events_per_stay)
        streams.append(stream)
        kept_outcomes.append(
            {
                "patientunitstayid": stay_id,
                "mortality": int(outcome_by_stay[stay_id]),
            }
        )
        sequence_lengths.append(len(stream.events))
        for token in stream.events:
            family_counts[token.partition("::")[0]] += 1

    filtered_outcomes = pd.DataFrame.from_records(
        kept_outcomes, columns=["patientunitstayid", "mortality"]
    )
    if not filtered_outcomes.empty:
        filtered_outcomes["mortality"] = filtered_outcomes["mortality"].astype(int)
    filtered_outcomes.attrs["outcome_stats"] = outcomes.attrs.get("outcome_stats", {})

    stats = EventStats(
        total_stays=len(all_stay_ids),
        kept_stays=len(streams),
        skipped_stays=len(all_stay_ids) - len(streams),
        min_sequence_length=min(sequence_lengths, default=0),
        max_sequence_length=max(sequence_lengths, default=0),
        median_sequence_length=(
            float(pd.Series(sequence_lengths, dtype=float).median())
            if sequence_lengths
            else 0.0
        ),
        token_family_counts=family_counts,
    )
    validate_event_stats(stats)
    return streams, filtered_outcomes, stats
