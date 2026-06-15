"""Tests for ICU event-stream data and serialization contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from icu_pretrain.constants import EVENT_FAMILIES
from icu_pretrain.data.eicu_event_builder import (
    CheckpointContract,
    CohortSummary,
    EligibilityDecision,
    EventStats,
    EventStream,
    FittedPreprocessing,
    ICUStayRecord,
    OutcomeRecord,
    PublicAggregateSummary,
    RunState,
    SplitRecord,
    StageManifest,
    event_stream_from_dict,
    read_event_streams_jsonl,
    read_stage_manifest_json,
    serialize_model_input,
    validate_checkpoint_contract,
    validate_cohort_summary,
    validate_eligibility_decision,
    validate_event_stats,
    validate_event_stream,
    validate_fitted_preprocessing,
    validate_icu_stay_record,
    validate_outcome_record,
    validate_outcomes_for_eligible_stays,
    validate_public_summary,
    validate_run_state,
    validate_split_record,
    validate_stage_manifest,
    write_split_metadata,
    write_stage_manifest_json,
    write_event_streams_jsonl,
)


ARTIFACT_HASHES = {
    "config": "hash-config",
    "vocabulary": "hash-vocab",
    "split": "hash-split",
    "preprocessing": "hash-preprocessing",
    "encoded_dataset": "hash-encoded",
}


def make_stream(**overrides: object) -> EventStream:
    values = {
        "patientunitstayid": "stay-1",
        "events": ["STATIC::GENDER::F", "DX::SEPSIS", "LAB::CREATININE::Q2"],
        "event_times": [None, 10.0, 10.0],
        "representation": "timegap_static",
        "metadata": {"source": "synthetic"},
        "split_name": "train",
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


def test_event_stream_serialized_record_requires_split_name() -> None:
    record = {
        "patientunitstayid": "stay-1",
        "events": ["DX::SEPSIS"],
        "event_times": [5],
        "representation": "basic",
    }

    with pytest.raises(ValueError, match="split_name"):
        event_stream_from_dict(record)


def test_event_stream_accepts_duplicate_timestamps() -> None:
    assert validate_event_stream(make_stream()) == make_stream()


def test_event_stream_allows_explicitly_empty_dynamic_contract() -> None:
    stream = make_stream(events=[], event_times=[])

    assert validate_event_stream(stream, min_events_per_stay=0) == stream


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


def test_event_stream_rejects_nonfinite_timestamps_and_unknown_split() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_event_stream(make_stream(event_times=[None, float("nan"), 10]))

    with pytest.raises(ValueError, match="split_name"):
        validate_event_stream(make_stream(split_name="dev"))


def test_model_input_serialization_excludes_local_identifiers() -> None:
    serialized = serialize_model_input(make_stream(metadata={"source": "synthetic"}))

    assert "patientunitstayid" not in serialized
    assert "uniquepid" not in serialized
    assert "hospitalid" not in serialized
    assert serialized["split_name"] == "train"

    with pytest.raises(ValueError, match="patient-level"):
        serialize_model_input(make_stream(metadata={"hospitalid": "local-group"}))

    with pytest.raises(ValueError, match="grouping identifier"):
        serialize_model_input(
            make_stream(events=["STATIC::HOSPITAL_ID::10"], event_times=[0])
        )


def test_outcome_record_requires_binary_mortality() -> None:
    assert validate_outcome_record(OutcomeRecord("stay-1", 1)) == OutcomeRecord(
        "stay-1", 1
    )

    with pytest.raises(ValueError, match="mortality"):
        validate_outcome_record(OutcomeRecord("stay-1", None))

    with pytest.raises(ValueError, match="mortality"):
        validate_outcome_record(OutcomeRecord("stay-1", 2))


def test_each_eligible_stay_has_exactly_one_outcome() -> None:
    records = [OutcomeRecord("stay-1", 0), OutcomeRecord("stay-2", 1)]

    assert (
        validate_outcomes_for_eligible_stays(records, ["stay-1", "stay-2"])
        == records
    )

    with pytest.raises(ValueError, match="exactly one"):
        validate_outcomes_for_eligible_stays(
            [OutcomeRecord("stay-1", 0), OutcomeRecord("stay-1", 1)],
            ["stay-1", "stay-2"],
        )


def test_icu_stay_and_eligibility_contracts_are_validated() -> None:
    stay = ICUStayRecord(
        patientunitstayid="stay-1",
        uniquepid="patient-group-1",
        hospitalid="hospital-group-1",
        mortality=1,
    )
    assert validate_icu_stay_record(stay) == stay

    eligible = EligibilityDecision(patientunitstayid="stay-1", eligible=True)
    ineligible = EligibilityDecision(
        patientunitstayid="stay-2",
        eligible=False,
        exclusion_reasons=["missing_label"],
    )
    assert validate_eligibility_decision(eligible) == eligible
    assert validate_eligibility_decision(ineligible) == ineligible

    with pytest.raises(ValueError, match="uniquepid"):
        validate_icu_stay_record(
            ICUStayRecord("stay-1", "", "hospital-group-1", 0)
        )
    with pytest.raises(ValueError, match="exclusion_reasons"):
        validate_eligibility_decision(
            EligibilityDecision("stay-2", eligible=False)
        )


def test_split_metadata_keeps_grouping_ids_local_to_processed_root(
    tmp_path: Path,
) -> None:
    record = SplitRecord(
        patientunitstayid="stay-1",
        uniquepid="patient-group-1",
        hospitalid="hospital-group-1",
        split_name="validation",
    )
    assert validate_split_record(record) == record

    processed_root = tmp_path / "data" / "processed"
    output = processed_root / "eicu_demo" / "split_metadata.json"
    write_split_metadata(output, [record], processed_root=processed_root)

    assert "patient-group-1" in output.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="processed_root"):
        write_split_metadata(
            tmp_path / "public_split.json",
            [record],
            processed_root=processed_root,
        )

    with pytest.raises(ValueError, match="split_name"):
        validate_split_record(
            SplitRecord("stay-1", "patient", "hospital", "holdout")
        )


def test_fitted_preprocessing_requires_training_only_metadata() -> None:
    fitted = FittedPreprocessing(
        artifact_hash="hash-preprocessing",
        fit_split="train",
        numeric_bins={"LAB::SODIUM": [135.0, 140.0, 145.0]},
        category_maps={"gender": {"Female": "F"}},
    )
    assert validate_fitted_preprocessing(fitted) == fitted

    with pytest.raises(ValueError, match="fit_split"):
        validate_fitted_preprocessing(
            FittedPreprocessing("hash-preprocessing", fit_split="validation")
        )

    with pytest.raises(ValueError, match="finite"):
        validate_fitted_preprocessing(
            FittedPreprocessing("hash", numeric_bins={"LAB::X": [1.0, float("inf")]})
        )


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


def test_cohort_summary_contains_rule_exclusions_and_class_counts() -> None:
    summary = CohortSummary(
        total_stays=4,
        eligible_stays=2,
        exclusion_counts={"missing_label": 1, "short_stay": 1},
        class_counts={"Alive": 1, "Expired": 1},
    )

    assert validate_cohort_summary(summary) == summary

    with pytest.raises(ValueError, match="class_counts"):
        validate_cohort_summary(
            CohortSummary(4, 2, {"missing_label": 1}, {"Alive": 2})
        )


def test_public_aggregate_summary_rejects_patient_level_fields() -> None:
    summary = PublicAggregateSummary(
        {
            "experiment_id": "EXP-02",
            "aggregate_counts": {"stays": 12, "expired": 3},
            "metrics": {"average_precision": 0.42},
        }
    )
    assert validate_public_summary(summary) == summary

    for key in ("patientunitstayid", "uniquepid", "hospitalid", "events"):
        with pytest.raises(ValueError, match="patient-level"):
            validate_public_summary(PublicAggregateSummary({key: ["not-public"]}))


def test_stage_manifest_records_restartable_stage_state(tmp_path: Path) -> None:
    manifest = StageManifest(
        stage_name="assemble_stream_shards",
        status="complete",
        config_hash="hash-config",
        input_hashes={"patient.csv.gz": "hash-patient"},
        upstream_hashes={"split": "hash-split"},
        shard_count=2,
        completed_shards=[0, 1],
        aggregate_counts={"rows": 10, "stays": 3},
        skipped_counts={"missing_value": 2},
        started_at="2026-06-07T00:00:00Z",
        updated_at="2026-06-07T00:01:00Z",
        completed_at="2026-06-07T00:01:00Z",
    )
    assert validate_stage_manifest(manifest) == manifest
    manifest_path = tmp_path / "manifest.json"
    write_stage_manifest_json(manifest_path, manifest)
    assert read_stage_manifest_json(manifest_path) == manifest

    failed = StageManifest(
        stage_name="fit_vocabulary",
        status="failed",
        config_hash="hash-config",
        input_hashes={},
        upstream_hashes={"streams": "hash-streams"},
        shard_count=1,
        completed_shards=[],
        aggregate_counts={},
        skipped_counts={},
        started_at="2026-06-07T00:00:00Z",
        updated_at="2026-06-07T00:01:00Z",
        failure={"type": "ValueError", "message": "synthetic aggregate failure"},
    )
    assert validate_stage_manifest(failed) == failed

    with pytest.raises(ValueError, match="completed shard"):
        validate_stage_manifest(
            StageManifest(
                stage_name="assemble_stream_shards",
                status="complete",
                config_hash="hash-config",
                input_hashes={},
                upstream_hashes={},
                shard_count=2,
                completed_shards=[0],
                aggregate_counts={},
                skipped_counts={},
                started_at="2026-06-07T00:00:00Z",
                updated_at="2026-06-07T00:01:00Z",
                completed_at="2026-06-07T00:01:00Z",
            )
        )


def test_run_state_and_checkpoint_contract_require_artifact_compatibility() -> None:
    state = RunState(
        run_id="run-synthetic",
        status="running",
        updated_at="2026-06-07T00:00:00Z",
        artifact_hashes=ARTIFACT_HASHES,
        last_checkpoint="checkpoints/step-100.pt",
    )
    assert validate_run_state(state) == state

    checkpoint = make_checkpoint()
    assert (
        validate_checkpoint_contract(
            checkpoint,
            expected_artifact_hashes=ARTIFACT_HASHES,
        )
        == checkpoint
    )

    incompatible = dict(ARTIFACT_HASHES)
    incompatible["split"] = "other"
    with pytest.raises(ValueError, match="incompatible"):
        validate_checkpoint_contract(
            checkpoint,
            expected_artifact_hashes=incompatible,
        )

    with pytest.raises(ValueError, match="artifact"):
        validate_run_state(
            RunState(
                "run-synthetic",
                "running",
                "2026-06-07T00:00:00Z",
                {"config": "only-one-hash"},
            )
        )


def make_checkpoint(**overrides: object) -> CheckpointContract:
    values = {
        "run_id": "run-synthetic",
        "model_state": {"encoder.weight": [0.0]},
        "prediction_head_state": {"head.weight": [0.0]},
        "optimizer_state": {"state": {}},
        "scheduler_state": {"last_epoch": 0},
        "gradient_state": {"micro_step": 0},
        "accumulation_step": 0,
        "epoch": 1,
        "next_batch_cursor": 2,
        "global_batch": 12,
        "optimizer_step": 3,
        "best_metric": 0.25,
        "best_epoch": 1,
        "early_stopping_state": {"bad_epochs": 0},
        "threshold_state": {"threshold": 0.5},
        "rng_state": {"python": "state", "numpy": "state", "torch": "state"},
        "sampler_state": {
            "permutation": [0, 1],
            "generator_state": "state",
            "cursor": 2,
        },
        "artifact_hashes": ARTIFACT_HASHES,
        "training_history": {"loss": [1.0]},
        "creation_reason": "periodic",
    }
    values.update(overrides)
    return CheckpointContract(**values)
