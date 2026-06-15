from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import gzip
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from icu_pretrain.constants import (
    CONTRACT_SCHEMA_VERSION,
    EVENT_FAMILIES,
)
from icu_pretrain.data.eicu_demo import load_tables, discover_raw_dir, table_path
from icu_pretrain.data.eicu_event_builder import (
    EventStream,
    EventStats,
    SplitRecord,
    FittedPreprocessing,
    CohortSummary,
    StageManifest,
    RunState,
    validate_event_stream,
    validate_event_stats,
    validate_cohort_summary,
    write_stage_manifest_json,
    read_stage_manifest_json,
    write_split_metadata,
    extract_outcomes,
    extract_categorical_events,
    _iter_numeric_measurements,
    fit_numeric_bins,
    apply_numeric_bin,
    _family_priority,
    _insert_time_gap_events,
    _numeric_value,
)
from icu_pretrain.data.splits import assign_patient_splits
from icu_pretrain.utils import load_yaml, validate_final_config


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare local eICU demo CSVs into patient-level artifacts."
    )
    parser.add_argument(
        "--raw_dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--resume",
        choices=["auto", "no"],
        default="auto",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
    )
    parser.add_argument(
        "--restart-stage",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--representation",
        choices=["basic", "timegap", "timegap_static"],
        default=None,
    )
    parser.add_argument(
        "--min_events_per_stay",
        type=_positive_integer,
        default=None,
    )
    return parser


def _get_config_hash(config: dict[str, Any]) -> str:
    serialized = json.dumps(config, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _get_manifest_hash(manifest: StageManifest) -> str:
    return hashlib.sha256(json.dumps(asdict(manifest), sort_keys=True).encode("utf-8")).hexdigest()


def _log(out_dir: Path, stage: str, status: str, message: str, extra: dict[str, Any] | None = None) -> None:
    print(f"[{stage}] {status.upper()}: {message}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = out_dir / "run.log"
    timestamp = datetime.utcnow().isoformat() + "Z"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [{stage}] {status.upper()}: {message}\n")
        
    events_file = out_dir / "events.jsonl"
    event_data = {
        "timestamp": timestamp,
        "stage": stage,
        "status": status,
        "message": message,
    }
    if extra:
        event_data.update(extra)
    with open(events_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event_data) + "\n")


def _write_run_state(out_dir: Path, state: RunState) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "state.json"
    temporary_path = output_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(asdict(state), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def _clear_stage_outputs_single(out_dir: Path, stage_name: str) -> None:
    if stage_name == "discover_inputs":
        pass
    elif stage_name == "build_cohort_and_splits":
        for name in ["cohort_summary.json", "split_metadata.json"]:
            p = out_dir / name
            if p.exists():
                p.unlink()
        p = out_dir / "work" / "build_cohort_and_splits"
        if p.exists():
            shutil.rmtree(p)
    elif stage_name == "extract_partitioned_events":
        p = out_dir / "work" / "extract_partitioned_events"
        if p.exists():
            shutil.rmtree(p)
    elif stage_name == "fit_training_preprocessing":
        p = out_dir / "preprocessing_metadata.json"
        if p.exists():
            p.unlink()
    elif stage_name == "assemble_stream_shards":
        for name in ["event_stats.json", "event_streams.jsonl", "outcomes.csv"]:
            p = out_dir / name
            if p.exists():
                p.unlink()
        p = out_dir / "event_shards"
        if p.exists():
            shutil.rmtree(p)


def _clear_stage_and_downstream(out_dir: Path, stage_name: str) -> None:
    stages_list = [
        "discover_inputs",
        "build_cohort_and_splits",
        "extract_partitioned_events",
        "fit_training_preprocessing",
        "assemble_stream_shards",
    ]
    if stage_name not in stages_list:
        raise ValueError(f"unknown stage name: {stage_name}")
    idx = stages_list.index(stage_name)
    for s in stages_list[idx:]:
        manifest_path = out_dir / "manifests" / s / "manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()
        _clear_stage_outputs_single(out_dir, s)


def _load_default_config() -> dict[str, Any]:
    return {
        "experiment": {
            "name": "eicu_demo_final_tiny",
            "seed": 42,
        },
        "data": {
            "dataset": "eicu_demo",
            "raw_dir": "data/raw/eicu_demo",
            "processed_dir": "data/processed/eicu_demo",
            "representation": "timegap_static",
            "outcome": {
                "name": "hospital_mortality",
                "source_field": "hospitaldischargestatus",
            },
            "observation_window": {
                "start_minutes": 0,
                "end_minutes": 1440,
            },
            "max_seq_len": 256,
            "min_dynamic_events": 5,
        },
        "split": {
            "strategy": "patient_grouped",
            "group_key": "uniquepid",
            "stratify_by": "hospital_mortality",
            "seed": 42,
            "ratios": {
                "train": 0.70,
                "validation": 0.15,
                "test": 0.15,
            },
        },
        "preprocessing": {
            "fit_on": "train",
            "vital_bucket_minutes": 60,
            "lab_min_observations": 50,
            "vital_min_observations": 50,
        },
        "model": {
            "type": "icu_tiny_transformer",
            "max_seq_len": 256,
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.1,
        },
        "pretraining": {
            "enabled": True,
            "objective": "masked_event_modeling",
            "mask_probability": 0.15,
            "epochs": 5,
            "batch_size": 8,
            "gradient_accumulation_steps": 4,
            "learning_rate": 0.0005,
            "weight_decay": 0.01,
        },
        "finetuning": {
            "task": "hospital_mortality",
            "epochs": 10,
            "batch_size": 8,
            "learning_rate": 0.0003,
            "weight_decay": 0.01,
            "early_stopping_patience": 3,
            "selection_metric": "validation_average_precision",
            "freeze_encoder": False,
        },
        "evaluation": {
            "split": "patient_grouped_test",
            "selection_metric": "validation_average_precision",
            "metrics": [
                "auroc",
                "average_precision",
                "f1",
                "balanced_accuracy",
            ],
        },
        "runtime": {
            "device": "cpu",
            "num_workers": 0,
            "seed": 42,
        },
        "data_processing": {
            "csv_chunk_rows": 50000,
            "partition_shards": 64,
            "encoded_shard_stays": 128,
        },
        "recovery": {
            "enabled": True,
            "log_every_batches": 10,
            "checkpoint_every_optimizer_steps": 100,
            "keep_last_checkpoints": 2,
            "resume": "auto",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    
    resume_mode = "no" if args.no_resume else args.resume
    
    config_dict = None
    if args.config:
        try:
            config_dict = load_yaml(args.config)
        except Exception as error:
            parser.error(f"failed to load config file: {error}")
    else:
        root_path = Path(__file__).resolve().parents[1]
        default_config_path = root_path / "configs" / "final" / "eicu_demo_final_tiny.yaml"
        if default_config_path.is_file():
            try:
                config_dict = load_yaml(default_config_path)
            except Exception:
                config_dict = _load_default_config()
        else:
            config_dict = _load_default_config()
            
    try:
        validated_config = validate_final_config(config_dict)
    except ValueError as error:
        parser.error(str(error))

    if args.seed is not None:
        validated_config["runtime"]["seed"] = args.seed
        validated_config["split"]["seed"] = args.seed
        validated_config["experiment"]["seed"] = args.seed

    if args.representation is not None:
        validated_config["data"]["representation"] = args.representation

    if args.min_events_per_stay is not None:
        validated_config["data"]["min_dynamic_events"] = args.min_events_per_stay
        
    try:
        resolved_raw = discover_raw_dir(args.raw_dir)
    except Exception as error:
        err_msg = str(error)
        if "does not exist" in err_msg or "could not discover" in err_msg:
            err_msg = f"raw directory does not exist or is missing required tables: {args.raw_dir} (patient.csv)"
        parser.error(err_msg)

    from icu_pretrain.data.eicu_demo import MVP_TABLES
    found_tables = {}
    for table_name in MVP_TABLES:
        is_optional = (table_name == "apachePatientResult")
        p = table_path(resolved_raw, table_name)
        if p.is_file():
            found_tables[table_name] = p
        elif not is_optional:
            err_msg = f"patient.csv.gz or patient.csv is missing" if table_name == "patient" else f"required eICU demo table file is missing: {p}"
            parser.error(err_msg)

    config_hash = _get_config_hash(validated_config)
    
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    run_id = f"prep_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    state = RunState(
        run_id=run_id,
        status="running",
        updated_at=datetime.utcnow().isoformat() + "Z",
        artifact_hashes={},
    )
    _write_run_state(out_dir, state)
    
    _log(out_dir, "pipeline", "started", f"preparation run started with ID {run_id}")
    
    if args.restart_stage:
        try:
            _clear_stage_and_downstream(out_dir, args.restart_stage)
            _log(out_dir, "pipeline", "info", f"restarted stage {args.restart_stage} and downstream")
        except ValueError as error:
            _log(out_dir, "pipeline", "failed", str(error))
            state.status = "failed"
            state.updated_at = datetime.utcnow().isoformat() + "Z"
            _write_run_state(out_dir, state)
            parser.error(str(error))
            
    if resume_mode == "no":
        _log(out_dir, "pipeline", "info", "resume disabled: clearing all stages")
        _clear_stage_and_downstream(out_dir, "discover_inputs")
        
    stages_manifests = {}
    
    input_hashes = {}
    for t_name, t_path in found_tables.items():
        input_hashes[t_name] = f"size:{t_path.stat().st_size}|mtime:{t_path.stat().st_mtime}"
        
    def check_and_run_stage(
        stage_name: str,
        upstream_names: list[str],
        shard_count: int,
        run_fn,
    ) -> StageManifest:
        manifest_path = out_dir / "manifests" / stage_name / "manifest.json"
        
        upstream_hashes = {}
        for up in upstream_names:
            upstream_hashes[up] = _get_manifest_hash(stages_manifests[up])
            
        if resume_mode == "auto" and manifest_path.is_file():
            try:
                existing = read_stage_manifest_json(manifest_path)
                compat = True
                if existing.config_hash != config_hash:
                    compat = False
                if existing.input_hashes != input_hashes:
                    compat = False
                if existing.upstream_hashes != upstream_hashes:
                    compat = False
                if not compat:
                    raise ValueError(f"stage manifest for {stage_name} is incompatible")
                if existing.status == "complete":
                    _log(out_dir, stage_name, "skipped", f"stage {stage_name} is already complete")
                    stages_manifests[stage_name] = existing
                    return existing
            except Exception as error:
                _log(out_dir, stage_name, "incompatible", f"resume compatibility check failed: {error}")
                state.status = "failed"
                state.updated_at = datetime.utcnow().isoformat() + "Z"
                _write_run_state(out_dir, state)
                raise ValueError(f"incompatible manifest: {error}")
                
        started_at = datetime.utcnow().isoformat() + "Z"
        manifest = StageManifest(
            stage_name=stage_name,
            status="running",
            config_hash=config_hash,
            input_hashes=input_hashes,
            upstream_hashes=upstream_hashes,
            shard_count=shard_count,
            completed_shards=[],
            aggregate_counts={},
            skipped_counts={},
            started_at=started_at,
            updated_at=started_at,
        )
        write_stage_manifest_json(manifest_path, manifest)
        
        try:
            run_fn(manifest, manifest_path)
            manifest.status = "complete"
            manifest.completed_at = datetime.utcnow().isoformat() + "Z"
            manifest.updated_at = manifest.completed_at
            write_stage_manifest_json(manifest_path, manifest)
            stages_manifests[stage_name] = manifest
            _log(out_dir, stage_name, "completed", f"stage {stage_name} completed successfully")
            return manifest
        except Exception as error:
            manifest.status = "failed"
            manifest.failure = {"type": type(error).__name__, "message": str(error)}
            manifest.updated_at = datetime.utcnow().isoformat() + "Z"
            write_stage_manifest_json(manifest_path, manifest)
            _log(out_dir, stage_name, "failed", f"stage {stage_name} failed: {error}")
            raise error

    def run_discover_inputs(manifest: StageManifest, manifest_path: Path):
        manifest.completed_shards = [0]
        manifest.aggregate_counts = {"tables_found": len(found_tables)}
        
    check_and_run_stage("discover_inputs", [], 1, run_discover_inputs)
    
    def run_build_cohort_and_splits(manifest: StageManifest, manifest_path: Path):
        patient_df = load_tables(args.raw_dir, ["patient"])["patient"]
        tables_to_load = ["patient"]
        if "apachePatientResult" in found_tables:
            tables_to_load.append("apachePatientResult")
        tables = load_tables(args.raw_dir, tables_to_load)
        outcomes = extract_outcomes(tables)
        summary = outcomes.attrs["cohort_summary"]
        
        cohort_path = out_dir / "cohort_summary.json"
        cohort_tmp = cohort_path.with_suffix(".tmp")
        cohort_tmp.write_text(
            json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        cohort_tmp.replace(cohort_path)
        
        split_records = assign_patient_splits(outcomes, patient_df, seed=validated_config["split"]["seed"])
        write_split_metadata(out_dir / "split_metadata.json", split_records, processed_root=out_dir)
        
        outcomes_path = out_dir / "outcomes.csv"
        outcomes_tmp = outcomes_path.with_suffix(".tmp")
        outcomes.to_csv(outcomes_tmp, index=False)
        outcomes_tmp.replace(outcomes_path)
        
        work_cohort_dir = out_dir / "work" / "build_cohort_and_splits"
        work_cohort_dir.mkdir(parents=True, exist_ok=True)
        cohort_outcomes_path = work_cohort_dir / "cohort_outcomes.csv"
        cohort_outcomes_tmp = cohort_outcomes_path.with_suffix(".tmp")
        outcomes.to_csv(cohort_outcomes_tmp, index=False)
        cohort_outcomes_tmp.replace(cohort_outcomes_path)
        
        manifest.completed_shards = [0]
        manifest.aggregate_counts = {"eligible_stays": summary.eligible_stays, "total_stays": summary.total_stays}
        manifest.skipped_counts = summary.exclusion_counts

    check_and_run_stage("build_cohort_and_splits", ["discover_inputs"], 1, run_build_cohort_and_splits)
    
    def run_extract_partitioned_events(manifest: StageManifest, manifest_path: Path):
        split_data_path = out_dir / "split_metadata.json"
        with open(split_data_path, "r", encoding="utf-8") as f:
            records_raw = json.load(f)
        stay_ids = {int(r["patientunitstayid"]) for r in records_raw}
        
        tables = load_tables(args.raw_dir)
        representation = validated_config["data"]["representation"]
        categorical = extract_categorical_events(tables, representation=representation)
        
        numeric_raw = {}
        for stay_id, measurement, val, offset in _iter_numeric_measurements(tables):
            s_id = int(stay_id)
            if s_id not in stay_ids:
                continue
            numeric_raw.setdefault(s_id, []).append((measurement, val, offset))
            
        work_dir = out_dir / "work" / "extract_partitioned_events"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        total_extracted = 0
        for shard_id in range(64):
            if shard_id in manifest.completed_shards:
                continue
            shard_stays = {s for s in stay_ids if s % 64 == shard_id}
            shard_data = {}
            for s in shard_stays:
                shard_data[str(s)] = {
                    "categorical": categorical.get(s, []),
                    "numeric": numeric_raw.get(s, []),
                }
            shard_path = work_dir / f"part-{shard_id}.json"
            shard_tmp = shard_path.with_suffix(".tmp")
            with open(shard_tmp, "w", encoding="utf-8") as f:
                json.dump(shard_data, f)
            shard_tmp.replace(shard_path)
            
            manifest.completed_shards.append(shard_id)
            manifest.updated_at = datetime.utcnow().isoformat() + "Z"
            write_stage_manifest_json(manifest_path, manifest)
            total_extracted += len(shard_stays)
            
        manifest.aggregate_counts = {"extracted_stays": len(stay_ids)}

    check_and_run_stage("extract_partitioned_events", ["build_cohort_and_splits"], 64, run_extract_partitioned_events)
    
    def run_fit_training_preprocessing(manifest: StageManifest, manifest_path: Path):
        split_data_path = out_dir / "split_metadata.json"
        with open(split_data_path, "r", encoding="utf-8") as f:
            records_raw = json.load(f)
        train_stays = {int(r["patientunitstayid"]) for r in records_raw if r["split_name"] == "train"}
        
        values_by_measurement = {}
        work_dir = out_dir / "work" / "extract_partitioned_events"
        for shard_id in range(64):
            shard_path = work_dir / f"part-{shard_id}.json"
            with open(shard_path, "r", encoding="utf-8") as f:
                shard_data = json.load(f)
            for stay_id_str, val_dict in shard_data.items():
                s_id = int(stay_id_str)
                if s_id not in train_stays:
                    continue
                for measurement, val, offset in val_dict["numeric"]:
                    num_val = _numeric_value(val)
                    if measurement is None or num_val is None:
                        continue
                    values_by_measurement.setdefault(measurement, []).append(num_val)
                    
        thresholds = {}
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            for measurement in sorted(values_by_measurement):
                vals = values_by_measurement[measurement]
                if len(vals) < 50:
                    continue
                try:
                    from icu_pretrain.data.eicu_event_builder import compute_quantiles_externally
                    q = compute_quantiles_externally(vals, tmp_path)
                    thresholds[measurement] = q
                except ValueError:
                    continue
                    
        preprocessing_metadata = FittedPreprocessing(
            artifact_hash=config_hash,
            fit_split="train",
            numeric_bins={k: list(v) for k, v in thresholds.items()},
            category_maps={},
        )
        
        meta_path = out_dir / "preprocessing_metadata.json"
        meta_tmp = meta_path.with_suffix(".tmp")
        meta_tmp.write_text(
            json.dumps(asdict(preprocessing_metadata), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        meta_tmp.replace(meta_path)
        
        manifest.completed_shards = [0]
        manifest.aggregate_counts = {"fitted_measurements": len(thresholds)}

    check_and_run_stage("fit_training_preprocessing", ["extract_partitioned_events"], 1, run_fit_training_preprocessing)
    
    def run_assemble_stream_shards(manifest: StageManifest, manifest_path: Path):
        meta_path = out_dir / "preprocessing_metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_raw = json.load(f)
        numeric_bins = meta_raw["numeric_bins"]
        
        split_data_path = out_dir / "split_metadata.json"
        with open(split_data_path, "r", encoding="utf-8") as f:
            records_raw = json.load(f)
        stay_splits = {int(r["patientunitstayid"]): r["split_name"] for r in records_raw}
        
        outcomes_path = out_dir / "work" / "build_cohort_and_splits" / "cohort_outcomes.csv"
        outcomes_df = pd.read_csv(outcomes_path)
        outcome_by_stay = dict(zip(outcomes_df["patientunitstayid"], outcomes_df["mortality"]))
        
        representation = validated_config["data"]["representation"]
        min_events_per_stay = validated_config["data"]["min_dynamic_events"]
        
        shards_dir = out_dir / "event_shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        
        work_dir = out_dir / "work" / "extract_partitioned_events"
        
        for shard_id in range(64):
            if shard_id in manifest.completed_shards:
                continue
            shard_data_path = work_dir / f"part-{shard_id}.json"
            with open(shard_data_path, "r", encoding="utf-8") as f:
                shard_data = json.load(f)
                
            shard_streams = []
            for stay_id_str, val_dict in shard_data.items():
                stay_id = int(stay_id_str)
                if stay_id not in outcome_by_stay:
                    continue
                split_name = stay_splits.get(stay_id, "train")
                
                static_events = [
                    (token, offset)
                    for token, offset in val_dict["categorical"]
                    if token.startswith("STATIC::")
                ]
                clinical_events = [
                    (token, offset)
                    for token, offset in val_dict["categorical"]
                    if not token.startswith("STATIC::")
                ]
                
                for measurement, val, offset in val_dict["numeric"]:
                    num_val = _numeric_value(val)
                    if num_val is None or measurement not in numeric_bins:
                        continue
                    bin_label = apply_numeric_bin(num_val, numeric_bins[measurement])
                    clinical_events.append((f"{measurement}::{bin_label}", offset))
                    
                clinical_events = list(dict.fromkeys(clinical_events))
                clinical_events.sort(
                    key=lambda event: (
                        event[1] is None,
                        event[1] if event[1] is not None else 0,
                        _family_priority(event[0]),
                        event[0],
                    )
                )
                
                original_counts = {family: 0 for family in EVENT_FAMILIES}
                for token, _ in static_events:
                    original_counts[token.partition("::")[0]] += 1
                for token, _ in clinical_events:
                    original_counts[token.partition("::")[0]] += 1
                    
                if len(clinical_events) < 5:
                    continue
                    
                def build_seq(retained_clin):
                    if representation in {"timegap", "timegap_static"}:
                        gapped = _insert_time_gap_events(retained_clin)
                        return [*static_events, *gapped]
                    return [*static_events, *retained_clin]
                    
                full_seq = build_seq(clinical_events)
                if len(full_seq) <= 256:
                    retained_combined = full_seq
                else:
                    first_dyn = clinical_events[0]
                    last_dyn = clinical_events[-1]
                    middle_dyn = clinical_events[1:-1]
                    low = 0
                    high = len(middle_dyn)
                    best_seq = None
                    while low <= high:
                        mid = (low + high) // 2
                        if mid == 0:
                            sampled = []
                        elif mid == 1:
                            sampled = [middle_dyn[len(middle_dyn) // 2]]
                        else:
                            L = len(middle_dyn)
                            indices = [int(i * L / mid) for i in range(mid)]
                            sampled = [middle_dyn[idx] for idx in indices]
                        retained = [first_dyn] + sampled + [last_dyn]
                        candidate_seq = build_seq(retained)
                        if len(candidate_seq) <= 256:
                            best_seq = candidate_seq
                            low = mid + 1
                        else:
                            high = mid - 1
                    if best_seq is not None:
                        retained_combined = best_seq
                    else:
                        candidate_seq = build_seq([first_dyn, last_dyn])
                        retained_combined = candidate_seq[:256]
                        
                dyn_count = sum(1 for token, _ in retained_combined if not token.startswith("STATIC::") and not token.startswith("TIME_GAP::"))
                if dyn_count < 5:
                    continue
                if len(retained_combined) < min_events_per_stay:
                    continue
                    
                retained_counts = {family: 0 for family in EVENT_FAMILIES}
                for token, _ in retained_combined:
                    retained_counts[token.partition("::")[0]] += 1
                    
                stream = EventStream(
                    patientunitstayid=stay_id,
                    events=[token for token, _ in retained_combined],
                    event_times=[event_time for _, event_time in retained_combined],
                    representation=representation,
                    metadata={
                        "original_counts": original_counts,
                        "retained_counts": retained_counts,
                    },
                    split_name=split_name,
                )
                validate_event_stream(stream, min_events_per_stay=min_events_per_stay)
                shard_streams.append(stream)
                
            shard_path = shards_dir / f"part-{shard_id}.jsonl.gz"
            shard_tmp = shard_path.with_suffix(".tmp")
            with gzip.open(shard_tmp, "wt", encoding="utf-8") as f:
                for stream in shard_streams:
                    f.write(json.dumps(asdict(stream), sort_keys=True) + "\n")
            shard_tmp.replace(shard_path)
            
            manifest.completed_shards.append(shard_id)
            manifest.updated_at = datetime.utcnow().isoformat() + "Z"
            write_stage_manifest_json(manifest_path, manifest)
            
        final_streams = []
        sequence_lengths = []
        family_counts = {family: 0 for family in EVENT_FAMILIES}
        for shard_id in range(64):
            shard_path = shards_dir / f"part-{shard_id}.jsonl.gz"
            with gzip.open(shard_path, "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    from icu_pretrain.data.eicu_event_builder import event_stream_from_dict
                    stream = event_stream_from_dict(record)
                    final_streams.append(stream)
                    sequence_lengths.append(len(stream.events))
                    for token in stream.events:
                        family_counts[token.partition("::")[0]] += 1
                        
        event_streams_path = out_dir / "event_streams.jsonl"
        event_streams_tmp = event_streams_path.with_suffix(".tmp")
        with open(event_streams_tmp, "w", encoding="utf-8") as f:
            for stream in final_streams:
                f.write(json.dumps(asdict(stream), sort_keys=True) + "\n")
        event_streams_tmp.replace(event_streams_path)
        
        all_stay_ids = set(stay_splits.keys())
        stats = EventStats(
            total_stays=len(all_stay_ids),
            kept_stays=len(final_streams),
            skipped_stays=len(all_stay_ids) - len(final_streams),
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
        
        stats_path = out_dir / "event_stats.json"
        stats_tmp = stats_path.with_suffix(".tmp")
        stats_tmp.write_text(
            json.dumps(asdict(stats), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stats_tmp.replace(stats_path)
        
        kept_stay_ids = {stream.patientunitstayid for stream in final_streams}
        final_outcomes_df = outcomes_df[outcomes_df["patientunitstayid"].isin(kept_stay_ids)]
        final_outcomes_df = final_outcomes_df.sort_values("patientunitstayid", kind="stable").reset_index(drop=True)
        final_outcomes_df["mortality"] = final_outcomes_df["mortality"].astype(int)
        
        outcomes_path = out_dir / "outcomes.csv"
        outcomes_tmp = outcomes_path.with_suffix(".tmp")
        final_outcomes_df.to_csv(outcomes_tmp, index=False)
        outcomes_tmp.replace(outcomes_path)
        
        shards_index = {
            "shard_count": 64,
            "shards": [f"part-{i}.jsonl.gz" for i in range(64)],
        }
        shards_index_path = shards_dir / "index.json"
        shards_index_tmp = shards_index_path.with_suffix(".tmp")
        shards_index_tmp.write_text(
            json.dumps(shards_index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shards_index_tmp.replace(shards_index_path)
        
        manifest.aggregate_counts = {
            "total_stays": stats.total_stays,
            "kept_stays": stats.kept_stays,
            "skipped_stays": stats.skipped_stays,
        }

    check_and_run_stage("assemble_stream_shards", ["fit_training_preprocessing"], 64, run_assemble_stream_shards)
    
    _log(out_dir, "pipeline", "completed", "preparation pipeline completed successfully")
    state.status = "completed"
    state.updated_at = datetime.utcnow().isoformat() + "Z"
    _write_run_state(out_dir, state)
    
    
    final_stats_path = out_dir / "event_stats.json"
    with open(final_stats_path, "r", encoding="utf-8") as f:
        final_stats = json.load(f)
        
    print(f"Prepared {final_stats['kept_stays']} event stream(s); skipped {final_stats['skipped_stays']} stay(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
