import json
import random
import numpy as np
import pytest
import torch
from pathlib import Path
from icu_pretrain.training.finetune import train_finetuning_model, TrainingInterruptedException
from icu_pretrain.training.pretrain import train_model
from icu_pretrain.data.dataset import EncodedDataset

def create_synthetic_processed_dir_ft(tmp_path: Path) -> Path:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    vocab = {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2, "[CLS]": 3, "TOKEN_A": 4, "TOKEN_B": 5, "TOKEN_C": 6, "STATIC::AGE": 7}
    vocab_path = processed_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    stages = ["fit_vocabulary", "build_cohort_and_splits", "fit_training_preprocessing", "encode_split_shards"]
    for s in stages:
        m_dir = processed_dir / "manifests" / s
        m_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "stage_name": s,
            "status": "completed",
            "config_hash": f"hash_{s}",
            "input_hashes": {},
            "upstream_hashes": {},
            "shard_count": 1,
            "completed_shards": [0],
            "aggregate_counts": {},
            "skipped_counts": {},
            "started_at": "2026-06-15T00:00:00Z",
            "updated_at": "2026-06-15T00:00:00Z",
            "completed_at": "2026-06-15T00:00:00Z",
        }
        with open(m_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)
    
    split_metadata = []
    encoded_dir = processed_dir / "encoded"
    for split in ["train", "validation", "test"]:
        split_dir = encoded_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        stays = []
        for i in range(16):
            stay_id = f"100{i}"
            patient_id = f"P100{i}"
            stays.append({
                "patientunitstayid": stay_id,
                "tokens": [3, 7, 4, 5, 6, 4, 5],
                "label": i % 2,
                "split_name": split,
            })
            split_metadata.append({
                "patientunitstayid": stay_id,
                "uniquepid": patient_id,
                "hospitalid": "H1",
                "split_name": split
            })
        EncodedDataset.write_shard(stays, 0, split_dir)
    with open(processed_dir / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(split_metadata, f)
    return processed_dir

def get_default_config_ft(processed_dir: Path) -> dict:
    return {
        "experiment": {
            "id": "EXP-01",
            "name": "test_finetune_run",
            "seed": 42,
        },
        "data": {
            "dataset": "eicu_demo",
            "raw_dir": "data/raw",
            "processed_dir": str(processed_dir),
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
            "d_model": 16,
            "n_heads": 2,
            "n_layers": 1,
            "dim_feedforward": 32,
            "dropout": 0.0,
        },
        "pretraining": {
            "enabled": True,
            "objective": "masked_event_modeling",
            "mask_probability": 0.15,
            "epochs": 2,
            "batch_size": 4,
            "gradient_accumulation_steps": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
        },
        "finetuning": {
            "task": "hospital_mortality",
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "early_stopping_patience": 3,
            "selection_metric": "validation_average_precision",
            "freeze_encoder": False,
        },
        "evaluation": {
            "split": "patient_grouped_test",
            "selection_metric": "validation_average_precision",
            "metrics": ["auroc", "average_precision", "f1", "balanced_accuracy"],
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
            "log_every_batches": 1,
            "checkpoint_every_optimizer_steps": 2,
            "keep_last_checkpoints": 2,
            "resume": "auto",
        },
    }

def test_finetuning_run_uninterrupted(tmp_path: Path):
    processed_dir = create_synthetic_processed_dir_ft(tmp_path)
    config = get_default_config_ft(processed_dir)
    run_dir = tmp_path / "run_uninterrupted"
    results = train_finetuning_model(config, processed_dir, run_dir, resume="no")
    assert "model_state" in results
    assert "prediction_head_state" in results
    assert "val_loss" in results
    assert (run_dir / "state.json").exists()
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "run.log").exists()
    assert (run_dir / "results.json").exists()
    
    with open(run_dir / "results.json", "r", encoding="utf-8") as f:
        res_data = json.load(f)
    assert res_data["experiment_id"] == "EXP-01"
    assert "auroc" in res_data
    assert "auroc_ci" in res_data
    assert "average_precision" in res_data
    assert "average_precision_ci" in res_data
    assert "f1" in res_data
    assert "balanced_accuracy" in res_data
    assert res_data["parameter_count"] > 0
    assert res_data["runtime"] >= 0.0

def test_resumable_finetuning_equivalence(tmp_path: Path):
    processed_dir = create_synthetic_processed_dir_ft(tmp_path)
    config = get_default_config_ft(processed_dir)
    run_dir_uninterrupted = tmp_path / "run_uninterrupted"
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    res_uninterrupted = train_finetuning_model(config, processed_dir, run_dir_uninterrupted, resume="no")
    run_dir_interrupted = tmp_path / "run_interrupted"
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    with pytest.raises(TrainingInterruptedException):
        train_finetuning_model(config, processed_dir, run_dir_interrupted, resume="no", interrupt_after_batches=2)
    with open(run_dir_interrupted / "state.json", "r", encoding="utf-8") as f:
        state_data = json.load(f)
    assert state_data["status"] == "interrupted"
    res_resumed = train_finetuning_model(config, processed_dir, run_dir_interrupted, resume="auto")
    assert abs(res_uninterrupted["val_loss"] - res_resumed["val_loss"]) < 1e-6
    for k in res_uninterrupted["model_state"]:
        diff = torch.max(torch.abs(res_uninterrupted["model_state"][k] - res_resumed["model_state"][k]))
        assert diff.item() < 1e-6

def test_finetuning_run_with_pretrained_checkpoint(tmp_path: Path):
    processed_dir = create_synthetic_processed_dir_ft(tmp_path)
    config = get_default_config_ft(processed_dir)
    pretrain_run_dir = tmp_path / "pretrain_run"
    train_model(config, processed_dir, pretrain_run_dir, resume="no")
    best_pt_checkpoint = pretrain_run_dir / "checkpoints" / "best.pt"
    run_dir = tmp_path / "finetune_run"
    results = train_finetuning_model(config, processed_dir, run_dir, resume="no", pretrain_checkpoint=best_pt_checkpoint)
    assert results is not None
    assert (run_dir / "results.json").exists()

def test_finetuning_nan_loss(tmp_path: Path, monkeypatch):
    processed_dir = create_synthetic_processed_dir_ft(tmp_path)
    config = get_default_config_ft(processed_dir)
    run_dir = tmp_path / "run_nan"
    import icu_pretrain.training.finetune as finetune_mod
    monkeypatch.setattr(finetune_mod, "compute_mortality_loss", lambda *args, **kwargs: torch.tensor(float("nan"), requires_grad=True))
    with pytest.raises(ValueError, match="Loss is NaN"):
        train_finetuning_model(config, processed_dir, run_dir, resume="no")


def test_incompatible_checkpoint(tmp_path: Path):
    processed_dir = create_synthetic_processed_dir_ft(tmp_path)
    config = get_default_config_ft(processed_dir)
    run_dir = tmp_path / "run_incompatible"
    with pytest.raises(TrainingInterruptedException):
        train_finetuning_model(config, processed_dir, run_dir, resume="no", interrupt_after_batches=1)
    manifest_path = processed_dir / "manifests" / "fit_vocabulary" / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["config_hash"] = "changed_hash"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with pytest.raises(ValueError, match="checkpoint artifact hashes are incompatible"):
        train_finetuning_model(config, processed_dir, run_dir, resume="auto")
