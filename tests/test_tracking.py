import json
import csv
import numpy as np
import pytest
import torch
from pathlib import Path
from icu_pretrain.experiments.tracking import (
    validate_public_safe,
    log_event,
    write_run_state,
    recover_truncated_jsonl,
    record_metrics,
    save_best_config,
)

def test_validate_public_safe():
    safe_data = {
        "experiment_id": "EXP-01",
        "representation": "timegap_static",
        "metrics": {"auroc": 0.85},
    }
    validate_public_safe(safe_data)

    unsafe_keys = ["uniquepid", "patientunitstayid", "tokens", "tensor", "patient_id", "stay_id"]
    for k in unsafe_keys:
        bad_data = {k: "some_val"}
        with pytest.raises(ValueError):
            validate_public_safe(bad_data)
            
        bad_nested = {"nested": {k: "some_val"}}
        with pytest.raises(ValueError):
            validate_public_safe(bad_nested)

    with pytest.raises(ValueError):
        validate_public_safe(torch.tensor([1, 2, 3]))

    with pytest.raises(ValueError):
        validate_public_safe(np.array([1, 2, 3]))

def test_log_event(tmp_path: Path):
    run_dir = tmp_path / "run"
    event = {
        "timestamp": "2026-06-15T00:00:00Z",
        "stage": "pretrain",
        "status": "running",
        "epoch": 1,
        "batch": 10,
        "loss": 0.5432,
    }
    log_event(run_dir, event)
    
    events_file = run_dir / "events.jsonl"
    run_log = run_dir / "run.log"
    assert events_file.exists()
    assert run_log.exists()
    
    with open(events_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == event
    
    with open(run_log, "r", encoding="utf-8") as f:
        log_content = f.read()
    assert "Stage: pretrain" in log_content
    assert "Epoch: 1" in log_content
    assert "Loss: 0.5432" in log_content

    bad_event = {"uniquepid": "123"}
    with pytest.raises(ValueError):
        log_event(run_dir, bad_event)

def test_write_run_state(tmp_path: Path):
    run_dir = tmp_path / "run"
    class DummyState:
        run_id = "run-01"
        status = "running"
        updated_at = "2026-06-15T00:00:00Z"
        artifact_hashes = {"config": "hash"}
        last_checkpoint = "checkpoint.pt"
    
    write_run_state(run_dir, DummyState())
    state_file = run_dir / "state.json"
    assert state_file.exists()
    with open(state_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["run_id"] == "run-01"
    assert data["status"] == "running"

def test_recover_truncated_jsonl(tmp_path: Path):
    events_file = tmp_path / "events.jsonl"
    lines = [
        '{"timestamp": "2026-06-15T00:00:00Z", "stage": "pretrain"}\n',
        '{"timestamp": "2026-06-15T00:01:00Z", "stage":\n'
    ]
    with open(events_file, "w", encoding="utf-8") as f:
        f.writelines(lines)
        
    recover_truncated_jsonl(events_file)
    
    with open(events_file, "r", encoding="utf-8") as f:
        recovered = f.readlines()
    assert len(recovered) == 1
    assert json.loads(recovered[0])["timestamp"] == "2026-06-15T00:00:00Z"

def test_record_metrics(tmp_path: Path):
    summary_dir = tmp_path / "summary"
    run_data_1 = {
        "experiment_id": "EXP-01",
        "representation": "timegap_static",
        "num_patients": 100,
        "num_stays": 120,
        "alive_count": 100,
        "expired_count": 20,
        "split_strategy": "patient_grouped",
        "seed": 42,
        "auroc": 0.85,
        "auroc_ci_lower": 0.80,
        "auroc_ci_upper": 0.90,
        "average_precision": 0.75,
        "average_precision_ci_lower": 0.70,
        "average_precision_ci_upper": 0.80,
        "f1": 0.65,
        "balanced_accuracy": 0.72,
        "parameter_count": 1000,
        "runtime": 12.5,
        "exclusions": "none",
        "failure_notes": "",
    }
    
    record_metrics(summary_dir, run_data_1)
    
    comp_csv = summary_dir / "experiment_comparison.csv"
    final_csv = summary_dir / "final_results.csv"
    assert comp_csv.exists()
    assert final_csv.exists()
    
    with open(comp_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "EXP-01"
    assert float(rows[0]["auroc"]) == 0.85

    run_data_duplicate = run_data_1.copy()
    run_data_duplicate["auroc"] = 0.88
    record_metrics(summary_dir, run_data_duplicate)
    
    with open(comp_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert float(rows[0]["auroc"]) == 0.88

    run_data_2 = run_data_1.copy()
    run_data_2["experiment_id"] = "EXP-02"
    record_metrics(summary_dir, run_data_2)
    
    with open(comp_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2
    assert rows[1]["experiment_id"] == "EXP-02"

    run_data_nan = run_data_1.copy()
    run_data_nan["experiment_id"] = "EXP-03"
    run_data_nan["auroc"] = float("nan")
    record_metrics(summary_dir, run_data_nan)
    
    with open(comp_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 3
    assert rows[2]["auroc"] == ""

def test_save_best_config(tmp_path: Path):
    summary_dir = tmp_path / "summary"
    config = {"model": {"d_model": 16}}
    save_best_config(summary_dir, config)
    best_config_file = summary_dir / "best_config.json"
    assert best_config_file.exists()
    with open(best_config_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["model"]["d_model"] == 16
