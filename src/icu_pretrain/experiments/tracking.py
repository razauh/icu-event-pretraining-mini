import json
import csv
import numpy as np
from pathlib import Path
from typing import Any

def validate_public_safe(data: Any) -> None:
    import torch
    forbidden_keys = {
        "uniquepid", "patientunitstayid", "tokens", "tensor", 
        "patient_id", "stay_id", "event_stream", "events", 
        "patient_ids", "stay_ids"
    }
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(k, str) and k.lower() in forbidden_keys:
                raise ValueError(f"Forbidden patient-level key: {k}")
            validate_public_safe(v)
    elif isinstance(data, (list, tuple, set)):
        for item in data:
            validate_public_safe(item)
    elif isinstance(data, torch.Tensor):
        raise ValueError("Tensors not allowed in tracking data")
    elif isinstance(data, np.ndarray):
        raise ValueError("Numpy arrays not allowed in tracking data")
    elif hasattr(data, "__dict__"):
        validate_public_safe(data.__dict__)

def log_event(run_dir: Path, event_data: dict[str, Any], stage: str = "pretrain") -> None:
    validate_public_safe(event_data)
    run_dir.mkdir(parents=True, exist_ok=True)
    events_file = run_dir / "events.jsonl"
    with open(events_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event_data) + "\n")
        f.flush()
    log_file = run_dir / "run.log"
    timestamp = event_data.get("timestamp", "")
    msg = f"[{timestamp}] Stage: {event_data.get('stage', stage)} | Status: {event_data.get('status', '')}"
    if "epoch" in event_data:
        msg += f" | Epoch: {event_data['epoch']}"
    if "batch" in event_data:
        msg += f" | Batch: {event_data['batch']}"
    if "loss" in event_data:
        msg += f" | Loss: {event_data['loss']:.4f}"
    if "val_loss" in event_data:
        msg += f" | Val Loss: {event_data['val_loss']:.4f}"
    if "checkpoint_path" in event_data:
        msg += f" | Checkpoint: {event_data['checkpoint_path']}"
    if "val_ap" in event_data:
        msg += f" | Val AP: {event_data['val_ap']:.4f}"
    if "error_type" in event_data:
        msg += f" | Error Type: {event_data['error_type']}"
    if "traceback" in event_data:
        msg += f" | Traceback: {event_data['traceback']}"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

def write_run_state(run_dir: Path, state: Any) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(state, "run_id"):
        state_dict = {
            "run_id": state.run_id,
            "status": state.status,
            "updated_at": state.updated_at,
            "artifact_hashes": state.artifact_hashes,
            "last_checkpoint": state.last_checkpoint,
        }
    else:
        state_dict = {
            "run_id": state.get("run_id"),
            "status": state.get("status"),
            "updated_at": state.get("updated_at"),
            "artifact_hashes": state.get("artifact_hashes"),
            "last_checkpoint": state.get("last_checkpoint"),
        }
    validate_public_safe(state_dict)
    output_path = run_dir / "state.json"
    temp_path = output_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(state_dict, f, indent=2, sort_keys=True)
    temp_path.replace(output_path)

def recover_truncated_jsonl(events_file_path: Path) -> None:
    if not events_file_path.exists():
        return
    valid_lines = []
    with open(events_file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                json.loads(line)
                valid_lines.append(line)
            except ValueError:
                break
    with open(events_file_path, "w", encoding="utf-8") as f:
        f.writelines(valid_lines)

def record_metrics(summary_dir: Path, run_data: dict[str, Any]) -> None:
    validate_public_safe(run_data)
    summary_dir.mkdir(parents=True, exist_ok=True)
    headers = [
        "experiment_id",
        "representation",
        "num_patients",
        "num_stays",
        "alive_count",
        "expired_count",
        "split_strategy",
        "seed",
        "auroc",
        "auroc_ci_lower",
        "auroc_ci_upper",
        "average_precision",
        "average_precision_ci_lower",
        "average_precision_ci_upper",
        "f1",
        "balanced_accuracy",
        "parameter_count",
        "runtime",
        "exclusions",
        "failure_notes"
    ]
    cleaned_data = {}
    for h in headers:
        val = run_data.get(h, "")
        if isinstance(val, float) and not np.isfinite(val):
            val = ""
        cleaned_data[h] = val

    for filename in ["experiment_comparison.csv", "final_results.csv"]:
        csv_path = summary_dir / filename
        rows = []
        if csv_path.exists():
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
        
        updated = False
        for i, r in enumerate(rows):
            if (
                r.get("experiment_id") == str(cleaned_data["experiment_id"]) and
                r.get("representation") == str(cleaned_data["representation"]) and
                r.get("seed") == str(cleaned_data["seed"]) and
                r.get("split_strategy") == str(cleaned_data["split_strategy"])
            ):
                rows[i] = cleaned_data
                updated = True
                break
        if not updated:
            rows.append(cleaned_data)
        
        temp_path = csv_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        temp_path.replace(csv_path)

def save_best_config(summary_dir: Path, config: dict[str, Any]) -> None:
    validate_public_safe(config)
    summary_dir.mkdir(parents=True, exist_ok=True)
    output_path = summary_dir / "best_config.json"
    temp_path = output_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    temp_path.replace(output_path)
