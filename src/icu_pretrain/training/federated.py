"""FedAvg-style pseudo-client simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from icu_pretrain.data.collate import ResumableDeterministicSampler, SupervisedCollator, create_dataloader
from icu_pretrain.data.dataset import EncodedDataset
from icu_pretrain.data.eicu_event_builder import SplitRecord, write_split_metadata
from icu_pretrain.experiments.tracking import log_event, write_run_state
from icu_pretrain.models.heads import MortalityPredictionHead
from icu_pretrain.models.transformer import ICUTinyTransformer
from icu_pretrain.training.evaluate import bootstrap_patient_metrics, compute_binary_metrics, find_best_f1_threshold
from icu_pretrain.training.finetune import train_finetuning_model
from icu_pretrain.training.pretrain import load_artifact_hashes, train_model
from icu_pretrain.utils import load_yaml, validate_final_config, validate_fedavg_config


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    temporary_path.replace(path)


def _hash_payload(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_records(processed_dir: Path) -> list[dict[str, Any]]:
    split_metadata_path = processed_dir / "split_metadata.json"
    if not split_metadata_path.is_file():
        raise FileNotFoundError(f"split_metadata.json not found in {processed_dir}")
    split_metadata = _load_json(split_metadata_path)
    stay_metadata = {str(record["patientunitstayid"]): record for record in split_metadata}
    encoded_root = processed_dir / "encoded"
    if not encoded_root.is_dir():
        raise FileNotFoundError(f"encoded directory not found in {processed_dir}")

    records: list[dict[str, Any]] = []
    for split_name in ("train", "validation", "test"):
        split_dir = encoded_root / split_name
        dataset = EncodedDataset(split_dir)
        for stay in dataset.stays():
            stay_id = str(stay.patientunitstayid)
            if stay_id not in stay_metadata:
                raise ValueError(f"split metadata missing stay {stay_id}")
            metadata = stay_metadata[stay_id]
            records.append(
                {
                    "patientunitstayid": stay_id,
                    "uniquepid": str(metadata["uniquepid"]),
                    "hospitalid": metadata["hospitalid"],
                    "label": int(stay.label),
                    "split_name": split_name,
                    "tokens": list(stay.tokens),
                }
            )
    return records


def _group_records_by_split(records: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {"train": [], "validation": [], "test": []}
    for record in records:
        split_name = str(record["split_name"])
        if split_name not in grouped:
            raise ValueError(f"unsupported split_name: {split_name}")
        grouped[split_name].append(dict(record))
    return grouped


def _split_records(records: Sequence[dict[str, Any]], *, seed: int) -> dict[str, list[dict[str, Any]]]:
    ordered = [dict(record) for record in records]
    rng = np.random.default_rng(seed)
    rng.shuffle(ordered)
    n_total = len(ordered)
    if n_total == 0:
        return {"train": [], "validation": [], "test": []}
    train_end = max(1, int(round(n_total * 0.7)))
    validation_end = max(train_end + 1 if n_total >= 2 else train_end, int(round(n_total * 0.85)))
    if n_total >= 2:
        validation_end = min(validation_end, n_total - 1)
    train_records = ordered[:train_end]
    validation_records = ordered[train_end:validation_end]
    test_records = ordered[validation_end:]
    if not validation_records and len(train_records) > 1:
        validation_records = [train_records.pop()]
    if not test_records and len(train_records) > 1:
        test_records = [train_records.pop()]
    for record in train_records:
        record["split_name"] = "train"
    for record in validation_records:
        record["split_name"] = "validation"
    for record in test_records:
        record["split_name"] = "test"
    return {
        "train": train_records,
        "validation": validation_records,
        "test": test_records,
    }


def _copy_scaffold(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = source_dir / "vocab.json"
    if not vocab_path.is_file():
        raise FileNotFoundError(f"vocab.json not found in {source_dir}")
    shutil.copy2(vocab_path, target_dir / "vocab.json")
    manifests_src = source_dir / "manifests"
    if not manifests_src.is_dir():
        raise FileNotFoundError(f"manifests directory not found in {source_dir}")
    manifests_dst = target_dir / "manifests"
    if manifests_dst.exists():
        shutil.rmtree(manifests_dst)
    shutil.copytree(manifests_src, manifests_dst)


def _write_encoded_split(split_dir: Path, stays: Sequence[dict[str, Any]], shard_size: int) -> None:
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    shard_id = 0
    for start in range(0, len(stays), shard_size):
        chunk = stays[start : start + shard_size]
        encoded_chunk = [
            {
                "patientunitstayid": stay["patientunitstayid"],
                "tokens": list(stay["tokens"]),
                "label": int(stay["label"]),
                "split_name": str(stay["split_name"]),
            }
            for stay in chunk
        ]
        EncodedDataset.write_shard(encoded_chunk, shard_id, split_dir)
        shard_id += 1
    if not stays:
        (split_dir / "index.json").write_text("[]\n", encoding="utf-8")


def _materialize_split_processed_dir(
    source_dir: Path,
    target_dir: Path,
    splits: dict[str, Sequence[dict[str, Any]]],
    *,
    shard_size: int,
) -> Path:
    _copy_scaffold(source_dir, target_dir)
    grouped = {split_name: [dict(record) for record in split_records] for split_name, split_records in splits.items()}
    all_records: list[dict[str, Any]] = []
    for split_records in grouped.values():
        all_records.extend(split_records)
    split_metadata = [
        SplitRecord(
            patientunitstayid=record["patientunitstayid"],
            uniquepid=record["uniquepid"],
            hospitalid=record["hospitalid"],
            split_name=record["split_name"],
        )
        for record in all_records
    ]
    write_split_metadata(target_dir / "split_metadata.json", split_metadata, processed_root=target_dir)
    encoded_root = target_dir / "encoded"
    for split_name, split_records in grouped.items():
        _write_encoded_split(encoded_root / split_name, split_records, shard_size=shard_size)
    return target_dir


def build_hospital_grouped_folds(
    records: Sequence[dict[str, Any]],
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len(records) == 0:
        raise ValueError("records must not be empty")

    labels = np.asarray([int(record["label"]) for record in records], dtype=int)
    groups = np.asarray([str(record["hospitalid"]) for record in records], dtype=object)
    unique_hospitals = sorted(set(groups.tolist()))
    if len(unique_hospitals) < n_splits:
        raise ValueError("insufficient hospitals for the requested number of folds")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[dict[str, Any]] = []
    seen_test_hospitals: set[str] = set()
    all_indices = np.arange(len(records)).reshape(-1, 1)

    for fold_number, (train_indices, test_indices) in enumerate(splitter.split(all_indices, labels, groups), start=1):
        train_hospitals = sorted({str(groups[idx]) for idx in train_indices})
        test_hospitals = sorted({str(groups[idx]) for idx in test_indices})
        overlap = seen_test_hospitals.intersection(test_hospitals)
        if overlap:
            raise ValueError("hospitals crossed test folds")
        seen_test_hospitals.update(test_hospitals)
        test_labels = labels[test_indices]
        fold_status = "defined" if len(set(test_labels.tolist())) > 1 else "undefined"
        folds.append(
            {
                "fold_number": fold_number,
                "train_hospitals": train_hospitals,
                "test_hospitals": test_hospitals,
                "train_count": int(train_indices.size),
                "test_count": int(test_indices.size),
                "status": fold_status,
            }
        )

    if seen_test_hospitals != set(unique_hospitals):
        raise ValueError("each hospital must appear in exactly one test fold")
    return folds


def assign_hospital_clients(
    records: Sequence[dict[str, Any]],
    *,
    n_clients: int = 3,
) -> dict[str, Any]:
    if n_clients < 1:
        raise ValueError("n_clients must be at least 1")

    hospital_counts: dict[str, int] = {}
    for record in records:
        hospital = str(record["hospitalid"])
        hospital_counts[hospital] = hospital_counts.get(hospital, 0) + 1

    client_assignments = {client_idx: {"hospitals": [], "stay_count": 0} for client_idx in range(n_clients)}
    for hospital in sorted(hospital_counts, key=lambda hid: (-hospital_counts[hid], hid)):
        client_idx = min(client_assignments, key=lambda idx: (client_assignments[idx]["stay_count"], idx))
        client_assignments[client_idx]["hospitals"].append(hospital)
        client_assignments[client_idx]["stay_count"] += hospital_counts[hospital]

    return {
        "hospital_counts": hospital_counts,
        "client_assignments": client_assignments,
    }


def average_state_dicts(
    state_dicts: Sequence[dict[str, torch.Tensor]],
    weights: Sequence[float],
) -> dict[str, torch.Tensor]:
    if len(state_dicts) != len(weights):
        raise ValueError("state_dicts and weights must have the same length")
    if not state_dicts:
        raise ValueError("state_dicts must not be empty")
    total_weight = float(sum(weights))
    if total_weight <= 0:
        raise ValueError("weights must sum to a positive value")

    reference = state_dicts[0]
    averaged: dict[str, torch.Tensor] = {}
    for key, reference_value in reference.items():
        stacked = []
        for state in state_dicts:
            if key not in state:
                raise ValueError(f"incompatible model states missing key: {key}")
            current_value = state[key]
            if current_value.shape != reference_value.shape:
                raise ValueError(f"incompatible tensor shape for key: {key}")
            stacked.append(current_value)
        if torch.is_floating_point(reference_value) or torch.is_complex(reference_value):
            accumulator = torch.zeros_like(reference_value, dtype=torch.float32)
            for current_value, weight in zip(stacked, weights, strict=True):
                accumulator = accumulator + current_value.to(accumulator.dtype) * float(weight)
            averaged[key] = (accumulator / total_weight).to(reference_value.dtype)
        else:
            first_value = stacked[0]
            if not all(torch.equal(first_value, current_value) for current_value in stacked[1:]):
                raise ValueError(f"incompatible non-floating tensor for key: {key}")
            averaged[key] = first_value.clone()
    return averaged


def _instantiate_model_and_head(config: dict[str, Any], vocab_size: int) -> tuple[ICUTinyTransformer, MortalityPredictionHead]:
    model_conf = config.get("model", {})
    model = ICUTinyTransformer(
        vocab_size=vocab_size,
        max_seq_len=int(model_conf.get("max_seq_len", 256)),
        d_model=int(model_conf.get("d_model", 64)),
        n_heads=int(model_conf.get("n_heads", 4)),
        n_layers=int(model_conf.get("n_layers", 2)),
        dim_feedforward=int(model_conf.get("dim_feedforward", 256)),
        dropout=float(model_conf.get("dropout", 0.1)),
    )
    head = MortalityPredictionHead(d_model=int(model_conf.get("d_model", 64)))
    return model, head


def _evaluate_model(
    model: ICUTinyTransformer,
    head: MortalityPredictionHead,
    dataset: EncodedDataset,
    vocab: dict[str, int],
    *,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    if len(dataset) == 0:
        return {
            "patient_ids": [],
            "targets": [],
            "probs": [],
            "metrics": {"auroc": float("nan"), "ap": float("nan"), "f1": float("nan"), "balanced_accuracy": float("nan")},
        }
    device = next(model.parameters()).device
    loader = create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=ResumableDeterministicSampler(len(dataset), seed=seed, epoch=0),
        collator=SupervisedCollator(vocab=vocab),
        num_workers=0,
    )
    probs: list[float] = []
    targets: list[int] = []
    patient_ids = [str(stay.patientunitstayid) for stay in dataset.stays()]
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            _, cls_output = model(input_ids, padding_mask=attention_mask)
            logits = head(cls_output)
            batch_probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
            probs.extend(float(value) for value in batch_probs)
            targets.extend(int(value) for value in labels.detach().cpu().numpy().tolist())
    metrics = compute_binary_metrics(targets, probs)
    return {
        "patient_ids": patient_ids,
        "targets": targets,
        "probs": probs,
        "metrics": metrics,
    }


def _checkpoint_namespace(model_state: dict[str, torch.Tensor], head_state: dict[str, torch.Tensor]) -> SimpleNamespace:
    return SimpleNamespace(model_state=model_state, prediction_head_state=head_state)


def _save_checkpoint(path: Path, model_state: dict[str, torch.Tensor], head_state: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_namespace(model_state, head_state), path)


def _write_processed_eval_summary(run_dir: Path, summary_name: str, payload: dict[str, Any]) -> Path:
    output_path = run_dir / summary_name
    _atomic_write_json(output_path, payload)
    return output_path


def run_hospital_grouped_evaluation(
    config: dict[str, Any],
    processed_dir: Path,
    run_dir: Path,
    *,
    resume: str = "auto",
) -> dict[str, Any]:
    validated = validate_final_config(deepcopy(config))
    processed_dir = Path(processed_dir)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    records = _load_records(processed_dir)
    folds = build_hospital_grouped_folds(records, n_splits=5, seed=validated["runtime"]["seed"])
    vocab = _load_json(processed_dir / "vocab.json")
    artifact_hashes = load_artifact_hashes(processed_dir)
    config_hash = _hash_payload(validated)
    artifact_hashes["config"] = config_hash

    run_state = {
        "run_id": validated.get("experiment", {}).get("name", "hospital_grouped_eval"),
        "status": "running",
        "updated_at": _now(),
        "artifact_hashes": artifact_hashes,
        "last_checkpoint": None,
    }
    write_run_state(run_dir, run_state)
    log_event(
        run_dir,
        {
            "timestamp": _now(),
            "stage": "hospital_grouped_evaluation",
            "status": "started",
            "fold_count": len(folds),
        },
        stage="hospital_grouped_evaluation",
    )

    fold_summaries: list[dict[str, Any]] = []
    pooled_test_probs: list[float] = []
    pooled_test_targets: list[int] = []
    pooled_test_patients: list[str] = []
    pooled_val_probs: list[float] = []
    pooled_val_targets: list[int] = []
    batch_size = int(validated.get("finetuning", {}).get("batch_size", 8))
    shard_size = int(validated.get("data_processing", {}).get("encoded_shard_stays", 128))

    pretrain_enabled = bool(validated.get("pretraining", {}).get("enabled", True))

    for fold in folds:
        fold_number = int(fold["fold_number"])
        fold_dir = run_dir / "folds" / f"fold_{fold_number:02d}"
        fold_state_path = fold_dir / "state.json"
        results_path = fold_dir / "results.json"
        if resume == "auto" and fold_state_path.is_file() and results_path.is_file():
            state_data = _load_json(fold_state_path)
            if state_data.get("status") == "completed":
                fold_summary = _load_json(results_path)
                fold_summaries.append(fold_summary)
                pooled_test_probs.extend(fold_summary.get("test_probs", []))
                pooled_test_targets.extend(fold_summary.get("test_targets", []))
                pooled_test_patients.extend(fold_summary.get("test_patients", []))
                pooled_val_probs.extend(fold_summary.get("val_probs", []))
                pooled_val_targets.extend(fold_summary.get("val_targets", []))
                continue

        fold_dir.mkdir(parents=True, exist_ok=True)
        log_event(
            fold_dir,
            {
                "timestamp": _now(),
                "stage": "hospital_grouped_evaluation",
                "status": "fold_started",
                "fold_number": fold_number,
            },
            stage="hospital_grouped_evaluation",
        )

        train_hospitals = set(fold["train_hospitals"])
        test_hospitals = set(fold["test_hospitals"])
        local_train_records = [
            record
            for record in records
            if str(record["hospitalid"]) in train_hospitals
        ]
        local_train_splits = _split_records(local_train_records, seed=validated["runtime"]["seed"] + fold_number)
        heldout_test_records = [
            {**record, "split_name": "test"}
            for record in records
            if str(record["hospitalid"]) in test_hospitals
        ]
        fold_processed_dir = _materialize_split_processed_dir(
            processed_dir,
            fold_dir / "processed",
            {
                "train": local_train_splits["train"],
                "validation": local_train_splits["validation"],
                "test": heldout_test_records,
            },
            shard_size=shard_size,
        )

        train_split = EncodedDataset(fold_processed_dir / "encoded" / "train")
        val_split = EncodedDataset(fold_processed_dir / "encoded" / "validation")
        test_split = EncodedDataset(fold_processed_dir / "encoded" / "test")
        if len(train_split) == 0 or len(val_split) == 0 or len(test_split) == 0:
            fold_summary = {
                "fold_number": fold_number,
                "status": "undefined",
                "train_count": len(train_split),
                "validation_count": len(val_split),
                "test_count": len(test_split),
                "auroc": float("nan"),
                "average_precision": float("nan"),
                "f1": float("nan"),
                "balanced_accuracy": float("nan"),
                "val_probs": [],
                "val_targets": [],
                "test_probs": [],
                "test_targets": [],
                "test_patients": [],
            }
            _atomic_write_json(results_path, fold_summary)
            write_run_state(
                fold_dir,
                {
                    "run_id": f"{run_state['run_id']}_fold_{fold_number:02d}",
                    "status": "completed",
                    "updated_at": _now(),
                    "artifact_hashes": {"config": config_hash},
                    "last_checkpoint": None,
                },
            )
            fold_summaries.append(fold_summary)
            continue

        pretrain_checkpoint = None
        if pretrain_enabled:
            pretrain_dir = fold_dir / "pretrain"
            fold_pretrain_config = deepcopy(validated)
            fold_pretrain_config["experiment"] = dict(fold_pretrain_config.get("experiment", {}))
            fold_pretrain_config["experiment"]["name"] = f"{validated.get('experiment', {}).get('name', 'hospital_grouped_eval')}_fold_{fold_number:02d}_pretrain"
            pretrain_result = train_model(fold_pretrain_config, fold_processed_dir, pretrain_dir, resume=resume)
            pretrain_checkpoint = pretrain_dir / "checkpoints" / "best.pt"
            _save_checkpoint(fold_dir / "checkpoint.pt", pretrain_result["model_state"], pretrain_result["prediction_head_state"])

        fold_finetune_config = deepcopy(validated)
        fold_finetune_config["experiment"] = dict(fold_finetune_config.get("experiment", {}))
        fold_finetune_config["experiment"]["name"] = f"{validated.get('experiment', {}).get('name', 'hospital_grouped_eval')}_fold_{fold_number:02d}"
        fold_finetune_config.setdefault("finetuning", {})["evaluate_on_test"] = True
        fold_finetune_config["evaluation"] = dict(fold_finetune_config.get("evaluation", {}))
        fold_finetune_config["evaluation"]["split"] = "patient_grouped_test"

        train_result = train_finetuning_model(
            fold_finetune_config,
            fold_processed_dir,
            fold_dir / "finetune",
            resume=resume,
            pretrain_checkpoint=pretrain_checkpoint,
        )
        _save_checkpoint(fold_dir / "checkpoint.pt", train_result["model_state"], train_result["prediction_head_state"])
        model, head = _instantiate_model_and_head(fold_finetune_config, len(vocab))
        model.load_state_dict(train_result["model_state"])
        head.load_state_dict(train_result["prediction_head_state"])
        model.eval()
        head.eval()

        val_eval = _evaluate_model(
            model,
            head,
            val_split,
            vocab,
            batch_size=batch_size,
            seed=validated["runtime"]["seed"],
        )
        test_eval = _evaluate_model(
            model,
            head,
            test_split,
            vocab,
            batch_size=batch_size,
            seed=validated["runtime"]["seed"],
        )
        threshold = find_best_f1_threshold(val_eval["targets"], val_eval["probs"])
        test_metrics = compute_binary_metrics(test_eval["targets"], test_eval["probs"], threshold=threshold)
        test_patients = [
            str(record["uniquepid"])
            for record in _load_json(fold_processed_dir / "split_metadata.json")
            if str(record["split_name"]) == "test"
        ]
        bootstrap = bootstrap_patient_metrics(
            patients=test_patients,
            y_true=test_eval["targets"],
            y_prob=test_eval["probs"],
            threshold=threshold,
            n_replicates=1000,
            seed=validated["runtime"]["seed"],
        )

        fold_summary = {
            "fold_number": fold_number,
            "status": fold["status"] if len(test_eval["targets"]) > 0 else "undefined",
            "train_hospitals": len(train_hospitals),
            "test_hospitals": len(test_hospitals),
            "train_count": len(train_split),
            "validation_count": len(val_split),
            "test_count": len(test_split),
            "auroc": test_metrics["auroc"],
            "auroc_ci": bootstrap["auroc_ci"],
            "average_precision": test_metrics["ap"],
            "average_precision_ci": bootstrap["ap_ci"],
            "f1": test_metrics["f1"],
            "balanced_accuracy": test_metrics["balanced_accuracy"],
            "val_probs": val_eval["probs"],
            "val_targets": val_eval["targets"],
            "test_probs": test_eval["probs"],
            "test_targets": test_eval["targets"],
            "test_patients": test_patients,
        }
        _atomic_write_json(results_path, fold_summary)
        write_run_state(
            fold_dir,
            {
                "run_id": f"{run_state['run_id']}_fold_{fold_number:02d}",
                "status": "completed",
                "updated_at": _now(),
                "artifact_hashes": {"config": config_hash},
                "last_checkpoint": None,
            },
        )
        log_event(
            fold_dir,
            {
                "timestamp": _now(),
                "stage": "hospital_grouped_evaluation",
                "status": "fold_completed",
                "fold_number": fold_number,
                "auroc": test_metrics["auroc"],
                "average_precision": test_metrics["ap"],
            },
            stage="hospital_grouped_evaluation",
        )
        fold_summaries.append(fold_summary)
        pooled_test_probs.extend(test_eval["probs"])
        pooled_test_targets.extend(test_eval["targets"])
        pooled_test_patients.extend(test_patients)
        pooled_val_probs.extend(val_eval["probs"])
        pooled_val_targets.extend(val_eval["targets"])

    pooled_threshold = find_best_f1_threshold(pooled_val_targets, pooled_val_probs)
    pooled_metrics = compute_binary_metrics(pooled_test_targets, pooled_test_probs, threshold=pooled_threshold)
    pooled_bootstrap = bootstrap_patient_metrics(
        patients=pooled_test_patients,
        y_true=pooled_test_targets,
        y_prob=pooled_test_probs,
        threshold=pooled_threshold,
        n_replicates=1000,
        seed=validated["runtime"]["seed"],
    )
    summary = {
        "evaluation_type": "exploratory_hospital_grouped_cv",
        "disclaimer": "Exploratory held-out-hospital evaluation within the demo dataset.",
        "fold_count": len(folds),
        "folds": fold_summaries,
        "pooled": {
            "auroc": pooled_metrics["auroc"],
            "auroc_ci": pooled_bootstrap["auroc_ci"],
            "average_precision": pooled_metrics["ap"],
            "average_precision_ci": pooled_bootstrap["ap_ci"],
            "f1": pooled_metrics["f1"],
            "balanced_accuracy": pooled_metrics["balanced_accuracy"],
            "threshold": pooled_threshold,
        },
    }
    _write_processed_eval_summary(run_dir, "hospital_grouped_summary.json", summary)
    write_run_state(
        run_dir,
        {
            "run_id": run_state["run_id"],
            "status": "completed",
            "updated_at": _now(),
            "artifact_hashes": artifact_hashes,
            "last_checkpoint": None,
        },
    )
    log_event(
        run_dir,
        {
            "timestamp": _now(),
            "stage": "hospital_grouped_evaluation",
            "status": "completed",
            "fold_count": len(folds),
            "pooled_auroc": pooled_metrics["auroc"],
            "pooled_average_precision": pooled_metrics["ap"],
        },
        stage="hospital_grouped_evaluation",
    )
    return summary


def _save_model_checkpoint(run_dir: Path, model_state: dict[str, torch.Tensor], head_state: dict[str, torch.Tensor]) -> Path:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "global.pt"
    torch.save(_checkpoint_namespace(model_state, head_state), checkpoint_path)
    return checkpoint_path


def _evaluate_global_model(
    config: dict[str, Any],
    processed_dir: Path,
    model_state: dict[str, torch.Tensor],
    head_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    vocab = _load_json(processed_dir / "vocab.json")
    model, head = _instantiate_model_and_head(config, len(vocab))
    model.load_state_dict(model_state)
    head.load_state_dict(head_state)
    model.eval()
    head.eval()
    encoded_root = processed_dir / "encoded"
    val_split = EncodedDataset(encoded_root / "validation")
    test_split = EncodedDataset(encoded_root / "test")
    batch_size = int(config.get("finetuning", {}).get("batch_size", 8))
    seed = int(config.get("runtime", {}).get("seed", 42))
    val_eval = _evaluate_model(model, head, val_split, vocab, batch_size=batch_size, seed=seed)
    test_eval = _evaluate_model(model, head, test_split, vocab, batch_size=batch_size, seed=seed)
    threshold = find_best_f1_threshold(val_eval["targets"], val_eval["probs"])
    test_metrics = compute_binary_metrics(test_eval["targets"], test_eval["probs"], threshold=threshold)
    split_metadata = _load_json(processed_dir / "split_metadata.json")
    stay_to_patient = {str(record["patientunitstayid"]): str(record["uniquepid"]) for record in split_metadata}
    test_patients = [stay_to_patient[str(stay.patientunitstayid)] for stay in test_split.stays()]
    bootstrap = bootstrap_patient_metrics(
        patients=test_patients,
        y_true=test_eval["targets"],
        y_prob=test_eval["probs"],
        threshold=threshold,
        n_replicates=1000,
        seed=seed,
    )
    return {
        "val_probs": val_eval["probs"],
        "val_targets": val_eval["targets"],
        "test_probs": test_eval["probs"],
        "test_targets": test_eval["targets"],
        "test_patients": test_patients,
        "threshold": threshold,
        "metrics": test_metrics,
        "bootstrap": bootstrap,
    }


def run_fedavg_simulation(
    config: dict[str, Any] | None = None,
    processed_dir: Path | None = None,
    run_dir: Path | None = None,
    *,
    resume: str = "auto",
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    if argv is not None:
        parser = argparse.ArgumentParser(description="Run FedAvg-style pseudo-client simulation.")
        parser.add_argument("--config", type=Path, default=Path("configs/fedavg/eicu_demo_fedavg_sim.yaml"))
        parser.add_argument("--processed_dir", type=Path, default=None)
        parser.add_argument("--run_dir", type=Path, default=None)
        parser.add_argument("--resume", choices=["auto", "no"], default="auto")
        args = parser.parse_args(list(argv))
        config = load_yaml(args.config)
        processed_dir = args.processed_dir
        run_dir = args.run_dir
        resume = args.resume

    if config is None:
        config = load_yaml(Path("configs/fedavg/eicu_demo_fedavg_sim.yaml"))
    validated = validate_fedavg_config(deepcopy(config))
    if processed_dir is None:
        processed_dir = Path(validated["data"]["processed_dir"])
    if run_dir is None:
        run_dir = Path("results") / "runs" / validated.get("experiment", {}).get("name", "fedavg_run")

    processed_dir = Path(processed_dir)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(processed_dir)
    train_records = [record for record in records if str(record["split_name"]) == "train"]
    assignment = assign_hospital_clients(train_records, n_clients=int(validated["fedavg"]["clients"]))
    artifact_hashes = load_artifact_hashes(processed_dir)
    config_hash = _hash_payload(validated)
    artifact_hashes["config"] = config_hash

    write_run_state(
        run_dir,
        {
            "run_id": validated.get("experiment", {}).get("name", "fedavg_run"),
            "status": "running",
            "updated_at": _now(),
            "artifact_hashes": artifact_hashes,
            "last_checkpoint": None,
        },
    )
    log_event(
        run_dir,
        {
            "timestamp": _now(),
            "stage": "fedavg",
            "status": "started",
            "client_count": int(validated["fedavg"]["clients"]),
            "round_count": int(validated["fedavg"]["rounds"]),
        },
        stage="fedavg",
    )

    shard_size = int(validated.get("data_processing", {}).get("encoded_shard_stays", 128))
    clients: dict[int, dict[str, Any]] = {}
    for client_idx, client_payload in assignment["client_assignments"].items():
        client_hospitals = set(client_payload["hospitals"])
        client_train_records = [
            record
            for record in records
            if str(record["hospitalid"]) in client_hospitals and str(record["split_name"]) == "train"
        ]
        client_splits = _split_records(client_train_records, seed=validated["runtime"]["seed"] + client_idx)
        client_dir = run_dir / "clients" / f"client_{client_idx:02d}"
        client_processed_dir = _materialize_split_processed_dir(
            processed_dir,
            client_dir / "processed",
            client_splits,
            shard_size=shard_size,
        )
        clients[client_idx] = {
            "client_dir": client_dir,
            "processed_dir": client_processed_dir,
            "stay_count": int(client_payload["stay_count"]),
            "hospitals": list(client_hospitals),
        }

    global_reference = train_model(validated, processed_dir, run_dir / "pretrain", resume=resume)
    global_model_state = global_reference["model_state"]
    global_head_state = global_reference["prediction_head_state"]
    global_checkpoint_path = _save_model_checkpoint(run_dir, global_model_state, global_head_state)

    central_run_dir = run_dir / "central_reference"
    central_config = deepcopy(validated)
    central_config.setdefault("finetuning", {})["epochs"] = int(validated["fedavg"]["rounds"])
    central_reference = train_finetuning_model(
        central_config,
        processed_dir,
        central_run_dir,
        resume=resume,
        pretrain_checkpoint=global_checkpoint_path,
    )
    central_eval = _evaluate_global_model(validated, processed_dir, central_reference["model_state"], central_reference["prediction_head_state"])

    round_summaries: list[dict[str, Any]] = []
    for round_idx in range(1, int(validated["fedavg"]["rounds"]) + 1):
        round_dir = run_dir / "rounds" / f"round_{round_idx:02d}"
        round_state_path = round_dir / "state.json"
        if resume == "auto" and round_state_path.is_file():
            state_data = _load_json(round_state_path)
            if state_data.get("status") == "completed" and (round_dir / "results.json").is_file():
                round_summaries.append(_load_json(round_dir / "results.json"))
                checkpoint_path = round_dir / "checkpoint.pt"
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(f"checkpoint not found for round {round_idx}")
                checkpoint_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                global_model_state = checkpoint_state.model_state
                global_head_state = checkpoint_state.prediction_head_state
                continue

        round_dir.mkdir(parents=True, exist_ok=True)
        log_event(
            round_dir,
            {
                "timestamp": _now(),
                "stage": "fedavg",
                "status": "round_started",
                "round_number": round_idx,
            },
            stage="fedavg",
        )

        client_states: list[dict[str, torch.Tensor]] = []
        head_states: list[dict[str, torch.Tensor]] = []
        weights: list[float] = []
        client_summaries: list[dict[str, Any]] = []
        for client_idx, client_payload in clients.items():
            client_dir = client_payload["client_dir"]
            client_state_path = client_dir / f"round_{round_idx:02d}" / "state.json"
            client_results_path = client_dir / f"round_{round_idx:02d}" / "results.json"
            if resume == "auto" and client_state_path.is_file() and client_results_path.is_file():
                state_data = _load_json(client_state_path)
                client_summary = _load_json(client_results_path)
                if state_data.get("status") == "completed":
                    checkpoint_path = client_dir / f"round_{round_idx:02d}" / "checkpoint.pt"
                    if not checkpoint_path.is_file():
                        raise FileNotFoundError(f"checkpoint not found for client {client_idx} round {round_idx}")
                    checkpoint_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    client_states.append(checkpoint_state.model_state)
                    head_states.append(checkpoint_state.prediction_head_state)
                    weights.append(float(client_payload["stay_count"]))
                    client_summaries.append(client_summary)
                    continue
                if state_data.get("status") == "skipped":
                    client_summaries.append(client_summary)
                    continue

            client_round_dir = client_dir / f"round_{round_idx:02d}"
            client_round_dir.mkdir(parents=True, exist_ok=True)
            if client_payload["stay_count"] == 0:
                client_summary = {
                    "client_idx": client_idx,
                    "round_number": round_idx,
                    "status": "skipped",
                    "stay_count": 0,
                }
                _atomic_write_json(client_results_path, client_summary)
                write_run_state(
                    client_round_dir,
                    {
                        "run_id": f"{validated.get('experiment', {}).get('name', 'fedavg_run')}_client_{client_idx:02d}_round_{round_idx:02d}",
                        "status": "completed",
                        "updated_at": _now(),
                        "artifact_hashes": {"config": config_hash},
                        "last_checkpoint": None,
                    },
                )
                client_summaries.append(client_summary)
                continue

            client_config = deepcopy(validated)
            client_config.setdefault("experiment", {})["name"] = f"{validated.get('experiment', {}).get('name', 'fedavg_run')}_client_{client_idx:02d}_round_{round_idx:02d}"
            client_config.setdefault("finetuning", {})["epochs"] = int(validated["fedavg"]["local_epochs"])
            client_config["finetuning"]["evaluate_on_test"] = False
            checkpoint_path = client_round_dir / "global_init.pt"
            torch.save(_checkpoint_namespace(global_model_state, global_head_state), checkpoint_path)
            result = train_finetuning_model(
                client_config,
                client_payload["processed_dir"],
                client_round_dir,
                resume=resume,
                pretrain_checkpoint=checkpoint_path,
            )
            _save_checkpoint(client_round_dir / "checkpoint.pt", result["model_state"], result["prediction_head_state"])
            client_state = {
                "client_idx": client_idx,
                "round_number": round_idx,
                "status": "completed",
                "stay_count": client_payload["stay_count"],
                "val_loss": result.get("val_loss"),
            }
            _atomic_write_json(client_results_path, client_state)
            write_run_state(
                client_round_dir,
                {
                    "run_id": f"{validated.get('experiment', {}).get('name', 'fedavg_run')}_client_{client_idx:02d}_round_{round_idx:02d}",
                    "status": "completed",
                    "updated_at": _now(),
                    "artifact_hashes": {"config": config_hash},
                    "last_checkpoint": None,
                },
            )
            client_states.append(result["model_state"])
            head_states.append(result["prediction_head_state"])
            weights.append(float(client_payload["stay_count"]))
            client_summaries.append(client_state)

        if not client_states:
            round_summary = {
                "round_number": round_idx,
                "status": "skipped",
                "client_count": 0,
            }
            _atomic_write_json(round_dir / "results.json", round_summary)
            write_run_state(
                round_dir,
                {
                    "run_id": f"{validated.get('experiment', {}).get('name', 'fedavg_run')}_round_{round_idx:02d}",
                    "status": "completed",
                    "updated_at": _now(),
                    "artifact_hashes": {"config": config_hash},
                    "last_checkpoint": None,
                },
            )
            round_summaries.append(round_summary)
            continue

        global_model_state = average_state_dicts(client_states, weights)
        global_head_state = average_state_dicts(head_states, weights)
        _save_checkpoint(round_dir / "checkpoint.pt", global_model_state, global_head_state)
        round_summary = {
            "round_number": round_idx,
            "status": "completed",
            "client_count": len(client_states),
            "client_summaries": [
                {
                    "client_idx": client_summary["client_idx"],
                    "status": client_summary["status"],
                    "stay_count": client_summary.get("stay_count", 0),
                }
                for client_summary in client_summaries
            ],
        }
        _atomic_write_json(round_dir / "results.json", round_summary)
        write_run_state(
            round_dir,
            {
                "run_id": f"{validated.get('experiment', {}).get('name', 'fedavg_run')}_round_{round_idx:02d}",
                "status": "completed",
                "updated_at": _now(),
                "artifact_hashes": {"config": config_hash},
                "last_checkpoint": None,
            },
        )
        log_event(
            round_dir,
            {
                "timestamp": _now(),
                "stage": "fedavg",
                "status": "round_completed",
                "round_number": round_idx,
                "client_count": len(client_states),
            },
            stage="fedavg",
        )
        round_summaries.append(round_summary)

    final_eval = _evaluate_global_model(validated, processed_dir, global_model_state, global_head_state)
    central_metrics = central_eval["metrics"]
    final_metrics = final_eval["metrics"]
    summary = {
        "evaluation_type": "simulated_hospital_cluster_fedavg",
        "disclaimer": "Single-process simulated FedAvg on hospital clusters within the demo dataset.",
        "client_count": int(validated["fedavg"]["clients"]),
        "round_count": int(validated["fedavg"]["rounds"]),
        "clients": {
            str(client_idx): {
                "stay_count": client_payload["stay_count"],
                "hospitals": client_payload["hospitals"],
            }
            for client_idx, client_payload in clients.items()
        },
        "rounds": round_summaries,
        "central_reference": {
            "auroc": central_metrics["auroc"],
            "average_precision": central_metrics["ap"],
            "f1": central_metrics["f1"],
            "balanced_accuracy": central_metrics["balanced_accuracy"],
        },
        "federated": {
            "auroc": final_metrics["auroc"],
            "auroc_ci": final_eval["bootstrap"]["auroc_ci"],
            "average_precision": final_metrics["ap"],
            "average_precision_ci": final_eval["bootstrap"]["ap_ci"],
            "f1": final_metrics["f1"],
            "balanced_accuracy": final_metrics["balanced_accuracy"],
            "threshold": final_eval["threshold"],
        },
    }
    _write_processed_eval_summary(run_dir, "fedavg_summary.json", summary)
    write_run_state(
        run_dir,
        {
            "run_id": validated.get("experiment", {}).get("name", "fedavg_run"),
            "status": "completed",
            "updated_at": _now(),
            "artifact_hashes": artifact_hashes,
            "last_checkpoint": None,
        },
    )
    log_event(
        run_dir,
        {
            "timestamp": _now(),
            "stage": "fedavg",
            "status": "completed",
            "round_count": int(validated["fedavg"]["rounds"]),
            "federated_auroc": final_metrics["auroc"],
        },
        stage="fedavg",
    )
    return summary
