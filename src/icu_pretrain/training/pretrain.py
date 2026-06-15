"""Masked event pretraining loop."""


import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from icu_pretrain.constants import CONTRACT_SCHEMA_VERSION
from icu_pretrain.utils import load_yaml, validate_final_config
from icu_pretrain.data.dataset import EncodedDataset
from icu_pretrain.data.collate import ResumableDeterministicSampler, MLMCollator, create_dataloader
from icu_pretrain.models.transformer import ICUTinyTransformer
from icu_pretrain.models.heads import MaskedEventPredictionHead, compute_mlm_loss
from icu_pretrain.data.eicu_event_builder import CheckpointContract, RunState, validate_checkpoint_contract


class CollateWrapper:
    def __init__(self, collator, epoch: int = 0) -> None:
        self.collator = collator
        self.epoch = epoch

    def __call__(self, batch):
        return self.collator(batch, epoch=self.epoch)


class TrainingInterruptedException(Exception):
    pass


def get_config_hash(config: dict[str, Any]) -> str:
    serialized = json.dumps(config, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_artifact_hashes(processed_dir: Path) -> dict[str, str]:
    stages = {
        "vocabulary": "fit_vocabulary",
        "split": "build_cohort_and_splits",
        "preprocessing": "fit_training_preprocessing",
        "encoded_dataset": "encode_split_shards",
    }
    hashes = {}
    for key, stage_name in stages.items():
        manifest_path = processed_dir / "manifests" / stage_name / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing manifest for stage {stage_name} at {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            h = manifest_data.get("config_hash")
            if not h:
                raise ValueError(f"Missing config_hash in {manifest_path}")
            hashes[key] = h
    return hashes


def write_run_state(run_dir: Path, state: RunState) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "state.json"
    temporary_path = output_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(
            {
                "run_id": state.run_id,
                "status": state.status,
                "updated_at": state.updated_at,
                "artifact_hashes": state.artifact_hashes,
                "last_checkpoint": state.last_checkpoint,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def log_event(run_dir: Path, event_data: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    events_file = run_dir / "events.jsonl"
    with open(events_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event_data) + "\n")
    log_file = run_dir / "run.log"
    timestamp = event_data.get("timestamp", "")
    msg = f"[{timestamp}] Stage: {event_data.get('stage', 'pretrain')} | Status: {event_data.get('status', '')}"
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
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def save_checkpoint(
    checkpoint: CheckpointContract,
    run_dir: Path,
    reason: str,
    periodic_checkpoints: list[Path],
    keep_last: int = 2
) -> Path:
    validate_checkpoint_contract(checkpoint)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{reason}.pt"
    if reason.startswith("step_") or reason.startswith("epoch_"):
        filename = f"checkpoint_{reason}.pt"
    checkpoint_path = checkpoints_dir / filename
    torch.save(checkpoint, checkpoint_path)
    if reason.startswith("step_") or reason.startswith("epoch_"):
        periodic_checkpoints.append(checkpoint_path)
        if len(periodic_checkpoints) > keep_last:
            old_checkpoint = periodic_checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
    return checkpoint_path


def train_model(
    config: dict[str, Any],
    processed_dir: Path,
    run_dir: Path,
    resume: str = "auto",
    interrupt_after_batches: int | None = None,
) -> dict[str, Any]:
    if not processed_dir.is_dir():
        raise FileNotFoundError(f"processed_dir {processed_dir} does not exist")
    vocab_path = processed_dir / "vocab.json"
    if not vocab_path.is_file():
        raise FileNotFoundError(f"vocab.json not found in {processed_dir}")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    encoded_dir = processed_dir / "encoded"
    train_dir = encoded_dir / "train"
    val_dir = encoded_dir / "validation"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(f"split directories not found in {encoded_dir}")
    train_dataset = EncodedDataset(train_dir)
    val_dataset = EncodedDataset(val_dir)
    if len(train_dataset) == 0:
        raise ValueError("Training dataset has zero stays")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset has zero stays")
    artifact_hashes = load_artifact_hashes(processed_dir)
    config_hash = get_config_hash(config)
    artifact_hashes["config"] = config_hash

    last_checkpoint_path = None
    state_path = run_dir / "state.json"
    if resume == "auto" and state_path.is_file():
        with open(state_path, "r", encoding="utf-8") as f:
            state_data = json.load(f)
        if state_data.get("last_checkpoint"):
            last_checkpoint_path = Path(state_data["last_checkpoint"])

    model_conf = config.get("model", {})
    model = ICUTinyTransformer(
        vocab_size=len(vocab),
        max_seq_len=model_conf.get("max_seq_len", 256),
        d_model=model_conf.get("d_model", 64),
        n_heads=model_conf.get("n_heads", 4),
        n_layers=model_conf.get("n_layers", 2),
        dim_feedforward=model_conf.get("dim_feedforward", 256),
        dropout=model_conf.get("dropout", 0.1),
    )
    head = MaskedEventPredictionHead(
        d_model=model_conf.get("d_model", 64),
        vocab_size=len(vocab),
    )

    train_conf = config.get("pretraining", {})
    lr = train_conf.get("learning_rate", 0.0005)
    wd = train_conf.get("weight_decay", 0.01)
    optimizer = AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=lr,
        weight_decay=wd,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    run_conf = config.get("runtime", {})
    seed = run_conf.get("seed", 42)
    sampler = ResumableDeterministicSampler(
        dataset_size=len(train_dataset),
        seed=seed,
        epoch=0,
        cursor=0,
    )
    mask_prob = train_conf.get("mask_probability", 0.15)
    collator = MLMCollator(vocab=vocab, mlm_probability=mask_prob, seed=seed)
    collate_wrapper = CollateWrapper(collator, epoch=0)

    start_epoch = 0
    global_batch = 0
    optimizer_step = 0
    accumulation_step = 0
    best_metric = float("inf")
    best_epoch = None
    training_history = {"train_losses": [], "val_losses": []}
    periodic_checkpoints = []

    if last_checkpoint_path is not None:
        if not last_checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {last_checkpoint_path}")
        checkpoint = torch.load(last_checkpoint_path, map_location="cpu", weights_only=False)
        validate_checkpoint_contract(checkpoint, expected_artifact_hashes=artifact_hashes)
        model.load_state_dict(checkpoint.model_state)
        head.load_state_dict(checkpoint.prediction_head_state)
        optimizer.load_state_dict(checkpoint.optimizer_state)
        scheduler.load_state_dict(checkpoint.scheduler_state)
        for name, param in model.named_parameters():
            grad_key = f"model.{name}"
            if grad_key in checkpoint.gradient_state:
                param.grad = checkpoint.gradient_state[grad_key].to(param.device)
        for name, param in head.named_parameters():
            grad_key = f"head.{name}"
            if grad_key in checkpoint.gradient_state:
                param.grad = checkpoint.gradient_state[grad_key].to(param.device)
        accumulation_step = checkpoint.accumulation_step
        start_epoch = checkpoint.epoch
        global_batch = checkpoint.global_batch
        optimizer_step = checkpoint.optimizer_step
        if checkpoint.best_metric is not None:
            best_metric = checkpoint.best_metric
        best_epoch = checkpoint.best_epoch
        training_history = checkpoint.training_history
        random.setstate(checkpoint.rng_state["python"])
        np.random.set_state(checkpoint.rng_state["numpy"])
        torch.set_rng_state(checkpoint.rng_state["torch"])
        sampler.load_state_dict({
            "seed": checkpoint.sampler_state.get("seed", seed),
            "epoch": checkpoint.epoch,
            "cursor": checkpoint.sampler_state["cursor"],
        })
        sampler.permutation = checkpoint.sampler_state["permutation"]

    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.exists():
        candidates = []
        for p in checkpoints_dir.glob("checkpoint_step_*.pt"):
            try:
                step_val = int(p.stem.split("_")[-1])
                candidates.append((step_val, p))
            except ValueError:
                pass
        for p in checkpoints_dir.glob("checkpoint_epoch_*.pt"):
            try:
                epoch_val = int(p.stem.split("_")[-1])
                candidates.append((epoch_val * 1000000, p))
            except ValueError:
                pass
        candidates.sort()
        periodic_checkpoints = [p for _, p in candidates]

    run_id = config.get("experiment", {}).get("name", "pretrain_run")
    state = RunState(
        run_id=run_id,
        status="running",
        updated_at=datetime.utcnow().isoformat() + "Z",
        artifact_hashes=artifact_hashes,
        last_checkpoint=str(last_checkpoint_path) if last_checkpoint_path else None
    )
    write_run_state(run_dir, state)

    log_event(run_dir, {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "stage": "pretrain",
        "status": "started",
        "epoch": start_epoch,
        "batch": global_batch,
        "optimizer_step": optimizer_step,
        "learning_rate": optimizer.param_groups[0]["lr"],
    })

    device = torch.device(run_conf.get("device", "cpu"))
    model.to(device)
    head.to(device)

    class TrainingState:
        def __init__(self) -> None:
            self.interrupted = False

    ts = TrainingState()

    def handle_sig(signum, frame):
        ts.interrupted = True

    try:
        signal.signal(signal.SIGINT, handle_sig)
        signal.signal(signal.SIGTERM, handle_sig)
    except ValueError:
        pass

    recovery_conf = config.get("recovery", {})
    checkpoint_every_opt_steps = recovery_conf.get("checkpoint_every_optimizer_steps", 100)
    log_every_batches = recovery_conf.get("log_every_batches", 10)
    keep_last = recovery_conf.get("keep_last_checkpoints", 2)
    num_epochs = train_conf.get("epochs", 5)
    batch_size = train_conf.get("batch_size", 8)
    grad_accum_steps = train_conf.get("gradient_accumulation_steps", 4)
    start_time = time.time()
    epoch_start_cursor = sampler.cursor
    epoch_start_global_batch = global_batch

    try:
        for epoch in range(start_epoch, num_epochs):
            if epoch > start_epoch or last_checkpoint_path is None:
                sampler.set_epoch(epoch)
            collate_wrapper.epoch = epoch
            epoch_start_cursor = sampler.cursor
            epoch_start_global_batch = global_batch
            train_loader = create_dataloader(
                dataset=train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                collator=collate_wrapper,
                num_workers=run_conf.get("num_workers", 0)
            )

            model.train()
            head.train()

            for batch in train_loader:
                if ts.interrupted or (interrupt_after_batches is not None and global_batch == interrupt_after_batches):
                    reason = "emergency" if ts.interrupted else "interrupted"
                    grad_state = {}
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_state[f"model.{name}"] = param.grad.detach().clone()
                    for name, param in head.named_parameters():
                        if param.grad is not None:
                            grad_state[f"head.{name}"] = param.grad.detach().clone()
                    current_epoch_batches = global_batch - epoch_start_global_batch
                    correct_cursor = min(epoch_start_cursor + current_epoch_batches * batch_size, len(sampler.permutation))
                    checkpoint = CheckpointContract(
                        run_id=run_id,
                        model_state=model.state_dict(),
                        prediction_head_state=head.state_dict(),
                        optimizer_state=optimizer.state_dict(),
                        scheduler_state=scheduler.state_dict(),
                        gradient_state=grad_state,
                        accumulation_step=accumulation_step,
                        epoch=epoch,
                        next_batch_cursor=correct_cursor,
                        global_batch=global_batch,
                        optimizer_step=optimizer_step,
                        best_metric=best_metric if best_metric != float("inf") else None,
                        best_epoch=best_epoch,
                        early_stopping_state={},
                        threshold_state={},
                        rng_state={
                            "python": random.getstate(),
                            "numpy": np.random.get_state(),
                            "torch": torch.get_rng_state(),
                        },
                        sampler_state={
                            "permutation": sampler.permutation,
                            "generator_state": {},
                            "cursor": correct_cursor,
                        },
                        artifact_hashes=artifact_hashes,
                        training_history=training_history,
                        creation_reason=reason,
                    )
                    checkpoint_path = save_checkpoint(checkpoint, run_dir, reason, periodic_checkpoints, keep_last)
                    state.status = "interrupted"
                    state.updated_at = datetime.utcnow().isoformat() + "Z"
                    state.last_checkpoint = str(checkpoint_path)
                    write_run_state(run_dir, state)
                    log_event(run_dir, {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "stage": "pretrain",
                        "status": "interrupted",
                        "epoch": epoch,
                        "batch": global_batch,
                        "optimizer_step": optimizer_step,
                        "checkpoint_path": str(checkpoint_path),
                    })
                    raise TrainingInterruptedException("Training was interrupted")

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                mlm_labels = batch["mlm_labels"].to(device)

                if input_ids.shape[1] > 256:
                    raise ValueError("Batch sequence length exceeds 256")

                x, _ = model(input_ids, padding_mask=attention_mask)
                logits = head(x)
                loss = compute_mlm_loss(logits, mlm_labels)
                if torch.isnan(loss):
                    raise ValueError("NaN loss encountered during training")

                loss_scaled = loss / grad_accum_steps
                loss_scaled.backward()

                global_batch += 1
                accumulation_step += 1

                if accumulation_step % grad_accum_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    optimizer_step += 1
                    accumulation_step = 0

                    if optimizer_step > 0 and optimizer_step % checkpoint_every_opt_steps == 0:
                        grad_state = {}
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                grad_state[f"model.{name}"] = param.grad.detach().clone()
                        for name, param in head.named_parameters():
                            if param.grad is not None:
                                grad_state[f"head.{name}"] = param.grad.detach().clone()
                        current_epoch_batches = global_batch - epoch_start_global_batch
                        correct_cursor = min(epoch_start_cursor + current_epoch_batches * batch_size, len(sampler.permutation))
                        checkpoint = CheckpointContract(
                            run_id=run_id,
                            model_state=model.state_dict(),
                            prediction_head_state=head.state_dict(),
                            optimizer_state=optimizer.state_dict(),
                            scheduler_state=scheduler.state_dict(),
                            gradient_state=grad_state,
                            accumulation_step=accumulation_step,
                            epoch=epoch,
                            next_batch_cursor=correct_cursor,
                            global_batch=global_batch,
                            optimizer_step=optimizer_step,
                            best_metric=best_metric if best_metric != float("inf") else None,
                            best_epoch=best_epoch,
                            early_stopping_state={},
                            threshold_state={},
                            rng_state={
                                "python": random.getstate(),
                                "numpy": np.random.get_state(),
                                "torch": torch.get_rng_state(),
                            },
                            sampler_state={
                                "permutation": sampler.permutation,
                                "generator_state": {},
                                "cursor": correct_cursor,
                            },
                            artifact_hashes=artifact_hashes,
                            training_history=training_history,
                            creation_reason=f"step_{optimizer_step}",
                        )
                        checkpoint_path = save_checkpoint(checkpoint, run_dir, f"step_{optimizer_step}", periodic_checkpoints, keep_last)
                        state.updated_at = datetime.utcnow().isoformat() + "Z"
                        state.last_checkpoint = str(checkpoint_path)
                        write_run_state(run_dir, state)

                if global_batch % log_every_batches == 0:
                    log_event(run_dir, {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "stage": "pretrain",
                        "status": "running",
                        "epoch": epoch,
                        "batch": global_batch,
                        "optimizer_step": optimizer_step,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "loss": loss.item(),
                        "elapsed_time": time.time() - start_time,
                    })

            val_loader = create_dataloader(
                dataset=val_dataset,
                batch_size=batch_size,
                sampler=ResumableDeterministicSampler(len(val_dataset), seed=seed, epoch=epoch),
                collator=collate_wrapper,
                num_workers=0
            )

            model.eval()
            head.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_input_ids = val_batch["input_ids"].to(device)
                    val_attention_mask = val_batch["attention_mask"].to(device)
                    val_mlm_labels = val_batch["mlm_labels"].to(device)
                    val_x, _ = model(val_input_ids, padding_mask=val_attention_mask)
                    val_logits = head(val_x)
                    val_loss += compute_mlm_loss(val_logits, val_mlm_labels).item()
                    val_batches += 1
            mean_val_loss = val_loss / val_batches if val_batches > 0 else 0.0

            training_history["train_losses"].append(0.0)
            training_history["val_losses"].append(mean_val_loss)

            is_best = False
            if mean_val_loss < best_metric:
                best_metric = mean_val_loss
                best_epoch = epoch
                is_best = True

            grad_state = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_state[f"model.{name}"] = param.grad.detach().clone()
            for name, param in head.named_parameters():
                if param.grad is not None:
                    grad_state[f"head.{name}"] = param.grad.detach().clone()
            current_epoch_batches = global_batch - epoch_start_global_batch
            correct_cursor = min(epoch_start_cursor + current_epoch_batches * batch_size, len(sampler.permutation))
            checkpoint = CheckpointContract(
                run_id=run_id,
                model_state=model.state_dict(),
                prediction_head_state=head.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                gradient_state=grad_state,
                accumulation_step=accumulation_step,
                epoch=epoch,
                next_batch_cursor=correct_cursor,
                global_batch=global_batch,
                optimizer_step=optimizer_step,
                best_metric=best_metric if best_metric != float("inf") else None,
                best_epoch=best_epoch,
                early_stopping_state={},
                threshold_state={},
                rng_state={
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                },
                sampler_state={
                    "permutation": sampler.permutation,
                    "generator_state": {},
                    "cursor": correct_cursor,
                },
                artifact_hashes=artifact_hashes,
                training_history=training_history,
                creation_reason=f"epoch_{epoch}",
            )
            checkpoint_path = save_checkpoint(checkpoint, run_dir, f"epoch_{epoch}", periodic_checkpoints, keep_last)
            state.updated_at = datetime.utcnow().isoformat() + "Z"
            state.last_checkpoint = str(checkpoint_path)
            write_run_state(run_dir, state)

            if is_best:
                save_checkpoint(checkpoint, run_dir, "best", periodic_checkpoints, keep_last)

            log_event(run_dir, {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "stage": "pretrain",
                "status": "validation",
                "epoch": epoch,
                "batch": global_batch,
                "optimizer_step": optimizer_step,
                "val_loss": mean_val_loss,
            })

        grad_state = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_state[f"model.{name}"] = param.grad.detach().clone()
        for name, param in head.named_parameters():
            if param.grad is not None:
                grad_state[f"head.{name}"] = param.grad.detach().clone()
        current_epoch_batches = global_batch - epoch_start_global_batch
        correct_cursor = min(epoch_start_cursor + current_epoch_batches * batch_size, len(sampler.permutation))
        checkpoint = CheckpointContract(
            run_id=run_id,
            model_state=model.state_dict(),
            prediction_head_state=head.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            gradient_state=grad_state,
            accumulation_step=accumulation_step,
            epoch=num_epochs - 1,
            next_batch_cursor=correct_cursor,
            global_batch=global_batch,
            optimizer_step=optimizer_step,
            best_metric=best_metric if best_metric != float("inf") else None,
            best_epoch=best_epoch,
            early_stopping_state={},
            threshold_state={},
            rng_state={
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
            sampler_state={
                "permutation": sampler.permutation,
                "generator_state": {},
                "cursor": correct_cursor,
            },
            artifact_hashes=artifact_hashes,
            training_history=training_history,
            creation_reason="final",
        )
        checkpoint_path = save_checkpoint(checkpoint, run_dir, "final", periodic_checkpoints, keep_last)
        state.status = "completed"
        state.updated_at = datetime.utcnow().isoformat() + "Z"
        state.last_checkpoint = str(checkpoint_path)
        write_run_state(run_dir, state)

        log_event(run_dir, {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "stage": "pretrain",
            "status": "completed",
            "epoch": num_epochs - 1,
            "batch": global_batch,
            "optimizer_step": optimizer_step,
        })

    except TrainingInterruptedException:
        raise
    except Exception as e:
        state.status = "failed"
        state.updated_at = datetime.utcnow().isoformat() + "Z"
        write_run_state(run_dir, state)
        log_event(run_dir, {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "stage": "pretrain",
            "status": "failed",
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
        })
        raise

    return {
        "model_state": model.state_dict(),
        "prediction_head_state": head.state_dict(),
        "val_loss": best_metric,
    }


def run_pretraining() -> None:
    parser = argparse.ArgumentParser(description="Run masked event pretraining.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--processed_dir", type=Path, default=None)
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--resume", choices=["auto", "no"], default="auto")
    args = parser.parse_args()

    config = load_yaml(args.config)
    validate_final_config(config)

    processed_dir = args.processed_dir
    if processed_dir is None:
        processed_dir = Path(config["data"]["processed_dir"])

    run_dir = args.run_dir
    if run_dir is None:
        run_name = config.get("experiment", {}).get("name", "pretrain_run")
        run_dir = Path("results/runs") / run_name

    train_model(
        config=config,
        processed_dir=processed_dir,
        run_dir=run_dir,
        resume=args.resume,
    )
