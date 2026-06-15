import time
import json
import random
import signal
import hashlib
import traceback
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW

from icu_pretrain.data.dataset import EncodedDataset
from icu_pretrain.data.collate import (
    ResumableDeterministicSampler,
    SupervisedCollator,
    create_dataloader
)
from icu_pretrain.models.transformer import ICUTinyTransformer
from icu_pretrain.models.heads import MortalityPredictionHead, compute_mortality_loss
from icu_pretrain.data.eicu_event_builder import CheckpointContract, RunState, validate_checkpoint_contract
from icu_pretrain.training.evaluate import (
    compute_binary_metrics,
    find_best_f1_threshold,
    bootstrap_patient_metrics
)
from icu_pretrain.utils import load_yaml, validate_final_config
from icu_pretrain.training.pretrain import (
    get_config_hash,
    load_artifact_hashes,
    write_run_state,
    log_event,
    save_checkpoint
)

class TrainingInterruptedException(Exception):
    pass

class TrainingState:
    def __init__(self) -> None:
        self.interrupted = False


def train_finetuning_model(
    config: dict[str, Any],
    processed_dir: Path,
    run_dir: Path,
    resume: str = "auto",
    pretrain_checkpoint: Path | None = None,
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
    test_dir = encoded_dir / "test"
    if not train_dir.is_dir() or not val_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(f"split directories not found in {encoded_dir}")
    train_dataset = EncodedDataset(train_dir)
    val_dataset = EncodedDataset(val_dir)
    test_dataset = EncodedDataset(test_dir)
    if len(train_dataset) == 0:
        raise ValueError("Training dataset has zero stays")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset has zero stays")
    if len(test_dataset) == 0:
        raise ValueError("Test dataset has zero stays")
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
    head = MortalityPredictionHead(
        d_model=model_conf.get("d_model", 64),
    )

    if last_checkpoint_path is None and pretrain_checkpoint is not None and pretrain_checkpoint.is_file():
        pt_checkpoint = torch.load(pretrain_checkpoint, map_location="cpu", weights_only=False)
        if hasattr(pt_checkpoint, "model_state"):
            model.load_state_dict(pt_checkpoint.model_state)
        else:
            model.load_state_dict(pt_checkpoint)

    train_conf = config.get("finetuning", {})
    lr = train_conf.get("learning_rate", 0.0003)
    wd = train_conf.get("weight_decay", 0.01)
    
    freeze = train_conf.get("freeze_encoder", False)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        opt_params = list(head.parameters())
    else:
        for param in model.parameters():
            param.requires_grad = True
        opt_params = list(model.parameters()) + list(head.parameters())

    optimizer = AdamW(
        opt_params,
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
    collator = SupervisedCollator(vocab=vocab)

    start_epoch = 0
    global_batch = 0
    optimizer_step = 0
    accumulation_step = 0
    best_metric = -1.0
    best_epoch = None
    patience_counter = 0
    training_history = {"train_losses": [], "val_losses": []}
    periodic_checkpoints = []

    if last_checkpoint_path is not None:
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
        patience_counter = checkpoint.early_stopping_state.get("patience_counter", 0)
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

    run_id = config.get("experiment", {}).get("name", "finetune_run")
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
        "stage": "finetune",
        "status": "started",
        "epoch": start_epoch,
        "batch": global_batch,
        "optimizer_step": optimizer_step,
        "learning_rate": optimizer.param_groups[0]["lr"],
    })

    device = torch.device(run_conf.get("device", "cpu"))
    model.to(device)
    head.to(device)

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
    num_epochs = train_conf.get("epochs", 10)
    batch_size = train_conf.get("batch_size", 8)
    patience = train_conf.get("early_stopping_patience", 3)
    grad_accum_steps = train_conf.get("gradient_accumulation_steps", 1)
    start_time = time.time()
    
    train_stays = train_dataset.stays()
    train_labels = [stay.label for stay in train_stays]
    num_neg = sum(1 for label in train_labels if label == 0)
    num_pos = sum(1 for label in train_labels if label == 1)
    if num_pos == 0:
        pos_weight = torch.tensor(1.0)
    else:
        pos_weight = torch.tensor(num_neg / num_pos)

    epoch_start_cursor = sampler.cursor
    epoch_start_global_batch = global_batch

    try:
        early_stopped = False
        for epoch in range(start_epoch, num_epochs):
            if early_stopped:
                break
            if epoch > start_epoch or last_checkpoint_path is None:
                sampler.set_epoch(epoch)
            epoch_start_cursor = sampler.cursor
            epoch_start_global_batch = global_batch

            train_loader = create_dataloader(
                dataset=train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                collator=collator,
                num_workers=0
            )

            model.train()
            head.train()

            for batch in train_loader:
                if ts.interrupted:
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
                        best_metric=best_metric if best_metric >= 0.0 else None,
                        best_epoch=best_epoch,
                        early_stopping_state={"patience_counter": patience_counter},
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
                        creation_reason="interrupted",
                    )
                    checkpoint_path = save_checkpoint(checkpoint, run_dir, "interrupted", periodic_checkpoints, keep_last)
                    state.status = "interrupted"
                    state.updated_at = datetime.utcnow().isoformat() + "Z"
                    state.last_checkpoint = str(checkpoint_path)
                    write_run_state(run_dir, state)
                    log_event(run_dir, {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "stage": "finetune",
                        "status": "interrupted",
                        "epoch": epoch,
                        "batch": global_batch,
                        "optimizer_step": optimizer_step,
                        "checkpoint_path": str(checkpoint_path),
                    })
                    raise TrainingInterruptedException("Training was interrupted")

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["labels"].to(device)

                if input_ids.shape[1] > 256:
                    raise ValueError("Batch sequence length exceeds 256")

                _, cls_output = model(input_ids, padding_mask=attention_mask)
                logits = head(cls_output)
                loss = compute_mortality_loss(logits, targets, pos_weight=pos_weight.to(device))
                if torch.isnan(loss):
                    raise ValueError("Loss is NaN")

                loss = loss / grad_accum_steps
                loss.backward()

                accumulation_step += 1
                if accumulation_step % grad_accum_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    optimizer_step += 1

                    if optimizer_step % checkpoint_every_opt_steps == 0:
                        grad_state = {}
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                grad_state[f"model.{name}"] = param.grad.detach().clone()
                        for name, param in head.named_parameters():
                            if param.grad is not None:
                                grad_state[f"head.{name}"] = param.grad.detach().clone()
                        current_epoch_batches = global_batch + 1 - epoch_start_global_batch
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
                            best_metric=best_metric if best_metric >= 0.0 else None,
                            best_epoch=best_epoch,
                            early_stopping_state={"patience_counter": patience_counter},
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
                        save_checkpoint(checkpoint, run_dir, f"step_{optimizer_step}", periodic_checkpoints, keep_last)

                if interrupt_after_batches is not None and global_batch >= interrupt_after_batches:
                    ts.interrupted = True

                if global_batch % log_every_batches == 0:
                    log_event(run_dir, {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "stage": "finetune",
                        "status": "running",
                        "epoch": epoch,
                        "batch": global_batch,
                        "optimizer_step": optimizer_step,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "loss": loss.item() * grad_accum_steps,
                        "elapsed_time": time.time() - start_time,
                    })

                global_batch += 1

            val_loader = create_dataloader(
                dataset=val_dataset,
                batch_size=batch_size,
                sampler=ResumableDeterministicSampler(len(val_dataset), seed=seed, epoch=epoch),
                collator=collator,
                num_workers=0
            )

            model.eval()
            head.eval()
            val_probs = []
            val_targets = []
            with torch.no_grad():
                for val_batch in val_loader:
                    val_input_ids = val_batch["input_ids"].to(device)
                    val_attention_mask = val_batch["attention_mask"].to(device)
                    val_y = val_batch["labels"].to(device)
                    _, cls_output = model(val_input_ids, padding_mask=val_attention_mask)
                    val_logits = head(cls_output)
                    probs = torch.sigmoid(val_logits)
                    val_probs.extend(probs.cpu().numpy())
                    val_targets.extend(val_y.cpu().numpy())

            val_targets = np.array(val_targets)
            val_probs = np.array(val_probs)
            val_metrics = compute_binary_metrics(val_targets, val_probs)
            val_ap = val_metrics["ap"]
            if np.isnan(val_ap):
                val_ap = 0.0

            training_history["train_losses"].append(0.0)
            training_history["val_losses"].append(val_ap)

            is_best = False
            if val_ap > best_metric:
                best_metric = val_ap
                best_epoch = epoch
                is_best = True
                patience_counter = 0
            else:
                patience_counter += 1

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
                best_metric=best_metric if best_metric >= 0.0 else None,
                best_epoch=best_epoch,
                early_stopping_state={"patience_counter": patience_counter},
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
                "stage": "finetune",
                "status": "validation",
                "epoch": epoch,
                "batch": global_batch,
                "optimizer_step": optimizer_step,
                "val_ap": val_ap,
            })

            if patience_counter >= patience:
                log_event(run_dir, {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "stage": "finetune",
                    "status": "early_stopping",
                    "epoch": epoch,
                    "batch": global_batch,
                    "optimizer_step": optimizer_step,
                })
                early_stopped = True

        best_checkpoint_path = run_dir / "checkpoints" / "best.pt"
        if not best_checkpoint_path.exists():
            checkpoint_path = save_checkpoint(checkpoint, run_dir, "best", periodic_checkpoints, keep_last)
        best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best_checkpoint.model_state)
        head.load_state_dict(best_checkpoint.prediction_head_state)

        model.eval()
        head.eval()

        val_probs = []
        val_targets = []
        val_loader = create_dataloader(
            dataset=val_dataset,
            batch_size=batch_size,
            sampler=ResumableDeterministicSampler(len(val_dataset), seed=seed, epoch=0),
            collator=collator,
            num_workers=0
        )
        with torch.no_grad():
            for batch in val_loader:
                val_input_ids = batch["input_ids"].to(device)
                val_attention_mask = batch["attention_mask"].to(device)
                val_y = batch["labels"].to(device)
                _, cls_output = model(val_input_ids, padding_mask=val_attention_mask)
                val_logits = head(cls_output)
                probs = torch.sigmoid(val_logits)
                val_probs.extend(probs.cpu().numpy())
                val_targets.extend(val_y.cpu().numpy())

        val_targets = np.array(val_targets)
        val_probs = np.array(val_probs)
        best_threshold = find_best_f1_threshold(val_targets, val_probs)

        test_stays = test_dataset.stays()
        test_loader = create_dataloader(
            dataset=test_dataset,
            batch_size=batch_size,
            sampler=ResumableDeterministicSampler(len(test_dataset), seed=seed, epoch=0),
            collator=collator,
            num_workers=0
        )
        test_probs = []
        test_targets = []
        with torch.no_grad():
            for batch in test_loader:
                test_input_ids = batch["input_ids"].to(device)
                test_attention_mask = batch["attention_mask"].to(device)
                test_y = batch["labels"].to(device)
                _, cls_output = model(test_input_ids, padding_mask=test_attention_mask)
                test_logits = head(cls_output)
                probs = torch.sigmoid(test_logits)
                test_probs.extend(probs.cpu().numpy())
                test_targets.extend(test_y.cpu().numpy())

        test_targets = np.array(test_targets)
        test_probs = np.array(test_probs)
        test_metrics = compute_binary_metrics(test_targets, test_probs, threshold=best_threshold)

        split_metadata_path = processed_dir / "split_metadata.json"
        if not split_metadata_path.exists():
            raise FileNotFoundError(f"split_metadata.json not found in {processed_dir}")
        with open(split_metadata_path, "r", encoding="utf-8") as f:
            split_metadata = json.load(f)
        stay_to_patient = {str(r["patientunitstayid"]): str(r["uniquepid"]) for r in split_metadata}
        test_patients = [stay_to_patient[str(stay.patientunitstayid)] for stay in test_stays]

        bootstrap_results = bootstrap_patient_metrics(
            patients=test_patients,
            y_true=test_targets,
            y_prob=test_probs,
            threshold=best_threshold,
            n_replicates=1000,
            seed=seed
        )

        runtime = time.time() - start_time
        param_count = model.parameter_count + sum(p.numel() for p in head.parameters())
        num_patients = len(np.unique(test_patients))
        num_stays = len(test_stays)
        alive_count = int(np.sum(test_targets == 0))
        expired_count = int(np.sum(test_targets == 1))

        results = {
            "experiment_id": config.get("experiment", {}).get("id", "EXP-01"),
            "representation": config.get("representation", "timegap_static"),
            "num_patients": num_patients,
            "num_stays": num_stays,
            "alive_count": alive_count,
            "expired_count": expired_count,
            "split_strategy": config.get("evaluation", {}).get("split", "patient_grouped_test"),
            "seed": seed,
            "auroc": test_metrics["auroc"],
            "auroc_ci": bootstrap_results["auroc_ci"],
            "average_precision": test_metrics["ap"],
            "average_precision_ci": bootstrap_results["ap_ci"],
            "f1": test_metrics["f1"],
            "balanced_accuracy": test_metrics["balanced_accuracy"],
            "parameter_count": param_count,
            "runtime": runtime,
        }

        results_path = run_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        state.status = "completed"
        state.updated_at = datetime.utcnow().isoformat() + "Z"
        write_run_state(run_dir, state)

        log_event(run_dir, {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "stage": "finetune",
            "status": "completed",
            "epoch": epoch,
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
            "stage": "finetune",
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

def run_finetuning() -> None:
    parser = argparse.ArgumentParser(description="Run outcome fine-tuning.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--processed_dir", type=Path, default=None)
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--resume", choices=["auto", "no"], default="auto")
    parser.add_argument("--pretrain_checkpoint", type=Path, default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    validate_final_config(config)
    processed_dir = args.processed_dir
    if processed_dir is None:
        processed_dir = Path(config["data"]["processed_dir"])
    run_dir = args.run_dir
    if run_dir is None:
        run_name = config.get("experiment", {}).get("name", "finetune_run")
        run_dir = Path("results/runs") / run_name
    train_finetuning_model(
        config=config,
        processed_dir=processed_dir,
        run_dir=run_dir,
        resume=args.resume,
        pretrain_checkpoint=args.pretrain_checkpoint,
    )
