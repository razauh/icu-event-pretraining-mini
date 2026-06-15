"""Optuna search for CPU-friendly Transformer configs."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from icu_pretrain.experiments.tracking import log_event, write_run_state
from icu_pretrain.training.finetune import TrainingInterruptedException, train_finetuning_model
from icu_pretrain.training.pretrain import train_model
from icu_pretrain.utils import load_yaml, validate_final_config, validate_tuning_config


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    temporary_path.replace(path)


def _hash_config(config: dict[str, Any]) -> str:
    serialized = json.dumps(config, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _import_optuna() -> Any:
    try:
        return importlib.import_module("optuna")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Optuna is required for tuning. Install the optional tuning extra with `python -m pip install -e .[tuning]`."
        ) from exc


def _default_run_dir(config: dict[str, Any], config_path: Path) -> Path:
    study = config["study"]
    storage = str(study.get("storage", ""))
    if storage.startswith("sqlite:///"):
        db_path = Path(storage.removeprefix("sqlite:///"))
        return db_path.parent / study["name"]
    return config_path.parent.parent.parent / "results" / "tuning" / study["name"]


def _sample_trial_params(trial: Any, search_space: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, choices in search_space.items():
        if key == "max_seq_len":
            continue
        params[key] = trial.suggest_categorical(key, list(choices))
    return params


def _trial_state_name(trial: Any) -> str:
    state = getattr(trial, "state", None)
    if hasattr(state, "name"):
        return str(state.name)
    return str(state)


def _completed_trials(study: Any) -> list[Any]:
    return [trial for trial in getattr(study, "trials", []) if _trial_state_name(trial).upper() == "COMPLETE"]


def _build_trial_config(
    base_config: dict[str, Any],
    tuning_config: dict[str, Any],
    trial_params: dict[str, Any],
    trial_name: str,
) -> dict[str, Any]:
    trial_config = deepcopy(base_config)
    experiment = trial_config.setdefault("experiment", {})
    experiment["id"] = "OPTUNA_TUNING"
    experiment["name"] = trial_name
    model = trial_config.setdefault("model", {})
    for key in ("d_model", "n_layers", "n_heads", "dim_feedforward", "dropout"):
        model[key] = trial_params[key]
    pretraining = trial_config.setdefault("pretraining", {})
    finetuning = trial_config.setdefault("finetuning", {})
    pretraining["epochs"] = tuning_config["trial_training"]["pretrain_epochs"]
    finetuning["epochs"] = tuning_config["trial_training"]["finetune_epochs"]
    pretraining["mask_probability"] = trial_params["mask_probability"]
    pretraining["learning_rate"] = trial_params["learning_rate"]
    finetuning["learning_rate"] = trial_params["learning_rate"]
    finetuning["evaluate_on_test"] = False
    return validate_final_config(trial_config)


def _trial_distribution_map(optuna_module: Any, search_space: dict[str, Any]) -> dict[str, Any]:
    distributions: dict[str, Any] = {}
    for key, choices in search_space.items():
        if key == "max_seq_len":
            continue
        distributions[key] = optuna_module.distributions.CategoricalDistribution(tuple(choices))
    return distributions


def _create_completed_trial(optuna_module: Any, search_space: dict[str, Any], params: dict[str, Any], value: float) -> Any:
    distributions = _trial_distribution_map(optuna_module, search_space)
    return optuna_module.trial.create_trial(
        params=params,
        distributions={key: distributions[key] for key in params},
        value=value,
        state=optuna_module.trial.TrialState.COMPLETE,
    )


def _write_trial_manifest(
    trial_dir: Path,
    trial_number: int,
    params: dict[str, Any],
    status: str,
    value: float | None = None,
    error: str | None = None,
) -> None:
    manifest = {
        "trial_number": trial_number,
        "params": params,
        "status": status,
        "updated_at": _now(),
    }
    if value is not None:
        manifest["value"] = value
    if error is not None:
        manifest["error"] = error
    _atomic_write_json(trial_dir / "trial.json", manifest)


def _run_trial(
    base_config: dict[str, Any],
    tuning_config: dict[str, Any],
    processed_dir: Path,
    trial_dir: Path,
    trial_number: int,
    trial_params: dict[str, Any],
    resume: str,
) -> float:
    trial_name = f"{base_config.get('experiment', {}).get('name', 'optuna_tuning')}_trial_{trial_number:03d}"
    trial_config = _build_trial_config(base_config, tuning_config, trial_params, trial_name)
    pretrain_dir = trial_dir / "pretrain"
    finetune_dir = trial_dir / "finetune"
    _write_trial_manifest(trial_dir, trial_number, trial_params, "running")
    try:
        train_model(trial_config, processed_dir, pretrain_dir, resume=resume)
        pretrain_checkpoint = pretrain_dir / "checkpoints" / "best.pt"
        train_finetuning_model(
            trial_config,
            processed_dir,
            finetune_dir,
            resume=resume,
            pretrain_checkpoint=pretrain_checkpoint,
        )
        results_path = finetune_dir / "results.json"
        if not results_path.is_file():
            raise FileNotFoundError(f"results.json not found in {finetune_dir}")
        with open(results_path, "r", encoding="utf-8") as handle:
            results = json.load(handle)
        value = results.get("val_ap")
        if value is None:
            raise ValueError(f"val_ap missing from {results_path}")
        value = float(value)
    except TrainingInterruptedException as exc:
        _write_trial_manifest(trial_dir, trial_number, trial_params, "interrupted", error=f"{type(exc).__name__}: {exc}")
        raise
    except Exception as exc:
        _write_trial_manifest(trial_dir, trial_number, trial_params, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    _write_trial_manifest(trial_dir, trial_number, trial_params, "completed", value=value)
    return value


def _resume_incomplete_trials(
    study: Any,
    optuna_module: Any,
    run_dir: Path,
    search_space: dict[str, Any],
    base_config: dict[str, Any],
    tuning_config: dict[str, Any],
    processed_dir: Path,
    resume: str,
) -> int:
    resumed = 0
    for manifest_path in sorted(run_dir.glob("trials/trial_*/trial.json")):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if str(manifest.get("status")) not in {"running", "interrupted"}:
            continue
        trial_number = int(manifest["trial_number"])
        trial_dir = manifest_path.parent
        params = dict(manifest.get("params", {}))
        value = _run_trial(
            base_config=base_config,
            tuning_config=tuning_config,
            processed_dir=processed_dir,
            trial_dir=trial_dir,
            trial_number=trial_number,
            trial_params=params,
            resume=resume,
        )
        study.add_trial(_create_completed_trial(optuna_module, search_space, params, value))
        resumed += 1
    return resumed


def _make_summary(study: Any, requested_trials: int, study_name: str, storage: str) -> dict[str, Any]:
    completed = _completed_trials(study)
    best_trial = max(completed, key=lambda trial: float(trial.value)) if completed else None
    summary: dict[str, Any] = {
        "study_name": study_name,
        "storage": storage,
        "requested_trials": requested_trials,
        "completed_trials": len(completed),
        "failed_trials": sum(1 for trial in getattr(study, "trials", []) if _trial_state_name(trial).upper() == "FAIL"),
        "pruned_trials": sum(1 for trial in getattr(study, "trials", []) if _trial_state_name(trial).upper() == "PRUNED"),
        "best_trial_number": None,
        "best_value": None,
        "best_params": None,
    }
    if best_trial is not None:
        summary["best_trial_number"] = getattr(best_trial, "number", None)
        summary["best_value"] = float(best_trial.value)
        summary["best_params"] = dict(getattr(best_trial, "params", {}))
    return summary


def run_optuna_search(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter tuning.")
    parser.add_argument("--config", type=Path, default=Path("configs/tuning/optuna_eicu_tiny.yaml"))
    parser.add_argument("--base_config", type=Path, default=Path("configs/final/eicu_demo_final_tiny.yaml"))
    parser.add_argument("--processed_dir", type=Path, default=None)
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--n_trials", type=int, default=None)
    parser.add_argument("--resume", choices=["auto", "no"], default="auto")
    args = parser.parse_args(argv)

    tuning_config = validate_tuning_config(load_yaml(args.config))
    base_config = validate_final_config(load_yaml(args.base_config))

    if args.n_trials is not None:
        if not 1 <= args.n_trials <= 8:
            raise ValueError("n_trials must be between 1 and 8")
        tuning_config["study"]["n_trials"] = args.n_trials

    processed_dir = args.processed_dir or Path(base_config["data"]["processed_dir"])
    if not processed_dir.is_dir():
        raise FileNotFoundError(f"processed_dir {processed_dir} does not exist")

    run_dir = args.run_dir or _default_run_dir(tuning_config, args.config)
    run_dir.mkdir(parents=True, exist_ok=True)

    artifact_hashes = {
        "base_config": _hash_config(base_config),
        "tuning_config": _hash_config(tuning_config),
    }
    write_run_state(
        run_dir,
        {
            "run_id": tuning_config["study"]["name"],
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
            "stage": "tuning",
            "status": "started",
            "study_name": tuning_config["study"]["name"],
            "requested_trials": tuning_config["study"]["n_trials"],
        },
        stage="tuning",
    )

    optuna_module = _import_optuna()
    study = optuna_module.create_study(
        study_name=tuning_config["study"]["name"],
        storage=tuning_config["study"]["storage"],
        direction=tuning_config["study"]["direction"],
        load_if_exists=True,
    )
    search_space = tuning_config["search_space"]
    _resume_incomplete_trials(
        study=study,
        optuna_module=optuna_module,
        run_dir=run_dir,
        search_space=search_space,
        base_config=base_config,
        tuning_config=tuning_config,
        processed_dir=processed_dir,
        resume=args.resume,
    )

    target_trials = min(int(tuning_config["study"]["n_trials"]), 8)
    executed_trials = len(getattr(study, "trials", []))
    while executed_trials < target_trials:
        trial = study.ask()
        trial_params = _sample_trial_params(trial, search_space)
        trial_number = int(getattr(trial, "number", executed_trials))
        trial_dir = run_dir / "trials" / f"trial_{trial_number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        log_event(
            run_dir,
            {
                "timestamp": _now(),
                "stage": "tuning",
                "status": "trial_started",
                "trial_number": trial_number,
            },
            stage="tuning",
        )
        try:
            value = _run_trial(
                base_config=base_config,
                tuning_config=tuning_config,
                processed_dir=processed_dir,
                trial_dir=trial_dir,
                trial_number=trial_number,
                trial_params=trial_params,
                resume=args.resume,
            )
        except TrainingInterruptedException:
            study.tell(trial, state=optuna_module.trial.TrialState.FAIL)
            break
        except Exception as exc:
            study.tell(trial, state=optuna_module.trial.TrialState.FAIL)
            log_event(
                run_dir,
                {
                    "timestamp": _now(),
                    "stage": "tuning",
                    "status": "trial_failed",
                    "trial_number": trial_number,
                    "error_type": type(exc).__name__,
                },
                stage="tuning",
            )
            executed_trials = len(getattr(study, "trials", []))
            continue
        study.tell(trial, float(value))
        executed_trials = len(getattr(study, "trials", []))
        log_event(
            run_dir,
            {
                "timestamp": _now(),
                "stage": "tuning",
                "status": "trial_completed",
                "trial_number": trial_number,
                "value": float(value),
            },
            stage="tuning",
        )

    summary = _make_summary(
        study=study,
        requested_trials=target_trials,
        study_name=tuning_config["study"]["name"],
        storage=tuning_config["study"]["storage"],
    )
    _atomic_write_json(run_dir / "study_summary.json", summary)
    if summary["best_params"] is not None:
        _atomic_write_json(run_dir / "best_params.json", summary["best_params"])
    write_run_state(
        run_dir,
        {
            "run_id": tuning_config["study"]["name"],
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
            "stage": "tuning",
            "status": "completed",
            "completed_trials": summary["completed_trials"],
            "failed_trials": summary["failed_trials"],
            "best_value": summary["best_value"],
        },
        stage="tuning",
    )
    return summary
