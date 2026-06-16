import argparse
import json
import os
import sys
import traceback
import yaml
from pathlib import Path
from typing import Any

from icu_pretrain.experiments.registry import EXPERIMENT_REGISTRY, EXPERIMENT_IDS
from icu_pretrain.training.baselines import train_and_evaluate_logistic_baseline
from icu_pretrain.training.federated import run_hospital_grouped_evaluation, run_fedavg_simulation
from icu_pretrain.training.pretrain import train_model
from icu_pretrain.training.finetune import train_finetuning_model
from icu_pretrain.experiments.tracking import record_metrics, save_best_config, log_event

root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from scripts.prepare_eicu_demo import main as prepare_main

def load_val_ap(run_dir: Path) -> float:
    results_path = run_dir / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"results.json not found in {run_dir}")
    with open(results_path, "r", encoding="utf-8") as f:
        res = json.load(f)
    val_ap = res.get("val_ap")
    if val_ap is None:
        val_ap = res.get("average_precision", 0.0)
    return float(val_ap)

def record_finetune_results(exp_id: str, representation: str, results_path: Path, summary_dir: Path) -> None:
    with open(results_path, "r", encoding="utf-8") as f:
        res = json.load(f)
    run_data = {
        "experiment_id": exp_id,
        "representation": representation,
        "num_patients": res.get("num_patients", 0),
        "num_stays": res.get("num_stays", 0),
        "alive_count": res.get("alive_count", 0),
        "expired_count": res.get("expired_count", 0),
        "split_strategy": res.get("split_strategy", "patient_grouped"),
        "seed": res.get("seed", 42),
        "auroc": res.get("auroc", "") if res.get("auroc", 0.0) != 0.0 else "",
        "auroc_ci_lower": res.get("auroc_ci", [0.0, 0.0])[0] if res.get("auroc", 0.0) != 0.0 else "",
        "auroc_ci_upper": res.get("auroc_ci", [0.0, 0.0])[1] if res.get("auroc", 0.0) != 0.0 else "",
        "average_precision": res.get("average_precision", "") if res.get("average_precision", 0.0) != 0.0 else "",
        "average_precision_ci_lower": res.get("average_precision_ci", [0.0, 0.0])[0] if res.get("average_precision", 0.0) != 0.0 else "",
        "average_precision_ci_upper": res.get("average_precision_ci", [0.0, 0.0])[1] if res.get("average_precision", 0.0) != 0.0 else "",
        "f1": res.get("f1", "") if res.get("f1", 0.0) != 0.0 else "",
        "balanced_accuracy": res.get("balanced_accuracy", "") if res.get("balanced_accuracy", 0.0) != 0.0 else "",
        "parameter_count": res.get("parameter_count", 0),
        "runtime": res.get("runtime", 0.0),
        "exclusions": "",
        "failure_notes": ""
    }
    record_metrics(summary_dir, run_data)

def prepare_data_for_representation(representation: str, base_config: dict, raw_dir: Path, processed_dir: Path) -> None:
    if (processed_dir / "vocab.json").is_file():
        return
    config_copy = base_config.copy()
    if "data" not in config_copy:
        config_copy["data"] = {}
    config_copy["data"]["representation"] = representation
    config_copy["data"]["processed_dir"] = str(processed_dir)
    config_copy["data"]["raw_dir"] = str(raw_dir)
    
    temp_config_path = processed_dir.parent / f"config_temp_{representation}.yaml"
    temp_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_copy, f)
        
    try:
        prepare_main([
            "--raw_dir", str(raw_dir),
            "--out_dir", str(processed_dir),
            "--config", str(temp_config_path),
            "--resume", "auto"
        ])
    finally:
        if temp_config_path.exists():
            temp_config_path.unlink()

def run_single_experiment(
    exp_id: str,
    base_config: dict[str, Any],
    processed_dir_root: Path,
    run_dir: Path,
    summary_dir: Path,
    resume: str = "auto",
    evaluate_on_test: bool = False,
) -> None:
    if exp_id not in EXPERIMENT_REGISTRY:
        raise ValueError(f"Experiment {exp_id} is not registered.")
    
    spec = EXPERIMENT_REGISTRY[exp_id]
    representation = spec["representation"]
    
    raw_dir = Path(base_config.get("data", {}).get("raw_dir", "data/raw/eicu_demo"))
    if representation == "basic":
        processed_dir = processed_dir_root / "eicu_demo_basic"
    else:
        processed_dir = processed_dir_root / "eicu_demo"
        
    prepare_data_for_representation(representation, base_config, raw_dir, processed_dir)
    
    exp_run_dir = run_dir / exp_id
    state_file = exp_run_dir / "state.json"
    
    if resume == "auto" and state_file.is_file():
        with open(state_file, "r", encoding="utf-8") as f:
            state_data = json.load(f)
        if state_data.get("status") == "completed":
            results_path = exp_run_dir / "results.json"
            if results_path.is_file():
                if evaluate_on_test:
                    with open(results_path, "r", encoding="utf-8") as f:
                        res = json.load(f)
                    if res.get("auroc") != 0.0 and res.get("auroc") != "":
                        record_finetune_results(exp_id, representation, results_path, summary_dir)
                        return
                else:
                    record_finetune_results(exp_id, representation, results_path, summary_dir)
                    return
                    
    config_copy = yaml.safe_load(yaml.safe_dump(base_config))
    if "experiment" not in config_copy:
        config_copy["experiment"] = {}
    config_copy["experiment"]["id"] = exp_id
    config_copy["experiment"]["name"] = exp_id
    
    if "data" not in config_copy:
        config_copy["data"] = {}
    config_copy["data"]["representation"] = representation
    config_copy["data"]["processed_dir"] = str(processed_dir)
    
    if "finetuning" not in config_copy:
        config_copy["finetuning"] = {}
    config_copy["finetuning"]["evaluate_on_test"] = evaluate_on_test
    config_copy["finetuning"]["check_selection_freeze"] = True

    try:
        if exp_id == "EXP-00":
            train_and_evaluate_logistic_baseline(config_copy, processed_dir, exp_run_dir)
            results_path = exp_run_dir / "results.json"
            record_finetune_results(exp_id, representation, results_path, summary_dir)
        elif exp_id == "EXP-01":
            train_finetuning_model(config_copy, processed_dir, exp_run_dir, resume=resume, pretrain_checkpoint=None)
            results_path = exp_run_dir / "results.json"
            record_finetune_results(exp_id, representation, results_path, summary_dir)
        elif exp_id in ("EXP-02", "EXP-03"):
            pretrain_run_dir = run_dir / f"{exp_id}_pretrain"
            pretrain_state_file = pretrain_run_dir / "state.json"
            pretrain_completed = False
            if resume == "auto" and pretrain_state_file.is_file():
                with open(pretrain_state_file, "r", encoding="utf-8") as f:
                    p_state = json.load(f)
                if p_state.get("status") == "completed":
                    pretrain_completed = True
            if not pretrain_completed:
                pretrain_config = yaml.safe_load(yaml.safe_dump(config_copy))
                pretrain_config["experiment"]["name"] = f"{exp_id}_pretrain"
                pretrain_config["pretraining"]["enabled"] = True
                train_model(pretrain_config, processed_dir, pretrain_run_dir, resume=resume)
                
            pretrain_checkpoint = pretrain_run_dir / "checkpoints" / "best.pt"
            train_finetuning_model(
                config_copy,
                processed_dir,
                exp_run_dir,
                resume=resume,
                pretrain_checkpoint=pretrain_checkpoint
            )
            results_path = exp_run_dir / "results.json"
            record_finetune_results(exp_id, representation, results_path, summary_dir)
        elif exp_id == "EXP-04":
            run_hospital_grouped_evaluation(config_copy, processed_dir, exp_run_dir, resume=resume)
        elif exp_id == "EXP-05":
            run_fedavg_simulation(config_copy, processed_dir, exp_run_dir, resume=resume)
    except Exception as err:
        failed_data = {
            "experiment_id": exp_id,
            "representation": representation,
            "num_patients": 0,
            "num_stays": 0,
            "alive_count": 0,
            "expired_count": 0,
            "split_strategy": "patient_grouped",
            "seed": 42,
            "auroc": "",
            "average_precision": "",
            "f1": "",
            "balanced_accuracy": "",
            "parameter_count": "",
            "runtime": "",
            "exclusions": "",
            "failure_notes": f"{type(err).__name__}: {str(err)}"
        }
        record_metrics(summary_dir, failed_data)
        raise err

def run_experiment(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run experiment runner")
    parser.add_argument("--config", type=Path, default=Path("configs/final/eicu_demo_final_tiny.yaml"))
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--run_dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--summary_dir", type=Path, default=Path("results/summary"))
    parser.add_argument("--experiment_id", type=str, default=None)
    parser.add_argument("--resume", type=str, default="auto")
    args = parser.parse_args(argv)
    
    if not args.config.is_file():
        raise FileNotFoundError(f"Config file not found: {args.config}")
        
    with open(args.config, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)
        
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    
    if args.experiment_id:
        if args.experiment_id not in EXPERIMENT_IDS:
            raise ValueError(f"Invalid experiment_id: {args.experiment_id}")
        if args.experiment_id == "EXP-04":
            freeze_file = args.summary_dir / "selection_frozen.txt"
            if not freeze_file.exists():
                raise ValueError("Cannot run EXP-04 before selection freeze")
        run_single_experiment(
            exp_id=args.experiment_id,
            base_config=base_config,
            processed_dir_root=args.processed_dir,
            run_dir=args.run_dir,
            summary_dir=args.summary_dir,
            resume=args.resume,
            evaluate_on_test=True
        )
    else:
        log_event(args.run_dir, {"timestamp": "suite_start", "stage": "suite", "status": "started"})
        
        for exp_id in ("EXP-00", "EXP-01", "EXP-02", "EXP-03"):
            try:
                run_single_experiment(
                    exp_id=exp_id,
                    base_config=base_config,
                    processed_dir_root=args.processed_dir,
                    run_dir=args.run_dir,
                    summary_dir=args.summary_dir,
                    resume=args.resume,
                    evaluate_on_test=False
                )
            except Exception:
                pass
                
        candidate_ids = ["EXP-01", "EXP-02", "EXP-03"]
        val_aps = {}
        for c_id in candidate_ids:
            try:
                val_aps[c_id] = load_val_ap(args.run_dir / c_id)
            except Exception:
                pass
                
        if not val_aps:
            raise ValueError("All candidate runs failed; cannot proceed with model selection.")
            
        selected_id = max(candidate_ids, key=lambda x: val_aps.get(x, -1.0))
        
        freeze_file = args.summary_dir / "selection_frozen.txt"
        with open(freeze_file, "w", encoding="utf-8") as f:
            f.write(selected_id + "\n")
            
        best_config = yaml.safe_load(yaml.safe_dump(base_config))
        if "experiment" not in best_config:
            best_config["experiment"] = {}
        best_config["experiment"]["selected_id"] = selected_id
        save_best_config(args.summary_dir, best_config)
        
        run_single_experiment(
            exp_id=selected_id,
            base_config=base_config,
            processed_dir_root=args.processed_dir,
            run_dir=args.run_dir,
            summary_dir=args.summary_dir,
            resume=args.resume,
            evaluate_on_test=True
        )
        
        for exp_id in ("EXP-04", "EXP-05"):
            try:
                run_single_experiment(
                    exp_id=exp_id,
                    base_config=base_config,
                    processed_dir_root=args.processed_dir,
                    run_dir=args.run_dir,
                    summary_dir=args.summary_dir,
                    resume=args.resume,
                    evaluate_on_test=True
                )
            except Exception:
                pass
                
        log_event(args.run_dir, {"timestamp": "suite_end", "stage": "suite", "status": "completed"})
