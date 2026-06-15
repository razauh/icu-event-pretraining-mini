import json
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import yaml

from icu_pretrain.experiments.registry import EXPERIMENT_IDS, EXPERIMENT_REGISTRY
from icu_pretrain.experiments.runner import run_experiment, load_val_ap

@pytest.fixture
def temp_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = {
        "experiment": {"name": "test_exp", "seed": 42},
        "data": {
            "dataset": "eicu_demo",
            "raw_dir": str(workspace / "raw"),
            "processed_dir": str(workspace / "processed"),
            "representation": "timegap_static"
        },
        "model": {"type": "icu_tiny_transformer"},
        "pretraining": {"enabled": False},
        "finetuning": {"evaluate_on_test": False},
        "evaluation": {"split": "patient_grouped_test"}
    }
    
    config_file = workspace / "test_config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
        
    (workspace / "raw").mkdir()
    (workspace / "processed").mkdir()
    
    return workspace, config_file

def test_registry_properties():
    assert len(EXPERIMENT_IDS) == 6
    assert "EXP-00" in EXPERIMENT_REGISTRY
    assert "EXP-01" in EXPERIMENT_REGISTRY
    assert "EXP-02" in EXPERIMENT_REGISTRY
    assert "EXP-03" in EXPERIMENT_REGISTRY
    assert "EXP-04" in EXPERIMENT_REGISTRY
    assert "EXP-05" in EXPERIMENT_REGISTRY

def test_attempted_test_evaluation_before_freeze_raises(temp_workspace):
    workspace, config_file = temp_workspace
    run_dir = workspace / "runs"
    summary_dir = workspace / "summary"
    
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    config["finetuning"]["evaluate_on_test"] = True
    config["experiment"]["id"] = "EXP-01"
    
    run_dir.mkdir()
    summary_dir.mkdir()
    
    processed_dir = workspace / "processed" / "eicu_demo"
    processed_dir.mkdir(parents=True)
    
    vocab_path = processed_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"a": 0}, f)
        
    (processed_dir / "encoded" / "train").mkdir(parents=True)
    (processed_dir / "encoded" / "validation").mkdir(parents=True)
    (processed_dir / "encoded" / "test").mkdir(parents=True)
    
    with open(processed_dir / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump([], f)
        
    with patch("icu_pretrain.training.finetune.ICUTinyTransformer"), \
         patch("icu_pretrain.training.finetune.EncodedDataset") as mock_dataset, \
         patch("icu_pretrain.training.finetune.load_artifact_hashes") as mock_hashes:
         
        mock_dataset.return_value.__len__.return_value = 1
        mock_stay = MagicMock()
        mock_stay.tokens = [0, 1]
        mock_stay.label = 1
        mock_dataset.return_value.__getitem__.return_value = mock_stay
        mock_hashes.return_value = {"a": "1"}
        
        with pytest.raises(ValueError, match="Attempted test evaluation before selection freeze"):
            run_experiment([
                "--config", str(config_file),
                "--processed_dir", str(workspace / "processed"),
                "--run_dir", str(run_dir),
                "--summary_dir", str(summary_dir),
                "--experiment_id", "EXP-01",
                "--resume", "no"
            ])

def test_selection_tie_breaker(temp_workspace):
    workspace, config_file = temp_workspace
    run_dir = workspace / "runs"
    summary_dir = workspace / "summary"
    
    run_dir.mkdir()
    summary_dir.mkdir()
    
    (run_dir / "EXP-01").mkdir()
    (run_dir / "EXP-02").mkdir()
    (run_dir / "EXP-03").mkdir()
    
    with open(run_dir / "EXP-01" / "results.json", "w", encoding="utf-8") as f:
        json.dump({"val_ap": 0.8}, f)
    with open(run_dir / "EXP-02" / "results.json", "w", encoding="utf-8") as f:
        json.dump({"val_ap": 0.8}, f)
    with open(run_dir / "EXP-03" / "results.json", "w", encoding="utf-8") as f:
        json.dump({"val_ap": 0.7}, f)
        
    with patch("icu_pretrain.experiments.runner.run_single_experiment") as mock_run:
        run_experiment([
            "--config", str(config_file),
            "--processed_dir", str(workspace / "processed"),
            "--run_dir", str(run_dir),
            "--summary_dir", str(summary_dir),
            "--resume", "no"
        ])
        
        freeze_file = summary_dir / "selection_frozen.txt"
        assert freeze_file.exists()
        with open(freeze_file, "r", encoding="utf-8") as f:
            selected_id = f.read().strip()
        assert selected_id == "EXP-01"

def test_failure_isolation(temp_workspace):
    workspace, config_file = temp_workspace
    run_dir = workspace / "runs"
    summary_dir = workspace / "summary"
    
    run_dir.mkdir()
    summary_dir.mkdir()
    
    processed_dir = workspace / "processed" / "eicu_demo"
    processed_dir.mkdir(parents=True)
    with open(processed_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump({"a": 0}, f)
        
    processed_dir_basic = workspace / "processed" / "eicu_demo_basic"
    processed_dir_basic.mkdir(parents=True)
    with open(processed_dir_basic / "vocab.json", "w", encoding="utf-8") as f:
        json.dump({"a": 0}, f)
        
    with patch("icu_pretrain.experiments.runner.train_and_evaluate_logistic_baseline") as mock_baseline, \
         patch("icu_pretrain.experiments.runner.train_finetuning_model") as mock_ft, \
         patch("icu_pretrain.experiments.runner.train_model") as mock_pretrain, \
         patch("icu_pretrain.experiments.runner.load_val_ap") as mock_val_ap:
         
        mock_baseline.side_effect = Exception("Isolated failure")
        mock_val_ap.return_value = 0.8
        
        def ft_side_effect(config, processed_dir, exp_run_dir, *args, **kwargs):
            exp_run_dir.mkdir(parents=True, exist_ok=True)
            with open(exp_run_dir / "state.json", "w", encoding="utf-8") as f:
                json.dump({"status": "completed"}, f)
            with open(exp_run_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump({"val_ap": 0.8}, f)
        mock_ft.side_effect = ft_side_effect
        
        run_experiment([
            "--config", str(config_file),
            "--processed_dir", str(workspace / "processed"),
            "--run_dir", str(run_dir),
            "--summary_dir", str(summary_dir),
            "--resume", "no"
        ])
        
        assert (summary_dir / "experiment_comparison.csv").exists()

def test_no_rerun_completed_experiment(temp_workspace):
    workspace, config_file = temp_workspace
    run_dir = workspace / "runs"
    summary_dir = workspace / "summary"
    
    run_dir.mkdir()
    summary_dir.mkdir()
    
    processed_dir = workspace / "processed" / "eicu_demo"
    processed_dir.mkdir(parents=True)
    vocab_path = processed_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"a": 0}, f)
        
    exp_run_dir = run_dir / "EXP-01"
    exp_run_dir.mkdir()
    
    with open(exp_run_dir / "state.json", "w", encoding="utf-8") as f:
        json.dump({"status": "completed"}, f)
    with open(exp_run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"val_ap": 0.9}, f)
        
    with patch("icu_pretrain.experiments.runner.train_finetuning_model") as mock_ft:
        run_experiment([
            "--config", str(config_file),
            "--processed_dir", str(workspace / "processed"),
            "--run_dir", str(run_dir),
            "--summary_dir", str(summary_dir),
            "--experiment_id", "EXP-01",
            "--resume", "auto"
        ])
        
        mock_ft.assert_not_called()
