import json
import pytest
import numpy as np
from pathlib import Path
from icu_pretrain.data.dataset import EncodedDataset
from icu_pretrain.training.baselines import train_and_evaluate_logistic_baseline

def create_mock_processed_dir(tmp_path: Path, single_class_split=None) -> Path:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    vocab = {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2, "[CLS]": 3, "TOKEN_A": 4, "TOKEN_B": 5}
    with open(processed_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    split_metadata = []
    encoded_dir = processed_dir / "encoded"
    for split in ["train", "validation", "test"]:
        split_dir = encoded_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        stays = []
        for i in range(16):
            stay_id = f"100{i}"
            patient_id = f"P100{i}"
            if single_class_split == split:
                label = 0
            else:
                label = i % 2
            stays.append({
                "patientunitstayid": stay_id,
                "tokens": [3, 4, 5],
                "label": label,
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

def test_train_and_evaluate_logistic_baseline(tmp_path):
    processed_dir = create_mock_processed_dir(tmp_path)
    config = {
        "experiment": {"id": "EXP-00", "name": "logistic_test"},
        "representation": "timegap_static",
        "evaluation": {"split": "patient_grouped_test"},
        "runtime": {"seed": 42}
    }
    run_dir = tmp_path / "run"
    res = train_and_evaluate_logistic_baseline(config, processed_dir, run_dir)
    assert res["experiment_id"] == "EXP-00"
    assert res["representation"] == "timegap_static"
    assert res["num_patients"] == 16
    assert res["num_stays"] == 16
    assert res["alive_count"] == 8
    assert res["expired_count"] == 8
    assert res["split_strategy"] == "patient_grouped_test"
    assert res["seed"] == 42
    assert 0.0 <= res["auroc"] <= 1.0
    assert len(res["auroc_ci"]) == 2
    assert 0.0 <= res["average_precision"] <= 1.0
    assert len(res["average_precision_ci"]) == 2
    assert 0.0 <= res["f1"] <= 1.0
    assert 0.0 <= res["balanced_accuracy"] <= 1.0
    assert res["parameter_count"] > 0
    assert res["runtime"] >= 0.0
    results_json = run_dir / "results.json"
    assert results_json.exists()

def test_train_and_evaluate_logistic_baseline_single_class(tmp_path):
    for split in ["train", "validation", "test"]:
        sub_tmp = tmp_path / f"class_{split}"
        processed_dir = create_mock_processed_dir(sub_tmp, single_class_split=split)
        config = {
            "experiment": {"id": "EXP-00", "name": "logistic_test"},
            "representation": "timegap_static",
            "evaluation": {"split": "patient_grouped_test"},
            "runtime": {"seed": 42}
        }
        run_dir = sub_tmp / "run"
        with pytest.raises(ValueError):
            train_and_evaluate_logistic_baseline(config, processed_dir, run_dir)

def test_train_and_evaluate_logistic_baseline_empty_dataset(tmp_path):
    processed_dir = create_mock_processed_dir(tmp_path)
    (processed_dir / "encoded" / "train" / "shard_0.json").unlink()
    (processed_dir / "encoded" / "train" / "index.json").unlink()
    config = {
        "experiment": {"id": "EXP-00", "name": "logistic_test"},
        "representation": "timegap_static",
        "evaluation": {"split": "patient_grouped_test"},
        "runtime": {"seed": 42}
    }
    run_dir = tmp_path / "run"
    with pytest.raises(ValueError):
        train_and_evaluate_logistic_baseline(config, processed_dir, run_dir)
