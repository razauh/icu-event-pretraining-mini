import time
import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from icu_pretrain.data.dataset import EncodedDataset
from icu_pretrain.training.evaluate import (
    compute_binary_metrics,
    find_best_f1_threshold,
    bootstrap_patient_metrics
)
from icu_pretrain.utils import load_yaml, validate_experiment_config

def train_and_evaluate_logistic_baseline(config: dict, processed_dir: Path, run_dir: Path) -> dict:
    processed_dir = Path(processed_dir)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = processed_dir / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"vocab.json not found in {processed_dir}")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    train_dir = processed_dir / "encoded" / "train"
    val_dir = processed_dir / "encoded" / "validation"
    test_dir = processed_dir / "encoded" / "test"
    train_dataset = EncodedDataset(train_dir)
    val_dataset = EncodedDataset(val_dir)
    test_dataset = EncodedDataset(test_dir)
    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty.")
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty.")
    train_stays = train_dataset.stays()
    val_stays = val_dataset.stays()
    test_stays = test_dataset.stays()
    X_train = np.zeros((len(train_stays), vocab_size))
    y_train = np.zeros(len(train_stays))
    for idx, stay in enumerate(train_stays):
        for token in stay.tokens:
            if 0 <= token < vocab_size:
                X_train[idx, token] += 1
        y_train[idx] = stay.label
    X_val = np.zeros((len(val_stays), vocab_size))
    y_val = np.zeros(len(val_stays))
    for idx, stay in enumerate(val_stays):
        for token in stay.tokens:
            if 0 <= token < vocab_size:
                X_val[idx, token] += 1
        y_val[idx] = stay.label
    X_test = np.zeros((len(test_stays), vocab_size))
    y_test = np.zeros(len(test_stays))
    for idx, stay in enumerate(test_stays):
        for token in stay.tokens:
            if 0 <= token < vocab_size:
                X_test[idx, token] += 1
        y_test[idx] = stay.label
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training set has only one unique class.")
    if len(np.unique(y_val)) < 2:
        raise ValueError("Validation set has only one unique class.")
    if len(np.unique(y_test)) < 2:
        raise ValueError("Test set has only one unique class.")
    seed = config.get("runtime", {}).get("seed", 42)
    start_time = time.time()
    model = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000)
    model.fit(X_train, y_train)
    runtime = time.time() - start_time
    param_count = int(np.prod(model.coef_.shape) + np.prod(model.intercept_.shape))
    y_val_prob = model.predict_proba(X_val)[:, 1]
    best_threshold = find_best_f1_threshold(y_val, y_val_prob)
    y_test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_binary_metrics(y_test, y_test_prob, threshold=best_threshold)
    split_metadata_path = processed_dir / "split_metadata.json"
    if not split_metadata_path.exists():
        raise FileNotFoundError(f"split_metadata.json not found in {processed_dir}")
    with open(split_metadata_path, "r", encoding="utf-8") as f:
        split_metadata = json.load(f)
    stay_to_patient = {str(r["patientunitstayid"]): str(r["uniquepid"]) for r in split_metadata}
    test_patients = [stay_to_patient[str(stay.patientunitstayid)] for stay in test_stays]
    bootstrap_results = bootstrap_patient_metrics(
        patients=test_patients,
        y_true=y_test,
        y_prob=y_test_prob,
        threshold=best_threshold,
        n_replicates=1000,
        seed=seed
    )
    num_patients = len(np.unique(test_patients))
    num_stays = len(test_stays)
    alive_count = int(np.sum(y_test == 0))
    expired_count = int(np.sum(y_test == 1))
    results = {
        "experiment_id": config.get("experiment", {}).get("id", "EXP-00"),
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
    return results

def run_baselines() -> None:
    parser = argparse.ArgumentParser(description="Run bag-of-events baselines.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--processed_dir", type=Path, default=None)
    parser.add_argument("--run_dir", type=Path, default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    validate_experiment_config(config)
    processed_dir = args.processed_dir
    if processed_dir is None:
        processed_dir = Path("data/processed")
    run_dir = args.run_dir
    if run_dir is None:
        run_name = config.get("experiment", {}).get("name", "baseline_run")
        run_dir = Path("results/runs") / run_name
    train_and_evaluate_logistic_baseline(
        config=config,
        processed_dir=processed_dir,
        run_dir=run_dir,
    )
