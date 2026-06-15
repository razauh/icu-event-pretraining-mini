import pytest
import numpy as np
from icu_pretrain.training.evaluate import (
    compute_binary_metrics,
    find_best_f1_threshold,
    bootstrap_patient_metrics
)

def test_compute_binary_metrics_basic():
    y_true = [0, 1, 0, 1]
    y_prob = [0.1, 0.9, 0.2, 0.8]
    metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["auroc"] == 1.0
    assert metrics["ap"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0

def test_compute_binary_metrics_mismatched_and_empty():
    with pytest.raises(ValueError):
        compute_binary_metrics([0, 1], [0.5])
    with pytest.raises(ValueError):
        compute_binary_metrics([], [])

def test_compute_binary_metrics_nans():
    with pytest.raises(ValueError):
        compute_binary_metrics([0, np.nan], [0.1, 0.2])
    with pytest.raises(ValueError):
        compute_binary_metrics([0, 1], [0.1, np.nan])

def test_compute_binary_metrics_invalid_ranges():
    with pytest.raises(ValueError):
        compute_binary_metrics([0, 2], [0.1, 0.2])
    with pytest.raises(ValueError):
        compute_binary_metrics([0, 1], [-0.1, 0.2])
    with pytest.raises(ValueError):
        compute_binary_metrics([0, 1], [0.1, 1.2])

def test_compute_binary_metrics_single_class():
    metrics = compute_binary_metrics([1, 1, 1], [0.1, 0.2, 0.3], threshold=0.5)
    assert np.isnan(metrics["auroc"])
    assert np.isnan(metrics["ap"])
    assert np.isnan(metrics["f1"])
    assert np.isnan(metrics["balanced_accuracy"])

def test_find_best_f1_threshold():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]
    thresh = find_best_f1_threshold(y_true, y_prob)
    assert 0.2 < thresh <= 0.8
    single_class_thresh = find_best_f1_threshold([0, 0], [0.1, 0.2])
    assert single_class_thresh == 0.5

def test_bootstrap_patient_metrics_basic():
    patients = ["P1", "P1", "P2", "P3", "P4", "P4"]
    y_true = [0, 0, 1, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.3, 0.9, 0.95]
    res = bootstrap_patient_metrics(patients, y_true, y_prob, threshold=0.5, n_replicates=50, seed=42)
    assert "auroc_ci" in res
    assert "ap_ci" in res
    assert "f1_ci" in res
    assert "balanced_accuracy_ci" in res
    assert res["skipped_replicates"] >= 0
    assert len(res["replicate_aurocs"]) + res["skipped_replicates"] == 50

def test_bootstrap_patient_metrics_one_class_replicates():
    patients = ["P1", "P2"]
    y_true = [1, 1]
    y_prob = [0.9, 0.9]
    res = bootstrap_patient_metrics(patients, y_true, y_prob, threshold=0.5, n_replicates=10, seed=42)
    assert res["skipped_replicates"] == 10
    assert np.isnan(res["auroc_ci"][0])
    assert np.isnan(res["auroc_ci"][1])
