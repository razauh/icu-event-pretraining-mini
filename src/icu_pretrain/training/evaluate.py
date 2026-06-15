import numpy as np
from collections import defaultdict
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    balanced_accuracy_score
)

def compute_binary_metrics(y_true, y_prob, threshold=None) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")
    if len(y_true) == 0:
        raise ValueError("Arrays cannot be empty.")
    if np.isnan(y_true).any() or np.isnan(y_prob).any():
        raise ValueError("Input contains NaNs.")
    if not np.all((y_true == 0) | (y_true == 1)):
        raise ValueError("y_true must contain only binary values (0 or 1).")
    if not np.all((y_prob >= 0.0) & (y_prob <= 1.0)):
        raise ValueError("y_prob must contain probabilities in the range [0.0, 1.0].")
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return {
            "auroc": float("nan"),
            "ap": float("nan"),
            "f1": float("nan") if threshold is not None else None,
            "balanced_accuracy": float("nan") if threshold is not None else None,
        }
    auroc = float(roc_auc_score(y_true, y_prob))
    ap = float(average_precision_score(y_true, y_prob))
    f1 = None
    bal_acc = None
    if threshold is not None:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = float(f1_score(y_true, y_pred, zero_division=0.0))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    return {
        "auroc": auroc,
        "ap": ap,
        "f1": f1,
        "balanced_accuracy": bal_acc,
    }

def find_best_f1_threshold(y_true, y_prob) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")
    if len(y_true) == 0:
        raise ValueError("Arrays cannot be empty.")
    if np.isnan(y_true).any() or np.isnan(y_prob).any():
        raise ValueError("Input contains NaNs.")
    if not np.all((y_true == 0) | (y_true == 1)):
        raise ValueError("y_true must contain only binary values (0 or 1).")
    if not np.all((y_prob >= 0.0) & (y_prob <= 1.0)):
        raise ValueError("y_prob must contain probabilities in the range [0.0, 1.0].")
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return 0.5
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = []
    for p, r in zip(precisions[:-1], recalls[:-1]):
        if p + r == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * p * r / (p + r))
    if not f1s:
        return 0.5
    best_idx = np.argmax(f1s)
    return float(thresholds[best_idx])

def bootstrap_patient_metrics(patients, y_true, y_prob, threshold, n_replicates=1000, seed=42) -> dict:
    patients = np.asarray(patients)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(patients) != len(y_true) or len(y_true) != len(y_prob):
        raise ValueError("patients, y_true, and y_prob must have the same length.")
    if len(y_true) == 0:
        raise ValueError("Arrays cannot be empty.")
    if np.isnan(y_true).any() or np.isnan(y_prob).any():
        raise ValueError("Input contains NaNs.")
    if not np.all((y_true == 0) | (y_true == 1)):
        raise ValueError("y_true must contain only binary values (0 or 1).")
    if not np.all((y_prob >= 0.0) & (y_prob <= 1.0)):
        raise ValueError("y_prob must contain probabilities in the range [0.0, 1.0].")
    unique_patients = np.unique(patients)
    n_patients = len(unique_patients)
    rng = np.random.default_rng(seed)
    patient_to_indices = defaultdict(list)
    for idx, p in enumerate(patients):
        patient_to_indices[p].append(idx)
    boot_aurocs = []
    boot_aps = []
    boot_f1s = []
    boot_bal_accs = []
    skipped_replicates = 0
    for _ in range(n_replicates):
        sampled_patients = rng.choice(unique_patients, size=n_patients, replace=True)
        sampled_indices = []
        for p in sampled_patients:
            sampled_indices.extend(patient_to_indices[p])
        rep_y_true = y_true[sampled_indices]
        rep_y_prob = y_prob[sampled_indices]
        if len(np.unique(rep_y_true)) < 2:
            skipped_replicates += 1
            continue
        metrics = compute_binary_metrics(rep_y_true, rep_y_prob, threshold=threshold)
        boot_aurocs.append(metrics["auroc"])
        boot_aps.append(metrics["ap"])
        boot_f1s.append(metrics["f1"])
        boot_bal_accs.append(metrics["balanced_accuracy"])
    if boot_aurocs:
        auroc_ci = (float(np.percentile(boot_aurocs, 2.5)), float(np.percentile(boot_aurocs, 97.5)))
        ap_ci = (float(np.percentile(boot_aps, 2.5)), float(np.percentile(boot_aps, 97.5)))
        f1_ci = (float(np.percentile(boot_f1s, 2.5)), float(np.percentile(boot_f1s, 97.5)))
        bal_acc_ci = (float(np.percentile(boot_bal_accs, 2.5)), float(np.percentile(boot_bal_accs, 97.5)))
    else:
        auroc_ci = (float("nan"), float("nan"))
        ap_ci = (float("nan"), float("nan"))
        f1_ci = (float("nan"), float("nan"))
        bal_acc_ci = (float("nan"), float("nan"))
    return {
        "auroc_ci": auroc_ci,
        "ap_ci": ap_ci,
        "f1_ci": f1_ci,
        "balanced_accuracy_ci": bal_acc_ci,
        "skipped_replicates": skipped_replicates,
        "replicate_aurocs": boot_aurocs,
        "replicate_aps": boot_aps,
        "replicate_f1s": boot_f1s,
        "replicate_balanced_accuracies": boot_bal_accs,
    }
