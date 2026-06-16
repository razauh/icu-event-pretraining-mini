from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from icu_pretrain.data.dataset import EncodedDataset
from icu_pretrain.data.eicu_event_builder import SplitRecord, write_split_metadata
from icu_pretrain.models.heads import MortalityPredictionHead
from icu_pretrain.models.transformer import ICUTinyTransformer
from icu_pretrain.training.evaluate import compute_binary_metrics
from icu_pretrain.training.federated import (
    average_state_dicts,
    assign_hospital_clients,
    build_hospital_grouped_folds,
    run_fedavg_simulation,
    run_hospital_grouped_evaluation,
)
from icu_pretrain.utils import load_yaml


ROOT = Path(__file__).resolve().parents[1]


def make_stay(stay_id: str, patient_id: str, hospital_id: str, split_name: str, label: int) -> dict[str, object]:
    return {
        "patientunitstayid": stay_id,
        "uniquepid": patient_id,
        "hospitalid": hospital_id,
        "split_name": split_name,
        "label": label,
        "tokens": [3, 4, 5],
    }


def make_state_pair(vocab_size: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    model = ICUTinyTransformer(
        vocab_size=vocab_size,
        max_seq_len=256,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    )
    head = MortalityPredictionHead(d_model=64)
    return model.state_dict(), head.state_dict()


def write_processed_dir(root: Path, split_records: dict[str, list[dict[str, object]]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    vocab = {f"token_{idx}": idx for idx in range(8)}
    (root / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    for stage in (
        "build_cohort_and_splits",
        "encode_split_shards",
        "fit_training_preprocessing",
        "fit_vocabulary",
    ):
        stage_dir = root / "manifests" / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "manifest.json").write_text(
            json.dumps({"config_hash": "synthetic-hash"}, indent=2),
            encoding="utf-8",
        )
    all_records: list[SplitRecord] = []
    for split_name, stays in split_records.items():
        split_dir = root / "encoded" / split_name
        EncodedDataset.write_shard(stays, 0, split_dir)
        for stay in stays:
            all_records.append(
                SplitRecord(
                    patientunitstayid=str(stay["patientunitstayid"]),
                    uniquepid=str(stay["uniquepid"]),
                    hospitalid=str(stay["hospitalid"]),
                    split_name=str(stay["split_name"]),
                )
            )
    write_split_metadata(root / "split_metadata.json", all_records, processed_root=root)


def load_state_pair_from_dir(processed_dir: Path) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    vocab = json.loads((processed_dir / "vocab.json").read_text(encoding="utf-8"))
    return make_state_pair(len(vocab))


def fail_if_called(*args: object, **kwargs: object) -> None:
    raise AssertionError("unexpected call")


def test_build_hospital_grouped_folds_keeps_each_hospital_in_one_test_fold() -> None:
    records = [
        {"patientunitstayid": f"stay-{idx}", "label": idx % 2, "hospitalid": f"hospital-{idx // 2}"}
        for idx in range(10)
    ]

    folds = build_hospital_grouped_folds(records, n_splits=5, seed=42)

    assert len(folds) == 5
    seen_hospitals: set[str] = set()
    seen_test_hospitals: set[str] = set()
    for fold in folds:
        train_hospitals = set(fold["train_hospitals"])
        test_hospitals = set(fold["test_hospitals"])
        assert train_hospitals.isdisjoint(test_hospitals)
        seen_hospitals.update(train_hospitals)
        seen_test_hospitals.update(test_hospitals)

    assert seen_test_hospitals == {f"hospital-{idx}" for idx in range(5)}
    assert seen_hospitals == {f"hospital-{idx}" for idx in range(5)}


def test_assign_hospital_clients_uses_greedy_tie_breaking() -> None:
    records = [
        {"hospitalid": "hospital-a"},
        {"hospitalid": "hospital-a"},
        {"hospitalid": "hospital-a"},
        {"hospitalid": "hospital-b"},
        {"hospitalid": "hospital-b"},
        {"hospitalid": "hospital-c"},
        {"hospitalid": "hospital-c"},
        {"hospitalid": "hospital-d"},
    ]

    assignment = assign_hospital_clients(records, n_clients=3)

    assert assignment["hospital_counts"] == {
        "hospital-a": 3,
        "hospital-b": 2,
        "hospital-c": 2,
        "hospital-d": 1,
    }
    assert assignment["client_assignments"][0]["hospitals"] == ["hospital-a"]
    assert assignment["client_assignments"][1]["hospitals"] == ["hospital-b", "hospital-d"]
    assert assignment["client_assignments"][2]["hospitals"] == ["hospital-c"]


def test_average_state_dicts_uses_weighted_float_averaging_and_rejects_incompatible_states() -> None:
    state_one = {
        "float_weight": torch.tensor([1.0, 3.0]),
        "int_count": torch.tensor([2], dtype=torch.int64),
    }
    state_two = {
        "float_weight": torch.tensor([3.0, 5.0]),
        "int_count": torch.tensor([2], dtype=torch.int64),
    }

    averaged = average_state_dicts([state_one, state_two], [1.0, 3.0])

    assert torch.allclose(averaged["float_weight"], torch.tensor([2.5, 4.5]))
    assert torch.equal(averaged["int_count"], torch.tensor([2], dtype=torch.int64))

    with pytest.raises(ValueError, match="incompatible tensor shape"):
        average_state_dicts([state_one, {"float_weight": torch.tensor([1.0])}], [1.0, 1.0])


def test_run_hospital_grouped_evaluation_refits_training_hospitals_from_all_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed_dir = tmp_path / "processed"
    run_dir = tmp_path / "run"
    train_stays = [
        make_stay("stay-a1", "patient-a", "hospital-a", "train", 0),
        make_stay("stay-a2", "patient-a2", "hospital-a", "train", 1),
        make_stay("stay-b1", "patient-b", "hospital-b", "train", 0),
        make_stay("stay-b2", "patient-b2", "hospital-b", "train", 1),
        make_stay("stay-c1", "patient-c", "hospital-c", "train", 0),
        make_stay("stay-c2", "patient-c2", "hospital-c", "train", 1),
        make_stay("stay-d1", "patient-d", "hospital-d", "train", 0),
        make_stay("stay-d2", "patient-d2", "hospital-d", "train", 1),
        make_stay("stay-e1", "patient-e", "hospital-e", "train", 0),
        make_stay("stay-e2", "patient-e2", "hospital-e", "train", 1),
    ]
    validation_stays = [
        make_stay("stay-f1-val", "patient-f", "hospital-f", "validation", 1),
    ]
    test_stays = [
        make_stay("stay-f1-test", "patient-f", "hospital-f", "test", 1),
    ]
    write_processed_dir(
        processed_dir,
        {
            "train": train_stays,
            "validation": validation_stays,
            "test": test_stays,
        },
    )

    config = load_yaml(ROOT / "configs/final/eicu_demo_final_tiny.yaml")
    config["data"]["processed_dir"] = str(processed_dir)
    config["pretraining"]["enabled"] = False
    config["experiment"]["name"] = "synthetic_hospital_grouped"

    model_state, head_state = load_state_pair_from_dir(processed_dir)

    def fake_train_finetuning_model(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "model_state": model_state,
            "prediction_head_state": head_state,
            "val_loss": 0.0,
        }

    def fake_evaluate_model(
        _model: object,
        _head: object,
        dataset: EncodedDataset,
        _vocab: dict[str, int],
        *,
        batch_size: int,
        seed: int,
    ) -> dict[str, object]:
        targets = [int(stay.label) for stay in dataset.stays()]
        probs = [0.9 if target else 0.1 for target in targets]
        return {
            "patient_ids": [str(stay.patientunitstayid) for stay in dataset.stays()],
            "targets": targets,
            "probs": probs,
            "metrics": compute_binary_metrics(targets, probs, threshold=0.5),
        }

    monkeypatch.setattr("icu_pretrain.training.federated.train_model", fail_if_called)
    monkeypatch.setattr("icu_pretrain.training.federated.train_finetuning_model", fake_train_finetuning_model)
    monkeypatch.setattr("icu_pretrain.training.federated._evaluate_model", fake_evaluate_model)

    summary = run_hospital_grouped_evaluation(config, processed_dir, run_dir, resume="no")

    assert summary["fold_count"] == 5
    folds = build_hospital_grouped_folds(train_stays + validation_stays + test_stays, n_splits=5, seed=42)
    found_training_fold = False
    for fold in folds:
        if "hospital-e" in fold["train_hospitals"]:
            found_training_fold = True
            fold_processed_dir = run_dir / "folds" / f"fold_{int(fold['fold_number']):02d}" / "processed"
            split_metadata = json.loads((fold_processed_dir / "split_metadata.json").read_text(encoding="utf-8"))
            assert any(record["hospitalid"] == "hospital-e" for record in split_metadata)
    assert found_training_fold


def test_run_fedavg_simulation_resumes_completed_client_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed_dir = tmp_path / "processed"
    run_dir = tmp_path / "run"
    write_processed_dir(
        processed_dir,
        {
            "train": [
                make_stay("stay-a1", "patient-a", "hospital-a", "train", 0),
                make_stay("stay-a2", "patient-a2", "hospital-a", "train", 1),
                make_stay("stay-b1", "patient-b", "hospital-b", "train", 0),
                make_stay("stay-c1", "patient-c", "hospital-c", "train", 1),
            ],
            "validation": [
                make_stay("stay-d1", "patient-d", "hospital-d", "validation", 0),
            ],
            "test": [
                make_stay("stay-e1", "patient-e", "hospital-e", "test", 1),
            ],
        },
    )

    config = load_yaml(ROOT / "configs/fedavg/eicu_demo_fedavg_sim.yaml")
    config["data"]["processed_dir"] = str(processed_dir)
    config["experiment"]["name"] = "synthetic_fedavg"

    model_state, head_state = load_state_pair_from_dir(processed_dir)

    def fake_train_model(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "model_state": model_state,
            "prediction_head_state": head_state,
            "val_loss": 0.0,
        }

    def fake_train_finetuning_model(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "model_state": model_state,
            "prediction_head_state": head_state,
            "val_loss": 0.0,
        }

    def fake_evaluate_global_model(
        *_args: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        return {
            "val_probs": [0.1, 0.9],
            "val_targets": [0, 1],
            "test_probs": [0.2, 0.8],
            "test_targets": [0, 1],
            "test_patients": ["patient-a", "patient-b"],
            "threshold": 0.5,
            "metrics": {
                "auroc": 1.0,
                "ap": 1.0,
                "f1": 1.0,
                "balanced_accuracy": 1.0,
            },
            "bootstrap": {
                "auroc_ci": [1.0, 1.0],
                "ap_ci": [1.0, 1.0],
            },
        }

    monkeypatch.setattr("icu_pretrain.training.federated.train_model", fake_train_model)
    monkeypatch.setattr("icu_pretrain.training.federated.train_finetuning_model", fake_train_finetuning_model)
    monkeypatch.setattr("icu_pretrain.training.federated._evaluate_global_model", fake_evaluate_global_model)

    completed_client_dir = run_dir / "clients" / "client_00" / "round_01"
    completed_client_dir.mkdir(parents=True, exist_ok=True)
    (completed_client_dir / "state.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (completed_client_dir / "results.json").write_text(
        json.dumps(
            {
                "client_idx": 0,
                "round_number": 1,
                "status": "completed",
                "stay_count": 2,
                "val_loss": 0.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.save(
        SimpleNamespace(model_state=model_state, prediction_head_state=head_state),
        completed_client_dir / "checkpoint.pt",
    )

    summary = run_fedavg_simulation(config, processed_dir, run_dir, resume="auto")

    assert summary["client_count"] == 3
    assert summary["round_count"] == 3
    assert summary["rounds"][0]["client_summaries"][0]["status"] == "completed"
