"""Validation tests for repository YAML configuration contracts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from icu_pretrain.experiments.registry import EXPERIMENT_IDS
from icu_pretrain.utils import (
    load_yaml,
    validate_experiment_config,
    validate_fedavg_config,
    validate_final_config,
    validate_tuning_config,
)


ROOT = Path(__file__).resolve().parents[1]
METRICS = ["auroc", "average_precision", "f1", "balanced_accuracy"]
EXPERIMENT_CONFIGS = (
    "configs/experiments/exp_00_logistic_full.yaml",
    "configs/experiments/exp_01_scratch_full.yaml",
    "configs/experiments/exp_02_pretrained_full.yaml",
    "configs/experiments/exp_03_pretrained_ablation.yaml",
    "configs/experiments/exp_04_hospital_grouped.yaml",
    "configs/experiments/exp_05_fedavg.yaml",
)


def load_config(relative_path: str) -> dict[str, object]:
    return load_yaml(ROOT / relative_path)


def test_final_config_matches_dataset_grounded_protocol() -> None:
    validated = validate_final_config(
        load_config("configs/final/eicu_demo_final_tiny.yaml")
    )

    assert validated["runtime"] == {"device": "cpu", "num_workers": 0, "seed": 42}
    assert validated["data"]["representation"] == "timegap_static"
    assert validated["data"]["observation_window"] == {
        "start_minutes": 0,
        "end_minutes": 1440,
    }
    assert validated["data"]["max_seq_len"] == 256
    assert validated["data"]["min_dynamic_events"] == 5
    assert validated["split"]["strategy"] == "patient_grouped"
    assert validated["split"]["ratios"] == {
        "train": 0.70,
        "validation": 0.15,
        "test": 0.15,
    }
    assert validated["preprocessing"]["vital_bucket_minutes"] == 60
    assert validated["evaluation"] == {
        "split": "patient_grouped_test",
        "selection_metric": "validation_average_precision",
        "metrics": METRICS,
    }
    assert validated["data_processing"] == {
        "csv_chunk_rows": 50000,
        "partition_shards": 64,
        "encoded_shard_stays": 128,
    }
    assert validated["recovery"] == {
        "enabled": True,
        "log_every_batches": 10,
        "checkpoint_every_optimizer_steps": 100,
        "keep_last_checkpoints": 2,
        "resume": "auto",
    }


def test_tuning_config_is_limited_to_eight_trials() -> None:
    validated = validate_tuning_config(
        load_config("configs/tuning/optuna_eicu_tiny.yaml")
    )

    assert validated["study"]["n_trials"] == 8
    assert validated["study"]["metric"] == "val_average_precision"
    assert validated["search_space"]["d_model"] == [32, 64, 96]
    assert validated["search_space"]["max_seq_len"] == [256]
    assert validated["trial_training"] == {
        "pretrain_epochs": 2,
        "finetune_epochs": 5,
    }


def test_fedavg_config_uses_three_hospital_cluster_clients_and_rounds() -> None:
    validated = validate_fedavg_config(
        load_config("configs/fedavg/eicu_demo_fedavg_sim.yaml")
    )

    assert validated["runtime"]["device"] == "cpu"
    assert validated["split"]["strategy"] == "patient_grouped"
    assert validated["fedavg"] == {
        "client_grouping": "hospital_cluster",
        "clients": 3,
        "rounds": 3,
        "local_epochs": 1,
        "client_fraction": 1.0,
        "aggregation": "weighted_average_by_num_training_stays",
    }


def test_experiment_registry_and_configs_match_reduced_matrix() -> None:
    assert EXPERIMENT_IDS == (
        "EXP-00",
        "EXP-01",
        "EXP-02",
        "EXP-03",
        "EXP-04",
        "EXP-05",
    )

    validated_configs = [
        validate_experiment_config(load_config(path)) for path in EXPERIMENT_CONFIGS
    ]

    assert [config["experiment"]["id"] for config in validated_configs] == list(
        EXPERIMENT_IDS
    )
    assert validated_configs[0]["model"] == "logistic_regression"
    assert validated_configs[1]["pretraining"] is False
    assert validated_configs[2]["pretraining"] is True
    assert validated_configs[3]["representation"] == "basic"
    assert validated_configs[4]["evaluation"]["split"] == "hospital_grouped_cv"
    assert validated_configs[5]["optional"] is True


def test_core_suite_excludes_optional_fedavg_experiment() -> None:
    validated = validate_experiment_config(
        load_config("configs/experiments/exp_suite_core.yaml")
    )

    assert len(validated["suite"]["experiments"]) == 5
    assert all("exp_05" not in path for path in validated["suite"]["experiments"])
    assert validated["runtime"] == {"device": "cpu", "num_workers": 0, "seed": 42}
    assert validated["data_processing"]["csv_chunk_rows"] == 50000
    assert validated["recovery"]["resume"] == "auto"


@pytest.mark.parametrize(
    ("validator", "config_path"),
    [
        (validate_final_config, "configs/final/eicu_demo_final_tiny.yaml"),
        (validate_tuning_config, "configs/tuning/optuna_eicu_tiny.yaml"),
        (validate_fedavg_config, "configs/fedavg/eicu_demo_fedavg_sim.yaml"),
        (validate_experiment_config, "configs/experiments/exp_00_logistic_full.yaml"),
        (validate_experiment_config, "configs/experiments/exp_suite_core.yaml"),
    ],
)
def test_validation_does_not_mutate_input(validator: object, config_path: str) -> None:
    config = load_config(config_path)
    original = deepcopy(config)

    validator(config)

    assert config == original


def test_empty_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="config"):
        validate_final_config({})


def test_missing_required_section_is_rejected() -> None:
    config = load_config("configs/final/eicu_demo_final_tiny.yaml")
    del config["model"]

    with pytest.raises(ValueError, match="model"):
        validate_final_config(config)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("data", "max_seq_len"), 128, "data.max_seq_len"),
        (("data", "min_dynamic_events"), 4, "data.min_dynamic_events"),
        (("preprocessing", "vital_bucket_minutes"), 0, "vital_bucket_minutes"),
        (("runtime", "device"), "cuda", "runtime.device"),
        (("runtime", "seed"), 7, "runtime.seed"),
        (("recovery", "resume"), "never", "recovery.resume"),
    ],
)
def test_stale_or_unsupported_protocol_values_are_rejected(
    path: tuple[str, str], value: object, message: str
) -> None:
    config = load_config("configs/final/eicu_demo_final_tiny.yaml")
    config[path[0]][path[1]] = value

    with pytest.raises(ValueError, match=message):
        validate_final_config(config)


def test_bounded_memory_and_recovery_intervals_are_configurable() -> None:
    config = load_config("configs/final/eicu_demo_final_tiny.yaml")
    config["data_processing"] = {
        "csv_chunk_rows": 10000,
        "partition_shards": 32,
        "encoded_shard_stays": 64,
    }
    config["recovery"]["log_every_batches"] = 5
    config["recovery"]["checkpoint_every_optimizer_steps"] = 50
    config["recovery"]["keep_last_checkpoints"] = 3

    validated = validate_final_config(config)

    assert validated["data_processing"]["csv_chunk_rows"] == 10000
    assert validated["recovery"]["checkpoint_every_optimizer_steps"] == 50


def test_invalid_configurable_interval_is_rejected() -> None:
    config = load_config("configs/final/eicu_demo_final_tiny.yaml")
    config["data_processing"]["csv_chunk_rows"] = 0

    with pytest.raises(ValueError, match="data_processing.csv_chunk_rows"):
        validate_final_config(config)


@pytest.mark.parametrize(
    ("start", "end"),
    [(-60, 1440), (0, 720), (1440, 0)],
)
def test_invalid_observation_window_is_rejected(start: int, end: int) -> None:
    config = load_config("configs/final/eicu_demo_final_tiny.yaml")
    config["data"]["observation_window"] = {
        "start_minutes": start,
        "end_minutes": end,
    }

    with pytest.raises(ValueError, match="observation_window"):
        validate_final_config(config)


def test_stay_random_split_is_rejected() -> None:
    config = load_config("configs/final/eicu_demo_final_tiny.yaml")
    config["split"]["strategy"] = "stay_random"

    with pytest.raises(ValueError, match="split.strategy"):
        validate_final_config(config)


@pytest.mark.parametrize(
    "ratios",
    [
        {"train": 0.8, "validation": 0.1, "test": 0.1},
        {"train": 0.7, "validation": 0.15, "test": 0.2},
        {"train": 0.85, "validation": 0.15, "test": 0.0},
    ],
)
def test_invalid_split_ratios_are_rejected(ratios: dict[str, float]) -> None:
    config = load_config("configs/final/eicu_demo_final_tiny.yaml")
    config["split"]["ratios"] = ratios

    with pytest.raises(ValueError, match="split.ratios"):
        validate_final_config(config)


@pytest.mark.parametrize(
    "feature",
    [
        "apache_iv_score",
        "hospitaldischargestatus",
        "unit_discharge_offset",
        "activeupondischarge",
    ],
)
def test_prohibited_leakage_features_are_rejected(feature: str) -> None:
    config = load_config("configs/final/eicu_demo_final_tiny.yaml")
    config["data"]["feature_fields"] = ["age", feature]

    with pytest.raises(ValueError, match="prohibited feature"):
        validate_final_config(config)


def test_unsupported_metric_is_rejected() -> None:
    config = load_config("configs/final/eicu_demo_final_tiny.yaml")
    config["evaluation"]["metrics"] = ["accuracy"]

    with pytest.raises(ValueError, match="evaluation.metrics"):
        validate_final_config(config)


def test_unsupported_training_selection_metric_is_rejected() -> None:
    config = load_config("configs/final/eicu_demo_final_tiny.yaml")
    config["finetuning"]["selection_metric"] = "validation_auroc"

    with pytest.raises(ValueError, match="finetuning.selection_metric"):
        validate_final_config(config)


def test_incompatible_model_dimensions_are_rejected() -> None:
    config = load_config("configs/final/eicu_demo_final_tiny.yaml")
    config["model"]["d_model"] = 62

    with pytest.raises(ValueError, match="model.d_model"):
        validate_final_config(config)


def test_unknown_experiment_id_is_rejected() -> None:
    config = load_config("configs/experiments/exp_00_logistic_full.yaml")
    config["experiment"]["id"] = "EXP-06"

    with pytest.raises(ValueError, match="experiment.id"):
        validate_experiment_config(config)


def test_optional_exp_05_cannot_be_added_to_core_suite() -> None:
    config = load_config("configs/experiments/exp_suite_core.yaml")
    config["suite"]["experiments"].append(
        "configs/experiments/exp_05_fedavg.yaml"
    )

    with pytest.raises(ValueError, match="optional EXP-05"):
        validate_experiment_config(config)


def test_more_than_eight_tuning_trials_is_rejected() -> None:
    config = load_config("configs/tuning/optuna_eicu_tiny.yaml")
    config["study"]["n_trials"] = 9

    with pytest.raises(ValueError, match="study.n_trials"):
        validate_tuning_config(config)


def test_tuning_rejects_incompatible_model_dimensions() -> None:
    config = load_config("configs/tuning/optuna_eicu_tiny.yaml")
    config["search_space"]["d_model"] = [62]

    with pytest.raises(ValueError, match="search_space.d_model"):
        validate_tuning_config(config)


def test_pseudoclient_fedavg_grouping_is_rejected() -> None:
    config = load_config("configs/fedavg/eicu_demo_fedavg_sim.yaml")
    config["fedavg"]["client_grouping"] = "event_density"

    with pytest.raises(ValueError, match="fedavg.client_grouping"):
        validate_fedavg_config(config)
