"""Validation tests for repository YAML configuration contracts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from icu_pretrain.utils import (
    load_yaml,
    validate_experiment_config,
    validate_fedavg_config,
    validate_final_config,
    validate_tuning_config,
)


ROOT = Path(__file__).resolve().parents[1]
METRICS = ["auroc", "average_precision", "f1", "balanced_accuracy"]


def load_config(relative_path: str) -> dict[str, object]:
    return load_yaml(ROOT / relative_path)


def test_final_config_is_valid_and_cpu_only() -> None:
    config = load_config("configs/final/eicu_demo_final_tiny.yaml")

    validated = validate_final_config(config)

    assert validated["runtime"]["device"] == "cpu"
    assert validated["data"]["representation"] == "timegap_static"
    assert validated["model"]["type"] == "icu_tiny_transformer"
    assert validated["evaluation"] == {"split": "random", "metrics": METRICS}


def test_tuning_config_is_valid_and_defaults_to_cpu() -> None:
    config = load_config("configs/tuning/optuna_eicu_tiny.yaml")

    validated = validate_tuning_config(config)

    assert validated["runtime"]["device"] == "cpu"
    assert validated["study"]["metric"] == "val_average_precision"
    assert validated["search_space"]["d_model"] == [32, 64, 96, 128]


def test_fedavg_config_is_valid_and_cpu_only() -> None:
    config = load_config("configs/fedavg/eicu_demo_fedavg_sim.yaml")

    validated = validate_fedavg_config(config)

    assert validated["runtime"]["device"] == "cpu"
    assert validated["data"]["representation"] == "timegap_static"
    assert validated["fedavg"]["split_strategy"] == "event_density"
    assert validated["model"]["type"] == "icu_tiny_transformer"


def test_experiment_suite_and_all_short_configs_are_valid() -> None:
    suite = validate_experiment_config(
        load_config("configs/experiments/exp_suite_core.yaml")
    )

    assert suite["runtime"]["device"] == "cpu"
    assert len(suite["suite"]["experiments"]) == 8

    for config_path in suite["suite"]["experiments"]:
        validated = validate_experiment_config(load_config(config_path))
        assert validated["runtime"]["device"] == "cpu"
        assert validated["representation"] in {"basic", "timegap", "timegap_static"}
        assert validated["model"] in {"logistic_regression", "icu_tiny_transformer"}
        assert validated["evaluation"]["split"] == "random"
        assert validated["evaluation"]["metrics"] == METRICS


@pytest.mark.parametrize(
    ("validator", "config_path"),
    [
        (validate_final_config, "configs/final/eicu_demo_final_tiny.yaml"),
        (validate_tuning_config, "configs/tuning/optuna_eicu_tiny.yaml"),
        (validate_fedavg_config, "configs/fedavg/eicu_demo_fedavg_sim.yaml"),
        (validate_experiment_config, "configs/experiments/exp_00_logistic_basic.yaml"),
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


def test_unsupported_representation_is_rejected() -> None:
    config = load_config("configs/final/eicu_demo_final_tiny.yaml")
    config["data"]["representation"] = "raw_values"

    with pytest.raises(ValueError, match="data.representation"):
        validate_final_config(config)


def test_unsupported_metric_is_rejected() -> None:
    config = load_config("configs/final/eicu_demo_final_tiny.yaml")
    config["evaluation"]["metrics"] = ["accuracy"]

    with pytest.raises(ValueError, match="evaluation.metrics"):
        validate_final_config(config)


def test_invalid_device_is_rejected() -> None:
    config = load_config("configs/final/eicu_demo_final_tiny.yaml")
    config["runtime"]["device"] = "cuda"

    with pytest.raises(ValueError, match="runtime.device"):
        validate_final_config(config)


def test_missing_runtime_device_is_rejected_for_full_config() -> None:
    config = load_config("configs/final/eicu_demo_final_tiny.yaml")
    del config["runtime"]["device"]

    with pytest.raises(ValueError, match="runtime.device"):
        validate_final_config(config)


def test_incompatible_model_dimensions_are_rejected() -> None:
    config = load_config("configs/final/eicu_demo_final_tiny.yaml")
    config["model"]["d_model"] = 62

    with pytest.raises(ValueError, match="model.d_model"):
        validate_final_config(config)


def test_unknown_experiment_id_is_rejected() -> None:
    config = load_config("configs/experiments/exp_00_logistic_basic.yaml")
    config["experiment"]["id"] = "EXP-99"

    with pytest.raises(ValueError, match="experiment.id"):
        validate_experiment_config(config)


@pytest.mark.parametrize(
    "learning_rate",
    [
        [],
        {"type": "uniform", "low": 0.0001, "high": 0.001},
        {"type": "loguniform", "low": 0.001, "high": 0.0001},
    ],
)
def test_malformed_optuna_search_space_is_rejected(learning_rate: object) -> None:
    config = load_config("configs/tuning/optuna_eicu_tiny.yaml")
    config["search_space"]["learning_rate"] = learning_rate

    with pytest.raises(ValueError, match="search_space.learning_rate"):
        validate_tuning_config(config)
