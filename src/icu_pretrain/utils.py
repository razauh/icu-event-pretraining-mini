"""General utility helpers."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from icu_pretrain.experiments.registry import EXPERIMENT_IDS


SUPPORTED_REPRESENTATIONS = {"basic", "timegap", "timegap_static"}
SUPPORTED_METRICS = {"auroc", "average_precision", "f1", "balanced_accuracy"}
DEFAULT_METRICS = ["auroc", "average_precision", "f1", "balanced_accuracy"]
SUPPORTED_MODELS = {"logistic_regression", "icu_tiny_transformer"}
SUPPORTED_EVALUATION_SPLITS = {"random", "pseudoclient", "event_density", "diagnosis_group"}
SUPPORTED_PSEUDOCLIENT_SPLITS = {"event_density", "diagnosis_group"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return data


def _copy_config(config: dict[str, Any], config_name: str) -> dict[str, Any]:
    if not isinstance(config, dict) or not config:
        raise ValueError(f"{config_name} config must be a non-empty mapping")
    return deepcopy(config)


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _require_value(mapping: dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise ValueError(f"{path} is required")
    return mapping[key]


def _validate_choice(value: Any, allowed: set[str], path: str) -> None:
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{path} must be one of: {choices}")


def _validate_runtime(config: dict[str, Any], *, default: bool) -> None:
    if default:
        runtime = config.setdefault("runtime", {})
        if not isinstance(runtime, dict):
            raise ValueError("runtime must be a mapping")
        runtime.setdefault("device", "cpu")
        runtime.setdefault("num_workers", 0)
    else:
        runtime = _require_mapping(config, "runtime")

    device = _require_value(runtime, "device", "runtime.device")
    if device != "cpu":
        raise ValueError("runtime.device must be 'cpu'")


def _validate_representation(value: Any, path: str = "data.representation") -> None:
    _validate_choice(value, SUPPORTED_REPRESENTATIONS, path)


def _validate_model(model: dict[str, Any]) -> None:
    model_type = _require_value(model, "type", "model.type")
    _validate_choice(model_type, {"icu_tiny_transformer"}, "model.type")

    d_model = _require_value(model, "d_model", "model.d_model")
    n_heads = _require_value(model, "n_heads", "model.n_heads")
    if not isinstance(d_model, int) or d_model <= 0:
        raise ValueError("model.d_model must be a positive integer")
    if not isinstance(n_heads, int) or n_heads <= 0:
        raise ValueError("model.n_heads must be a positive integer")
    if d_model % n_heads != 0:
        raise ValueError("model.d_model must be divisible by model.n_heads")


def _validate_evaluation(evaluation: dict[str, Any], *, add_metrics: bool) -> None:
    split = _require_value(evaluation, "split", "evaluation.split")
    _validate_choice(split, SUPPORTED_EVALUATION_SPLITS, "evaluation.split")

    if add_metrics:
        evaluation.setdefault("metrics", list(DEFAULT_METRICS))
    metrics = _require_value(evaluation, "metrics", "evaluation.metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("evaluation.metrics must be a non-empty list")
    unsupported = [metric for metric in metrics if metric not in SUPPORTED_METRICS]
    if unsupported:
        raise ValueError(f"evaluation.metrics contains unsupported metric: {unsupported[0]}")


def validate_final_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and copy the full training configuration."""
    validated = _copy_config(config, "final")
    for section in (
        "experiment",
        "data",
        "model",
        "pretraining",
        "finetuning",
        "evaluation",
        "runtime",
    ):
        _require_mapping(validated, section)

    data = validated["data"]
    _validate_choice(_require_value(data, "dataset", "data.dataset"), {"eicu_demo"}, "data.dataset")
    _validate_representation(_require_value(data, "representation", "data.representation"))
    _validate_model(validated["model"])
    _validate_evaluation(validated["evaluation"], add_metrics=False)
    _validate_runtime(validated, default=False)
    return validated


def _validate_search_choices(search_space: dict[str, Any], key: str) -> list[Any]:
    value = _require_value(search_space, key, f"search_space.{key}")
    if not isinstance(value, list) or not value:
        raise ValueError(f"search_space.{key} must be a non-empty list")
    return value


def validate_tuning_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and copy an Optuna search configuration."""
    validated = _copy_config(config, "tuning")
    study = _require_mapping(validated, "study")
    search_space = _require_mapping(validated, "search_space")
    _require_mapping(validated, "trial_training")

    _validate_choice(
        _require_value(study, "direction", "study.direction"),
        {"maximize", "minimize"},
        "study.direction",
    )
    _validate_choice(
        _require_value(study, "metric", "study.metric"),
        {"val_auroc", "val_average_precision", "val_f1", "val_balanced_accuracy"},
        "study.metric",
    )

    d_models = _validate_search_choices(search_space, "d_model")
    n_heads = _validate_search_choices(search_space, "n_heads")
    for key in (
        "n_layers",
        "dim_feedforward",
        "dropout",
        "mask_probability",
        "max_seq_len",
    ):
        _validate_search_choices(search_space, key)

    if any(not isinstance(value, int) or value <= 0 for value in d_models):
        raise ValueError("search_space.d_model must contain positive integers")
    if any(not isinstance(value, int) or value <= 0 for value in n_heads):
        raise ValueError("search_space.n_heads must contain positive integers")
    if any(d_model % heads != 0 for d_model in d_models for heads in n_heads):
        raise ValueError("search_space.d_model values must be divisible by search_space.n_heads values")

    learning_rate = _require_value(
        search_space, "learning_rate", "search_space.learning_rate"
    )
    if not isinstance(learning_rate, dict):
        raise ValueError("search_space.learning_rate must be a mapping")
    if learning_rate.get("type") != "loguniform":
        raise ValueError("search_space.learning_rate.type must be 'loguniform'")
    low = learning_rate.get("low")
    high = learning_rate.get("high")
    if (
        not isinstance(low, (int, float))
        or isinstance(low, bool)
        or not isinstance(high, (int, float))
        or isinstance(high, bool)
        or low <= 0
        or low >= high
    ):
        raise ValueError("search_space.learning_rate requires 0 < low < high")

    _validate_runtime(validated, default=True)
    return validated


def validate_fedavg_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and copy a FedAvg-style simulation configuration."""
    validated = _copy_config(config, "fedavg")
    for section in ("experiment", "data", "model", "fedavg", "runtime"):
        _require_mapping(validated, section)

    data = validated["data"]
    _validate_choice(_require_value(data, "dataset", "data.dataset"), {"eicu_demo"}, "data.dataset")
    _validate_representation(_require_value(data, "representation", "data.representation"))
    data_split = _require_value(data, "split_strategy", "data.split_strategy")
    _validate_choice(data_split, SUPPORTED_PSEUDOCLIENT_SPLITS, "data.split_strategy")

    fedavg = validated["fedavg"]
    fedavg_split = _require_value(fedavg, "split_strategy", "fedavg.split_strategy")
    _validate_choice(fedavg_split, SUPPORTED_PSEUDOCLIENT_SPLITS, "fedavg.split_strategy")
    if data_split != fedavg_split:
        raise ValueError("data.split_strategy must match fedavg.split_strategy")

    _validate_model(validated["model"])
    _validate_runtime(validated, default=False)
    return validated


def validate_experiment_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and copy a short experiment or experiment-suite config."""
    validated = _copy_config(config, "experiment")
    if "suite" in validated:
        suite = _require_mapping(validated, "suite")
        _require_value(suite, "name", "suite.name")
        experiments = _require_value(suite, "experiments", "suite.experiments")
        if not isinstance(experiments, list) or not experiments:
            raise ValueError("suite.experiments must be a non-empty list")
        for path in experiments:
            if not isinstance(path, str):
                raise ValueError("suite.experiments entries must be paths")
            stem = Path(path).stem
            parts = stem.split("_", maxsplit=2)
            experiment_id = f"EXP-{parts[1]}" if len(parts) >= 2 else ""
            if experiment_id not in EXPERIMENT_IDS:
                raise ValueError(f"suite.experiments contains unknown experiment: {path}")
        _validate_runtime(validated, default=True)
        return validated

    experiment = _require_mapping(validated, "experiment")
    experiment_id = _require_value(experiment, "id", "experiment.id")
    if experiment_id not in EXPERIMENT_IDS:
        raise ValueError(f"experiment.id is unknown: {experiment_id}")
    _require_value(experiment, "name", "experiment.name")

    _validate_representation(
        _require_value(validated, "representation", "representation"),
        "representation",
    )
    _validate_choice(
        _require_value(validated, "model", "model"), SUPPORTED_MODELS, "model"
    )
    if not isinstance(_require_value(validated, "pretraining", "pretraining"), bool):
        raise ValueError("pretraining must be a boolean")

    evaluation = _require_mapping(validated, "evaluation")
    _validate_evaluation(evaluation, add_metrics=True)
    _validate_runtime(validated, default=True)
    return validated
