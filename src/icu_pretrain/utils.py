"""General utility helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml

from icu_pretrain.experiments.registry import EXPERIMENT_IDS


SUPPORTED_REPRESENTATIONS = {"basic", "timegap_static"}
SUPPORTED_METRICS = {"auroc", "average_precision", "f1", "balanced_accuracy"}
DEFAULT_METRICS = ["auroc", "average_precision", "f1", "balanced_accuracy"]
SUPPORTED_MODELS = {"logistic_regression", "icu_tiny_transformer", "selected_model"}
SUPPORTED_EVALUATION_SPLITS = {
    "patient_grouped_test",
    "hospital_grouped_cv",
    "hospital_cluster_simulation",
}
PROHIBITED_FEATURE_FRAGMENTS = (
    "apache",
    "actualhospitalmortality",
    "actualicumortality",
    "predictedhospitalmortality",
    "predictedicumortality",
    "diedinhospital",
    "hospitaldischargestatus",
    "unitdischargestatus",
    "discharge",
    "lengthofstay",
    "length_of_stay",
    "los",
    "activeupondischarge",
)
FEATURE_KEYS = {
    "features",
    "feature_fields",
    "input_fields",
    "static_fields",
    "dynamic_fields",
}


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


def _optional_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.setdefault(key, {})
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


def _validate_positive_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _validate_number(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{path} must be numeric")
    return float(value)


def _validate_runtime(config: dict[str, Any], *, default: bool) -> None:
    runtime = _optional_mapping(config, "runtime") if default else _require_mapping(config, "runtime")
    if default:
        runtime.setdefault("device", "cpu")
        runtime.setdefault("num_workers", 0)
        runtime.setdefault("seed", 42)

    device = _require_value(runtime, "device", "runtime.device")
    if device != "cpu":
        raise ValueError("runtime.device must be 'cpu'")

    seed = _require_value(runtime, "seed", "runtime.seed")
    if seed != 42:
        raise ValueError("runtime.seed must be 42")

    num_workers = _require_value(runtime, "num_workers", "runtime.num_workers")
    if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 0:
        raise ValueError("runtime.num_workers must be a non-negative integer")


def _validate_representation(value: Any, path: str = "data.representation") -> None:
    _validate_choice(value, SUPPORTED_REPRESENTATIONS, path)


def _validate_model(model: dict[str, Any]) -> None:
    model_type = _require_value(model, "type", "model.type")
    _validate_choice(model_type, {"icu_tiny_transformer"}, "model.type")

    d_model = _validate_positive_int(_require_value(model, "d_model", "model.d_model"), "model.d_model")
    n_heads = _validate_positive_int(_require_value(model, "n_heads", "model.n_heads"), "model.n_heads")
    if d_model % n_heads != 0:
        raise ValueError("model.d_model must be divisible by model.n_heads")

    _validate_positive_int(
        _require_value(model, "max_seq_len", "model.max_seq_len"),
        "model.max_seq_len",
    )
    if model["max_seq_len"] != 256:
        raise ValueError("model.max_seq_len must be 256")


def _validate_data_contract(data: dict[str, Any]) -> None:
    _validate_choice(
        _require_value(data, "dataset", "data.dataset"),
        {"eicu_demo"},
        "data.dataset",
    )
    _validate_representation(_require_value(data, "representation", "data.representation"))

    outcome = _require_mapping(data, "outcome")
    _validate_choice(
        _require_value(outcome, "name", "data.outcome.name"),
        {"hospital_mortality"},
        "data.outcome.name",
    )
    _validate_choice(
        _require_value(outcome, "source_field", "data.outcome.source_field"),
        {"hospitaldischargestatus"},
        "data.outcome.source_field",
    )

    window = _require_mapping(data, "observation_window")
    start = _require_value(window, "start_minutes", "data.observation_window.start_minutes")
    end = _require_value(window, "end_minutes", "data.observation_window.end_minutes")
    if start != 0 or end != 1440:
        raise ValueError("data.observation_window must be 0..1440 minutes")

    if _require_value(data, "max_seq_len", "data.max_seq_len") != 256:
        raise ValueError("data.max_seq_len must be 256")
    if _require_value(data, "min_dynamic_events", "data.min_dynamic_events") != 5:
        raise ValueError("data.min_dynamic_events must be 5")


def _validate_split(split: dict[str, Any]) -> None:
    strategy = _require_value(split, "strategy", "split.strategy")
    if strategy != "patient_grouped":
        raise ValueError("split.strategy must be patient_grouped")
    if _require_value(split, "group_key", "split.group_key") != "uniquepid":
        raise ValueError("split.group_key must be uniquepid")
    if _require_value(split, "seed", "split.seed") != 42:
        raise ValueError("split.seed must be 42")

    ratios = _require_mapping(split, "ratios")
    train = _validate_number(
        _require_value(ratios, "train", "split.ratios.train"),
        "split.ratios.train",
    )
    validation = _validate_number(
        _require_value(ratios, "validation", "split.ratios.validation"),
        "split.ratios.validation",
    )
    test = _validate_number(
        _require_value(ratios, "test", "split.ratios.test"),
        "split.ratios.test",
    )
    if any(value <= 0 for value in (train, validation, test)):
        raise ValueError("split.ratios values must be positive")
    if abs((train + validation + test) - 1.0) > 1e-8:
        raise ValueError("split.ratios must sum to 1")
    if (train, validation, test) != (0.7, 0.15, 0.15):
        raise ValueError("split.ratios must be 70/15/15")


def _validate_preprocessing(preprocessing: dict[str, Any]) -> None:
    if _require_value(preprocessing, "fit_on", "preprocessing.fit_on") != "train":
        raise ValueError("preprocessing.fit_on must be train")
    if (
        _require_value(
            preprocessing,
            "vital_bucket_minutes",
            "preprocessing.vital_bucket_minutes",
        )
        != 60
    ):
        raise ValueError("preprocessing.vital_bucket_minutes must be 60")
    if (
        _require_value(
            preprocessing,
            "lab_min_observations",
            "preprocessing.lab_min_observations",
        )
        != 50
    ):
        raise ValueError("preprocessing.lab_min_observations must be 50")
    if (
        _require_value(
            preprocessing,
            "vital_min_observations",
            "preprocessing.vital_min_observations",
        )
        != 50
    ):
        raise ValueError("preprocessing.vital_min_observations must be 50")


def _validate_data_processing(config: dict[str, Any], *, default: bool) -> None:
    processing = (
        _optional_mapping(config, "data_processing")
        if default
        else _require_mapping(config, "data_processing")
    )
    if default:
        processing.setdefault("csv_chunk_rows", 50000)
        processing.setdefault("partition_shards", 64)
        processing.setdefault("encoded_shard_stays", 128)

    for key in ("csv_chunk_rows", "partition_shards", "encoded_shard_stays"):
        _validate_positive_int(
            _require_value(processing, key, f"data_processing.{key}"),
            f"data_processing.{key}",
        )


def _validate_recovery(config: dict[str, Any], *, default: bool) -> None:
    recovery = (
        _optional_mapping(config, "recovery")
        if default
        else _require_mapping(config, "recovery")
    )
    if default:
        recovery.setdefault("enabled", True)
        recovery.setdefault("log_every_batches", 10)
        recovery.setdefault("checkpoint_every_optimizer_steps", 100)
        recovery.setdefault("keep_last_checkpoints", 2)
        recovery.setdefault("resume", "auto")

    if _require_value(recovery, "enabled", "recovery.enabled") is not True:
        raise ValueError("recovery.enabled must be true")
    _validate_positive_int(
        _require_value(recovery, "log_every_batches", "recovery.log_every_batches"),
        "recovery.log_every_batches",
    )
    _validate_positive_int(
        _require_value(
            recovery,
            "checkpoint_every_optimizer_steps",
            "recovery.checkpoint_every_optimizer_steps",
        ),
        "recovery.checkpoint_every_optimizer_steps",
    )
    _validate_positive_int(
        _require_value(
            recovery,
            "keep_last_checkpoints",
            "recovery.keep_last_checkpoints",
        ),
        "recovery.keep_last_checkpoints",
    )
    if _require_value(recovery, "resume", "recovery.resume") != "auto":
        raise ValueError("recovery.resume must be auto")


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

    selection_metric = evaluation.get("selection_metric")
    if (
        selection_metric is not None
        and selection_metric != "validation_average_precision"
    ):
        raise ValueError("evaluation.selection_metric must be validation_average_precision")


def _validate_training(
    pretraining: dict[str, Any],
    finetuning: dict[str, Any],
) -> None:
    mask_probability = _validate_number(
        _require_value(
            pretraining,
            "mask_probability",
            "pretraining.mask_probability",
        ),
        "pretraining.mask_probability",
    )
    if not 0.0 < mask_probability < 1.0:
        raise ValueError("pretraining.mask_probability must be between 0 and 1")

    for section_name, section in (
        ("pretraining", pretraining),
        ("finetuning", finetuning),
    ):
        for key in ("epochs", "batch_size"):
            _validate_positive_int(
                _require_value(section, key, f"{section_name}.{key}"),
                f"{section_name}.{key}",
            )

    if (
        _require_value(
            finetuning,
            "selection_metric",
            "finetuning.selection_metric",
        )
        != "validation_average_precision"
    ):
        raise ValueError(
            "finetuning.selection_metric must be validation_average_precision"
        )


def _iter_feature_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _iter_feature_values(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_feature_values(item)


def _validate_no_prohibited_features(config: dict[str, Any]) -> None:
    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if key in FEATURE_KEYS:
                    for feature in _iter_feature_values(item):
                        normalized = feature.replace("_", "").replace("-", "").lower()
                        if any(
                            fragment.replace("_", "") in normalized
                            for fragment in PROHIBITED_FEATURE_FRAGMENTS
                        ):
                            raise ValueError(f"{child_path} contains prohibited feature: {feature}")
                visit(item, child_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(config, "")


def _validate_common_protocol(config: dict[str, Any], *, default_runtime: bool) -> None:
    _validate_data_contract(_require_mapping(config, "data"))
    _validate_split(_require_mapping(config, "split"))
    _validate_preprocessing(_require_mapping(config, "preprocessing"))
    _validate_runtime(config, default=default_runtime)
    _validate_data_processing(config, default=default_runtime)
    _validate_recovery(config, default=default_runtime)
    _validate_no_prohibited_features(config)


def validate_final_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and copy the full training configuration."""
    validated = _copy_config(config, "final")
    for section in (
        "experiment",
        "data",
        "split",
        "preprocessing",
        "model",
        "pretraining",
        "finetuning",
        "evaluation",
        "runtime",
        "data_processing",
        "recovery",
    ):
        _require_mapping(validated, section)

    _validate_common_protocol(validated, default_runtime=False)
    _validate_model(validated["model"])
    _validate_training(validated["pretraining"], validated["finetuning"])
    _validate_evaluation(validated["evaluation"], add_metrics=False)
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
    trial_training = _require_mapping(validated, "trial_training")
    _validate_common_protocol(validated, default_runtime=True)

    _validate_choice(
        _require_value(study, "direction", "study.direction"),
        {"maximize"},
        "study.direction",
    )
    _validate_choice(
        _require_value(study, "metric", "study.metric"),
        {"val_average_precision"},
        "study.metric",
    )
    n_trials = _require_value(study, "n_trials", "study.n_trials")
    if (
        not isinstance(n_trials, int)
        or isinstance(n_trials, bool)
        or not 1 <= n_trials <= 8
    ):
        raise ValueError("study.n_trials must be between 1 and 8")

    d_models = _validate_search_choices(search_space, "d_model")
    n_heads = _validate_search_choices(search_space, "n_heads")
    for key in (
        "n_layers",
        "dim_feedforward",
        "dropout",
        "mask_probability",
        "learning_rate",
    ):
        _validate_search_choices(search_space, key)

    if "max_seq_len" in search_space and any(
        value != 256
        for value in _validate_search_choices(search_space, "max_seq_len")
    ):
        raise ValueError("search_space.max_seq_len must remain fixed at 256")
    if any(not isinstance(value, int) or value <= 0 for value in d_models):
        raise ValueError("search_space.d_model must contain positive integers")
    if any(not isinstance(value, int) or value <= 0 for value in n_heads):
        raise ValueError("search_space.n_heads must contain positive integers")
    if any(d_model % heads != 0 for d_model in d_models for heads in n_heads):
        raise ValueError("search_space.d_model values must be divisible by search_space.n_heads values")

    if (
        _require_value(
            trial_training,
            "pretrain_epochs",
            "trial_training.pretrain_epochs",
        )
        != 2
    ):
        raise ValueError("trial_training.pretrain_epochs must be 2")
    if (
        _require_value(
            trial_training,
            "finetune_epochs",
            "trial_training.finetune_epochs",
        )
        != 5
    ):
        raise ValueError("trial_training.finetune_epochs must be 5")

    return validated


def validate_fedavg_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and copy a FedAvg-style simulation configuration."""
    validated = _copy_config(config, "fedavg")
    for section in (
        "experiment",
        "data",
        "split",
        "preprocessing",
        "model",
        "fedavg",
        "runtime",
    ):
        _require_mapping(validated, section)

    _validate_common_protocol(validated, default_runtime=True)
    _validate_model(validated["model"])

    fedavg = validated["fedavg"]
    if (
        _require_value(fedavg, "client_grouping", "fedavg.client_grouping")
        != "hospital_cluster"
    ):
        raise ValueError("fedavg.client_grouping must be hospital_cluster")
    if _require_value(fedavg, "clients", "fedavg.clients") != 3:
        raise ValueError("fedavg.clients must be 3")
    if _require_value(fedavg, "rounds", "fedavg.rounds") != 3:
        raise ValueError("fedavg.rounds must be 3")
    if _require_value(fedavg, "local_epochs", "fedavg.local_epochs") != 1:
        raise ValueError("fedavg.local_epochs must be 1")
    if _require_value(fedavg, "client_fraction", "fedavg.client_fraction") != 1.0:
        raise ValueError("fedavg.client_fraction must be 1.0")
    if (
        _require_value(fedavg, "aggregation", "fedavg.aggregation")
        != "weighted_average_by_num_training_stays"
    ):
        raise ValueError(
            "fedavg.aggregation must be weighted_average_by_num_training_stays"
        )

    return validated


def _experiment_id_from_path(path: str) -> str:
    stem = Path(path).stem
    parts = stem.split("_", maxsplit=2)
    return f"EXP-{parts[1]}" if len(parts) >= 2 else ""


def _validate_short_experiment_shape(validated: dict[str, Any]) -> None:
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
        _require_value(validated, "model", "model"),
        SUPPORTED_MODELS,
        "model",
    )
    pretraining = _require_value(validated, "pretraining", "pretraining")
    if not isinstance(pretraining, bool) and pretraining != "selected":
        raise ValueError("pretraining must be a boolean or selected")

    optional = validated.get("optional", False)
    if not isinstance(optional, bool):
        raise ValueError("optional must be a boolean")


def validate_experiment_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and copy a short experiment or experiment-suite config."""
    validated = _copy_config(config, "experiment")
    if "suite" in validated:
        suite = _require_mapping(validated, "suite")
        _require_value(suite, "name", "suite.name")
        experiments = _require_value(suite, "experiments", "suite.experiments")
        if not isinstance(experiments, list) or not experiments:
            raise ValueError("suite.experiments must be a non-empty list")
        seen: set[str] = set()
        for path in experiments:
            if not isinstance(path, str):
                raise ValueError("suite.experiments entries must be paths")
            experiment_id = _experiment_id_from_path(path)
            if experiment_id not in EXPERIMENT_IDS:
                raise ValueError(f"suite.experiments contains unknown experiment: {path}")
            seen.add(experiment_id)
        if "EXP-05" in seen:
            raise ValueError("suite.experiments must not include optional EXP-05")
        _validate_runtime(validated, default=True)
        _validate_data_processing(validated, default=True)
        _validate_recovery(validated, default=True)
        return validated

    _validate_short_experiment_shape(validated)
    evaluation = _require_mapping(validated, "evaluation")
    _validate_evaluation(evaluation, add_metrics=True)
    _validate_runtime(validated, default=True)
    _validate_data_processing(validated, default=True)
    _validate_recovery(validated, default=True)
    _validate_no_prohibited_features(validated)
    return validated
