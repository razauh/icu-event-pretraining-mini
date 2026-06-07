"""Package-level tests that stay independent of ICU data and optional tools."""

from __future__ import annotations

import builtins
import importlib
from pathlib import Path

import pytest

import icu_pretrain
from icu_pretrain.utils import load_yaml


PACKAGE_MODULES = (
    "icu_pretrain.data",
    "icu_pretrain.data.collate",
    "icu_pretrain.data.dataset",
    "icu_pretrain.data.eicu_demo",
    "icu_pretrain.data.eicu_event_builder",
    "icu_pretrain.data.splits",
    "icu_pretrain.data.tokenizer",
    "icu_pretrain.models",
    "icu_pretrain.models.heads",
    "icu_pretrain.models.transformer",
    "icu_pretrain.training",
    "icu_pretrain.training.baselines",
    "icu_pretrain.training.evaluate",
    "icu_pretrain.training.federated",
    "icu_pretrain.training.finetune",
    "icu_pretrain.training.pretrain",
    "icu_pretrain.experiments",
    "icu_pretrain.experiments.registry",
    "icu_pretrain.experiments.runner",
    "icu_pretrain.experiments.tracking",
    "icu_pretrain.tuning",
    "icu_pretrain.tuning.optuna_search",
    "icu_pretrain.analysis",
    "icu_pretrain.analysis.memorisation",
    "icu_pretrain.analysis.plots",
    "icu_pretrain.analysis.tables",
)


def test_package_exposes_version() -> None:
    assert icu_pretrain.__version__ == "0.1.0"


def test_package_modules_import_without_data_or_optional_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__
    original_open = Path.open

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".", maxsplit=1)[0] in {"jupyter", "optuna"}:
            raise AssertionError(f"Package import unexpectedly required {name}")
        return original_import(name, *args, **kwargs)

    def guarded_open(path: Path, *args: object, **kwargs: object) -> object:
        normalized = path.as_posix()
        if "data/raw/" in normalized or "data/processed/" in normalized:
            raise AssertionError(f"Package import unexpectedly read {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(Path, "open", guarded_open)

    for module_name in PACKAGE_MODULES:
        assert importlib.import_module(module_name).__name__ == module_name


def test_load_yaml_returns_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("runtime:\n  device: cpu\n", encoding="utf-8")

    assert load_yaml(config_path) == {"runtime": {"device": "cpu"}}


def test_load_yaml_returns_empty_mapping_for_empty_file(tmp_path: Path) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    assert load_yaml(config_path) == {}


def test_load_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "list.yaml"
    config_path.write_text("- first\n- second\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected mapping"):
        load_yaml(config_path)
