from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from icu_pretrain.tuning.optuna_search import _resume_incomplete_trials, run_optuna_search
from icu_pretrain.utils import load_yaml


ROOT = Path(__file__).resolve().parents[1]


class FakeTrialState:
    COMPLETE = "COMPLETE"
    FAIL = "FAIL"
    PRUNED = "PRUNED"


class FakeDistribution:
    def __init__(self, choices: tuple[object, ...]) -> None:
        self.choices = choices


class FakeTrialRecord:
    def __init__(self, number: int, params: dict[str, object], state: str, value: float | None = None) -> None:
        self.number = number
        self.params = dict(params)
        self.state = state
        self.value = value


class FakeTrial:
    def __init__(self, number: int, params: dict[str, object]) -> None:
        self.number = number
        self._params = dict(params)
        self.params: dict[str, object] = {}

    def suggest_categorical(self, name: str, choices: list[object]) -> object:
        value = self._params[name]
        self.params[name] = value
        return value


class FakeStudy:
    def __init__(self, param_queue: list[dict[str, object]]) -> None:
        self._param_queue = list(param_queue)
        self.trials: list[FakeTrialRecord] = []
        self._next_number = 0

    def ask(self) -> FakeTrial:
        if not self._param_queue:
            raise AssertionError("unexpected trial request")
        params = self._param_queue.pop(0)
        trial = FakeTrial(self._next_number, params)
        self._next_number += 1
        return trial

    def tell(self, trial: FakeTrial, value: float | None = None, state: str | None = None) -> None:
        if state is None:
            self.trials.append(FakeTrialRecord(trial.number, trial.params, FakeTrialState.COMPLETE, value))
            return
        self.trials.append(FakeTrialRecord(trial.number, trial.params, state, value))

    def add_trial(self, trial: FakeTrialRecord) -> None:
        self.trials.append(trial)


class FakeOptunaModule:
    def __init__(self, study: FakeStudy) -> None:
        self._study = study
        self.distributions = SimpleNamespace(CategoricalDistribution=FakeDistribution)
        self.trial = SimpleNamespace(TrialState=FakeTrialState, create_trial=self._create_trial)

    def create_study(self, **kwargs: object) -> FakeStudy:
        return self._study

    def _create_trial(self, *, params: dict[str, object], distributions: dict[str, object], value: float, state: str) -> FakeTrialRecord:
        return FakeTrialRecord(len(self._study.trials), params, state, value)


def _write_config(path: Path, config: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _prepare_configs(tmp_path: Path) -> tuple[Path, Path, Path]:
    tuning_config = load_yaml(ROOT / "configs" / "tuning" / "optuna_eicu_tiny.yaml")
    base_config = load_yaml(ROOT / "configs" / "final" / "eicu_demo_final_tiny.yaml")
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    tuning_path = _write_config(tmp_path / "tuning.yaml", tuning_config)
    base_path = _write_config(tmp_path / "base.yaml", base_config)
    return tuning_path, base_path, processed_dir


def _fake_train_model(calls: list[tuple[str, dict[str, object]]]):
    def _inner(config: dict[str, object], processed_dir: Path, run_dir: Path, resume: str = "auto") -> dict[str, object]:
        calls.append(("pretrain", deepcopy(config)))
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "best.pt").write_text("checkpoint", encoding="utf-8")
        return {"model_state": {}, "prediction_head_state": {}, "val_loss": 0.0}

    return _inner


def _fake_train_finetuning_model(calls: list[tuple[str, dict[str, object]]], val_ap: float = 0.73):
    def _inner(
        config: dict[str, object],
        processed_dir: Path,
        run_dir: Path,
        resume: str = "auto",
        pretrain_checkpoint: Path | None = None,
    ) -> dict[str, object]:
        calls.append(("finetune", deepcopy(config), pretrain_checkpoint))
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "results.json").write_text(json.dumps({"val_ap": val_ap}), encoding="utf-8")
        return {"val_loss": val_ap}

    return _inner


def test_missing_optuna_raises_actionable_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tuning_path, base_path, processed_dir = _prepare_configs(tmp_path)
    run_dir = tmp_path / "run"
    monkeypatch.setattr(
        "icu_pretrain.tuning.optuna_search.importlib.import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError("optuna")) if name == "optuna" else __import__(name),
    )

    with pytest.raises(RuntimeError, match="optional tuning extra"):
        run_optuna_search(
            [
                "--config",
                str(tuning_path),
                "--base_config",
                str(base_path),
                "--processed_dir",
                str(processed_dir),
                "--run_dir",
                str(run_dir),
                "--n_trials",
                "1",
            ]
        )


def test_successful_trial_keeps_fixed_data_and_writes_best_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tuning_path, base_path, processed_dir = _prepare_configs(tmp_path)
    run_dir = tmp_path / "run"
    params_queue = [
        {
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 4,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "learning_rate": 0.0005,
            "mask_probability": 0.15,
        }
    ]
    study = FakeStudy(params_queue)
    fake_optuna = FakeOptunaModule(study)
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        "icu_pretrain.tuning.optuna_search.importlib.import_module",
        lambda name: fake_optuna if name == "optuna" else __import__(name),
    )
    monkeypatch.setattr("icu_pretrain.tuning.optuna_search.train_model", _fake_train_model(calls))
    monkeypatch.setattr(
        "icu_pretrain.tuning.optuna_search.train_finetuning_model",
        _fake_train_finetuning_model(calls),
    )

    summary = run_optuna_search(
        [
            "--config",
            str(tuning_path),
            "--base_config",
            str(base_path),
            "--processed_dir",
            str(processed_dir),
            "--run_dir",
            str(run_dir),
            "--n_trials",
            "1",
        ]
    )

    assert summary["completed_trials"] == 1
    assert summary["best_value"] == pytest.approx(0.73)
    assert (run_dir / "best_params.json").is_file()
    assert (run_dir / "study_summary.json").is_file()
    assert len(calls) == 2
    pretrain_config = calls[0][1]
    finetune_config = calls[1][1]
    assert pretrain_config["data"] == load_yaml(base_path)["data"]
    assert pretrain_config["split"] == load_yaml(base_path)["split"]
    assert pretrain_config["preprocessing"] == load_yaml(base_path)["preprocessing"]
    assert pretrain_config["model"]["d_model"] == 64
    assert pretrain_config["pretraining"]["epochs"] == 2
    assert pretrain_config["pretraining"]["learning_rate"] == 0.0005
    assert finetune_config["finetuning"]["epochs"] == 5
    assert finetune_config["finetuning"]["learning_rate"] == 0.0005
    assert finetune_config["finetuning"]["evaluate_on_test"] is False
    assert isinstance(calls[1][2], Path)
    with open(run_dir / "best_params.json", "r", encoding="utf-8") as handle:
        best_params = json.load(handle)
    assert best_params == params_queue[0]


def test_invalid_sampled_pair_does_not_create_fake_best_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tuning_path, base_path, processed_dir = _prepare_configs(tmp_path)
    run_dir = tmp_path / "run"
    study = FakeStudy(
        [
            {
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "dim_feedforward": 256,
                "dropout": 0.1,
                "learning_rate": 0.0005,
                "mask_probability": 0.15,
            }
        ]
    )
    fake_optuna = FakeOptunaModule(study)
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        "icu_pretrain.tuning.optuna_search.importlib.import_module",
        lambda name: fake_optuna if name == "optuna" else __import__(name),
    )
    monkeypatch.setattr("icu_pretrain.tuning.optuna_search.train_model", _fake_train_model(calls))
    monkeypatch.setattr(
        "icu_pretrain.tuning.optuna_search.train_finetuning_model",
        _fake_train_finetuning_model(calls),
    )
    monkeypatch.setattr(
        "icu_pretrain.tuning.optuna_search._sample_trial_params",
        lambda trial, search_space: {
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 3,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "learning_rate": 0.0005,
            "mask_probability": 0.15,
        },
    )

    summary = run_optuna_search(
        [
            "--config",
            str(tuning_path),
            "--base_config",
            str(base_path),
            "--processed_dir",
            str(processed_dir),
            "--run_dir",
            str(run_dir),
            "--n_trials",
            "1",
        ]
    )

    assert summary["completed_trials"] == 0
    assert summary["best_params"] is None
    assert not (run_dir / "best_params.json").exists()
    assert len(calls) == 0


def test_resume_incomplete_trial_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tuning_path, base_path, processed_dir = _prepare_configs(tmp_path)
    run_dir = tmp_path / "run"
    trial_dir = run_dir / "trials" / "trial_000"
    trial_dir.mkdir(parents=True)
    manifest = {
        "trial_number": 0,
        "params": {
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 4,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "learning_rate": 0.0005,
            "mask_probability": 0.15,
        },
        "status": "interrupted",
        "updated_at": "2026-06-15T00:00:00Z",
    }
    (trial_dir / "trial.json").write_text(json.dumps(manifest), encoding="utf-8")
    study = FakeStudy([])
    fake_optuna = FakeOptunaModule(study)
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr("icu_pretrain.tuning.optuna_search.train_model", _fake_train_model(calls))
    monkeypatch.setattr(
        "icu_pretrain.tuning.optuna_search.train_finetuning_model",
        _fake_train_finetuning_model(calls),
    )

    resumed = _resume_incomplete_trials(
        study=study,
        optuna_module=fake_optuna,
        run_dir=run_dir,
        search_space=load_yaml(tuning_path)["search_space"],
        base_config=load_yaml(base_path),
        tuning_config=load_yaml(tuning_path),
        processed_dir=processed_dir,
        resume="auto",
    )

    assert resumed == 1
    assert len(calls) == 2
    with open(trial_dir / "trial.json", "r", encoding="utf-8") as handle:
        updated = json.load(handle)
    assert updated["status"] == "completed"
    assert study.trials[-1].state == FakeTrialState.COMPLETE
