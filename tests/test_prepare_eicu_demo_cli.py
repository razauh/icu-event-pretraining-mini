"""Tests for the local eICU demo preparation command."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

from icu_pretrain.data.eicu_demo import MVP_TABLES, table_path
from icu_pretrain.data.eicu_event_builder import read_event_streams_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "prepare_eicu_demo.py"


def _write_synthetic_tables(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True)
    for table_name in MVP_TABLES:
        frame = pd.DataFrame({"patientunitstayid": pd.Series(dtype="int64")})
        frame.to_csv(table_path(raw_dir, table_name), index=False)

    pd.DataFrame(
        {
            "patientunitstayid": [101],
            "age": [65],
            "gender": ["Female"],
            "hospitaldischargestatus": ["Alive"],
        }
    ).to_csv(table_path(raw_dir, "patient"), index=False)
    pd.DataFrame(
        {
            "patientunitstayid": [101, 101],
            "diagnosisoffset": [0, 60],
            "diagnosisstring": ["Sepsis", "Acidosis"],
        }
    ).to_csv(table_path(raw_dir, "diagnosis"), index=False)


def _run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    source_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_path, env.get("PYTHONPATH", "")) if part
    )
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        cwd=cwd or REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_writes_expected_processed_artifacts(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    _write_synthetic_tables(raw_dir)

    result = _run_cli(
        "--raw_dir",
        str(raw_dir),
        "--out_dir",
        str(out_dir),
    )

    assert result.returncode == 0, result.stderr
    assert {path.name for path in out_dir.iterdir()} == {
        "event_streams.jsonl",
        "outcomes.csv",
        "event_stats.json",
    }
    streams = read_event_streams_jsonl(out_dir / "event_streams.jsonl")
    assert len(streams) == 1
    assert streams[0].representation == "timegap_static"
    assert pd.read_csv(out_dir / "outcomes.csv").to_dict("records") == [
        {"patientunitstayid": 101, "mortality": 0}
    ]
    stats = json.loads((out_dir / "event_stats.json").read_text(encoding="utf-8"))
    assert stats["kept_stays"] == 1
    assert "Prepared 1 event stream" in result.stdout
    assert "patientunitstayid" not in result.stdout
    assert "DX::" not in result.stdout


def test_cli_supports_relative_paths_and_replaces_existing_outputs(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    _write_synthetic_tables(raw_dir)
    out_dir.mkdir()
    for filename in ("event_streams.jsonl", "outcomes.csv", "event_stats.json"):
        (out_dir / filename).write_text("stale", encoding="utf-8")

    result = _run_cli(
        "--raw_dir",
        "raw",
        "--out_dir",
        "processed",
        "--representation",
        "basic",
        "--min_events_per_stay",
        "1",
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert "stale" not in (out_dir / "event_streams.jsonl").read_text(
        encoding="utf-8"
    )
    assert json.loads((out_dir / "event_stats.json").read_text(encoding="utf-8"))[
        "kept_stays"
    ] == 1


def test_cli_writes_valid_empty_outputs_after_filtering(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    _write_synthetic_tables(raw_dir)

    result = _run_cli(
        "--raw_dir",
        str(raw_dir),
        "--out_dir",
        str(out_dir),
        "--min_events_per_stay",
        "100",
    )

    assert result.returncode == 0, result.stderr
    assert (out_dir / "event_streams.jsonl").read_text(encoding="utf-8") == ""
    assert list(pd.read_csv(out_dir / "outcomes.csv").columns) == [
        "patientunitstayid",
        "mortality",
    ]
    stats = json.loads((out_dir / "event_stats.json").read_text(encoding="utf-8"))
    assert stats["kept_stays"] == 0


def test_cli_fails_before_writing_when_raw_tables_are_missing(tmp_path: Path) -> None:
    out_dir = tmp_path / "processed"

    result = _run_cli(
        "--raw_dir",
        str(tmp_path / "missing"),
        "--out_dir",
        str(out_dir),
    )

    assert result.returncode != 0
    assert "patient.csv" in result.stderr
    assert not out_dir.exists()


def test_cli_rejects_invalid_arguments_without_writing(tmp_path: Path) -> None:
    out_dir = tmp_path / "processed"

    unsupported = _run_cli(
        "--raw_dir",
        str(tmp_path / "raw"),
        "--out_dir",
        str(out_dir),
        "--representation",
        "unsupported",
    )
    non_positive = _run_cli(
        "--raw_dir",
        str(tmp_path / "raw"),
        "--out_dir",
        str(out_dir),
        "--min_events_per_stay",
        "0",
    )

    assert unsupported.returncode != 0
    assert "invalid choice" in unsupported.stderr
    assert non_positive.returncode != 0
    assert "positive integer" in non_positive.stderr
    assert not out_dir.exists()


def test_cli_keeps_all_outputs_within_requested_directory(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "nested" / "processed"
    _write_synthetic_tables(raw_dir)

    result = _run_cli(
        "--raw_dir",
        str(raw_dir),
        "--out_dir",
        str(out_dir),
        "--min_events_per_stay",
        "1",
    )

    assert result.returncode == 0, result.stderr
    generated_names = {"event_streams.jsonl", "outcomes.csv", "event_stats.json"}
    assert generated_names.isdisjoint(path.name for path in tmp_path.iterdir())
    assert generated_names == {path.name for path in out_dir.iterdir()}
