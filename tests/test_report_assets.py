from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from icu_pretrain.analysis.plots import make_plots
from icu_pretrain.analysis.tables import make_tables


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_make_report_assets_writes_expected_outputs(tmp_path: Path) -> None:
    summary_dir = tmp_path / "results" / "summary"
    run_root = tmp_path / "results" / "runs"
    figures_dir = tmp_path / "results" / "figures"

    fieldnames = [
        "experiment_id",
        "representation",
        "num_patients",
        "num_stays",
        "alive_count",
        "expired_count",
        "split_strategy",
        "seed",
        "auroc",
        "auroc_ci_lower",
        "auroc_ci_upper",
        "average_precision",
        "average_precision_ci_lower",
        "average_precision_ci_upper",
        "f1",
        "balanced_accuracy",
        "parameter_count",
        "runtime",
        "exclusions",
        "failure_notes",
    ]
    row = {
        "experiment_id": "EXP-01",
        "representation": "timegap_static",
        "num_patients": 5,
        "num_stays": 7,
        "alive_count": 5,
        "expired_count": 2,
        "split_strategy": "patient_grouped",
        "seed": 42,
        "auroc": 0.876,
        "auroc_ci_lower": "",
        "auroc_ci_upper": "",
        "average_precision": 0.812,
        "average_precision_ci_lower": "",
        "average_precision_ci_upper": "",
        "f1": 0.667,
        "balanced_accuracy": 0.701,
        "parameter_count": 1200,
        "runtime": 12.5,
        "exclusions": "none",
        "failure_notes": "",
    }
    _write_csv(summary_dir / "experiment_comparison.csv", fieldnames, [row])
    _write_csv(summary_dir / "final_results.csv", fieldnames, [row])
    _write_json(summary_dir / "best_config.json", {"model": {"d_model": 64}, "recovery": {"resume": "auto"}})
    _write_json(
        run_root / "hospital_fold_01" / "hospital_grouped_summary.json",
        {
            "fold_count": 5,
            "pooled": {
                "auroc": 0.8,
                "average_precision": 0.75,
                "f1": 0.7,
                "balanced_accuracy": 0.72,
            },
        },
    )

    payload = make_tables(summary_dir=summary_dir, run_root=run_root)
    figures = make_plots(summary_dir=summary_dir, run_root=run_root, figures_dir=figures_dir)

    cohort_summary = json.loads((summary_dir / "cohort_summary.json").read_text(encoding="utf-8"))
    assert cohort_summary["icu_stays"] == 2520
    assert cohort_summary["patients"] == 1841
    assert cohort_summary["hospitals"] == 186
    assert cohort_summary["wards"] == 292

    report_json = json.loads((summary_dir / "report_tables.json").read_text(encoding="utf-8"))
    assert report_json["cohort_summary"]["icu_stays"] == 2520
    assert report_json["cohort_summary"]["patients"] == 1841
    assert report_json["selected_model"][0]["experiment_id"] == "EXP-01"
    assert report_json["hospital_grouped_runs"][0]["run_id"] == "hospital_fold_01"

    report_md = (summary_dir / "report_tables.md").read_text(encoding="utf-8")
    assert "<10" in report_md
    assert "patient_grouped" in report_md

    assert payload["cohort_summary"]["icu_stays"] == 2520
    assert len(figures) == 3
    for figure_path in figures:
        assert figure_path.exists()
        assert figure_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_make_report_assets_handles_header_only_tables(tmp_path: Path) -> None:
    summary_dir = tmp_path / "results" / "summary"
    run_root = tmp_path / "results" / "runs"

    fieldnames = [
        "experiment_id",
        "representation",
        "num_patients",
        "num_stays",
        "alive_count",
        "expired_count",
        "split_strategy",
        "seed",
        "auroc",
        "auroc_ci_lower",
        "auroc_ci_upper",
        "average_precision",
        "average_precision_ci_lower",
        "average_precision_ci_upper",
        "f1",
        "balanced_accuracy",
        "parameter_count",
        "runtime",
        "exclusions",
        "failure_notes",
    ]
    _write_csv(summary_dir / "experiment_comparison.csv", fieldnames, [])
    _write_csv(summary_dir / "final_results.csv", fieldnames, [])

    payload = make_tables(summary_dir=summary_dir, run_root=run_root)

    report_md = (summary_dir / "report_tables.md").read_text(encoding="utf-8")
    assert "No rows available." not in report_md
    assert "| experiment_id | representation | num_patients |" in report_md
    assert payload["selected_model"] == []


def test_make_report_assets_rejects_patient_level_fields(tmp_path: Path) -> None:
    summary_dir = tmp_path / "results" / "summary"
    run_root = tmp_path / "results" / "runs"

    fieldnames = [
        "experiment_id",
        "representation",
        "patientunitstayid",
        "num_patients",
        "num_stays",
        "alive_count",
        "expired_count",
        "split_strategy",
        "seed",
        "auroc",
        "average_precision",
        "f1",
        "balanced_accuracy",
        "parameter_count",
        "runtime",
        "exclusions",
        "failure_notes",
    ]
    row = {
        "experiment_id": "EXP-01",
        "representation": "timegap_static",
        "patientunitstayid": "123",
        "num_patients": 5,
        "num_stays": 7,
        "alive_count": 5,
        "expired_count": 2,
        "split_strategy": "patient_grouped",
        "seed": 42,
        "auroc": 0.876,
        "average_precision": 0.812,
        "f1": 0.667,
        "balanced_accuracy": 0.701,
        "parameter_count": 1200,
        "runtime": 12.5,
        "exclusions": "none",
        "failure_notes": "",
    }
    _write_csv(summary_dir / "experiment_comparison.csv", fieldnames, [row])
    _write_csv(summary_dir / "final_results.csv", fieldnames, [row])

    with pytest.raises(ValueError):
        make_tables(summary_dir=summary_dir, run_root=run_root)
