from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from icu_pretrain.experiments.tracking import validate_public_safe


DEFAULT_SUMMARY_DIR = Path("results/summary")
DEFAULT_RUN_ROOT = Path("results/runs")
DEFAULT_DOCS_DIR = Path("docs")
DEFAULT_FIGURES_DIR = Path("results/figures")

PUBLIC_COHORT_SUMMARY = {
    "dataset": "eicu_demo",
    "dataset_version": "2.0.1",
    "icu_stays": 2520,
    "patients": 1841,
    "hospitals": 186,
    "wards": 292,
    "notes": "Verified demo profile from docs/plan.md.",
}

COUNT_COLUMNS = {
    "num_patients",
    "num_stays",
    "alive_count",
    "expired_count",
    "total_stays",
    "kept_stays",
    "skipped_stays",
    "train_count",
    "validation_count",
    "test_count",
    "client_count",
    "round_count",
    "fold_count",
    "training_token_count",
    "training_rare_ngram_count",
}

METRIC_COLUMNS = {
    "auroc",
    "average_precision",
    "f1",
    "balanced_accuracy",
    "ap",
    "threshold",
}

FORBIDDEN_PUBLIC_FIELDS = {
    "uniquepid",
    "patientunitstayid",
    "patient_ids",
    "stay_ids",
    "tokens",
    "event_stream",
    "events",
}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _is_numeric(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _coerce_number(value: Any) -> float | None:
    if not _is_numeric(value):
        return None
    number = float(value)
    if not number == number:
        return None
    return number


def _format_value(column: str, value: Any) -> str:
    if value in (None, ""):
        return ""
    number = _coerce_number(value)
    if column in COUNT_COLUMNS and number is not None:
        if 0 <= number < 10:
            return "<10"
        if float(number).is_integer():
            return str(int(number))
        return f"{number:.2f}".rstrip("0").rstrip(".")
    if column in METRIC_COLUMNS and number is not None:
        return f"{number:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _flatten_mapping(mapping: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for key in sorted(mapping):
        value = mapping[key]
        label = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            items.extend(_flatten_mapping(value, label))
        else:
            items.append((label, value))
    return items


def _markdown_table(headers: list[str], rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    if not rows:
        return "\n".join(lines)
    for row in rows:
        lines.append("| " + " | ".join(_format_value(header, row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def _load_optional_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.is_file():
        return None
    payload = _load_json(path)
    validate_public_safe(payload)
    return payload


def _load_run_summaries(run_root: Path, filename: str) -> list[dict[str, Any]]:
    if not run_root.is_dir():
        return []
    summaries: list[dict[str, Any]] = []
    for path in sorted(run_root.rglob(filename)):
        payload = _load_json(path)
        validate_public_safe(payload)
        relative = path.relative_to(run_root)
        run_id = relative.parts[0] if relative.parts else path.stem
        summaries.append({"run_id": run_id, "path": str(relative), "summary": payload})
    return summaries


def _selected_model_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_keys = [
        "experiment_id",
        "representation",
        "split_strategy",
        "auroc",
        "average_precision",
        "f1",
        "balanced_accuracy",
    ]
    selected_rows = []
    for row in rows:
        selected_rows.append({key: row.get(key, "") for key in selected_keys})
    return selected_rows


def _prepare_cohort_summary(summary_dir: Path) -> dict[str, Any]:
    cohort_path = summary_dir / "cohort_summary.json"
    payload = _load_optional_json(cohort_path)
    if isinstance(payload, dict):
        return payload
    cohort_summary = dict(PUBLIC_COHORT_SUMMARY)
    validate_public_safe(cohort_summary)
    _atomic_write_json(cohort_path, cohort_summary)
    return cohort_summary


def load_report_payload(
    summary_dir: Path | None = None,
    run_root: Path | None = None,
) -> dict[str, Any]:
    summary_dir = Path(summary_dir or DEFAULT_SUMMARY_DIR)
    run_root = Path(run_root or DEFAULT_RUN_ROOT)

    cohort_summary = _prepare_cohort_summary(summary_dir)
    experiment_comparison = _load_csv_rows(summary_dir / "experiment_comparison.csv")
    final_results = _load_csv_rows(summary_dir / "final_results.csv")
    best_config = _load_optional_json(summary_dir / "best_config.json")
    hospital_grouped_runs = _load_run_summaries(run_root, "hospital_grouped_summary.json")
    fedavg_runs = _load_run_summaries(run_root, "fedavg_summary.json")
    memorisation_runs = _load_run_summaries(run_root, "memorisation_probe.json")

    for row in experiment_comparison + final_results:
        validate_public_safe(row)

    payload = {
        "cohort_summary": cohort_summary,
        "experiment_comparison": experiment_comparison,
        "final_results": final_results,
        "best_config": best_config or {},
        "selected_model": _selected_model_rows(final_results or experiment_comparison),
        "hospital_grouped_runs": hospital_grouped_runs,
        "fedavg_runs": fedavg_runs,
        "memorisation_runs": memorisation_runs,
    }
    validate_public_safe(payload)
    return payload


def _write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    sections = []
    sections.append("# Result Tables")
    sections.append("")
    sections.append("## Cohort Summary")
    sections.append(_markdown_table(["field", "value"], [{"field": key, "value": value} for key, value in _flatten_mapping(payload["cohort_summary"]) ]))
    sections.append("")
    sections.append("## Selected Config")
    sections.append(_markdown_table(["field", "value"], [{"field": key, "value": value} for key, value in _flatten_mapping(payload["best_config"]) ]))
    sections.append("")
    sections.append("## Model Comparison")
    sections.append(_markdown_table([
        "experiment_id",
        "representation",
        "split_strategy",
        "auroc",
        "average_precision",
        "f1",
        "balanced_accuracy",
    ], payload["selected_model"]))
    sections.append("")
    sections.append("## Experiment Comparison")
    sections.append(_markdown_table([
        "experiment_id",
        "representation",
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
        "failure_notes",
    ], payload["experiment_comparison"]))
    sections.append("")
    sections.append("## Final Results")
    sections.append(_markdown_table([
        "experiment_id",
        "representation",
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
        "failure_notes",
    ], payload["final_results"]))
    sections.append("")

    grouped_rows = []
    for item in payload["hospital_grouped_runs"]:
        summary = item["summary"]
        pooled = summary.get("pooled", {})
        grouped_rows.append(
            {
                "run_id": item["run_id"],
                "fold_count": summary.get("fold_count", ""),
                "auroc": pooled.get("auroc", ""),
                "average_precision": pooled.get("average_precision", ""),
                "f1": pooled.get("f1", ""),
                "balanced_accuracy": pooled.get("balanced_accuracy", ""),
            }
        )
    sections.append("## Hospital-Grouped Runs")
    sections.append(_markdown_table(["run_id", "fold_count", "auroc", "average_precision", "f1", "balanced_accuracy"], grouped_rows))
    sections.append("")

    fedavg_rows = []
    for item in payload["fedavg_runs"]:
        summary = item["summary"]
        fedavg = summary.get("federated", {})
        central = summary.get("central_reference", {})
        fedavg_rows.append(
            {
                "run_id": item["run_id"],
                "client_count": summary.get("client_count", ""),
                "round_count": summary.get("round_count", ""),
                "central_auroc": central.get("auroc", ""),
                "federated_auroc": fedavg.get("auroc", ""),
                "federated_average_precision": fedavg.get("average_precision", ""),
            }
        )
    sections.append("## FedAvg Runs")
    sections.append(_markdown_table(["run_id", "client_count", "round_count", "central_auroc", "federated_auroc", "federated_average_precision"], fedavg_rows))
    sections.append("")

    memorisation_rows = []
    for item in payload["memorisation_runs"]:
        summary = item["summary"]
        overlap = summary.get("rare_ngram_overlap", {})
        memorisation_rows.append(
            {
                "run_id": item["run_id"],
                "status": summary.get("status", ""),
                "training_rare_ngram_count": summary.get("training_rare_ngram_count", ""),
                "rare_overlap_rate": overlap.get("rate", ""),
            }
        )
    sections.append("## Memorisation Runs")
    sections.append(_markdown_table(["run_id", "status", "training_rare_ngram_count", "rare_overlap_rate"], memorisation_rows))
    sections.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")


def make_tables(
    summary_dir: Path | None = None,
    run_root: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    summary_dir = Path(summary_dir or DEFAULT_SUMMARY_DIR)
    run_root = Path(run_root or DEFAULT_RUN_ROOT)
    output_path = Path(output_path or summary_dir / "report_tables.md")
    payload = load_report_payload(summary_dir=summary_dir, run_root=run_root)
    _atomic_write_json(summary_dir / "report_tables.json", payload)
    _write_markdown_report(output_path, payload)
    return payload
