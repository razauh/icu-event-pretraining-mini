# ICU Event Pretraining Mini

This repository is a CPU-friendly technical prototype built on the eICU demo
v2.0.1 dataset. It constructs leakage-controlled first-24-hour ICU stay event
streams, pretrains a compact Transformer with masked event modelling, and
evaluates hospital-mortality prediction against scratch and bag-of-events
baselines.

The verified demo cohort contains 2,520 raw ICU stays from 1,841 patients
across 186 hospitals and 292 wards. The repository keeps raw and processed
patient-level data local and does not claim clinical readiness or deployment
utility.

## Scope

- eICU demo dataset only.
- Event construction, tokenisation, masked pretraining, downstream evaluation,
  pseudo-client grouped evaluation, Optuna tuning, and FedAvg-style simulation.
- Aggregate reporting only. Public summaries distinguish patient-grouped,
  hospital-grouped, and simulated-client results.

## Public Artifacts

- `results/summary/cohort_summary.json`
- `results/summary/experiment_comparison.csv`
- `results/summary/final_results.csv`
- `results/summary/best_config.json`
- `results/figures/model_comparison.png`
- `results/figures/hospital_grouped_performance.png`
- `results/figures/fedavg_vs_centralised.png`

## Reporting

Generate public result tables and figures with:

```bash
python scripts/make_report_assets.py
```

The script writes aggregate tables to `results/summary/report_tables.json` and
`results/summary/report_tables.md`, then produces the public figures in
`results/figures/`.

## Recovery

Preprocessing and training use local run directories under `results/runs/<run_id>/`
with `run.log`, `events.jsonl`, `state.json`, and `checkpoints/`. Default data
processing uses `csv_chunk_rows: 50000`, `partition_shards: 64`, and
`encoded_shard_stays: 128`. Resume defaults to `auto`, and `--restart-stage`
can discard a failed stage and downstream outputs when compatibility checks
detect a mismatch.
