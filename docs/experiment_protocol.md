# Experiment Protocol

## Study Order

Run the staged experiments in this order: representation ablations, pretrained
versus scratch finetuning, constrained Optuna tuning, final verification,
patient-grouped evaluation, hospital-grouped evaluation, FedAvg-style
simulation, and the optional memorisation diagnostic.

## Processing Defaults

The preprocessing pipeline streams local CSVs in chunks of 50,000 rows, splits
normalised patient data into 64 deterministic shards, and encodes at most 128
stays per output shard. These defaults are configurable, but compatibility
checks require matching values when resuming a run.

## Run State

Every preprocessing, training, tuning, grouped-evaluation, and FedAvg run uses
`results/runs/<run_id>/run.log`, `results/runs/<run_id>/events.jsonl`,
`results/runs/<run_id>/state.json`, and `results/runs/<run_id>/checkpoints/`.
Logs are local only and never include patient identifiers, raw tokens, or raw
rows.

## Checkpoints

The training loops save a full checkpoint every 100 optimizer steps, at epoch
boundaries, after new best validation metrics, and on controlled interruption.
The newest two periodic checkpoints are retained along with the best and final
checkpoints.

## Resume and Recovery

Resume defaults to `auto`. Compatible runs continue from the last valid
checkpoint or child run state. If the config, inputs, schema, or upstream hashes
differ, automatic resume stops with an incompatibility error rather than mixing
artifacts. `--restart-stage <name>` can discard a failed stage and all
downstream outputs before the next run.

Unexpected interruptions resume from the next valid batch when a compatible
checkpoint exists. Incompatible artifacts, missing checkpoints, or changed
defaults require rerunning the affected stage.

## Public Reporting

Aggregate reports are generated from completed result rows only. Public tables
and figures distinguish patient-grouped, hospital-grouped, and simulated-client
results and stay limited to the eICU demo scope.
