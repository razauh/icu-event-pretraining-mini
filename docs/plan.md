# ICU Event Pretraining Mini: Dataset-Grounded Research and Implementation Plan

## 1. Project Positioning

Repository name:

```text
icu-event-pretraining-mini
```

Subtitle:

```text
A CPU-friendly PyTorch prototype for masked pretraining on heterogeneous ICU
event streams using the eICU demo dataset.
```

This repository demonstrates how the locally available eICU Collaborative
Research Database demo v2.0.1 can be converted into ICU stay-level event
streams and used to train a compact Transformer encoder with masked event
modelling. The project includes leakage-controlled hospital-mortality
prediction, controlled representation experiments, exploratory hospital-grouped
evaluation, and a small FedAvg-style simulation over hospital clusters.

The project is exploratory. It does not claim:

- clinical deployment readiness or clinical decision support;
- prospective, causal, or temporal generalisation;
- reliable performance for individual hospitals;
- true multicentre generalisation beyond this small demo sample;
- a clinical foundation model;
- secure or privacy-preserving federated learning;
- formal privacy guarantees.

## 2. Verified Dataset Profile

Use only the local eICU demo v2.0.1 files under:

```text
data/raw/eicu_demo/physionet.org/files/eicu-crd-demo/2.0.1/
```

The raw files are the source of truth. The local distribution contains:

| Item | Verified count |
|---|---:|
| ICU unit stays | 2,520 |
| Hospital-system encounters | 2,174 |
| De-identified patients | 1,841 |
| De-identified hospitals | 186 |
| De-identified wards | 292 |
| CSV tables | 31 |

Important identifiers in `patient.csv.gz`:

- `patientunitstayid`: one ICU unit stay and the event-table join key;
- `patienthealthsystemstayid`: one hospital-system encounter;
- `uniquepid`: one de-identified patient and the primary split-group key;
- `hospitalid`: de-identified hospital grouping variable;
- `wardid`: de-identified ward grouping variable.

Hospital and ward identifiers are present and complete in the local demo. They
permit exploratory grouped analysis, but they do not make this dataset large
enough for strong hospital-level claims. Hospitals contain only 10-40 stays,
and many have no hospital deaths in the demo.

Raw and processed ICU data remain local and ignored by Git. No patient-level
records, event streams, split assignments, vocabularies derived from protected
data, checkpoints, or local databases may be committed.

## 3. Research Objective

The primary research question is:

```text
Does masked event pretraining improve hospital-mortality prediction over a
scratch Transformer and a bag-of-events baseline when all models use the same
leakage-controlled first-24-hour ICU representation?
```

The pipeline is:

```text
eICU demo CSV tables
    -> eligible ICU stays and patient-grouped splits
    -> first-24-hour ICU event streams
    -> training-only tokenisation and numeric binning
    -> masked event pretraining
    -> hospital-mortality fine-tuning
    -> controlled model comparison
    -> exploratory hospital-grouped evaluation
    -> optional FedAvg-style hospital-cluster simulation
```

The unit of representation is an ICU stay. The unit of split isolation is a
patient.

## 4. Cohort and Outcome Definition

### 4.1 Primary outcome

The primary label is hospital mortality from `patient.csv.gz`:

```text
hospital_mortality = 1 if hospitaldischargestatus == "Expired" else 0
```

Rows with missing or unrecognised `hospitaldischargestatus` are excluded. In
the unfiltered raw patient table, 2,492 stays have a recognised label: 2,280
Alive and 212 Expired.

Do not substitute APACHE mortality fields for this label. ICU mortality from
`unitdischargestatus` may be implemented later as a separately named secondary
task; it must not be mixed with the primary outcome.

### 4.2 Prediction time and observation window

The prediction time is 24 hours after ICU admission. The primary representation
uses dynamic events with offsets in the inclusive interval:

```text
0 <= event_offset <= 1440 minutes
```

The task is therefore hospital-mortality prediction after observing the first
24 hours of an ICU stay.

Negative-offset events are excluded from the primary experiment. They may only
be evaluated in a clearly labelled future ablation. Events after minute 1440
must never enter the primary model input.

### 4.3 Eligibility rules

A stay is eligible when all of the following hold:

1. `patientunitstayid`, `uniquepid`, and `hospitalid` are present.
2. `hospitaldischargestatus` is Alive or Expired.
3. `unitdischargeoffset >= 1440`, so the patient remains in the ICU through the
   complete observation window.
4. At least five retained dynamic events remain after filtering,
   normalization, aggregation, and the 24-hour time restriction.

Apply eligibility before constructing final split assignments. Report the
number of exclusions at each rule, split by mortality label where possible.
Do not silently replace a 24-hour cohort with shorter-stay padding.

## 5. Source Tables and Leakage Controls

### 5.1 MVP tables

Use these tables:

| Logical table | Exact raw file | Purpose | Event time |
|---|---|---|---|
| patient | `patient.csv.gz` | IDs, eligibility, static context, outcome | Static |
| diagnosis | `diagnosis.csv.gz` | Diagnosis tokens | `diagnosisoffset` |
| lab | `lab.csv.gz` | Laboratory tokens | `labresultoffset` |
| medication | `medication.csv.gz` | Medication tokens | `drugstartoffset` |
| infusiondrug | `infusiondrug.csv.gz` | Infusion tokens | `infusionoffset` |
| treatment | `treatment.csv.gz` | Treatment/procedure tokens | `treatmentoffset` |
| vitalPeriodic | `vitalPeriodic.csv.gz` | Periodic vital tokens | `observationoffset` |
| vitalAperiodic | `vitalAperiodic.csv.gz` | Aperiodic vital tokens | `observationoffset` |

The loader must either accept the exact versioned raw directory above or
discover exactly one `eicu-crd-demo/2.0.1` directory below the configured raw
root. Ambiguous or missing matches must fail with a clear error.

Large event tables must not be loaded in full. Read compressed CSV files with
`pandas.read_csv(..., chunksize=50000)` by default, requesting only required
columns and explicit dtypes where practical. `patient.csv.gz` may be loaded in
full because it contains only 2,520 rows, but the implementation must also
support chunked patient loading for consistency and synthetic stress tests.

As rows are normalized, partition them into 64 deterministic local shards using
`sha256(str(patientunitstayid)) mod 64`. Process and reduce one shard at a time.
This keeps all rows for one ICU stay together without retaining complete event
tables in memory. Row chunk size and shard count are configurable; defaults are
chosen for an 8 GB CPU-only system.

### 5.2 Tables excluded from model inputs

`apachePatientResult.csv.gz` is not an input table for the mortality model. It
contains actual and predicted mortality and length-of-stay fields and has two
APACHE-version rows for each covered stay. It may be used only for aggregate
label-quality cross-checks that do not affect training.

The following fields are prohibited as model inputs:

- all actual or predicted ICU/hospital mortality fields;
- `diedinhospital`;
- all actual or predicted ICU/hospital length-of-stay fields;
- hospital or unit discharge status, location, time, year, or offset;
- `activeupondischarge` fields;
- APACHE scores or prediction outputs in the primary experiment;
- identifiers, including patient, encounter, stay, hospital, and ward IDs.

Hospital and ward IDs are grouping metadata only. They must not be converted to
model tokens.

## 6. Split Protocol and Fitted Preprocessing

### 6.1 Primary split

Create one reproducible 70/15/15 train/validation/test split using seed 42.
Group by `uniquepid` so every stay from one patient is assigned to exactly one
partition. Preserve mortality prevalence as closely as possible with a
stratified group assignment.

The implementation must verify:

- no `uniquepid` appears in more than one partition;
- no `patientunitstayid` appears in more than one partition;
- every eligible stay appears exactly once;
- both outcome classes occur in every partition.

The test partition remains untouched until the model configuration and decision
threshold are selected from training and validation data.

### 6.2 Training-only fitting

After split assignment, fit all data-dependent transformations on training
stays only:

- numeric quantile boundaries;
- token vocabulary and minimum-frequency filtering;
- category normalization maps derived from observed values;
- unknown-category handling;
- any clipping or imputation statistics.

Persist split metadata and fitted transformation metadata only under ignored
processed-data directories. Validation and test data may use fitted rules but
must not update them.

## 7. Event Construction

### 7.1 Static context

Prepend these static tokens to every stream:

```text
STATIC::AGE_BIN::<bin>
STATIC::GENDER::<value>
STATIC::UNIT_ADMIT_SOURCE::<value>
STATIC::UNIT_TYPE::<value>
```

Rules:

- Use `unitadmitsource`, not `hospitaladmitsource`.
- Normalize categorical text by trimming whitespace, uppercasing, and replacing
  internal whitespace runs with `_`.
- Map missing and unrecognised values to `UNKNOWN`.
- Parse numeric ages into `0_17`, `18_39`, `40_59`, `60_79`, and `80_PLUS`.
- Map the de-identified age value `> 89` to `80_PLUS`.
- Do not include ethnicity in the core representation; reserve it for subgroup
  reporting after minimum-count checks.

Static tokens have offset 0 and are always preserved during truncation.

### 7.2 Diagnosis events

Create diagnosis identity from `diagnosisstring`. Normalize text using the same
categorical normalization rule. Use `icd9code` only as optional metadata, not
as the primary identity, because it is missing in a material fraction of rows.

Token format:

```text
DX::<NORMALIZED_DIAGNOSIS>
```

Drop rows without a usable diagnosis string. Remove exact duplicate diagnosis
tokens within the same stay and minute.

### 7.3 Laboratory events

Use `labname` as the variable identity. Parse `labresult` as a finite numeric
value. Rows without a usable name or numeric value are excluded from the MVP
representation and counted in the preprocessing report.

For each normalized lab name with at least 50 numeric observations among
training stays, fit the 25th, 50th, and 75th percentiles. Encode values as:

```text
LAB::<NAME>::Q1
LAB::<NAME>::Q2
LAB::<NAME>::Q3
LAB::<NAME>::Q4
```

Variables below the support threshold map to `LAB::RARE::<bin>` only if a
pooled binning rule is explicitly disabled; the default is to drop them. Do not
label quantiles as clinically LOW, NORMAL, or HIGH.

For duplicate measurements of the same lab within one minute, retain the last
row in stable source order before tokenization.

### 7.4 Medication events

Use a normalized nonempty `drugname` as the medication identity. If `drugname`
is missing, retain the row only when `drughiclseqno` is present, using:

```text
MED::HICL::<drughiclseqno>
```

Otherwise drop the row and report it. Do not use dose, route, frequency, or
order-cancellation state as identity tokens in the MVP.

Remove exact duplicate medication tokens within the same stay and minute.

### 7.5 Infusion events

Use normalized `drugname` as the identity and encode:

```text
INFUSION::<NORMALIZED_DRUG>
```

Do not tokenize rates or amounts in the MVP because units and completeness are
not sufficiently standardized. Drop rows without a usable drug name. Remove
exact duplicates within the same stay and minute.

### 7.6 Treatment events

Use normalized `treatmentstring` and encode:

```text
TREATMENT::<NORMALIZED_TREATMENT>
```

Ignore `activeupondischarge`. Drop unusable treatment strings and remove exact
duplicates within the same stay and minute.

### 7.7 Vital events

Use these numeric variables:

- periodic: `temperature`, `sao2`, `heartrate`, `respiration`;
- aperiodic: `noninvasivesystolic`, `noninvasivediastolic`,
  `noninvasivemean`.

For each stay, variable, and fixed 60-minute bucket starting at ICU admission,
compute the median of finite numeric values. Fit variable-specific Q1-Q4
boundaries from training-bucket medians using the same rules as labs.

Token format:

```text
VITAL::<NAME>::Q1
VITAL::<NAME>::Q2
VITAL::<NAME>::Q3
VITAL::<NAME>::Q4
```

Assign each aggregated token the final minute of its bucket, capped at 1440.
Do not forward-fill missing measurements.

### 7.8 Ordering and time-gap tokens

Sort dynamic events by:

1. event offset;
2. event-family priority: `DX`, `LAB`, `VITAL`, `MED`, `INFUSION`,
   `TREATMENT`;
3. token text.

Insert one time-gap token between consecutive dynamic-event timestamps. Do not
insert gap tokens between events sharing a timestamp.

Use these bins:

```text
TIME_GAP::0_15M
TIME_GAP::16_60M
TIME_GAP::61_180M
TIME_GAP::181_360M
TIME_GAP::GT_360M
```

Static tokens precede all dynamic events and do not affect gap calculation.

### 7.9 Sequence length

Use `max_seq_len: 256`, including static and time-gap tokens.

If a sequence exceeds 256 tokens:

1. Preserve all static tokens.
2. Preserve the first and last dynamic token.
3. Select the remaining dynamic positions by deterministic uniform sampling
   across the ordered stream until the limit is met.
4. Recompute time-gap tokens from the retained dynamic timestamps.
5. If recomputed gaps still exceed the limit, uniformly remove additional
   nonendpoint dynamic tokens and recompute again.

Record original and retained token counts by event family. This policy is
designed to retain temporal coverage instead of keeping only the beginning or
end of the stay.

## 8. Tokenizer and Masked Event Objective

The vocabulary must include:

```text
[PAD]
[UNK]
[MASK]
[CLS]
```

Build the vocabulary from training streams only. Keep tokens with at least five
training occurrences; map all others to `[UNK]`.

For masked event modelling:

- select 15% of maskable tokens;
- do not mask `[PAD]`, `[CLS]`, or static context tokens;
- replace 80% of selected tokens with `[MASK]`;
- replace 10% with a random training-vocabulary token;
- leave 10% unchanged;
- compute cross-entropy loss only at selected positions.

Use a deterministic epoch-aware random seed so masking varies by epoch while
remaining reproducible.

## 9. Model

Model name:

```text
ICU-TinyTransformer
```

Architecture:

```text
token IDs
  -> token embeddings + position embeddings + event-family embeddings
  -> two-layer Transformer encoder
  -> masked-token prediction head for pretraining
  -> [CLS] binary classification head for fine-tuning
```

Default CPU-friendly configuration:

```yaml
model:
  max_seq_len: 256
  d_model: 64
  n_heads: 4
  n_layers: 2
  dim_feedforward: 256
  dropout: 0.1

pretraining:
  mask_probability: 0.15
  batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 0.0005
  weight_decay: 0.01
  epochs: 5

finetuning:
  batch_size: 8
  learning_rate: 0.0003
  weight_decay: 0.01
  epochs: 10
  early_stopping_patience: 3
  selection_metric: validation_average_precision

runtime:
  device: cpu
  num_workers: 0
  seed: 42

data_processing:
  csv_chunk_rows: 50000
  partition_shards: 64
  encoded_shard_stays: 128

recovery:
  enabled: true
  checkpoint_every_optimizer_steps: 100
  log_every_batches: 10
  keep_last_checkpoints: 2
  resume: auto
```

Use class-weighted binary cross-entropy for Transformer fine-tuning, with the
positive-class weight computed from training labels only. Select the binary F1
threshold on validation predictions only. AUROC and average precision use raw
prediction scores and do not use that threshold.

## 10. Experiment Design

### 10.1 Core experiment matrix

Run this reduced matrix:

| ID | Representation | Model | Pretraining | Evaluation | Purpose |
|---|---|---|---|---|---|
| EXP-00 | Full 24-hour representation | Logistic regression | No | Patient-grouped test | Bag-of-events baseline |
| EXP-01 | Full 24-hour representation | Tiny Transformer | No | Patient-grouped test | Scratch neural baseline |
| EXP-02 | Full 24-hour representation | Tiny Transformer | Yes | Patient-grouped test | Primary pretrained model |
| EXP-03 | No static or time-gap tokens | Tiny Transformer | Yes | Patient-grouped test | Representation ablation |
| EXP-04 | Full 24-hour representation | Selected model | As selected | Hospital-grouped CV | Exploratory site shift |
| EXP-05 | Full 24-hour representation | Tiny Transformer | Yes | Hospital-cluster simulation | Optional FedAvg demonstration |

All core model comparisons use the same eligible cohort and primary patient
split. EXP-00 uses token counts with class-weighted logistic regression. EXP-01
and EXP-02 must use identical architecture, fine-tuning settings, and selection
rules so the only intended difference is pretraining.

### 10.2 Model selection

Select between EXP-01, EXP-02, and EXP-03 using validation average precision.
Do not use test metrics for architecture, preprocessing, epoch, or threshold
selection. Evaluate the selected configuration once on the primary test set.

Report all prespecified experiment results, including negative results. Do not
report only the winning configuration.

### 10.3 Optional tuning

Run tuning only after EXP-00 through EXP-03 complete successfully. Limit Optuna
to eight trials and optimize validation average precision.

```yaml
tuning:
  n_trials: 8
  pretrain_epochs_per_trial: 2
  finetune_epochs_per_trial: 5
  d_model: [32, 64, 96]
  n_layers: [1, 2]
  n_heads: [2, 4]
  dim_feedforward: [128, 256]
  dropout: [0.0, 0.1, 0.2]
  learning_rate: [0.0001, 0.0003, 0.0005, 0.001]
  mask_probability: [0.10, 0.15, 0.20]
```

Reject invalid combinations where `d_model` is not divisible by `n_heads`.
Keep preprocessing and data splits fixed across trials. The Optuna SQLite study
is a local ignored artifact and must not be committed.

## 11. Evaluation and Reporting

Primary metrics:

- AUROC;
- average precision.

Secondary metrics:

- F1 at the validation-selected threshold;
- balanced accuracy at the same threshold.

For the final primary test evaluation, compute 95% confidence intervals using
1,000 patient-level bootstrap replicates sampled by `uniquepid` with replacement.
Skip and count replicates containing only one outcome class. Use seed 42.

Every result table must include:

- experiment ID and representation;
- number of patients and ICU stays;
- Alive and Expired counts;
- split strategy and seed;
- AUROC and average precision with confidence intervals for final results;
- F1 and balanced accuracy;
- parameter count and runtime;
- explicit notes on exclusions or failed runs.

Do not fabricate metrics. Use placeholders only in planning documents before
experiments are run.

## 12. Exploratory Hospital-Grouped Evaluation

Use five-fold `StratifiedGroupKFold` with `hospitalid` as the group and seed 42.
Each hospital must appear in exactly one test fold per cross-validation round.
Fit all preprocessing independently inside each training fold; do not reuse
quantiles or vocabularies fitted on held-out hospitals.

Because individual hospitals are small, report fold-level and pooled
out-of-fold metrics, not per-hospital AUROC. If any fold has one outcome class,
report the failure and do not invent an AUROC.

Describe this experiment as exploratory generalisation to held-out de-identified
hospitals within the demo. It does not establish multicentre generalisation for
the full eICU population or new health systems.

Event-density grouping may be retained only as an optional distribution-shift
stress test. It must be called a synthetic cohort analysis, not a site or client
evaluation.

## 13. FedAvg-Style Hospital-Cluster Simulation

EXP-05 is optional and begins only after the centralised experiments are stable.

Construct three clients from hospitals in the training partition:

1. Count eligible training stays per hospital.
2. Sort hospitals by descending stay count, breaking ties by `hospitalid`.
3. Assign each hospital to the client with the smallest current stay count,
   breaking client ties by client number.
4. Keep all stays from one hospital in one client.

Use the validation and test partitions from the primary patient-grouped split
for global evaluation. Never distribute their stays to clients.

```yaml
fedavg:
  clients: 3
  rounds: 3
  local_epochs: 1
  client_fraction: 1.0
  aggregation: weighted_average_by_num_training_stays
```

Initialize every client from the same pretrained encoder and classification
head. Aggregate all trainable parameters after each round. Compare the final
global model with the centralised EXP-02 configuration using the same test set.

This is a single-process simulation. It does not implement secure aggregation,
differential privacy, network communication, client authentication, or threat
modelling.

## 14. Bounded-Memory Execution, Logging, and Recovery

### 14.1 Restartable preprocessing

Preprocessing is a staged, bounded-memory pipeline:

```text
discover_inputs
  -> build_cohort_and_splits
  -> extract_partitioned_events
  -> fit_training_preprocessing
  -> assemble_stream_shards
  -> fit_vocabulary
  -> encode_split_shards
```

Each stage writes to an ignored working directory and owns a `manifest.json`
containing:

- stage name and schema version;
- status: `running`, `complete`, or `failed`;
- normalized config hash;
- input file fingerprints using relative path, size, modification time, and
  distributed SHA256 value when available;
- upstream manifest hashes;
- shard count and completed shard IDs;
- aggregate row/stay counts and skipped-record counts;
- start, update, and completion timestamps;
- failure type and message without patient-level values.

Chunk and shard outputs are written to `*.tmp`, flushed, and atomically renamed
only after validation. A shard is complete only after its checksum and aggregate
counts are recorded in the manifest. On restart, the pipeline reuses completed
shards and resumes at the first incomplete shard. If config, inputs, schema, or
upstream hashes differ, automatic resume must stop with an incompatibility error
instead of mixing artifacts from different runs. `--restart-stage <name>` may
explicitly discard that stage and all downstream local outputs.

Training-only numeric quantiles must also remain bounded-memory. Write normalized
training values into disk-backed per-variable runs, sort fixed-size runs, and
use an external k-way merge to select exact quartile ranks. Do not accumulate all
lab or vital values in a Python list or DataFrame. Delete temporary numeric runs
only after fitted preprocessing metadata is atomically complete.

Vocabulary counting and encoded dataset creation must stream completed event
shards. Encoded data is written as multiple shards of at most 128 stays plus an
index, not one monolithic `.pt` object.

### 14.2 Run logs

Every preprocessing, training, tuning, grouped-evaluation, and FedAvg run gets a
local `run_id` and ignored run directory:

```text
results/runs/<run_id>/run.log
results/runs/<run_id>/events.jsonl
results/runs/<run_id>/state.json
results/runs/<run_id>/checkpoints/
```

`run.log` is human-readable and mirrors console progress. `events.jsonl` is
append-only, one JSON object per line, flushed after every event. Both logs must
include timestamps, run/stage name, status transitions, epoch, batch, optimizer
step, learning rate, loss, validation metrics, elapsed time, checkpoint path,
and aggregate throughput where applicable. They must never include patient IDs,
stay IDs, raw tokens, event streams, secrets, or raw rows.

`state.json` is an atomically replaced summary pointing to the last valid
checkpoint and recording `running`, `completed`, `failed`, or `interrupted`.
Uncaught exceptions are logged with exception type and traceback. SIGINT and
SIGTERM are handled as controlled interruption requests: stop after the current
batch reaches a consistent state, save an emergency checkpoint, flush logs, and
mark the run interrupted.

### 14.3 Resumable training

Save a full checkpoint every 100 completed optimizer steps by default, at every
epoch boundary, whenever validation produces a new best model, and on controlled
interruption. A checkpoint contains:

- format/schema version and run ID;
- model and prediction-head state;
- optimizer and scheduler state;
- current parameter gradients and gradient-accumulation micro-step;
- epoch, next batch cursor, global batch, and optimizer step;
- best metric, best epoch, early-stopping state, and selected threshold state;
- Python, NumPy, and PyTorch RNG states;
- sampler permutation/generator state and cursor;
- config, vocabulary, split, preprocessing, and encoded-dataset hashes;
- training-history summary and checkpoint creation reason.

With `num_workers: 0`, saved sampler state, epoch-aware masking, and restored RNG
state, resume must continue from the next unprocessed batch rather than restart
the epoch. If an unexpected process or machine failure occurs, resume from the
last completed periodic checkpoint; at most the configured checkpoint interval
is repeated. Checkpoint loading must fail on incompatible config or artifact
hashes unless an explicit weights-only initialization mode is requested for a
new run.

Keep the newest two periodic checkpoints plus the best and final checkpoints.
Delete older periodic checkpoints only after the replacement checkpoint is
validated. Checkpoints and detailed run logs are local ignored artifacts.

The same run-state contract applies to fine-tuning, each Optuna trial, each
hospital-grouped fold, each FedAvg client update, and each FedAvg round. Parent
runs record child run IDs and resume only incomplete children.

## 15. Optional Memorisation Diagnostic

After selecting the final pretrained model, an optional diagnostic may measure:

- overlap of rare training token n-grams with generated top-k masked-token
  predictions;
- masked prediction accuracy stratified by token frequency.

The diagnostic must use aggregate counts only and must not emit patient-level
streams or identifiers. It is not a privacy audit, membership-inference test,
or privacy guarantee.

## 16. Reproducible Artifacts

Processed local outputs:

```text
data/processed/eicu_demo/cohort_summary.json
data/processed/eicu_demo/split_metadata.json
data/processed/eicu_demo/preprocessing_metadata.json
data/processed/eicu_demo/event_stats.json
data/processed/eicu_demo/vocab.json
data/processed/eicu_demo/manifests/<stage>/manifest.json
data/processed/eicu_demo/event_shards/part-*.jsonl.gz
data/processed/eicu_demo/event_shards/index.json
data/processed/eicu_demo/encoded/train/part-*.pt
data/processed/eicu_demo/encoded/validation/part-*.pt
data/processed/eicu_demo/encoded/test/part-*.pt
data/processed/eicu_demo/encoded/index.json
```

These outputs are ignored and must not be committed because they derive from
patient-level data.

Trackable aggregate outputs may include:

```text
results/summary/cohort_summary.json
results/summary/experiment_comparison.csv
results/summary/final_results.csv
results/summary/best_config.json
results/figures/model_comparison.png
results/figures/hospital_grouped_performance.png
results/figures/fedavg_vs_centralised.png
```

Trackable summaries must contain aggregate results only. Suppress or combine
any subgroup cell with fewer than 10 stays.

## 17. Repository Structure

Keep the planned package boundaries focused on the experiment:

```text
configs/                 # final, experiment, tuning, and FedAvg configs
data/                    # local raw and processed data; ignored
docs/                    # protocol, data statement, limitations, model card
results/                 # curated aggregate summaries and figures
  runs/                  # ignored detailed logs, state, and checkpoints
scripts/                 # preparation, training, evaluation, reporting CLIs
src/icu_pretrain/
  data/                  # loading, cohort, events, tokenization, splits
  models/                # Transformer and prediction heads
  training/              # pretraining, fine-tuning, baselines, FedAvg
  experiments/           # run registry and orchestration
  tuning/                # constrained Optuna study
  analysis/              # aggregate plots, tables, diagnostics
tests/                   # synthetic-data and unit tests only
```

Do not add a full eICU or MIMIC dependency, remote experiment tracking, cloud
logging, telemetry, or heavyweight model architecture.

## 18. Implementation Order

### Phase 1: Cohort and data contract

- Resolve the versioned raw directory and exact filenames.
- Add chunk iterators, deterministic stay partitioning, stage manifests, and
  restart compatibility checks.
- Parse `patient.csv.gz`, apply eligibility rules, and create the patient-grouped
  split.
- Produce aggregate cohort and exclusion summaries.
- Add synthetic tests for label parsing, eligibility, and split isolation.

Acceptance criteria:

- no patient overlaps partitions;
- every retained stay has one label and one partition;
- both classes occur in all primary partitions;
- no patient-level artifact is written outside ignored processed directories.

### Phase 2: Event builder and tokenizer

- Implement table-specific event extraction and 24-hour filtering.
- Add 60-minute vital aggregation and training-only quantile fitting.
- Implement deterministic ordering, time-gap tokens, and sequence sampling.
- Build a training-only vocabulary and encode all partitions.
- Stream every stage through restartable shards; never materialize a complete
  large event table or encoded split in memory.

Acceptance criteria:

- no retained dynamic event has an offset outside 0-1440;
- prohibited leakage fields cannot become tokens;
- validation/test values do not affect fitted preprocessing;
- repeated runs with seed 42 produce identical streams and encodings.

### Phase 3: Core models

- Implement logistic regression, scratch Transformer, masked pretraining, and
  fine-tuning.
- Run EXP-00 through EXP-03.
- Add append-only JSONL/text logs, full-state checkpoints, controlled signal
  handling, and exact mid-epoch resume before long runs.
- Select configuration and threshold from validation data only.

Acceptance criteria:

- EXP-01 and EXP-02 differ only by pretrained initialization;
- masked loss is computed only at selected positions;
- test results are generated only after selection is frozen.

### Phase 4: Grouped evaluation and optional extensions

- Run five-fold hospital-grouped evaluation for EXP-04.
- Run the eight-trial tuning study only if needed.
- Run FedAvg and memorisation diagnostics only after core results are complete.
- Generate aggregate result tables, figures, limitations, and model card.

Acceptance criteria:

- hospitals never cross grouped-evaluation folds;
- every FedAvg hospital belongs to one client;
- all reports distinguish patient-grouped, hospital-grouped, and simulated
  client results;
- all public claims remain limited to the eICU demo.

## 19. Final Application Framing

Use this framing in the README and application materials:

```text
This repository is a CPU-friendly technical prototype built on the eICU demo
v2.0.1 dataset. It constructs leakage-controlled first-24-hour ICU stay event
streams, pretrains a compact Transformer with masked event modelling, and
evaluates hospital-mortality prediction against scratch and bag-of-events
baselines. The demo includes de-identified hospital and ward identifiers, which
support exploratory grouped evaluation and a small FedAvg-style simulation over
hospital clusters. Because the dataset is small and selectively sampled, the
project does not claim clinical readiness or generalisation to new hospitals.
```

The final repository should emphasize reproducible event construction,
leakage-aware evaluation, controlled comparisons, and honest limitations rather
than headline performance.
