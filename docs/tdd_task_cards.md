# TDD Task Cards for `icu-event-pretraining-mini`

Source of truth: `docs/plan.md`.

These cards describe the required behavior after the dataset-vs-plan audit.
Existing code and tests may be reused, but no card is complete until the revised
tests and acceptance criteria pass. All tests must use synthetic data and must
not inspect `data/raw/` or `data/processed/`.

## Implementation Status

Last updated: 2026-06-07

| Task card | Status |
|---|---|
| TDD-00 - Test Harness and Package Safety | Implemented |
| TDD-01 - Dataset-Grounded Configuration Contracts | Remaining |
| TDD-02 - Cohort, Outcome, Event, and Artifact Contracts | Remaining |
| TDD-03 - Versioned eICU Demo Table Discovery and Loading | Remaining |
| TDD-04 - Primary Outcome and 24-Hour Cohort Eligibility | Remaining |
| TDD-05 - Patient-Grouped Train/Validation/Test Splits | Remaining |
| TDD-06 - Static and Categorical Event Extraction | Remaining |
| TDD-07 - Training-Only Numeric Fitting and Vital Aggregation | Remaining |
| TDD-08 - Leakage-Controlled Event Stream Assembly | Remaining |
| TDD-09 - eICU Preparation CLI and Local Artifacts | Remaining |
| TDD-10 - Training-Only Tokenizer and Vocabulary | Remaining |
| TDD-11 - Encoded Dataset Creation | Remaining |
| TDD-12 - Padding, Attention Masks, and MLM Corruption | Remaining |
| TDD-13 - ICU-TinyTransformer Encoder | Remaining |
| TDD-14 - Masked Event and Mortality Heads | Remaining |
| TDD-15 - Masked Event Pretraining Workflow | Remaining |
| TDD-16 - Metrics, Threshold Selection, and Patient Bootstrap | Remaining |
| TDD-17 - Bag-of-Events Logistic Baseline | Remaining |
| TDD-18 - Scratch and Pretrained Fine-Tuning | Remaining |
| TDD-19 - Run Logging, Recovery State, and Aggregate Tracking | Remaining |
| TDD-20 - Core Experiment Registry and Runner | Remaining |
| TDD-21 - Optional Eight-Trial Optuna Search | Remaining |
| TDD-22 - Hospital-Grouped Evaluation and FedAvg Simulation | Remaining |
| TDD-23 - Optional Memorisation Diagnostic | Remaining |
| TDD-24 - Result Assets, Documentation, and Application Polish | Remaining |

Progress: **1 implemented, 24 remaining.**

Resetting a card to Remaining does not require deleting reusable code. It means
the implementation must be reviewed and changed until it satisfies this card.

## Dependency Map

Primary chain:

```text
TDD-00 -> TDD-01 -> TDD-02 -> TDD-03 -> TDD-04 -> TDD-05
       -> TDD-06 -> TDD-07 -> TDD-08 -> TDD-09 -> TDD-10
       -> TDD-11 -> TDD-12 -> TDD-13 -> TDD-14 -> TDD-15
```

Downstream dependencies:

```text
TDD-12 -> TDD-16 -> TDD-17
TDD-15 + TDD-16 -> TDD-18
TDD-01 -> TDD-19
TDD-17 + TDD-18 + TDD-19 -> TDD-20
TDD-20 -> TDD-21
TDD-05 + TDD-16 + TDD-18 -> TDD-22
TDD-15 + TDD-18 -> TDD-23
TDD-19 + TDD-20 + TDD-22 -> TDD-24
```

`TDD-21` and `TDD-23` are optional and do not block the central result.
`TDD-24` must handle absent optional outputs.

## Global Batching and Recovery Contract

All cards must preserve bounded memory and resumability:

- Read compressed raw tables in configurable chunks, default 50,000 rows.
- Partition normalized events into 64 deterministic stay shards using SHA256 of
  `patientunitstayid`; process one shard at a time.
- Write temporary outputs atomically and track completed stages/shards in
  manifests with input, config, and upstream hashes.
- Resume only compatible incomplete work; never combine stale and current
  artifacts.
- Encode at most 128 stays per model-data shard and load shards lazily.
- Write append-only `events.jsonl` plus human-readable `run.log`, without any
  patient-level content.
- Save complete training state every 100 optimizer steps, at epoch boundaries,
  on new best models, and on controlled interruption.
- Restore model, optimizer, scheduler, gradients, accumulation state, epoch,
  batch cursor, RNG, sampler, early stopping, and artifact hashes.
- Keep the newest two periodic checkpoints plus best and final checkpoints.

---

## TDD-00 - Test Harness and Package Safety

**Status:** Implemented.

**Goal:** Provide a synthetic-only, CPU-only test harness for all later cards.

**Source plan sections:** 1, 2, 17, and 18.

**Files likely to change:** `pyproject.toml`, `tests/conftest.py`,
`tests/test_package_sanity.py`.

**Dependencies:** None.

**Test-first requirements:**

- Verify package and planned submodule imports without raw data, processed data,
  GPU, Optuna, or generated artifacts.
- Verify YAML loading accepts mappings, treats an empty file as `{}`, and
  rejects non-mappings.
- Provide reusable synthetic patient and event-table fixtures containing no
  real identifiers or values.

**Implementation steps:**

1. Configure test discovery and a documented test dependency.
2. Add import and utility sanity tests.
3. Add small synthetic table factories under `tests/` only.

**Acceptance criteria:** Tests can be collected and run locally without reading
protected data or requiring network access.

**Edge cases:** Empty YAML, malformed YAML, missing optional dependency, import
from a clean process.

**Definition of done:** Later cards can be implemented test-first using only
synthetic fixtures.

---

## TDD-01 - Dataset-Grounded Configuration Contracts

**Status:** Remaining.

**Goal:** Validate configuration against the revised 24-hour protocol and
reduced experiment matrix.

**Source plan sections:** 4-13.

**Files likely to change:** `src/icu_pretrain/utils.py`, `configs/`,
`tests/test_config_validation.py`.

**Dependencies:** `TDD-00`.

**Test-first requirements:**

- Accept observation window `0..1440`, `max_seq_len: 256`, seed 42, CPU runtime,
  70/15/15 patient-grouped splits, 60-minute vital buckets, and minimum five
  retained dynamic events.
- Accept experiments `EXP-00` through `EXP-05`, with EXP-05 optional.
- Accept eight-trial maximum tuning and three-client, three-round FedAvg.
- Accept configurable defaults for 50,000-row CSV chunks, 64 processing shards,
  128 stays per encoded shard, logs every 10 batches, checkpoints every 100
  optimizer steps, two retained periodic checkpoints, and `resume: auto`.
- Reject APACHE/discharge fields as configured features, stay-random splitting,
  invalid ratios, unsupported metrics, invalid devices, and incompatible
  `d_model`/`n_heads`.

**Implementation steps:**

1. Define normalized data, split, preprocessing, model, training, evaluation,
   tuning, and FedAvg config contracts.
2. Update scaffolded YAML files to the revised defaults and experiment IDs.
3. Return explicit key-oriented validation errors without mutating inputs.

**Acceptance criteria:** Every active config agrees with `docs/plan.md`; stale
128-token, pseudo-site, and unrestricted mortality settings are rejected.

**Edge cases:** Ratios not summing to one, end before start, nonpositive bucket
size, more than eight trials, unknown experiment ID.

**Definition of done:** All CLIs and runners can rely on one dataset-grounded
configuration layer.

---

## TDD-02 - Cohort, Outcome, Event, and Artifact Contracts

**Status:** Remaining.

**Goal:** Define stable contracts for ICU stay records, eligibility decisions,
splits, fitted preprocessing, event streams, stage manifests, run state,
checkpoints, outcomes, and aggregate summaries.

**Source plan sections:** 4, 6, 7, and 14.

**Files likely to change:** `src/icu_pretrain/data/eicu_event_builder.py`,
`src/icu_pretrain/constants.py`, `tests/test_event_contracts.py`.

**Dependencies:** `TDD-01`.

**Test-first requirements:**

- An event stream contains `patientunitstayid`, ordered tokens and timestamps,
  representation metadata, split name, and no raw patient/site identifier in
  serialized model input.
- An outcome is exactly one hospital-mortality label per eligible stay.
- Cohort summaries include rule-level exclusions and class counts.
- Split metadata supports `uniquepid` and `hospitalid` internally but is written
  only under ignored processed paths.
- Aggregate public summaries reject patient IDs and event dumps.
- Stage manifests record compatibility hashes, completed shards, aggregate
  counts, timestamps, and failure metadata.
- Checkpoint contracts include model/training/RNG/sampler/cursor state and reject
  incompatible artifact hashes.

**Implementation steps:** Define typed records, validators, JSON/JSONL schemas,
and safe serialization helpers.

**Acceptance criteria:** Synthetic records round-trip; invalid IDs, labels,
timestamps, families, split names, or patient-level public fields fail clearly.

**Edge cases:** Duplicate timestamps, empty dynamic events, missing label,
unknown family, non-finite timestamp.

**Definition of done:** All later data and experiment stages share explicit,
leakage-aware contracts.

---

## TDD-03 - Versioned eICU Demo Table Discovery and Loading

**Status:** Remaining.

**Goal:** Discover and load the exact required `.csv.gz` tables safely.

**Source plan section:** 5.1.

**Files likely to change:** `src/icu_pretrain/data/eicu_demo.py`,
`tests/test_eicu_demo_loader.py`.

**Dependencies:** `TDD-02`.

**Test-first requirements:**

- Accept either the exact v2.0.1 directory or a root containing exactly one
  `eicu-crd-demo/2.0.1` directory.
- Load `patient`, `diagnosis`, `lab`, `medication`, `infusiondrug`, `treatment`,
  `vitalPeriodic`, and `vitalAperiodic` from exact case-sensitive filenames.
- Read compressed CSVs and only requested columns when possible.
- Yield DataFrame chunks with a configurable default of 50,000 rows; no large
  event table loader may concatenate every chunk into one DataFrame.
- Validate each table's required ID, value, and offset columns.
- Do not require `apachePatientResult`.

**Implementation steps:** Add deterministic discovery, logical-name-to-file
mapping, required-column schemas, and reusable chunk iterators. Keep a bounded
full-table helper only for explicitly small tables such as `patient`.

**Acceptance criteria:** Missing, ambiguous, mis-cased, empty, or malformed
tables produce actionable errors; extra columns are allowed.

**Edge cases:** Direct versus parent root, two version matches, empty optional
event table, duplicate event rows.

**Definition of done:** Downstream code consumes validated DataFrames without
assuming uncompressed files or a flat raw directory.

---

## TDD-04 - Primary Outcome and 24-Hour Cohort Eligibility

**Status:** Remaining.

**Goal:** Produce the fixed hospital-mortality label and eligible 24-hour cohort.

**Source plan section:** 4.

**Files likely to change:** `src/icu_pretrain/data/eicu_event_builder.py`,
`tests/test_cohort_selection.py`.

**Dependencies:** `TDD-03`.

**Test-first requirements:**

- Map case-insensitive Alive to 0 and Expired to 1 from
  `patient.hospitaldischargestatus` only.
- Exclude missing/unrecognised labels, missing required IDs, and
  `unitdischargeoffset < 1440`.
- Reject duplicate or conflicting patient rows for one stay.
- Do not fall back to unit mortality or any APACHE field.
- Defer the five-retained-event criterion until normalized events are available,
  while retaining a provisional cohort record.

**Implementation steps:** Implement label parsing, fixed eligibility rules,
rule-level exclusion reasons, and aggregate cohort summaries.

**Acceptance criteria:** Every provisional stay has one valid label,
`uniquepid`, `hospitalid`, and complete 24-hour ICU exposure.

**Edge cases:** Numeric/string offsets, exactly 1440 minutes, mixed-case labels,
duplicate identical rows, unknown status.

**Definition of done:** The primary target and non-event eligibility cohort are
fixed without outcome fallback or leakage.

---

## TDD-05 - Patient-Grouped Train/Validation/Test Splits

**Status:** Remaining.

**Goal:** Create reproducible 70/15/15 partitions grouped by patient before any
data-dependent preprocessing is fitted.

**Source plan section:** 6.

**Files likely to change:** `src/icu_pretrain/data/splits.py`,
`tests/test_splits.py`.

**Dependencies:** `TDD-04`.

**Test-first requirements:**

- Group all stays for one `uniquepid` into one partition.
- Approximate 70/15/15 stay counts while stratifying mortality as closely as
  possible with seed 42.
- Require both outcome classes in every partition or fail with a data-size
  explanation.
- Verify complete, exclusive assignment of eligible stays.
- Persist sensitive split assignments only to an explicitly supplied ignored
  processed-data path.

**Implementation steps:** Implement deterministic stratified group assignment,
split validation, and aggregate split summaries.

**Acceptance criteria:** No patient or stay crosses partitions; repeated runs
with the same input and seed are identical.

**Edge cases:** One-class cohort, one patient with many stays, tiny strata,
duplicate stay IDs, impossible class distribution.

**Definition of done:** All fitted preprocessing and modelling can operate from
an immutable patient-isolated split.

---

## TDD-06 - Static and Categorical Event Extraction

**Status:** Remaining.

**Goal:** Extract deterministic static, diagnosis, medication, infusion, and
treatment events within the primary time window.

**Source plan sections:** 7.1, 7.2, and 7.4-7.6.

**Files likely to change:** `src/icu_pretrain/data/eicu_event_builder.py`,
`tests/test_event_builder_categorical.py`.

**Dependencies:** `TDD-05`.

**Test-first requirements:**

- Emit age bins `0_17`, `18_39`, `40_59`, `60_79`, `80_PLUS`; map `> 89` to
  `80_PLUS`.
- Emit normalized gender, `unitadmitsource`, and `unittype`; use `UNKNOWN` for
  missing/unrecognised static values.
- Never emit ethnicity, IDs, site tokens, discharge fields, APACHE fields, or
  `activeupondischarge`.
- Use diagnosis string identity; drug name with HICL fallback for medication;
  drug name for infusion; treatment string for treatment.
- Retain only offsets from 0 through 1440 and deduplicate identical tokens in
  the same stay and minute.
- Consume row chunks and append normalized events to deterministic temporary
  stay shards; never retain all source rows in memory.

**Implementation steps:** Implement text normalization, static extraction,
table-specific categorical extraction, leakage deny-list checks, and coverage
counters.

**Acceptance criteria:** Synthetic output follows exact token formats and is
deterministic without exposing prohibited fields.

**Edge cases:** Symbols/whitespace, missing offset, negative offset, minute
1440, missing drug name with/without HICL, duplicate rows.

**Definition of done:** All nonnumeric MVP events are available under the fixed
24-hour and leakage rules.

---

## TDD-07 - Training-Only Numeric Fitting and Vital Aggregation

**Status:** Remaining.

**Goal:** Fit lab and vital discretization from training stays only and apply it
without validation/test leakage.

**Source plan sections:** 6.2, 7.3, and 7.7.

**Files likely to change:** `src/icu_pretrain/data/eicu_event_builder.py`,
`tests/test_numeric_discretization.py`.

**Dependencies:** `TDD-06`.

**Test-first requirements:**

- Parse finite numeric labs and the seven specified vital variables only.
- Aggregate vital values by stay, variable, and 60-minute bucket using median.
- Fit per-variable 25th/50th/75th percentiles from training data only when at
  least 50 training observations exist.
- Drop unsupported variables by default; do not pool them into `RARE` bins.
- Encode Q1-Q4 and handle repeated/equal thresholds deterministically.
- Prove that changing validation/test values cannot change fitted metadata.
- Write training numeric values to bounded sorted disk runs and use an external
  k-way merge to select exact quartile ranks without collecting all values in
  memory.

**Implementation steps:** Separate fit and transform APIs, add stable same-minute
lab selection, vital aggregation, metadata serialization, and skip counters.

**Acceptance criteria:** No raw numeric values become tokens; transform never
updates fitted thresholds; all emitted offsets remain in 0..1440.

**Edge cases:** All-equal values, exactly 50 observations, invalid strings,
infinities, unseen variable, empty bucket, boundary equality.

**Definition of done:** Numeric tokenization is reproducible and isolated to
training-derived statistics.

---

## TDD-08 - Leakage-Controlled Event Stream Assembly

**Status:** Remaining.

**Goal:** Merge extracted events into final first-24-hour ICU stay streams and
apply deterministic length control.

**Source plan sections:** 7.8 and 7.9.

**Files likely to change:** `src/icu_pretrain/data/eicu_event_builder.py`,
`tests/test_event_stream_builder.py`.

**Dependencies:** `TDD-07`.

**Test-first requirements:**

- Sort by offset, family priority `DX/LAB/VITAL/MED/INFUSION/TREATMENT`, then
  token text; prepend static tokens.
- Insert the five defined time-gap bins only between distinct dynamic times.
- Exclude stays with fewer than five retained dynamic events.
- Cap at 256 total tokens while preserving static tokens and first/last dynamic
  events and uniformly sampling temporal positions.
- Recompute gaps after sampling and guarantee deterministic output.
- Record original/retained counts by family and final exclusion counts.
- Reduce one deterministic stay shard at a time and checkpoint completed shard
  IDs in an atomic stage manifest.

**Implementation steps:** Add merge/order logic, gap calculation, iterative
uniform sampling, final eligibility filter, and aggregate statistics.

**Acceptance criteria:** No post-1440, negative, or prohibited event survives;
every stream validates and is at most 256 tokens.

**Edge cases:** Same-minute events, no gap, one dynamic timestamp, gap over 360
minutes, sampling dominated by gap tokens, exactly five events.

**Definition of done:** The canonical full and no-static/no-gap representations
are ready for tokenization.

---

## TDD-09 - eICU Preparation CLI and Local Artifacts

**Status:** Remaining.

**Goal:** Orchestrate cohort selection, splitting, fitted preprocessing, and
event construction into ignored local artifacts.

**Source plan sections:** 14, 16, and 18 phases 1-2.

**Files likely to change:** `scripts/prepare_eicu_demo.py`,
`tests/test_prepare_eicu_demo_cli.py`.

**Dependencies:** `TDD-08`.

**Test-first requirements:**

- Run end to end on compressed synthetic tables.
- Accept raw root, processed output directory, config, and seed.
- Write cohort summary, sensitive split metadata, preprocessing metadata, event
  stats, and event streams beneath the supplied output directory.
- Fail before partial publication on invalid input; use atomic final writes.
- Print aggregate counts only, never patient/stay IDs or event streams.
- Support `--resume auto`, `--no-resume`, and `--restart-stage`; compatible runs
  skip completed shards and continue at the first incomplete shard.
- Produce append-only JSONL and text logs with stage/chunk/shard progress and an
  atomic `state.json`.

**Implementation steps:** Replace old representation CLI flow with config-driven
pipeline orchestration, stage manifests, run directories, atomic shard writing,
and safe restart handling.

**Acceptance criteria:** Artifacts are deterministic, remain under the output
directory, and reflect split-before-fit ordering.

**Edge cases:** Existing artifacts, missing raw root, ambiguous dataset version,
empty final cohort, unwritable output path.

**Definition of done:** One command reproducibly prepares leakage-controlled
local data without exposing patient-level content.

---

## TDD-10 - Training-Only Tokenizer and Vocabulary

**Status:** Remaining.

**Goal:** Fit and persist a stable event vocabulary from training streams only.

**Source plan section:** 8.

**Files likely to change:** `src/icu_pretrain/data/tokenizer.py`,
`tests/test_tokenizer.py`.

**Dependencies:** `TDD-09`.

**Test-first requirements:** Reserve stable IDs for `[PAD]`, `[UNK]`, `[MASK]`,
and `[CLS]`; retain training tokens with frequency at least five; map all unseen
tokens to `[UNK]`; prove validation/test tokens do not affect vocabulary.

Count tokens by streaming completed training event shards and checkpoint count
shards in the vocabulary-stage manifest; do not load all streams together.

**Implementation steps:** Implement fit, encode, decode, save, load, frequency
metadata, and deterministic token ordering.

**Acceptance criteria:** Save/load preserves IDs and metadata; fitting empty or
non-training input fails clearly.

**Edge cases:** Token equal to a special token, frequency exactly five, corrupt
vocabulary, duplicate token definitions.

**Definition of done:** Every partition can be encoded with one immutable
training-derived vocabulary.

---

## TDD-11 - Encoded Dataset Creation

**Status:** Remaining.

**Goal:** Encode split-specific streams and labels into restartable local shards
without losing provenance.

**Source plan section:** 16.

**Files likely to change:** `src/icu_pretrain/data/dataset.py`,
`tests/test_dataset.py`.

**Dependencies:** `TDD-10`.

**Test-first requirements:** Add `[CLS]`, encode tokens, preserve label and split,
enforce 256-token maximum, and reject overlaps or unlabeled stays. Patient/site
grouping metadata may remain in ignored metadata but must not enter tensors.
Write at most 128 stays per encoded shard plus an atomic index and per-shard
checksum; resume at the first incomplete shard.

**Implementation steps:** Implement dataset records, serialization for
train/validation/test tensor shards, lazy shard iteration, and aggregate
unknown-token reporting.

**Acceptance criteria:** Encoded artifacts align one-to-one with final streams
and use only tokenizer IDs.

**Edge cases:** Empty partition, all unknown tokens, missing outcome, duplicate
stay, corrupt encoded file.

**Definition of done:** Stable encoded datasets exist for all central workflows.

---

## TDD-12 - Padding, Attention Masks, and MLM Corruption

**Status:** Remaining.

**Goal:** Build CPU-friendly batches and reproducible masked-event targets.

**Source plan section:** 8.

**Files likely to change:** `src/icu_pretrain/data/collate.py`,
`tests/test_collate.py`.

**Dependencies:** `TDD-11`.

**Test-first requirements:** Pad batches, create attention masks, select 15% of
maskable tokens, exclude `[PAD]`, `[CLS]`, and static tokens, and implement the
80/10/10 replacement rule with `-100` labels elsewhere.

**Implementation steps:** Add supervised and MLM collators with deterministic
epoch-aware seeding and shape validation. Data loaders must lazily read encoded
shards and use a resumable deterministic sampler with an explicit permutation
and cursor; the default remains `num_workers: 0`.

**Acceptance criteria:** Same seed/epoch reproduces masks, different epochs can
change masks, and loss targets exist only at selected positions.

**Edge cases:** No maskable token, batch size one, all-static sequence, random
replacement selecting a special token, maximum-length sequence.

**Definition of done:** Encoded data can be batched for pretraining and
fine-tuning without leakage or nondeterministic tests.

---

## TDD-13 - ICU-TinyTransformer Encoder

**Status:** Remaining.

**Goal:** Implement the compact shared encoder for all neural experiments.

**Source plan section:** 9.

**Files likely to change:** `src/icu_pretrain/models/transformer.py`,
`tests/test_transformer.py`.

**Dependencies:** `TDD-12`.

**Test-first requirements:** Verify token, position, and event-family embeddings;
two-layer default encoder; padding-mask behavior; `[CLS]` output; maximum length
256; CPU forward pass; and parameter-count reporting.

**Implementation steps:** Build `nn.TransformerEncoder`, validate dimensions,
and return sequence plus pooled `[CLS]` embeddings.

**Acceptance criteria:** Defaults are `d_model=64`, four heads, two layers,
feedforward 256, dropout 0.1; invalid shapes fail clearly.

**Edge cases:** Length overflow, invalid family IDs, empty batch, incompatible
heads, deterministic evaluation mode.

**Definition of done:** One encoder supports scratch, pretrained, and FedAvg
workflows.

---

## TDD-14 - Masked Event and Mortality Heads

**Status:** Remaining.

**Goal:** Implement masked-token and binary hospital-mortality heads.

**Source plan section:** 9.

**Files likely to change:** `src/icu_pretrain/models/heads.py`,
`tests/test_heads.py`.

**Dependencies:** `TDD-13`.

**Test-first requirements:** Check MLM logits `[batch, seq, vocab]`, mortality
logits `[batch]`, `[CLS]` pooling, ignored MLM labels, and weighted binary loss.

**Implementation steps:** Add heads and loss helpers with shape/device checks.

**Acceptance criteria:** Both heads integrate with the encoder and support batch
size one without implicit probability conversion.

**Edge cases:** All MLM labels ignored, all-one mini-batch, invalid positive
weight, vocabulary mismatch.

**Definition of done:** Model outputs support the two fixed learning objectives.

---

## TDD-15 - Masked Event Pretraining Workflow

**Status:** Remaining.

**Goal:** Train and validate masked event modelling on training-derived data.

**Source plan sections:** 8, 9, and 10.

**Files likely to change:** `src/icu_pretrain/training/pretrain.py`,
`scripts/pretrain.py`, `tests/test_pretraining_loop.py`.

**Dependencies:** `TDD-14`.

**Test-first requirements:** Run a tiny CPU epoch, apply gradient accumulation,
compute loss only at selected positions, validate without updating weights, and
write local checkpoint/aggregate metrics without patient data.

Interrupt a synthetic run mid-epoch, restore its full checkpoint, and verify
that resumed model/optimizer state, next batch, masking sequence, and final
metrics match an uninterrupted run.

**Implementation steps:** Load validated config/artifacts, construct model and
collators, create JSONL/text logs, save full-state periodic/best/final/emergency
checkpoints, handle SIGINT/SIGTERM, and resume compatible runs automatically.

**Acceptance criteria:** Default five-epoch configuration is reproducible;
missing data, no masked positions, NaN loss, and incompatible checkpoints fail
clearly.

**Edge cases:** Zero batches, existing checkpoint, disabled pretraining,
interrupted epoch.

**Definition of done:** A reusable pretrained encoder can be produced locally.

---

## TDD-16 - Metrics, Threshold Selection, and Patient Bootstrap

**Status:** Remaining.

**Goal:** Centralize leakage-safe binary evaluation and uncertainty reporting.

**Source plan section:** 11.

**Files likely to change:** `src/icu_pretrain/training/evaluate.py`,
`tests/test_evaluate_metrics.py`.

**Dependencies:** `TDD-12`.

**Test-first requirements:** Compute AUROC, average precision, F1, and balanced
accuracy; select the F1 threshold on validation scores only; apply it unchanged
to test scores; compute 1,000 `uniquepid` bootstrap replicates with seed 42;
skip/count one-class replicates.

**Implementation steps:** Separate score metrics, threshold selection,
thresholded metrics, and grouped bootstrap APIs.

**Acceptance criteria:** Test labels cannot influence threshold selection;
undefined metrics return explicit metadata rather than fabricated values.

**Edge cases:** Empty arrays, NaNs, mismatched lengths, tied thresholds,
single-class labels, repeated stays per patient.

**Definition of done:** Every central, grouped, and FedAvg experiment shares the
same evaluation contract.

---

## TDD-17 - Bag-of-Events Logistic Baseline

**Status:** Remaining.

**Goal:** Implement EXP-00 on the full 24-hour token representation.

**Source plan section:** 10.1.

**Files likely to change:** `src/icu_pretrain/training/baselines.py`,
`scripts/run_baselines.py`, `tests/test_baselines.py`.

**Dependencies:** `TDD-16`.

**Test-first requirements:** Build training-vocabulary token counts, train
class-weighted logistic regression with seed 42, select threshold on validation,
and evaluate the untouched test partition.

**Implementation steps:** Add sparse vectorization, model fitting, shared metric
calls, and aggregate result output.

**Acceptance criteria:** No validation/test vocabulary fitting occurs; failures
from one-class or empty data are explicit.

**Edge cases:** All-zero row, vocabulary larger than sample count, unknown test
tokens, severe imbalance.

**Definition of done:** EXP-00 provides a reproducible non-neural comparator.

---

## TDD-18 - Scratch and Pretrained Fine-Tuning

**Status:** Remaining.

**Goal:** Implement EXP-01 through EXP-03 with controlled initialization and
validation-only selection.

**Source plan sections:** 9, 10.1, and 10.2.

**Files likely to change:** `src/icu_pretrain/training/finetune.py`,
`scripts/finetune_outcome.py`, `tests/test_finetuning_loop.py`.

**Dependencies:** `TDD-15`, `TDD-16`.

**Test-first requirements:** Train scratch and pretrained models with identical
architecture/fine-tuning settings; use training-only positive class weight;
early-stop on validation average precision; select validation F1 threshold;
support the no-static/no-gap ablation.

Interrupt synthetic scratch and pretrained runs mid-epoch and verify full-state
resume produces the same next batch, optimizer trajectory, early-stopping state,
and final metrics as uninterrupted execution.

**Implementation steps:** Implement common fine-tuning path with optional encoder
checkpoint, JSONL/text logging, periodic and emergency full-state checkpoints,
automatic compatible resume, and frozen test evaluation after model selection.

**Acceptance criteria:** EXP-01 and EXP-02 differ only by initialization;
EXP-03 differs only by representation; test data never drives selection.

**Edge cases:** Missing/incompatible checkpoint, empty validation set, one-class
partition, batch size one, early-stop tie.

**Definition of done:** Controlled scratch, pretrained, and representation
ablation runs are available.

---

## TDD-19 - Run Logging, Recovery State, and Aggregate Tracking

**Status:** Remaining.

**Goal:** Record complete local execution progress, recovery pointers, aggregate
metadata, and metrics without remote tracking or patient-level content.

**Source plan sections:** 11, 14, and 16.

**Files likely to change:** `src/icu_pretrain/experiments/tracking.py`,
`tests/test_tracking.py`.

**Dependencies:** `TDD-01`.

**Test-first requirements:** Write stable CSV/JSON schemas with experiment ID,
representation, patient/stay/class/hospital counts, split, seed, metrics,
confidence intervals, parameter count, runtime, exclusions, and failure notes.
Reject IDs, raw tokens, tensors, or event dumps.

Also verify append-only, line-flushed `events.jsonl`, mirrored human-readable
`run.log`, atomic `state.json`, status transitions, exception traceback logging,
checkpoint references, and safe recovery after a truncated final JSONL line.

**Implementation steps:** Add run IDs/directories, structured event logger,
text logger, atomic state writer, checkpoint registry, aggregate writers,
duplicate-run policy, and best-config metadata writer.

**Acceptance criteria:** Missing/undefined metrics remain null with explanation;
no value is fabricated.

**Edge cases:** Existing file, duplicate experiment, non-finite metric, failed
run, absent optional output.

**Definition of done:** Every runner emits compatible, public-safe summaries.

---

## TDD-20 - Core Experiment Registry and Runner

**Status:** Remaining.

**Goal:** Orchestrate the prespecified central experiments and final selection.

**Source plan section:** 10.

**Files likely to change:** `src/icu_pretrain/experiments/registry.py`,
`src/icu_pretrain/experiments/runner.py`, `tests/test_experiment_runner.py`.

**Dependencies:** `TDD-17`, `TDD-18`, `TDD-19`.

**Test-first requirements:** Register EXP-00 logistic full, EXP-01 scratch full,
EXP-02 pretrained full, EXP-03 pretrained ablation, EXP-04 hospital grouped,
and optional EXP-05 FedAvg. Select among EXP-01/02/03 by validation average
precision and evaluate the selected central model once on test.

Resume an interrupted suite by reusing completed compatible child runs and
starting only incomplete experiments; parent logs must link child run IDs.

**Implementation steps:** Add immutable registry, dispatch, suite ordering,
selection record, failure isolation, and no-rerun test guard.

**Acceptance criteria:** Unsupported combinations fail; all prespecified central
results are retained, including negative results.

**Edge cases:** Tie in selection metric, failed experiment, existing result,
missing optional config, attempted test evaluation before selection freeze.

**Definition of done:** The reduced controlled experiment matrix runs through
one reproducible interface.

---

## TDD-21 - Optional Eight-Trial Optuna Search

**Status:** Remaining.

**Goal:** Run the constrained optional tuning study after core experiments.

**Source plan section:** 10.3.

**Files likely to change:** `src/icu_pretrain/tuning/optuna_search.py`,
`scripts/run_optuna_tuning.py`, `tests/test_optuna_search.py`.

**Dependencies:** `TDD-20`.

**Test-first requirements:** Limit to eight trials, optimize validation average
precision, keep data/preprocessing/splits fixed, reject dimensions not divisible
by heads, and never access test metrics.

Each trial is a resumable child run. Restarting the study must preserve completed
trials and resume or rerun only the interrupted trial from its latest checkpoint.

**Implementation steps:** Lazily import Optuna, merge sampled values into the
base config, run trial training, and write local trial/best-parameter artifacts.

**Acceptance criteria:** Missing Optuna gives an actionable message; zero
completed trials does not create a fake best configuration.

**Edge cases:** Invalid sampled pair, pruned/failed trials, existing study,
override above eight trials.

**Definition of done:** Optional CPU-friendly tuning is available without
changing the primary protocol.

---

## TDD-22 - Hospital-Grouped Evaluation and FedAvg Simulation

**Status:** Remaining.

**Goal:** Implement EXP-04 held-out-hospital evaluation and optional EXP-05
FedAvg over deterministic hospital clusters.

**Source plan sections:** 12 and 13.

**Files likely to change:** `src/icu_pretrain/training/federated.py`,
`scripts/run_fedavg_sim.py`, `tests/test_federated.py`.

**Dependencies:** `TDD-05`, `TDD-16`, `TDD-18`.

**Test-first requirements:**

- Use five-fold `StratifiedGroupKFold` grouped by `hospitalid`, seed 42, with
  preprocessing refitted inside each training fold.
- Ensure hospitals never cross folds; report fold and pooled out-of-fold metrics.
- Build three training clients by descending hospital size and greedy assignment
  to the smallest client, with deterministic ties.
- Keep validation/test stays out of clients; run three rounds, one local epoch,
  full participation, and stay-count-weighted parameter averaging.
- Log and checkpoint each fold, client update, and completed FedAvg round as a
  child run so recovery skips completed work and resumes the incomplete child.

**Implementation steps:** Add hospital folds, fold-local fitting hooks, client
clustering, state aggregation, central comparison, and disclaimer metadata.

**Acceptance criteria:** No per-hospital AUROC is reported; one-class folds are
reported as undefined; output says exploratory/simulated and makes no secure-FL
claim.

**Edge cases:** One-class fold/client, empty client, nonfloating state tensor,
hospital count ties, incompatible model states.

**Definition of done:** Site-aware evaluation and optional non-pooled simulation
match the actual dataset and stated limitations.

---

## TDD-23 - Optional Memorisation Diagnostic

**Status:** Remaining.

**Goal:** Add aggregate rare-pattern diagnostics without presenting them as a
privacy audit.

**Source plan section:** 15.

**Files likely to change:** `src/icu_pretrain/analysis/memorisation.py`,
`scripts/run_memorisation_probe.py`, `tests/test_memorisation_probe.py`.

**Dependencies:** `TDD-15`, `TDD-18`.

**Test-first requirements:** Measure rare training n-gram overlap with held-out
streams and masked prediction accuracy by training-token frequency; write only
aggregate counts and mandatory disclaimer metadata.

**Implementation steps:** Add deterministic frequency strata, overlap metrics,
optional model scoring, and aggregate JSON output.

**Acceptance criteria:** No patient IDs or token sequences are emitted; absent
model/rare patterns are handled without fabricated results.

**Edge cases:** No rare n-grams, all rare, top-k above vocabulary, vocabulary
mismatch, empty held-out data.

**Definition of done:** The optional diagnostic is reproducible and explicitly
non-guaranteeing.

---

## TDD-24 - Result Assets, Documentation, and Application Polish

**Status:** Remaining.

**Goal:** Generate public aggregate summaries and documentation consistent with
the verified dataset and completed experiments.

**Source plan sections:** 1, 2, 11, 14, 16, and 19.

**Files likely to change:** `src/icu_pretrain/analysis/`,
`scripts/make_report_assets.py`, `README.md`, and `docs/` reports.

**Dependencies:** `TDD-19`, `TDD-20`, `TDD-22`.

**Test-first requirements:** Build tables and figures from real aggregate rows;
handle absent tuning, FedAvg, or memorisation outputs; suppress/combine subgroup
cells below 10 stays; reject patient-level fields; never fabricate metrics.

Document chunk/shard defaults, run-log locations, checkpoint retention, resume
commands, compatibility failures, and recovery limitations.

**Implementation steps:** Generate cohort summary, experiment comparison, final
results, selected config, model comparison, hospital-grouped figure, and optional
FedAvg figure; update model card, data statement, protocol, limitations, README,
and application note.

**Acceptance criteria:** Public text states 2,520 raw ICU stays, 1,841 patients,
186 hospitals, and 292 wards; distinguishes patient-grouped, hospital-grouped,
and simulated-client results; limits claims to the eICU demo.

**Edge cases:** Results with headers only, undefined confidence interval, failed
fold, absent optional artifact, stale claim in documentation.

**Definition of done:** The repository is application-polished, reproducible,
and honest about dataset scope, uncertainty, and incomplete optional work.

---

## Recommended Execution Order

1. Complete `TDD-00` through `TDD-15` in dependency order.
2. Implement `TDD-16`, then `TDD-17` and `TDD-18`.
3. Complete tracking and central orchestration with `TDD-19` and `TDD-20`.
4. Complete hospital-grouped evaluation in `TDD-22`.
5. Run optional `TDD-21` and `TDD-23` only after the core pipeline is stable.
6. Finish aggregate reporting and documentation with `TDD-24`.
