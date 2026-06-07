# TDD Task Cards for `icu-event-pretraining-mini`

Source of truth: `docs/junk/plan.md`.

## Implementation Status

Last updated: 2026-06-07

| Task card | Status |
|---|---|
| TDD-00 - Establish Test Harness and Package Sanity | Implemented |
| TDD-01 - Configuration Loading and Validation Contracts | Implemented |
| TDD-02 - Event Stream Data Contracts | Implemented |
| TDD-03 - eICU Demo Table Loading | Remaining |
| TDD-04 - Outcome Extraction | Remaining |
| TDD-05 - Static, Diagnosis, Medication, Infusion, and Treatment Events | Remaining |
| TDD-06 - Lab and Vital Numeric Discretization | Remaining |
| TDD-07 - Time-Gap Tokens and Full Event Stream Builder | Remaining |
| TDD-08 - Prepare eICU Demo CLI | Remaining |
| TDD-09 - Event Tokenizer and Vocabulary Artifacts | Remaining |
| TDD-10 - Splits and Pseudo-Client Grouping | Remaining |
| TDD-11 - Encoded Dataset Creation | Remaining |
| TDD-12 - Collation, Padding, Attention Masks, and Masked LM Targets | Remaining |
| TDD-13 - Tiny Transformer Encoder | Remaining |
| TDD-14 - Prediction Heads | Remaining |
| TDD-15 - Masked Event Pretraining Loop and CLI | Remaining |
| TDD-16 - Evaluation Metrics | Remaining |
| TDD-17 - Bag-of-Events Logistic Baselines | Remaining |
| TDD-18 - Fine-Tuning Workflow | Remaining |
| TDD-19 - Local Experiment Tracking | Remaining |
| TDD-20 - Experiment Registry and Runner | Remaining |
| TDD-21 - Optuna Tuning Search | Remaining |
| TDD-22 - Pseudo-Client Evaluation and FedAvg Simulation | Remaining |
| TDD-23 - Rare-Pattern Memorisation Probe | Remaining |
| TDD-24 - Result Tables, Figures, and Application Polish | Remaining |

Progress: **3 implemented, 22 remaining.**

Notes from scaffold inspection:
- The planned package directories already exist under `src/icu_pretrain/`.
- Most implementation modules and scripts are placeholders raising `NotImplementedError`.
- `tests/` does not exist yet.
- `pyproject.toml` has no pytest/dev optional dependency yet.
- Experiment configs exist but are minimal; final/tuning/FedAvg configs match the plan more closely.
- `docs/plan.md` is deleted in the current worktree; the active source plan is `docs/junk/plan.md`.
- This file is under `docs/junk/`, which is currently ignored by `.gitignore`.

## Dependency Map

Core chain:

`TDD-00` -> `TDD-01` -> `TDD-02` -> `TDD-03` -> `TDD-04` -> `TDD-05` -> `TDD-06` -> `TDD-07` -> `TDD-08` -> `TDD-09` -> `TDD-10` -> `TDD-11` -> `TDD-12` -> `TDD-13` -> `TDD-14` -> `TDD-15` -> `TDD-16` -> `TDD-17` -> `TDD-18` -> `TDD-19` -> `TDD-20` -> `TDD-21` -> `TDD-22` -> `TDD-23` -> `TDD-24`

Parallel-safe after dependencies:

- `TDD-17` baselines can proceed after `TDD-16`.
- `TDD-19` tracking can proceed after `TDD-01`.
- `TDD-24` reporting can proceed after `TDD-19`, though final documentation polish should remain last.

---

## TDD-00 - Establish Test Harness and Package Sanity [Implemented]

**Goal:** Add a minimal local test harness so all later work can be developed test-first.

**Source plan section or requirement:** TDD requirement from user request; plan section 13 repo structure; plan section 20 development schedule.

**Files likely to be created or modified:**
- `pyproject.toml`
- `tests/test_package_sanity.py`
- Optional `tests/conftest.py`

**Dependencies:** None.

**Test-first requirements:**
- Create `tests/test_package_sanity.py` first.
- Add failing tests that assert `icu_pretrain.__version__` imports, `load_yaml()` loads mappings, expected package submodules import, and placeholder modules are importable.
- Minimal implementation should add pytest/dev dependency metadata if needed and any missing test discovery configuration.
- Add regression tests asserting imports do not require raw eICU data, GPU, Optuna, notebooks, or generated processed data.

**Implementation steps:**
1. Add pytest as a dev/test dependency or documented optional extra.
2. Add sanity tests importing `icu_pretrain`, data, model, training, experiment, tuning, and analysis modules.
3. Keep tests CPU-only and independent from real data.
4. Do not remove placeholders unless needed to make imports stable.

**Acceptance criteria:**
- `python -m pytest` discovers tests.
- Package imports succeed from editable install or configured `PYTHONPATH`.
- No test reads `data/raw/` or `data/processed/`.

**Edge cases to cover:**
- Empty YAML file returns `{}` through `load_yaml()`.
- Non-mapping YAML raises `ValueError`.

**Definition of done:** A developer can run the test harness locally before implementing pipeline features.

---

## TDD-01 - Configuration Loading and Validation Contracts [Implemented]

**Goal:** Treat YAML configs as validated contracts for scripts, experiments, tuning, and FedAvg workflows.

**Source plan section or requirement:** Plan section 14 config design; plan section 15 scripts and commands.

**Files likely to be created or modified:**
- `src/icu_pretrain/utils.py`
- Optional `src/icu_pretrain/experiments/registry.py`
- `tests/test_config_validation.py`

**Dependencies:** `TDD-00`.

**Test-first requirements:**
- Create `tests/test_config_validation.py` first.
- Add failing tests that load `configs/final/eicu_demo_final_tiny.yaml`, `configs/tuning/optuna_eicu_tiny.yaml`, `configs/fedavg/eicu_demo_fedavg_sim.yaml`, and `configs/experiments/exp_suite_core.yaml`.
- Tests should assert required keys, CPU defaults, metric names, representation names, split names, and supported model names.
- Minimal implementation should add validation helpers returning normalized dictionaries.
- Add regression tests rejecting missing required sections, unsupported representation, unsupported metric, invalid device, incompatible `d_model`/`n_heads`, and malformed Optuna search spaces.

**Implementation steps:**
1. Add `validate_final_config`, `validate_tuning_config`, `validate_fedavg_config`, and `validate_experiment_config`.
2. Normalize short experiment configs without expanding them into unrelated features.
3. Validate known representations: `basic`, `timegap`, `timegap_static`.
4. Validate metrics: `auroc`, `average_precision`, `f1`, `balanced_accuracy`.
5. Keep error messages explicit and key-oriented.

**Acceptance criteria:**
- All scaffolded YAML configs pass validation.
- Invalid configs fail before training starts.
- Validation does not mutate config files.

**Edge cases to cover:**
- Empty YAML file.
- Unknown experiment ID.
- `d_model` not divisible by `n_heads`.
- Missing `runtime.device`.

**Definition of done:** Later CLIs and runners can rely on one config validation layer.

---

## TDD-02 - Event Stream Data Contracts [Implemented]

**Goal:** Define stable in-memory and serialized contracts for patient event streams, outcomes, and stats.

**Source plan section or requirement:** Plan section 5 event-stream representation; plan section 20 Day 1 outputs.

**Files likely to be created or modified:**
- `src/icu_pretrain/data/eicu_event_builder.py`
- `src/icu_pretrain/constants.py`
- `tests/test_event_contracts.py`

**Dependencies:** `TDD-01`.

**Test-first requirements:**
- Create `tests/test_event_contracts.py` first.
- Add failing tests asserting an event stream record contains `patientunitstayid`, ordered `events`, optional event times, `representation`, and metadata.
- Tests should assert valid token families are only `STATIC`, `DX`, `LAB`, `VITAL`, `MED`, `INFUSION`, `TREATMENT`, and `TIME_GAP`.
- Minimal implementation should add typed dataclasses or structured dictionaries plus validation helpers.
- Add regression tests rejecting empty IDs, empty event lists after filtering, unknown token families, non-string event tokens, and unsorted event timestamps.

**Implementation steps:**
1. Define an `EventStream` contract and validation function.
2. Define an `OutcomeRecord` contract for mortality-style binary labels.
3. Define `event_stats.json` schema fields: counts, min/max/median sequence length, token family counts, skipped stays.
4. Add JSONL serialization and deserialization helpers.

**Acceptance criteria:**
- Synthetic event streams round-trip through JSONL.
- Invalid event records fail with clear errors.
- Tests do not embed protected/raw data assumptions.

**Edge cases to cover:**
- Static-only stay below `min_events_per_stay`.
- Duplicate timestamps.
- Missing outcome label.
- Unknown event token prefix.

**Definition of done:** Later data, tokenizer, and dataset tasks share one event-stream contract.

---

## TDD-03 - eICU Demo Table Loading

**Goal:** Load expected eICU demo CSV tables safely and fail clearly when required files or columns are missing.

**Source plan section or requirement:** Plan section 3 dataset plan and MVP tables; AGENTS data-safety rules.

**Files likely to be created or modified:**
- `src/icu_pretrain/data/eicu_demo.py`
- `tests/test_eicu_demo_loader.py`

**Dependencies:** `TDD-02`.

**Test-first requirements:**
- Create `tests/test_eicu_demo_loader.py` using tiny synthetic CSVs in `tmp_path`.
- Add failing tests that assert `table_path()` resolves `<raw_dir>/<table>.csv`, MVP table names match the plan, and the loader reads only requested tables.
- Minimal implementation should add `load_tables(raw_dir, tables=MVP_TABLES)` with pandas.
- Add regression tests for missing table, missing required `patientunitstayid`, empty CSV, and extra columns.

**Implementation steps:**
1. Declare expected minimal required columns per MVP table.
2. Add CSV loading with explicit table-name validation.
3. Validate `patientunitstayid` where required.
4. Avoid writing loaded raw data anywhere.

**Acceptance criteria:**
- Synthetic MVP CSVs load into a dict of DataFrames.
- Missing required files or columns produce actionable exceptions.
- Loader supports optional table subsets for tests.

**Edge cases to cover:**
- Capitalization-sensitive table names.
- Empty optional table.
- Duplicate `patientunitstayid` rows where a table naturally has many events.

**Definition of done:** Event builder can consume validated table DataFrames without direct filesystem assumptions.

---

## TDD-04 - Outcome Extraction

**Goal:** Extract one mortality/discharge-style binary outcome per `patientunitstayid`.

**Source plan section or requirement:** Plan section 8 downstream task; plan section 20 Day 1 `outcomes.csv`.

**Files likely to be created or modified:**
- `src/icu_pretrain/data/eicu_event_builder.py`
- `tests/test_outcome_extraction.py`

**Dependencies:** `TDD-03`.

**Test-first requirements:**
- Create `tests/test_outcome_extraction.py`.
- Add failing tests using synthetic `patient` and/or `apachePatientResult` rows to produce `outcomes.csv` columns `patientunitstayid` and `mortality`.
- Minimal implementation should choose available mortality/discharge-status fields conservatively.
- Add regression tests for missing label fields, ambiguous labels, duplicate patient rows, and non-binary labels.

**Implementation steps:**
1. Implement `extract_outcomes(tables)` returning one row per stay.
2. Prefer explicit hospital/discharge mortality-style fields when present.
3. Fall back to binary discharge-status style fields only when unambiguous.
4. Drop or mark stays with unavailable labels and report counts in stats.

**Acceptance criteria:**
- Output labels are integer `0`/`1`.
- Stays without reliable labels do not silently become negative labels.
- Extraction does not claim clinical deployment readiness.

**Edge cases to cover:**
- Text labels with case differences.
- Conflicting duplicate label rows.
- Label available in one table but not another.

**Definition of done:** Downstream training has a validated binary target contract.

---

## TDD-05 - Static, Diagnosis, Medication, Infusion, and Treatment Events

**Goal:** Build deterministic non-numeric event tokens from synthetic eICU table rows.

**Source plan section or requirement:** Plan section 5 event families; plan section 10 R1 representation.

**Files likely to be created or modified:**
- `src/icu_pretrain/data/eicu_event_builder.py`
- `tests/test_event_builder_categorical.py`

**Dependencies:** `TDD-04`.

**Test-first requirements:**
- Create `tests/test_event_builder_categorical.py`.
- Add failing tests asserting tokens such as `STATIC::AGE_BIN::*`, `STATIC::GENDER::*`, `DX::*`, `MED::*`, `INFUSION::*`, and `TREATMENT::*`.
- Minimal implementation should normalize text tokens, bucket age, and sort events chronologically when offsets exist.
- Add regression tests for nulls, duplicate rows, odd whitespace, unknown gender/admission source, and missing event offsets.

**Implementation steps:**
1. Implement text normalization for token suffixes.
2. Add static context tokens for age bin, gender, admission source, and unit context when available.
3. Add diagnosis, medication, infusion, and treatment token extraction.
4. Keep representation filtering clear: R1 includes clinical events, R2 adds time gaps, R3 adds static context.

**Acceptance criteria:**
- Synthetic stays produce deterministic token sequences.
- R1 excludes static and time-gap tokens unless requested by representation.
- Missing optional fields do not crash event construction.

**Edge cases to cover:**
- Age `> 89` or masked age values.
- Multiple diagnosis rows with same offset.
- Medication names with spaces or symbols.
- Treatment strings with nested categories.

**Definition of done:** Basic categorical ICU event streams can be built without numeric discretization.

---

## TDD-06 - Lab and Vital Numeric Discretization

**Goal:** Convert continuous lab and vital values into compact quantile-bin event tokens.

**Source plan section or requirement:** Plan section 5 numeric value handling; plan section 10 representation variants.

**Files likely to be created or modified:**
- `src/icu_pretrain/data/eicu_event_builder.py`
- `tests/test_numeric_discretization.py`

**Dependencies:** `TDD-05`.

**Test-first requirements:**
- Create `tests/test_numeric_discretization.py`.
- Add failing tests that fit quantile bins on synthetic values and emit tokens such as `LAB::CREATININE::Q1` and `VITAL::HEARTRATE::Q4`.
- Minimal implementation should add fit/apply bin helpers per measurement name.
- Add regression tests for nonnumeric values, all-equal values, missing measurement names, values outside fitted range, and small sample sizes.

**Implementation steps:**
1. Extract lab values from `lab` and vital values from `vitalPeriodic`/`vitalAperiodic`.
2. Fit quantile bin thresholds on available rows.
3. Apply bins deterministically with stable labels `Q1` to `Q4`.
4. Include skipped-value counts in stats.

**Acceptance criteria:**
- No raw continuous values are emitted as Transformer tokens.
- Bin behavior is deterministic for the same input and seed.
- Small demo-sized data does not crash bin fitting.

**Edge cases to cover:**
- Fewer than four unique values.
- Negative lab values.
- String values like `"5.2"` and invalid strings.
- Missing event time offsets.

**Definition of done:** Numeric ICU measurements become model-ready event tokens.

---

## TDD-07 - Time-Gap Tokens and Full Event Stream Builder

**Goal:** Assemble chronological patient-level sequences for `basic`, `timegap`, and `timegap_static` representations.

**Source plan section or requirement:** Plan section 5 event-stream representation; plan section 10 R1/R2/R3; plan section 20 Day 1 outputs.

**Files likely to be created or modified:**
- `src/icu_pretrain/data/eicu_event_builder.py`
- `tests/test_event_stream_builder.py`

**Dependencies:** `TDD-06`.

**Test-first requirements:**
- Create `tests/test_event_stream_builder.py`.
- Add failing tests asserting one sequence per `patientunitstayid`, chronological ordering, `TIME_GAP::*` insertion for R2/R3, static tokens for R3, and filtering by `min_events_per_stay`.
- Minimal implementation should expose `build_event_streams(tables, representation, min_events_per_stay)`.
- Add regression tests for zero/negative time gaps, missing offsets, multiple rows at same time, and stays with no outcome.

**Implementation steps:**
1. Merge categorical and numeric event extractors.
2. Sort events by stay ID, event offset, and deterministic token tie-breaker.
3. Insert time-gap tokens only between non-static chronological events for time-aware representations.
4. Return event streams, outcomes, and stats.

**Acceptance criteria:**
- R1/R2/R3 behavior matches the plan.
- Event streams validate against `TDD-02` contracts.
- Stats include kept/skipped stays and token family counts.

**Edge cases to cover:**
- Same patient in only one event table.
- Missing all offsets.
- Huge sequence, which should be truncated later rather than here.
- Static context unavailable.

**Definition of done:** The core patient-level ICU event-stream builder works on synthetic demo-shaped data.

---

## TDD-08 - Prepare eICU Demo CLI

**Goal:** Implement `scripts/prepare_eicu_demo.py` to produce local processed artifacts.

**Source plan section or requirement:** Plan section 15 basic polished run; plan section 20 Day 1 outputs.

**Files likely to be created or modified:**
- `scripts/prepare_eicu_demo.py`
- `src/icu_pretrain/data/eicu_demo.py`
- `tests/test_prepare_eicu_demo_cli.py`

**Dependencies:** `TDD-07`.

**Test-first requirements:**
- Create `tests/test_prepare_eicu_demo_cli.py`.
- Add failing tests invoking the script against synthetic CSVs and asserting `event_streams.jsonl`, `outcomes.csv`, and `event_stats.json` are created under `--out_dir`.
- Minimal implementation should add argparse and call loader/builder functions.
- Add regression tests for missing raw dir, unsupported representation, and refusal to write outside specified output path.

**Implementation steps:**
1. Add CLI args: `--raw_dir`, `--out_dir`, `--representation`, `--min_events_per_stay`.
2. Create output directory if needed.
3. Write JSONL, CSV, and stats JSON.
4. Print a concise summary without patient-level details.

**Acceptance criteria:**
- CLI succeeds on synthetic data.
- CLI fails clearly when required raw CSVs are absent.
- Outputs remain under ignored `data/processed/` by default.

**Edge cases to cover:**
- Existing output files.
- Empty output after filtering.
- Relative and absolute paths.

**Definition of done:** Day 1 data preparation can run locally without modifying tracked data.

---

## TDD-09 - Event Tokenizer and Vocabulary Artifacts

**Goal:** Fit, persist, load, and apply an event tokenizer with special tokens.

**Source plan section or requirement:** Plan section 5 event tokenisation; plan section 20 Day 2 `vocab.json`.

**Files likely to be created or modified:**
- `src/icu_pretrain/data/tokenizer.py`
- `tests/test_tokenizer.py`

**Dependencies:** `TDD-08`.

**Test-first requirements:**
- Create `tests/test_tokenizer.py`.
- Add failing tests asserting `[PAD]`, `[UNK]`, `[MASK]`, and `[CLS]` IDs are stable, fit counts events, encode/decode works, and unknown tokens map to `[UNK]`.
- Minimal implementation should complete `EventTokenizer.fit`, `encode`, `decode`, `save`, and `load`.
- Add regression tests for deterministic ordering, min frequency, empty sequences, duplicate tokens, and corrupt vocab files.

**Implementation steps:**
1. Use `SPECIAL_TOKENS` from constants.
2. Fit vocabulary from training sequences only.
3. Persist JSON with token-to-id, ordered vocabulary, frequencies, and metadata.
4. Keep token strings unchanged after event builder normalization.

**Acceptance criteria:**
- Vocabulary is deterministic for identical inputs.
- Round-trip save/load preserves IDs.
- Unknown event tokens never crash encoding.

**Edge cases to cover:**
- Token equal to a special token.
- Empty fit input.
- Very long token strings.
- Non-list sequence input.

**Definition of done:** Event streams can be converted into stable integer IDs.

---

## TDD-10 - Splits and Pseudo-Client Grouping

**Goal:** Create random splits and pseudo-client grouping strategies from processed streams.

**Source plan section or requirement:** Plan section 10 evaluation variants and pseudo-client strategies; plan section 20 Day 2 `split_metadata.json`.

**Files likely to be created or modified:**
- `src/icu_pretrain/data/splits.py`
- `tests/test_splits.py`

**Dependencies:** `TDD-09`.

**Test-first requirements:**
- Create `tests/test_splits.py`.
- Add failing tests asserting deterministic random train/val/test splits and event-density pseudo-client groups `low`, `medium`, `high`.
- Minimal implementation should replace `make_random_split()` and add pseudo-client split helpers.
- Add regression tests for tiny datasets, duplicate IDs, missing labels, diagnosis grouping fallback, and admission-context fallback.

**Implementation steps:**
1. Implement seeded random split with configurable proportions.
2. Implement event-density grouping based on sequence length quantiles.
3. Implement diagnosis-group and admission-context grouping when metadata exists.
4. Emit `split_metadata.json` with IDs, strategy, seed, counts, and skipped records.

**Acceptance criteria:**
- Splits are disjoint and cover all eligible stays.
- Default pseudo-client strategy is `event_density`.
- Grouped split never claims true hospital/site split.

**Edge cases to cover:**
- Fewer than three stays.
- Class imbalance.
- All stays have the same event count.
- Missing diagnosis/admission fields.

**Definition of done:** Experiment workflows can select random or pseudo-client evaluation splits reproducibly.

---

## TDD-11 - Encoded Dataset Creation

**Goal:** Create PyTorch dataset objects and encoded `.pt` artifacts from tokenized streams and labels.

**Source plan section or requirement:** Plan section 20 Day 2 outputs; plan section 6 model input shape.

**Files likely to be created or modified:**
- `src/icu_pretrain/data/dataset.py`
- `tests/test_dataset.py`

**Dependencies:** `TDD-10`.

**Test-first requirements:**
- Create `tests/test_dataset.py`.
- Add failing tests instantiating `ICUEventDataset` from encoded records and asserting `__len__`, `__getitem__`, truncation, label handling, and metadata fields.
- Minimal implementation should make `ICUEventDataset` a `torch.utils.data.Dataset`.
- Add regression tests for missing labels for pretraining, label-required fine-tuning, max sequence truncation, empty sequences, and loaded `.pt` artifacts.

**Implementation steps:**
1. Define encoded record structure with `input_ids`, `patientunitstayid`, optional `label`, and optional group metadata.
2. Add creation helper from event streams, tokenizer, outcomes, split IDs, and `max_seq_len`.
3. Save `encoded_train.pt`, `encoded_val.pt`, and `encoded_test.pt`.
4. Keep patient IDs as metadata, not model inputs.

**Acceptance criteria:**
- Dataset supports pretraining records without labels and fine-tuning records with labels.
- Encoded artifacts load back into equivalent datasets.
- Long sequences are truncated deterministically.

**Edge cases to cover:**
- Sequence exactly `max_seq_len`.
- Sequence shorter than one token.
- Outcome label missing for a split member.
- Unknown tokens after vocab fitting.

**Definition of done:** Data can flow from prepared JSONL/CSV into PyTorch datasets.

---

## TDD-12 - Collation, Padding, Attention Masks, and Masked LM Targets

**Goal:** Batch variable-length event sequences and create masked-event pretraining inputs.

**Source plan section or requirement:** Plan section 7 masked event modelling; plan section 6 CPU-friendly config.

**Files likely to be created or modified:**
- `src/icu_pretrain/data/collate.py`
- `tests/test_collate.py`

**Dependencies:** `TDD-11`.

**Test-first requirements:**
- Create `tests/test_collate.py`.
- Add failing tests asserting padded `input_ids`, `attention_mask`, labels tensor, and masked-event `target_ids` where unmasked positions are ignored.
- Minimal implementation should replace `collate_event_batch()` and add a pretraining collator.
- Add regression tests for no special tokens masked, at least one eligible mask when possible, all-padding prevention, deterministic masking with seed, and empty batch rejection.

**Implementation steps:**
1. Implement supervised collator for fine-tuning.
2. Implement masked pretraining collator using `mask_probability`.
3. Use `[PAD]` for padding, `[MASK]` for selected positions, and `-100` for ignored target positions.
4. Return tensors on CPU by default.

**Acceptance criteria:**
- Batch output is directly usable by PyTorch models.
- Masked loss is computed only on selected event positions.
- Padding does not affect attention or metrics.

**Edge cases to cover:**
- Single-token sequence.
- Batch with mixed lengths.
- `mask_probability` 0 or 1.
- Sequence containing `[CLS]` or special tokens.

**Definition of done:** Training loops can consume batched tensors without custom per-loop padding logic.

---

## TDD-13 - Tiny Transformer Encoder

**Goal:** Implement `ICU-TinyTransformer` as a CPU-friendly PyTorch encoder.

**Source plan section or requirement:** Plan section 6 model plan and final CPU config.

**Files likely to be created or modified:**
- `src/icu_pretrain/models/transformer.py`
- `tests/test_transformer.py`

**Dependencies:** `TDD-12`.

**Test-first requirements:**
- Create `tests/test_transformer.py`.
- Add failing tests instantiating the model with vocab size, max seq len, d_model, heads, layers, feedforward dim, dropout, and asserting output shape `[batch, seq, d_model]`.
- Minimal implementation should subclass `torch.nn.Module` and use `torch.nn.TransformerEncoderLayer`.
- Add regression tests for attention masks, max length overflow, invalid `d_model/n_heads`, deterministic eval mode, and parameter count reporting.

**Implementation steps:**
1. Add token embeddings and positional embeddings.
2. Apply transformer encoder with padding mask support.
3. Return sequence embeddings and optionally pooled embedding.
4. Keep defaults aligned with final config: max_seq_len 128, d_model 64, 4 heads, 2 layers.

**Acceptance criteria:**
- Forward pass works on CPU for small synthetic batches.
- Padding mask is accepted and shape-checked.
- No GPU requirement is introduced.

**Edge cases to cover:**
- Sequence length greater than `max_seq_len`.
- Empty batch.
- Invalid attention mask shape.
- Dropout disabled in eval mode.

**Definition of done:** Encoder is ready for masked pretraining and fine-tuning heads.

---

## TDD-14 - Prediction Heads

**Goal:** Implement masked event and binary outcome prediction heads.

**Source plan section or requirement:** Plan section 6 architecture; plan section 7 objective; plan section 8 downstream task.

**Files likely to be created or modified:**
- `src/icu_pretrain/models/heads.py`
- `tests/test_heads.py`

**Dependencies:** `TDD-13`.

**Test-first requirements:**
- Create `tests/test_heads.py`.
- Add failing tests asserting masked head logits shape `[batch, seq, vocab_size]` and outcome head logits shape `[batch]` or `[batch, 1]`.
- Minimal implementation should add `MaskedEventPredictionHead` and `OutcomePredictionHead`.
- Add regression tests for ignored labels with cross entropy, pooled output for outcome prediction, and dtype/device consistency.

**Implementation steps:**
1. Add linear masked-event head over sequence embeddings.
2. Add binary outcome head over pooled `[CLS]` or masked mean pooling.
3. Provide small loss helper functions or documented expected loss inputs.
4. Avoid clinical calibration claims.

**Acceptance criteria:**
- Heads integrate with encoder outputs.
- Loss can be computed for synthetic masked and binary targets.
- Shapes remain stable for batch size 1.

**Edge cases to cover:**
- All target positions ignored.
- All labels same class in a tiny batch.
- Missing pooled representation.

**Definition of done:** Model components support both pretraining and downstream supervised learning.

---

## TDD-15 - Masked Event Pretraining Loop and CLI

**Goal:** Train the encoder with masked event modelling and write pretraining metrics/checkpoints locally.

**Source plan section or requirement:** Plan section 7 main objective; plan section 15 `scripts/pretrain.py`; plan section 20 Day 3 output.

**Files likely to be created or modified:**
- `src/icu_pretrain/training/pretrain.py`
- `scripts/pretrain.py`
- `tests/test_pretraining_loop.py`
- `tests/test_pretrain_cli.py`

**Dependencies:** `TDD-14`.

**Test-first requirements:**
- Create `tests/test_pretraining_loop.py` and `tests/test_pretrain_cli.py`.
- Add failing tests running one tiny CPU epoch on synthetic encoded data and asserting masked loss is returned and metrics JSON/CSV is written.
- Minimal implementation should add `run_pretraining(config)` and script argparse `--config`.
- Add regression tests for missing encoded files, invalid mask probability, gradient accumulation, zero train batches, and checkpoint path safety.

**Implementation steps:**
1. Load and validate config.
2. Load tokenizer/vocab and encoded train/val datasets.
3. Build encoder plus masked head.
4. Train masked positions only with `CrossEntropyLoss(ignore_index=-100)`.
5. Save local checkpoint/metrics under ignored or configured output path; do not commit checkpoints.

**Acceptance criteria:**
- Tiny synthetic pretraining reports finite loss.
- CLI exits nonzero with clear message when processed data is missing.
- Default runtime stays CPU.

**Edge cases to cover:**
- No masked positions in a batch.
- NaN loss.
- Existing checkpoint file.
- Config with `pretraining.enabled: false`.

**Definition of done:** The core masked-event pretraining objective works end to end on small data.

---

## TDD-16 - Evaluation Metrics

**Goal:** Compute downstream metrics consistently for binary outcome prediction.

**Source plan section or requirement:** Plan section 8 metrics; plan section 19 result fields.

**Files likely to be created or modified:**
- `src/icu_pretrain/training/evaluate.py`
- `tests/test_evaluate_metrics.py`

**Dependencies:** `TDD-12`.

**Test-first requirements:**
- Create `tests/test_evaluate_metrics.py`.
- Add failing tests asserting AUROC, Average Precision, F1, and Balanced Accuracy on known arrays.
- Minimal implementation should replace `evaluate_predictions()` with a function accepting labels, scores, optional threshold, and metrics list.
- Add regression tests for single-class labels, NaN scores, mismatched lengths, empty arrays, and threshold behavior.

**Implementation steps:**
1. Use scikit-learn metrics already declared in dependencies.
2. Return a dictionary with plan-aligned metric names.
3. Handle undefined AUROC/AP cases explicitly with `None` or `nan` plus warning metadata.
4. Keep score-based metrics separate from threshold-based metrics.

**Acceptance criteria:**
- Metric keys match config and result CSV schema.
- Undefined metrics do not crash full experiment runs unless configured strict.
- Tests document all fallback behavior.

**Edge cases to cover:**
- All-zero labels.
- All-one predictions.
- Scores outside `[0, 1]`.
- Different threshold values.

**Definition of done:** Baselines, fine-tuning, experiments, and FedAvg all share one metric implementation.

---

## TDD-17 - Bag-of-Events Logistic Baselines

**Goal:** Implement EXP-00 and EXP-01 logistic regression baselines.

**Source plan section or requirement:** Plan section 11 Block A; plan section 20 Day 4 `baseline_results.csv`.

**Files likely to be created or modified:**
- `src/icu_pretrain/training/baselines.py`
- `scripts/run_baselines.py`
- `tests/test_baselines.py`

**Dependencies:** `TDD-16`.

**Test-first requirements:**
- Create `tests/test_baselines.py`.
- Add failing tests training logistic regression on synthetic token-count features and writing `results/summary/baseline_results.csv`.
- Minimal implementation should vectorize bag-of-events counts and call scikit-learn logistic regression.
- Add regression tests for unknown tokens in validation, no positive labels, tiny train set fallback, and deterministic seed.

**Implementation steps:**
1. Convert encoded or tokenized sequences into bag-of-events features.
2. Train logistic regression for `basic` and `timegap` configs.
3. Evaluate with shared metrics.
4. Write result rows with experiment ID/name and plan schema.

**Acceptance criteria:**
- EXP-00 and EXP-01 can run on synthetic splits.
- Baseline output schema aligns with result summary fields.
- Failures explain data insufficiency instead of fabricating metrics.

**Edge cases to cover:**
- Vocabulary larger than sample count.
- Empty feature rows.
- Class imbalance.
- Missing split metadata.

**Definition of done:** Non-neural baselines are available for controlled comparison.

---

## TDD-18 - Fine-Tuning Workflow

**Goal:** Fine-tune the tiny Transformer for mortality-style outcome prediction, from scratch or pretrained weights.

**Source plan section or requirement:** Plan section 8 downstream task; plan section 11 Blocks B/C/E; plan section 15 `finetune_outcome.py`.

**Files likely to be created or modified:**
- `src/icu_pretrain/training/finetune.py`
- `scripts/finetune_outcome.py`
- `tests/test_finetuning_loop.py`

**Dependencies:** `TDD-16`, `TDD-15`.

**Test-first requirements:**
- Create `tests/test_finetuning_loop.py`.
- Add failing tests running one tiny CPU fine-tuning epoch from scratch and from a synthetic pretrained checkpoint.
- Minimal implementation should add `run_finetuning(config)` with optional `pretrained_checkpoint`.
- Add regression tests for `freeze_encoder`, missing checkpoint, label-free dataset, batch size 1, and finite metrics.

**Implementation steps:**
1. Load config, tokenizer, encoded datasets, and split metadata.
2. Build encoder plus outcome head.
3. Optionally load pretrained encoder weights.
4. Train with binary classification loss.
5. Evaluate using shared metrics and write result records.

**Acceptance criteria:**
- Scratch and pretrained workflows share most code.
- Outputs include AUROC/AP/F1/Balanced Accuracy when defined.
- CLI matches `python scripts/finetune_outcome.py --config ...`.

**Edge cases to cover:**
- All labels same in validation.
- Checkpoint has incompatible shape.
- Empty validation set.
- `freeze_encoder: true`.

**Definition of done:** Transformer supervised experiments can run after data preparation.

---

## TDD-19 - Local Experiment Tracking

**Goal:** Record metrics and experiment metadata in local, reproducible files without remote tracking.

**Source plan section or requirement:** Plan section 19 result summary schemas; non-goal: no wandb.

**Files likely to be created or modified:**
- `src/icu_pretrain/experiments/tracking.py`
- `tests/test_tracking.py`

**Dependencies:** `TDD-01`.

**Test-first requirements:**
- Create `tests/test_tracking.py`.
- Add failing tests appending result rows to CSV and writing/updating JSON summaries using the exact plan fields.
- Minimal implementation should replace `record_metrics()` with structured writers.
- Add regression tests for duplicate experiment IDs, missing metric keys, atomic writes, and no patient-level data in result files.

**Implementation steps:**
1. Define canonical result columns from plan section 19.
2. Add CSV append/update helper for experiment comparison.
3. Add JSON writer for best config and run metadata.
4. Ensure output dirs are created safely.

**Acceptance criteria:**
- Result files have stable headers.
- Missing metrics are blank/null, not fabricated.
- Writers do not store raw sequences, patient IDs, tensors, or checkpoints.

**Edge cases to cover:**
- Existing empty CSV.
- Existing row for same experiment.
- Non-finite metrics.
- Relative output directory.

**Definition of done:** All runners can write consistent local summaries.

---

## TDD-20 - Experiment Registry and Runner

**Goal:** Orchestrate EXP-00 through EXP-07 and final selected runs from YAML configs.

**Source plan section or requirement:** Plan section 10 experiment matrix; plan section 11 run list; plan section 15 suite commands.

**Files likely to be created or modified:**
- `src/icu_pretrain/experiments/registry.py`
- `src/icu_pretrain/experiments/runner.py`
- `scripts/run_experiment.py`
- `scripts/run_experiment_suite.py`
- `tests/test_experiment_runner.py`

**Dependencies:** `TDD-19`, `TDD-17`, `TDD-18`.

**Test-first requirements:**
- Create `tests/test_experiment_runner.py`.
- Add failing tests loading scaffolded experiment YAML files, dispatching baselines or transformer workflows, and recording result rows using stubbed or synthetic training functions.
- Minimal implementation should map experiment model/pretraining fields to baseline, scratch fine-tune, or pretrain-plus-fine-tune paths.
- Add regression tests for unknown experiment ID, missing config path, suite order, partial failure behavior, and minimal experiment configs.

**Implementation steps:**
1. Expand `EXPERIMENT_IDS` into a registry with names and expected representation/model/pretraining/split.
2. Implement `run_experiment(config_path, base_config=None)` with validation.
3. Implement suite runner over `configs/experiments/exp_suite_core.yaml`.
4. Keep actual training delegated to training modules.

**Acceptance criteria:**
- EXP-00 to EXP-07 configs can be validated and dispatched.
- Suite preserves configured order.
- Runner never invents metrics for failed experiments.

**Edge cases to cover:**
- Config missing `experiment.id`.
- Unsupported `model`.
- Pretraining true for logistic regression.
- Existing results file.

**Definition of done:** Controlled experiment matrix can be launched through planned scripts.

---

## TDD-21 - Optuna Tuning Search

**Goal:** Run a constrained CPU-friendly Optuna search and write best params/trials artifacts.

**Source plan section or requirement:** Plan section 11 Block D; plan section 14 tuning config; plan section 20 Day 6 outputs.

**Files likely to be created or modified:**
- `src/icu_pretrain/tuning/optuna_search.py`
- `scripts/run_optuna_tuning.py`
- `tests/test_optuna_search.py`

**Dependencies:** `TDD-20`.

**Test-first requirements:**
- Create `tests/test_optuna_search.py`.
- Add failing tests running a 1-2 trial synthetic or monkeypatched objective and asserting `trials.csv` and `best_params.json`.
- Minimal implementation should import Optuna lazily and expose `run_optuna_search(config, n_trials_override=None)`.
- Add regression tests for missing Optuna dependency, invalid search space, incompatible head/model values, pruning flag, and timeout handling.

**Implementation steps:**
1. Validate tuning config.
2. Sample only plan-supported hyperparameters.
3. Build trial config by merging sampled values into base training config.
4. Optimize validation average precision.
5. Save trial table and best params without committing SQLite study DB.

**Acceptance criteria:**
- Tuning works in synthetic fast mode.
- Missing optional Optuna gives an actionable install message.
- Best params are CPU-friendly and valid for model construction.

**Edge cases to cover:**
- `d_model=96`, `n_heads=4`.
- `d_model` incompatible with sampled heads.
- Zero completed trials.
- Existing study storage.

**Definition of done:** Hyperparameter tuning workflow is available without brute-force grid search.

---

## TDD-22 - Pseudo-Client Evaluation and FedAvg Simulation

**Goal:** Implement grouped evaluation and FedAvg-style pseudo-client training without true multicentre claims.

**Source plan section or requirement:** Plan section 4 multicentre wording; plan section 10 E2/E3; plan section 11 Blocks F/G.

**Files likely to be created or modified:**
- `src/icu_pretrain/training/federated.py`
- `scripts/run_pseudoclient_eval.py`
- `scripts/run_fedavg_sim.py`
- `tests/test_federated.py`

**Dependencies:** `TDD-18`, `TDD-10`.

**Test-first requirements:**
- Create `tests/test_federated.py`.
- Add failing tests grouping synthetic stays by event density, running one FedAvg round over tiny client models or monkeypatched local updates, and aggregating by number of examples.
- Minimal implementation should add pseudo-client evaluation helper and `run_fedavg_simulation(config)`.
- Add regression tests for empty client, uneven client sizes, `client_fraction` less than 1, incompatible model states, and disclaimer metadata.

**Implementation steps:**
1. Reuse split/group helpers from `splits.py`.
2. Evaluate final model per pseudo-client group.
3. Implement FedAvg rounds: select clients, local train, weighted-average state dicts, central evaluation.
4. Write `results/fedavg/fedavg_metrics.json` and comparison-ready summary rows.

**Acceptance criteria:**
- Output explicitly labels groups as pseudo-clients.
- Aggregation is weighted by number of examples.
- No secure FL, differential privacy, secure aggregation, or true site-generalization claims are made.

**Edge cases to cover:**
- One client only.
- Client with no positive labels.
- Non-floating tensors in state dict.
- Missing grouped split metadata.

**Definition of done:** Non-pooled training awareness is demonstrated through pseudo-client simulation.

---

## TDD-23 - Rare-Pattern Memorisation Probe

**Goal:** Add a lightweight diagnostic for rare-pattern memorisation/privacy concerns.

**Source plan section or requirement:** Plan section 11 Block H; plan section 20 Day 7 output.

**Files likely to be created or modified:**
- `src/icu_pretrain/analysis/memorisation.py`
- `scripts/run_memorisation_probe.py`
- `tests/test_memorisation_probe.py`

**Dependencies:** `TDD-15`, `TDD-18`.

**Test-first requirements:**
- Create `tests/test_memorisation_probe.py`.
- Add failing tests computing rare n-gram overlap and top-k masked prediction overlap on synthetic token sequences.
- Minimal implementation should add deterministic probe functions and JSON output.
- Add regression tests for no rare n-grams, duplicate sequences, top-k larger than vocab, empty test set, and disclaimer field presence.

**Implementation steps:**
1. Count train/test n-grams and identify rare train patterns.
2. Compute overlap diagnostics on held-out streams.
3. Optionally run top-k masked prediction overlap using trained model outputs.
4. Write `results/summary/memorisation_probe.json`.

**Acceptance criteria:**
- Probe is labeled as diagnostic, not a formal privacy audit.
- No patient-level token sequences are written to tracked results.
- Metrics are reproducible on synthetic inputs.

**Edge cases to cover:**
- All n-grams common.
- All n-grams rare.
- Model unavailable.
- Vocabulary mismatch.

**Definition of done:** The repo includes the planned privacy/memorisation sanity check.

---

## TDD-24 - Result Tables, Figures, and Application Polish

**Goal:** Generate polished public summaries and documentation without fabricating metrics.

**Source plan section or requirement:** Plan section 12 README summary table; plan sections 16-19 docs/results; plan section 20 Day 8.

**Files likely to be created or modified:**
- `src/icu_pretrain/analysis/tables.py`
- `src/icu_pretrain/analysis/plots.py`
- `scripts/make_report_assets.py`
- `README.md`
- `docs/model_card.md`
- `docs/data_statement.md`
- `docs/experiment_protocol.md`
- `docs/limitations.md`
- `docs/application_note.md`
- `tests/test_report_assets.py`

**Dependencies:** `TDD-19`, `TDD-20`, `TDD-22`, `TDD-23`.

**Test-first requirements:**
- Create `tests/test_report_assets.py`.
- Add failing tests transforming synthetic `experiment_comparison.csv` into README-ready tables and generating placeholder-safe figure files only from real result rows.
- Minimal implementation should add `make_tables()`, `make_plots()`, and CLI behavior.
- Add regression tests for empty results, pending placeholders, missing optional matplotlib, non-finite metrics, and no raw/patient-level data in reports.

**Implementation steps:**
1. Generate `final_results.csv` and curated comparison tables from result summaries.
2. Generate simple figures: model comparison, pretraining loss, tuning history, FedAvg vs centralized when inputs exist.
3. Update documentation to match plan wording: eICU demo only, no clinical deployment, no true hospital-level generalization.
4. Keep placeholders clearly labeled where experiments have not been run.

**Acceptance criteria:**
- Public-facing docs follow plan structure and disclaimers.
- README result table never invents AUROC/AP values.
- Generated assets are reproducible from scripts/configs.

**Edge cases to cover:**
- Results file with only headers.
- Missing FedAvg metrics.
- Tuning results absent.
- Existing docs with outdated claims.

**Definition of done:** The repository is application-polished while remaining honest about scope and available metrics.

---

## Recommended Execution Order

1. `TDD-00` test harness.
2. `TDD-01` config validation.
3. `TDD-02` event/outcome contracts.
4. `TDD-03` eICU CSV loading.
5. `TDD-04` outcome extraction.
6. `TDD-05` categorical event extraction.
7. `TDD-06` numeric discretization.
8. `TDD-07` full event stream builder.
9. `TDD-08` preparation CLI.
10. `TDD-09` tokenizer.
11. `TDD-10` splits and pseudo-client groups.
12. `TDD-11` encoded dataset.
13. `TDD-12` collators and masking.
14. `TDD-13` Transformer encoder.
15. `TDD-14` prediction heads.
16. `TDD-15` pretraining loop.
17. `TDD-16` shared metrics.
18. `TDD-17` logistic baselines.
19. `TDD-18` fine-tuning.
20. `TDD-19` tracking.
21. `TDD-20` experiment runner.
22. `TDD-21` Optuna tuning.
23. `TDD-22` pseudo-client/FedAvg simulation.
24. `TDD-23` memorisation probe.
25. `TDD-24` reporting and application polish.
