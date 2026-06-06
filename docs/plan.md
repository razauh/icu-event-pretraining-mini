Below is the **complete updated plan** for the repo, using your final decisions:

```text
Dataset: eICU demo only
Repo style: application-polished, but backed by multiple controlled experiments
Model: tiny custom PyTorch Transformer encoder
Objective: masked ICU event modelling
Hardware target: CPU, 8 GB RAM
Multi-location: no true multicentre claim
Replacement: pseudo-client grouped evaluation + FedAvg-style simulation
Experiment design: controlled matrix + small Optuna tuning, not brute-force full permutation
```

The eICU demo is a good fit because it contains data for **over 2,500 ICU unit stays selected from 20 larger hospitals**, but the demo removes hospital and unit identifiers, so your repo must not claim real hospital-level or multicentre generalisation. ([PhysioNet][1])

---

# 1. Repo name and final positioning

Recommended repo name:

```text
icu-event-pretraining-mini
```

Recommended subtitle:

```text
A CPU-friendly PyTorch prototype for masked pretraining on heterogeneous ICU event streams using the eICU demo dataset.
```

Core positioning:

```text
This repository demonstrates how heterogeneous ICU records from the eICU demo dataset can be converted into chronological event streams and used to pretrain a compact Transformer encoder with masked event modelling. It includes controlled ablation experiments, automated hyperparameter tuning, downstream outcome prediction, pseudo-client grouped evaluation, and a small FedAvg-style simulation.

The repository does not claim to train a clinical foundation model. It also does not claim true hospital-level multicentre generalisation, because hospital and unit identifiers are removed from the eICU demo dataset.
```

This is the right balance: it looks **scientifically serious**, but it avoids overclaiming.

---

# 2. Main project objective

The repo should prove that you can do the following:

```text
eICU demo CSV tables
↓
patient-level ICU event streams
↓
event tokenisation
↓
masked event pretraining
↓
downstream outcome prediction
↓
controlled experiment comparison
↓
pseudo-client / FedAvg-style simulation
↓
application-polished README and result summary
```

The PhD project is about ICU event-stream foundation modelling. Your repo should therefore focus on **event construction, representation learning, pretraining, evaluation, and non-pooled training constraints**, not on building a huge model.

---

# 3. Dataset plan

Use **only eICU demo**.

The eICU Collaborative Research Database includes ICU-relevant information such as vital signs, care-plan documentation, severity-of-illness measures, diagnosis information, and treatment information. ([PhysioNet][1]) The demo is appropriate for your laptop because it is much smaller than the full eICU database, while still containing heterogeneous ICU data.

## Tables to use

Use these tables in the MVP:

```text
patient
apachePatientResult
diagnosis
lab
medication
infusionDrug
treatment
vitalPeriodic
vitalAperiodic
```

Why these tables?

```text
patient                → static/context/outcome fields
apachePatientResult    → severity / outcome-related information
diagnosis              → diagnostic event tokens
lab                    → laboratory event tokens
medication             → medication event tokens
infusionDrug           → infusion/drug event tokens
treatment              → treatment/procedure event tokens
vitalPeriodic          → periodic vital-sign event tokens
vitalAperiodic         → aperiodic vital-sign event tokens
```

Do **not** upload raw eICU data into GitHub. The repo should contain code, configs, docs, and small generated sample outputs only.

---

# 4. What to do about multicentre / multi-location

Do not skip the topic completely, because the PhD description cares about non-pooled and multicentre ICU learning. But do not claim true multicentre evaluation.

Use this wording:

```text
Because hospital and unit identifiers are removed from the eICU demo dataset, this repository does not perform true hospital-level generalisation. Instead, it implements pseudo-client grouped evaluation and FedAvg-style simulation to demonstrate how the pipeline could support grouped or non-pooled learning when real site identifiers are available.
```

This is accurate because the demo removes hospital and unit identifiers. ([PhysioNet][1])

---

# 5. Event-stream representation

Each `patientunitstayid` becomes one chronological event sequence.

Example:

```text
STATIC::AGE_BIN::60_70
STATIC::GENDER::M
STATIC::ADMISSION_SOURCE::EMERGENCY
DX::RESPIRATORY_FAILURE
LAB::CREATININE::HIGH
TIME_GAP::1H_3H
VITAL::HEARTRATE::90_110
MED::NOREPINEPHRINE
TREATMENT::MECHANICAL_VENTILATION
```

## Event categories

Use these token families:

```text
STATIC::*       → age bin, gender, admission source, unit type if available
DX::*           → diagnosis tokens
LAB::*          → discretised lab results
VITAL::*        → discretised vital signs
MED::*          → medication tokens
INFUSION::*     → infusion drug tokens
TREATMENT::*    → treatment/procedure tokens
TIME_GAP::*     → time interval between events
```

## Numeric value handling

For labs and vitals, do not feed raw continuous values directly into the Transformer. Use discretised bins.

Recommended first version:

```text
BIN_LOW
BIN_NORMAL
BIN_HIGH
BIN_VERY_HIGH
```

or quantile-style bins:

```text
Q1
Q2
Q3
Q4
```

For a one-week implementation, quantile bins are easier. For a polished README, you can describe them as:

```text
Continuous ICU measurements are discretised using training-set quantile bins to produce compact event tokens suitable for CPU-friendly masked event modelling.
```

---

# 6. Model plan

Use a tiny custom Transformer encoder implemented in PyTorch.

PyTorch provides `TransformerEncoderLayer`, which is suitable for building an encoder stack from self-attention and feedforward layers. ([PyTorch Documentation][2])

## Model name

```text
ICU-TinyTransformer
```

## Architecture

```text
event token IDs
   ↓
event embeddings
   +
position embeddings
   +
event-type/time-gap embeddings
   ↓
tiny Transformer encoder
   ↓
masked event prediction head
   ↓
optional downstream outcome head
```

## Final CPU-friendly config

```yaml
model:
  name: icu_tiny_transformer
  max_seq_len: 128
  d_model: 64
  n_heads: 4
  n_layers: 2
  dim_feedforward: 256
  dropout: 0.1

pretraining:
  objective: masked_event_modeling
  mask_probability: 0.15
  batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 0.0005
  weight_decay: 0.01
  epochs: 5
  device: cpu
```

This should be your final recommended config unless tuning clearly finds a better CPU-friendly setting.

---

# 7. Main pretraining objective

Use **masked event modelling**.

Original sequence:

```text
STATIC::AGE_BIN::60_70
DX::SEPSIS
LAB::CREATININE::HIGH
TIME_GAP::1H_3H
VITAL::HEARTRATE::90_110
```

Masked sequence:

```text
STATIC::AGE_BIN::60_70
DX::SEPSIS
[MASK]
TIME_GAP::1H_3H
VITAL::HEARTRATE::90_110
```

Target:

```text
LAB::CREATININE::HIGH
```

Loss:

```text
CrossEntropyLoss over masked positions only
```

This is the core of the repo. Everything else should support this.

---

# 8. Downstream task

Use one simple outcome-prediction task.

Recommended primary task:

```text
hospital mortality / discharge mortality-style outcome
```

Fallback:

```text
ICU mortality or binary discharge-status style label, depending on available parsed fields.
```

Metrics:

```text
AUROC
Average Precision
F1
Balanced Accuracy
```

Use **AUROC** and **Average Precision** as the main metrics. Scikit-learn’s `roc_auc_score` computes area under the ROC curve from prediction scores, and `average_precision_score` summarizes the precision-recall curve as a weighted mean of precisions at each threshold. ([scikit-learn.org][3])

---

# 9. Recommended experiment strategy

Do not run every possible full permutation. That would become too large and messy.

Use a **controlled permutation-inspired matrix**:

```text
Stage 1: Representation ablations
Stage 2: Pretraining vs no-pretraining comparison
Stage 3: Small Optuna tuning
Stage 4: Final verification runs
Stage 5: Pseudo-client and FedAvg-style simulation
Stage 6: Rare-pattern memorisation probe
```

Optuna is a good fit because it is designed for automatic hyperparameter optimization and supports dynamic search spaces through a define-by-run API. ([optuna.readthedocs.io][4])

---

# 10. Recommended experiment matrix

## Representation variants

```text
R1 = basic clinical events
     DX + LAB + VITAL + MED + INFUSION + TREATMENT

R2 = basic clinical events + time gaps
     R1 + TIME_GAP tokens

R3 = basic clinical events + time gaps + static context
     R2 + age bin + gender + admission/admit-source context
```

## Model variants

```text
M0 = bag-of-events logistic regression
M1 = tiny Transformer trained from scratch
M2 = tiny Transformer with masked event pretraining
M3 = tuned tiny Transformer with masked event pretraining
```

## Evaluation variants

```text
E1 = random split
E2 = pseudo-client grouped split
E3 = FedAvg-style pseudo-client simulation
```

## Pseudo-client split strategies

```text
S1 = event_density
     Group stays by number of events: low, medium, high.

S2 = diagnosis_group
     Group stays by broad diagnosis category where available.

S3 = admission_context
     Group stays by available admission/admit-source context where available.
```

Default pseudo-client split:

```text
event_density
```

Reason: it is always available after event-stream construction.

---

# 11. Concrete experiment run list

This is the run list I recommend. It is systematic but still realistic for CPU.

## Block A — Baselines

```text
EXP-00
Name: logistic_basic_random
Representation: R1
Model: M0 bag-of-events logistic regression
Training: supervised only
Evaluation: E1 random split
Purpose: simple non-neural baseline

EXP-01
Name: logistic_timegap_random
Representation: R2
Model: M0 bag-of-events logistic regression
Training: supervised only
Evaluation: E1 random split
Purpose: check whether time-gap tokens help even in count-based baseline
```

## Block B — Transformer from scratch

```text
EXP-02
Name: scratch_basic_random
Representation: R1
Model: M1 tiny Transformer
Training: supervised outcome training only
Evaluation: E1 random split
Purpose: neural baseline without pretraining

EXP-03
Name: scratch_timegap_random
Representation: R2
Model: M1 tiny Transformer
Training: supervised outcome training only
Evaluation: E1 random split
Purpose: test time-gap value without pretraining

EXP-04
Name: scratch_timegap_static_random
Representation: R3
Model: M1 tiny Transformer
Training: supervised outcome training only
Evaluation: E1 random split
Purpose: test static context without pretraining
```

## Block C — Masked pretraining

```text
EXP-05
Name: pretrain_basic_random
Representation: R1
Model: M2 tiny Transformer
Training: masked event pretraining + fine-tuning
Evaluation: E1 random split
Purpose: test pretraining with basic events

EXP-06
Name: pretrain_timegap_random
Representation: R2
Model: M2 tiny Transformer
Training: masked event pretraining + fine-tuning
Evaluation: E1 random split
Purpose: test pretraining with time-gap-aware streams

EXP-07
Name: pretrain_timegap_static_random
Representation: R3
Model: M2 tiny Transformer
Training: masked event pretraining + fine-tuning
Evaluation: E1 random split
Purpose: test full representation before tuning
```

## Block D — Automated tuning

```text
EXP-08
Name: optuna_tiny_search
Representation: best of R1/R2/R3 from EXP-05 to EXP-07
Model: M2/M3
Training: masked event pretraining + fine-tuning
Evaluation: validation AUROC / Average Precision
Purpose: find best CPU-friendly hyperparameters
Trials: 10–20
Epochs per trial: 1–3
```

Tuning search space:

```yaml
search_space:
  d_model: [32, 64, 96, 128]
  n_layers: [1, 2, 3]
  n_heads: [2, 4]
  dim_feedforward: [128, 256, 512]
  dropout: [0.0, 0.1, 0.2]
  learning_rate:
    type: loguniform
    low: 0.0001
    high: 0.001
  mask_probability: [0.10, 0.15, 0.20]
  max_seq_len: [64, 128, 256]
```

## Block E — Final polished model

```text
EXP-09
Name: final_tuned_random
Representation: best representation from earlier experiments
Model: M3 tuned tiny Transformer
Training: masked event pretraining + fine-tuning
Evaluation: E1 random split
Purpose: final main reported model
```

## Block F — Pseudo-client grouped evaluation

```text
EXP-10
Name: final_tuned_pseudoclient_event_density
Representation: best representation
Model: M3 tuned tiny Transformer
Training: masked event pretraining + fine-tuning
Evaluation: E2 pseudo-client split using event_density
Purpose: grouped evaluation without claiming true site generalisation

EXP-11
Name: final_tuned_pseudoclient_diagnosis
Representation: best representation
Model: M3 tuned tiny Transformer
Training: masked event pretraining + fine-tuning
Evaluation: E2 pseudo-client split using diagnosis_group
Purpose: optional second grouped split if diagnosis grouping is reliable
```

## Block G — FedAvg-style simulation

```text
EXP-12
Name: fedavg_event_density_sim
Representation: best representation
Model: M3 tuned tiny Transformer
Training: FedAvg-style pseudo-client simulation
Evaluation: pseudo-client held-out evaluation
Purpose: demonstrate non-pooled training awareness
```

FedAvg config:

```yaml
fedavg:
  split_strategy: event_density
  clients: 3
  rounds: 3
  local_epochs: 1
  client_fraction: 1.0
  aggregation: weighted_average_by_num_examples
```

README disclaimer:

```text
This is a FedAvg-style simulation over pseudo-clients, not a secure federated-learning deployment. It does not implement secure aggregation, differential privacy, or cross-institution communication.
```

## Block H — Rare-pattern memorisation probe

```text
EXP-13
Name: rare_pattern_memorisation_probe
Representation: best representation
Model: final tuned pretrained Transformer
Evaluation: rare n-gram overlap / top-k masked prediction overlap
Purpose: lightweight diagnostic linked to memorisation/privacy concerns
```

README disclaimer:

```text
This probe is a diagnostic sanity check for rare-pattern memorisation. It is not a formal privacy audit and does not provide privacy guarantees.
```

---

# 12. Final experiment summary table for README

Your README should contain a polished table like this:

```text
| ID | Representation | Model | Pretraining | Evaluation | AUROC | Avg Precision | Notes |
|---|---|---|---|---|---:|---:|---|
| EXP-00 | Basic events | Logistic regression | No | Random | x.xx | x.xx | Bag-of-events baseline |
| EXP-01 | Events + time gaps | Logistic regression | No | Random | x.xx | x.xx | Time-gap baseline |
| EXP-02 | Basic events | Tiny Transformer | No | Random | x.xx | x.xx | Scratch neural baseline |
| EXP-03 | Events + time gaps | Tiny Transformer | No | Random | x.xx | x.xx | Time-aware scratch model |
| EXP-04 | Events + time gaps + static | Tiny Transformer | No | Random | x.xx | x.xx | Full scratch representation |
| EXP-05 | Basic events | Tiny Transformer | Yes | Random | x.xx | x.xx | Masked pretraining |
| EXP-06 | Events + time gaps | Tiny Transformer | Yes | Random | x.xx | x.xx | Time-aware pretraining |
| EXP-07 | Events + time gaps + static | Tiny Transformer | Yes | Random | x.xx | x.xx | Full pretraining representation |
| EXP-09 | Best representation | Tuned Tiny Transformer | Yes | Random | x.xx | x.xx | Final selected config |
| EXP-10 | Best representation | Tuned Tiny Transformer | Yes | Pseudo-client | x.xx | x.xx | Not true site split |
| EXP-12 | Best representation | FedAvg simulation | Yes | Pseudo-client | x.xx | x.xx | Simulated clients |
```

Do **not** over-emphasize the numbers. Emphasize the experimental workflow.

---

# 13. Repo structure

Use this final structure:

```text
icu-event-pretraining-mini/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── final/
│   │   └── eicu_demo_final_tiny.yaml
│   ├── experiments/
│   │   ├── exp_00_logistic_basic.yaml
│   │   ├── exp_01_logistic_timegap.yaml
│   │   ├── exp_02_scratch_basic.yaml
│   │   ├── exp_03_scratch_timegap.yaml
│   │   ├── exp_04_scratch_timegap_static.yaml
│   │   ├── exp_05_pretrain_basic.yaml
│   │   ├── exp_06_pretrain_timegap.yaml
│   │   ├── exp_07_pretrain_timegap_static.yaml
│   │   └── exp_suite_core.yaml
│   ├── tuning/
│   │   └── optuna_eicu_tiny.yaml
│   └── fedavg/
│       └── eicu_demo_fedavg_sim.yaml
├── data/
│   ├── raw/                  # ignored
│   ├── processed/            # ignored
│   └── README.md
├── src/
│   └── icu_pretrain/
│       ├── __init__.py
│       ├── constants.py
│       ├── utils.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── eicu_demo.py
│       │   ├── eicu_event_builder.py
│       │   ├── tokenizer.py
│       │   ├── dataset.py
│       │   ├── collate.py
│       │   └── splits.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── transformer.py
│       │   └── heads.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── pretrain.py
│       │   ├── finetune.py
│       │   ├── evaluate.py
│       │   ├── baselines.py
│       │   └── federated.py
│       ├── experiments/
│       │   ├── __init__.py
│       │   ├── runner.py
│       │   ├── registry.py
│       │   └── tracking.py
│       ├── tuning/
│       │   ├── __init__.py
│       │   └── optuna_search.py
│       └── analysis/
│           ├── __init__.py
│           ├── plots.py
│           ├── tables.py
│           └── memorisation.py
├── scripts/
│   ├── prepare_eicu_demo.py
│   ├── pretrain.py
│   ├── finetune_outcome.py
│   ├── run_baselines.py
│   ├── run_experiment.py
│   ├── run_experiment_suite.py
│   ├── run_optuna_tuning.py
│   ├── run_pseudoclient_eval.py
│   ├── run_fedavg_sim.py
│   ├── run_memorisation_probe.py
│   └── make_report_assets.py
├── results/
│   ├── README.md
│   ├── summary/
│   │   ├── final_results.csv
│   │   ├── experiment_comparison.csv
│   │   └── best_config.json
│   ├── tuning/
│   │   ├── best_params.json
│   │   └── trials.csv
│   ├── fedavg/
│   │   └── fedavg_metrics.json
│   └── figures/
│       ├── architecture.png
│       ├── pretraining_loss.png
│       ├── tuning_history.png
│       ├── model_comparison.png
│       └── fedavg_vs_centralised.png
├── notebooks/
│   └── 01_eicu_event_stream_exploration.ipynb
└── docs/
    ├── model_card.md
    ├── data_statement.md
    ├── experiment_protocol.md
    ├── limitations.md
    └── application_note.md
```

---

# 14. Config design

## Final config

```yaml
experiment:
  name: final_tuned_random
  seed: 42

data:
  dataset: eicu_demo
  raw_dir: data/raw/eicu_demo
  processed_dir: data/processed/eicu_demo
  representation: timegap_static
  max_seq_len: 128
  min_events_per_stay: 5

model:
  type: icu_tiny_transformer
  d_model: 64
  n_heads: 4
  n_layers: 2
  dim_feedforward: 256
  dropout: 0.1

pretraining:
  enabled: true
  objective: masked_event_modeling
  mask_probability: 0.15
  epochs: 5
  batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 0.0005
  weight_decay: 0.01

finetuning:
  task: mortality
  epochs: 5
  batch_size: 8
  learning_rate: 0.0003
  freeze_encoder: false

evaluation:
  split: random
  metrics:
    - auroc
    - average_precision
    - f1
    - balanced_accuracy

runtime:
  device: cpu
  num_workers: 0
```

## Tuning config

```yaml
study:
  name: optuna_eicu_tiny
  storage: sqlite:///results/tuning/optuna_study.db
  direction: maximize
  metric: val_average_precision
  n_trials: 15
  timeout_minutes: null

search_space:
  d_model: [32, 64, 96, 128]
  n_layers: [1, 2, 3]
  n_heads: [2, 4]
  dim_feedforward: [128, 256, 512]
  dropout: [0.0, 0.1, 0.2]
  learning_rate:
    type: loguniform
    low: 0.0001
    high: 0.001
  mask_probability: [0.10, 0.15, 0.20]
  max_seq_len: [64, 128, 256]

trial_training:
  pretrain_epochs: 2
  finetune_epochs: 2
  subset_fraction: 0.5
  pruning: true
```

---

# 15. Scripts and commands

## Basic polished run

```bash
python scripts/prepare_eicu_demo.py \
  --raw_dir data/raw/eicu_demo \
  --out_dir data/processed/eicu_demo

python scripts/pretrain.py \
  --config configs/final/eicu_demo_final_tiny.yaml

python scripts/finetune_outcome.py \
  --config configs/final/eicu_demo_final_tiny.yaml

python scripts/make_report_assets.py
```

## Experiment suite

```bash
python scripts/run_experiment_suite.py \
  --suite configs/experiments/exp_suite_core.yaml
```

## Tuning

```bash
python scripts/run_optuna_tuning.py \
  --config configs/tuning/optuna_eicu_tiny.yaml \
  --n_trials 15
```

## Pseudo-client evaluation

```bash
python scripts/run_pseudoclient_eval.py \
  --config configs/final/eicu_demo_final_tiny.yaml \
  --split_strategy event_density
```

## FedAvg-style simulation

```bash
python scripts/run_fedavg_sim.py \
  --config configs/fedavg/eicu_demo_fedavg_sim.yaml
```

## Memorisation probe

```bash
python scripts/run_memorisation_probe.py \
  --config configs/final/eicu_demo_final_tiny.yaml
```

---

# 16. README structure

Use this exact README layout:

```text
# ICU Event Pretraining Mini

## Motivation

## What this repository demonstrates

## Why eICU demo?

## Scope and limitations

## Dataset

## Event-stream construction

## Model architecture

## Pretraining objective

## Experiment design

## Recommended experiment matrix

## Automated hyperparameter tuning

## Downstream evaluation

## Pseudo-client grouped evaluation

## FedAvg-style simulation

## Results

## How to run

## Repository structure

## Model card

## Data and citation

## Author contribution
```

The most important README sections are:

```text
What this repository demonstrates
Scope and limitations
Experiment design
Results
Author contribution
```

---

# 17. `docs/experiment_protocol.md`

This file should explain the experiment matrix in a professional way.

Suggested text:

```text
The experiment design follows a staged strategy rather than a full exhaustive grid. First, a compact ablation matrix evaluates event representation choices and the effect of masked event pretraining. Second, Optuna is used to search over a constrained CPU-friendly hyperparameter space. Third, the selected configuration is used for final random-split, pseudo-client, FedAvg-style, and rare-pattern memorisation evaluations.

This avoids an infeasible full factorial search while preserving systematic comparison.
```

---

# 18. `docs/application_note.md`

This is useful because the job specifically asks for one relevant GitHub repository and your exact contribution.

Suggested text:

```text
This repository was developed as a focused technical prototype for ICU event-stream pretraining. It demonstrates my ability to parse heterogeneous ICU data tables, construct patient-level chronological event streams, design compact token representations for irregular clinical events, implement masked event modelling with a small Transformer encoder, compare pretrained and non-pretrained models, run controlled experiments and automated hyperparameter tuning, evaluate downstream outcome prediction, and simulate grouped/pseudo-client learning without claiming true hospital-level generalisation.

The project is intentionally CPU-friendly and uses the open eICU demo dataset. It is not a clinical deployment system and does not make clinical claims.
```

---

# 19. What to include in final results

Your `results/summary/experiment_comparison.csv` should include:

```text
experiment_id
experiment_name
representation
model
pretraining
split_strategy
auroc
average_precision
f1
balanced_accuracy
pretrain_val_loss
finetune_val_loss
num_parameters
max_seq_len
runtime_minutes
notes
```

Your `results/summary/best_config.json` should include:

```json
{
  "selected_experiment": "EXP-09",
  "selection_metric": "validation_average_precision",
  "representation": "timegap_static",
  "model": "icu_tiny_transformer",
  "pretraining": "masked_event_modeling",
  "reason": "Best CPU-friendly validation performance among controlled experiments."
}
```

Your README should show only the polished summary table, not every raw log.

---

# 20. Development schedule

## Day 1 — eICU parser and event stream builder

Deliverables:

```text
src/icu_pretrain/data/eicu_demo.py
src/icu_pretrain/data/eicu_event_builder.py
scripts/prepare_eicu_demo.py
data/README.md
```

Output:

```text
data/processed/eicu_demo/event_streams.jsonl
data/processed/eicu_demo/outcomes.csv
data/processed/eicu_demo/event_stats.json
```

## Day 2 — Tokenizer, datasets, splits

Deliverables:

```text
tokenizer.py
dataset.py
collate.py
splits.py
```

Output:

```text
vocab.json
encoded_train.pt
encoded_val.pt
encoded_test.pt
split_metadata.json
```

## Day 3 — Model and pretraining

Deliverables:

```text
models/transformer.py
models/heads.py
training/pretrain.py
scripts/pretrain.py
```

Output:

```text
results/figures/pretraining_loss.png
```

## Day 4 — Fine-tuning and baselines

Deliverables:

```text
training/finetune.py
training/evaluate.py
training/baselines.py
scripts/run_baselines.py
scripts/finetune_outcome.py
```

Output:

```text
results/summary/baseline_results.csv
```

## Day 5 — Experiment runner and core matrix

Deliverables:

```text
experiments/runner.py
experiments/registry.py
experiments/tracking.py
scripts/run_experiment.py
scripts/run_experiment_suite.py
```

Output:

```text
results/summary/experiment_comparison.csv
```

## Day 6 — Optuna tuning

Deliverables:

```text
tuning/optuna_search.py
scripts/run_optuna_tuning.py
configs/tuning/optuna_eicu_tiny.yaml
```

Output:

```text
results/tuning/trials.csv
results/tuning/best_params.json
results/figures/tuning_history.png
```

## Day 7 — Pseudo-client, FedAvg, memorisation probe

Deliverables:

```text
training/federated.py
analysis/memorisation.py
scripts/run_pseudoclient_eval.py
scripts/run_fedavg_sim.py
scripts/run_memorisation_probe.py
```

Output:

```text
results/fedavg/fedavg_metrics.json
results/figures/fedavg_vs_centralised.png
results/summary/memorisation_probe.json
```

## Day 8 — Application polish

Deliverables:

```text
README.md
docs/model_card.md
docs/data_statement.md
docs/experiment_protocol.md
docs/limitations.md
docs/application_note.md
results/summary/final_results.csv
```

This last day is critical. The repo must look like a polished submission, not an unfinished experiment folder.

---

# 21. Non-goals

Explicitly avoid these:

```text
No full eICU dependency.
No MIMIC dependency.
No true multicentre claim.
No clinical deployment claim.
No large pretrained LLM.
No raw data committed to GitHub.
No wandb requirement.
No huge model checkpoints committed.
No exhaustive brute-force grid over hundreds of runs.
```

---

# 22. Final application framing

When sending the repo, describe it like this:

```text
I am sharing one relevant GitHub repository that I built as a small-scale technical prototype for this application. The repository uses the eICU demo dataset to construct heterogeneous ICU event streams and pretrain a compact Transformer encoder using masked event modelling. It includes controlled ablation experiments, automated hyperparameter tuning, downstream outcome prediction, pseudo-client grouped evaluation, and a small FedAvg-style simulation. Since the eICU demo removes hospital and unit identifiers, I explicitly avoid claiming true hospital-level multicentre generalisation. My goal was to demonstrate my understanding of ICU event representation, pretraining, evaluation, and non-pooled training constraints under limited compute.
```

That is the final direction I recommend.

[1]: https://physionet.org/content/eicu-crd-demo/?utm_source=chatgpt.com "eICU Collaborative Research Database Demo v2.0.1"
[2]: https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html?utm_source=chatgpt.com "TransformerEncoderLayer — PyTorch 2.12 documentation"
[3]: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html?utm_source=chatgpt.com "roc_auc_score — scikit-learn 1.9.0 documentation"
[4]: https://optuna.readthedocs.io/?utm_source=chatgpt.com "Optuna: A hyperparameter optimization framework — Optuna ..."

