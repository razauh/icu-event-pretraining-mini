# Graph Report - icu-event-pretraining-mini  (2026-06-06)

## Corpus Check
- 39 files · ~4,832 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 113 nodes · 74 edges · 36 communities detected
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]

## God Nodes (most connected - your core abstractions)
1. `ICUEventDataset` - 3 edges
2. `load_yaml()` - 2 edges
3. `run_finetuning()` - 2 edges
4. `run_fedavg_simulation()` - 2 edges
5. `run_baselines()` - 2 edges
6. `evaluate_predictions()` - 2 edges
7. `run_pretraining()` - 2 edges
8. `run_memorisation_probe()` - 2 edges
9. `make_plots()` - 2 edges
10. `make_tables()` - 2 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Communities

### Community 0 - "Community 0"
Cohesion: 0.4
Nodes (3): ICUEventDataset, PyTorch dataset definitions for encoded ICU event streams., Placeholder dataset for encoded event streams.

### Community 1 - "Community 1"
Cohesion: 0.5
Nodes (3): load_yaml(), General utility helpers., Load a YAML file as a dictionary.

### Community 2 - "Community 2"
Cohesion: 0.5
Nodes (3): Outcome fine-tuning loop., Placeholder fine-tuning entry point., run_finetuning()

### Community 3 - "Community 3"
Cohesion: 0.5
Nodes (3): FedAvg-style pseudo-client simulation., Placeholder FedAvg simulation entry point., run_fedavg_simulation()

### Community 4 - "Community 4"
Cohesion: 0.5
Nodes (3): Bag-of-events baseline models., Placeholder baseline entry point., run_baselines()

### Community 5 - "Community 5"
Cohesion: 0.5
Nodes (3): evaluate_predictions(), Evaluation metrics for downstream outcome prediction., Placeholder evaluation entry point.

### Community 6 - "Community 6"
Cohesion: 0.5
Nodes (3): Masked event pretraining loop., Placeholder pretraining entry point., run_pretraining()

### Community 7 - "Community 7"
Cohesion: 0.5
Nodes (3): Rare-pattern memorisation probe., Placeholder memorisation probe entry point., run_memorisation_probe()

### Community 8 - "Community 8"
Cohesion: 0.5
Nodes (3): make_plots(), Plot generation helpers., Placeholder plot generation entry point.

### Community 9 - "Community 9"
Cohesion: 0.5
Nodes (3): make_tables(), Result table generation helpers., Placeholder table generation entry point.

### Community 10 - "Community 10"
Cohesion: 0.5
Nodes (3): collate_event_batch(), Batch collation for ICU event sequences., Placeholder collate function.

### Community 11 - "Community 11"
Cohesion: 0.5
Nodes (3): make_random_split(), Random and pseudo-client split helpers., Placeholder random split helper.

### Community 12 - "Community 12"
Cohesion: 0.5
Nodes (3): eICU demo table loading utilities., Return the expected CSV path for an eICU demo table., table_path()

### Community 13 - "Community 13"
Cohesion: 0.5
Nodes (3): build_event_streams(), Build chronological ICU event streams from eICU demo tables., Placeholder for the event stream builder implementation.

### Community 14 - "Community 14"
Cohesion: 0.5
Nodes (3): Optuna search for CPU-friendly Transformer configs., Placeholder Optuna search entry point., run_optuna_search()

### Community 15 - "Community 15"
Cohesion: 0.5
Nodes (3): ICUTinyTransformer, Tiny Transformer encoder for ICU event streams., Placeholder for the PyTorch Transformer encoder.

### Community 16 - "Community 16"
Cohesion: 0.5
Nodes (3): MaskedEventPredictionHead, Prediction heads for pretraining and downstream tasks., Placeholder masked event prediction head.

### Community 17 - "Community 17"
Cohesion: 0.5
Nodes (3): Local experiment tracking helpers., Placeholder metrics recording helper., record_metrics()

### Community 18 - "Community 18"
Cohesion: 0.67
Nodes (2): Placeholder experiment runner., run_experiment()

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (1): ICU event pretraining mini package.

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (1): Shared constants for ICU event pretraining.

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (1): Analysis and reporting helpers.

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (1): Data loading and event stream construction.

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (1): Hyperparameter tuning.

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (1): Experiment orchestration.

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): Run bag-of-events baselines.

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): Fine-tune the encoder for outcome prediction.

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): Run FedAvg-style pseudo-client simulation.

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): Run the core experiment suite.

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): Run masked event pretraining.

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): Run rare-pattern memorisation diagnostics.

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Run a single configured experiment.

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): Run pseudo-client grouped evaluation.

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): Generate README/report tables and figures.

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): Prepare eICU demo tables into patient-level event streams.

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): Run Optuna hyperparameter tuning.

## Knowledge Gaps
- **54 isolated node(s):** `ICU event pretraining mini package.`, `Shared constants for ICU event pretraining.`, `General utility helpers.`, `Load a YAML file as a dictionary.`, `Outcome fine-tuning loop.` (+49 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 18`** (3 nodes): `Placeholder experiment runner.`, `run_experiment()`, `runner.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (2 nodes): `ICU event pretraining mini package.`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (2 nodes): `Shared constants for ICU event pretraining.`, `constants.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (2 nodes): `Analysis and reporting helpers.`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (2 nodes): `Data loading and event stream construction.`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (2 nodes): `__init__.py`, `Hyperparameter tuning.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (2 nodes): `Experiment orchestration.`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (2 nodes): `run_baselines.py`, `Run bag-of-events baselines.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (2 nodes): `finetune_outcome.py`, `Fine-tune the encoder for outcome prediction.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (2 nodes): `run_fedavg_sim.py`, `Run FedAvg-style pseudo-client simulation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (2 nodes): `run_experiment_suite.py`, `Run the core experiment suite.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (2 nodes): `pretrain.py`, `Run masked event pretraining.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (2 nodes): `run_memorisation_probe.py`, `Run rare-pattern memorisation diagnostics.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (2 nodes): `run_experiment.py`, `Run a single configured experiment.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (2 nodes): `run_pseudoclient_eval.py`, `Run pseudo-client grouped evaluation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (2 nodes): `make_report_assets.py`, `Generate README/report tables and figures.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (2 nodes): `prepare_eicu_demo.py`, `Prepare eICU demo tables into patient-level event streams.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (2 nodes): `run_optuna_tuning.py`, `Run Optuna hyperparameter tuning.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What connects `ICU event pretraining mini package.`, `Shared constants for ICU event pretraining.`, `General utility helpers.` to the rest of the system?**
  _54 weakly-connected nodes found - possible documentation gaps or missing edges._