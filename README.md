# ICU Event Pretraining Mini

A CPU-friendly PyTorch prototype for masked pretraining on heterogeneous ICU
event streams using the eICU demo dataset.

This repository is being scaffolded from the project plan in `docs/plan.md`.
It intentionally avoids committing raw eICU data, processed patient streams,
large model checkpoints, and local experiment state.

## Scope

- eICU demo dataset only.
- Compact Transformer encoder for masked event modelling.
- Controlled ablations, downstream outcome prediction, pseudo-client grouped
  evaluation, Optuna tuning, and FedAvg-style simulation.
- No true hospital-level generalisation claim, because the eICU demo removes
  hospital and unit identifiers.

