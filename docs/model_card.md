# Model Card

## Overview

ICU-TinyTransformer is a compact Transformer encoder for masked event
pretraining on the eICU demo v2.0.1 dataset. It is a CPU-friendly research
prototype, not a clinical system.

## Training Data

The verified demo cohort contains 2,520 raw ICU stays from 1,841 patients
across 186 hospitals and 292 wards. Raw and processed patient-level data stay
local and are not committed to this repository.

## Intended Use

The model is intended for exploratory representation learning, controlled
downstream benchmarking, patient-grouped evaluation, hospital-grouped
evaluation, and simulated-client FedAvg analysis on the eICU demo only.

## Metrics

Public results report AUROC, average precision, F1, and balanced accuracy.
Selected thresholds are chosen from validation data only.

## Limitations

The prototype does not claim clinical readiness, external hospital
generalisation, secure federated learning, or privacy guarantees. Performance
should be interpreted as exploratory on a small, selectively sampled demo
dataset.

## Ethical Considerations

The repository avoids patient-level public artifacts and keeps raw data local.
Public reporting stays at the aggregate cohort, experiment, fold, and client
levels.
