"""Shared constants for ICU event pretraining."""

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[MASK]", "[CLS]"]

EVENT_FAMILIES = [
    "STATIC",
    "DX",
    "LAB",
    "VITAL",
    "MED",
    "INFUSION",
    "TREATMENT",
    "TIME_GAP",
]

EVENT_REPRESENTATIONS = ["basic", "timegap", "timegap_static"]

SPLIT_NAMES = ["train", "validation", "test"]

MANIFEST_STATUSES = ["running", "complete", "failed"]

RUN_STATUSES = ["running", "completed", "failed", "interrupted"]

CONTRACT_SCHEMA_VERSION = 1

ARTIFACT_HASH_KEYS = [
    "config",
    "vocabulary",
    "split",
    "preprocessing",
    "encoded_dataset",
]
