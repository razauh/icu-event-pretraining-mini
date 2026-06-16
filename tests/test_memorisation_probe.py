from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from icu_pretrain.analysis.memorisation import compute_memorisation_summary, run_memorisation_probe
from icu_pretrain.data.dataset import EncodedDataset
from icu_pretrain.models.heads import MaskedEventPredictionHead
from icu_pretrain.models.transformer import ICUTinyTransformer
from icu_pretrain.utils import load_yaml


ROOT = Path(__file__).resolve().parents[1]


def write_processed_dir(root: Path, vocab_size: int = 6) -> None:
    root.mkdir(parents=True, exist_ok=True)
    vocab = {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2, "[CLS]": 3}
    for idx in range(4, vocab_size):
        vocab[f"TOKEN_{idx}"] = idx
    (root / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    train_dir = root / "encoded" / "train"
    val_dir = root / "encoded" / "validation"
    EncodedDataset.write_shard(
        [
            {"patientunitstayid": "train-1", "tokens": [3, 7, 8], "label": 1, "split_name": "train"},
            {"patientunitstayid": "train-2", "tokens": [3, 8, 9], "label": 0, "split_name": "train"},
        ],
        0,
        train_dir,
    )
    EncodedDataset.write_shard(
        [
            {"patientunitstayid": "val-1", "tokens": [3, 7, 8], "label": 1, "split_name": "validation"},
        ],
        0,
        val_dir,
    )


def test_compute_memorisation_summary_counts_rare_overlap_and_frequency_accuracy() -> None:
    train_sequences = [[3, 7], [3, 8]]
    heldout_sequences = [[3, 7], [3, 8], [3, 9]]

    summary = compute_memorisation_summary(
        train_sequences,
        heldout_sequences,
        predict_topk=lambda sequence, mask_index, top_k: [sequence[mask_index]],
        top_k=10,
        ngram_size=2,
        rare_threshold=1,
        vocab_size=6,
    )

    assert summary["status"] == "completed"
    assert summary["effective_top_k"] == 6
    assert summary["training_rare_ngram_count"] == 2
    assert summary["rare_ngram_overlap"] == {"total": 2, "hit_count": 2, "rate": 1.0}
    bins = {entry["frequency_bin"]: entry for entry in summary["masked_accuracy_by_frequency"]}
    assert bins["0"]["total"] == 1
    assert bins["0"]["correct"] == 1
    assert bins["0"]["accuracy"] == 1.0
    assert bins["1"]["total"] == 2
    assert bins["1"]["correct"] == 2
    assert bins["1"]["accuracy"] == 1.0


def test_compute_memorisation_summary_handles_empty_heldout_and_no_model() -> None:
    summary = compute_memorisation_summary(
        [[3, 7], [3, 8]],
        [],
        predict_topk=None,
        top_k=5,
        ngram_size=2,
        rare_threshold=1,
    )

    assert summary["status"] == "empty_heldout"
    assert summary["model_scoring_available"] is False
    assert summary["rare_ngram_overlap"] == {"total": 0, "hit_count": 0, "rate": None}
    assert all(entry["total"] == 0 for entry in summary["masked_accuracy_by_frequency"])


def test_run_memorisation_probe_raises_on_vocabulary_mismatch(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    write_processed_dir(processed_dir, vocab_size=6)

    config = load_yaml(ROOT / "configs/final/eicu_demo_final_tiny.yaml")
    config["data"]["processed_dir"] = str(processed_dir)

    mismatch_vocab_size = 7
    model = ICUTinyTransformer(
        vocab_size=mismatch_vocab_size,
        max_seq_len=256,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    )
    head = MaskedEventPredictionHead(d_model=64, vocab_size=mismatch_vocab_size)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        SimpleNamespace(model_state=model.state_dict(), prediction_head_state=head.state_dict()),
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="vocabulary mismatch"):
        run_memorisation_probe(
            config=config,
            processed_dir=processed_dir,
            checkpoint_path=checkpoint_path,
            heldout_split="validation",
        )
