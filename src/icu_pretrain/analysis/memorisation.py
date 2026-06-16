from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from icu_pretrain.data.dataset import EncodedDataset
from icu_pretrain.models.heads import MaskedEventPredictionHead
from icu_pretrain.models.transformer import ICUTinyTransformer
from icu_pretrain.utils import load_yaml, validate_final_config


SequenceTokens = Sequence[int]
PredictTopK = Callable[[SequenceTokens, int, int], Sequence[int]]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_sequences(sequences: Sequence[SequenceTokens]) -> list[list[int]]:
    normalized: list[list[int]] = []
    for sequence in sequences:
        normalized.append([int(token) for token in sequence])
    return normalized


def _count_token_frequencies(sequences: Sequence[SequenceTokens]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for sequence in sequences:
        counts.update(int(token) for token in sequence)
    return counts


def _count_ngrams(sequences: Sequence[SequenceTokens], n: int) -> Counter[tuple[int, ...]]:
    counts: Counter[tuple[int, ...]] = Counter()
    if n <= 0:
        raise ValueError("ngram_size must be positive")
    for sequence in sequences:
        if len(sequence) < n:
            continue
        for start in range(len(sequence) - n + 1):
            counts[tuple(int(token) for token in sequence[start : start + n])] += 1
    return counts


def _frequency_bucket(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 4:
        return "2-4"
    if count <= 19:
        return "5-19"
    return "20+"


def _build_frequency_bins() -> list[dict[str, Any]]:
    return [
        {"frequency_bin": "0", "lower": 0, "upper": 0, "total": 0, "correct": 0, "accuracy": None},
        {"frequency_bin": "1", "lower": 1, "upper": 1, "total": 0, "correct": 0, "accuracy": None},
        {"frequency_bin": "2-4", "lower": 2, "upper": 4, "total": 0, "correct": 0, "accuracy": None},
        {"frequency_bin": "5-19", "lower": 5, "upper": 19, "total": 0, "correct": 0, "accuracy": None},
        {"frequency_bin": "20+", "lower": 20, "upper": None, "total": 0, "correct": 0, "accuracy": None},
    ]


def _build_frequency_bin_index(bins: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {entry["frequency_bin"]: entry for entry in bins}


def _topk_contains(predicted: Sequence[int], target: int) -> bool:
    return int(target) in {int(token) for token in predicted}


def compute_memorisation_summary(
    train_sequences: Sequence[SequenceTokens],
    heldout_sequences: Sequence[SequenceTokens],
    *,
    predict_topk: PredictTopK | None = None,
    top_k: int = 5,
    ngram_size: int = 3,
    rare_threshold: int = 1,
    vocab_size: int | None = None,
) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if rare_threshold < 1:
        raise ValueError("rare_threshold must be at least 1")

    train_sequences = _normalize_sequences(train_sequences)
    heldout_sequences = _normalize_sequences(heldout_sequences)
    token_counts = _count_token_frequencies(train_sequences)
    ngram_counts = _count_ngrams(train_sequences, ngram_size)
    rare_ngram_count = sum(1 for count in ngram_counts.values() if count <= rare_threshold)

    frequency_bins = _build_frequency_bins()
    frequency_index = _build_frequency_bin_index(frequency_bins)
    effective_top_k = min(top_k, vocab_size) if vocab_size is not None else top_k
    if effective_top_k < 1:
        effective_top_k = 1

    status = "completed"
    if not heldout_sequences:
        status = "empty_heldout"
    elif predict_topk is None:
        status = "no_model"

    rare_overlap_total = 0
    rare_overlap_hits = 0
    scored_positions = 0

    if predict_topk is not None and heldout_sequences:
        for sequence in heldout_sequences:
            if len(sequence) <= 1:
                continue
            for mask_index in range(1, len(sequence)):
                target = int(sequence[mask_index])
                predicted = list(predict_topk(sequence, mask_index, effective_top_k))
                if not predicted:
                    continue
                scored_positions += 1
                bin_entry = frequency_index[_frequency_bucket(int(token_counts.get(target, 0)))]
                bin_entry["total"] += 1
                if _topk_contains(predicted, target):
                    bin_entry["correct"] += 1
                if mask_index + 1 >= ngram_size:
                    context = tuple(int(token) for token in sequence[mask_index - ngram_size + 1 : mask_index + 1])
                    if ngram_counts.get(context, 0) <= rare_threshold and ngram_counts.get(context, 0) > 0:
                        rare_overlap_total += 1
                        if _topk_contains(predicted, target):
                            rare_overlap_hits += 1

        for entry in frequency_bins:
            if entry["total"] > 0:
                entry["accuracy"] = entry["correct"] / entry["total"]

        if scored_positions == 0:
            status = "empty_heldout"

    summary = {
        "evaluation_type": "rare_pattern_memorisation_probe",
        "disclaimer": "Exploratory aggregate memorisation diagnostic within the demo dataset. It is not a privacy audit or guarantee.",
        "status": status,
        "requested_top_k": int(top_k),
        "effective_top_k": int(effective_top_k),
        "ngram_size": int(ngram_size),
        "rare_threshold": int(rare_threshold),
        "model_scoring_available": bool(predict_topk is not None and scored_positions > 0),
        "training_token_count": int(sum(token_counts.values())),
        "training_rare_ngram_count": int(rare_ngram_count),
        "rare_ngram_overlap": {
            "total": int(rare_overlap_total),
            "hit_count": int(rare_overlap_hits),
            "rate": (rare_overlap_hits / rare_overlap_total) if rare_overlap_total > 0 else None,
        },
        "masked_accuracy_by_frequency": frequency_bins,
    }
    return summary


def _load_sequences_from_processed_dir(processed_dir: Path, split_name: str) -> list[list[int]]:
    dataset = EncodedDataset(processed_dir / "encoded" / split_name)
    return [list(stay.tokens) for stay in dataset.stays()]


def _build_predictor(
    config: dict[str, Any],
    processed_dir: Path,
    checkpoint_path: Path,
) -> tuple[PredictTopK, int]:
    vocab = _load_json(processed_dir / "vocab.json")
    if not isinstance(vocab, dict):
        raise ValueError("vocab.json must contain a mapping")
    model_conf = config.get("model", {})
    model = ICUTinyTransformer(
        vocab_size=len(vocab),
        max_seq_len=int(model_conf.get("max_seq_len", 256)),
        d_model=int(model_conf.get("d_model", 64)),
        n_heads=int(model_conf.get("n_heads", 4)),
        n_layers=int(model_conf.get("n_layers", 2)),
        dim_feedforward=int(model_conf.get("dim_feedforward", 256)),
        dropout=float(model_conf.get("dropout", 0.1)),
    )
    head = MaskedEventPredictionHead(d_model=int(model_conf.get("d_model", 64)), vocab_size=len(vocab))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = getattr(checkpoint, "model_state", None)
    head_state = getattr(checkpoint, "prediction_head_state", None)
    if model_state is None and isinstance(checkpoint, dict):
        model_state = checkpoint.get("model_state")
        head_state = checkpoint.get("prediction_head_state")
    if model_state is None:
        raise ValueError("checkpoint missing model_state")
    try:
        model.load_state_dict(model_state)
        if head_state is not None:
            head.load_state_dict(head_state)
    except RuntimeError as error:
        raise ValueError("vocabulary mismatch") from error
    model.eval()
    head.eval()
    mask_id = int(vocab.get("[MASK]", 2))

    def predict_topk(sequence: SequenceTokens, mask_index: int, top_k: int) -> list[int]:
        input_ids = torch.tensor([list(int(token) for token in sequence)], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        input_ids[0, mask_index] = mask_id
        with torch.no_grad():
            encoded, _ = model(input_ids, padding_mask=attention_mask)
            logits = head(encoded)[0, mask_index]
            effective_k = min(max(1, int(top_k)), logits.shape[-1])
            return torch.topk(logits, k=effective_k).indices.tolist()

    return predict_topk, len(vocab)


def run_memorisation_probe(
    *,
    train_sequences: Sequence[SequenceTokens] | None = None,
    heldout_sequences: Sequence[SequenceTokens] | None = None,
    predict_topk: PredictTopK | None = None,
    top_k: int = 5,
    ngram_size: int = 3,
    rare_threshold: int = 1,
    vocab_size: int | None = None,
    config: dict[str, Any] | None = None,
    processed_dir: Path | None = None,
    run_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    heldout_split: str = "validation",
    output_path: Path | None = None,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    if train_sequences is None or heldout_sequences is None:
        if argv is not None:
            parser = argparse.ArgumentParser(description="Run rare-pattern memorisation diagnostics.")
            parser.add_argument("--config", type=Path, default=Path("configs/final/eicu_demo_final_tiny.yaml"))
            parser.add_argument("--processed_dir", type=Path, default=None)
            parser.add_argument("--run_dir", type=Path, default=None)
            parser.add_argument("--checkpoint", type=Path, default=None)
            parser.add_argument("--heldout_split", choices=("validation", "test"), default="validation")
            parser.add_argument("--top_k", type=int, default=5)
            parser.add_argument("--ngram_size", type=int, default=3)
            parser.add_argument("--rare_threshold", type=int, default=1)
            args = parser.parse_args(list(argv))
            config = load_yaml(args.config)
            processed_dir = args.processed_dir
            run_dir = args.run_dir
            checkpoint_path = args.checkpoint
            heldout_split = args.heldout_split
            top_k = args.top_k
            ngram_size = args.ngram_size
            rare_threshold = args.rare_threshold
        elif config is None:
            config = load_yaml(Path("configs/final/eicu_demo_final_tiny.yaml"))
        validated = validate_final_config(json.loads(json.dumps(config)))
        if processed_dir is None:
            processed_dir = Path(validated["data"]["processed_dir"])
        processed_dir = Path(processed_dir)
        train_sequences = _load_sequences_from_processed_dir(processed_dir, "train")
        if heldout_sequences is None:
            if heldout_split not in {"validation", "test"}:
                raise ValueError("heldout_split must be validation or test")
            heldout_sequences = _load_sequences_from_processed_dir(processed_dir, heldout_split)
        if checkpoint_path is not None:
            predict_topk, vocab_size = _build_predictor(validated, processed_dir, Path(checkpoint_path))
        if output_path is None and run_dir is not None:
            output_path = Path(run_dir) / "memorisation_probe.json"
        elif output_path is None:
            output_path = Path("results") / "summary" / "memorisation_probe.json"

    summary = compute_memorisation_summary(
        train_sequences or [],
        heldout_sequences or [],
        predict_topk=predict_topk,
        top_k=top_k,
        ngram_size=ngram_size,
        rare_threshold=rare_threshold,
        vocab_size=vocab_size,
    )

    if output_path is not None:
        _atomic_write_json(Path(output_path), summary)
    return summary
