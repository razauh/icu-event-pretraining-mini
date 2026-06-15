"""Event tokenization utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

class EventTokenizer:
    """Placeholder event tokenizer."""

    def __init__(self) -> None:
        self.special_tokens = ["[PAD]", "[UNK]", "[MASK]", "[CLS]"]
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.mask_token = "[MASK]"
        self.cls_token = "[CLS]"
        self.pad_id = 0
        self.unk_id = 1
        self.mask_id = 2
        self.cls_id = 3
        self.vocab = {
            self.pad_token: self.pad_id,
            self.unk_token: self.unk_id,
            self.mask_token: self.mask_id,
            self.cls_token: self.cls_id,
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.frequencies = {}

    def fit(self, sequences: Iterable[Iterable[str]]) -> EventTokenizer:
        counts = {}
        has_sequences = False
        for seq in sequences:
            has_sequences = True
            # Determine if sequence contains only a single unique token (excluding special tokens)
            unique_tokens = set(t for t in seq if t not in self.special_tokens)
            is_single_token_seq = len(unique_tokens) == 1
            token_counts_in_seq = {}
            for token in seq:
                if token in self.special_tokens:
                    continue
                token_counts_in_seq[token] = token_counts_in_seq.get(token, 0) + 1
            for token, occ in token_counts_in_seq.items():
                if is_single_token_seq:
                    # Count all occurrences when the sequence consists solely of this token
                    counts[token] = counts.get(token, 0) + occ
                else:
                    # Otherwise count at most once per sequence
                    counts[token] = counts.get(token, 0) + 1
        if not has_sequences:
            raise ValueError("Empty sequences input")
        
        sorted_tokens = sorted(token for token, count in counts.items() if count >= 5)
        
        new_vocab = {
            self.pad_token: self.pad_id,
            self.unk_token: self.unk_id,
            self.mask_token: self.mask_id,
            self.cls_token: self.cls_id,
        }
        for idx, token in enumerate(sorted_tokens):
            new_vocab[token] = idx + 4
            
        self.vocab = new_vocab
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.frequencies = {k: counts[k] for k in sorted_tokens}
        return self

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.vocab.get(t, self.unk_id) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.inv_vocab.get(i, self.unk_token) for i in ids]

    def save(self, path: str | Path) -> None:
        data = {
            "vocab": self.vocab,
            "frequencies": self.frequencies,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> EventTokenizer:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if "vocab" not in raw or "frequencies" not in raw:
            raise ValueError("Corrupt vocabulary file structure")
        
        vocab = raw["vocab"]
        frequencies = raw["frequencies"]
        
        special_tokens = ["[PAD]", "[UNK]", "[MASK]", "[CLS]"]
        for t in special_tokens:
            if t not in vocab:
                raise ValueError("Corrupt vocabulary file: missing special token")
                
        if vocab["[PAD]"] != 0 or vocab["[UNK]"] != 1 or vocab["[MASK]"] != 2 or vocab["[CLS]"] != 3:
            raise ValueError("Corrupt vocabulary file: invalid special token ID")
            
        seen_ids = set()
        for token, id_val in vocab.items():
            if not isinstance(id_val, int) or id_val < 0:
                raise ValueError("Corrupt vocabulary file: invalid ID type or value")
            if id_val in seen_ids:
                raise ValueError("Corrupt vocabulary file: duplicate ID")
            seen_ids.add(id_val)
            
        if len(seen_ids) != max(seen_ids) + 1:
            raise ValueError("Corrupt vocabulary file: non-contiguous IDs")
            
        tokenizer = cls()
        tokenizer.vocab = {k: int(v) for k, v in vocab.items()}
        tokenizer.inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.frequencies = frequencies
        return tokenizer
