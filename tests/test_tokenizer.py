from __future__ import annotations

from pathlib import Path
import pytest
import json

from icu_pretrain.data.tokenizer import EventTokenizer

def test_tokenizer_special_tokens() -> None:
    tokenizer = EventTokenizer()
    assert tokenizer.vocab["[PAD]"] == 0
    assert tokenizer.vocab["[UNK]"] == 1
    assert tokenizer.vocab["[MASK]"] == 2
    assert tokenizer.vocab["[CLS]"] == 3

def test_tokenizer_fit_and_encode() -> None:
    sequences = [
        ["DX::Sepsis", "LAB::Sodium::Q1", "DX::Sepsis"],
        ["DX::Sepsis", "LAB::Sodium::Q1", "VITAL::HeartRate::Q3"],
        ["DX::Sepsis", "LAB::Sodium::Q1", "DX::Shock"],
        ["DX::Sepsis", "LAB::Sodium::Q1", "DX::Shock"],
        ["DX::Sepsis", "LAB::Sodium::Q1", "DX::Shock"],
        ["DX::Shock", "DX::Shock"],
    ]
    tokenizer = EventTokenizer()
    tokenizer.fit(sequences)
    
    assert "DX::Sepsis" in tokenizer.vocab
    assert "LAB::Sodium::Q1" in tokenizer.vocab
    assert "DX::Shock" in tokenizer.vocab
    assert "VITAL::HeartRate::Q3" not in tokenizer.vocab
    
    assert tokenizer.frequencies["DX::Sepsis"] == 5
    assert tokenizer.frequencies["LAB::Sodium::Q1"] == 5
    assert tokenizer.frequencies["DX::Shock"] == 5
    
    encoded = tokenizer.encode(["DX::Sepsis", "VITAL::HeartRate::Q3", "[CLS]"])
    assert encoded[0] == tokenizer.vocab["DX::Sepsis"]
    assert encoded[1] == tokenizer.vocab["[UNK]"]
    assert encoded[2] == tokenizer.vocab["[CLS]"]
    
    decoded = tokenizer.decode(encoded)
    assert decoded == ["DX::Sepsis", "[UNK]", "[CLS]"]

def test_tokenizer_fit_empty_fails() -> None:
    tokenizer = EventTokenizer()
    with pytest.raises(ValueError):
        tokenizer.fit([])

def test_tokenizer_save_load(tmp_path: Path) -> None:
    sequences = [["DX::Sepsis"] * 5]
    tokenizer = EventTokenizer()
    tokenizer.fit(sequences)
    
    vocab_path = tmp_path / "vocab.json"
    tokenizer.save(vocab_path)
    
    loaded = EventTokenizer.load(vocab_path)
    assert loaded.vocab == tokenizer.vocab
    assert loaded.frequencies == tokenizer.frequencies
    assert loaded.special_tokens == tokenizer.special_tokens

def test_tokenizer_corrupt_load(tmp_path: Path) -> None:
    vocab_path = tmp_path / "corrupt_vocab.json"
    
    vocab_path.write_text(json.dumps({"vocab": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        EventTokenizer.load(vocab_path)
        
    vocab_path.write_text(json.dumps({"vocab": {"[PAD]": 0}, "frequencies": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        EventTokenizer.load(vocab_path)

    bad_ids = {
        "[PAD]": 1,
        "[UNK]": 1,
        "[MASK]": 2,
        "[CLS]": 3
    }
    vocab_path.write_text(json.dumps({"vocab": bad_ids, "frequencies": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        EventTokenizer.load(vocab_path)

    duplicate_ids = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[MASK]": 2,
        "[CLS]": 3,
        "DX::Sepsis": 3
    }
    vocab_path.write_text(json.dumps({"vocab": duplicate_ids, "frequencies": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        EventTokenizer.load(vocab_path)

    non_contiguous_ids = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[MASK]": 2,
        "[CLS]": 3,
        "DX::Sepsis": 5
    }
    vocab_path.write_text(json.dumps({"vocab": non_contiguous_ids, "frequencies": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        EventTokenizer.load(vocab_path)

def test_tokenizer_special_token_in_input() -> None:
    sequences = [
        ["DX::Sepsis", "[PAD]", "DX::Sepsis"],
        ["DX::Sepsis", "[PAD]"],
        ["DX::Sepsis", "[PAD]"],
        ["DX::Sepsis", "[PAD]"],
        ["DX::Sepsis", "[PAD]"],
    ]
    tokenizer = EventTokenizer()
    tokenizer.fit(sequences)
    assert "[PAD]" not in tokenizer.frequencies
    assert tokenizer.vocab["[PAD]"] == 0

def test_tokenizer_validation_test_isolation() -> None:
    train_sequences = [["DX::Sepsis"] * 5]
    val_sequences = [["DX::Shock"] * 10]
    
    tokenizer = EventTokenizer()
    tokenizer.fit(train_sequences)
    
    assert "DX::Sepsis" in tokenizer.vocab
    assert "DX::Shock" not in tokenizer.vocab
