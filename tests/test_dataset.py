from pathlib import Path
import pytest

from icu_pretrain.data.dataset import EncodedStay, EncodedDataset


def make_stay(uid: str, tokens: list[int], label: int, split: str):
    return {
        "patientunitstayid": uid,
        "tokens": tokens,
        "label": label,
        "split_name": split,
    }


def test_dataset_loads_shards_and_enforces_contracts(tmp_path: Path):
    # Create two valid shards
    shard0 = [
        make_stay("stay-1", [3, 5, 6], 0, "train"),
        make_stay("stay-2", [3, 7], 1, "train"),
    ]
    shard1 = [
        make_stay("stay-3", [3, 8, 9, 10], 0, "validation"),
    ]
    EncodedDataset.write_shard(shard0, shard_id=0, root=tmp_path)
    EncodedDataset.write_shard(shard1, shard_id=1, root=tmp_path)

    ds = EncodedDataset(tmp_path)
    assert len(ds) == 3
    stays = ds.stays()
    ids = [s.patientunitstayid for s in stays]
    assert ids == ["stay-1", "stay-2", "stay-3"]
    # Verify token length enforcement (no error for short sequences)
    for s in stays:
        assert len(s.tokens) <= 256


def test_duplicate_stay_raises(tmp_path: Path):
    shard0 = [make_stay("stay-1", [3, 5], 0, "train")]
    shard1 = [make_stay("stay-1", [3, 6], 1, "validation")]
    EncodedDataset.write_shard(shard0, shard_id=0, root=tmp_path)
    EncodedDataset.write_shard(shard1, shard_id=1, root=tmp_path)
    with pytest.raises(ValueError, match="duplicate stay identifier"):
        EncodedDataset(tmp_path)


def test_missing_label_raises(tmp_path: Path):
    shard0 = [{
        "patientunitstayid": "stay-1",
        "tokens": [3, 5],
        # label omitted deliberately
        "split_name": "train",
    }]
    EncodedDataset.write_shard(shard0, shard_id=0, root=tmp_path)
    with pytest.raises(ValueError, match="stay record missing fields"):
        EncodedDataset(tmp_path)


def test_token_length_exceeds_limit(tmp_path: Path):
    long_tokens = [3] + list(range(256))  # length 257 > 256
    shard0 = [make_stay("stay-1", long_tokens, 0, "train")]
    EncodedDataset.write_shard(shard0, shard_id=0, root=tmp_path)
    with pytest.raises(ValueError, match="tokens exceed max_seq_len"):
        EncodedDataset(tmp_path)


def test_missing_shard_raises(tmp_path: Path):
    shard0 = [make_stay("stay-1", [3, 5], 0, "train")]
    EncodedDataset.write_shard(shard0, shard_id=0, root=tmp_path)
    (tmp_path / "shard_0.json").unlink()
    with pytest.raises(ValueError, match="shard file missing"):
        EncodedDataset(tmp_path)


def test_checksum_mismatch_raises(tmp_path: Path):
    shard0 = [make_stay("stay-1", [3, 5], 0, "train")]
    EncodedDataset.write_shard(shard0, shard_id=0, root=tmp_path)
    (tmp_path / "shard_0.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        EncodedDataset(tmp_path)
