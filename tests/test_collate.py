import torch
import pytest
from icu_pretrain.data.collate import (
    collate_event_batch,
    SupervisedCollator,
    MLMCollator,
    ResumableDeterministicSampler,
    create_dataloader,
)
from icu_pretrain.data.dataset import EncodedStay

def test_collate_padding_and_attention_mask():
    # Two stays of different lengths
    stay1 = EncodedStay(patientunitstayid="stay1", tokens=[3, 10, 20], label=0, split_name="train")
    stay2 = EncodedStay(patientunitstayid="stay2", tokens=[3, 5, 6, 7, 8], label=1, split_name="train")
    batch = [stay1, stay2]
    out = collate_event_batch(batch)
    input_ids = out["input_ids"]
    attention = out["attention_mask"]
    # Expect shape (2,5) with PAD id 0
    assert input_ids.shape == (2, 5)
    assert torch.all(input_ids[0, :3] == torch.tensor([3, 10, 20]))
    assert torch.all(input_ids[0, 3:] == 0)
    assert torch.all(input_ids[1] == torch.tensor([3, 5, 6, 7, 8]))
    # Attention mask 1 for real tokens, 0 for padding
    assert torch.all(attention[0, :3] == 1)
    assert torch.all(attention[0, 3:] == 0)
    assert torch.all(attention[1] == 1)

def test_collate_mlm_masking_respects_unmaskable_tokens():
    # Fix random seed for reproducibility
    torch.manual_seed(42)
    stay = EncodedStay(patientunitstayid="s", tokens=[3, 4, 5, 6, 0, 0], label=0, split_name="train")
    # tokens: CLS=3, then three maskable tokens (4,5,6), then two PAD=0
    out = collate_event_batch([stay])
    input_ids = out["input_ids"][0]
    mlm_labels = out["mlm_labels"][0]
    # Ensure PAD and CLS were not masked (labels remain -100)
    assert mlm_labels[0].item() == -100  # CLS
    assert mlm_labels[4:].tolist() == [-100, -100]  # PAD positions
    # With only three maskable tokens, 15% -> 0 masks, so no MLM labels
    assert (mlm_labels != -100).sum().item() == 0
    # Create longer sequence to trigger masking (26 maskable tokens)
    long_tokens = [3] + list(range(4, 30))
    stay_long = EncodedStay(patientunitstayid="l", tokens=long_tokens, label=0, split_name="train")
    torch.manual_seed(0)
    out_long = collate_event_batch([stay_long])
    mlm_long = out_long["mlm_labels"][0]
    maskable_positions = (mlm_long != -100).nonzero(as_tuple=False).flatten()
    # Expected number of masked tokens: floor(26 * 0.15) = 3
    assert len(maskable_positions) == 3
    # Ensure CLS (index 0) not masked
    assert 0 not in maskable_positions.tolist()


def test_supervised_collator():
    vocab = {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2, "[CLS]": 3, "a": 4, "b": 5}
    collator = SupervisedCollator(vocab)
    stay1 = EncodedStay(patientunitstayid="1", tokens=[3, 4], label=1, split_name="train")
    stay2 = EncodedStay(patientunitstayid="2", tokens=[3, 5, 4], label=0, split_name="train")
    out = collator([stay1, stay2])
    assert out["input_ids"].shape == (2, 3)
    assert out["input_ids"][0, 2].item() == 0
    assert out["attention_mask"][0, 2].item() == 0
    assert out["attention_mask"][0, 1].item() == 1
    assert out["labels"].tolist() == [1, 0]


def test_mlm_collator_excludes_static_and_specials():
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[MASK]": 2,
        "[CLS]": 3,
        "a": 4,
        "b": 5,
        "STATIC::AGE": 6,
        "STATIC::GENDER": 7,
    }
    collator = MLMCollator(vocab, mlm_probability=0.5, seed=42)
    tokens = [3, 6, 7] + [4] * 20
    stay = EncodedStay(patientunitstayid="1", tokens=tokens, label=0, split_name="train")
    out = collator([stay], epoch=0)
    mlm_labels = out["mlm_labels"][0]
    assert mlm_labels[0].item() == -100
    assert mlm_labels[1].item() == -100
    assert mlm_labels[2].item() == -100


def test_mlm_collator_epoch_determinism():
    vocab = {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2, "[CLS]": 3, "a": 4, "b": 5}
    collator = MLMCollator(vocab, mlm_probability=0.5, seed=42)
    tokens = [3] + [4] * 20
    stay = EncodedStay(patientunitstayid="1", tokens=tokens, label=0, split_name="train")
    out1 = collator([stay], epoch=1)
    out2 = collator([stay], epoch=1)
    out3 = collator([stay], epoch=2)
    assert torch.equal(out1["input_ids"], out2["input_ids"])
    assert torch.equal(out1["mlm_labels"], out2["mlm_labels"])
    assert not torch.equal(out1["input_ids"], out3["input_ids"])


def test_mlm_collator_replacement_rules():
    vocab = {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2, "[CLS]": 3}
    for i in range(4, 1004):
        vocab[f"token_{i}"] = i
    collator = MLMCollator(vocab, mlm_probability=1.0, seed=123)
    tokens = [3] + list(range(4, 104))
    stay = EncodedStay(patientunitstayid="1", tokens=tokens, label=0, split_name="train")
    out = collator([stay], epoch=0)
    input_ids = out["input_ids"][0]
    mlm_labels = out["mlm_labels"][0]
    masked_indices = (mlm_labels != -100).nonzero(as_tuple=False).flatten()
    assert len(masked_indices) == 100
    mask_count = 0
    rand_count = 0
    same_count = 0
    for idx in masked_indices:
        orig = mlm_labels[idx].item()
        curr = input_ids[idx].item()
        if curr == 2:
            mask_count += 1
        elif curr == orig:
            same_count += 1
        else:
            assert curr >= 4
            rand_count += 1
    assert mask_count > 0
    assert rand_count > 0
    assert same_count > 0


def test_resumable_deterministic_sampler():
    sampler = ResumableDeterministicSampler(dataset_size=10, seed=42, epoch=0)
    indices1 = list(sampler)
    sampler.set_epoch(0)
    indices2 = list(sampler)
    assert indices1 == indices2
    sampler.set_epoch(1)
    indices3 = list(sampler)
    assert indices1 != indices3
    sampler = ResumableDeterministicSampler(dataset_size=10, seed=42, epoch=0, cursor=4)
    resumed_indices = list(sampler)
    assert resumed_indices == indices1[4:]
    sampler.set_epoch(0)
    sampler.cursor = 7
    resumed_indices2 = list(sampler)
    assert resumed_indices2 == indices1[7:]


def test_sampler_state_dict():
    sampler = ResumableDeterministicSampler(dataset_size=10, seed=42, epoch=2, cursor=3)
    state = sampler.state_dict()
    assert state["seed"] == 42
    assert state["epoch"] == 2
    assert state["cursor"] == 3
    sampler2 = ResumableDeterministicSampler(dataset_size=10)
    sampler2.load_state_dict(state)
    assert sampler2.seed == 42
    assert sampler2.epoch == 2
    assert sampler2.cursor == 3
    assert list(sampler2) == list(sampler)


def test_collate_edge_cases():
    vocab = {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2, "[CLS]": 3, "STATIC::1": 4}
    collator = MLMCollator(vocab, mlm_probability=0.15, seed=42)
    stay_no_maskable = EncodedStay(patientunitstayid="1", tokens=[3, 4], label=0, split_name="train")
    out = collator([stay_no_maskable], epoch=0)
    assert (out["mlm_labels"] != -100).sum().item() == 0
    stay_max_len = EncodedStay(patientunitstayid="2", tokens=[3] + [4]*255, label=0, split_name="train")
    out2 = collator([stay_max_len], epoch=0)
    assert out2["input_ids"].shape == (1, 256)
    with pytest.raises(ValueError, match="vocabulary must contain non-special tokens"):
        collator_empty = MLMCollator({"[PAD]": 0, "[UNK]": 1, "[MASK]": 2, "[CLS]": 3}, mlm_probability=1.0)
        stay_err = EncodedStay(patientunitstayid="3", tokens=[3, 1], label=0, split_name="train")
        collator_empty([stay_err], epoch=0)
