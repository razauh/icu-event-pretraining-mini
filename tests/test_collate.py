import torch
from icu_pretrain.data.collate import collate_event_batch
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
