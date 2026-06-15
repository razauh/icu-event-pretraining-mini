"""Batch collation for ICU event sequences."""


import torch
from typing import List, Dict
from .dataset import EncodedStay

def collate_event_batch(batch: List[EncodedStay]) -> Dict[str, torch.Tensor]:
    """Collate a batch of :class:`EncodedStay` objects.

    Returns a dict with keys ``input_ids``, ``attention_mask`` and ``mlm_labels``.
    ``input_ids`` is padded to the longest sequence in *batch* using the PAD
    token ID (0). ``attention_mask`` has ``1`` for real tokens and ``0`` for PAD.
    ``mlm_labels`` follows the masked‑language‑modeling scheme:
    * 15% of *maskable* tokens are selected (maskable = not PAD and not CLS).
    * 80% of selected tokens are replaced with the MASK token ID (2).
    * 10% are replaced with a random token ID from the vocabulary.
    * 10% are left unchanged.
    Unselected positions have a label of -100 so that loss is ignored.
    """
    if not batch:
        raise ValueError("batch must contain at least one EncodedStay")

    pad_id = 0
    cls_id = 3
    mask_id = 2

    # Determine max sequence length in the batch (capped at 256 by contract)
    max_len = max(len(stay.tokens) for stay in batch)
    if max_len > 256:
        raise ValueError("sequence length exceeds max_seq_len of 256")

    batch_size = len(batch)
    # Prepare tensors
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, stay in enumerate(batch):
        seq_len = len(stay.tokens)
        input_ids[i, :seq_len] = torch.tensor(stay.tokens, dtype=torch.long)
        attention_mask[i, :seq_len] = 1

    # MLM label preparation
    mlm_labels = torch.full_like(input_ids, -100)
    # Identify maskable positions (exclude PAD and CLS)
    maskable = (input_ids != pad_id) & (input_ids != cls_id)
    maskable_indices = torch.nonzero(maskable, as_tuple=False)
    num_maskable = maskable_indices.size(0)
    num_to_mask = int(num_maskable * 0.15)
    if num_to_mask == 0:
        return {"input_ids": input_ids, "attention_mask": attention_mask, "mlm_labels": mlm_labels}

    # Randomly select positions to mask
    perm = torch.randperm(num_maskable)
    selected = maskable_indices[perm[:num_to_mask]]
    # Record original tokens for loss computation
    original_tokens = input_ids[selected[:, 0], selected[:, 1]].clone()
    mlm_labels[selected[:, 0], selected[:, 1]] = original_tokens

    # Decide replacement type for each selected position
    rand_vals = torch.rand(num_to_mask)
    # 80% -> MASK token
    mask_mask = rand_vals < 0.80
    # 10% -> random token
    random_mask = (rand_vals >= 0.80) & (rand_vals < 0.90)
    # 10% -> keep original (already recorded in labels)

    # Apply MASK token
    if mask_mask.any():
        rows, cols = selected[mask_mask][:, 0], selected[mask_mask][:, 1]
        input_ids[rows, cols] = mask_id

    # Apply random token replacement
    if random_mask.any():
        rows, cols = selected[random_mask][:, 0], selected[random_mask][:, 1]
        # Vocabulary IDs are contiguous starting at 0; exclude special tokens [0-3]
        vocab_max_id = int(input_ids.max().item())
        if vocab_max_id < 4:
            raise ValueError("vocabulary must contain non‑special tokens for random replacement")
        random_ids = torch.randint(4, vocab_max_id + 1, (rows.shape[0],), dtype=torch.long)
        input_ids[rows, cols] = random_ids

    return {"input_ids": input_ids, "attention_mask": attention_mask, "mlm_labels": mlm_labels}


