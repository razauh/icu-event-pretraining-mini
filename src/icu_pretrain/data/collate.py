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


class SupervisedCollator:
    def __init__(self, vocab: Dict[str, int]) -> None:
        self.vocab = vocab
        self.pad_id = vocab.get("[PAD]", 0)

    def __call__(self, batch: List[EncodedStay]) -> Dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("batch must contain at least one EncodedStay")
        max_len = max(len(stay.tokens) for stay in batch)
        if max_len > 256:
            raise ValueError("sequence length exceeds max_seq_len of 256")
        batch_size = len(batch)
        input_ids = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.zeros(batch_size, dtype=torch.long)

        for i, stay in enumerate(batch):
            seq_len = len(stay.tokens)
            input_ids[i, :seq_len] = torch.tensor(stay.tokens, dtype=torch.long)
            attention_mask[i, :seq_len] = 1
            labels[i] = stay.label

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class MLMCollator:
    def __init__(self, vocab: Dict[str, int], mlm_probability: float = 0.15, seed: int = 42) -> None:
        self.vocab = vocab
        self.mlm_probability = mlm_probability
        self.seed = seed
        self.pad_id = vocab.get("[PAD]", 0)
        self.mask_id = vocab.get("[MASK]", 2)
        self.cls_id = vocab.get("[CLS]", 3)
        self.static_token_ids = {v for k, v in vocab.items() if k.startswith("STATIC::")}

    def __call__(self, batch: List[EncodedStay], epoch: int = 0) -> Dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("batch must contain at least one EncodedStay")
        max_len = max(len(stay.tokens) for stay in batch)
        if max_len > 256:
            raise ValueError("sequence length exceeds max_seq_len of 256")
        batch_size = len(batch)
        input_ids = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, stay in enumerate(batch):
            seq_len = len(stay.tokens)
            input_ids[i, :seq_len] = torch.tensor(stay.tokens, dtype=torch.long)
            attention_mask[i, :seq_len] = 1

        mlm_labels = torch.full_like(input_ids, -100)
        g = torch.Generator()
        g.manual_seed(self.seed + epoch)

        is_static = torch.zeros_like(input_ids, dtype=torch.bool)
        for static_id in self.static_token_ids:
            is_static |= (input_ids == static_id)

        maskable = (input_ids != self.pad_id) & (input_ids != self.cls_id) & ~is_static
        maskable_indices = torch.nonzero(maskable, as_tuple=False)
        num_maskable = maskable_indices.size(0)
        num_to_mask = int(num_maskable * self.mlm_probability)

        if num_to_mask > 0:
            perm = torch.randperm(num_maskable, generator=g)
            selected = maskable_indices[perm[:num_to_mask]]
            rows = selected[:, 0]
            cols = selected[:, 1]
            original_tokens = input_ids[rows, cols].clone()
            mlm_labels[rows, cols] = original_tokens

            rand_vals = torch.rand(num_to_mask, generator=g)
            mask_mask = rand_vals < 0.80
            random_mask = (rand_vals >= 0.80) & (rand_vals < 0.90)

            if mask_mask.any():
                input_ids[rows[mask_mask], cols[mask_mask]] = self.mask_id
            if random_mask.any():
                vocab_size = len(self.vocab)
                if vocab_size < 5:
                    raise ValueError("vocabulary must contain non-special tokens for random replacement")
                random_ids = torch.randint(4, vocab_size, (random_mask.sum().item(),), dtype=torch.long, generator=g)
                input_ids[rows[random_mask], cols[random_mask]] = random_ids

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mlm_labels": mlm_labels,
        }


class ResumableDeterministicSampler(torch.utils.data.Sampler[int]):
    def __init__(self, dataset_size: int, seed: int = 42, epoch: int = 0, cursor: int = 0) -> None:
        self.dataset_size = dataset_size
        self.seed = seed
        self.epoch = epoch
        self.cursor = cursor
        self._generate_permutation()

    def _generate_permutation(self) -> None:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        self.permutation = torch.randperm(self.dataset_size, generator=g).tolist()

    def __iter__(self):
        for idx in self.permutation[self.cursor:]:
            yield idx
            self.cursor += 1

    def __len__(self) -> int:
        return self.dataset_size - self.cursor

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.cursor = 0
        self._generate_permutation()

    def state_dict(self) -> Dict[str, int]:
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "cursor": self.cursor,
        }

    def load_state_dict(self, state: Dict[str, int]) -> None:
        self.seed = state.get("seed", self.seed)
        self.epoch = state.get("epoch", self.epoch)
        self.cursor = state.get("cursor", self.cursor)
        self._generate_permutation()


def create_dataloader(
    dataset,
    batch_size: int,
    sampler: ResumableDeterministicSampler,
    collator,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
    )
