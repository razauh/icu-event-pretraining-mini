"""Prediction heads for pretraining and downstream tasks."""


import torch
import torch.nn as nn

class MaskedEventPredictionHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("input tensor must have shape [batch, seq_len, d_model]")
        return self.projection(x)


class MortalityPredictionHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, 1)

    def forward(self, cls_output: torch.Tensor) -> torch.Tensor:
        if cls_output.ndim != 2:
            raise ValueError("cls_output must have shape [batch, d_model]")
        logits = self.projection(cls_output)
        return logits.squeeze(-1)


def compute_mlm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("logits must be 3D and labels must be 2D")
    if logits.shape[:2] != labels.shape:
        raise ValueError("batch and sequence dimensions must match between logits and labels")
    vocab_size = logits.shape[-1]
    valid_mask = (labels == -100) | ((labels >= 0) & (labels < vocab_size))
    if not valid_mask.all():
        raise ValueError("labels contain out-of-vocabulary indices")
    if (labels == -100).all():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return nn.functional.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)


def compute_mortality_loss(logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    if logits.ndim != 1 or targets.ndim != 1:
        raise ValueError("logits and targets must be 1D tensors")
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same shape")
    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("targets must be binary (0 or 1)")
    if pos_weight is not None:
        if pos_weight.ndim not in (0, 1):
            raise ValueError("pos_weight must be a scalar or 1D tensor")
        if (pos_weight <= 0).any():
            raise ValueError("pos_weight must be strictly positive")
    targets_float = targets.to(logits.dtype)
    return nn.functional.binary_cross_entropy_with_logits(logits, targets_float, pos_weight=pos_weight)
