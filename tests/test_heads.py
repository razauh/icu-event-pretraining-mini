import pytest
import torch
from icu_pretrain.models.heads import (
    MaskedEventPredictionHead,
    MortalityPredictionHead,
    compute_mlm_loss,
    compute_mortality_loss,
)

def test_masked_event_prediction_head():
    head = MaskedEventPredictionHead(d_model=64, vocab_size=100)
    x = torch.randn(4, 32, 64)
    out = head(x)
    assert out.shape == (4, 32, 100)
    with pytest.raises(ValueError, match="input tensor must have shape"):
        head(torch.randn(4, 64))

def test_mortality_prediction_head():
    head = MortalityPredictionHead(d_model=64)
    x = torch.randn(4, 64)
    out = head(x)
    assert out.shape == (4,)
    x_single = torch.randn(1, 64)
    out_single = head(x_single)
    assert out_single.shape == (1,)
    with pytest.raises(ValueError, match="cls_output must have shape"):
        head(torch.randn(4, 32, 64))

def test_compute_mlm_loss():
    logits = torch.randn(2, 5, 10)
    labels = torch.tensor([[1, 2, -100, -100, 3], [-100, 4, 5, -100, -100]], dtype=torch.long)
    loss = compute_mlm_loss(logits, labels)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    labels_all_ignored = torch.full((2, 5), -100, dtype=torch.long)
    loss_ignored = compute_mlm_loss(logits, labels_all_ignored)
    assert loss_ignored.item() == 0.0
    assert loss_ignored.requires_grad
    labels_mismatch = torch.tensor([[1, 2, 15, -100, 3], [-100, 4, 5, -100, -100]], dtype=torch.long)
    with pytest.raises(ValueError, match="out-of-vocabulary"):
        compute_mlm_loss(logits, labels_mismatch)
    with pytest.raises(ValueError, match="must be 3D"):
        compute_mlm_loss(torch.randn(2, 10), labels)

def test_compute_mortality_loss():
    logits = torch.tensor([0.5, -0.2, 1.2, -1.5], dtype=torch.float)
    targets = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    loss = compute_mortality_loss(logits, targets)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    targets_all_one = torch.tensor([1, 1, 1, 1], dtype=torch.long)
    loss_all_one = compute_mortality_loss(logits, targets_all_one)
    assert loss_all_one.ndim == 0
    pos_weight = torch.tensor(2.0)
    loss_weighted = compute_mortality_loss(logits, targets, pos_weight=pos_weight)
    assert loss_weighted.ndim == 0
    with pytest.raises(ValueError, match="must be binary"):
        compute_mortality_loss(logits, torch.tensor([1, 2, 0, 1], dtype=torch.long))
    with pytest.raises(ValueError, match="must be strictly positive"):
        compute_mortality_loss(logits, targets, pos_weight=torch.tensor(-1.0))
