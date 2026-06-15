import pytest
import torch
from icu_pretrain.models.transformer import ICUTinyTransformer

def test_default_parameters():
    model = ICUTinyTransformer(vocab_size=100)
    assert model.max_seq_len == 256
    assert model.token_embedding.embedding_dim == 64
    assert model.position_embedding.embedding_dim == 64
    assert model.family_embedding.embedding_dim == 64
    assert model.token_embedding.num_embeddings == 100
    assert model.family_embedding.num_embeddings == 16

def test_forward_output_shapes():
    model = ICUTinyTransformer(vocab_size=100)
    token_ids = torch.randint(1, 100, (4, 32))
    seq_out, cls_out = model(token_ids)
    assert seq_out.shape == (4, 32, 64)
    assert cls_out.shape == (4, 64)

def test_embedding_addition():
    model = ICUTinyTransformer(vocab_size=100)
    token_ids = torch.randint(1, 100, (2, 16))
    family_ids = torch.randint(0, 8, (2, 16))
    seq_out_no_fam, _ = model(token_ids)
    seq_out_fam, _ = model(token_ids, family_ids=family_ids)
    assert not torch.allclose(seq_out_no_fam, seq_out_fam)

def test_padding_mask_behavior():
    model = ICUTinyTransformer(vocab_size=100, dropout=0.0)
    model.eval()
    token_ids = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, 4, 5]], dtype=torch.long)
    padding_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.long)
    with torch.no_grad():
        seq_out, _ = model(token_ids, padding_mask=padding_mask)
    token_ids_alt = torch.tensor([[1, 2, 3, 9, 9], [1, 2, 3, 4, 5]], dtype=torch.long)
    with torch.no_grad():
        seq_out_alt, _ = model(token_ids_alt, padding_mask=padding_mask)
    assert torch.allclose(seq_out[0, :3], seq_out_alt[0, :3], atol=1e-5)

def test_length_overflow():
    model = ICUTinyTransformer(vocab_size=100, max_seq_len=16)
    token_ids = torch.randint(1, 100, (2, 17))
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(token_ids)

def test_invalid_shapes():
    model = ICUTinyTransformer(vocab_size=100)
    token_ids_1d = torch.randint(1, 100, (16,))
    with pytest.raises(ValueError, match="2D tensor"):
        model(token_ids_1d)
    token_ids = torch.randint(1, 100, (2, 16))
    family_ids_mismatch = torch.randint(0, 8, (2, 15))
    with pytest.raises(ValueError, match="family_ids must have the same shape"):
        model(token_ids, family_ids=family_ids_mismatch)
    padding_mask_mismatch = torch.randint(0, 2, (2, 17))
    with pytest.raises(ValueError, match="padding_mask must have the same shape"):
        model(token_ids, padding_mask=padding_mask_mismatch)

def test_empty_batch():
    model = ICUTinyTransformer(vocab_size=100)
    token_ids = torch.randint(1, 100, (0, 16))
    with pytest.raises(ValueError, match="batch size must be greater than 0"):
        model(token_ids)

def test_incompatible_heads():
    with pytest.raises(ValueError, match="divisible by n_heads"):
        ICUTinyTransformer(vocab_size=100, d_model=64, n_heads=5)

def test_parameter_count():
    model = ICUTinyTransformer(vocab_size=100)
    count = model.parameter_count
    assert isinstance(count, int)
    assert count > 0

def test_deterministic_eval_mode():
    model = ICUTinyTransformer(vocab_size=100, dropout=0.5)
    token_ids = torch.randint(1, 100, (2, 16))
    model.eval()
    with torch.no_grad():
        out1_eval, _ = model(token_ids)
        out2_eval, _ = model(token_ids)
    assert torch.allclose(out1_eval, out2_eval)
    model.train()
    out1_train, _ = model(token_ids)
    out2_train, _ = model(token_ids)
    assert not torch.allclose(out1_train, out2_train)
