"""Tiny Transformer encoder for ICU event streams."""


import torch
import torch.nn as nn

class ICUTinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 256, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.family_embedding = nn.Embedding(16, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.max_seq_len = max_seq_len

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, token_ids: torch.Tensor, family_ids: torch.Tensor | None = None, padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must be a 2D tensor of shape [batch, seq_len]")
        batch, seq_len = token_ids.shape
        if batch == 0:
            raise ValueError("batch size must be greater than 0")
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        if family_ids is not None:
            if family_ids.shape != token_ids.shape:
                raise ValueError("family_ids must have the same shape as token_ids")
        if padding_mask is not None:
            if padding_mask.shape != token_ids.shape:
                raise ValueError("padding_mask must have the same shape as token_ids")
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch, -1)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        if family_ids is not None:
            x = x + self.family_embedding(family_ids)
        key_padding_mask = None
        if padding_mask is not None:
            if padding_mask.dtype == torch.bool:
                key_padding_mask = padding_mask
            else:
                key_padding_mask = (padding_mask == 0)
        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
        cls_output = x[:, 0, :]
        return x, cls_output

