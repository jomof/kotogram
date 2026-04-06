"""BPD model architecture: encoder -> KC bottleneck -> reconstruction decoder.

Shared between training (scratch/recon_bpd.py) and inference
(scripts/recon_bpd/inference.py).  Do not add training-specific code here.
"""

import math
from dataclasses import dataclass
from typing import Tuple, cast

import torch
from torch import nn


@dataclass
class BpdModelConfig:
    """Minimal config for the encoder -> KC -> recon architecture."""

    surface_vocab_size: int
    surface_embed_dim: int = 300  # matches chiVe pretrained vectors
    d_model: int = 512
    ffn_dim: int = 2048
    num_layers: int = 4
    num_heads: int = 16
    dropout: float = 0.1
    max_seq_len: int = 512
    kc_vocab_size: int = 1024
    recon_pos_embed_dim: int = 64
    recon_hidden_dim: int = 256
    layer_drop_prob: float = 0.5


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + cast(torch.Tensor, self.pe)[:, : x.size(1), :]
        return cast(torch.Tensor, self.dropout(x))


class AttentionPooler(nn.Module):
    """Attention-weighted pooling with a learnable query vector."""

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self, encoder_output: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        query = self.query.expand(encoder_output.size(0), -1, -1)
        attn_output, _ = self.attention(
            query=query,
            key=encoder_output,
            value=encoder_output,
            key_padding_mask=(attention_mask == 0),
        )
        return cast(torch.Tensor, self.layer_norm(attn_output.squeeze(1)))


class KCHead(nn.Module):  # pylint: disable=abstract-method
    """MLP: pooled representation -> KC logits.  Two hidden layers with expansion."""

    def __init__(self, d_model: int, kc_vocab_size: int, dropout: float = 0.1):
        super().__init__()
        mid = d_model * 2
        self.hidden1 = nn.Linear(d_model, mid)
        self.hidden2 = nn.Linear(mid, d_model)
        self.output = nn.Linear(d_model, kc_vocab_size)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(kc_vocab_size)

    def forward_with_raw(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (raw_logits, layer-normed logits)."""
        x = self.drop(self.act(self.hidden1(x)))
        x = self.drop(self.act(self.hidden2(x)))
        raw = self.output(x)
        return raw, cast(torch.Tensor, self.norm(raw))


class ReconDecoder(nn.Module):  # pylint: disable=abstract-method
    """Position-aware MLP: (kc_probs, position) -> hidden -> surface logits.

    Dual positional encoding: end-relative (0 = last content token) and
    start-relative (0 = first content token).  Together they implicitly
    encode both absolute position AND sentence length -- a token at
    start_rel=2, end_rel=5 is the third token in an 8-token sentence.

    ``output_head`` is exposed as a plain ``nn.Linear`` so the caller can
    chunk the expensive [H -> V] projection externally.
    """

    def __init__(
        self,
        kc_vocab_size: int,
        surface_vocab_size: int,
        *,
        pos_embed_dim: int = 64,
        hidden_dim: int = 256,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.pos_embed_end = nn.Embedding(max_seq_len, pos_embed_dim)
        self.pos_embed_start = nn.Embedding(max_seq_len, pos_embed_dim)
        self.hidden1 = nn.Linear(kc_vocab_size + 2 * pos_embed_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, surface_vocab_size, bias=False)
        self.semantic_head = nn.Linear(
            hidden_dim, 300, bias=False
        )  # 300D Chive early-exit projection
        self.act = nn.ReLU()

    def forward_hidden(
        self, kc_probs: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pre-logit hidden states for every position.  Shape: [B, T, H]."""
        bsz, seq_len = attention_mask.shape
        lengths = attention_mask.bool().sum(dim=1)
        abs_pos = (
            torch.arange(seq_len, device=kc_probs.device).unsqueeze(0).expand(bsz, -1)
        )

        end_rel = (lengths.unsqueeze(1) - 1 - abs_pos).clamp(min=0)
        pos_emb_end = self.pos_embed_end(end_rel)

        start_rel = abs_pos.clamp(max=lengths.unsqueeze(1) - 1)
        pos_emb_start = self.pos_embed_start(start_rel)

        kc_exp = kc_probs.unsqueeze(1).expand(-1, seq_len, -1)
        h = torch.cat([kc_exp, pos_emb_end, pos_emb_start], dim=-1)
        h = self.act(self.hidden1(h))
        result: torch.Tensor = self.act(self.hidden2(h))
        return result


class BpdModel(nn.Module):  # pylint: disable=abstract-method
    """Encoder -> KC bottleneck -> reconstruction decoder.

    Minimal architecture for the BPD training objective.  Omits all
    classification heads (formality, gender, grammaticality, register,
    grammar-point) present in the full TrainingClassifier.
    """

    def __init__(self, cfg: BpdModelConfig):
        super().__init__()
        self.cfg = cfg

        self.surface_embed = nn.Embedding(
            cfg.surface_vocab_size, cfg.surface_embed_dim, padding_idx=0
        )
        self.embed_proj = nn.Linear(cfg.surface_embed_dim, cfg.d_model)
        self.embed_norm = nn.LayerNorm(cfg.d_model)
        self.embed_drop = nn.Dropout(cfg.dropout)

        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.max_seq_len, cfg.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, cfg.num_layers, enable_nested_tensor=False
        )

        std_scale = 1.0 / math.sqrt(2.0 * max(1, cfg.num_layers))
        for layer in self.encoder.layers:
            nn.init.normal_(
                layer.self_attn.out_proj.weight, mean=0.0, std=0.02 * std_scale
            )
            nn.init.normal_(layer.linear2.weight, mean=0.0, std=0.02 * std_scale)

        self.pooler = AttentionPooler(cfg.d_model, cfg.num_heads, cfg.dropout)
        self.kc_head = KCHead(cfg.d_model, cfg.kc_vocab_size, cfg.dropout)
        self.recon = ReconDecoder(
            cfg.kc_vocab_size,
            cfg.surface_vocab_size,
            pos_embed_dim=cfg.recon_pos_embed_dim,
            hidden_dim=cfg.recon_hidden_dim,
            max_seq_len=cfg.max_seq_len,
        )

        # Diagnostic head: predict sentence length from KC probs alone.
        self.length_head = nn.Sequential(
            nn.Linear(cfg.kc_vocab_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def encode(
        self, surface_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Surface token IDs -> pooled representation [B, d_model]."""
        x = self.surface_embed(surface_ids)
        x = self.embed_proj(x)
        x = self.embed_norm(x)
        x = self.embed_drop(x)
        x = self.pos_enc(x)
        # Stochastic Depth (LayerDrop) -- training only
        pad_mask = attention_mask == 0
        if self.training and self.cfg.layer_drop_prob > 0:
            for i, layer in enumerate(self.encoder.layers):
                if i > 0 and torch.rand(1).item() < self.cfg.layer_drop_prob:
                    continue
                x = layer(x, src_key_padding_mask=pad_mask)
        else:
            x = self.encoder(x, src_key_padding_mask=pad_mask)
        return cast(torch.Tensor, self.pooler(x, attention_mask))
