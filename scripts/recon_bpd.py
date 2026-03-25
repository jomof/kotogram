#!/usr/bin/env python3
"""Full-sequence BPD KC training loop — standalone reference implementation.

Primary metric: bpd (bits per attended token-position)
  - Karpathy-style comparable compression metric adapted to token positions
    rather than raw bytes.  Normalization denominator is attention_mask.sum().
  - Lower is better.
  - Designed for run-to-run comparison within this experiment family.

Model architecture (inlined):
  surface embedding → transformer encoder → attention pooler → KC head
  → Gumbel-sampled KC probs → position-aware recon MLP → surface logits

External dependencies limited to data loading:
  kotogram.locations, kotogram.tokenizer, train.paths, train.dataset

Usage:
    source requirements.sh
    python -m scripts.recon_bpd
"""

import math
import time
from dataclasses import dataclass
from typing import Tuple, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

# Data loading only — the sole external dependencies from this project.
from kotogram import locations
from kotogram.tokenizer import Tokenizer
from train import paths as train_paths
from train.dataset import StyleDataset, collate_fn

# Fused linear cross-entropy (apple/ml-cross-entropy): computes the
# output_head matmul + CE in a single kernel without materializing [B,T,V].
# Falls back to chunked approach when unavailable (e.g. MPS without Triton).
try:
    from cut_cross_entropy import linear_cross_entropy as _cce_linear_ce

    _HAS_CCE = True
except ImportError:
    _HAS_CCE = False

# ── Training config ──────────────────────────────────────────────────
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
IS_CUDA = DEVICE == "cuda"
USE_FUSED_CE = IS_CUDA and _HAS_CCE
BATCH_SIZE = 256
EPOCHS = 1000
SAMPLE_RATIO = 1 if IS_CUDA else 0.3
LR = 1e-4
TEMPERATURE = 1.8
GRAD_CAP = 5.0
INPUT_MASK_RATIO = 0.15
RECON_CHUNK = 8 if IS_CUDA else 4  # CUDA has VRAM for larger chunks
SEED = 42

KL_SPARSE_WEIGHT = 0.0001
KL_TARGET_RHO = 0.03
COV_PENALTY_WEIGHT = 5.0

LOG2 = math.log(2.0)


# ═════════════════════════════════════════════════════════════════════
# Model architecture
# (inlined from kotogram.model and train.models — only the pieces
#  needed for encoder → KC bottleneck → full-sequence reconstruction)
# ═════════════════════════════════════════════════════════════════════


@dataclass
class BpdModelConfig:
    """Minimal config for the encoder → KC → recon architecture."""

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


class KCHead(nn.Module):
    """MLP: pooled representation → KC logits.  Two hidden layers with expansion."""

    def __init__(self, d_model: int, kc_vocab_size: int, dropout: float = 0.1):
        super().__init__()
        mid = d_model * 2
        self.hidden1 = nn.Linear(d_model, mid)
        self.hidden2 = nn.Linear(mid, d_model)
        self.output = nn.Linear(d_model, kc_vocab_size)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(kc_vocab_size)

    def forward_with_raw(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (raw_logits, layer-normed logits)."""
        x = self.drop(self.act(self.hidden1(x)))
        x = self.drop(self.act(self.hidden2(x)))
        raw = self.output(x)
        return raw, cast(torch.Tensor, self.norm(raw))


class ReconDecoder(nn.Module):
    """Position-aware MLP: (kc_probs, position) → hidden → surface logits.

    End-relative position embeddings: 0 = last content token, 1 = second-to-last, …
    This lets the model learn sentence-final patterns (particles, copulas)
    directly from positional features.

    ``output_head`` is exposed as a plain ``nn.Linear`` so the caller can
    chunk the expensive [H → V] projection externally.
    """

    def __init__(
        self,
        kc_vocab_size: int,
        surface_vocab_size: int,
        pos_embed_dim: int = 64,
        hidden_dim: int = 256,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.pos_embed = nn.Embedding(max_seq_len, pos_embed_dim)
        self.hidden1 = nn.Linear(kc_vocab_size + pos_embed_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, surface_vocab_size, bias=False)
        self.act = nn.ReLU()

    def forward_hidden(
        self, kc_probs: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pre-logit hidden states for every position.  Shape: [B, T, H]."""
        B, T = attention_mask.shape
        lengths = attention_mask.bool().sum(dim=1)
        abs_pos = torch.arange(T, device=kc_probs.device).unsqueeze(0).expand(B, -1)
        end_rel = (lengths.unsqueeze(1) - 1 - abs_pos).clamp(min=0)
        pos_emb = self.pos_embed(end_rel)
        kc_exp = kc_probs.unsqueeze(1).expand(-1, T, -1)
        h = torch.cat([kc_exp, pos_emb], dim=-1)
        h = self.act(self.hidden1(h))
        return self.act(self.hidden2(h))


class BpdModel(nn.Module):
    """Encoder → KC bottleneck → reconstruction decoder.

    Minimal architecture for the BPD training objective.  Omits all
    classification heads (formality, gender, grammaticality, register,
    grammar-point) present in the full TrainingClassifier.
    """

    def __init__(self, cfg: BpdModelConfig):
        super().__init__()
        self.cfg = cfg

        # Surface embedding → d_model
        self.surface_embed = nn.Embedding(
            cfg.surface_vocab_size, cfg.surface_embed_dim, padding_idx=0
        )
        self.embed_proj = nn.Linear(cfg.surface_embed_dim, cfg.d_model)
        self.embed_norm = nn.LayerNorm(cfg.d_model)
        self.embed_drop = nn.Dropout(cfg.dropout)

        # Positional encoding + Transformer encoder
        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.max_seq_len, cfg.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, cfg.num_layers, enable_nested_tensor=False
        )

        # Attention pooler → KC head → recon decoder
        self.pooler = AttentionPooler(cfg.d_model, cfg.num_heads, cfg.dropout)
        self.kc_head = KCHead(cfg.d_model, cfg.kc_vocab_size, cfg.dropout)
        self.recon = ReconDecoder(
            cfg.kc_vocab_size,
            cfg.surface_vocab_size,
            cfg.recon_pos_embed_dim,
            cfg.recon_hidden_dim,
            cfg.max_seq_len,
        )

    def encode(
        self, surface_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Surface token IDs → pooled representation [B, d_model]."""
        x = self.surface_embed(surface_ids)
        x = self.embed_proj(x)
        x = self.embed_norm(x)
        x = self.embed_drop(x)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
        return cast(torch.Tensor, self.pooler(x, attention_mask))


# ═════════════════════════════════════════════════════════════════════
# Training loop
# ═════════════════════════════════════════════════════════════════════


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device(DEVICE)
    print(f"Device: {device}")
    print(f"Fused CE: {USE_FUSED_CE}")

    if IS_CUDA:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # ── Data loading ─────────────────────────────────────────────────
    output_dir = locations.get_style_output_dir()
    tokenizer = Tokenizer.load(f"{output_dir}/tokenizer.json")

    cache_dir = train_paths.get_style_dataset_cache_dir()
    dataset = StyleDataset(
        cache_dir, tokenizer, sample_ratio=SAMPLE_RATIO, verbose=True
    )
    gram_ds = dataset.filter_by_grammaticality(label=1)
    print(f"Gram sentences: {len(gram_ds)}")

    n_total_batches = (len(gram_ds) + BATCH_SIZE - 1) // BATCH_SIZE
    dl_generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        gram_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2 if IS_CUDA else 0,
        pin_memory=IS_CUDA,
        drop_last=False,
        generator=dl_generator,
    )

    # ── Model ────────────────────────────────────────────────────────
    surface_vocab = tokenizer.get_vocab_sizes()["surface"]
    cfg = BpdModelConfig(surface_vocab_size=surface_vocab)
    model = BpdModel(cfg)
    model.to(device)
    if IS_CUDA:
        model = torch.compile(model)
    model.train()

    optimizer = Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler(device.type, enabled=IS_CUDA)

    # ── Training loop ────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        t0 = time.perf_counter()
        total_loss_sum = 0.0
        epoch_total_bits = 0.0
        epoch_num_units = 0
        total_kl_sum = 0.0
        total_cov_sum = 0.0
        epoch_t1_correct = 0
        epoch_t1_units = 0
        total_elements = 0
        n_batches = 0

        for batch in loader:
            surface_ids = batch.feature_inputs["input_ids_surface"].to(
                device, non_blocking=IS_CUDA
            )
            attention_mask = batch.attention_mask.to(
                device, non_blocking=IS_CUDA
            )
            B, T = attention_mask.shape

            # Snapshot targets before masking
            recon_targets = surface_ids.clone()

            # BERT-style input corruption (encoder input only)
            if INPUT_MASK_RATIO > 0:
                maskable = attention_mask.bool()
                rand_mask = (
                    torch.rand_like(surface_ids.float()) < INPUT_MASK_RATIO
                ) & maskable
                surface_ids = surface_ids.masked_fill(rand_mask, 0)

            # ── Forward (under autocast, matching KCTrainer) ─────────
            with torch.amp.autocast(device.type):
                pooled = model.encode(surface_ids, attention_mask)
                kc_logits_raw, _ = model.kc_head.forward_with_raw(pooled)

                # Gumbel noise + gradient capping
                u = torch.rand_like(kc_logits_raw).clamp_(1e-6, 1 - 1e-6)
                g = -torch.log(-torch.log(u))
                logits_select = kc_logits_raw + 0.6 * g

                if kc_logits_raw.requires_grad:
                    kc_logits_raw.register_hook(
                        lambda grad: grad.clamp(min=-GRAD_CAP, max=GRAD_CAP)
                    )

                logits_select = logits_select.clamp(-12, 12)
                kc_probs = torch.sigmoid(logits_select / TEMPERATURE)
                kc_probs = torch.nan_to_num(
                    kc_probs, nan=0.0, posinf=1.0, neginf=0.0
                )

                # Recon decoder: pre-logit hidden states [B, T, H]
                h_recon = model.recon.forward_hidden(kc_probs, attention_mask)

            # ── Output projection + CE ────────────────────────────────
            assert h_recon.shape[:2] == recon_targets.shape
            assert attention_mask.shape == recon_targets.shape
            mask_f = attention_mask.float()

            out_weight = model.recon.output_head.weight

            if USE_FUSED_CE:
                # Fused linear CE: h_recon × W^T → CE in one kernel,
                # never materializes [B, T, V] in global memory.
                ce_targets = recon_targets.clone()
                ce_targets[~attention_mask.bool()] = -100
                nll_per_token = _cce_linear_ce(
                    h_recon, out_weight, ce_targets, reduction="none",
                )
                total_nll_nats = nll_per_token.sum()

                # Top-1 every 100 batches (separate forward pass is expensive)
                if n_batches % 100 == 0:
                    epoch_t1_units += int(mask_f.sum().item())
                    with torch.no_grad():
                        for c0 in range(0, T, RECON_CHUNK):
                            c1 = min(c0 + RECON_CHUNK, T)
                            with torch.amp.autocast(device.type):
                                chunk_logits = F.linear(
                                    h_recon[:, c0:c1, :], out_weight
                                )
                            preds = chunk_logits.argmax(dim=-1)
                            chunk_mask = attention_mask[:, c0:c1].bool()
                            epoch_t1_correct += int(
                                ((preds == recon_targets[:, c0:c1]) & chunk_mask)
                                .sum().item()
                            )
            else:
                # Chunked fallback (MPS / no CCE): each chunk is
                # [B, RECON_CHUNK, V] to bound peak memory.
                total_nll_nats = torch.tensor(0.0, device=device)
                for c0 in range(0, T, RECON_CHUNK):
                    c1 = min(c0 + RECON_CHUNK, T)
                    with torch.amp.autocast(device.type):
                        chunk_logits = F.linear(
                            h_recon[:, c0:c1, :], out_weight
                        )
                    chunk_logits = chunk_logits.float()
                    chunk_targets = recon_targets[:, c0:c1]
                    chunk_mask = mask_f[:, c0:c1]
                    chunk_nll = F.cross_entropy(
                        chunk_logits.reshape(-1, chunk_logits.size(-1)),
                        chunk_targets.reshape(-1),
                        reduction="none",
                    ).reshape(B, -1)
                    total_nll_nats = total_nll_nats + (chunk_nll * chunk_mask).sum()
                    with torch.no_grad():
                        preds = chunk_logits.argmax(dim=-1)
                        epoch_t1_correct += int(
                            ((preds == chunk_targets) & chunk_mask.bool())
                            .sum().item()
                        )

            # nats → bits, normalize by attended token count
            # Primary run-to-run fitness metric.  Lower is better.
            total_bits = total_nll_nats / LOG2
            num_units = mask_f.sum().clamp_min(1)
            bpd = total_bits / num_units

            # ── Regularizers ─────────────────────────────────────────
            loss = bpd
            kl_contrib = 0.0
            cov_contrib = 0.0

            if KL_SPARSE_WEIGHT > 0:
                rho_hat = kc_probs.mean(dim=0).clamp(1e-7, 1 - 1e-7)
                rho = KL_TARGET_RHO
                kl_term = (
                    rho_hat * torch.log(rho_hat / rho)
                    + (1 - rho_hat) * torch.log((1 - rho_hat) / (1 - rho))
                ).sum()
                kl_scaled = KL_SPARSE_WEIGHT * kl_term
                loss = loss + kl_scaled
                kl_contrib = kl_scaled.item()

            if COV_PENALTY_WEIGHT > 0:
                centered = kc_probs - kc_probs.mean(dim=0)
                cov = (centered.T @ centered) / max(1, B)
                cov.fill_diagonal_(0.0)
                cov_term = (cov**2).mean()
                cov_scaled = COV_PENALTY_WEIGHT * cov_term
                loss = loss + cov_scaled
                cov_contrib = cov_scaled.item()

            # ── Backward + step ──────────────────────────────────────
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # ── Epoch stats ──────────────────────────────────────────
            batch_units = int(num_units.item())
            total_loss_sum += loss.item()
            epoch_total_bits += total_bits.item()
            epoch_num_units += batch_units
            total_kl_sum += kl_contrib
            total_cov_sum += cov_contrib
            total_elements += B
            n_batches += 1

            del loss, h_recon, total_nll_nats, total_bits, bpd
            if device.type == "mps" and n_batches % 8 == 0:
                torch.mps.empty_cache()

            dt_batch = time.perf_counter() - t0
            t1_denom = epoch_t1_units if USE_FUSED_CE else epoch_num_units
            t1_pct = 100.0 * epoch_t1_correct / max(1, t1_denom)
            print(
                f"\r  batch {n_batches}/{n_total_batches}  "
                f"bpd={epoch_total_bits / max(1, epoch_num_units):.4f}  "
                f"To-1={t1_pct:.1f}%  "
                f"{total_elements / dt_batch:.1f} el/s  "
                f"{dt_batch:.1f}s",
                end="",
                flush=True,
            )

        dt = time.perf_counter() - t0
        avg_bpd = epoch_total_bits / max(1, epoch_num_units)
        avg_loss = total_loss_sum / max(1, n_batches)
        avg_kl = total_kl_sum / max(1, n_batches)
        avg_cov = total_cov_sum / max(1, n_batches)
        t1_denom = epoch_t1_units if USE_FUSED_CE else epoch_num_units
        t1_pct = 100.0 * epoch_t1_correct / max(1, t1_denom)
        els = total_elements / dt
        print()  # finish \r progress line
        print(
            f"Epoch {epoch + 1}/{EPOCHS}  "
            f"bpd={avg_bpd:.4f}  "
            f"To-1={t1_pct:.1f}%  "
            f"loss={avg_loss:.4f}  "
            f"sparsity={avg_kl:.4f}  "
            f"orthogonality={avg_cov:.4f}  "
            f"{els:.1f} el/s  "
            f"{total_elements} samples  "
            f"{dt:.1f}s"
        )


if __name__ == "__main__":
    main()
