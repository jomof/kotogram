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
    python -m scratch.recon_bpd
"""

import contextlib
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Data loading only — the sole external dependencies from this project.
from kotogram import locations
from kotogram.tokenizer import Tokenizer
from train import paths as train_paths
from train.chive import load_chive_for_vocab
from train.dataset import StyleDataset, collate_fn

# Fused linear cross-entropy (apple/ml-cross-entropy): computes the
# output_head matmul + CE in a single kernel without materializing [B,T,V].
# Falls back to chunked approach when unavailable (e.g. MPS without Triton).
try:
    from cut_cross_entropy import linear_cross_entropy as _cce_linear_ce

    _HAS_CCE = True
except ImportError:
    _HAS_CCE = False

# ── Hardware detection ────────────────────────────────────────────────
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
IS_CUDA = DEVICE == "cuda"
USE_FUSED_CE = IS_CUDA and _HAS_CCE
LOG2 = math.log(2.0)
AUTOCAST = (lambda: torch.amp.autocast(DEVICE)) if IS_CUDA else contextlib.nullcontext


# ── Configuration ─────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    """All tunable hyperparameters for the BPD training loop."""

    # Training
    batch_size: int = 256
    epochs: int = 1000
    sample_ratio: float = 1.0
    lr: float = 1e-4  # Original lr: 1e-4
    temperature: float = 1.8  # Original temperature: 1.8
    weight_decay: float = 0.01
    grad_cap: float = 5.0  # Original grad_cap: 5.0
    input_mask_ratio: float = 0.15  # Original input_mask_ratio: 0.15
    seed: int = 42  # Original seed: 42

    # Regularization
    kl_sparse_weight: float = 0.0001  # Original kl_sparse_weight: 0.0001
    kl_target_rho: float = 0.03  # Original kl_target_rho: 0.03
    cov_penalty_weight: float = 5.0  # Original cov_penalty_weight: 5.0
    consistency_weight: float = 0.0001  # dual-mask KC consistency (0 = disabled)

    # Model architecture
    d_model: int = 512  # Original d_model: 512
    ffn_dim: int = 2048  # Original ffn_dim: 2048
    num_layers: int = 6  # Original num_layers: 4
    num_heads: int = 16  # Original num_heads: 16
    dropout: float = 0.1  # Original dropout: 0.1
    kc_vocab_size: int = 1024  # Original kc_vocab_size: 1024
    recon_pos_embed_dim: int = 64  # Original recon_pos_embed_dim: 64
    recon_hidden_dim: int = 256  # Original recon_hidden_dim: 256


@dataclass
class TrainResult:
    """Metrics returned from a training run."""

    final_bpd: float
    final_top1_pct: float
    final_cossim: float
    final_loss: float


@dataclass
class TrainCheckpoint:
    """Opaque checkpoint for resuming a partially-completed training run.

    The caller should treat this as a black box — pass it back to ``train()``
    to resume from the last completed epoch.
    """

    model_state: Dict[str, torch.Tensor]
    optimizer_state: Dict[str, object]
    scaler_state: Dict[str, object]
    scheduler_state: Dict[str, object]
    epoch: int  # last completed epoch (0-indexed)
    latest_metrics: Dict[str, float]
    epoch_history: list  # [(epoch, metrics_dict), ...]


EpochEndCallback = Callable[[int, Dict[str, float]], None]
EpochStartCallback = Callable[[int], None]


def save_checkpoint(checkpoint: TrainCheckpoint, path: str) -> None:
    """Persist a checkpoint to disk (atomic write)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(
        {
            "model_state": checkpoint.model_state,
            "optimizer_state": checkpoint.optimizer_state,
            "scaler_state": checkpoint.scaler_state,
            "scheduler_state": checkpoint.scheduler_state,
            "epoch": checkpoint.epoch,
            "latest_metrics": checkpoint.latest_metrics,
            "epoch_history": checkpoint.epoch_history,
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def load_checkpoint(path: str) -> Optional[TrainCheckpoint]:
    """Load a checkpoint from disk, or return None if not found."""
    if not os.path.exists(path):
        return None
    data = torch.load(path, weights_only=False, map_location=DEVICE)
    return TrainCheckpoint(**data)


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

    def forward_with_raw(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, cfg.num_layers, enable_nested_tensor=False
        )

        # Scale residual stream projections to maintain unit variance regardless of depth
        std_scale = 1.0 / math.sqrt(2.0 * max(1, cfg.num_layers))
        for layer in self.encoder.layers:
            nn.init.normal_(layer.self_attn.out_proj.weight, mean=0.0, std=0.02 * std_scale)
            nn.init.normal_(layer.linear2.weight, mean=0.0, std=0.02 * std_scale)

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

class _SetupCache:
    def __init__(self):
        self.dataset_subsets: Dict[float, object] = {}
        self.tokenizer: Optional[Tokenizer] = None
        self.chive_weights: Optional[torch.Tensor] = None
        self.chive_normed: Optional[torch.Tensor] = None
        self.cached_model: Optional[BpdModel] = None
        self.cached_model_cfg: Optional[BpdModelConfig] = None

GLOBAL_SETUP_CACHE = _SetupCache()


def train(
    config: TrainConfig,
    on_epoch_start: EpochStartCallback,
    on_epoch_end: EpochEndCallback,
    checkpoint_path: str,
    checkpoint: Optional[TrainCheckpoint] = None,
) -> Tuple[TrainResult, TrainCheckpoint]:
    """Run the BPD training loop and return final metrics + checkpoint.

    Args:
        config: Training configuration.
        on_epoch_start: Callback invoked at the start of each epoch with
            ``(epoch_index,)``.
        on_epoch_end: Callback invoked at the end of each epoch with
            ``(epoch_index, metrics_dict)``.  May raise any exception
            (e.g. ``optuna.TrialPruned``) to abort training early.
        checkpoint: Optional checkpoint from a previous ``train()`` call.
            When provided, model/optimizer/scaler state is restored and
            training resumes from ``checkpoint.epoch + 1``.
        checkpoint_path: Path for per-epoch checkpoint persistence.
            The checkpoint is saved to this path after every epoch
            (before the ``on_epoch_end`` callback) for crash recovery.
    """

    torch.manual_seed(config.seed)
    device = torch.device(DEVICE)
    sample_ratio = config.sample_ratio
    recon_chunk = 8 if IS_CUDA else 4

    print(f"Device: {device}")
    print(f"Fused CE: {USE_FUSED_CE}")

    if IS_CUDA:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # ── Data loading ─────────────────────────────────────────────────
    global GLOBAL_SETUP_CACHE
    if GLOBAL_SETUP_CACHE.tokenizer is None:
        output_dir = locations.get_style_output_dir()
        GLOBAL_SETUP_CACHE.tokenizer = Tokenizer.load(f"{output_dir}/tokenizer.json")
    tokenizer = GLOBAL_SETUP_CACHE.tokenizer

    if sample_ratio not in GLOBAL_SETUP_CACHE.dataset_subsets:
        cache_dir = train_paths.get_style_dataset_cache_dir()
        dataset = StyleDataset(
            cache_dir,
            tokenizer,
            sample_ratio=sample_ratio,
        )
        GLOBAL_SETUP_CACHE.dataset_subsets[sample_ratio] = dataset.filter_by_grammaticality(label=1)
        print(f"Gram sentences: {len(GLOBAL_SETUP_CACHE.dataset_subsets[sample_ratio])}")
    gram_ds = GLOBAL_SETUP_CACHE.dataset_subsets[sample_ratio]

    n_total_batches = (len(gram_ds) + config.batch_size - 1) // config.batch_size
    dl_generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        gram_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2 if IS_CUDA else 0,
        pin_memory=IS_CUDA,
        drop_last=False,
        generator=dl_generator,
    )

    # ── Model ────────────────────────────────────────────────────────
    surface_vocab = tokenizer.get_vocab_sizes()["surface"]
    cfg = BpdModelConfig(
        surface_vocab_size=surface_vocab,
        d_model=config.d_model,
        ffn_dim=config.ffn_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        kc_vocab_size=config.kc_vocab_size,
        recon_pos_embed_dim=config.recon_pos_embed_dim,
        recon_hidden_dim=config.recon_hidden_dim,
    )
    # Load chiVe pretrained surface embeddings and freeze
    if GLOBAL_SETUP_CACHE.chive_weights is None:
        cw = load_chive_for_vocab(tokenizer.field_vocabs["surface"])
        
        # Missing chiVe words are all zeros. Randomize them so OOV words aren't squashed together.
        norms = cw.norm(dim=-1)
        missing_mask = (norms == 0.0)
        missing_mask[0] = False  # Keep index 0 [PAD] strictly at zero
        
        if missing_mask.any():
            # Match the variance of the known chiVe vectors
            std = cw[~missing_mask].std().item() if (~missing_mask).any() else 0.1
            cw[missing_mask] = torch.randn_like(cw[missing_mask]) * std
            
        GLOBAL_SETUP_CACHE.chive_weights = cw
    chive_weights = GLOBAL_SETUP_CACHE.chive_weights

    if GLOBAL_SETUP_CACHE.cached_model_cfg == cfg and GLOBAL_SETUP_CACHE.cached_model is not None:
        import copy
        model = copy.deepcopy(GLOBAL_SETUP_CACHE.cached_model)
    else:
        model = BpdModel(cfg)
        with torch.no_grad():
            n = min(model.surface_embed.weight.size(0), chive_weights.size(0))
            model.surface_embed.weight[:n] = chive_weights[:n]
            model.surface_embed.weight[0].zero_()  # keep padding at zero
        model.surface_embed.weight.requires_grad = False
        import copy
        GLOBAL_SETUP_CACHE.cached_model_cfg = cfg
        GLOBAL_SETUP_CACHE.cached_model = copy.deepcopy(model)

    # L2-normalized chiVe embeddings for cosine-similarity Top-1 metric.
    # Tokens without chiVe vectors (zero rows) get zero norm → excluded.
    if GLOBAL_SETUP_CACHE.chive_normed is None:
        chive_normed = F.normalize(chive_weights[:surface_vocab], dim=-1)
        GLOBAL_SETUP_CACHE.chive_normed = chive_normed.to(device)
    chive_normed = GLOBAL_SETUP_CACHE.chive_normed

    model.to(device)
    if IS_CUDA:
        model = torch.compile(model)
    model.train()

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=IS_CUDA)

    # LR schedule: smooth token-based linear warmup + cosine decay.
    # Bound definitions to absolute batch quantities so schedule shape remains
    # invariant across different sample ratios and creates a smooth curve inside long epochs.
    effective_ratio = config.sample_ratio
    warmup_epochs = max(1, round(5 / effective_ratio))
    total_lr_epochs = round(100 / effective_ratio)
    min_lr_ratio = 0.01  # decay to 1% of peak LR

    warmup_batches = warmup_epochs * n_total_batches
    total_lr_batches = total_lr_epochs * n_total_batches

    def _lr_lambda(current_batch: int) -> float:
        if current_batch < warmup_batches:
            return (current_batch + 1) / max(1, warmup_batches)
        progress = (current_batch - warmup_batches) / max(1, total_lr_batches - warmup_batches)
        progress = min(progress, 1.0)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
            1.0 + math.cos(math.pi * progress)
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # ── Restore from checkpoint if provided ──────────────────────────
    start_epoch = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint.model_state)
        optimizer.load_state_dict(checkpoint.optimizer_state)
        scaler.load_state_dict(checkpoint.scaler_state)
        scheduler.load_state_dict(checkpoint.scheduler_state)
        start_epoch = checkpoint.epoch + 1

    # ── Training loop ────────────────────────────────────────────────
    latest_metrics: Dict[str, float] = (
        checkpoint.latest_metrics
        if checkpoint is not None
        else {
            "bpd": float("inf"),
            "To-1": 0.0,
            "cos": 0.0,
            "loss": float("inf"),
        }
    )
    epoch_history: list = (
        list(checkpoint.epoch_history) if checkpoint is not None else []
    )
    cumulative_tokens_trained = int(latest_metrics.get("cumulative_tokens_trained", 0))
    cumulative_elapsed_ms = float(latest_metrics.get("cumulative_elapsed_ms", 0.0))

    epoch = max(0, start_epoch - 1)
    for epoch in range(start_epoch, config.epochs):
        on_epoch_start(epoch)
        t0 = time.perf_counter()
        total_loss_sum = 0.0
        epoch_total_bits = 0.0
        epoch_num_units = 0
        total_kl_sum = 0.0
        total_cov_sum = 0.0
        epoch_t1_correct = 0
        epoch_t1_units = 0
        epoch_cossim_sum = 0.0
        epoch_sharpness_sum = 0.0
        total_consistency_sum = 0.0
        epoch_cossim_pair_sum = 0.0
        epoch_pooled_std_sum = 0.0
        total_elements = 0
        n_batches = 0
        s1_count = 0
        s0_count = 0
        fuzzy_count = 0
        kc_prob_count = 0

        def _get_bin_label(length: int) -> str:
            if length <= 3: return "1-3"
            if length <= 7: return "4-7"
            if length <= 15: return "8-15"
            if length <= 31: return "16-31"
            return "32+"

        bin_labels = ["1-3", "4-7", "8-15", "16-31", "32+"]
        s1_count_by_bin = {k: 0 for k in bin_labels}
        s0_count_by_bin = {k: 0 for k in bin_labels}
        fuzzy_count_by_bin = {k: 0 for k in bin_labels}
        kc_prob_count_by_bin = {k: 0 for k in bin_labels}
        raw_consistency_sum = 0.0
        logit_abs_sum = 0.0
        logit_sq_sum = 0.0
        logit_sum = 0.0
        logit_count = 0

        for batch in loader:
            ids = batch.feature_inputs["input_ids_surface"].to(device, non_blocking=IS_CUDA)
            recon_targets = ids
            attention_mask = batch.attention_mask.to(device, non_blocking=IS_CUDA)

            B_actual = ids.size(0)

            if config.consistency_weight > 0:
                # Instantly double the batch dimension so the rest of the loop vectorize-processes 
                # exactly two distinct masked variations of each sentence concurrently.
                recon_targets = torch.cat([recon_targets, recon_targets], dim=0)
                attention_mask = torch.cat([attention_mask, attention_mask], dim=0)

            B = recon_targets.size(0)
            T = attention_mask.shape[1]
            maskable = attention_mask.bool() if config.input_mask_ratio > 0 else None

            def _apply_mask(ids: torch.Tensor) -> torch.Tensor:
                if maskable is None:
                    return ids
                rand_mask = (
                    torch.rand_like(ids.float()) < config.input_mask_ratio
                ) & maskable
                return ids.masked_fill(rand_mask, 0)

            surface_ids = _apply_mask(recon_targets)

            # ── Forward (under autocast, matching KCTrainer) ─────────
            with AUTOCAST():
                pooled = model.encode(surface_ids, attention_mask)
                # Track the standard deviation of encoder embeddings to verify depth invariance
                epoch_pooled_std_sum += float(pooled.std(dim=-1).mean().item()) * B_actual
                kc_logits_raw, _ = model.kc_head.forward_with_raw(pooled)

                # ── Dual-mask consistency regularization ──────────────
                consistency_loss = torch.tensor(0.0, device=device)
                if config.consistency_weight > 0:
                    half = B // 2
                    cos = F.cosine_similarity(
                        kc_logits_raw[:half],
                        kc_logits_raw[half:],
                        dim=-1,
                    )
                    consistency_loss = (1.0 - cos).mean()
                    epoch_cossim_pair_sum += cos.detach().mean().item() * half
                    raw_consistency_sum += consistency_loss.detach().item()

                # Gumbel noise + gradient capping
                u = torch.rand_like(kc_logits_raw).clamp_(1e-6, 1 - 1e-6)
                g = -torch.log(-torch.log(u))
                logits_select = kc_logits_raw + 0.6 * g

                if kc_logits_raw.requires_grad:
                    _cap = config.grad_cap
                    kc_logits_raw.register_hook(
                        lambda grad, c=_cap: grad.clamp(min=-c, max=c)
                    )

                logits_select = logits_select.clamp(-12, 12)
                kc_probs = torch.sigmoid(logits_select / config.temperature)
                kc_probs = torch.nan_to_num(kc_probs, nan=0.0, posinf=1.0, neginf=0.0)

                # Bernoulli entropy of KC probs: 0 = pure binary, 1 = all at 0.5
                _p = kc_probs.detach().clamp(1e-7, 1 - 1e-7)
                _h = -_p * torch.log2(_p) - (1 - _p) * torch.log2(1 - _p)
                epoch_sharpness_sum += (1.0 - _h.mean().item()) * B_actual

                # s0/fuzzy/s1 sharpness (matches kc_trainer_view thresholds)
                _det = kc_probs.detach()
                s1_mask = _det > 0.9
                s0_mask = _det < 0.1
                fuzzy_mask = (_det >= 0.2) & (_det <= 0.8)

                s1_count += int(s1_mask.sum().item())
                s0_count += int(s0_mask.sum().item())
                fuzzy_count += int(fuzzy_mask.sum().item())
                kc_prob_count += _det.numel()

                # Binned accumulation by sequence length
                s1_per_row = s1_mask.sum(dim=1).cpu().tolist()
                s0_per_row = s0_mask.sum(dim=1).cpu().tolist()
                fuzzy_per_row = fuzzy_mask.sum(dim=1).cpu().tolist()
                lengths = attention_mask.sum(dim=1).cpu().tolist()
                V = _det.size(1)

                for i, length in enumerate(lengths):
                    label = _get_bin_label(length)
                    s1_count_by_bin[label] += s1_per_row[i]
                    s0_count_by_bin[label] += s0_per_row[i]
                    fuzzy_count_by_bin[label] += fuzzy_per_row[i]
                    kc_prob_count_by_bin[label] += V

                # KC logit magnitude stats
                _logits_det = kc_logits_raw.detach()
                logit_abs_sum += _logits_det.abs().sum().item()
                logit_sq_sum += (_logits_det**2).sum().item()
                logit_sum += _logits_det.sum().item()
                logit_count += _logits_det.numel()

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
                    h_recon,
                    out_weight,
                    ce_targets,
                    reduction="none",
                )
                total_nll_nats = nll_per_token.sum()

                # Top-1 + cosine sim every 100 batches (forward pass is expensive)
                if n_batches % 100 == 0:
                    batch_units = int(mask_f.sum().item())
                    epoch_t1_units += batch_units
                    with torch.no_grad():
                        for c0 in range(0, T, recon_chunk):
                            c1 = min(c0 + recon_chunk, T)
                            with AUTOCAST():
                                chunk_logits = F.linear(
                                    h_recon[:, c0:c1, :], out_weight
                                )
                            preds = chunk_logits.argmax(dim=-1)
                            chunk_mask = attention_mask[:, c0:c1].bool()
                            epoch_t1_correct += int(
                                ((preds == recon_targets[:, c0:c1]) & chunk_mask)
                                .sum()
                                .item()
                            )
                            pred_emb = chive_normed[preds]
                            tgt_emb = chive_normed[recon_targets[:, c0:c1]]
                            cos = (pred_emb * tgt_emb).sum(dim=-1)
                            epoch_cossim_sum += float((cos * chunk_mask).sum().item())
            else:
                # Chunked fallback (MPS / no CCE): each chunk is
                # [B, RECON_CHUNK, V] to bound peak memory.
                total_nll_nats = torch.tensor(0.0, device=device)
                for c0 in range(0, T, recon_chunk):
                    c1 = min(c0 + recon_chunk, T)
                    with AUTOCAST():
                        chunk_logits = F.linear(h_recon[:, c0:c1, :], out_weight)
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
                        valid = chunk_mask.bool()
                        epoch_t1_correct += int(
                            ((preds == chunk_targets) & valid).sum().item()
                        )
                        pred_emb = chive_normed[preds]
                        tgt_emb = chive_normed[chunk_targets]
                        cos = (pred_emb * tgt_emb).sum(dim=-1)
                        epoch_cossim_sum += float((cos * chunk_mask).sum().item())

            # nats → bits, normalize by attended token count
            # Primary run-to-run fitness metric.  Lower is better.
            total_bits = total_nll_nats / LOG2
            num_units = mask_f.sum().clamp_min(1)
            bpd = total_bits / num_units

            # ── Regularizers ─────────────────────────────────────────
            loss = bpd
            kl_contrib = 0.0
            cov_contrib = 0.0
            consist_contrib = 0.0

            if config.consistency_weight > 0:
                consist_scaled = config.consistency_weight * consistency_loss
                loss = loss + consist_scaled
                consist_contrib = consist_scaled.item()

            if config.kl_sparse_weight > 0:
                rho_hat = kc_probs.mean(dim=0).clamp(1e-7, 1 - 1e-7)
                rho = config.kl_target_rho
                kl_term = (
                    rho_hat * torch.log(rho_hat / rho)
                    + (1 - rho_hat) * torch.log((1 - rho_hat) / (1 - rho))
                ).sum()
                kl_scaled = config.kl_sparse_weight * kl_term
                loss = loss + kl_scaled
                kl_contrib = kl_scaled.item()

            if config.cov_penalty_weight > 0:
                centered = kc_probs - kc_probs.mean(dim=0)
                cov = (centered.T @ centered) / max(1, B)
                cov.fill_diagonal_(0.0)
                cov_term = (cov**2).mean()
                cov_scaled = config.cov_penalty_weight * cov_term
                loss = loss + cov_scaled
                cov_contrib = cov_scaled.item()

            # ── Backward + step ──────────────────────────────────────
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Step the smooth per-batch scheduler
            scheduler.step()

            # ── Epoch stats ──────────────────────────────────────────
            batch_units = int(num_units.item())
            total_loss_sum += loss.item()
            epoch_total_bits += total_bits.item()
            epoch_num_units += batch_units
            total_kl_sum += kl_contrib
            total_cov_sum += cov_contrib
            total_consistency_sum += consist_contrib
            total_elements += B_actual
            n_batches += 1

            del loss, h_recon, total_nll_nats, total_bits, bpd, consistency_loss
            if device.type == "mps" and n_batches % 8 == 0:
                torch.mps.empty_cache()

            dt_batch = time.perf_counter() - t0
            t1_denom = epoch_t1_units if USE_FUSED_CE else epoch_num_units
            t1_pct = 100.0 * epoch_t1_correct / max(1, t1_denom)
            avg_cos = epoch_cossim_sum / max(1, t1_denom)
            print(
                f"\r  batch {n_batches}/{n_total_batches}  "
                f"bpd={epoch_total_bits / max(1, epoch_num_units):.4f}  "
                f"To-1={t1_pct:.1f}%  "
                f"cos={avg_cos:.3f}  "
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
        avg_consist = total_consistency_sum / max(1, n_batches)
        avg_pair_cos = epoch_cossim_pair_sum / max(1, total_elements)
        avg_pooled_std = epoch_pooled_std_sum / max(1, total_elements)
        t1_denom = epoch_t1_units if USE_FUSED_CE else epoch_num_units
        t1_pct = 100.0 * epoch_t1_correct / max(1, t1_denom)
        avg_cos = epoch_cossim_sum / max(1, t1_denom)
        avg_sharpness = epoch_sharpness_sum / max(1, total_elements)
        s1_pct = s1_count / max(1, kc_prob_count)
        s0_pct = s0_count / max(1, kc_prob_count)
        fuzzy_pct = fuzzy_count / max(1, kc_prob_count)
        avg_raw_consist = raw_consistency_sum / max(1, n_batches)
        mean_abs_logit = logit_abs_sum / max(1, logit_count)
        logit_std = (
            logit_sq_sum / max(1, logit_count) - (logit_sum / max(1, logit_count)) ** 2
        ) ** 0.5
        current_lr = scheduler.get_last_lr()[0]
        els = total_elements / dt

        cumulative_tokens_trained += epoch_num_units
        cumulative_elapsed_ms += (dt * 1000.0)

        latest_metrics = {
            "bpd": avg_bpd,
            "To-1": t1_pct,
            "cos": avg_cos,
            "sharp": avg_sharpness,
            "s1": s1_pct,
            "s0": s0_pct,
            "fuzzy": fuzzy_pct,
        }

        for label in bin_labels:
            mlflow_label = label.replace("+", "_plus")
            n = max(1, kc_prob_count_by_bin[label])
            latest_metrics[f"s1_{mlflow_label}"] = s1_count_by_bin[label] / n
            latest_metrics[f"s0_{mlflow_label}"] = s0_count_by_bin[label] / n
            latest_metrics[f"fuzzy_{mlflow_label}"] = fuzzy_count_by_bin[label] / n

        latest_metrics.update({
            "raw_consistency": avg_raw_consist,
            "mean_abs_logit": mean_abs_logit,
            "logit_std": logit_std,
            "pooled_std": avg_pooled_std,
            "loss": avg_loss,
            "sparsity": avg_kl,
            "orthogonality": avg_cov,
            "consistency": avg_consist,
            "mask-agree": avg_pair_cos,
            "lr": current_lr,
            "el_per_sec": els,
            "samples": total_elements,
            "epoch_secs": dt,
            "tokens_trained": epoch_num_units,
            "cumulative_tokens_trained": cumulative_tokens_trained,
            "elapsed_ms": cumulative_elapsed_ms,
        })

        print()  # finish \r progress line

        # Per-epoch checkpoint save (before callback, so pruned trials
        # still have their checkpoint persisted for later reuse).
        epoch_history.append((epoch, dict(latest_metrics)))
        save_checkpoint(
            TrainCheckpoint(
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scaler_state=scaler.state_dict(),
                scheduler_state=scheduler.state_dict(),
                epoch=epoch,
                latest_metrics=latest_metrics,
                epoch_history=epoch_history,
            ),
            checkpoint_path,
        )

        on_epoch_end(epoch, latest_metrics)

    final_checkpoint = TrainCheckpoint(
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scaler_state=scaler.state_dict(),
        scheduler_state=scheduler.state_dict(),
        epoch=epoch,
        latest_metrics=latest_metrics,
        epoch_history=epoch_history,
    )
    return (
        TrainResult(
            final_bpd=latest_metrics["bpd"],
            final_top1_pct=latest_metrics["To-1"],
            final_cossim=latest_metrics["cos"],
            final_loss=latest_metrics["loss"],
        ),
        final_checkpoint,
    )
