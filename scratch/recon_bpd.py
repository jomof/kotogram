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
import time
import warnings
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
from scratch.recon_bpd_checkpoint import (
    EpochContext,
    TrainCheckpoint,
    save_checkpoint,
)
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
    lr: float = 3e-4  # Original lr: 1e-4
    temperature: float = 1.2  # Original temperature: 1.8
    # Temperature annealing: start warm (soft KC probs) and cool to
    # target temperature over this many effective epochs.
    # Jang, Gu & Poole, "Categorical Reparameterization with
    # Gumbel-Softmax," ICLR 2017
    # Maddison, Mnih & Teh, "The Concrete Distribution," ICLR 2017
    temperature_start_multiplier: float = 3.0  # initial temp = temperature * this
    temperature_anneal_epochs: float = 30.0  # effective epochs to reach target temp
    weight_decay: float = 0.01
    grad_cap: float = 5.0  # Original grad_cap: 5.0
    input_mask_ratio: float = 0.15  # Original input_mask_ratio: 0.15
    seed: int = 42  # Original seed: 42

    # Regularization
    kl_sparse_weight: float = 0.0001  # Original kl_sparse_weight: 0.0001
    kl_target_rho: float = 0.06  # Original kl_target_rho: 0.03
    cov_penalty_weight: float = 5.0  # Original cov_penalty_weight: 5.0
    consistency_weight: float = 0.0001  # dual-mask KC consistency (0 = disabled)
    # VICReg regularization on encoder pooled output
    # Bardes, Ponce & LeCun, "VICReg: Variance-Invariance-Covariance
    # Regularization for Self-Supervised Learning," ICLR 2022
    vicreg_var_weight: float = 11.0  # variance term coefficient
    vicreg_cov_weight: float = 5.0  # covariance term coefficient
    vicreg_gamma: float = 0.3  # target std per dimension
    # Sentence length prediction from KC vector (diagnostic head).
    # Low weight: this is primarily a diagnostic, not a training driver.
    length_pred_weight: float = 0.01

    # Model architecture
    d_model: int = 512  # Original d_model: 512
    ffn_dim: int = 2048  # Original ffn_dim: 2048
    num_layers: int = 2  # Original num_layers: 4
    num_heads: int = 16  # Original num_heads: 16
    dropout: float = 0.1  # Original dropout: 0.1
    kc_vocab_size: int = 1024  # Original kc_vocab_size: 1024
    recon_pos_embed_dim: int = 64  # Original recon_pos_embed_dim: 64
    recon_hidden_dim: int = 256  # Original recon_hidden_dim: 256
    # Stochastic depth: probability of dropping each encoder layer
    # Fan et al., "Reducing Transformer Depth on Demand with
    # Structured Dropout," ICLR 2020
    layer_drop_prob: float = 0.5
    semantic_gating_threshold: float = 0.85  # Set to > 0.0 to enable throughput skips


@dataclass
class TrainResult:
    """Metrics returned from a training run."""

    final_bpd: float
    final_top1_pct: float
    final_cossim: float
    final_loss: float


EpochEndCallback = Callable[[int, Dict[str, float], EpochContext], None]
EpochStartCallback = Callable[[int], None]


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

    Dual positional encoding: end-relative (0 = last content token) and
    start-relative (0 = first content token).  Together they implicitly
    encode both absolute position AND sentence length — a token at
    start_rel=2, end_rel=5 is the third token in an 8-token sentence.

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
        # End-relative position: 0 = last content token, 1 = second-to-last, ...
        self.pos_embed_end = nn.Embedding(max_seq_len, pos_embed_dim)
        # Start-relative position: 0 = first content token, 1 = second, ...
        # Together with end-relative, these implicitly encode both
        # absolute position and sentence length without a separate signal.
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
        B, T = attention_mask.shape
        lengths = attention_mask.bool().sum(dim=1)
        abs_pos = torch.arange(T, device=kc_probs.device).unsqueeze(0).expand(B, -1)

        # End-relative: 0 = last content token, counting backward
        end_rel = (lengths.unsqueeze(1) - 1 - abs_pos).clamp(min=0)
        pos_emb_end = self.pos_embed_end(end_rel)

        # Start-relative: 0 = first token, counting forward
        # Clamp to length-1 so padding positions don't get
        # out-of-bounds indices (they'll be masked out anyway).
        start_rel = abs_pos.clamp(max=lengths.unsqueeze(1) - 1)
        pos_emb_start = self.pos_embed_start(start_rel)

        kc_exp = kc_probs.unsqueeze(1).expand(-1, T, -1)
        h = torch.cat([kc_exp, pos_emb_end, pos_emb_start], dim=-1)
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
            nn.init.normal_(
                layer.self_attn.out_proj.weight, mean=0.0, std=0.02 * std_scale
            )
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

        # Diagnostic head: predict sentence length from KC probs alone.
        # Simple MLP: 1024 → 128 → 1. Detects whether the KC bottleneck
        # encodes sentence length information.
        self.length_head = nn.Sequential(
            nn.Linear(cfg.kc_vocab_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
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
        # ── Stochastic Depth (LayerDrop) ────────────────────────
        # Fan et al., "Reducing Transformer Depth on Demand with
        # Structured Dropout," ICLR 2020
        #
        # Randomly skip entire transformer layers during training.
        # This prevents the over-smoothing cascade where deep layers
        # progressively erase token-level distinctions, and ensures
        # representations are robust at every effective depth.
        pad_mask = attention_mask == 0
        if self.training and self.cfg.layer_drop_prob > 0:
            for i, layer in enumerate(self.encoder.layers):
                if i > 0 and torch.rand(1).item() < self.cfg.layer_drop_prob:
                    continue
                x = layer(x, src_key_padding_mask=pad_mask)
        else:
            x = self.encoder(x, src_key_padding_mask=pad_mask)
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
        self.content_mask: Optional[torch.Tensor] = None


GLOBAL_SETUP_CACHE = _SetupCache()


def train(
    config: TrainConfig,
    on_epoch_start: EpochStartCallback,
    on_epoch_end: EpochEndCallback,
    checkpoint_path: str,
    checkpoint: Optional[TrainCheckpoint] = None,
    run_name: str = "",
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

    if GLOBAL_SETUP_CACHE.content_mask is None:
        mask_path = f"{train_paths.get_style_dataset_cache_dir()}/content_mask.bin"
        GLOBAL_SETUP_CACHE.content_mask = torch.from_file(
            mask_path,
            shared=True,
            size=tokenizer.get_vocab_sizes()["surface"],
            dtype=torch.uint8,
        ).bool()

    content_mask_tensor = GLOBAL_SETUP_CACHE.content_mask.to(
        device, non_blocking=IS_CUDA
    )

    if sample_ratio not in GLOBAL_SETUP_CACHE.dataset_subsets:
        cache_dir = train_paths.get_style_dataset_cache_dir()
        dataset = StyleDataset(
            cache_dir,
            tokenizer,
            sample_ratio=sample_ratio,
        )
        GLOBAL_SETUP_CACHE.dataset_subsets[sample_ratio] = (
            dataset.filter_by_grammaticality(label=1)
        )
        print(
            f"Gram sentences: {len(GLOBAL_SETUP_CACHE.dataset_subsets[sample_ratio])}"
        )
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
        layer_drop_prob=config.layer_drop_prob,
    )
    # Load chiVe pretrained surface embeddings and freeze
    if GLOBAL_SETUP_CACHE.chive_weights is None:
        cw = load_chive_for_vocab(tokenizer.field_vocabs["surface"])

        # Missing chiVe words are all zeros. Randomize them so OOV words aren't squashed together.
        norms = cw.norm(dim=-1)
        missing_mask = norms == 0.0
        missing_mask[0] = False  # Keep index 0 [PAD] strictly at zero

        if missing_mask.any():
            # Match the variance of the known chiVe vectors
            std = cw[~missing_mask].std().item() if (~missing_mask).any() else 0.1
            cw[missing_mask] = torch.randn_like(cw[missing_mask]) * std

        GLOBAL_SETUP_CACHE.chive_weights = cw
    chive_weights = GLOBAL_SETUP_CACHE.chive_weights

    if (
        GLOBAL_SETUP_CACHE.cached_model_cfg == cfg
        and GLOBAL_SETUP_CACHE.cached_model is not None
    ):
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

    optimizer = AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
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
        progress = (current_batch - warmup_batches) / max(
            1, total_lr_batches - warmup_batches
        )
        progress = min(progress, 1.0)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
            1.0 + math.cos(math.pi * progress)
        )

    # LambdaLR.__init__ calls step() internally, triggering a spurious
    # "scheduler.step() before optimizer.step()" warning.  The actual
    # training loop has the correct order (scaler.step → scheduler.step).
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "Detected call of .lr_scheduler.step", UserWarning
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # ── Restore from checkpoint if provided ──────────────────────────
    start_epoch = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint.model_state, strict=False)
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
    cumulative_elapsed_ms = float(latest_metrics.get("elapsed_ms", 0.0))

    epoch = max(0, start_epoch - 1)
    for epoch in range(start_epoch, config.epochs):
        # Deterministic per-epoch seed: same epoch always sees same
        # batch order and stochastic ops, even after resume.
        epoch_seed = config.seed + epoch
        torch.manual_seed(epoch_seed)
        if IS_CUDA:
            torch.cuda.manual_seed(epoch_seed)
        dl_generator.manual_seed(epoch_seed)
        on_epoch_start(epoch)
        t0 = time.perf_counter()
        # All accumulators on GPU — single .item() batch at epoch end
        total_loss_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
        epoch_total_bits = torch.tensor(0.0, device=device, dtype=torch.float32)
        epoch_num_units = torch.tensor(0, device=device, dtype=torch.long)
        total_kl_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
        total_cov_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
        epoch_t1_correct = 0
        epoch_t1_units = 0
        epoch_cossim_sum = 0.0
        epoch_sharpness_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
        total_consistency_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
        total_vicreg_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
        epoch_cossim_pair_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
        epoch_pooled_std_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
        total_elements = 0
        n_batches = 0
        s1_count = torch.tensor(0, device=device, dtype=torch.long)
        s0_count = torch.tensor(0, device=device, dtype=torch.long)
        fuzzy_count = torch.tensor(0, device=device, dtype=torch.long)
        kc_prob_count = torch.tensor(0, device=device, dtype=torch.long)

        bin_labels = ["1-3", "4-7", "8-15", "16-31", "32+"]
        _NUM_BINS = len(bin_labels)
        # Bin boundaries on GPU for torch.bucketize: (0,3], (3,7], (7,15], (15,31], (31,inf]
        _bin_edges = torch.tensor([3, 7, 15, 31], device=device, dtype=torch.float32)
        s1_count_by_bin_t = torch.zeros(_NUM_BINS, device=device, dtype=torch.float32)
        s0_count_by_bin_t = torch.zeros(_NUM_BINS, device=device, dtype=torch.float32)
        fuzzy_count_by_bin_t = torch.zeros(
            _NUM_BINS, device=device, dtype=torch.float32
        )
        kc_prob_count_by_bin_t = torch.zeros(
            _NUM_BINS, device=device, dtype=torch.float32
        )
        bpd_bits_by_bin_t = torch.zeros(_NUM_BINS, device=device, dtype=torch.float32)
        bpd_tokens_by_bin_t = torch.zeros(_NUM_BINS, device=device, dtype=torch.float32)
        raw_consistency_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
        logit_abs_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
        logit_sq_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
        logit_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
        logit_count = torch.tensor(0, device=device, dtype=torch.long)

        total_length_pred_loss_sum = torch.tensor(
            0.0, device=device, dtype=torch.float32
        )
        total_length_pred_mae_sum = torch.tensor(
            0.0, device=device, dtype=torch.float32
        )
        total_length_pred_count = 0

        total_semantic_loss_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
        total_semantic_tokens = torch.tensor(0, device=device, dtype=torch.long)
        total_semantic_skipped = torch.tensor(0, device=device, dtype=torch.long)

        # Dynamic Semantic Thresholding (1.0 -> 0.85 over 30 effective epochs)
        # Scales cleanly by sample_ratio to match LR warmup geometry.
        base_threshold = config.semantic_gating_threshold
        if 0.0 < base_threshold < 1.0:
            eff_epochs = epoch * config.sample_ratio
            current_threshold = max(base_threshold, 1.0 - 0.005 * eff_epochs)
        else:
            current_threshold = base_threshold

        # ── Temperature Annealing ───────────────────────────────
        # Jang, Gu & Poole, "Categorical Reparameterization with
        # Gumbel-Softmax," ICLR 2017
        # Maddison, Mnih & Teh, "The Concrete Distribution,"
        # ICLR 2017
        #
        # Start with high temperature (soft, high-entropy KC probs)
        # and anneal down to the target temperature. This gives the
        # length-proportional KL sparsity time to organize logit
        # allocation across sentence lengths BEFORE the sigmoid
        # sharpens assignments into irreversible binary decisions.
        eff_epoch = epoch * config.sample_ratio
        if eff_epoch < config.temperature_anneal_epochs:
            temp_start = config.temperature * config.temperature_start_multiplier
            temp_end = config.temperature
            anneal_progress = eff_epoch / config.temperature_anneal_epochs
            current_temperature = temp_start + (temp_end - temp_start) * anneal_progress
        else:
            current_temperature = config.temperature

        # ── KL Sparsity Warmup ──────────────────────────────────
        # Ramp the KL weight from ~0 to full strength on the same
        # schedule as temperature annealing. Quadratic ramp: near-
        # zero for the first ~half of the anneal period, then
        # accelerates to full strength as temperature cools and
        # assignments begin to sharpen. This ensures the length-
        # proportional allocation is negotiated under soft probs.
        if eff_epoch < config.temperature_anneal_epochs:
            kl_warmup = (eff_epoch / config.temperature_anneal_epochs) ** 2
        else:
            kl_warmup = 1.0

        for batch in loader:
            ids = batch.feature_inputs["input_ids_surface"].to(
                device, non_blocking=IS_CUDA
            )
            attention_mask = batch.attention_mask.to(device, non_blocking=IS_CUDA)

            # ── Data Augmentation: Drop Non-Content Tokens (50%) ──────
            is_content = content_mask_tensor[ids]
            # Must also protect special tokens (ids < 4) from being dropped
            is_non_content_valid = (~is_content) & attention_mask.bool() & (ids >= 4)
            drop_mask = is_non_content_valid & (torch.rand_like(ids.float()) < 0.5)

            # Drop means turning off the attention mask and replacing the ID with PAD
            ids = ids.masked_fill(drop_mask, 0)
            attention_mask = attention_mask.masked_fill(drop_mask, 0)

            recon_targets = ids
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
                epoch_pooled_std_sum += pooled.detach().std(dim=-1).sum().float()

                half = B // 2

                # ── VICReg: Variance-Covariance Regularization ──────
                # Bardes, Ponce & LeCun, "VICReg: Variance-Invariance-
                # Covariance Regularization for Self-Supervised
                # Learning," ICLR 2022
                #
                # Variance term: prevent dimensional collapse by
                # requiring each dimension to maintain std >= gamma.
                # Covariance term: decorrelate dimensions to prevent
                # redundant encoding.
                # Applied to the pooled encoder output BEFORE the KC
                # head, so the upstream representation is forced to
                # remain high-rank and variable across the batch.
                vicreg_loss = torch.tensor(0.0, device=device)
                if config.vicreg_var_weight > 0 or config.vicreg_cov_weight > 0:
                    # Use only the first view to avoid double-counting
                    z = pooled[:half] if config.consistency_weight > 0 else pooled

                    # Cast to float32 for numerical stability in
                    # variance/covariance computation. AUTOCAST may
                    # use TF32/FP16 which causes noisy squared terms.
                    z = z.float()
                    z_centered = z - z.mean(dim=0)

                    # Variance: hinge loss on per-dimension std
                    std_z = torch.sqrt(z.var(dim=0) + 1e-4)
                    var_loss = F.relu(config.vicreg_gamma - std_z).mean()

                    # Covariance: penalize off-diagonal correlations
                    n = max(1, z.size(0) - 1)
                    cov_matrix = (z_centered.T @ z_centered) / n
                    cov_matrix.fill_diagonal_(0.0)
                    cov_loss = (cov_matrix**2).mean()

                    vicreg_loss = (
                        config.vicreg_var_weight * var_loss
                        + config.vicreg_cov_weight * cov_loss
                    )

                kc_logits_raw, _ = model.kc_head.forward_with_raw(pooled)

                # ── Dual-mask consistency regularization ──────────────
                consistency_loss = torch.tensor(0.0, device=device)
                if config.consistency_weight > 0:
                    # ── Stop-gradient consistency (asymmetric + symmetrized) ────
                    # Grill et al., "Bootstrap Your Own Latent" (BYOL), NeurIPS 2020
                    # Chen & He, "Exploring Simple Siamese Representation Learning"
                    # (SimSiam), CVPR 2021
                    #
                    # Applying stop_grad to one branch eliminates the symmetric
                    # collapse attractor where both views converge to a constant.
                    # Symmetrizing ensures neither branch is privileged.
                    cos_ab = F.cosine_similarity(
                        kc_logits_raw[:half],
                        kc_logits_raw[half:].detach(),  # stop-gradient on view B
                        dim=-1,
                    )
                    cos_ba = F.cosine_similarity(
                        kc_logits_raw[:half].detach(),  # stop-gradient on view A
                        kc_logits_raw[half:],
                        dim=-1,
                    )
                    consistency_loss = (
                        0.5 * (1.0 - cos_ab).mean() + 0.5 * (1.0 - cos_ba).mean()
                    )
                    epoch_cossim_pair_sum += (
                        (0.5 * (cos_ab + cos_ba)).detach().sum().float()
                    )
                    raw_consistency_sum += consistency_loss.detach().float()

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
                kc_probs = torch.sigmoid(logits_select / current_temperature)
                kc_probs = torch.nan_to_num(kc_probs, nan=0.0, posinf=1.0, neginf=0.0)

                # Bernoulli entropy of KC probs: 0 = pure binary, 1 = all at 0.5
                _p = kc_probs.detach().clamp(1e-7, 1 - 1e-7)
                _h = -_p * torch.log2(_p) - (1 - _p) * torch.log2(1 - _p)
                epoch_sharpness_sum += (1.0 - _h.mean()).float() * B_actual

                # s0/fuzzy/s1 sharpness (matches kc_trainer_view thresholds)
                _det = kc_probs.detach()
                s1_mask = _det > 0.9
                s0_mask = _det < 0.1
                fuzzy_mask = (_det >= 0.2) & (_det <= 0.8)

                s1_count += s1_mask.sum()
                s0_count += s0_mask.sum()
                fuzzy_count += fuzzy_mask.sum()
                kc_prob_count += _det.numel()

                # Binned accumulation by sequence length — fully on GPU
                s1_per_row = s1_mask.sum(dim=1)  # [B]
                s0_per_row = s0_mask.sum(dim=1)  # [B]
                fuzzy_per_row = fuzzy_mask.sum(dim=1)  # [B]
                lengths_gpu = attention_mask.sum(dim=1).float()  # [B]
                V = _det.size(1)

                bin_idx = torch.bucketize(
                    lengths_gpu, _bin_edges
                )  # [B], values in 0.._NUM_BINS-1
                s1_count_by_bin_t.scatter_add_(0, bin_idx.long(), s1_per_row.float())
                s0_count_by_bin_t.scatter_add_(0, bin_idx.long(), s0_per_row.float())
                fuzzy_count_by_bin_t.scatter_add_(
                    0, bin_idx.long(), fuzzy_per_row.float()
                )
                kc_prob_count_by_bin_t.scatter_add_(
                    0,
                    bin_idx.long(),
                    torch.full_like(bin_idx, V, dtype=torch.float32),
                )

                # KC logit magnitude stats — stay on GPU
                _logits_det = kc_logits_raw.detach()
                logit_abs_sum += _logits_det.abs().sum().float()
                logit_sq_sum += (_logits_det**2).sum().float()
                logit_sum += _logits_det.sum().float()
                logit_count += _logits_det.numel()

                # Recon decoder: pre-logit hidden states [B, T, H]
                h_recon = model.recon.forward_hidden(kc_probs, attention_mask)

                # ── Sentence length prediction (diagnostic) ─────────
                # Predict number of content tokens from KC probs alone.
                # Uses the raw KC probs (not logits) as input since
                # that's the representation the recon decoder sees.
                length_pred_loss = torch.tensor(0.0, device=device)
                if config.length_pred_weight > 0:
                    pred_lengths = model.length_head(
                        kc_probs.detach()
                        if config.length_pred_weight < 0.1
                        else kc_probs
                    ).squeeze(-1)
                    # Use only first half if consistency doubling is active
                    if config.consistency_weight > 0:
                        true_lengths_lp = attention_mask[:half].sum(dim=1).float()
                        pred_lengths_lp = pred_lengths[:half]
                    else:
                        true_lengths_lp = attention_mask.sum(dim=1).float()
                        pred_lengths_lp = pred_lengths
                    # Normalized MSE: divide by mean length squared so the
                    # loss magnitude is independent of sentence length scale
                    mean_len = true_lengths_lp.mean().clamp_min(1.0)
                    length_pred_loss = F.mse_loss(pred_lengths_lp, true_lengths_lp) / (
                        mean_len**2
                    )

            # ── Output projection + CE ────────────────────────────────
            assert h_recon.shape[:2] == recon_targets.shape
            assert attention_mask.shape == recon_targets.shape
            mask_f = attention_mask.float()

            out_weight = model.recon.output_head.weight

            nats_per_row = torch.zeros(B, device=device)
            semantic_distillation_loss = torch.tensor(0.0, device=device)

            if USE_FUSED_CE:
                # Fused linear CE: h_recon × W^T → CE in one kernel,
                # never materializes [B, T, V] in global memory.
                valid_mask = attention_mask.bool()
                ce_targets = recon_targets.where(
                    valid_mask, torch.tensor(-100, device=device)
                )

                threshold = current_threshold
                if threshold > 0.0:
                    # 1. Project hidden to 300D and normalize
                    with AUTOCAST():
                        pred_emb = model.recon.semantic_head(h_recon)
                    pred_emb = F.normalize(pred_emb.float(), p=2, dim=-1)

                    # 2. Get true 300D targets
                    tgt_emb = chive_normed[ce_targets.clamp(min=0)]

                    # 3. Calculate similarities
                    cos_sim = (pred_emb * tgt_emb).sum(dim=-1)

                    num_valid = valid_mask.sum()
                    total_semantic_tokens += num_valid

                    # 4. Auxiliary semantic loss for active tokens
                    sem_loss_sum = ((1.0 - cos_sim) * valid_mask.float()).sum()
                    semantic_distillation_loss = (
                        sem_loss_sum / num_valid.clamp_min(1).float()
                    )
                    total_semantic_loss_sum += sem_loss_sum.detach().float()

                    # 5. Stochastic semantic gating: deterministically
                    # keep hard tokens (cos_sim < threshold), and
                    # randomly rescue easy tokens with probability
                    # (1 - threshold). This drops most easy tokens
                    # like the original gate but softens the boundary.
                    is_easy = cos_sim >= threshold
                    rescue = torch.rand_like(cos_sim) > threshold
                    is_hard = (~is_easy | rescue) & valid_mask

                    num_hard = is_hard.sum()
                    total_semantic_skipped += num_valid - num_hard

                    flat_h = h_recon.reshape(-1, h_recon.size(-1))
                    flat_tgt = ce_targets.reshape(-1)
                    flat_hard = is_hard.reshape(-1)

                    h_hard = flat_h[flat_hard]
                    tgt_hard = flat_tgt[flat_hard]

                    if h_hard.size(0) > 0:
                        nll_hard = _cce_linear_ce(
                            h_hard, out_weight, tgt_hard, reduction="none"
                        )
                        total_nll_nats = nll_hard.sum()

                        # Retain row-level metric alignment manually
                        b_indices = (
                            torch.arange(B, device=device)
                            .unsqueeze(1)
                            .expand(-1, T)
                            .reshape(-1)
                        )
                        nats_per_row.scatter_add_(0, b_indices[flat_hard], nll_hard)
                    else:
                        total_nll_nats = torch.tensor(0.0, device=device)

                else:
                    nll_per_token = _cce_linear_ce(
                        h_recon,
                        out_weight,
                        ce_targets,
                        reduction="none",
                    )
                    total_nll_nats = nll_per_token.sum()
                    nats_per_row = nll_per_token.reshape(B, -1).sum(dim=1)

                # Top-1 + cosine sim deferred to epoch end (avoids [B,T,V] materialization)
            else:
                # Chunked fallback (MPS / no CCE): each chunk is
                # [B, RECON_CHUNK, V] to bound peak memory.
                # Uses ignore_index=-100 to match the CUDA path.
                valid_mask = attention_mask.bool()
                ce_targets = recon_targets.where(
                    valid_mask, torch.tensor(-100, device=device)
                )
                total_nll_nats = torch.tensor(0.0, device=device)
                for c0 in range(0, T, recon_chunk):
                    c1 = min(c0 + recon_chunk, T)
                    with AUTOCAST():
                        chunk_logits = F.linear(h_recon[:, c0:c1, :], out_weight)
                    chunk_logits = chunk_logits.float()
                    chunk_targets = ce_targets[:, c0:c1]
                    chunk_nll = F.cross_entropy(
                        chunk_logits.reshape(-1, chunk_logits.size(-1)),
                        chunk_targets.reshape(-1),
                        ignore_index=-100,
                        reduction="none",
                    ).reshape(B, -1)
                    total_nll_nats = total_nll_nats + chunk_nll.sum()
                    nats_per_row += chunk_nll.sum(dim=1)

                # Top-1 + cosine sim deferred to epoch end (same as CUDA path)

            # nats → bits, normalize by attended token count
            # Primary run-to-run fitness metric.  Lower is better.
            total_bits = total_nll_nats / LOG2
            num_units = mask_f.sum().clamp_min(1)
            bpd = total_bits / num_units

            bits_per_row = nats_per_row / LOG2  # [B], stays on GPU
            row_lengths = mask_f.sum(dim=1)  # [B], stays on GPU
            bpd_bin_idx = torch.bucketize(row_lengths, _bin_edges)  # [B]
            bpd_bits_by_bin_t.scatter_add_(0, bpd_bin_idx.long(), bits_per_row.float())
            bpd_tokens_by_bin_t.scatter_add_(0, bpd_bin_idx.long(), row_lengths.float())

            # ── Regularizers ─────────────────────────────────────────
            loss = bpd + semantic_distillation_loss * 5.0

            if config.consistency_weight > 0:
                consist_scaled = config.consistency_weight * consistency_loss
                loss = loss + consist_scaled
                total_consistency_sum += consist_scaled.detach().float()

            # VICReg contribution
            if config.vicreg_var_weight > 0 or config.vicreg_cov_weight > 0:
                loss = loss + vicreg_loss
                total_vicreg_sum += vicreg_loss.detach().float()

            # Length prediction diagnostic
            if config.length_pred_weight > 0:
                loss = loss + config.length_pred_weight * length_pred_loss
                total_length_pred_loss_sum += length_pred_loss.detach().float()
                # Also track MAE for interpretability (in token units)
                with torch.no_grad():
                    total_length_pred_mae_sum += (
                        (pred_lengths_lp - true_lengths_lp).abs().mean().float()
                    )
                    total_length_pred_count += 1

            if config.kl_sparse_weight > 0:
                # Length-proportional sparsity adjustment:
                # Short sentences are heavily penalized for turning on logits to prevent them
                # from monopolizing the capacity budget with "easy memorization" features.
                # A 15-token sentence acts as the 1.0 baseline.
                seq_lengths = (
                    attention_mask.sum(dim=1, keepdim=True).float().clamp_min(1.0)
                )
                length_penalty = 15.0 / seq_lengths

                norm_probs = kc_probs * length_penalty
                rho_hat = norm_probs.mean(dim=0).clamp(1e-7, 1 - 1e-7)

                rho = config.kl_target_rho
                kl_term = (
                    rho_hat * torch.log(rho_hat / rho)
                    + (1 - rho_hat) * torch.log((1 - rho_hat) / (1 - rho))
                ).sum()
                kl_scaled = config.kl_sparse_weight * kl_warmup * kl_term
                loss = loss + kl_scaled
                total_kl_sum += kl_scaled.detach().float()

            if config.cov_penalty_weight > 0:
                centered = kc_probs - kc_probs.mean(dim=0)
                cov = (centered.T @ centered) / max(1, B)
                cov.fill_diagonal_(0.0)
                cov_term = (cov**2).mean()
                cov_scaled = config.cov_penalty_weight * cov_term
                loss = loss + cov_scaled
                total_cov_sum += cov_scaled.detach().float()

            # ── Backward + step ──────────────────────────────────────
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            # Step the smooth per-batch scheduler
            scheduler.step()

            # ── Epoch stats (all GPU, zero sync) ─────────────────────
            total_loss_sum += loss.detach().float()
            epoch_total_bits += total_bits.detach().float()
            epoch_num_units += num_units.detach().long()
            total_elements += B_actual
            n_batches += 1

            del (
                loss,
                h_recon,
                total_nll_nats,
                total_bits,
                bpd,
                consistency_loss,
                vicreg_loss,
                length_pred_loss,
            )
            if device.type == "mps" and n_batches % 8 == 0:
                torch.mps.empty_cache()

            dt_batch = time.perf_counter() - t0
            print(
                f"\r  batch {n_batches}/{n_total_batches}  "
                f"{total_elements / dt_batch:.1f} el/s  "
                f"{dt_batch:.1f}s",
                end="",
                flush=True,
            )

        dt = time.perf_counter() - t0

        # ── Single GPU→CPU sync for ALL accumulators ─────────────
        _total_loss_val = total_loss_sum.item()
        _total_bits_val = epoch_total_bits.item()
        _num_units_val = int(epoch_num_units.item())
        _kl_val = total_kl_sum.item()
        _lp_loss_val = total_length_pred_loss_sum.item()
        _lp_mae_val = total_length_pred_mae_sum.item()
        _sem_loss_val = total_semantic_loss_sum.item()
        _sem_tokens_val = int(total_semantic_tokens.item())
        _sem_skipped_val = int(total_semantic_skipped.item())

        avg_bpd = _total_bits_val / max(1, _num_units_val)
        avg_loss = _total_loss_val / max(1, n_batches)
        avg_kl = _kl_val / max(1, n_batches)
        epoch_num_units = _num_units_val  # reassign for downstream compat

        # ── Epoch-end Top-1 + cosine sim (single evaluation pass) ──
        # Both CUDA and MPS paths defer Top-1 computation to epoch end
        # to avoid per-batch [B,T,V] materialization and GPU→CPU syncs.
        model.eval()
        with torch.no_grad():
            # Fresh forward pass on the last batch's data
            pooled_eval = model.encode(surface_ids, attention_mask)
            kc_logits_eval, _ = model.kc_head.forward_with_raw(pooled_eval)
            kc_probs_eval = torch.sigmoid(kc_logits_eval / current_temperature)
            h_recon_eval = model.recon.forward_hidden(kc_probs_eval, attention_mask)
            for c0 in range(0, T, recon_chunk):
                c1 = min(c0 + recon_chunk, T)
                with AUTOCAST():
                    chunk_logits = F.linear(h_recon_eval[:, c0:c1, :], out_weight)
                preds = chunk_logits.argmax(dim=-1)
                chunk_mask = attention_mask[:, c0:c1].bool()
                epoch_t1_correct += int(
                    ((preds == recon_targets[:, c0:c1]) & chunk_mask).sum().item()
                )
                pred_emb = chive_normed[preds]
                tgt_emb = chive_normed[recon_targets[:, c0:c1]]
                cos = (pred_emb * tgt_emb).sum(dim=-1)
                epoch_cossim_sum += float((cos * chunk_mask).sum().item())
            epoch_t1_units = int(attention_mask.sum().item())
        model.train()

        # Single GPU→CPU sync for all per-batch GPU accumulators
        _cov_val = total_cov_sum.item()
        _consist_val = total_consistency_sum.item()
        _cossim_pair_val = epoch_cossim_pair_sum.item()
        _pooled_std_val = epoch_pooled_std_sum.item()
        _sharpness_val = epoch_sharpness_sum.item()
        _s1_val = s1_count.item()
        _s0_val = s0_count.item()
        _fuzzy_val = fuzzy_count.item()
        _kc_prob_val = kc_prob_count.item()
        _raw_consist_val = raw_consistency_sum.item()
        _logit_abs_val = logit_abs_sum.item()
        _logit_sq_val = logit_sq_sum.item()
        _logit_sum_val = logit_sum.item()
        _logit_count_val = logit_count.item()
        _vicreg_val = total_vicreg_sum.item()

        avg_cov = _cov_val / max(1, n_batches)
        avg_consist = _consist_val / max(1, n_batches)
        avg_pair_cos = _cossim_pair_val / max(1, total_elements)
        avg_pooled_std = _pooled_std_val / max(1, total_elements)
        t1_denom = epoch_t1_units
        t1_pct = 100.0 * epoch_t1_correct / max(1, t1_denom)
        avg_cos = epoch_cossim_sum / max(1, t1_denom)
        avg_sharpness = _sharpness_val / max(1, total_elements)
        s1_pct = _s1_val / max(1, _kc_prob_val)
        s0_pct = _s0_val / max(1, _kc_prob_val)
        fuzzy_pct = _fuzzy_val / max(1, _kc_prob_val)
        avg_raw_consist = _raw_consist_val / max(1, n_batches)
        mean_abs_logit = _logit_abs_val / max(1, _logit_count_val)
        logit_std = (
            _logit_sq_val / max(1, _logit_count_val)
            - (_logit_sum_val / max(1, _logit_count_val)) ** 2
        ) ** 0.5
        current_lr = scheduler.get_last_lr()[0]
        els = total_elements / dt

        cumulative_tokens_trained += _num_units_val
        cumulative_elapsed_ms += dt * 1000.0

        latest_metrics = {
            "bpd": avg_bpd,
            "To-1": t1_pct,
            "cos": avg_cos,
            "sharp": avg_sharpness,
            "s1": s1_pct,
            "s0": s0_pct,
            "fuzzy": fuzzy_pct,
        }

        # Pull GPU bin accumulators to CPU once at epoch end
        _s1_bins = s1_count_by_bin_t.cpu().tolist()
        _s0_bins = s0_count_by_bin_t.cpu().tolist()
        _fuzzy_bins = fuzzy_count_by_bin_t.cpu().tolist()
        _kc_bins = kc_prob_count_by_bin_t.cpu().tolist()
        _bpd_bits_bins = bpd_bits_by_bin_t.cpu().tolist()
        _bpd_tok_bins = bpd_tokens_by_bin_t.cpu().tolist()
        for bi, label in enumerate(bin_labels):
            mlflow_label = label.replace("+", "_plus")
            n = max(1, _kc_bins[bi])
            latest_metrics[f"s1_{mlflow_label}"] = _s1_bins[bi] / n
            latest_metrics[f"s0_{mlflow_label}"] = _s0_bins[bi] / n
            latest_metrics[f"fuzzy_{mlflow_label}"] = _fuzzy_bins[bi] / n

            t = max(1.0, _bpd_tok_bins[bi])
            latest_metrics[f"bpd_{mlflow_label}"] = _bpd_bits_bins[bi] / t

        if _sem_tokens_val > 0:
            latest_metrics["semantic_distillation_loss"] = (
                _sem_loss_val / _sem_tokens_val
            )
            latest_metrics["semantic_skip_ratio"] = _sem_skipped_val / _sem_tokens_val

        latest_metrics.update(
            {
                "raw_consistency": avg_raw_consist,
                "mean_abs_logit": mean_abs_logit,
                "logit_std": logit_std,
                "pooled_std": avg_pooled_std,
                "loss": avg_loss,
                "sparsity": avg_kl,
                "orthogonality": avg_cov,
                "consistency": avg_consist,
                "mask-agree": avg_pair_cos,
                "vicreg": _vicreg_val / max(1, n_batches),
                "lr": current_lr,
                "temperature": current_temperature,
                "kl_warmup": kl_warmup,
                "semantic_threshold": current_threshold,
                "length_pred_mse": _lp_loss_val / max(1, total_length_pred_count),
                "length_pred_mae": _lp_mae_val / max(1, total_length_pred_count),
                "el_per_sec": els,
                "samples": total_elements,
                "epoch_secs": dt,
                "tokens_trained": _num_units_val,
                "cumulative_tokens_trained": cumulative_tokens_trained,
                "elapsed_ms": cumulative_elapsed_ms,
            }
        )

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

        ctx = EpochContext(
            model=model,
            tokenizer=tokenizer,
            device=device,
            temperature=current_temperature,
            checkpoint_path=checkpoint_path,
            config=config,
            run_name=run_name,
        )
        on_epoch_end(epoch, latest_metrics, ctx)

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
