"""Load a recon_bpd model from checkpoint.lock and run BPD inference.

Provides the scoring backend for the CC sentence-selection pipeline:
  - ``load_model_from_checkpoint()`` builds a BpdModel from the locked
    checkpoint + dataset, returning the model, tokenizer, and checkpoint ID.
  - ``compute_bpd()`` runs the full encode -> KC -> recon forward pass and
    returns per-sentence BPD (bits per token).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from scripts.checkpoint import LOCAL_CACHE, download_checkpoint, read_lock
from scripts.recon_bpd.model import BpdModel, BpdModelConfig


def infer_config_from_state(
    state: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """Recover BpdModelConfig fields from checkpoint tensor shapes.

    Works for both raw and torch.compile checkpoints (``_orig_mod.`` prefix).
    """

    def _get(suffix: str) -> torch.Tensor:
        for key, val in state.items():
            if key.endswith(suffix):
                return val
        raise KeyError(f"No key ending with {suffix!r} in state dict")

    surface_vocab_size = _get("surface_embed.weight").shape[0]
    surface_embed_dim = _get("surface_embed.weight").shape[1]
    d_model = _get("embed_proj.weight").shape[0]
    ffn_dim = _get("encoder.layers.0.linear1.weight").shape[0]
    kc_vocab_size = _get("kc_head.output.weight").shape[0]
    recon_hidden_dim = _get("recon.hidden1.weight").shape[0]
    recon_pos_embed_dim = _get("recon.pos_embed_end.weight").shape[1]

    layer_nums = set()
    for k in state:
        if "encoder.layers." in k:
            parts = k.split(".")
            idx = parts.index("layers") + 1
            if idx < len(parts) and parts[idx].isdigit():
                layer_nums.add(int(parts[idx]))
    num_layers = max(layer_nums) + 1 if layer_nums else 1

    # num_heads: in_proj_weight is (3*d_model, d_model), but head count is
    # not directly inferrable from shapes -- use common defaults.
    num_heads = 16 if d_model % 16 == 0 else 8

    return {
        "surface_vocab_size": surface_vocab_size,
        "surface_embed_dim": surface_embed_dim,
        "d_model": d_model,
        "ffn_dim": ffn_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "kc_vocab_size": kc_vocab_size,
        "recon_pos_embed_dim": recon_pos_embed_dim,
        "recon_hidden_dim": recon_hidden_dim,
    }


def _cache_tokenizer(checkpoint_id: str, vocab: Dict[str, Dict[str, int]]) -> str:
    """Write a tokenizer.json alongside the checkpoint for worker processes."""
    os.makedirs(LOCAL_CACHE, exist_ok=True)
    path = os.path.join(LOCAL_CACHE, f"tokenizer-{checkpoint_id}.json")
    if not os.path.exists(path):
        data = {"field_vocabs": vocab, "frozen": True}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    return path


def _build_config(
    config_dict: Optional[Dict[str, Any]],
    cleaned_state: Dict[str, torch.Tensor],
) -> BpdModelConfig:
    """Build BpdModelConfig from checkpoint metadata or inferred shapes."""
    if config_dict:
        import dataclasses

        valid_fields = {f.name for f in dataclasses.fields(BpdModelConfig)}
        return BpdModelConfig(
            **{k: v for k, v in config_dict.items() if k in valid_fields}
        )
    return BpdModelConfig(
        **infer_config_from_state(cleaned_state),
        dropout=0.0,
        layer_drop_prob=0.0,
    )


def load_model_from_checkpoint(  # pylint: disable=too-many-locals
    lock: Optional[Dict[str, Any]] = None,
    *,
    distilled: bool = True,
    drop_layers: int = 0,
    output_rank: int = 0,
) -> Tuple[BpdModel, str, str]:
    """Build and load a BpdModel from checkpoint.lock + dataset.lock.

    When *distilled* is True (default), loads or creates the FP16 distilled
    variant for faster MPS inference.  *drop_layers* permanently removes
    encoder layers; *output_rank* applies low-rank SVD to the output head.
    The returned model has ``_distilled`` and (optionally) ``_output_u`` /
    ``_output_v`` attributes for the inference path.

    Returns ``(model, tokenizer_path, checkpoint_id)``.
    """
    if lock is None:
        lock = read_lock()
    if lock is None:
        raise FileNotFoundError(
            "checkpoint.lock not found. Run: scripts/cc checkpoint pull recon_bpd"
        )

    checkpoint_id: str = lock["checkpoint_id"]
    model_type: str = lock["model_type"]
    dataset_id: str = lock["dataset_id"]

    if distilled:
        from scripts.recon_bpd.distill import ensure_distilled

        local_pt = ensure_distilled(
            checkpoint_id,
            model_type,
            drop_layers=drop_layers,
            output_rank=output_rank,
        )
    else:
        local_pt = download_checkpoint(model_type, checkpoint_id)

    ckpt = torch.load(local_pt, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    # Extract low-rank factors before building the model
    output_u = cleaned.pop("recon.output_u", None)
    output_v = cleaned.pop("recon.output_v", None)

    cfg = _build_config(ckpt.get("config_dict"), cleaned)
    model = BpdModel(cfg)
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    model._distilled = distilled  # type: ignore[attr-defined,assignment]  # pylint: disable=protected-access

    if output_u is not None and output_v is not None:
        model.register_buffer("_output_u", output_u)  # [V, r]
        model.register_buffer("_output_v", output_v)  # [r, H]

    from scripts.dataset import resolve_dataset_by_id

    bundle = resolve_dataset_by_id(dataset_id)
    tokenizer_path = _cache_tokenizer(checkpoint_id, bundle["vocab"])

    return model, tokenizer_path, checkpoint_id


LN2 = 0.6931471805599453  # ln(2), for nat -> bit conversion


def embed_and_bpd(  # pylint: disable=too-many-locals
    model: BpdModel,
    surface_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    vectorized: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single forward pass returning pooled embeddings and per-sentence BPD.

    When *vectorized* is False (full FP32 model), the output projection is
    chunked over positions to avoid a [B, T, V] tensor (~17 GB at batch 512).
    When True (FP16 distilled model), uses a single matmul over all positions
    at once -- feasible because FP16 halves the tensor size and distilled
    models are run at smaller batch sizes.

    Returns:
        (pooled, bpd) where pooled is [B, d_model] and bpd is [B].
    """
    pooled = model.encode(surface_ids, attention_mask)
    kc_raw, _ = model.kc_head.forward_with_raw(pooled)
    kc_probs = torch.sigmoid(kc_raw)
    h_recon = model.recon.forward_hidden(kc_probs, attention_mask)  # [B, T, H]

    bsz, seq_len = surface_ids.shape

    output_u: Optional[torch.Tensor] = getattr(model, "_output_u", None)
    output_v: Optional[torch.Tensor] = getattr(model, "_output_v", None)
    has_lowrank = output_u is not None and output_v is not None

    if has_lowrank and vectorized:
        assert output_u is not None and output_v is not None  # narrows type
        # Full [B,T,V] two-step: feasible at small batch (FP16)
        mid = F.linear(h_recon, output_v)  # pylint: disable=not-callable
        logits = F.linear(mid, output_u)  # pylint: disable=not-callable
        log_probs = F.log_softmax(logits.float(), dim=-1)
        targets = surface_ids.unsqueeze(-1)
        nll = -log_probs.gather(2, targets).squeeze(-1)
        nll_sum = (nll * attention_mask.float()).sum(dim=1)
    elif has_lowrank:
        assert output_u is not None and output_v is not None  # narrows type
        # Per-position two-step: peak memory [B, V] not [B, T, V]
        nll_sum = torch.zeros(bsz, device=surface_ids.device)
        for t in range(seq_len):
            mid_t = F.linear(h_recon[:, t, :], output_v)  # pylint: disable=not-callable
            logits_t = F.linear(mid_t, output_u)  # pylint: disable=not-callable
            log_probs_t = F.log_softmax(logits_t.float(), dim=-1)
            target_t = surface_ids[:, t]
            nll_t = -log_probs_t.gather(1, target_t.unsqueeze(1)).squeeze(1)
            nll_sum += nll_t * attention_mask[:, t].float()
    elif vectorized:
        output_weight = model.recon.output_head.weight  # [V, H]
        logits = F.linear(h_recon, output_weight)  # pylint: disable=not-callable
        log_probs = F.log_softmax(logits.float(), dim=-1)
        targets = surface_ids.unsqueeze(-1)
        nll = -log_probs.gather(2, targets).squeeze(-1)
        nll_sum = (nll * attention_mask.float()).sum(dim=1)
    else:
        output_weight = model.recon.output_head.weight  # [V, H]
        nll_sum = torch.zeros(bsz, device=surface_ids.device)
        for t in range(seq_len):
            logits_t = F.linear(h_recon[:, t, :], output_weight)  # pylint: disable=not-callable
            log_probs_t = F.log_softmax(logits_t, dim=-1)
            target_t = surface_ids[:, t]
            nll_t = -log_probs_t.gather(1, target_t.unsqueeze(1)).squeeze(1)
            nll_sum += nll_t * attention_mask[:, t].float()

    lengths = attention_mask.float().sum(dim=1).clamp(min=1)
    return pooled, nll_sum / (lengths * LN2)
