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

    from scripts.recon_bpd import count_encoder_layers

    num_layers = count_encoder_layers(state)

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
    )


def load_model_from_checkpoint(  # pylint: disable=too-many-locals,redefined-builtin
    lock: Optional[Dict[str, Any]] = None,
    *,
    distilled: bool = True,
    layer_mask: str = "",
    output_rank: int = 0,
    token_percentile: float = 100.0,
    compile: bool = True,
) -> Tuple[BpdModel, str, str]:
    """Build and load a BpdModel from checkpoint.lock + dataset.lock.

    When *distilled* is True (default), loads or creates the FP16 distilled
    variant for faster MPS inference.  *layer_mask* is a binary string
    (``'1'`` = keep, ``'0'`` = drop) controlling which encoder layers to
    retain; *output_rank* applies low-rank SVD to the output head.
    The returned model has ``_distilled`` and (optionally) ``_output_u`` /
    ``_output_v`` attributes for the inference path.

    When *token_percentile* < 100, the distilled checkpoint uses a reduced
    surface vocabulary.  The model receives a ``_token_remap`` buffer mapping
    full-vocab IDs to reduced IDs so ``embed_and_bpd`` can transparently
    remap tokenizer output.

    When *compile* is True (default), the model is wrapped with
    ``torch.compile`` for fused-kernel acceleration.  Set to False
    for short-lived models where compilation overhead dominates.

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
            layer_mask=layer_mask,
            output_rank=output_rank,
            token_percentile=token_percentile,
        )
    else:
        local_pt = download_checkpoint(model_type, checkpoint_id)

    ckpt = torch.load(local_pt, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    # FP32 layer dropping: apply mask to state dict at load time
    has_drops = bool(layer_mask) and "0" in layer_mask
    if not distilled and has_drops:
        from scripts.recon_bpd.distill import _drop_encoder_layers

        cleaned = _drop_encoder_layers(cleaned, layer_mask)

    # Extract low-rank factors before building the model
    output_u = cleaned.pop("recon.output_u", None)
    output_v = cleaned.pop("recon.output_v", None)

    cfg = _build_config(ckpt.get("config_dict"), cleaned)
    # Override num_layers when FP32 layer dropping changed the state dict
    if not distilled and has_drops:
        cfg.num_layers = layer_mask.count("1")
    # Models with dropped layers must not apply stochastic depth
    # scaling — those layers are permanently gone, not randomly dropped.
    if has_drops or (
        ckpt.get("distilled") and ckpt.get("layer_mask", "").count("0") > 0
    ):
        cfg.layer_drop_prob = 0.0
    model = BpdModel(cfg)
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    model._distilled = distilled  # type: ignore[attr-defined,assignment]  # pylint: disable=protected-access

    if output_u is not None and output_v is not None:
        model.register_buffer("_output_u", output_u)  # [V, r]
        model.register_buffer("_output_v", output_v)  # [r, H]

    # Attach token remap for reduced-vocab models so embed_and_bpd can
    # transparently convert full-vocab tokenizer IDs to reduced IDs.
    # Two formats: raw old_to_new tensor (from training) or dict with
    # kept_indices/unk_id (from distillation).
    token_remap_raw = ckpt.get("token_remap")
    if isinstance(token_remap_raw, torch.Tensor):
        model.register_buffer("_token_remap", token_remap_raw)
    elif isinstance(token_remap_raw, dict):
        from scripts.recon_bpd.token_remap import NUM_SPECIAL

        kept = token_remap_raw["kept_indices"]
        unk_id = int(token_remap_raw["unk_id"])
        v_old = int(kept.max().item()) + 1 if len(kept) > 0 else NUM_SPECIAL
        v_old = max(v_old, NUM_SPECIAL) + 1
        old_to_new = torch.full((v_old,), unk_id, dtype=torch.long)
        for new_id, old_id in enumerate(kept.tolist()):
            if old_id < v_old:
                old_to_new[old_id] = new_id
        model.register_buffer("_token_remap", old_to_new)

    # KC inference parameters: match training's clamp + temperature scaling
    # so the reconstruction decoder sees the same KC probability distribution.
    metrics = ckpt.get("latest_metrics") or {}
    model._kc_temperature = float(metrics.get("temperature", 1.2))  # type: ignore[assignment]  # pylint: disable=protected-access
    model._kc_clamp = 12.0  # type: ignore[assignment]  # pylint: disable=protected-access

    from scripts.dataset import resolve_dataset_by_id

    bundle = resolve_dataset_by_id(dataset_id)
    tokenizer_path = _cache_tokenizer(checkpoint_id, bundle["vocab"])

    # Legacy fallback: checkpoint predates token_remap persistence.
    # Reconstruct the mapping via byte-level embedding matching (slow).
    if getattr(model, "_token_remap", None) is None:
        _maybe_register_inferred_remap(model, bundle)

    if compile:
        model = torch.compile(model)  # type: ignore[assignment]

    return model, tokenizer_path, checkpoint_id


def _maybe_register_inferred_remap(  # pylint: disable=too-many-locals
    model: BpdModel,
    bundle: Dict[str, Any],
) -> None:
    """Infer and register a ``_token_remap`` buffer when the bundle vocabulary
    exceeds the model's ``surface_vocab_size``.

    Derives the mapping by matching each chiVe vector in the full vocabulary
    against the checkpoint's embedding rows (exact byte-level match).  This
    is robust to changes in ``token_gram_freq`` ordering that would otherwise
    produce a different old→new mapping than what was used during training.
    """
    from scripts.dataset import download_chive, load_chive

    v_new = model.cfg.surface_vocab_size
    chive_id = bundle.get("chive_id")
    if chive_id is None:
        return

    # Check if remapping is needed
    surface_vocab = bundle.get("vocab", {}).get("surface", {})
    v_old = len(surface_vocab)
    if v_old <= v_new:
        return

    chive_path = download_chive(chive_id)
    chive = load_chive(chive_path)
    v_old = chive.shape[0]
    if v_old <= v_new:
        return

    ckpt_embed = model.surface_embed.weight.detach()  # [v_new, 300]
    unk_id = v_new - 1

    embed_index: Dict[bytes, int] = {}
    for new_id in range(v_new):
        key = ckpt_embed[new_id].numpy().tobytes()
        if key not in embed_index:
            embed_index[key] = new_id

    old_to_new = torch.full((v_old,), unk_id, dtype=torch.long)
    for old_id in range(v_old):
        key = chive[old_id].numpy().tobytes()
        if key in embed_index:
            old_to_new[old_id] = embed_index[key]

    model.register_buffer("_token_remap", old_to_new)


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
    # Remap full-vocab tokenizer IDs to reduced-vocab IDs if applicable.
    token_remap: Optional[torch.Tensor] = getattr(model, "_token_remap", None)
    if token_remap is not None:
        clamped = surface_ids.clamp(max=token_remap.size(0) - 1)
        surface_ids = token_remap.to(surface_ids.device)[clamped]

    pooled = model.encode(surface_ids, attention_mask)
    kc_raw, _ = model.kc_head.forward_with_raw(pooled)

    # Match training's KC logit processing: clamp to [-C, C] then scale by
    # temperature before sigmoid.  Without this, extreme logits (mean ≈ -36)
    # produce near-zero KC probabilities that starve the reconstruction decoder.
    kc_clamp: float = getattr(model, "_kc_clamp", 12.0)
    kc_temp: float = getattr(model, "_kc_temperature", 1.0)
    kc_raw = kc_raw.clamp(-kc_clamp, kc_clamp)
    kc_probs = torch.sigmoid(kc_raw / kc_temp)

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
