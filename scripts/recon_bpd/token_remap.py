"""Token percentile vocabulary reduction for recon_bpd.

Computes a remap that keeps only the most-frequent surface tokens covering a
target percentile of gram token-position mass, collapsing the rest into a
single UNK whose chiVe embedding is the frequency-weighted mean of the
removed tokens.

Shared between training (scratch/recon_bpd.py) and distillation
(scripts/recon_bpd/distill.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from kotogram.tokenizer import MASK_ID, Tokenizer

NUM_SPECIAL = MASK_ID + 1  # IDs 0..3 are PAD, UNK, CLS, MASK


@dataclass
class TokenRemap:
    """Result of a percentile-based vocabulary reduction."""

    old_to_new: torch.Tensor  # [V_old] int64 mapping old ID -> new ID
    kept_indices: torch.Tensor  # [V_new - 1] int64, old IDs that were kept (excl. UNK)
    v_new: int  # new vocabulary size (kept + 1 UNK)
    unk_id: int  # new ID for the merged UNK token
    unk_chive_row: torch.Tensor  # [300] float32 embedding for UNK
    percentile: float  # the percentile that produced this remap


def compute_token_remap(
    token_gram_freq: torch.Tensor,
    percentile: float,
    chive_weights: torch.Tensor,
) -> TokenRemap:
    """Compute a vocabulary reduction from a token-frequency vector.

    Args:
        token_gram_freq: ``[V]`` int64 tensor of per-token gram position counts.
        percentile: Keep tokens covering this percentage of total position mass
            (e.g. 99.0 keeps tokens accounting for 99% of positions).
        chive_weights: ``[V, 300]`` float32 chiVe embeddings.

    Returns:
        A ``TokenRemap`` describing the old-to-new mapping and UNK embedding.
    """
    freq = token_gram_freq.cpu().numpy().astype(np.float64)
    v_old = len(freq)
    total = freq.sum()

    sorted_idx = np.argsort(-freq)
    cumsum = np.cumsum(freq[sorted_idx])
    threshold = percentile / 100.0 * total
    n_kept = int(np.searchsorted(cumsum, threshold, side="right")) + 1
    n_kept = min(n_kept, v_old)

    kept_set = set(sorted_idx[:n_kept].tolist())

    # Always keep special tokens
    for sid in range(NUM_SPECIAL):
        kept_set.add(sid)

    kept_sorted = sorted(kept_set)
    kept_indices = torch.tensor(kept_sorted, dtype=torch.long)

    # UNK occupies the last slot in the new vocab
    v_new = len(kept_sorted) + 1
    unk_id = v_new - 1

    old_to_new = torch.full((v_old,), unk_id, dtype=torch.long)
    for new_id, old_id in enumerate(kept_sorted):
        old_to_new[old_id] = new_id

    # Frequency-weighted mean of removed tokens' chiVe vectors
    removed_mask = torch.ones(v_old, dtype=torch.bool)
    removed_mask[kept_indices] = False
    removed_freq = token_gram_freq[removed_mask].float()
    w_sum = removed_freq.sum().clamp_min(1.0)
    weights = removed_freq / w_sum
    unk_chive_row = (chive_weights[removed_mask] * weights.unsqueeze(1)).sum(dim=0)

    return TokenRemap(
        old_to_new=old_to_new,
        kept_indices=kept_indices,
        v_new=v_new,
        unk_id=unk_id,
        unk_chive_row=unk_chive_row,
        percentile=percentile,
    )


def apply_remap_to_bundle(
    bundle: Dict[str, Any],
    chive_weights: torch.Tensor,
    percentile: float,
) -> Tuple[Dict[str, Any], torch.Tensor, TokenRemap]:
    """Apply token percentile reduction to a dataset bundle in-place.

    Remaps ``bundle["features"]["surface"]`` IDs to the reduced vocabulary,
    rebuilds ``bundle["vocab"]["surface"]``, slices ``bundle["content_mask"]``,
    and constructs a new chiVe matrix for the reduced vocab.

    Args:
        bundle: Dataset bundle dict (modified in-place).
        chive_weights: ``[V_old, 300]`` chiVe embeddings.
        percentile: Token percentile to keep (e.g. 99.0).

    Returns:
        ``(bundle, new_chive, remap)`` tuple.
    """
    tgf = bundle.get("token_gram_freq")
    if tgf is None:
        raise KeyError(
            "bundle missing 'token_gram_freq'. "
            "Rebuild the dataset with the latest scripts/dataset.py."
        )

    remap = compute_token_remap(tgf, percentile, chive_weights)
    v_old = len(remap.old_to_new)

    # Remap surface feature IDs
    surface = bundle["features"]["surface"]
    bundle["features"]["surface"] = remap.old_to_new[surface.long()]

    # Rebuild surface vocab: kept tokens get new IDs, plus <UNK_REDUCED>
    old_vocab: Dict[str, int] = bundle["vocab"]["surface"]
    inv_old: Dict[int, str] = {v: k for k, v in old_vocab.items()}
    new_vocab: Dict[str, int] = {}
    for new_id, old_id in enumerate(remap.kept_indices.tolist()):
        token_str = inv_old.get(old_id, f"<id_{old_id}>")
        new_vocab[token_str] = new_id
    new_vocab["<UNK_REDUCED>"] = remap.unk_id
    bundle["vocab"]["surface"] = new_vocab

    # Slice content_mask to new vocab
    old_cm = bundle["content_mask"]
    new_cm = torch.zeros(remap.v_new, dtype=old_cm.dtype)
    new_cm[: len(remap.kept_indices)] = old_cm[remap.kept_indices]
    new_cm[remap.unk_id] = True  # treat merged UNK as content
    bundle["content_mask"] = new_cm

    # Build reduced chiVe: kept rows + UNK row
    new_chive = torch.zeros(remap.v_new, chive_weights.size(1), dtype=chive_weights.dtype)
    new_chive[: len(remap.kept_indices)] = chive_weights[remap.kept_indices]
    new_chive[remap.unk_id] = remap.unk_chive_row

    print(
        f"  Token remap (p={percentile}): "
        f"{v_old:,} -> {remap.v_new:,} surface tokens "
        f"({v_old - remap.v_new:,} removed, "
        f"{(v_old - remap.v_new) / v_old * 100:.1f}% reduction)"
    )

    return bundle, new_chive, remap


def apply_remap_to_state_dict(
    state: Dict[str, torch.Tensor],
    remap: TokenRemap,
    chive_weights: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Slice a full-vocab state dict down to a reduced vocabulary.

    Used during distillation to shrink an already-trained checkpoint.

    Args:
        state: Model state dict with full-vocab weights.
        remap: ``TokenRemap`` from ``compute_token_remap``.
        chive_weights: Optional ``[V_old, 300]`` chiVe for constructing the
            UNK embedding row.  If None, the UNK row is zero-initialized.

    Returns:
        New state dict with reduced-vocab weight matrices.
    """
    out = dict(state)

    embed_key = "surface_embed.weight"
    if embed_key in out:
        old_embed = out[embed_key]
        new_embed = torch.zeros(
            remap.v_new, old_embed.size(1), dtype=old_embed.dtype
        )
        new_embed[: len(remap.kept_indices)] = old_embed[remap.kept_indices]
        if chive_weights is not None:
            new_embed[remap.unk_id] = remap.unk_chive_row.to(old_embed.dtype)
        out[embed_key] = new_embed

    output_key = "recon.output_head.weight"
    if output_key in out:
        old_out = out[output_key]  # [V_old, H]
        new_out = torch.zeros(
            remap.v_new, old_out.size(1), dtype=old_out.dtype
        )
        new_out[: len(remap.kept_indices)] = old_out[remap.kept_indices]
        out[output_key] = new_out

    # Also handle low-rank SVD factors if present
    u_key = "recon.output_u"
    if u_key in out:
        old_u = out[u_key]  # [V_old, r]
        new_u = torch.zeros(remap.v_new, old_u.size(1), dtype=old_u.dtype)
        new_u[: len(remap.kept_indices)] = old_u[remap.kept_indices]
        out[u_key] = new_u

    # semantic_head projects to 300D (not V-dependent), no slicing needed

    return out
