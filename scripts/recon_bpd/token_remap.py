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

import torch

from kotogram.tokenizer import MASK_ID

NUM_SPECIAL = MASK_ID + 1  # IDs 0..3 are PAD, UNK, CLS, MASK

# Whole-token pristine mappings: only applied when the entire token matches.
_PRISTINE_EXACT: Dict[str, str] = {
    "...": "\u2026",  # three dots -> ellipsis
}

# Single-char pristine mappings: only applied to single-character tokens.
# NOTE: '.' and ASCII '"' are handled context-dependently in apply_pristine().
_PRISTINE_SINGLE: Dict[str, str] = {
    "!": "\uff01",
    "?": "\uff1f",
    ",": "\u3001",
    ":": "\uff1a",
    "~": "\uff5e",
    "\uff0e": "\u3002",  # fullwidth stop -> ideographic stop
    "\uff61": "\u3002",
    "\uff64": "\u3001",  # halfwidth variants
    "\uff62": "\u300c",
    "\uff63": "\u300d",  # halfwidth brackets
}


def pristine_surface(tok: str) -> str:
    """Map a dirty surface token to its pristine form, or return unchanged.

    NOTE: Does NOT handle '.' or ASCII '"' — those need sequence context.
    Use ``apply_pristine()`` for the full context-aware mapping.
    """
    if tok in _PRISTINE_EXACT:
        return _PRISTINE_EXACT[tok]
    if len(tok) == 1 and tok in _PRISTINE_SINGLE:
        return _PRISTINE_SINGLE[tok]
    return tok


def build_pristine_id_mapping(vocab: Dict[str, int]) -> torch.Tensor:
    """Build a [vocab_size] static ID mapping for non-context-dependent rules.

    Context-dependent rules ('.' handling) are applied by ``apply_pristine()``.
    Non-content tokens are left as identity; the dataloader's
    ``content_drop_ratio`` physically removes them from both input and target.
    """
    inv_vocab = {v: k for k, v in vocab.items()}
    v = max(vocab.values()) + 1
    mapping = torch.arange(v, dtype=torch.long)

    for tid in range(NUM_SPECIAL, v):
        tok = inv_vocab.get(tid)
        if tok is None:
            continue
        p = pristine_surface(tok)
        if p != tok:
            pid = vocab.get(p)
            if pid is not None:
                mapping[tid] = pid
                continue

    return mapping


def apply_pristine(
    ids: torch.Tensor,
    vocab: Dict[str, int],
    static_mapping: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply full pristine mapping to a 1-D token ID sequence.

    Handles context-dependent rules:
    - '.' as last token → '。'
    - ASCII '"' by occurrence → alternating 「 and 」 (1st, 3rd, … open; 2nd, 4th, … close)
    - All other rules via static mapping

    Never introduces PAD tokens; non-content tokens are left as identity
    and handled by the dataloader's ``content_drop_ratio``.

    Pass a pre-built ``static_mapping`` to avoid recomputing it per call.
    """
    if static_mapping is None:
        static_mapping = build_pristine_id_mapping(vocab)
    out = static_mapping[ids].clone()

    # Sentence-final '.' → '。'
    dot_id = vocab.get(".")
    maru_id = vocab.get("\u3002")  # 。
    n = len(ids)
    if dot_id is not None and maru_id is not None:
        if n > 0 and int(ids[n - 1]) == dot_id:
            out[n - 1] = maru_id

    # ASCII '"' → alternating 「 / 」 by occurrence parity
    quote_id = vocab.get('"')
    open_k = vocab.get("\u300c")
    close_k = vocab.get("\u300d")
    if quote_id is not None and open_k is not None and close_k is not None:
        dq_count = 0
        for i in range(n):
            if int(ids[i]) == quote_id:
                dq_count += 1
                out[i] = open_k if dq_count % 2 == 1 else close_k

    return out


def _pristine_token_ids(vocab: Dict[str, int]) -> set:
    """Return token IDs that participate in pristine mappings (both sides)."""
    ids: set = set()
    for tok, tid in vocab.items():
        if tid < NUM_SPECIAL:
            continue
        p = pristine_surface(tok)
        if p != tok:
            ids.add(tid)
            pid = vocab.get(p)
            if pid is not None:
                ids.add(pid)
    # Context-dependent: '.' runs, '" parity'
    for tok in (".", "\u3002", "\u2026", '"', "\u300c", "\u300d"):
        ctx_tid = vocab.get(tok)
        if ctx_tid is not None:
            ids.add(ctx_tid)
    return ids


def apply_pristine_batch(
    ids: torch.Tensor,
    attention_mask: torch.Tensor,
    vocab: Dict[str, int],
    static_mapping: torch.Tensor,
) -> torch.Tensor:
    """Apply pristine mapping to a batch of token ID sequences.

    Args:
        ids: [B, T] int64 tensor of dirty surface token IDs.
        attention_mask: [B, T] float tensor (1.0 for real tokens, 0.0 for padding).
        vocab: Surface vocabulary dict (token string -> ID).
        static_mapping: Pre-built mapping from ``build_pristine_id_mapping``.

    Returns:
        [B, T] int64 tensor of pristine surface token IDs.
    """
    bsz = ids.size(0)
    out = ids.clone()
    for b in range(bsz):
        seq_len = int(attention_mask[b].sum().item())
        row = ids[b, :seq_len]
        out[b, :seq_len] = apply_pristine(row, vocab, static_mapping=static_mapping)
    return out


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
    chive_percentile: float,
    chive_weights: torch.Tensor,
    chive_ranks: torch.Tensor,
    force_keep: Optional[set] = None,
) -> TokenRemap:
    """Compute a vocabulary reduction based on chiVe corpus frequency ranks.

    Args:
        token_gram_freq: ``[V]`` int64 tensor of per-token gram position counts.
        chive_percentile: Keep tokens in the top X% of chiVe's global vocabulary.
            (e.g., 50.0 keeps rank <= 1,265,396).
        chive_weights: ``[V, 300]`` float32 chiVe embeddings.
        chive_ranks: ``[V]`` float or int tensor of chiVe frequency ranks.
        force_keep: Optional set of token IDs that must survive the reduction
            regardless of chiVe rank (e.g. pristine target tokens).

    Returns:
        A ``TokenRemap`` describing the old-to-new mapping and UNK embedding.
    """
    v_old = len(token_gram_freq)

    # chiVe 1.3 mc5 has 2,530,792 total tokens.
    CHIVE_TOTAL_VOCAB = 2530792
    limit = int(CHIVE_TOTAL_VOCAB * (chive_percentile / 100.0))

    # Keep tokens that are within the chiVe rank limit AND appear in our corpus
    kept_mask = (chive_ranks <= limit) & (token_gram_freq > 0)
    kept_set = set(torch.nonzero(kept_mask).squeeze(-1).tolist())

    # Always keep special tokens
    for sid in range(NUM_SPECIAL):
        kept_set.add(sid)

    # Force-keep pristine target tokens (and their dirty sources)
    if force_keep:
        kept_set.update(fid for fid in force_keep if fid < v_old)

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

    # Convert the chive parameter naming back to generic "percentile" for the dataclass
    return TokenRemap(
        old_to_new=old_to_new,
        kept_indices=kept_indices,
        v_new=v_new,
        unk_id=unk_id,
        unk_chive_row=unk_chive_row,
        percentile=chive_percentile,
    )


def apply_remap_to_bundle(  # pylint: disable=too-many-locals
    bundle: Dict[str, Any],
    chive_weights: torch.Tensor,
    chive_percentile: float,
) -> Tuple[Dict[str, Any], torch.Tensor, TokenRemap]:
    """Apply token percentile reduction to a dataset bundle in-place.

    Remaps ``bundle["features"]["surface"]`` IDs to the reduced vocabulary,
    rebuilds ``bundle["vocab"]["surface"]``, slices ``bundle["content_mask"]``,
    and constructs a new chiVe matrix for the reduced vocab.

    Args:
        bundle: Dataset bundle dict (read-only, a shallow copy is returned).
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

    chive_ranks = bundle.get("chive_ranks")
    if chive_ranks is None:
        raise KeyError("bundle missing 'chive_ranks' required for chiVe-rank remap.")

    pristine_ids = _pristine_token_ids(bundle["vocab"]["surface"])
    remap = compute_token_remap(
        token_gram_freq=tgf,
        chive_percentile=chive_percentile,
        chive_weights=chive_weights,
        chive_ranks=chive_ranks,
        force_keep=pristine_ids,
    )
    v_old = len(remap.old_to_new)

    # Create a shallow copy to avoid destroying global shared state in Optuna
    new_bundle = dict(bundle)
    new_bundle["features"] = dict(bundle["features"])
    new_bundle["vocab"] = dict(bundle["vocab"])

    # Remap surface feature IDs
    surface = new_bundle["features"]["surface"]
    new_bundle["features"]["surface"] = remap.old_to_new[surface.long()]

    # Rebuild surface vocab: kept tokens get new IDs, plus <UNK_REDUCED>
    old_vocab: Dict[str, int] = new_bundle["vocab"]["surface"]
    inv_old: Dict[int, str] = {v: k for k, v in old_vocab.items()}
    new_vocab: Dict[str, int] = {}
    for new_id, old_id in enumerate(remap.kept_indices.tolist()):
        token_str = inv_old.get(old_id, f"<id_{old_id}>")
        new_vocab[token_str] = new_id
    new_vocab["<UNK_REDUCED>"] = remap.unk_id
    new_bundle["vocab"]["surface"] = new_vocab

    # Slice content_mask to new vocab
    old_cm = new_bundle["content_mask"]
    new_cm = torch.zeros(remap.v_new, dtype=old_cm.dtype)
    new_cm[: len(remap.kept_indices)] = old_cm[remap.kept_indices]
    new_cm[remap.unk_id] = True  # treat merged UNK as content
    new_bundle["content_mask"] = new_cm

    # Build reduced chiVe: kept rows + UNK row
    new_chive = torch.zeros(
        remap.v_new, chive_weights.size(1), dtype=chive_weights.dtype
    )
    new_chive[: len(remap.kept_indices)] = chive_weights[remap.kept_indices]
    new_chive[remap.unk_id] = remap.unk_chive_row

    print(
        f"  Token remap (chive_{chive_percentile}p): "
        f"{v_old:,} -> {remap.v_new:,} surface tokens "
        f"({v_old - remap.v_new:,} removed, "
        f"{(v_old - remap.v_new) / v_old * 100:.1f}% reduction)"
    )

    return new_bundle, new_chive, remap


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
        new_embed = torch.zeros(remap.v_new, old_embed.size(1), dtype=old_embed.dtype)
        new_embed[: len(remap.kept_indices)] = old_embed[remap.kept_indices]
        if chive_weights is not None:
            new_embed[remap.unk_id] = remap.unk_chive_row.to(old_embed.dtype)
        out[embed_key] = new_embed

    output_key = "recon.output_head.weight"
    if output_key in out:
        old_out = out[output_key]  # [V_old, H]
        new_out = torch.zeros(remap.v_new, old_out.size(1), dtype=old_out.dtype)
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
