"""Distill a recon_bpd checkpoint to FP16 for fast local MPS inference.

Converts all weights to float16, halving memory and enabling vectorized
output projection (single [B, T, V] matmul instead of per-position loop).
Optionally drops encoder layers via a binary mask and/or applies low-rank
SVD to the output head for further speedup.

Usage:
    python -m scripts.recon_bpd.distill                # distill checkpoint.lock
    python -m scripts.recon_bpd.distill --benchmark    # arena benchmark
    python -m scripts.recon_bpd.distill --force         # re-distill even if cached

The ``--benchmark`` flag runs an arena-style tournament: all layer-mask
variants compete at 100 sentences, the bottom 50% are eliminated each round
(adding 100 sentences per round), and the survivors are compared in a full
table.  Layer masks are binary strings (e.g. ``'110100111'``) where
``1`` = keep and ``0`` = drop.  The first layer is always kept, so every
mask produced by the arena starts with ``'1'``.
"""

# pylint: disable=too-many-lines
from __future__ import annotations

import argparse
import itertools
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from scripts.checkpoint import LOCAL_CACHE, download_checkpoint, read_lock

if TYPE_CHECKING:
    from rich.table import Table

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _distilled_path(
    checkpoint_id: str,
    layer_mask: str = "",
    output_rank: int = 0,
    token_percentile: float = 100.0,
) -> str:
    parts = [f"ckpt-{checkpoint_id}.distilled"]
    if layer_mask and "0" in layer_mask:
        parts.append(f"-mask{layer_mask}")
    if output_rank > 0:
        parts.append(f"-rank{output_rank}")
    if token_percentile < 100.0:
        parts.append(f"-tp{token_percentile:g}")
    parts.append(".pt")
    return os.path.join(LOCAL_CACHE, "".join(parts))


def _invalidate_stale_distilled(checkpoint_id: str) -> int:
    """Remove cached distilled checkpoints that lack ``token_remap``.

    Checks one sample file — if it's missing the remap, all cached distilled
    files for this checkpoint are stale and get deleted.  Returns the number
    of files removed.
    """
    import glob as glob_mod

    pattern = os.path.join(LOCAL_CACHE, f"ckpt-{checkpoint_id}.distilled*.pt")
    files = glob_mod.glob(pattern)
    if not files:
        return 0
    sample = files[0]
    ckpt = torch.load(sample, map_location="cpu", weights_only=False)
    if "token_remap" in ckpt:
        return 0
    for f in files:
        os.remove(f)
    return len(files)


# ---------------------------------------------------------------------------
# Benchmark score cache
# ---------------------------------------------------------------------------

_BENCH_CACHE_FIELDS = (
    "sps",
    "cos",
    "bpd_diff",
    "composite",
    "mean_bpd",
    "elapsed",
    "mem_mb",
    "num_layers",
)
_BENCH_CACHE_VERSION = 11  # bump when _make_test_batch or BPD semantics change


def _bench_cache_path(checkpoint_id: str) -> str:
    return os.path.join(LOCAL_CACHE, f"bench-{checkpoint_id}.json")


def _load_bench_cache(checkpoint_id: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load ``{variant_name: {n_sentences_str: {scores...}}}``.

    Returns an empty dict if the cache file is missing or was written by
    an older ``_BENCH_CACHE_VERSION`` (test-data semantics changed).
    """
    import json

    path = _bench_cache_path(checkpoint_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("_version") != _BENCH_CACHE_VERSION:
            print("    Bench cache version mismatch — discarding stale cache")
            return {}
        entries: Dict[str, Dict[str, Dict[str, Any]]] = data.get("entries", {})
        return entries
    return {}


def _save_bench_cache(
    checkpoint_id: str,
    cache: Dict[str, Dict[str, Dict[str, Any]]],
) -> None:
    """Atomically write the score cache to disk."""
    import json

    os.makedirs(LOCAL_CACHE, exist_ok=True)
    path = _bench_cache_path(checkpoint_id)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"_version": _BENCH_CACHE_VERSION, "entries": cache}, f)
    os.replace(tmp, path)


def _cache_result(
    cache: Dict[str, Dict[str, Dict[str, Any]]],
    name: str,
    n_sentences: int,
    result: Dict[str, Any],
) -> None:
    """Insert one result into the in-memory cache dict."""
    if name not in cache:
        cache[name] = {}
    cache[name][str(n_sentences)] = {
        k: result[k] for k in _BENCH_CACHE_FIELDS if k in result
    }


# ---------------------------------------------------------------------------
# Layer-mask helpers
# ---------------------------------------------------------------------------


def _mask_to_kept(mask: str) -> List[int]:
    """Convert a binary mask string to a sorted list of kept layer indices.

    ``'110100'`` → ``[0, 1, 3]``
    """
    return [i for i, c in enumerate(mask) if c == "1"]


def _all_masks(total_layers: int) -> List[str]:
    """Every valid layer mask (first layer kept, at least one layer dropped)."""
    masks: List[str] = []
    for bits in itertools.product("01", repeat=total_layers - 1):
        mask = "1" + "".join(bits)
        if "0" in mask:
            masks.append(mask)
    return masks


# ---------------------------------------------------------------------------
# Layer-drop (mask-based)
# ---------------------------------------------------------------------------


def _drop_encoder_layers(
    state: Dict[str, torch.Tensor],
    layer_mask: str,
) -> Dict[str, torch.Tensor]:
    """Remove encoder layers per *layer_mask* and renumber survivors to 0..K-1."""
    kept = _mask_to_kept(layer_mask)
    old_to_new = {old: new for new, old in enumerate(kept)}

    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if "encoder.layers." not in k:
            out[k] = v
            continue
        parts = k.split(".")
        idx = parts.index("layers") + 1
        if idx < len(parts) and parts[idx].isdigit():
            old_idx = int(parts[idx])
            if old_idx in old_to_new:
                parts[idx] = str(old_to_new[old_idx])
                out[".".join(parts)] = v
    return out


# ---------------------------------------------------------------------------
# Low-rank SVD for output head
# ---------------------------------------------------------------------------


def _apply_low_rank(
    state: Dict[str, torch.Tensor],
    rank: int,
) -> Dict[str, torch.Tensor]:
    """Replace recon.output_head.weight [V, H] with two SVD factors.

    Stores ``recon.output_u`` [V, r] and ``recon.output_v`` [r, H].
    The original weight is removed so load_state_dict(..., strict=False)
    leaves output_head uninitialised (the factors are used instead).
    """
    key = "recon.output_head.weight"
    weight = state[key].float()  # SVD in FP32 for precision
    u_full, s, vh = torch.linalg.svd(weight, full_matrices=False)  # pylint: disable=not-callable
    u_r = u_full[:, :rank]  # [V, r]
    s_r = s[:rank]  # [r]
    vh_r = vh[:rank, :]  # [r, H]

    # Absorb singular values into vh so inference is two plain matmuls
    v_factor = torch.diag(s_r) @ vh_r  # [r, H]

    out = dict(state)
    del out[key]
    out["recon.output_u"] = u_r.half()  # [V, r]
    out["recon.output_v"] = v_factor.half()  # [r, H]
    return out


# ---------------------------------------------------------------------------
# Core distillation
# ---------------------------------------------------------------------------


def _derive_remap_from_fp32_state(  # pylint: disable=too-many-locals
    state: Dict[str, torch.Tensor],
    checkpoint_id: str,
) -> Optional[torch.Tensor]:
    """Derive a token_remap from the FP32 source state dict.

    Uses byte-level matching between the checkpoint's surface_embed and
    the full ChiVe vocabulary — same logic as the inference fallback but
    applied before FP16 conversion so the bytes still match.

    Returns None if no remapping is needed.
    """
    embed_key = next((k for k in state if k.endswith("surface_embed.weight")), None)
    if embed_key is None:
        return None

    ckpt_embed = state[embed_key].detach().float()
    v_new = ckpt_embed.shape[0]

    lock = read_lock()
    if lock is None:
        return None

    from scripts.dataset import resolve_dataset_by_id

    bundle = resolve_dataset_by_id(lock["dataset_id"])
    surface_vocab = bundle.get("vocab", {}).get("surface", {})
    v_old = len(surface_vocab)
    if v_old <= v_new:
        return None

    chive_id = bundle.get("chive_id")
    if chive_id is None:
        return None

    from scripts.dataset import download_chive, load_chive

    chive_path = download_chive(chive_id)
    chive = load_chive(chive_path)
    v_old = chive.shape[0]
    if v_old <= v_new:
        return None

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

    n_mapped = (old_to_new != unk_id).sum().item()
    print(f"  Derived token remap for {checkpoint_id}: {n_mapped}/{v_old} mapped")
    return old_to_new


def distill_checkpoint(  # pylint: disable=too-many-locals
    checkpoint_id: str,
    model_type: str = "recon_bpd",
    *,
    layer_mask: str = "",
    output_rank: int = 0,
    token_percentile: float = 100.0,
    force: bool = False,
) -> str:
    """Convert a full checkpoint to FP16, optionally with layer-mask and low-rank.

    *layer_mask* is a binary string (``'1'`` = keep, ``'0'`` = drop) whose
    length must equal the model's encoder layer count.  An empty string or
    all-ones mask keeps every layer.

    When *token_percentile* < 100, also reduces the surface vocabulary by
    keeping only the tokens that cover that percentage of gram token-position
    mass (requires dataset.lock to resolve the dataset bundle).

    Returns the path to the distilled checkpoint.
    """
    has_drops = bool(layer_mask) and "0" in layer_mask

    out_path = _distilled_path(checkpoint_id, layer_mask, output_rank, token_percentile)
    if os.path.exists(out_path) and not force:
        return out_path

    full_path = download_checkpoint(model_type, checkpoint_id)
    extras = []
    if has_drops:
        extras.append(f"mask {layer_mask}")
    if output_rank:
        extras.append(f"rank-{output_rank} output head")
    if token_percentile < 100.0:
        extras.append(f"token-percentile {token_percentile:g}")
    label = f" ({', '.join(extras)})" if extras else ""
    print(f"  Distilling {checkpoint_id} to FP16{label}...")

    ckpt = torch.load(full_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]

    # Recover the source checkpoint's token_remap BEFORE converting to FP16,
    # since byte-level matching against ChiVe requires FP32 weights.
    source_token_remap: Optional[torch.Tensor] = None
    src_remap = ckpt.get("token_remap")
    if isinstance(src_remap, torch.Tensor):
        source_token_remap = src_remap
    elif src_remap is None:
        source_token_remap = _derive_remap_from_fp32_state(state, checkpoint_id)

    cleaned = {k.replace("_orig_mod.", ""): v.half() for k, v in state.items()}

    if has_drops:
        cleaned = _drop_encoder_layers(cleaned, layer_mask)
    if output_rank > 0:
        cleaned = _apply_low_rank(cleaned, output_rank)

    # Token percentile vocab reduction (additional reduction on top of source)
    token_remap_meta: Optional[Dict[str, Any]] = None
    if token_percentile < 100.0:
        from scripts.recon_bpd.token_remap import (
            apply_remap_to_state_dict,
            compute_token_remap,
        )

        lock = read_lock()
        if lock is None:
            raise FileNotFoundError(
                "checkpoint.lock needed for token remap (dataset resolution)"
            )
        from scripts.dataset import resolve_dataset

        bundle, chive_weights = resolve_dataset(None)
        tgf = bundle.get("token_gram_freq")
        if tgf is None:
            raise KeyError(
                "Dataset bundle missing 'token_gram_freq'. "
                "Rebuild with latest scripts/dataset.py."
            )
        remap = compute_token_remap(tgf, token_percentile, chive_weights)
        cleaned = apply_remap_to_state_dict(cleaned, remap, chive_weights)
        v_old = len(remap.old_to_new)
        print(
            f"  Token remap: {v_old:,} -> {remap.v_new:,} "
            f"({v_old - remap.v_new:,} removed)"
        )
        token_remap_meta = {
            "percentile": token_percentile,
            "v_new": remap.v_new,
            "unk_id": remap.unk_id,
            "kept_indices": remap.kept_indices,
        }

    distilled_ckpt: Dict[str, Any] = {
        "model_state": cleaned,
        "distilled": True,
        "layer_mask": layer_mask,
        "output_rank": output_rank,
        "source_checkpoint_id": checkpoint_id,
    }
    if token_remap_meta is not None:
        distilled_ckpt["token_remap"] = token_remap_meta
    elif source_token_remap is not None:
        distilled_ckpt["token_remap"] = source_token_remap
    if "config_dict" in ckpt:
        cfg = dict(ckpt["config_dict"])
        if has_drops:
            cfg["num_layers"] = layer_mask.count("1")
        if token_remap_meta is not None:
            cfg["surface_vocab_size"] = token_remap_meta["v_new"]
        distilled_ckpt["config_dict"] = cfg

    os.makedirs(LOCAL_CACHE, exist_ok=True)
    torch.save(distilled_ckpt, out_path)

    full_mb = os.path.getsize(full_path) / (1024 * 1024)
    dist_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  {full_mb:.0f} MB -> {dist_mb:.0f} MB ({dist_mb / full_mb:.0%})")
    return out_path


def ensure_distilled(
    checkpoint_id: str,
    model_type: str = "recon_bpd",
    *,
    layer_mask: str = "",
    output_rank: int = 0,
    token_percentile: float = 100.0,
) -> str:
    """Return path to distilled checkpoint, creating it if needed."""
    cached = _distilled_path(checkpoint_id, layer_mask, output_rank, token_percentile)
    if os.path.exists(cached):
        return cached
    opts = {
        "layer_mask": layer_mask,
        "output_rank": output_rank,
        "token_percentile": token_percentile,
    }
    return distill_checkpoint(checkpoint_id, model_type, **opts)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Bulk pre-distillation
# ---------------------------------------------------------------------------


def _pre_distill_configs(  # pylint: disable=too-many-locals
    checkpoint_id: str,
    model_type: str,
    configs: list[tuple[str, bool, str, int]],
) -> None:
    """Pre-distill all uncached variants in parallel.

    Loads the source checkpoint once and precomputes a single SVD of the
    output head, then fans out per-variant work (layer masking + save)
    across a thread pool.  PyTorch tensor ops release the GIL so threads
    give real parallelism for the heavy lifting.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_invalidated = _invalidate_stale_distilled(checkpoint_id)
    if n_invalidated:
        print(
            f"    Purged {n_invalidated} stale distilled checkpoints (missing token_remap)"
        )

    needed: list[tuple[str, int]] = []
    n_cached = 0
    for _name, distilled, mask, rank in configs:
        if not distilled:
            continue
        if os.path.exists(_distilled_path(checkpoint_id, mask, rank)):
            n_cached += 1
        else:
            needed.append((mask, rank))

    if not needed:
        if n_cached:
            print(f"    All {n_cached} distilled variants cached")
        return

    full_path = download_checkpoint(model_type, checkpoint_id)
    ckpt = torch.load(full_path, map_location="cpu", weights_only=False)
    source_state = ckpt["model_state"]
    config_dict = ckpt.get("config_dict")

    # Recover source token_remap (from FP32 weights, before FP16 conversion).
    source_token_remap: Optional[torch.Tensor] = None
    src_remap = ckpt.get("token_remap")
    if isinstance(src_remap, torch.Tensor):
        source_token_remap = src_remap
    elif src_remap is None:
        source_token_remap = _derive_remap_from_fp32_state(source_state, checkpoint_id)

    cleaned = {k.replace("_orig_mod.", ""): v.half() for k, v in source_state.items()}

    # One SVD, reused for every rank variant
    oh_key = "recon.output_head.weight"
    unique_ranks = sorted(set(r for _, r in needed if r > 0))
    rank_factors: Dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    if unique_ranks:
        weight = cleaned[oh_key].float()
        u_full, s, vh = torch.linalg.svd(weight, full_matrices=False)  # pylint: disable=not-callable
        for r in unique_ranks:
            u_r = u_full[:, :r].half()
            v_factor = (torch.diag(s[:r]) @ vh[:r, :]).half()
            rank_factors[r] = (u_r, v_factor)
        del u_full, s, vh, weight

    os.makedirs(LOCAL_CACHE, exist_ok=True)

    def _distill_one(mask: str, rank: int) -> None:
        out_path = _distilled_path(checkpoint_id, mask, rank)
        if os.path.exists(out_path):
            return
        has_drops = bool(mask) and "0" in mask
        state = _drop_encoder_layers(cleaned, mask) if has_drops else cleaned
        if rank > 0:
            state = dict(state)
            del state[oh_key]
            u_r, v_factor = rank_factors[rank]
            state["recon.output_u"] = u_r
            state["recon.output_v"] = v_factor

        payload: Dict[str, Any] = {
            "model_state": state,
            "distilled": True,
            "layer_mask": mask,
            "output_rank": rank,
            "source_checkpoint_id": checkpoint_id,
        }
        if source_token_remap is not None:
            payload["token_remap"] = source_token_remap
        if config_dict is not None:
            cfg = dict(config_dict)
            if has_drops:
                cfg["num_layers"] = mask.count("1")
            payload["config_dict"] = cfg
        torch.save(payload, out_path)

    detail_parts = []
    if unique_ranks:
        detail_parts.append(f"{len(unique_ranks)} unique ranks")
    if n_cached:
        detail_parts.append(f"{n_cached} cached")
    detail = f" ({', '.join(detail_parts)})" if detail_parts else ""
    print(f"    Distilling {len(needed)} variants{detail}...")

    workers = min(4, os.cpu_count() or 4)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_distill_one, m, r): (m, r) for m, r in needed}
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 50 == 0 or done == len(needed):
                print(f"    {done}/{len(needed)} distilled")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


_BENCH_BATCH = 64  # match production BPD_BATCH_SIZE


def _run_one_variant(  # pylint: disable=too-many-locals
    lock: Dict[str, Any],
    test_ids: torch.Tensor,
    test_mask: torch.Tensor,
    device: torch.device,
    n_sentences: int,
    *,
    distilled: bool,
    layer_mask: str = "",
    output_rank: int = 0,
) -> Dict[str, Any]:
    """Benchmark a single model variant matching production batching.

    Processes sentences in batches of 64 with vectorized=True for distilled
    models, mirroring the actual embed_and_score() path.
    """
    from scripts.recon_bpd.inference import embed_and_bpd, load_model_from_checkpoint

    model, _, _ = load_model_from_checkpoint(
        lock,
        distilled=distilled,
        layer_mask=layer_mask,
        output_rank=output_rank,
    )
    num_layers = model.cfg.num_layers
    use_vec = getattr(model, "_distilled", False)
    model.to(device)

    # Warmup: triggers torch.compile graph capture + first MPS kernel launch.
    # Not included in timing.
    with torch.inference_mode():
        embed_and_bpd(model, test_ids[:4], test_mask[:4], vectorized=use_vec)
        embed_and_bpd(model, test_ids[:4], test_mask[:4], vectorized=use_vec)
    if device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()

    # Timed batched run
    pooled_parts: list[torch.Tensor] = []
    bpd_parts: list[torch.Tensor] = []

    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, n_sentences, _BENCH_BATCH):
            end = min(start + _BENCH_BATCH, n_sentences)
            p, b = embed_and_bpd(
                model,
                test_ids[start:end],
                test_mask[start:end],
                vectorized=use_vec,
            )
            pooled_parts.append(p)
            bpd_parts.append(b)
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    pooled = torch.cat(pooled_parts, dim=0)
    bpd = torch.cat(bpd_parts, dim=0)

    mem_mb = 0.0
    if device.type == "mps":
        mem_mb = torch.mps.current_allocated_memory() / (1024 * 1024)

    del model
    import gc

    gc.collect()
    # torch.compile caches compiled graphs — reset to reclaim memory
    torch._dynamo.reset()  # pylint: disable=protected-access
    if device.type == "mps":
        torch.mps.empty_cache()

    return {
        "elapsed": elapsed,
        "sps": n_sentences / elapsed,
        "mem_mb": mem_mb,
        "mean_bpd": bpd.mean().item(),
        "pooled": pooled.float().cpu(),
        "bpd_vec": bpd.float().cpu(),
        "num_layers": num_layers,
    }


_ARENA_STEP = 500
_ARENA_INITIAL = 1000
_ARENA_FINAL_SIZE = 8
_ARENA_KEEP_FRAC = 0.25


def _arena_max_sentences(n_contestants: int) -> int:
    """Compute total sentences needed for the full arena."""
    if n_contestants <= _ARENA_FINAL_SIZE:
        return _ARENA_INITIAL
    pool = n_contestants
    n = _ARENA_INITIAL
    while True:
        keep = max(int(pool * _ARENA_KEEP_FRAC), 1)
        n += _ARENA_STEP
        if keep < _ARENA_FINAL_SIZE:
            break
        pool = keep
    return n


def _composite_score(
    ref_pooled: torch.Tensor, v: Dict[str, Any]
) -> tuple[float, float]:
    """Return (sps * cos^4, cosine_similarity)."""
    cos: float = (
        torch.nn.functional.cosine_similarity(  # pylint: disable=not-callable
            ref_pooled, v["pooled"], dim=-1
        )
        .mean()
        .item()
    )
    return v["sps"] * (cos**4), cos


def get_total_layers(lock: Dict[str, Any]) -> int:
    """Peek at checkpoint config to get encoder layer count without inference."""
    full_path = download_checkpoint(lock["model_type"], lock["checkpoint_id"])
    ckpt = torch.load(full_path, map_location="cpu", weights_only=False)
    if "config_dict" in ckpt:
        return int(ckpt["config_dict"].get("num_layers", 9))
    from scripts.recon_bpd import count_encoder_layers

    return count_encoder_layers(ckpt["model_state"])


def _build_all_configs(
    total_layers: int,
    rank_sweep: List[int],
) -> list[tuple[str, bool, str, int]]:
    """Generate (name, distilled, layer_mask, output_rank) for all variants.

    Includes baselines, every valid layer mask in both FP16 and FP32,
    and optionally rank variants.
    """
    masks = _all_masks(total_layers)

    configs: list[tuple[str, bool, str, int]] = [
        ("full fp32", False, "", 0),
        ("fp16", True, "", 0),
    ]
    for m in masks:
        configs.append((m, True, m, 0))

    for r in rank_sweep:
        configs.append((f"fp16 r{r}", True, "", r))

    for m in masks:
        for r in rank_sweep:
            configs.append((f"{m} r{r}", True, m, r))

    return configs


_BASELINE_NAMES = frozenset({"full fp32", "fp16"})


def _build_leaderboard(  # pylint: disable=too-many-locals
    entries: list[tuple[str, float, float, float, float]],
    progress_i: int,
    total: int,
    ref_sps: float,
) -> Table:
    """Build a Rich Table with all entries sorted by composite score."""
    from rich.table import Table

    table = Table(
        title=f"[bold]Leaderboard[/bold]  ({progress_i}/{total})",
        title_style="",
        show_edge=False,
        pad_edge=False,
    )
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Variant", min_width=14)
    table.add_column("s/s", justify="right", width=7)
    table.add_column("cos", justify="right", width=7)
    table.add_column("BPD", justify="right", width=8)
    table.add_column("Speed", justify="right", width=7)

    ranked = sorted(entries, key=lambda e: e[4], reverse=True)
    for rank_i, (name, sps, cos, bpd, _composite) in enumerate(ranked, 1):
        speedup = sps / max(ref_sps, 1e-9)
        is_baseline = name in _BASELINE_NAMES
        if rank_i == 1:
            style = "bold green"
        elif is_baseline:
            style = "dim"
        else:
            style = ""
        table.add_row(
            str(rank_i),
            name,
            f"{sps:.0f}",
            f"{cos:.4f}",
            f"{bpd:.4f}",
            f"{speedup:.2f}x",
            style=style,
        )

    return table


def _update_board(
    board: list[tuple[str, float, float, float, float]],
    board_limit: int,
    entry: tuple[str, float, float, float, float],
) -> None:
    """Insert *entry* into the leaderboard, evicting the worst non-baseline."""
    name = entry[0]
    is_baseline = name in _BASELINE_NAMES
    if is_baseline or len(board) < board_limit:
        board.append(entry)
    elif entry[4] > min(e[4] for e in board if e[0] not in _BASELINE_NAMES):
        worst_idx = min(
            (j for j, e in enumerate(board) if e[0] not in _BASELINE_NAMES),
            key=lambda j: board[j][4],
        )
        board[worst_idx] = entry


def _run_configs(  # pylint: disable=too-many-locals
    lock: Dict[str, Any],
    configs: list[tuple[str, bool, str, int]],
    test_ids: torch.Tensor,
    test_mask: torch.Tensor,
    *,
    device: torch.device,
    n_sentences: int,
) -> list[tuple[str, Dict[str, Any]]]:
    """Run all configs with a live-updating leaderboard.

    Baselines run first (deterministic), then contestants in random order
    so the leaderboard isn't biased by config-list ordering.

    Heavy tensors (pooled, bpd_vec) are scored eagerly against the
    reference and then discarded so memory stays bounded.  Each result
    dict gets ``cos``, ``bpd_diff``, and ``composite`` scalars instead.

    Results are cached to ``bench-<checkpoint_id>.json`` so a crashed run
    can resume without re-benchmarking already-scored variants.  The
    ``full fp32`` reference is always run fresh (we need its tensors to
    score new variants).
    """
    import random

    from rich.console import Console
    from rich.live import Live

    console = Console()
    checkpoint_id: str = lock["checkpoint_id"]
    cache = _load_bench_cache(checkpoint_id)
    n_key = str(n_sentences)

    variants: list[tuple[str, Dict[str, Any]]] = []
    ref_pooled: Optional[torch.Tensor] = None
    ref_bpd: Optional[torch.Tensor] = None
    ref_sps = 1.0
    board: list[tuple[str, float, float, float, float]] = []
    board_limit = _ARENA_FINAL_SIZE + len(_BASELINE_NAMES)

    baseline_configs = [c for c in configs if c[0] in _BASELINE_NAMES]
    contestant_configs = [c for c in configs if c[0] not in _BASELINE_NAMES]
    random.shuffle(contestant_configs)
    run_order = baseline_configs + contestant_configs
    total = len(run_order)
    n_cached = 0

    with Live(console=console, refresh_per_second=4, transient=True) as live:
        for i, (name, dist, mask, rank) in enumerate(run_order):
            # --- cache hit (never for full fp32 — need reference tensors) ---
            if name != "full fp32" and name in cache and n_key in cache[name]:
                result = dict(cache[name][n_key])
                variants.append((name, result))
                n_cached += 1
                entry = (
                    name,
                    result["sps"],
                    result["cos"],
                    result["mean_bpd"],
                    result["composite"],
                )
                _update_board(board, board_limit, entry)
                live.update(_build_leaderboard(board, i + 1, total, ref_sps))
                continue

            # --- cache miss: run the variant ---
            result = _run_one_variant(
                lock,
                test_ids,
                test_mask,
                device,
                n_sentences,
                distilled=dist,
                layer_mask=mask,
                output_rank=rank,
            )

            if name == "full fp32":
                ref_pooled = result["pooled"]
                ref_bpd = result["bpd_vec"]
                ref_sps = result["sps"]
                result["cos"] = 1.0
                result["bpd_diff"] = 0.0
                result["composite"] = result["sps"]
                del result["pooled"], result["bpd_vec"]
                variants.append((name, result))
                _cache_result(cache, name, n_sentences, result)
                _save_bench_cache(checkpoint_id, cache)
                entry = (
                    name,
                    result["sps"],
                    1.0,
                    result["mean_bpd"],
                    result["sps"],
                )
                _update_board(board, board_limit, entry)
                live.update(_build_leaderboard(board, i + 1, total, ref_sps))
                continue

            if ref_pooled is None:
                del result["pooled"], result["bpd_vec"]
                variants.append((name, result))
                continue

            composite, cos = _composite_score(ref_pooled, result)
            bpd_diff_val: float = (
                (ref_bpd - result["bpd_vec"]).abs().mean().item()  # type: ignore[union-attr]
            )
            result["cos"] = cos
            result["bpd_diff"] = bpd_diff_val
            result["composite"] = composite
            del result["pooled"], result["bpd_vec"]
            variants.append((name, result))

            _cache_result(cache, name, n_sentences, result)
            _save_bench_cache(checkpoint_id, cache)

            entry = (name, result["sps"], cos, result["mean_bpd"], composite)
            _update_board(board, board_limit, entry)

            live.update(_build_leaderboard(board, i + 1, total, ref_sps))

    del ref_pooled, ref_bpd

    if n_cached:
        console.print(f"    ({n_cached}/{total} loaded from cache)")
    if board:
        console.print(_build_leaderboard(board, total, total, ref_sps))

    return variants


def benchmark(  # pylint: disable=too-many-locals
    checkpoint_id: Optional[str] = None,
    *,
    rank_sweep: Optional[List[int]] = None,
) -> None:
    """Arena-style benchmark with progressive elimination.

    Generates every valid layer mask (first layer always kept) and combines
    them with the rank sweep.  Starts with _ARENA_INITIAL sentences,
    keeps the top 25% by sps * cos^4, adds _ARENA_STEP more sentences each
    round, and repeats until fewer than _ARENA_FINAL_SIZE would survive.
    The final round re-evaluates the top _ARENA_FINAL_SIZE variants at the
    accumulated sentence count and prints a full comparison table.
    """
    lock = read_lock()
    if lock is None:
        raise FileNotFoundError("checkpoint.lock not found")
    if checkpoint_id is None:
        checkpoint_id = lock["checkpoint_id"]
    if rank_sweep is None:
        rank_sweep = []

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    total_layers = get_total_layers(lock)
    configs = _build_all_configs(total_layers, rank_sweep)

    baselines = configs[:2]  # full fp32, fp16 — always run, never eliminated
    contestants = list(configs[2:])

    max_n = _arena_max_sentences(len(contestants))
    test_ids, test_mask = _make_test_batch(lock, max_n, device)

    criteria = lock.get("criteria", {})
    epoch = lock.get("epoch", "?")
    print(f"Benchmarking {checkpoint_id} on {device}  (epoch {epoch})")
    if criteria:
        parts = [f"{k}: {v:.4f}" for k, v in criteria.items()]
        print(f"  Best: {', '.join(parts)}")
    print(
        f"  {total_layers} layers, {len(configs)} variants, arena up to {max_n} sentences\n"
    )

    is_final = len(contestants) <= _ARENA_FINAL_SIZE
    round_num = 0

    while True:
        round_num += 1
        n = _ARENA_INITIAL + (round_num - 1) * _ARENA_STEP

        round_configs = baselines + contestants

        label = "Final round" if is_final else f"Round {round_num}"
        print(f"  {label} ({n} sentences, {len(contestants)} variants)")

        _pre_distill_configs(checkpoint_id, lock["model_type"], round_configs)
        results = _run_configs(
            lock,
            round_configs,
            test_ids[:n],
            test_mask[:n],
            device=device,
            n_sentences=n,
        )

        results_by_name = dict(results)

        scored: list[tuple[tuple[str, bool, str, int], str, float, float, float]] = []
        for cfg in contestants:
            cname = cfg[0]
            v = results_by_name[cname]
            scored.append((cfg, cname, v["composite"], v["cos"], v["sps"]))
        scored.sort(key=lambda x: x[2], reverse=True)

        if is_final:
            print()
            _print_sweep(results)
            break

        keep_n = max(int(len(scored) * _ARENA_KEEP_FRAC), 1)
        eliminated = [name for _, name, _, _, _ in scored[keep_n:]]
        if eliminated:
            elim_str = ", ".join(eliminated)
            if len(elim_str) > 72:
                elim_str = ", ".join(eliminated[:4]) + f" ... ({len(eliminated)} total)"
            print(f"    Eliminated {len(eliminated)}: {elim_str}")

        if keep_n < _ARENA_FINAL_SIZE:
            contestants = [cfg for cfg, _, _, _, _ in scored[:_ARENA_FINAL_SIZE]]
            is_final = True
        else:
            contestants = [cfg for cfg, _, _, _, _ in scored[:keep_n]]
        print()


# ---------------------------------------------------------------------------
# Benchmark display
# ---------------------------------------------------------------------------


def _print_sweep(
    variants: list[tuple[str, Dict[str, Any]]],
) -> None:
    """Print a vertical comparison table (one variant per row).

    Uses pre-computed ``cos``, ``bpd_diff``, and ``composite`` scalars
    already stored in each result dict by ``_run_configs``.
    """
    ref_sps = 1.0
    for name, v in variants:
        if name == "full fp32":
            ref_sps = v["sps"]
            break
    show_mem = any(v["mem_mb"] > 0 for _, v in variants)

    best_name = ""
    best_score = -1.0
    for name, v in variants:
        if name == "full fp32":
            continue
        if v["composite"] > best_score:
            best_score = v["composite"]
            best_name = name

    name_w = max(14, max(len(name) for name, _ in variants))
    headers = ["Variant", "Layers", "s/s", "Time", "BPD", "Emb cos", "BPD |d|", "Speed"]
    if show_mem:
        headers.insert(5, "Mem MB")
    widths = [name_w, 6, 7, 7, 9, 8, 8, 7]
    if show_mem:
        widths.insert(5, 6)

    print("  ".join(h.rjust(w) for h, w in zip(headers, widths)))
    print("  ".join("\u2500" * w for w in widths))

    sorted_variants = sorted(variants, key=lambda nv: nv[1]["sps"], reverse=True)
    for name, v in sorted_variants:
        speedup = v["sps"] / max(ref_sps, 1e-9)
        cols = [
            name.rjust(name_w),
            str(v["num_layers"]).rjust(6),
            f"{v['sps']:.0f}".rjust(7),
            f"{v['elapsed']:.2f}s".rjust(7),
            f"{v['mean_bpd']:.4f}".rjust(9),
            f"{v['cos']:.4f}".rjust(8),
            f"{v['bpd_diff']:.4f}".rjust(8),
            f"{speedup:.2f}x".rjust(7),
        ]
        if show_mem:
            cols.insert(5, f"{v['mem_mb']:.0f}".rjust(6))
        line = "  ".join(cols)
        if name == best_name:
            line += "  *"
        print(line)


# ---------------------------------------------------------------------------
# Test batch construction
# ---------------------------------------------------------------------------


def _make_test_batch(  # pylint: disable=too-many-locals
    lock: Dict[str, Any],
    n_sentences: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a padded batch of surface IDs from the dataset bundle.

    Uses the pre-encoded ``features["surface"]`` tensor and ``offsets``
    directly — the same token IDs the model was trained on.  Only
    grammatical sentences (``labels["gram"] == 1``) are included to
    match training data selection.
    """
    from scripts.dataset import resolve_dataset_by_id

    bundle = resolve_dataset_by_id(lock["dataset_id"])
    offsets = bundle["offsets"]
    surface = bundle["features"]["surface"]
    total = len(offsets) - 1

    gram = bundle.get("labels", {}).get("gram")
    if gram is not None:
        pool = [i for i in range(total) if int(gram[i].item()) == 1]
    else:
        pool = list(range(total))

    indices = pool[:n_sentences]
    if len(indices) < n_sentences:
        indices = (indices * ((n_sentences // len(indices)) + 1))[:n_sentences]

    encoded: list[torch.Tensor] = []
    for idx in indices:
        start = int(offsets[idx].item())
        end = int(offsets[idx + 1].item())
        encoded.append(surface[start:end])

    max_len = max(len(ids) for ids in encoded)
    padded = torch.zeros(n_sentences, max_len, dtype=torch.long)
    mask = torch.zeros(n_sentences, max_len, dtype=torch.float32)
    for i, ids in enumerate(encoded):
        padded[i, : len(ids)] = ids
        mask[i, : len(ids)] = 1.0

    return padded.to(device), mask.to(device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill recon_bpd checkpoint to FP16 for fast MPS inference.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Arena-style benchmark of layer-mask and low-rank variants",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-distill even if cached",
    )
    parser.add_argument(
        "--layer-mask",
        type=str,
        default="",
        help="Binary layer mask (1=keep, 0=drop), e.g. '110100000'",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="*",
        default=None,
        help="Output-head low-rank SVD ranks to benchmark (default: none)",
    )
    parser.add_argument(
        "--token-percentile",
        type=float,
        default=100.0,
        help="Surface token percentile to keep (default: 100.0 = no reduction)",
    )
    args = parser.parse_args()

    lock = read_lock()
    if lock is None:
        print("checkpoint.lock not found. Run: scripts/cc checkpoint pull recon_bpd")
        return

    checkpoint_id = lock["checkpoint_id"]
    print(f"Checkpoint: {checkpoint_id}")

    distill_checkpoint(
        checkpoint_id,
        lock["model_type"],
        layer_mask=args.layer_mask,
        token_percentile=args.token_percentile,
        force=args.force,
    )

    if args.benchmark:
        print()
        benchmark(
            checkpoint_id,
            rank_sweep=args.ranks,
        )


if __name__ == "__main__":
    main()
