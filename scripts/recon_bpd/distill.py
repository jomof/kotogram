"""Distill a recon_bpd checkpoint to FP16 for fast local MPS inference.

Converts all weights to float16, halving memory and enabling vectorized
output projection (single [B, T, V] matmul instead of per-position loop).
Optionally drops encoder layers and/or applies low-rank SVD to the output
head for further speedup.

Usage:
    python -m scripts.recon_bpd.distill                # distill checkpoint.lock
    python -m scripts.recon_bpd.distill --benchmark    # sweep layer-drop + rank
    python -m scripts.recon_bpd.distill --force         # re-distill even if cached
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional

import torch

from scripts.checkpoint import LOCAL_CACHE, download_checkpoint, read_lock

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _distilled_path(
    checkpoint_id: str,
    drop_layers: int = 0,
    output_rank: int = 0,
) -> str:
    parts = [f"ckpt-{checkpoint_id}.distilled"]
    if drop_layers > 0:
        parts.append(f"-drop{drop_layers}")
    if output_rank > 0:
        parts.append(f"-rank{output_rank}")
    parts.append(".pt")
    return os.path.join(LOCAL_CACHE, "".join(parts))


# ---------------------------------------------------------------------------
# Layer-drop helpers
# ---------------------------------------------------------------------------


def _max_droppable(total: int) -> int:
    """Max droppable layers (all except the first, 0-indexed layer 0)."""
    return total - 1


def _auto_sweep(max_drop: int) -> List[int]:
    """Generate a reasonable sweep of drop counts from 1 to *max_drop*."""
    if max_drop <= 10:
        return list(range(1, max_drop + 1))
    result: List[int] = torch.linspace(1, max_drop, 8).round().long().unique().tolist()
    return result


def _drop_order(total: int) -> List[int]:
    """Layer drop order: odd 1-rel first (3,5,7,...), then even 1-rel backwards (8,6,4,2).

    0-indexed phase 1: [2, 4, 6, ...]  (odd 1-relative positions)
    0-indexed phase 2: [..., 5, 3, 1]  (even 1-relative positions, high to low)
    Layer 0 is never dropped.
    """
    phase1 = list(range(2, total, 2))
    phase2 = list(reversed(range(1, total, 2)))
    return phase1 + phase2


def _select_kept_layers(total: int, drop: int) -> List[int]:
    """Select which layers to keep after dropping *drop* layers."""
    if drop == 0:
        return list(range(total))
    droppable = _drop_order(total)
    if drop > len(droppable):
        raise ValueError(
            f"Cannot drop {drop} layers: only {len(droppable)} "
            f"droppable positions in {total}-layer model"
        )
    to_drop = set(droppable[:drop])
    return [i for i in range(total) if i not in to_drop]


def _drop_encoder_layers(
    state: Dict[str, torch.Tensor],
    drop_layers: int,
) -> Dict[str, torch.Tensor]:
    """Remove encoder layers and renumber survivors to 0..K-1."""
    layer_nums = set()
    for k in state:
        if "encoder.layers." in k:
            parts = k.split(".")
            idx = parts.index("layers") + 1
            if idx < len(parts) and parts[idx].isdigit():
                layer_nums.add(int(parts[idx]))
    total = max(layer_nums) + 1 if layer_nums else 1

    kept = _select_kept_layers(total, drop_layers)
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


def distill_checkpoint(  # pylint: disable=too-many-locals
    checkpoint_id: str,
    model_type: str = "recon_bpd",
    *,
    drop_layers: int = 0,
    output_rank: int = 0,
    force: bool = False,
) -> str:
    """Convert a full checkpoint to FP16, optionally with layer-drop and low-rank.

    Returns the path to the distilled checkpoint.
    """
    out_path = _distilled_path(checkpoint_id, drop_layers, output_rank)
    if os.path.exists(out_path) and not force:
        return out_path

    full_path = download_checkpoint(model_type, checkpoint_id)
    extras = []
    if drop_layers:
        extras.append(f"dropping {drop_layers} layers")
    if output_rank:
        extras.append(f"rank-{output_rank} output head")
    label = f" ({', '.join(extras)})" if extras else ""
    print(f"  Distilling {checkpoint_id} to FP16{label}...")

    ckpt = torch.load(full_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    cleaned = {k.replace("_orig_mod.", ""): v.half() for k, v in state.items()}

    if drop_layers > 0:
        cleaned = _drop_encoder_layers(cleaned, drop_layers)
    if output_rank > 0:
        cleaned = _apply_low_rank(cleaned, output_rank)

    distilled_ckpt: Dict[str, Any] = {
        "model_state": cleaned,
        "distilled": True,
        "drop_layers": drop_layers,
        "output_rank": output_rank,
        "source_checkpoint_id": checkpoint_id,
    }
    if "config_dict" in ckpt:
        cfg = dict(ckpt["config_dict"])
        if drop_layers > 0:
            cfg["num_layers"] = cfg.get("num_layers", 16) - drop_layers
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
    drop_layers: int = 0,
    output_rank: int = 0,
) -> str:
    """Return path to distilled checkpoint, creating it if needed."""
    out_path = _distilled_path(checkpoint_id, drop_layers, output_rank)
    if os.path.exists(out_path):
        return out_path
    return distill_checkpoint(
        checkpoint_id,
        model_type,
        drop_layers=drop_layers,
        output_rank=output_rank,
    )


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
    drop_layers: int = 0,
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
        drop_layers=drop_layers,
        output_rank=output_rank,
    )
    num_layers = model.cfg.num_layers
    use_vec = getattr(model, "_distilled", False)
    model.to(device)

    # Warmup
    with torch.inference_mode():
        embed_and_bpd(
            model,
            test_ids[:4],
            test_mask[:4],
            vectorized=use_vec,
        )
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


_SCREEN_N = 200
_SHORTLIST_K = 5


def _get_total_layers(lock: Dict[str, Any]) -> int:
    """Peek at checkpoint config to get encoder layer count without inference."""
    full_path = download_checkpoint(lock["model_type"], lock["checkpoint_id"])
    ckpt = torch.load(full_path, map_location="cpu", weights_only=False)
    if "config_dict" in ckpt:
        return int(ckpt["config_dict"].get("num_layers", 9))
    layer_nums = set()
    for k in ckpt["model_state"]:
        if "encoder.layers." in k:
            parts = k.split(".")
            idx = parts.index("layers") + 1
            if idx < len(parts) and parts[idx].isdigit():
                layer_nums.add(int(parts[idx]))
    return max(layer_nums) + 1 if layer_nums else 1


def _build_all_configs(
    total_layers: int,
    drop_sweep: Optional[List[int]],
    rank_sweep: List[int],
) -> list[tuple[str, bool, int, int]]:
    """Generate (name, distilled, drop_layers, output_rank) for all variants.

    Includes baselines, individual sweeps, and the full grid of combinations.
    """
    max_drop = _max_droppable(total_layers)
    if drop_sweep is None:
        drop_sweep = _auto_sweep(max_drop)

    configs: list[tuple[str, bool, int, int]] = [
        ("full fp32", False, 0, 0),
        ("fp16", True, 0, 0),
    ]
    valid_drops: List[int] = []
    for d in drop_sweep:
        if d > max_drop:
            print(
                f"  Skipping drop {d}: only {max_drop} droppable in {total_layers} layers"
            )
            continue
        configs.append((f"fp16 -{d}L", True, d, 0))
        valid_drops.append(d)

    for r in rank_sweep:
        configs.append((f"fp16 r{r}", True, 0, r))

    for d in valid_drops:
        for r in rank_sweep:
            configs.append((f"fp16 -{d}L r{r}", True, d, r))

    return configs


def _run_configs(
    lock: Dict[str, Any],
    configs: list[tuple[str, bool, int, int]],
    test_ids: torch.Tensor,
    test_mask: torch.Tensor,
    *,
    device: torch.device,
    n_sentences: int,
) -> list[tuple[str, Dict[str, Any]]]:
    """Run all configs and return (name, result) pairs."""
    variants: list[tuple[str, Dict[str, Any]]] = []
    for name, dist, drop, rank in configs:
        result = _run_one_variant(
            lock,
            test_ids,
            test_mask,
            device,
            n_sentences,
            distilled=dist,
            drop_layers=drop,
            output_rank=rank,
        )
        variants.append((name, result))
    return variants


def benchmark(  # pylint: disable=too-many-locals
    checkpoint_id: Optional[str] = None,
    *,
    n_sentences: int = 200,
    drop_sweep: Optional[List[int]] = None,
    rank_sweep: Optional[List[int]] = None,
) -> None:
    """Compare distillation variants with two-phase screening.

    When *n_sentences* > 200, screens the full grid (individual + all drop x rank
    combinations) cheaply at 200 sentences, then re-runs only the top-K fastest
    at the requested sentence count for accurate throughput numbers.
    """
    lock = read_lock()
    if lock is None:
        raise FileNotFoundError("checkpoint.lock not found")
    if checkpoint_id is None:
        checkpoint_id = lock["checkpoint_id"]
    if rank_sweep is None:
        rank_sweep = [16, 32, 64, 128]

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    total_layers = _get_total_layers(lock)
    configs = _build_all_configs(total_layers, drop_sweep, rank_sweep)
    n_combos = sum(1 for _, _, d, r in configs if d > 0 and r > 0)

    print(f"Benchmarking {checkpoint_id} on {device} ({n_sentences} sentences)")
    use_screen = n_sentences > _SCREEN_N and len(configs) > _SHORTLIST_K + 2

    if use_screen:
        print(
            f"  Screening {len(configs)} variants at {_SCREEN_N} sentences"
            f" ({n_combos} combinations)\n"
        )
        test_ids, test_mask = _make_test_batch(lock, n_sentences, device)
        screen_results = _run_configs(
            lock,
            configs,
            test_ids[:_SCREEN_N],
            test_mask[:_SCREEN_N],
            device=device,
            n_sentences=_SCREEN_N,
        )
        # Always keep baselines (first two); shortlist rest by speed
        baselines = configs[:2]
        ranked = sorted(
            zip(configs[2:], screen_results[2:]),
            key=lambda cr: cr[1][1]["sps"],
            reverse=True,
        )
        top_configs = [c for c, _ in ranked[:_SHORTLIST_K]]
        top_names = [c[0] for c in top_configs]
        print(f"  \u2192 Top {_SHORTLIST_K}: {', '.join(top_names)}")
        print(
            f"  Full benchmark: {len(baselines) + len(top_configs)} variants"
            f" at {n_sentences} sentences\n"
        )

        final_configs = baselines + top_configs
        variants = _run_configs(
            lock,
            final_configs,
            test_ids,
            test_mask,
            device=device,
            n_sentences=n_sentences,
        )
    else:
        print()
        test_ids, test_mask = _make_test_batch(lock, n_sentences, device)
        variants = _run_configs(
            lock,
            configs,
            test_ids,
            test_mask,
            device=device,
            n_sentences=n_sentences,
        )

    ref = dict(variants)["full fp32"]
    _print_sweep(ref, variants)


# ---------------------------------------------------------------------------
# Benchmark display
# ---------------------------------------------------------------------------


def _print_sweep(
    ref: Dict[str, Any],
    variants: list[tuple[str, Dict[str, Any]]],
) -> None:
    """Print a vertical comparison table (one variant per row)."""
    ref_pooled, ref_bpd, ref_sps = ref["pooled"], ref["bpd_vec"], ref["sps"]
    show_mem = ref["mem_mb"] > 0

    headers = ["Variant", "Layers", "s/s", "Time", "BPD", "Emb cos", "BPD |d|", "Speed"]
    if show_mem:
        headers.insert(5, "Mem MB")
    widths = [14, 6, 7, 7, 9, 8, 8, 7]
    if show_mem:
        widths.insert(5, 6)

    print("  ".join(h.rjust(w) for h, w in zip(headers, widths)))
    print("  ".join("\u2500" * w for w in widths))

    sorted_variants = sorted(variants, key=lambda nv: nv[1]["sps"], reverse=True)
    for name, v in sorted_variants:
        _print_variant_row(
            name, v, ref_pooled, ref_bpd, ref_sps=ref_sps, show_mem=show_mem
        )


def _print_variant_row(
    name: str,
    v: Dict[str, Any],
    ref_pooled: torch.Tensor,
    ref_bpd: torch.Tensor,
    *,
    ref_sps: float,
    show_mem: bool,
) -> None:
    """Print one row of the vertical benchmark table."""
    cos = (
        torch.nn.functional.cosine_similarity(  # pylint: disable=not-callable
            ref_pooled,
            v["pooled"],
            dim=-1,
        )
        .mean()
        .item()
    )
    bdiff: float = (ref_bpd - v["bpd_vec"]).abs().mean().item()
    speedup = v["sps"] / max(ref_sps, 1e-9)

    cols = [
        name.rjust(14),
        str(v["num_layers"]).rjust(6),
        f"{v['sps']:.0f}".rjust(7),
        f"{v['elapsed']:.2f}s".rjust(7),
        f"{v['mean_bpd']:.4f}".rjust(9),
        f"{cos:.4f}".rjust(8),
        f"{bdiff:.4f}".rjust(8),
        f"{speedup:.2f}x".rjust(7),
    ]
    if show_mem:
        cols.insert(5, f"{v['mem_mb']:.0f}".rjust(6))
    print("  ".join(cols))


# ---------------------------------------------------------------------------
# Test batch construction
# ---------------------------------------------------------------------------


def _make_test_batch(  # pylint: disable=too-many-locals
    lock: Dict[str, Any],
    n_sentences: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a padded batch of surface IDs from the corpus."""
    from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
    from kotogram.tokenizer import Tokenizer
    from scripts.dataset import resolve_dataset_by_id

    bundle = resolve_dataset_by_id(lock["dataset_id"])
    tokenizer = Tokenizer()
    tokenizer.load_state({"field_vocabs": bundle["vocab"], "frozen": True})
    parser = SudachiJapaneseParser()

    sentences = bundle.get("sentences", [])[:n_sentences]
    if len(sentences) < n_sentences:
        sentences = sentences * ((n_sentences // max(len(sentences), 1)) + 1)
        sentences = sentences[:n_sentences]

    encoded = []
    for s in sentences:
        kotogram = parser.japanese_to_kotogram(s)
        enc = tokenizer.encode(kotogram)
        encoded.append(enc["surface"])

    max_len = max(len(ids) for ids in encoded)
    padded = torch.zeros(n_sentences, max_len, dtype=torch.long)
    mask = torch.zeros(n_sentences, max_len, dtype=torch.float32)
    for i, ids in enumerate(encoded):
        padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
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
        help="Sweep layer-drop and low-rank variants",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-distill even if cached",
    )
    parser.add_argument(
        "--n-sentences",
        type=int,
        default=200,
        help="Sentences for benchmark (default: 200)",
    )
    parser.add_argument(
        "--drop-layers",
        type=int,
        nargs="*",
        default=None,
        help="Layer-drop counts to benchmark (default: auto)",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="*",
        default=None,
        help="Output-head ranks to benchmark (default: 16 32 64 128)",
    )
    args = parser.parse_args()

    lock = read_lock()
    if lock is None:
        print("checkpoint.lock not found. Run: scripts/cc checkpoint pull recon_bpd")
        return

    checkpoint_id = lock["checkpoint_id"]
    print(f"Checkpoint: {checkpoint_id}")

    distill_checkpoint(checkpoint_id, lock["model_type"], force=args.force)

    if args.benchmark:
        print()
        benchmark(
            checkpoint_id,
            n_sentences=args.n_sentences,
            drop_sweep=args.drop_layers,
            rank_sweep=args.ranks,
        )


if __name__ == "__main__":
    main()
