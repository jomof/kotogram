"""Find identical KC-activation groups and top-5 closest non-identical pairs.

Uses the KC probability vector (1024d, sparse) as the sentence fingerprint
rather than the pooled encoder embedding.  Two sentences with the same KC
activations are genuinely semantically equivalent from the model's perspective.
"""

import heapq
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from rich.console import Console
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table

from scripts.cc_common import parallel_parse_and_encode

IDENTICAL_GROUPS_FILE = Path("scratch/identical_groups.txt")
KC_CACHE = Path(".cc/corpus-kc-probs.npy")


def compute_kc_probs(
    sents: list[str], console: Console
) -> np.ndarray:
    """Compute KC probability vectors for all sentences via batched inference."""
    from scripts.recon_bpd.inference import load_model_from_checkpoint

    console.print("Loading model...")
    model, _tok, _ckpt = load_model_from_checkpoint(layer_mask="")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    console.print(f"Parsing {len(sents):,} sentences...")
    encoded = parallel_parse_and_encode(sents)

    kc_dim = model.cfg.kc_vocab_size
    all_kc = np.zeros((len(sents), kc_dim), dtype=np.float32)
    lengths = np.array([len(e["surface"]) for e in encoded], dtype=np.int64)
    order = np.argsort(lengths)

    batch_size = 64
    t0 = time.monotonic()
    scored = 0

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Computing KC probs...", total=len(sents))
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            batch_enc = [encoded[i] for i in batch_idx]
            max_len = int(lengths[batch_idx].max())

            padded = np.zeros((len(batch_enc), max_len), dtype=np.int64)
            mask_np = np.zeros((len(batch_enc), max_len), dtype=np.float32)
            for i, enc in enumerate(batch_enc):
                ids = enc["surface"]
                padded[i, : len(ids)] = ids
                mask_np[i, : len(ids)] = 1.0

            surface_ids = torch.from_numpy(padded).to(device)
            mask = torch.from_numpy(mask_np).to(device)

            with torch.inference_mode():
                token_remap = getattr(model, "_token_remap", None)
                if token_remap is not None:
                    clamped = surface_ids.clamp(max=token_remap.size(0) - 1)
                    surface_ids = token_remap.to(device)[clamped]

                pooled = model.encode(surface_ids, mask)
                kc_raw, _ = model.kc_head.forward_with_raw(pooled)
                kc_clamp = getattr(model, "_kc_clamp", 12.0)
                kc_temp = getattr(model, "_kc_temperature", 1.0)
                kc_raw = kc_raw.clamp(-kc_clamp, kc_clamp)
                kc_probs = torch.sigmoid(kc_raw / kc_temp)

            all_kc[batch_idx] = kc_probs.cpu().numpy()

            if device.type == "mps":
                torch.mps.empty_cache()

            scored += len(batch_idx)
            elapsed = time.monotonic() - t0
            els = scored / elapsed if elapsed > 0 else 0
            progress.update(task, advance=len(batch_idx), description=f"KC probs ({els:.0f} el/s)")

    elapsed = time.monotonic() - t0
    console.print(f"  Done in {elapsed:.1f}s ({scored / elapsed:.0f} el/s)")

    hot = (all_kc > 0.9).sum(axis=1)
    console.print(
        f"  KC stats: dim={kc_dim}, hot per sentence: "
        f"mean={hot.mean():.1f}  median={np.median(hot):.0f}  "
        f"min={hot.min()}  max={hot.max()}"
    )

    np.save(str(KC_CACHE), all_kc)
    console.print(f"  Cached to {KC_CACHE}")
    return all_kc


def find_identical_groups(
    t: torch.Tensor, sents: list[str], console: Console
) -> list[list[int]]:
    """Hash binarized KC activations to find groups of identical vectors. O(n)."""
    console.print("Finding identical KC-activation groups...")
    t0 = time.monotonic()

    binary = (t > 0.9).to(torch.uint8).numpy()
    groups: dict[bytes, list[int]] = defaultdict(list)
    for i, row in enumerate(binary):
        groups[row.tobytes()].append(i)

    singletons = sum(1 for idxs in groups.values() if len(idxs) == 1)
    multi = [idxs for idxs in groups.values() if len(idxs) > 1]
    total_dupes = sum(len(g) for g in multi)
    elapsed = time.monotonic() - t0
    console.print(
        f"  {singletons:,} unique sentences, {len(multi):,} collision groups"
        f" containing {total_dupes:,} sentences ({elapsed:.1f}s)"
    )
    return multi


def write_identical_groups(
    groups: list[list[int]], sents: list[str], console: Console
) -> None:
    """Write identical groups to file, one group per line separated by |."""
    with open(IDENTICAL_GROUPS_FILE, "w", encoding="utf-8") as f:
        for idxs in sorted(groups, key=lambda g: -len(g)):
            f.write("|".join(sents[i] for i in idxs) + "\n")
    console.print(f"  Wrote {len(groups):,} groups to {IDENTICAL_GROUPS_FILE}")


def print_identical_summary(
    groups: list[list[int]], sents: list[str], hot_counts: np.ndarray, console: Console
) -> None:
    tbl = Table(
        title=f"Identical KC Groups ({len(groups):,} groups)", show_header=True
    )
    tbl.add_column("Size", justify="right", width=5)
    tbl.add_column("KCs", justify="right", width=4)
    tbl.add_column("Sample sentences")

    for g in sorted(groups, key=lambda g: -len(g))[:10]:
        samples = "  ↔  ".join(sents[i][:40] for i in g[:3])
        if len(g) > 3:
            samples += f"  (+{len(g) - 3} more)"
        tbl.add_row(str(len(g)), str(int(hot_counts[g[0]])), samples)

    if len(groups) > 10:
        tbl.add_row("...", "", f"({len(groups) - 10} more groups)")
    console.print(tbl)


def _edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = prev[j - 1] if ca == cb else 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[-1]


def build_table(
    heap: list[tuple[float, int, int]],
    sents: list[str],
    hot_counts: np.ndarray,
    qi: int,
    total_q: int,
    elapsed: float,
    eta: float,
) -> Table:
    pct = 100 * qi / total_q if total_q else 0
    tbl = Table(
        title=(
            f"Top Non-Identical Pairs  [{qi}/{total_q}] {pct:.0f}%  "
            f"elapsed={elapsed:.0f}s  eta={eta:.0f}s"
        ),
        show_header=True,
    )
    tbl.add_column("#", style="bold", width=3)
    tbl.add_column("sim", justify="right", width=8)
    tbl.add_column("ed", justify="right", width=4)
    tbl.add_column("KCs", justify="right", width=5)
    tbl.add_column("Sentence A")
    tbl.add_column("Sentence B")

    ranked = sorted(heap, key=lambda x: -x[0])
    for rank, (sim, i, j) in enumerate(ranked[:5], 1):
        kcs = f"{int(hot_counts[i])}/{int(hot_counts[j])}"
        ed = str(_edit_distance(sents[i], sents[j]))
        tbl.add_row(str(rank), f"{sim:.4f}", ed, kcs, sents[i][:60], sents[j][:60])
    for _ in range(5 - min(len(ranked), 5)):
        tbl.add_row("", "", "", "", "", "")

    return tbl


def main():
    console = Console()

    with open(".cc/corpus-sentences.txt", encoding="utf-8") as f:
        sents = [ln.rstrip("\n") for ln in f]
    console.print(f"{len(sents):,} corpus sentences")

    if KC_CACHE.exists():
        console.print(f"Loading cached KC probs from {KC_CACHE}...")
        kc = np.load(str(KC_CACHE))
        if kc.shape[0] != len(sents):
            console.print("  Cache size mismatch, recomputing...")
            kc = compute_kc_probs(sents, console)
    else:
        kc = compute_kc_probs(sents, console)

    n = min(kc.shape[0], len(sents))
    kc = kc[:n]
    sents = sents[:n]
    console.print(f"{n:,} sentences, {kc.shape[1]} KCs")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    console.print(f"Device: {device}")

    t = torch.from_numpy(kc).float()
    hot_counts = (kc > 0.9).sum(axis=1)

    # --- Phase 1: identical KC groups ---
    groups = find_identical_groups(t, sents, console)
    write_identical_groups(groups, sents, console)
    print_identical_summary(groups, sents, hot_counts, console)

    identical_pairs: set[tuple[int, int]] = set()
    for g in groups:
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                identical_pairs.add((g[a], g[b]) if g[a] < g[b] else (g[b], g[a]))
    console.print(f"  {len(identical_pairs):,} identical pairs to skip\n")

    # --- Phase 2: top-5 non-identical pairs (cosine on KC probs) ---
    norms = t.norm(dim=1, keepdim=True).clamp(min=1e-8)
    t = t / norms

    Q_CHUNK = 2048
    K_CHUNK = 200_000
    TOP_K = 5
    heap: list[tuple[float, int, int]] = []
    t0 = time.monotonic()
    total_q = (n + Q_CHUNK - 1) // Q_CHUNK

    with Live(
        build_table([], sents, hot_counts, 0, total_q, 0, 0),
        console=console,
        refresh_per_second=2,
    ) as live:
        for qi, q_start in enumerate(range(0, n, Q_CHUNK)):
            q_end = min(q_start + Q_CHUNK, n)
            q = t[q_start:q_end].to(device)
            q_size = q_end - q_start

            best_sim = torch.full((q_size,), -1.0, device=device)
            best_idx = torch.zeros(q_size, dtype=torch.long, device=device)

            q_global = torch.arange(q_start, q_end, device=device).unsqueeze(1)

            for k_start in range(0, n, K_CHUNK):
                k_end = min(k_start + K_CHUNK, n)
                k = t[k_start:k_end].to(device)
                sim = q @ k.T

                k_indices = torch.arange(k_start, k_end, device=device).unsqueeze(0)
                sim[k_indices <= q_global] = -1.0

                vals, idxs = sim.max(dim=1)
                better = vals > best_sim
                best_sim[better] = vals[better]
                best_idx[better] = idxs[better] + k_start
                del k, sim, k_indices

            best_sim_cpu = best_sim.cpu()
            best_idx_cpu = best_idx.cpu()
            for li in range(q_size):
                gi = q_start + li
                s = best_sim_cpu[li].item()
                j = best_idx_cpu[li].item()
                if j <= gi or s <= -1.0:
                    continue
                pair = (gi, j) if gi < j else (j, gi)
                if pair in identical_pairs:
                    continue
                if len(heap) < TOP_K:
                    heapq.heappush(heap, (s, gi, j))
                elif s > heap[0][0]:
                    heapq.heapreplace(heap, (s, gi, j))

            del q
            elapsed = time.monotonic() - t0
            rate = (qi + 1) / elapsed if elapsed > 0 else 1
            eta = (total_q - qi - 1) / rate
            live.update(build_table(list(heap), sents, hot_counts, qi + 1, total_q, elapsed, eta))

    elapsed = time.monotonic() - t0
    results = sorted(heap, key=lambda x: -x[0])

    console.print()
    for rank, (sim, i, j) in enumerate(results, 1):
        console.print(f"\n[bold]#{rank}  sim={sim:.6f}[/bold]")
        console.print(f"  [dim][{i:>7,}][/dim] {sents[i]}")
        console.print(f"  [dim][{j:>7,}][/dim] {sents[j]}")
    console.print(f"\n[dim]Completed in {elapsed:.1f}s[/dim]")


if __name__ == "__main__":
    main()
