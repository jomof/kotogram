"""Find identical embedding groups and top-5 closest non-identical pairs."""

import heapq
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.live import Live
from rich.table import Table

IDENTICAL_GROUPS_FILE = Path("scratch/identical_groups.txt")


def find_identical_groups(
    t: torch.Tensor, sents: list[str], console: Console
) -> list[list[int]]:
    """Hash normalized embeddings to find groups of identical vectors. O(n)."""
    console.print("Finding identical embedding groups...")
    t0 = time.monotonic()

    # Quantize to int16 for hashing (precision ~3e-5, well within threshold)
    quantized = (t * 16384).to(torch.int16).numpy()
    groups: dict[bytes, list[int]] = defaultdict(list)
    for i, row in enumerate(quantized):
        groups[row.tobytes()].append(i)

    multi = [idxs for idxs in groups.values() if len(idxs) > 1]
    total_dupes = sum(len(g) for g in multi)
    elapsed = time.monotonic() - t0
    console.print(
        f"  {len(multi):,} groups, {total_dupes:,} sentences with duplicates "
        f"({elapsed:.1f}s)"
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
    groups: list[list[int]], sents: list[str], console: Console
) -> None:
    tbl = Table(
        title=f"Identical Embedding Groups ({len(groups):,} groups)", show_header=True
    )
    tbl.add_column("Size", justify="right", width=5)
    tbl.add_column("Sample sentences")

    for g in sorted(groups, key=lambda g: -len(g))[:10]:
        samples = "  ↔  ".join(sents[i][:40] for i in g[:3])
        if len(g) > 3:
            samples += f"  (+{len(g) - 3} more)"
        tbl.add_row(str(len(g)), samples)

    if len(groups) > 10:
        tbl.add_row("...", f"({len(groups) - 10} more groups)")
    console.print(tbl)


def build_table(
    heap: list[tuple[float, int, int]],
    sents: list[str],
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
    tbl.add_column("Sentence A")
    tbl.add_column("Sentence B")

    ranked = sorted(heap, key=lambda x: -x[0])
    for rank, (sim, i, j) in enumerate(ranked[:5], 1):
        tbl.add_row(str(rank), f"{sim:.4f}", sents[i][:60], sents[j][:60])
    for _ in range(5 - min(len(ranked), 5)):
        tbl.add_row("", "", "", "")

    return tbl


def main():
    console = Console()
    console.print("Loading embeddings...")
    emb = np.load(".cc/corpus-embeddings.npy")
    with open(".cc/corpus-sentences.txt", encoding="utf-8") as f:
        sents = [ln.rstrip("\n") for ln in f]

    n = min(emb.shape[0], len(sents))
    emb = emb[:n]
    sents = sents[:n]
    console.print(f"{n:,} sentences, {emb.shape[1]}d")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    console.print(f"Device: {device}")

    t = torch.from_numpy(emb).float()
    norms = t.norm(dim=1, keepdim=True).clamp(min=1e-8)
    t = t / norms

    # --- Phase 1: identical groups (instant) ---
    groups = find_identical_groups(t, sents, console)
    write_identical_groups(groups, sents, console)
    print_identical_summary(groups, sents, console)

    # Build set of identical pairs for fast lookup during NN search
    identical_pairs: set[tuple[int, int]] = set()
    for g in groups:
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                identical_pairs.add((g[a], g[b]) if g[a] < g[b] else (g[b], g[a]))
    console.print(f"  {len(identical_pairs):,} identical pairs to skip\n")

    # --- Phase 2: top-5 non-identical pairs (live) ---
    Q_CHUNK = 2048
    K_CHUNK = 200_000
    TOP_K = 5
    heap: list[tuple[float, int, int]] = []
    t0 = time.monotonic()
    total_q = (n + Q_CHUNK - 1) // Q_CHUNK

    with Live(
        build_table([], sents, 0, total_q, 0, 0),
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
            live.update(build_table(list(heap), sents, qi + 1, total_q, elapsed, eta))

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
