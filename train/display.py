"""Display logic for training progress reporting."""

import math
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

console = Console()


def print_phase_header(phase: str, info: Optional[str] = None) -> None:
    """Print a header for a training phase."""
    icon = {
        "MLM": "📝 ",
        "KC": "🧠 ",
        "Style": "🎨 ",
    }.get(phase, "")
    text = (
        f"{icon}{phase} Pretraining" if "Pretraining" not in phase else f"{icon}{phase}"
    )
    if info:
        text += f" ({info})"
    console.print(f"\n[bold blue]{text}[/bold blue]")


def print_progress_bar(batch_idx: int, total_batches: int, loss: float) -> None:
    """Print a simple text-based progress bar."""
    progress = (batch_idx + 1) / max(1, total_batches)
    bar_width = 30
    filled_width = int(bar_width * progress)
    bar = "=" * filled_width + ">" + "." * (bar_width - filled_width - 1)
    import sys

    sys.stdout.write(f"\r  [{bar}] {batch_idx + 1}/{total_batches} loss={loss:.4f}")
    sys.stdout.flush()


def print_kc_first_batch_debug(
    epoch: int,
    kc_logits: Any,
    kc_probs: Any,
    sparse: Any,
    target_logits: Dict[str, Any],
    batch: Dict[str, Any],
    kc_topk: int,
    kc_vocab_size: int,
    device: Any,
    pos_weight_cap: float = 50.0,
    pos_weight_eps: float = 1e-6,
) -> None:
    """Detailed debug prints for the first batch of a KC epoch."""
    import torch

    print(f"\n  --- [KC Epoch {epoch} First Batch Debug] ---")

    # A) KC head stats
    dense_density = (kc_probs > 0.1).float().mean().item()
    nonzero_per_sample = (sparse > 0).sum(dim=-1).float().mean().item()
    print(
        f"  KC Activations: logits_mean={kc_logits.mean().item():.4f} std={kc_logits.std().item():.4f} min={kc_logits.amin().item():.4f} max={kc_logits.amax().item():.4f}"
    )
    print(
        f"  KC Probs:       mean={kc_probs.mean().item():.4f} std={kc_probs.std().item():.4f} min={kc_probs.amin().item():.4f} max={kc_probs.amax().item():.4f} density(>0.1)={dense_density:.4f}"
    )
    print(
        f"  KC Sparse:      mean={sparse.mean().item():.4f} nonzeros/sample={nonzero_per_sample:.1f}"
    )

    topk_vals, topk_inds = torch.topk(kc_probs, k=kc_topk, dim=-1)
    print(
        f"  KC Top-{kc_topk} Vals:  mean={topk_vals.mean().item():.4f} min={topk_vals.amin().item():.4f} max={topk_vals.amax().item():.4f}"
    )

    # B) Top-K diversity
    flat_inds = topk_inds.reshape(-1)
    unique_k = torch.unique(flat_inds).numel()
    coverage = unique_k / kc_vocab_size
    unique_per_sample = torch.unique(topk_inds[0]).numel()

    print(
        f"  KC Diversity:   unique_kcs_in_batch={unique_k} coverage={coverage:.4f} unique_sample_0={unique_per_sample}"
    )
    print(f"  KC Example Inds: {topk_inds[0][: min(10, kc_topk)].tolist()}")

    # C) Decoder output sanity per structural head
    print("  KC Decoder Sanity (Structural + Separation):")
    sorted_heads = sorted(target_logits.keys())
    for name in sorted_heads[:8]:
        logits = target_logits[name]
        target_key = f"kc_targets_{name}"
        if target_key not in batch:
            continue

        targets = batch[target_key].to(device).float()
        with torch.no_grad():
            # 1. Compute density and pos_weight as Loss will
            pos_count = targets.sum()
            total_count = targets.numel()
            p = (pos_count / (total_count + pos_weight_eps)).clamp(
                min=pos_weight_eps, max=1.0 - pos_weight_eps
            )
            pos_w = ((1.0 - p) / p).clamp(min=1.0, max=pos_weight_cap)
            # 2. Probability-based metrics
            probs = torch.sigmoid(logits)
            prob_mean = probs.mean().item()

            # Adaptive Thresholds
            p_prior = p.item()
            dp = (probs > p_prior).float().mean().item()
            d2p = (probs > min(0.9, 2 * p_prior)).float().mean().item()

            thresh_str = f"dp={dp:.2f} d2p={d2p:.2f}"
            if p_prior < 0.1:
                d005 = (probs > 0.05).float().mean().item()
                d002 = (probs > 0.02).float().mean().item()
                d001 = (probs > 0.01).float().mean().item()
                thresh_str += f" d05={d005:.2f} d02={d002:.2f} d01={d001:.2f}"
            else:
                dens_01 = (probs > 0.1).float().mean().item()
                dens_05 = (probs > 0.5).float().mean().item()
                thresh_str += f" d1={dens_01:.2f} d5={dens_05:.2f}"

            pos_mask = targets > 0.5
            neg_mask = ~pos_mask

            pos_probs = probs[pos_mask]
            neg_probs = probs[neg_mask]

            pos_p = pos_probs.mean().item() if pos_mask.any() else float("nan")
            neg_p = neg_probs.mean().item() if neg_mask.any() else float("nan")
            gap_p = pos_p - neg_p if (pos_mask.any() and neg_mask.any()) else 0.0

            # ROC-AUC calculation (subsampled for efficiency)
            auc = float("nan")
            if pos_mask.any() and neg_mask.any():
                n_pos_all = pos_mask.sum().item()
                n_neg_all = neg_mask.sum().item()

                # Subsample if too large
                max_samples = 2000
                if n_pos_all > max_samples or n_neg_all > max_samples:
                    # Random subsample
                    pos_idx = torch.where(pos_mask.view(-1))[0]
                    neg_idx = torch.where(neg_mask.view(-1))[0]

                    if pos_idx.numel() > max_samples:
                        pos_idx = pos_idx[torch.randperm(pos_idx.numel())[:max_samples]]
                    if neg_idx.numel() > max_samples:
                        neg_idx = neg_idx[torch.randperm(neg_idx.numel())[:max_samples]]

                    n_pos = pos_idx.numel()
                    n_neg = neg_idx.numel()

                    sub_probs = torch.cat(
                        [probs.view(-1)[pos_idx], probs.view(-1)[neg_idx]]
                    )
                    sub_labels = torch.cat(
                        [
                            torch.ones(n_pos, device=device),
                            torch.zeros(n_neg, device=device),
                        ]
                    )
                else:
                    n_pos = n_pos_all
                    n_neg = n_neg_all
                    sub_probs = probs.view(-1)
                    sub_labels = targets.view(-1)

                # Rank statistic AUC
                combined = torch.stack([sub_probs, sub_labels], dim=1)
                # Sort by probability
                indices = torch.argsort(combined[:, 0])
                sorted_labels = combined[indices, 1]
                # Ranks (1-indexed)
                ranks = torch.arange(
                    1, sorted_labels.numel() + 1, device=device
                ).float()
                pos_rank_sum = (ranks * sorted_labels).sum().item()
                auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

            # 3. Baseline & Loss Analysis
            bias_used = logits.mean().item()
            import torch.nn.functional as F

            pw_tensor = torch.tensor(pos_w.item(), device=device)

            # Actual head loss
            head_loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pw_tensor, reduction="mean"
            ).item()

            # Prior loss (constant prediction at mean logit)
            prior_loss = F.binary_cross_entropy_with_logits(
                torch.full_like(logits, fill_value=bias_used),
                targets,
                pos_weight=pw_tensor,
                reduction="mean",
            ).item()

            delta = head_loss - prior_loss

            # Drift Diagnostics
            # p_target_global = p (from 10 batches) or we can use true_pos_rate
            # bias_init = log(p / (1-p))
            # We want to see how far our current logits/probs moved from that prior.
            bias_prior = math.log(p_prior / (1 - p_prior)) if p_prior > 0 else -10.0
            bias_drift = bias_used - bias_prior

            true_pos_rate = targets.mean().item()

        print(
            f"    {name:20}: p={p:.5f} pos_w={pos_w:.1f} bias={bias_used:.2f} |\n"
            f"      tgt={true_pos_rate:.4f} p_avg={prob_mean:.4f} {thresh_str} |\n"
            f"      pos_p={pos_p:.4f} neg_p={neg_p:.4f} gap_p={gap_p:.4f} auc={auc:.4f} |\n"
            f"      loss={head_loss:.4f} prior={prior_loss:.4f} delta={delta:.4f} drift={bias_drift:.2f}"
        )

    if len(sorted_heads) > 8:
        print("    ...")
    print("  -------------------------------------------\n")


def print_kc_first_batch_summary(
    kc_stats: Dict[str, Any],
    head_diagnostics: List[Dict[str, Any]],
) -> None:
    """Compact summary of the first batch for minimal logging mode."""
    raw_str = ""
    if "raw_logits_mean" in kc_stats:
        raw_str = (
            f"raw μ/σ={kc_stats['raw_logits_mean']:.3f}/{kc_stats['raw_logits_std']:.3f} "
            f"[{kc_stats['raw_logits_min']:.2f}, {kc_stats['raw_logits_max']:.2f}] "
        )

    print(
        f"  FB: KC: {raw_str}nom μ/σ={kc_stats['logits_mean']:.4f}/{kc_stats['logits_std']:.4f} "
        f"probs μ/σ={kc_stats['probs_mean']:.3f}/{kc_stats['probs_std']:.3f} "
        f"(>0.5: {kc_stats.get('probs_gt05', 0.0):.1%} >0.9: {kc_stats.get('probs_gt09', 0.0):.1%}) "
        f"topk μ/min/max={kc_stats.get('topk_mean', 0.0):.2f}/{kc_stats.get('topk_min', 0.0):.2f}/{kc_stats.get('topk_max', 0.0):.2f} "
        f"sparse={kc_stats['sparse_mean']:.4f} nonzeros={kc_stats['nonzero']:.1f} "
        f"uniqKCs={kc_stats['unique_kcs']}"
    )

    for d in head_diagnostics:
        print(
            f"    {d['name']:20}: tgt={d['p']:.4g} w={d['pos_w']:.1f} pred={d['p_avg']:.4g} "
            f"auc={d['auc']:.3f} Δloss={d['delta']:+.4f}"
        )


def print_kc_epoch_compact_summary(
    epoch: int,
    total_epochs: int,
    total_loss: float,
    avg_prob: float,
    act_dens: float,
    struct_avg: float,
    top_losses: List[Any],
    amp_stats: Dict[str, Any],
    entropy_norm: Optional[float] = None,
    avg_kl_to_uniform: Optional[float] = None,
    uniq_kcs: Optional[int] = None,
    avg_p_max: Optional[float] = None,
) -> None:
    """Compact single-line summary of epoch results."""
    top_str = ", ".join([f"{n} {loss:.3f}" for n, loss in top_losses])
    print(
        f"  KC Epoch {epoch}/{total_epochs}: loss={total_loss:.4f} "
        f"prob={avg_prob:.2f} dens={act_dens:.4f} "
        f"struct={struct_avg:.4f} "
        f"{f'ent={entropy_norm:.3f} ' if entropy_norm is not None else ''}"
        f"{f'kl={avg_kl_to_uniform:.3f} ' if avg_kl_to_uniform is not None else ''}"
        f"{f'uniq={uniq_kcs} ' if uniq_kcs is not None else ''}"
        f"{f'pmax={avg_p_max:.3f} ' if avg_p_max is not None else ''}"
        f"top=[{top_str}] | "
        f"AMP scale {amp_stats['start']:.0f}->{amp_stats['end']:.0f} "
        f"skips={amp_stats['skips']} steps={amp_stats['opt_steps']}(+flush={amp_stats['flush_steps']})"
    )


def print_kc_loss_breakdown(parts: Dict[str, float], weights: Dict[str, float]) -> None:
    """Print breakdown of KC loss components."""
    # parts: base, struct, label, div, lb, collapse, sparsity
    # weights: div, lb, sparsity, collapse
    msg = (
        f"  KC LossParts: base={parts.get('base', 0):.4f} "
        f"struct={parts.get('struct', 0):.4f} "
        f"label={parts.get('label', 0):.4f} | "
        f"div={parts.get('div', 0):.4f} "
        f"lb={parts.get('lb', 0):.4f} "
        f"coll={parts.get('collapse', 0):.4f} "
        f"spar={parts.get('sparsity', 0):.4f} "
        f"(W: div={weights.get('div', 0):.2g} "
        f"lb={weights.get('lb', 0):.2g} "
        f"coll={weights.get('collapse', 0):.2g})"
    )
    print(msg)


def print_kc_usage_summary(
    uniq: int,
    total: int,
    max_top1: float,
    tv_mean: float,
    gap_mean: float,
    topk_counts: List[Tuple[int, int]],
    top1_counts: List[Tuple[int, int]],
    k: int,
) -> None:
    """Print compact KC usage stats (histograms)."""

    def fmt_hist(counts: List[Tuple[int, int]], div: int) -> str:
        parts = []
        for idx, count in counts:
            if count == 0:
                continue
            pct = count / max(1, div)
            parts.append(f"{idx}:{count}:{pct:.1%}")
        return ", ".join(parts)

    topk_str = fmt_hist(topk_counts, total * k)
    top1_str = fmt_hist(top1_counts, total)

    print(
        f"    KC Usage: uniqKCs={uniq} total={total} maxTop1={max_top1:.3f} "
        f"topkVals μ={tv_mean:.3f} gapμ={gap_mean:.3f}\n"
        f"      topK(topk): {topk_str}\n"
        f"      topK(top1): {top1_str}"
    )


def print_epoch_summary(
    epoch: int,
    total_epochs: int,
    primary_metrics: Dict[str, float],
    secondary_metrics: Optional[Dict[str, Any]] = None,
    phase: Optional[str] = None,
    kc_epoch_stats: Optional[Dict[str, Any]] = None,
) -> None:
    """Print a formatted summary of the epoch using Rich."""

    title = f"Epoch {epoch}/{total_epochs}"
    if phase:
        icon = {
            "MLM": "📝 ",
            "KC": "🧠 ",
            "Style": "🎨 ",
        }.get(phase, "")
        title = f"{icon}{phase} | {title}"

    # Primary Metrics Table
    p_table = Table(
        show_header=True, header_style="bold magenta", box=None, padding=(0, 2)
    )
    p_table.add_column("Primary Metric")
    p_table.add_column("Value", justify="right")

    for k, v in primary_metrics.items():
        p_table.add_row(k, f"[bold]{v:.4f}[/bold]")

    # Secondary Metrics Table
    s_table = None
    if secondary_metrics:
        is_grouped = any(isinstance(v, dict) for v in secondary_metrics.values())

        if is_grouped:
            s_table = Table(show_header=True, header_style="bold cyan")
            s_table.add_column("Field")
            s_table.add_column("Train Loss", justify="right")
            s_table.add_column("Val Loss", justify="right")
            s_table.add_column("Accuracy", justify="right")

            for group_name, metrics in secondary_metrics.items():
                if isinstance(metrics, dict):
                    t_loss = metrics.get("Train", 0.0)
                    v_loss = metrics.get("Val", 0.0)
                    acc = metrics.get("Acc", 0.0)
                    s_table.add_row(
                        group_name,
                        f"{t_loss:.4f}",
                        f"{v_loss:.4f}",
                        f"[bold green]{acc * 100:.2f}%[/bold green]",
                    )
        else:
            s_table = Table(
                show_header=True, header_style="bold yellow", title="Field Losses"
            )
            items = sorted(
                [(k, v) for k, v in secondary_metrics.items() if isinstance(v, float)],
                key=lambda x: -x[1],
            )

            # Use 2 columns if many items
            if len(items) > 6:
                s_table.add_column("Field")
                s_table.add_column("Loss", justify="right")
                s_table.add_column("Field")
                s_table.add_column("Loss", justify="right")

                num_rows = (len(items) + 1) // 2
                for i in range(num_rows):
                    k1, v1 = items[i]
                    c1 = "red" if v1 > 5.0 else "white"
                    val1 = f"[{c1}]{v1:.4f}[/{c1}]"

                    if i + num_rows < len(items):
                        k2, v2 = items[i + num_rows]
                        c2 = "red" if v2 > 5.0 else "white"
                        val2 = f"[{c2}]{v2:.4f}[/{c2}]"
                        s_table.add_row(k1, val1, k2, val2)
                    else:
                        s_table.add_row(k1, val1, "", "")
            else:
                s_table.add_column("Field")
                s_table.add_column("Loss", justify="right")
                for k, v in items:
                    c = "red" if v > 5.0 else "white"
                    s_table.add_row(k, f"[{c}]{v:.4f}[/{c}]")

    elements: List[Any] = [p_table]
    if s_table:
        elements.append(s_table)

    group = Group(*elements)
    console.print(
        Panel(group, title=f"[bold]{title}[/bold]", expand=False, border_style="blue")
    )

    if phase == "KC" and kc_epoch_stats:
        print("  KC Epoch Summary:")
        print(f"    total_loss={primary_metrics.get('Total Loss', 0.0):.4f}")
        print(
            f"    struct_loss(avg)={kc_epoch_stats.get('avg_struct_loss', 0.0):.4f} over {kc_epoch_stats.get('num_struct_heads_processed', 0)} struct batches"
        )
        print(
            f"    label_loss(avg)={kc_epoch_stats.get('avg_label_loss', 0.0):.4f} over {kc_epoch_stats.get('num_label_heads_processed', 0)} label batches"
        )
        print(f"    sparsity={kc_epoch_stats.get('avg_sparsity', 0.0):.4f}")
