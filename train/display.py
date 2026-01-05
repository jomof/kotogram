"""Display logic for training progress reporting."""

import math
from typing import Any, Dict, List, Optional, cast

from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console(force_terminal=True)


class RichTrainerProgressBar:
    """Stateful progress bar for training loops using Rich."""

    def __init__(
        self,
        desc: str,
        total_steps: int,
    ):
        # Use provided console or fall back to global forced-terminal console
        self.console = console
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[status]}"),
            console=self.console,
            transient=False,
        )
        self.task_id = self.progress.add_task(
            desc, total=total_steps, status="Initializing..."
        )
        self.progress.start()

    def update(
        self,
        step: int,
        loss: float,
    ) -> None:
        """Update progress bar state."""
        # Build extra fields
        fields = {}
        if loss is not None:
            fields["status"] = f"loss={loss:.4f}"

        # Cast fields to Any for Mypy safety with typed kwargs in Progress.update
        fields_any = cast(Dict[str, Any], fields)
        self.progress.update(self.task_id, completed=step + 1, **fields_any)

    def log(self, message: str) -> None:
        """Print a message above the progress bar."""
        # Use the progress console to print cleanly above the bar
        self.progress.console.print(message)

    def stop(self) -> None:
        """Stop and remove progress bar."""
        self.progress.stop()


def print_phase_header(
    phase: str,
    epoch: int,
    total_epochs: int,
    info: Optional[str] = None,
) -> None:
    """Print a header for a training phase."""
    icon = {
        "KC": "🧠 ",
        "Style": "🎨 ",
    }.get(phase, "")

    if epoch is not None and total_epochs is not None:
        text = f"{icon}Epoch {epoch}/{total_epochs} Training {phase}"
    else:
        # Fallback / Legacy behavior
        text = (
            f"{icon}{phase} Pretraining"
            if "Pretraining" not in phase
            else f"{icon}{phase}"
        )

    if info:
        text += f" ({info})"
    console.print(f"\n[bold blue]{text}[/bold blue]")


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
    # pylint: disable=too-many-locals, too-many-positional-arguments
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
        if f"kc_targets_{name}" not in batch:
            continue

        _analyze_kc_head_debug(
            name,
            target_logits[name],
            batch[f"kc_targets_{name}"],
            device,
            pos_weight_cap,
            pos_weight_eps,
        )

    if len(sorted_heads) > 8:
        print("    ...")
    print("  -------------------------------------------\n")


def _analyze_kc_head_debug(
    name: str,
    logits: Any,
    targets: Any,
    device: Any,
    pos_weight_cap: float,
    pos_weight_eps: float,
) -> None:
    """Analyze and print debug stats for a single KC head."""
    # pylint: disable=too-many-locals, too-many-positional-arguments
    import torch
    import torch.nn.functional as F

    targets = targets.to(device).float()
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

        # AUC Logic (simplified/inlined or kept if complexity allows)
        # Keeping logic here for now but simplified extraction would be better if still complex.
        # Given this is now in its own function, it should satisfy locals limit for the *outer* function.
        # This function might still have high locals, but let's clear the outer one first.

        auc = float("nan")
        if pos_mask.any() and neg_mask.any():
            # Subsample if too large
            max_samples = 2000
            n_pos_all = pos_mask.sum().item()
            n_neg_all = neg_mask.sum().item()

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

            combined = torch.stack([sub_probs, sub_labels], dim=1)
            indices = torch.argsort(combined[:, 0])
            sorted_labels = combined[indices, 1]
            ranks = torch.arange(1, sorted_labels.numel() + 1, device=device).float()
            pos_rank_sum = (ranks * sorted_labels).sum().item()
            auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

        # 3. Baseline & Loss Analysis
        bias_used = logits.mean().item()

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
        bias_prior = math.log(p_prior / (1 - p_prior)) if p_prior > 0 else -10.0
        bias_drift = bias_used - bias_prior
        true_pos_rate = targets.mean().item()

    print(
        f"    {name:20}: p={p:.5f} pos_w={pos_w:.1f} bias={bias_used:.2f} |\n"
        f"      tgt={true_pos_rate:.4f} p_avg={prob_mean:.4f} {thresh_str} |\n"
        f"      pos_p={pos_p:.4f} neg_p={neg_p:.4f} gap_p={gap_p:.4f} auc={auc:.4f} |\n"
        f"      loss={head_loss:.4f} prior={prior_loss:.4f} delta={delta:.4f} drift={bias_drift:.2f}"
    )


def format_kc_first_batch_summary(
    kc_stats: Dict[str, Any],
    head_diagnostics: List[Dict[str, Any]],
) -> str:
    """Compact summary of the first batch for minimal logging mode."""
    raw_str = ""
    if "raw_logits_mean" in kc_stats:
        raw_str = (
            f"raw μ/σ={kc_stats['raw_logits_mean']:.3f}/{kc_stats['raw_logits_std']:.3f} "
            f"[{kc_stats['raw_logits_min']:.2f}, {kc_stats['raw_logits_max']:.2f}] "
        )

    lines = [
        f"  FB: KC: {raw_str}nom μ/σ={kc_stats['logits_mean']:.4f}/{kc_stats['logits_std']:.4f} "
        f"probs μ/σ={kc_stats['probs_mean']:.3f}/{kc_stats['probs_std']:.3f} "
        f"(>0.5: {kc_stats.get('probs_gt05', 0.0):.1%} >0.9: {kc_stats.get('probs_gt09', 0.0):.1%}) "
        f"topk μ/min/max={kc_stats.get('topk_mean', 0.0):.2f}/{kc_stats.get('topk_min', 0.0):.2f}/{kc_stats.get('topk_max', 0.0):.2f} "
        f"sparse={kc_stats['sparse_mean']:.4f} nonzeros={kc_stats['nonzero']:.1f} "
        f"uniqKCs={kc_stats['unique_kcs']}"
    ]

    for d in head_diagnostics:
        lines.append(
            f"    {d['name']:20}: tgt={d['p']:.4g} w={d['pos_w']:.1f} pred={d['p_avg']:.4g} "
            f"auc={d['auc']:.3f} Δloss={d['delta']:+.4f}"
        )

    return "\n".join(lines)


def print_epoch_summary(
    epoch: int,
    total_epochs: int,
    primary_metrics: Dict[str, float],
    secondary_metrics: Dict[str, Any],
) -> None:
    """Print a formatted summary of the epoch using Rich."""
    # pylint: disable=too-many-locals, too-many-positional-arguments

    title = f"Epoch {epoch} of {total_epochs}"
    icon = "🎨 "
    title = f"{icon}Style | {title}"

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


def print_best_model_saved(path: str, val_loss: float) -> None:
    """Print success message when a new best model is saved.

    Args:
        path: Path where the model was saved.
        val_loss: The validation loss of this best model.
    """
    console.print(
        f"[bold green]New best model matched! Saving to {path} (Loss: {val_loss:.4f})[/bold green]"
    )
