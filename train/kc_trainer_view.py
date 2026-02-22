"""Training diagnostics view for KC trainer."""

# pylint: disable=too-many-lines

import math
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Protocol, Tuple

import torch
from rich.table import Table

from train.display import console, format_worst_sample_display, print_phase_header
from train.types import (
    KcDynSizingBinStats,
    KcEpochSummary,
    KCTrainingHistory,
    RunningLossComponents,
    TrainEpochResult,
    WorstSampleInfo,
)


class KCTrainerView(Protocol):
    """Interface for KC training visualization and logging."""

    kc_threshold: float

    def on_kc_train_start(
        self, epochs: int, start_epoch: int, start_batch: int
    ) -> None: ...

    def on_kc_epoch_start(
        self, epoch: int, total_epochs: int, encoder_frozen: bool
    ) -> None:
        _ = encoder_frozen

    def on_kc_epoch_end(self, epoch: int, epoch_result: TrainEpochResult) -> None:
        _ = epoch_result

    def on_kc_train_end(self, history: KCTrainingHistory) -> None: ...

    def on_kc_progress_init(self, desc: str, total_steps: int) -> None: ...

    def on_kc_progress_update(
        self, batch_idx: int, loss: float, total_steps: int
    ) -> None:
        _ = loss
        _ = total_steps

    def on_kc_progress_stop(self) -> None: ...

    def on_kc_bias_init(
        self, name: str, p_mean: float, bias: float, bias_count: int
    ) -> None:
        _ = name
        _ = p_mean
        _ = bias
        _ = bias_count

    def on_kc_warning(self, message: str) -> None: ...

    def on_kc_timing_summary(
        self,
        avg_total_ms: float,
        avg_data_ms: float,
        avg_compute_ms: float,
        data_frac: float,
    ) -> None:
        _ = avg_total_ms
        _ = avg_data_ms
        _ = avg_compute_ms
        _ = data_frac

    def on_line_flush(self) -> None: ...

    def on_style_oversampling_enabled(
        self, formality_boost: float, gender_boost: float
    ) -> None:
        """Called when style oversampling is enabled."""

    # pylint: disable=too-many-positional-arguments
    def on_kc_batch_stats(
        self,
        epoch: int,
        batch_idx: int,
        content_len: torch.Tensor,
        pmax_per_ex: torch.Tensor,
        kc_probs: torch.Tensor,
    ) -> None:
        pass

    def on_kc_epoch_summary(self, epoch: int, summary: KcEpochSummary) -> None:
        pass

    def on_kc_epoch_metrics_skipped(self, epoch: int, total_loss: float) -> None:
        """Called when metrics are skipped for early epochs."""


class KCTrainerDiagnosticsView(KCTrainerView):
    """Default implementation of KCTrainerView that prints diagnostic tables."""

    def __init__(self) -> None:  # pylint: disable=attribute-defined-outside-init
        self.kc_threshold: float = 0.5  # Adaptive threshold (persists across epochs)
        self.reset_epoch_stats()
        # Store previous epoch family stats for trajectory arrows
        self.prev_family_stats: Dict[str, Dict[str, float]] = {}
        self.prev_mse_stats: Dict[str, Dict[str, float]] = {}  # MSE family tracking
        # Store previous epoch loss components for delta arrows
        self.prev_loss_components: Optional[RunningLossComponents] = None
        # Store previous epoch Zipf KL for trajectory arrow
        self.prev_zipf_kl: Optional[float] = None
        # Store previous epoch spill stats for bin trajectory arrows
        self.prev_bin_spill: Dict[str, float] = {}
        self.prev_bin_kth: Dict[str, float] = {}
        self.prev_kc_threshold: Optional[float] = None  # Previous epoch's threshold

    # pylint: disable=attribute-defined-outside-init
    def reset_epoch_stats(self) -> None:
        # Sizing Stats
        self.bin_counts: Dict[str, int] = defaultdict(int)
        self.bin_len_sum: Dict[str, float] = defaultdict(float)
        self.bin_k_sum: Dict[str, float] = defaultdict(float)  # Count of KCs > 0.5
        self.bin_active_count: Dict[str, int] = defaultdict(int)  # Sentences with K≥1
        self.bin_spill_prob_sum: Dict[str, float] = defaultdict(
            float
        )  # (K+1)th prob, K≥1 only
        self.bin_kth_prob_sum: Dict[str, float] = defaultdict(
            float
        )  # Kth prob, K≥1 only
        self.bin_gap_sum: Dict[str, float] = defaultdict(float)  # Kth - Spill, K≥1 only

        # Reservoirs for percentiles
        self.bin_k_reservoirs: Dict[str, List[float]] = defaultdict(list)

        # Activation Stats
        self.pmax_reservoir: List[float] = []

        self.sat99_count: int = 0
        self.total_ex_count: int = 0

        # Dead KC tracking: per-slot boolean arrays (initialized on first batch)
        self.ever_above_half: Optional[torch.Tensor] = None  # [vocab_size] bool
        self.ever_below_half: Optional[torch.Tensor] = None  # [vocab_size] bool

        # Sharpness tracking (alive KCs only)
        self.sharp1_count: int = 0  # prob > 0.9 (committed "on")
        self.sharp0_count: int = 0  # prob < 0.1 (committed "off")
        self.fuzzy_count: int = 0  # prob in [0.2, 0.8]
        self.alive_prob_count: int = 0  # total alive KC-prob evaluations

        # Threshold sweep accumulators (for computing next epoch's threshold)
        self.max_prob_per_kc: Optional[torch.Tensor] = None  # (vocab_size,) running max
        self.threshold_candidates: List[float] = [
            round(0.50 + i * 0.01, 2)
            for i in range(50)  # 0.50..0.99
        ]
        self.threshold_hit_counts: Dict[float, int] = {
            t: 0 for t in self.threshold_candidates
        }
        self.threshold_k_sums: Dict[float, float] = {
            t: 0.0 for t in self.threshold_candidates
        }
        self.threshold_n_sentences: int = 0

    def on_kc_train_start(
        self, epochs: int, start_epoch: int, start_batch: int
    ) -> None:
        _ = epochs
        _ = start_epoch
        _ = start_batch

    def on_kc_epoch_start(
        self, epoch: int, total_epochs: int, encoder_frozen: bool
    ) -> None:
        self.reset_epoch_stats()
        print_phase_header(
            "KC",
            info="Encoder Frozen" if encoder_frozen else "Encoder Thawed",
            epoch=epoch + 1,
            total_epochs=total_epochs,
        )

    def on_kc_epoch_end(self, epoch: int, epoch_result: TrainEpochResult) -> None:
        _ = epoch
        _ = epoch_result

    def on_kc_train_end(self, history: KCTrainingHistory) -> None:
        _ = history

    def on_kc_progress_init(self, desc: str, total_steps: int) -> None:
        _ = desc
        _ = total_steps

    def on_kc_progress_update(
        self, batch_idx: int, loss: float, total_steps: int
    ) -> None:
        # Minimal progress bar logic usually handled by wrapper or simple print
        pass

    def on_kc_progress_stop(self) -> None:
        pass

    def on_kc_bias_init(
        self, name: str, p_mean: float, bias: float, bias_count: int
    ) -> None:
        console.print(
            f"[dim]BiasInit({name}): p_mean={p_mean:.4f} bias={bias:.4f} samples={bias_count}[/dim]"
        )

    def on_kc_warning(self, message: str) -> None:
        console.print(f"[yellow]KC WARNING: {message}[/yellow]")

    def on_kc_timing_summary(
        self,
        avg_total_ms: float,
        avg_data_ms: float,
        avg_compute_ms: float,
        data_frac: float,
    ) -> None:
        console.print(
            f"[dim]Timing: {avg_total_ms:.1f}ms/batch (Data: {avg_data_ms:.1f}ms {data_frac * 100:.1f}%, Compute: {avg_compute_ms:.1f}ms)[/dim]"
        )

    def on_line_flush(self) -> None:
        pass

    def on_style_oversampling_enabled(
        self, formality_boost: float, gender_boost: float
    ) -> None:
        console.print(
            f"[bold yellow]Style oversampling enabled:[/bold yellow] "
            f"formality×{formality_boost:.1f}, gender×{gender_boost:.1f}"
        )

    def _get_bin_label(self, length: int) -> str:
        if length <= 3:
            return "1-3"
        if length <= 7:
            return "4-7"
        if length <= 15:
            return "8-15"
        if length <= 31:
            return "16-31"
        return "32+"

    # pylint: disable=too-many-positional-arguments,too-many-locals
    def on_kc_batch_stats(
        self,
        epoch: int,
        batch_idx: int,
        content_len: torch.Tensor,
        pmax_per_ex: torch.Tensor,
        kc_probs: torch.Tensor,
    ) -> None:
        # Move to CPU for stats
        lens = content_len.cpu().tolist()
        pmax = pmax_per_ex.cpu().tolist()

        batch_size = kc_probs.size(0)

        # Dead KC tracking: update per-slot ever-above/below masks FIRST
        # so we can exclude stuck-on KCs from K/Kth/Spill/Gap
        thresh = self.kc_threshold
        batch_above = (kc_probs > thresh).any(dim=0)  # [vocab_size] on device
        batch_below = (kc_probs <= thresh).any(dim=0)  # [vocab_size] on device
        if self.ever_above_half is None:
            self.ever_above_half = batch_above.cpu()
            self.ever_below_half = batch_below.cpu()
        else:
            assert self.ever_below_half is not None  # Always set with ever_above_half
            self.ever_above_half = self.ever_above_half | batch_above.cpu()
            self.ever_below_half = self.ever_below_half | batch_below.cpu()

        # Mask out dead KCs for K/Kth/Spill/Gap
        # Alive = seen above threshold at least once AND below at least once
        # Dead-0s (stuck off) and dead-1s (stuck on) are excluded.
        alive_mask = (self.ever_below_half & self.ever_above_half).to(kc_probs.device)
        kc_probs_masked = kc_probs * alive_mask.unsqueeze(0)  # [B, vocab_size]

        # Compute K (count of KCs > threshold), Kth, Spill, Gap per example
        # Using masked probs (dead-1 KCs excluded)
        probs_sorted, _ = torch.sort(kc_probs_masked, dim=1, descending=True)
        alive_count = int(alive_mask.sum().item())

        k_counts = (kc_probs_masked > thresh).float().sum(dim=1).cpu().tolist()
        spill_probs = []
        kth_probs = []
        gaps = []
        for i in range(batch_size):
            k = int(k_counts[i])
            # Kth: prob of last KC above threshold (index k-1 in sorted)
            if k >= 1:
                kth = probs_sorted[i, k - 1].item()
            else:
                kth = 0.0
            # Spill: prob of first KC below threshold (index k in sorted)
            if k < alive_count:
                spill = probs_sorted[i, k].item()
            else:
                spill = 0.0
            kth_probs.append(kth)
            spill_probs.append(spill)
            gaps.append(kth - spill)

        # Sharpness tracking on alive KCs
        alive_probs = kc_probs_masked[:, alive_mask.bool()]  # [B, alive_count]
        self.alive_prob_count += alive_probs.numel()
        self.sharp1_count += int((alive_probs > 0.9).sum().item())
        self.sharp0_count += int((alive_probs < 0.1).sum().item())
        self.fuzzy_count += int(
            ((alive_probs >= 0.2) & (alive_probs <= 0.8)).sum().item()
        )

        # Update reservoirs
        self.pmax_reservoir.extend(pmax)
        if len(self.pmax_reservoir) > 50000:
            self.pmax_reservoir = random.sample(self.pmax_reservoir, 50000)

        # Saturation Tracking
        self.total_ex_count += len(pmax)
        for val in pmax:
            if val >= 0.99:
                self.sat99_count += 1

        # Binning
        for i, length in enumerate(lens):
            label = self._get_bin_label(length)

            self.bin_counts[label] += 1
            self.bin_len_sum[label] += length
            k = k_counts[i]
            self.bin_k_sum[label] += k
            # Only accumulate Kth/Spill/Gap for sentences with at least one active KC
            if k >= 1:
                self.bin_active_count[label] += 1
                self.bin_spill_prob_sum[label] += spill_probs[i]
                self.bin_kth_prob_sum[label] += kth_probs[i]
                self.bin_gap_sum[label] += gaps[i]

            # K Reservoir for percentiles
            if len(self.bin_k_reservoirs[label]) < 1000:
                self.bin_k_reservoirs[label].append(k)
            elif random.random() < 0.1:
                idx = random.randint(0, 999)
                self.bin_k_reservoirs[label][idx] = k

        # Threshold sweep: accumulate stats for adaptive threshold computation
        batch_max = kc_probs.max(dim=0).values.cpu()
        if self.max_prob_per_kc is None:
            self.max_prob_per_kc = batch_max
        else:
            self.max_prob_per_kc = torch.max(self.max_prob_per_kc, batch_max)
        self.threshold_n_sentences += batch_size
        for t in self.threshold_candidates:
            k_per_ex = (kc_probs > t).float().sum(dim=1)  # (B,)
            self.threshold_k_sums[t] += float(k_per_ex.sum().item())
            self.threshold_hit_counts[t] += int((k_per_ex > 0).sum().item())

    def on_kc_epoch_metrics_skipped(self, epoch: int, total_loss: float) -> None:
        """Display abbreviated summary when metrics are skipped."""
        console.print(
            f"[dim]KC EP{epoch + 1} (metrics skipped): loss={total_loss:.4f}[/dim]"
        )

    # pylint: disable=too-many-locals,too-many-nested-blocks
    def on_kc_epoch_summary(self, epoch: int, summary: KcEpochSummary) -> None:
        # BLOCK 0: Loss breakdown
        lc = summary.loss_components
        f_state = "Frozen" if summary.frozen else "Thawed"

        # Format metrics
        def _f(v: float) -> str:
            return f"{v:.4f}"

        # Loss breakdown as Table with delta arrows
        act = summary.activation_stats

        def _delta_arrow(curr: float, prev: Optional[float]) -> str:
            """Return colored arrow showing % change from previous epoch."""
            if prev is None or prev == 0.0:
                return ""
            diff = curr - prev
            if abs(diff) < 1e-6:
                return ""
            pct = (diff / prev) * 100
            if pct < 0:
                return f"[green]↓{abs(pct):.1f}%[/green]"
            return f"[red]↑{pct:.1f}%[/red]"

        prev_lc = self.prev_loss_components
        console.print(
            f"[bold]KC EP{summary.epoch_idx + 1}[/bold] {f_state} Loss Breakdown:"
        )
        table_loss = Table(
            show_header=True, header_style="bold", box=None, padding=(0, 1)
        )
        table_loss.add_column("Loss", style="dim", min_width=10)
        table_loss.add_column("Value", justify="right", min_width=10)
        table_loss.add_column("Δ", justify="left", min_width=8)
        table_loss.add_column("Detail", style="dim")

        # Helper for accuracy display with delta arrow
        def _acc(correct: int, total: int, prev_correct: int, prev_total: int) -> str:
            if total == 0:
                return ""
            pct = 100.0 * correct / total
            color = "green" if pct >= 50.0 else "yellow" if pct >= 25.0 else "red"
            base = f"[{color}]{pct:.1f}%[/{color}]"
            # Delta arrow (accuracy going UP is good)
            if prev_total > 0:
                prev_pct = 100.0 * prev_correct / prev_total
                diff = pct - prev_pct
                if abs(diff) >= 0.1:
                    if diff > 0:
                        base += f" [green]↑{diff:.1f}[/green]"
                    else:
                        base += f" [red]↓{abs(diff):.1f}[/red]"
            return base

        # Get weights and divisor from summary
        w = summary.weights
        nb = max(1, summary.n_batches)  # Avoid div-by-zero

        # Add rows with weighted and averaged values (so they sum to epoch loss)
        table_loss.add_row(
            "[cyan]struct[/cyan]",
            _f(lc.struct * w.struct / nb),
            _delta_arrow(lc.struct, prev_lc.struct if prev_lc else None),
            "",
        )
        # Prior KC losses (formality KC0-3, gender KC4-5, register KC6-18) removed
        # - now handled by style classifier
        table_loss.add_row(
            "diversity",
            _f(lc.div * w.div / nb),
            _delta_arrow(lc.div, prev_lc.div if prev_lc else None),
            f"Ent={act.ent_norm:.3f} AvgP={act.kc_probs_mean:.3f}",
        )
        table_loss.add_row(
            "entropy",
            _f(lc.entropy * w.entropy / nb),
            _delta_arrow(lc.entropy, prev_lc.entropy if prev_lc else None),
            f"AvgH={act.avg_entropy:.3f}",
        )
        table_loss.add_row(
            "collapse",
            _f(lc.collapse * w.collapse / nb),
            _delta_arrow(lc.collapse, prev_lc.collapse if prev_lc else None),
            f"Sat95={act.sat_contrib_mean:.2f} PMax={act.pmax_global_max:.3f}",
        )
        table_loss.add_row(
            "sparsity",
            _f(lc.kl_sparse * w.kl_sparse / nb),
            _delta_arrow(lc.kl_sparse, prev_lc.kl_sparse if prev_lc else None),
            f"S1={self.sharp1_count / max(1, self.alive_prob_count):.0%} "
            f"S0={self.sharp0_count / max(1, self.alive_prob_count):.0%} "
            f"Fuzzy={self.fuzzy_count / max(1, self.alive_prob_count):.0%}",
        )
        table_loss.add_row(
            "orthogonality",
            _f(lc.cov_penalty * w.cov_penalty / nb),
            _delta_arrow(lc.cov_penalty, prev_lc.cov_penalty if prev_lc else None),
            f"AvgP={act.kc_probs_mean:.3f}",
        )
        table_loss.add_row(
            "saturation",
            _f(lc.saturation / nb),
            _delta_arrow(lc.saturation, prev_lc.saturation if prev_lc else None),
            f"sc={act.sat_scale_mean:.0f} c={act.sat_contrib_ratio:.0%} pen={act.sat_pen_global:.2f}/{act.sat_pen_pos:.2f} LM={act.pmax_logit_mean_global:.1f}/{act.pmax_logit_mean_pos:.1f} P={act.frac_has_pos:.0%}>{act.frac_over_thr_pos:.0%}",
        )
        zipf_arrow = ""
        if self.prev_zipf_kl is not None:
            zipf_delta = summary.zipf_kl - self.prev_zipf_kl
            if zipf_delta < -1e-6:
                zipf_arrow = "[green]↓[/green]"
            elif zipf_delta > 1e-6:
                zipf_arrow = "[red]↑[/red]"
        zipf_detail = (
            f"Used={summary.kc_logits_used_count} "
            f"({summary.kc_logits_used_percent:.1f}%) "
            f"ZipfKL={_f(summary.zipf_kl)}{zipf_arrow}"
        )
        table_loss.add_row(
            "zipf",
            _f(lc.coverage * w.coverage / nb),
            _delta_arrow(lc.coverage, prev_lc.coverage if prev_lc else None),
            zipf_detail,
        )
        console.print(table_loss)

        # INVARIANT: total_loss = struct + div + entropy + collapse + sparsity + mean_penalty + saturation + coverage
        # All components are on the same scale (per-batch sums), so they add to epoch loss.
        loss_sum = (
            lc.struct * w.struct / nb
            # Prior KC losses (formality, gender, register) removed - handled by style classifier
            + lc.div * w.div / nb
            + lc.entropy * w.entropy / nb
            + lc.collapse * w.collapse / nb
            + lc.kl_sparse * w.kl_sparse / nb
            + lc.cov_penalty * w.cov_penalty / nb
            + lc.saturation / nb  # Already weighted in kc_trainer
            + lc.coverage * w.coverage / nb  # Already weighted in kc_trainer
        )
        expected_loss = summary.total_loss
        tolerance = 1e-3
        if abs(loss_sum - expected_loss) > tolerance:
            raise RuntimeError(
                f"Loss breakdown sum mismatch: sum={loss_sum:.4f} vs "
                f"epoch_loss={expected_loss:.4f} (diff={abs(loss_sum - expected_loss):.4f})"
            )

        # Store for next epoch comparison
        self.prev_loss_components = lc
        self.prev_zipf_kl = summary.zipf_kl

        # BLOCK 1: Dynamic Sizing
        # Compute aggregates from stored bins
        table_sizing = Table(
            show_header=True, header_style="bold magenta", box=None, padding=(0, 1)
        )
        thresh_str = f"{self.kc_threshold:.2f}"
        table_sizing.add_column("Bin")
        table_sizing.add_column("N")
        table_sizing.add_column("Len")
        table_sizing.add_column(f"K@{thresh_str}")
        table_sizing.add_column("Hit%")
        table_sizing.add_column("Kth")
        table_sizing.add_column("Spill")
        table_sizing.add_column("Gap")
        table_sizing.add_column("Canary")

        sorted_labels = ["1-3", "4-7", "8-15", "16-31", "32+"]

        # Populate summary.sizing_stats first (formalizing the aggregation)
        summary.sizing_stats = []
        for label in sorted_labels:
            n = self.bin_counts[label]
            if n == 0:
                continue

            # Percentiles
            k_res = sorted(self.bin_k_reservoirs[label])
            nk = len(k_res)
            kp10 = k_res[nk // 10] if nk else 0.0
            kp50 = k_res[nk // 2] if nk else 0.0
            kp90 = k_res[int(nk * 0.9)] if nk else 0.0

            na = self.bin_active_count[label]  # K≥1 count
            active_pct = na / n if n > 0 else 0.0
            stats = KcDynSizingBinStats(
                bin_label=label,
                count=n,
                len_mean=self.bin_len_sum[label] / n,
                k_mean=self.bin_k_sum[label] / n,
                k_p10=float(kp10),
                k_p50=float(kp50),
                k_p90=float(kp90),
                spill_prob_mean=self.bin_spill_prob_sum[label] / max(1, na),
                kth_prob_mean=self.bin_kth_prob_sum[label] / max(1, na),
                gap_mean=self.bin_gap_sum[label] / max(1, na),
                active_pct=active_pct,
            )
            summary.sizing_stats.append(stats)

        # Render
        for s in summary.sizing_stats:
            # Color for spill: red if high (>0.40), green if low (<0.10), dim otherwise
            c_spill = (
                "red"
                if s.spill_prob_mean > 0.40
                else ("green" if s.spill_prob_mean < 0.10 else "dim")
            )

            # Kth trajectory arrow (higher is better)
            kth_arrow = ""
            prev_kth = self.prev_bin_kth.get(s.bin_label)
            if prev_kth is not None:
                delta = s.kth_prob_mean - prev_kth
                if delta > 0.001:
                    kth_arrow = "[green]↑[/green]"
                elif delta < -0.001:
                    kth_arrow = "[red]↓[/red]"

            # Spill trajectory arrow (lower is better)
            spill_arrow = ""
            prev_spill = self.prev_bin_spill.get(s.bin_label)
            if prev_spill is not None:
                delta = s.spill_prob_mean - prev_spill
                if delta < -0.001:
                    spill_arrow = "[green]↓[/green]"
                elif delta > 0.001:
                    spill_arrow = "[red]↑[/red]"

            # Color for kth: green if high (>0.75), red if low (<0.25)
            c_kth = (
                "green"
                if s.kth_prob_mean > 0.75
                else ("red" if s.kth_prob_mean < 0.25 else "dim")
            )

            # Color for gap: green if wide (>0.3), red if narrow (<0.1)
            c_gap = (
                "green" if s.gap_mean > 0.3 else ("red" if s.gap_mean < 0.1 else "dim")
            )

            table_sizing.add_row(
                s.bin_label,
                str(s.count),
                f"{s.len_mean:.1f}",
                f"{s.k_mean:.1f}|{s.k_p10:.0f}/{s.k_p50:.0f}/{s.k_p90:.0f}",
                f"{s.active_pct:.0%}",
                f"[{c_kth}]{s.kth_prob_mean:.3f}[/{c_kth}]{kth_arrow}",
                f"[{c_spill}]{s.spill_prob_mean:.3f}[/{c_spill}]{spill_arrow}",
                f"[{c_gap}]{s.gap_mean:.3f}[/{c_gap}]",
                summary.canary_texts.get(s.bin_label, ""),
            )

        # Store current spill values for next epoch trajectory arrows
        self.prev_bin_spill = {
            s.bin_label: s.spill_prob_mean for s in summary.sizing_stats
        }
        self.prev_bin_kth = {s.bin_label: s.kth_prob_mean for s in summary.sizing_stats}

        # Total row (weighted averages across all bins)
        all_stats = summary.sizing_stats
        total_n = sum(s.count for s in all_stats)
        total_active = sum(int(s.count * s.active_pct) for s in all_stats)
        if total_n > 0:
            w_len = sum(s.len_mean * s.count for s in all_stats) / total_n
            w_k = sum(s.k_mean * s.count for s in all_stats) / total_n
            w_kth = sum(
                s.kth_prob_mean * s.count * s.active_pct for s in all_stats
            ) / max(1, total_active)
            w_spill = sum(
                s.spill_prob_mean * s.count * s.active_pct for s in all_stats
            ) / max(1, total_active)
            w_gap = sum(s.gap_mean * s.count * s.active_pct for s in all_stats) / max(
                1, total_active
            )
            w_active_pct = total_active / total_n if total_n > 0 else 0.0

            # Merge all K reservoirs for global percentiles
            all_k = []
            for label in sorted_labels:
                all_k.extend(self.bin_k_reservoirs[label])
            all_k.sort()
            nk = len(all_k)
            gp10 = all_k[nk // 10] if nk else 0.0
            gp50 = all_k[nk // 2] if nk else 0.0
            gp90 = all_k[int(nk * 0.9)] if nk else 0.0

            table_sizing.add_row(
                "[bold]Total[/bold]",
                f"[bold]{total_n}[/bold]",
                f"[bold]{w_len:.1f}[/bold]",
                f"[bold]{w_k:.1f}|{gp10:.0f}/{gp50:.0f}/{gp90:.0f}[/bold]",
                f"[bold]{w_active_pct:.0%}[/bold]",
                f"[bold]{w_kth:.3f}[/bold]",
                f"[bold]{w_spill:.3f}[/bold]",
                f"[bold]{w_gap:.3f}[/bold]",
                "",
            )

        console.print(table_sizing)

        # Dead KC summary (below sizing table)
        if self.ever_above_half is not None and self.ever_below_half is not None:
            vocab_size = self.ever_above_half.size(0)
            dead_0 = int((~self.ever_above_half).sum().item())  # Never > 0.5
            dead_1 = int((~self.ever_below_half).sum().item())  # Never <= 0.5
            dead_total = dead_0 + dead_1
            alive = vocab_size - dead_total
            # Color: red if many dead, green if few
            c_dead = (
                "red"
                if dead_total > vocab_size * 0.2
                else ("green" if dead_total < vocab_size * 0.05 else "dim")
            )
            console.print(
                f"  [{c_dead}]Dead KCs: {dead_total}/{vocab_size} "
                f"(0s={dead_0} 1s={dead_1} alive={alive})[/{c_dead}]"
            )

        # Adaptive threshold computation
        if self.max_prob_per_kc is not None and self.threshold_n_sentences > 0:
            n_sent = self.threshold_n_sentences
            vocab_size_t = self.max_prob_per_kc.size(0)
            # Find highest threshold with Hit% >= 99.9%
            new_threshold = self.kc_threshold  # fallback to current
            for t in sorted(self.threshold_candidates, reverse=True):
                hit_pct = self.threshold_hit_counts[t] / n_sent
                if hit_pct >= 0.999:
                    new_threshold = t
                    break
            # Compute stats for the chosen new threshold
            new_alive = int((self.max_prob_per_kc > new_threshold).sum().item())
            new_k_avg = self.threshold_k_sums.get(new_threshold, 0.0) / max(1, n_sent)
            new_hit_pct = self.threshold_hit_counts.get(new_threshold, 0) / max(
                1, n_sent
            )

            # Print threshold line with trajectory arrow
            old_t = self.prev_kc_threshold
            if old_t is not None and abs(new_threshold - old_t) > 0.001:
                arrow = "[green]↑[/green]" if new_threshold > old_t else "[red]↓[/red]"
                t_str = f"{old_t:.2f}→{new_threshold:.2f}{arrow}"
            else:
                t_str = f"{new_threshold:.2f}"
            console.print(
                f"  Threshold: {t_str}"
                f" (alive={new_alive}/{vocab_size_t},"
                f" K_avg={new_k_avg:.1f},"
                f" Hit={new_hit_pct:.1%})"
            )

            # Update for next epoch
            self.prev_kc_threshold = self.kc_threshold
            self.kc_threshold = new_threshold
            summary.kc_threshold = new_threshold

        # BLOCK 2: Activations
        # Compute percentiles from self.pmax_reservoir
        def _res_p(res: List[float], p: float) -> float:
            if not res:
                return 0.0
            res.sort()  # Sort in place is fine here
            return res[int(len(res) * p)]

        pmax_p50 = _res_p(self.pmax_reservoir, 0.5)
        pmax_p90 = _res_p(self.pmax_reservoir, 0.9)
        pmax_p99 = _res_p(self.pmax_reservoir, 0.99)

        act_stats = summary.activation_stats

        # Populate act_stats with reservoirs for correctness
        act_stats.pmax_p50 = pmax_p50
        act_stats.pmax_p90 = pmax_p90
        act_stats.pmax_p99 = pmax_p99

        # BLOCK 3a: MSE Families (Regression Diagnostics)
        if summary.diagnostics.mse_families:
            table_mse = Table(
                show_header=True, header_style="bold magenta", box=None, padding=(0, 1)
            )
            table_mse.add_column("MSE Family")
            table_mse.add_column("Loss")
            table_mse.add_column("Acc")  # Discrete label bucket match
            table_mse.add_column("Corr")  # Pearson correlation
            table_mse.add_column("Bias")  # Mean prediction bias
            table_mse.add_column("σ(Pred)")  # Prediction std
            table_mse.add_column("BΔ")  # Gradient flow

            mse_loss_sum = 0.0
            for name, mse in sorted(summary.diagnostics.mse_families.items()):
                mse_loss_sum += mse.loss_mean

                # Loss arrow: lower is better
                prev_mse = self.prev_mse_stats.get(name, {})
                prev_loss = prev_mse.get("loss", None)
                loss_arrow = ""
                if prev_loss is not None:
                    delta = mse.loss_mean - prev_loss
                    if delta < -0.001:
                        loss_arrow = "[green]↓[/green]"
                    elif delta > 0.001:
                        loss_arrow = "[red]↑[/red]"

                # Acc arrow: higher is better
                prev_acc = prev_mse.get("acc", None)
                acc_arrow = ""
                if prev_acc is not None:
                    delta = mse.discrete_accuracy - prev_acc
                    if delta > 0.01:
                        acc_arrow = "[green]↑[/green]"
                    elif delta < -0.01:
                        acc_arrow = "[red]↓[/red]"

                # Corr arrow: higher is better
                prev_corr = prev_mse.get("corr", None)
                corr_arrow = ""
                if prev_corr is not None and not math.isnan(mse.correlation):
                    delta = mse.correlation - prev_corr
                    if delta > 0.01:
                        corr_arrow = "[green]↑[/green]"
                    elif delta < -0.01:
                        corr_arrow = "[red]↓[/red]"

                # Colors
                c_acc = (
                    "green"
                    if mse.discrete_accuracy > 0.7
                    else ("red" if mse.discrete_accuracy < 0.3 else "dim")
                )
                c_corr = (
                    "green"
                    if mse.correlation > 0.5
                    else ("red" if mse.correlation < 0.1 else "dim")
                )
                c_bias = "red" if abs(mse.mean_bias) > 0.1 else "dim"
                c_std = "red" if mse.pred_std < 0.05 else "dim"  # Low diversity = bad
                c_bdelta = "green" if mse.bias_delta > 0.01 else "dim"

                table_mse.add_row(
                    f"{name}",
                    f"{mse.loss_mean:.4f}{loss_arrow}",
                    f"[{c_acc}]{mse.discrete_accuracy * 100:.1f}%[/{c_acc}]{acc_arrow}",
                    f"[{c_corr}]{mse.correlation:.3f}[/{c_corr}]{corr_arrow}",
                    f"[{c_bias}]{mse.mean_bias:.3f}[/{c_bias}]",
                    f"[{c_std}]{mse.pred_std:.3f}[/{c_std}]",
                    f"[{c_bdelta}]{mse.bias_delta:.3f}[/{c_bdelta}]",
                )

                # Store for next epoch
                self.prev_mse_stats[name] = {
                    "loss": mse.loss_mean,
                    "acc": mse.discrete_accuracy,
                    "corr": mse.correlation,
                }

            console.print(table_mse)

        # BLOCK 3b: Label Families (Classification Diagnostics)
        table_fam = Table(
            show_header=True, header_style="bold blue", box=None, padding=(0, 1)
        )
        table_fam.add_column("Family")
        table_fam.add_column("Loss")
        table_fam.add_column("Pos%")
        table_fam.add_column("PosDen")
        table_fam.add_column("PosP")  # avg probability for positives
        table_fam.add_column("Acc")  # overall accuracy at threshold 0.5
        table_fam.add_column("AvgPos")  # avg positive labels per positive example
        table_fam.add_column("MedPos")  # median positive labels per positive example
        table_fam.add_column("Logit(+/-)")
        table_fam.add_column("Gap")
        table_fam.add_column("Msk%")
        table_fam.add_column("Keys")
        table_fam.add_column("BΔ")  # bias delta for gradient flow check

        # Diagnosis Flag Accumulators
        flag_allneg05_count = 0
        flag_allneg01_count = 0

        num_fams = 0

        # Sort by "most suspicious asleep" first
        # Suspicion = 1.0 if pos_ex_frac < 0.001 (Starvation)
        #             Or small logit gap?
        # Let's sort by pos_ex_frac ascending, then logit gap ascending
        def _sus_score(item: Any) -> Tuple[float, float, str]:
            # item is (name, family_stats)
            stats = item[1]
            # Primary sort: low pos_ex_frac (starvation)
            # Secondary: low logit gap (thresholding)
            gap = stats.logit_pos_mean - stats.logit_neg_mean
            if math.isnan(gap):
                gap = -999.0
            return (stats.pos_ex_frac, gap, item[0])

        fam_items = sorted(summary.diagnostics.families.items(), key=_sus_score)

        # Track stats for trajectory arrows
        current_family_stats: Dict[str, Dict[str, float]] = {}

        for name, fam in fam_items:
            num_fams += 1

            # Checks for Flags
            if fam.recall_05 == 0.0 and fam.fp_rate == 0.0:
                flag_allneg05_count += 1
            if fam.recall_01 < 0.05:
                flag_allneg01_count += 1

            # Wakefulness Diagnostics
            gap = fam.logit_pos_mean - fam.logit_neg_mean
            if math.isnan(gap):
                s_gap = "nan"
            else:
                s_gap = f"{gap:.2f}"

            # Trajectory arrows: compare to previous epoch
            prev = self.prev_family_stats.get(name, {})
            prev_loss = prev.get("loss", None)
            prev_gap = prev.get("gap", None)
            prev_posp = prev.get("posp", None)
            prev_acc = prev.get("acc", None)

            # Loss arrow: lower is better (green ↓, red ↑)
            loss_arrow = ""
            if prev_loss is not None:
                delta = fam.loss_mean - prev_loss
                if delta < -0.001:
                    loss_arrow = "[green]↓[/green]"
                elif delta > 0.001:
                    loss_arrow = "[red]↑[/red]"

            # Gap arrow: higher is better (green ↑, red ↓)
            gap_arrow = ""
            if prev_gap is not None and not math.isnan(gap):
                delta = gap - prev_gap
                if delta > 0.01:
                    gap_arrow = "[green]↑[/green]"
                elif delta < -0.01:
                    gap_arrow = "[red]↓[/red]"

            # PosP arrow: higher is better (green ↑, red ↓)
            posp_arrow = ""
            if prev_posp is not None:
                delta = fam.prob_pos_mean - prev_posp
                if delta > 0.005:  # 0.5% threshold for significance
                    posp_arrow = "[green]↑[/green]"
                elif delta < -0.005:
                    posp_arrow = "[red]↓[/red]"

            # Accuracy arrow: higher is better (green ↑, red ↓)
            acc_arrow = ""
            if prev_acc is not None:
                delta = fam.accuracy - prev_acc
                if delta > 0.002:  # 0.2% threshold for significance
                    acc_arrow = "[green]↑[/green]"
                elif delta < -0.002:
                    acc_arrow = "[red]↓[/red]"

            # Colors
            c_pos = "red" if fam.pos_ex_frac < 0.001 else "green"
            c_gap = "red" if (not math.isnan(gap) and gap < 0.5) else "dim"
            # mask_coverage is "fraction of examples with any valid supervised entry"
            # low coverage is suspicious -> red
            c_msk = "red" if fam.mask_coverage < 0.5 else "dim"

            # Color for PosProb: red if low, green if good
            c_posp = (
                "red"
                if fam.prob_pos_mean < 0.3
                else ("green" if fam.prob_pos_mean > 0.7 else "dim")
            )
            # Compact "Single-Line" Summary style per family row
            # Color for bias delta: green if moving, dim if near-zero
            # bias_delta is now abs sum, so always positive. Use higher threshold.
            c_bdelta = "green" if fam.bias_delta > 0.01 else "dim"

            # Color for accuracy: green if good (>0.9), yellow if ok (>0.7), red if poor
            c_acc = (
                "green"
                if fam.accuracy > 0.9
                else ("yellow" if fam.accuracy > 0.7 else "red")
            )

            # Compute average predicted positive labels per positive example
            avg_pos_str = "-"
            med_pos_str = "-"
            if hasattr(summary, "accumulators") and name in summary.accumulators:
                acc = summary.accumulators[name]
                if acc.n_pos_ex > 0:
                    avg_pos = acc.cnt_pred_pos_on_pos / acc.n_pos_ex
                    # Color based on deviation from target density if helpful,
                    # but for now just display raw predicted density on positive examples.
                    avg_pos_str = f"{avg_pos:.2f}"
                    med_pos = acc.median_pred_pos_on_pos()
                    if med_pos is not None:
                        med_pos_str = f"{med_pos:.0f}"
            elif fam.pos_ex_frac > 1e-6:
                # Fallback to ground truth density if accumulator missing (legacy)
                avg_pos = fam.pos_label_density / fam.pos_ex_frac
                avg_pos_str = f"({avg_pos:.2f})"
                med_pos_str = "-"

            # Display true per-batch loss contribution (no scaling)
            table_fam.add_row(
                f"{name}",
                f"{fam.loss_mean:.4f}{loss_arrow}",
                f"[{c_pos}]{fam.pos_ex_frac * 100:.2f}%[/{c_pos}]",
                f"{fam.pos_label_density:.3f}",
                f"[{c_posp}]{fam.prob_pos_mean * 100:.0f}%[/{c_posp}]{posp_arrow}",
                f"[{c_acc}]{fam.accuracy * 100:.0f}%[/{c_acc}]{acc_arrow}",
                avg_pos_str,
                med_pos_str,
                f"{fam.logit_pos_mean:.1f}/{fam.logit_neg_mean:.1f}",
                f"[{c_gap}]{s_gap}[/{c_gap}]{gap_arrow}",
                f"[{c_msk}]{fam.mask_coverage * 100:.1f}%[/{c_msk}]",
                f"{fam.keys_present}",
                f"[{c_bdelta}]{fam.bias_delta:.3f}[/{c_bdelta}]",
            )

            # Store current stats for next epoch comparison
            current_family_stats[name] = {
                "loss": fam.loss_mean,
                "gap": gap,
                "posp": fam.prob_pos_mean,
                "acc": fam.accuracy,
            }

        # Save for next epoch
        self.prev_family_stats = current_family_stats
        console.print(table_fam)

        # INVARIANT: struct + gap = sum(all family losses)
        # Each family's loss_mean is its per-batch contribution to struct (including per-family gap).
        # This checksum validates that the diagnostic tracking matches the trainer.
        # Skip if no families (minimal test scenarios).
        all_label_families = list(summary.diagnostics.families.values())
        all_mse_families = list(summary.diagnostics.mse_families.values())
        if all_label_families or all_mse_families:
            all_batch_counts = [
                fam.batch_count for fam in all_label_families + all_mse_families
            ]
            do_checksum = not (
                all_batch_counts
                and any(bc != summary.n_batches for bc in all_batch_counts)
            )
            if do_checksum:
                label_loss_sum = sum(fam.loss_mean for fam in all_label_families)
                mse_loss_sum = sum(fam.loss_mean for fam in all_mse_families)
                family_loss_sum = label_loss_sum + mse_loss_sum
                # struct is BCE only (gap regularizer removed)
                struct_loss = lc.struct * w.struct / nb
                tolerance = 1e-3
                if abs(family_loss_sum - struct_loss) > tolerance:
                    raise RuntimeError(
                        f"Family loss sum mismatch: sum={family_loss_sum:.4f} vs "
                        f"struct={struct_loss:.4f} (diff={abs(family_loss_sum - struct_loss):.4f})"
                    )

        # Print Warns if shape mismatch detected?
        # (Not implemented in accumulator yet, relying on table visual for now)

        # BLOCK 4: Label Heads - now integrated into family table as LLoss column

        # BLOCK 4.5: Worst Samples (most problematic samples per family)
        if summary.worst_samples:
            console.print("[bold]Worst Samples (highest loss per family):[/bold]")
            sorted_items: List[Tuple[str, "WorstSampleInfo"]] = []
            for fam_name, tracker in sorted(summary.worst_samples.items()):
                sample = tracker.worst()
                if sample is None:
                    continue
                sorted_items.append((fam_name, sample))

            for fam_name, sample in sorted_items:
                sentence, loss_color = format_worst_sample_display(
                    sample.sentence, sample.loss
                )
                # Build label display - show labels if either is non-empty
                label_info = ""
                # Truncate pred_labels for console display
                display_pred = sample.pred_labels
                if display_pred and display_pred != "none":
                    parts = display_pred.split(",")
                    if len(parts) > 5:
                        display_pred = (
                            ",".join(parts[:5]) + f"...+{len(parts) - 5} more"
                        )
                if sample.target_labels and display_pred:
                    label_info = f" \\[{sample.target_labels}→{display_pred}]"
                elif sample.target_labels:
                    label_info = f" \\[{sample.target_labels}]"
                elif display_pred:
                    label_info = f" \\[?→{display_pred}]"
                console.print(
                    f"  [cyan]{fam_name}[/cyan]: "
                    f"[{loss_color}]loss={sample.loss:.4f}[/{loss_color}] "
                    f"tgt={sample.target:.2f} pred={sample.prediction:.2f}"
                    f"{label_info} "
                    f'[dim]idx={sample.sample_idx} "{sentence}"[/dim]'
                )

            # Export top-50 worst samples per family to TSV files
            tsv_dir = os.path.join(".cache", "training")
            os.makedirs(tsv_dir, exist_ok=True)
            # Remove stale worst-*.tsv files from previous epochs
            import glob

            for old_tsv in glob.glob(os.path.join(tsv_dir, "worst-*.tsv")):
                os.remove(old_tsv)
            family_label_keys = {
                "gender": "# Values: masc=-1.0, neutral=0.0, fem=1.0",
                "formality": (
                    "# Values: v_casual=-1.0, casual=-0.5,"
                    " neutral=0.0, formal=0.5, v_formal=1.0"
                ),
                "grammatic": "# Values: ungrammatical=0.0, grammatical=1.0",
                "gender_class": "# Labels: masc, neutral, fem",
                "formality_class": "# Labels: v_casual, casual, neutral, formal, v_formal",
                "register": (
                    "# Labels: comma-separated register tags"
                    " (e.g. sonkeigo, kenjogo, joseigo, etc.)"
                ),
                "grammar_point": None,
            }
            for fam_name, tracker in summary.worst_samples.items():
                tsv_path = os.path.join(tsv_dir, f"worst-{fam_name}.tsv")
                top_samples = tracker.top_n()
                with open(tsv_path, "w", encoding="utf-8") as f:
                    f.write(f"# Epoch: {summary.epoch_idx + 1}\n")
                    label_key = family_label_keys.get(fam_name)
                    if label_key:
                        f.write(f"{label_key}\n")
                    if fam_name == "grammar_point":
                        # Count GP frequency across top-50 samples
                        label_freq: dict[int, int] = {}
                        pred_freq: dict[int, int] = {}
                        for ws in top_samples:
                            if ws.target_labels and ws.target_labels != "none":
                                for gp_str in ws.target_labels.split(","):
                                    gp_str = gp_str.strip()
                                    if gp_str.startswith("gp"):
                                        gid = int(gp_str[2:])
                                        label_freq[gid] = label_freq.get(gid, 0) + 1
                            if ws.pred_labels and ws.pred_labels != "none":
                                for gp_str in ws.pred_labels.split(","):
                                    gp_str = gp_str.strip()
                                    if gp_str.startswith("gp"):
                                        gid = int(gp_str[2:])
                                        pred_freq[gid] = pred_freq.get(gid, 0) + 1
                        all_gids = set(label_freq) | set(pred_freq)
                        # Write GP listing
                        if all_gids:
                            # Look up GP names and pos/neg counts from corpus.db
                            gp_names: dict[int, str] = {}
                            gp_pos: dict[int, int] = {}
                            gp_neg: dict[int, int] = {}
                            db_path = os.path.join("data", "corpus.db")
                            if os.path.exists(db_path):
                                import sqlite3

                                conn = sqlite3.connect(db_path)
                                for gid in all_gids:
                                    gp_id_str = f"gp{gid:04d}"
                                    row = conn.execute(
                                        "SELECT name, pos_count, neg_count"
                                        " FROM grammar_stats WHERE gp_id = ?",
                                        (gp_id_str,),
                                    ).fetchone()
                                    if row:
                                        gp_names[gid] = f"data/grammar/{row[0]}.yaml"
                                        gp_pos[gid] = row[1]
                                        gp_neg[gid] = row[2]
                                conn.close()
                            # Build entries
                            # (gid, lf, pf, prior_val, prior_str, pos, neg, name)
                            entries: list[
                                tuple[int, int, int, float, str, int, int, str]
                            ] = []
                            for gid in all_gids:
                                lf = label_freq.get(gid, 0)
                                pf = pred_freq.get(gid, 0)
                                prior_val = -1.0
                                prior_str = ""
                                if (
                                    summary.gp_priors is not None
                                    and 0 <= gid < summary.gp_priors.numel()
                                ):
                                    p = float(summary.gp_priors[gid].item())
                                    if math.isfinite(p) and 0.0 <= p <= 1.0:
                                        prior_val = p
                                        prior_str = f"{p * 100.0:.2f}%"
                                entries.append(
                                    (
                                        gid,
                                        lf,
                                        pf,
                                        prior_val,
                                        prior_str,
                                        gp_pos.get(gid, 0),
                                        gp_neg.get(gid, 0),
                                        gp_names.get(gid, ""),
                                    )
                                )
                            # Sort by (lf + pf) desc, then prior desc
                            entries.sort(key=lambda e: (-(e[1] + e[2]), -e[3]))
                            entries = entries[:25]
                            # Column widths
                            lf_w = max(2, max(len(str(e[1])) for e in entries))
                            pf_w = max(2, max(len(str(e[2])) for e in entries))
                            prior_w = max(5, max(len(e[4]) for e in entries))
                            pos_w = max(3, max(len(str(e[5])) for e in entries))
                            neg_w = max(3, max(len(str(e[6])) for e in entries))
                            # Key
                            f.write(
                                "# lf:    label freq (appearances in target labels)\n"
                            )
                            f.write("# pf:    pred freq (appearances in predictions)\n")
                            f.write("# prior: prior probability from gp_priors.bin\n")
                            f.write("# pos:   positive sentence count in corpus\n")
                            f.write("# neg:   negative sentence count in corpus\n")
                            # Header row
                            f.write(
                                f"#   {'ID':<6s}  {'lf':>{lf_w}s}"
                                f"  {'pf':>{pf_w}s}"
                                f"  {'prior':>{prior_w}s}"
                                f"  {'pos':>{pos_w}s}"
                                f"  {'neg':>{neg_w}s}  name\n"
                            )
                            for (
                                gid,
                                lf,
                                pf,
                                _pv,
                                prior_str,
                                pos,
                                neg,
                                name_str,
                            ) in entries:
                                f.write(
                                    f"#   gp{gid:04d}  {lf:>{lf_w}d}"
                                    f"  {pf:>{pf_w}d}"
                                    f"  {prior_str:>{prior_w}s}"
                                    f"  {pos:>{pos_w}d}"
                                    f"  {neg:>{neg_w}d}  {name_str}\n"
                                )
                        # predicted_sample description after table
                        f.write(
                            "# predicted_sample: 5 randomly sampled"
                            " predicted gpXXXX IDs"
                            " (no-prior GPs sampled first)\n"
                        )
                        f.write(
                            "loss\ttarget\tpredicted_count"
                            "\tpredicted_sample\tsentence\n"
                        )
                        for ws in top_samples:
                            target_str = ws.target_labels or f"{ws.target:.4f}"
                            # Sample 5 GPs, prioritizing no-prior
                            sampled_str = "none"
                            if ws.pred_labels and ws.pred_labels != "none":
                                gp_ids = [
                                    int(g.strip()[2:])
                                    for g in ws.pred_labels.split(",")
                                    if g.strip().startswith("gp")
                                ]
                                no_prior = []
                                has_prior = []
                                for gid in gp_ids:
                                    has_p = False
                                    if (
                                        summary.gp_priors is not None
                                        and 0 <= gid < summary.gp_priors.numel()
                                    ):
                                        p = float(summary.gp_priors[gid].item())
                                        if math.isfinite(p) and 0.0 <= p <= 1.0:
                                            has_p = True
                                    if has_p:
                                        has_prior.append(gid)
                                    else:
                                        no_prior.append(gid)
                                random.shuffle(no_prior)
                                random.shuffle(has_prior)
                                picked = no_prior[:5]
                                if len(picked) < 5:
                                    picked += has_prior[: 5 - len(picked)]
                                picked.sort()
                                sampled_str = ",".join(f"gp{g:04d}" for g in picked)
                            f.write(
                                f"{ws.loss:.6f}\t{target_str}"
                                f"\t{int(ws.prediction)}"
                                f"\t{sampled_str}\t{ws.sentence}\n"
                            )
                    else:
                        f.write("loss\ttarget\tpredicted\tsentence\n")
                        for ws in top_samples:
                            target_str = ws.target_labels or f"{ws.target:.4f}"
                            pred_str = ws.pred_labels or f"{ws.prediction:.4f}"
                            f.write(
                                f"{ws.loss:.6f}\t{target_str}"
                                f"\t{pred_str}\t{ws.sentence}\n"
                            )
            console.print(f"  [dim]TSV files written to {tsv_dir}/worst-*.tsv[/dim]")

        # BLOCK 4.6: Top Loss Contributors (for PNU families like grammar_point)
        # Check accumulators for loss_by_label
        for name, acc in summary.accumulators.items():
            if acc.loss_by_label is not None and name.startswith("grammar_point"):
                # Sort by loss descending (values are sums, so higher = more contribution)
                # But wait, loss_by_label accumulates weighted means sum(mean_batch_loss) if I did it right?
                # No, in kc_trainer I did: loss_by_label += ... (sum(dim=0) / count)
                # So it's sum of per-batch average losses.
                # Just like total loss is sum of per-batch average losses.

                # Get indices of top 5
                vals, inds = torch.topk(
                    acc.loss_by_label, min(5, len(acc.loss_by_label))
                )
                vals_list = vals.cpu().tolist()
                inds_list = inds.cpu().tolist()

                parts = []
                for val, idx in zip(vals_list, inds_list):
                    if val < 1e-6:
                        continue
                    # Format as gpXXXX (assuming index matches grammar point ID)
                    parts.append(f"gp{idx:04d}: {val:.4f}")

                if parts:
                    console.print(
                        f"  [cyan]{name}[/cyan] Top Loss: [yellow]{', '.join(parts)}[/yellow]"
                    )
                    if name == "grammar_point" and summary.gp_priors is not None:
                        # Hint: build curate command using GPs whose priors are unset (NaN).
                        no_default = []
                        for idx in inds_list:
                            if 0 <= idx < int(summary.gp_priors.numel()):
                                p = float(summary.gp_priors[idx].item())
                                if not math.isfinite(p):
                                    no_default.append(f"gp{idx:04d}")
                        if no_default:
                            console.print(
                                "    [white dim]add priors with "
                                f"'scripts/curate study {' '.join(no_default)} --full-pos'[/white dim]"
                            )

        # BLOCK 4.7: Prior Drift (for grammar_point family)
        # Show top 5 GPs by |observed_freq - prior| to highlight priors needing update.
        for name, acc in summary.accumulators.items():
            if (
                name == "grammar_point"
                and acc.freq_by_label is not None
                and acc.n_ex > 0
                and summary.gp_priors is not None
            ):
                # Observed frequency = predicted-positive rate per GP
                # (sigmoid(logit) > 0.5 count / total examples through accumulator).
                # Matches the prior's definition: fraction of sentences the model
                # predicts as containing this GP.
                obs_freq = (acc.freq_by_label / acc.n_ex).cpu()

                # Build prior vector, filling NaN/unset with gp_default_prior
                priors = summary.gp_priors.detach().float().cpu()
                n_gp = min(obs_freq.numel(), priors.numel())
                obs_freq = obs_freq[:n_gp]
                priors = priors[:n_gp]
                prior_set = torch.isfinite(priors) & (priors >= 0.0) & (priors <= 1.0)
                priors_filled = priors.clone()
                priors_filled[~prior_set] = summary.gp_default_prior

                # Deviation = |observed - prior|
                deviation = (obs_freq - priors_filled).abs()

                # Top 5
                k = min(5, n_gp)
                vals, inds = torch.topk(deviation, k)
                drift_parts: list[str] = []
                drift_gp_ids: list[str] = []
                for dev_val, idx in zip(vals.tolist(), inds.tolist()):
                    if dev_val < 1e-8:
                        continue
                    obs_pct = obs_freq[idx].item() * 100.0
                    gp_id = f"gp{idx:04d}"
                    if prior_set[idx]:
                        pri_pct = priors[idx].item() * 100.0
                        drift_parts.append(f"{gp_id}: {obs_pct:.2g}%/{pri_pct:.2g}%")
                    else:
                        drift_parts.append(f"{gp_id}: {obs_pct:.2g}%")
                    drift_gp_ids.append(gp_id)

                if drift_parts:
                    console.print(
                        f"  [cyan]{name}[/cyan] Prior Drift: "
                        f"[yellow]{', '.join(drift_parts)}[/yellow]"
                    )
                    if drift_gp_ids:
                        console.print(
                            "    [white dim]add priors with "
                            f"'scripts/curate study {' '.join(drift_gp_ids)} "
                            f"--full-pos'[/white dim]"
                        )

                # Best-fit default prior: mean observed freq of GPs using default
                default_mask = ~prior_set  # GPs with NaN/unset priors
                if default_mask.any():
                    best_fit = obs_freq[default_mask].mean().item()
                    n_default = int(default_mask.sum().item())
                    cur_default = summary.gp_default_prior
                    console.print(
                        f"  [cyan]{name}[/cyan] Best-fit default prior: "
                        f"[yellow]{best_fit:.6f}[/yellow] "
                        f"[dim](current: {cur_default:.2g}, {n_default} GPs)[/dim]"
                    )

        # BLOCK 5: Diagnosis Flags
        flags = []

        # ALLNEG05: >70% families have 0 recall/0 fpr at .5
        if num_fams > 0 and (flag_allneg05_count / num_fams) > 0.7:
            flags.append("ALLNEG05")

        # ALLNEG01: >70% families have <5% recall at .1
        if num_fams > 0 and (flag_allneg01_count / num_fams) > 0.7:
            flags.append("ALLNEG01")

        # SAT: Sat99 > 0.5 or AvgProb > 0.7
        sat99_rate = self.sat99_count / max(1, self.total_ex_count)
        if sat99_rate > 0.5 or act_stats.kc_probs_mean > 0.7:
            flags.append("SAT")

        # COLL: EntNorm low, KL high (relaxed threshold)
        if act_stats.ent_norm < 0.5 and act_stats.kl_u_norm > 0.3:
            flags.append("COLL")

        if flags:
            console.print(f"[bold red]Flags: {' '.join(flags)}[/bold red]")
