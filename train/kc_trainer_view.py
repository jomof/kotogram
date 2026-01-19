# pylint: disable=duplicate-code
import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Protocol, Tuple

import torch
from rich.table import Table

from train.display import console, print_phase_header
from train.types import (
    KcDynSizingBinStats,
    KcEpochSummary,
    KCTrainingHistory,
    RunningLossComponents,
    TrainEpochResult,
)


class KCTrainerView(Protocol):
    """Interface for KC training visualization and logging."""

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

    def on_auto_batch_size(self, batch_size: int, device: Any) -> None: ...

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
        k_budget_t: torch.Tensor,
        topk_vals: torch.Tensor,
        pmax_per_ex: torch.Tensor,
        topk_sum_per_ex: torch.Tensor,
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
        self.reset_epoch_stats()
        # Store previous epoch family stats for trajectory arrows
        self.prev_family_stats: Dict[str, Dict[str, float]] = {}
        self.prev_mse_stats: Dict[str, Dict[str, float]] = {}  # MSE family tracking
        # Store previous epoch loss components for delta arrows
        self.prev_loss_components: Optional[RunningLossComponents] = None
        # Store previous epoch spill stats for bin trajectory arrows
        self.prev_bin_spill: Dict[str, float] = {}

    # pylint: disable=attribute-defined-outside-init
    def reset_epoch_stats(self) -> None:
        # Sizing Stats
        self.bin_counts: Dict[str, int] = defaultdict(int)
        self.bin_len_sum: Dict[str, float] = defaultdict(float)
        self.bin_k_budget_sum: Dict[str, float] = defaultdict(float)
        self.bin_budget_ratio_sum: Dict[str, float] = defaultdict(float)
        self.bin_masked_tail_sum: Dict[str, float] = defaultdict(float)
        self.bin_keff_sum: Dict[str, float] = defaultdict(float)
        self.bin_keff_minus_budget_sum: Dict[str, float] = defaultdict(float)
        self.bin_spill_prob_sum: Dict[str, float] = defaultdict(float)  # (k+1)th prob

        # Reservoirs for percentiles
        self.bin_k_reservoirs: Dict[str, List[float]] = defaultdict(list)

        # Activation Stats
        self.pmax_reservoir: List[float] = []
        self.topk_sum_reservoir: List[float] = []
        self.sat99_count: int = 0
        self.total_ex_count: int = 0

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

    def on_auto_batch_size(self, batch_size: int, device: Any) -> None:
        console.print(
            f"[bold cyan]Auto-tuning batch size: Detected device memory on {device}, selected batch size {batch_size}[/bold cyan]"
        )

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
        k_budget_t: torch.Tensor,
        topk_vals: torch.Tensor,
        pmax_per_ex: torch.Tensor,
        topk_sum_per_ex: torch.Tensor,
        kc_probs: torch.Tensor,
    ) -> None:
        # Move to CPU for stats
        lens = content_len.cpu().tolist()
        budgets = k_budget_t.cpu().tolist()
        pmax = pmax_per_ex.cpu().tolist()
        topk_sums = topk_sum_per_ex.cpu().tolist()

        # Calculate derived metrics
        # masked_tail_rate: fraction of topk_vals == 0
        # keff: count(topk_vals > 0)
        is_zero = (topk_vals == 0).float()
        masked_rate = is_zero.mean(dim=1).cpu().tolist()
        keff = (topk_vals > 0).float().sum(dim=1).cpu().tolist()

        # Compute spill probability: prob of (k+1)th KC (first outside budget)
        # Sort probs descending and get the (k+1)th value for each example
        probs_sorted, _ = torch.sort(kc_probs, dim=1, descending=True)
        batch_size = kc_probs.size(0)
        vocab_size = kc_probs.size(1)
        spill_probs = []
        for i in range(batch_size):
            k = int(budgets[i])
            if k < vocab_size:
                spill_probs.append(probs_sorted[i, k].item())  # (k+1)th is index k
            else:
                spill_probs.append(0.0)  # Budget exceeds vocab, no spill

        # Update reservoirs
        self.pmax_reservoir.extend(pmax)  # Allow growing large? N=50k max.
        if len(self.pmax_reservoir) > 50000:
            # Basic trim
            self.pmax_reservoir = random.sample(self.pmax_reservoir, 50000)

        self.topk_sum_reservoir.extend(topk_sums)
        if len(self.topk_sum_reservoir) > 50000:
            self.topk_sum_reservoir = random.sample(self.topk_sum_reservoir, 50000)

        # Saturation Tracking
        self.total_ex_count += len(pmax)
        # pmax is already a list of floats
        for val in pmax:
            if val >= 0.99:
                self.sat99_count += 1

        # Binning
        for i, length in enumerate(lens):
            label = self._get_bin_label(length)

            self.bin_counts[label] += 1
            self.bin_len_sum[label] += length
            k = budgets[i]
            self.bin_k_budget_sum[label] += k
            self.bin_budget_ratio_sum[label] += k / max(1, length)
            self.bin_masked_tail_sum[label] += masked_rate[i]
            kf = keff[i]
            self.bin_keff_sum[label] += kf
            self.bin_keff_minus_budget_sum[label] += kf - k
            self.bin_spill_prob_sum[label] += spill_probs[i]

            # K Budget Reservoir
            if len(self.bin_k_reservoirs[label]) < 1000:
                self.bin_k_reservoirs[label].append(k)
            elif random.random() < 0.1:  # simple sub-sampling
                idx = random.randint(0, 999)
                self.bin_k_reservoirs[label][idx] = k

    def on_kc_epoch_metrics_skipped(self, epoch: int, total_loss: float) -> None:
        """Display abbreviated summary when metrics are skipped."""
        console.print(
            f"[dim]KC EP{epoch + 1} (metrics skipped): loss={total_loss:.4f}[/dim]"
        )

    # pylint: disable=too-many-locals
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
        table_loss.add_column("Acc", justify="right", min_width=14)
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
            "",
        )
        # Prior KC losses (formality KC0-3, gender KC4-5, register KC6-18) removed
        # - now handled by style classifier
        table_loss.add_row(
            "diversity",
            _f(lc.div * w.div / nb),
            _delta_arrow(lc.div, prev_lc.div if prev_lc else None),
            "",
            f"Ent={act.ent_norm:.3f} AvgP={act.kc_probs_mean:.3f}",
        )
        table_loss.add_row(
            "load_bal",
            _f(lc.lb * w.lb / nb),
            _delta_arrow(lc.lb, prev_lc.lb if prev_lc else None),
            "",
            f"KL={act.kl_u_norm:.3f}",
        )
        table_loss.add_row(
            "collapse",
            _f(lc.collapse * w.collapse / nb),
            _delta_arrow(lc.collapse, prev_lc.collapse if prev_lc else None),
            "",
            f"Sat95={act.sat_contrib_mean:.2f} PMax={act.pmax_global_max:.3f}",
        )
        table_loss.add_row(
            "sparsity",
            _f(lc.sparsity * w.sparsity / nb),
            _delta_arrow(lc.sparsity, prev_lc.sparsity if prev_lc else None),
            "",
            f"Dens={act.act_dens_mean:.3f} K={act.topk_sum_p50:.0f}/{act.topk_sum_p90:.0f}/{act.topk_sum_p99:.0f}",
        )
        table_loss.add_row(
            "saturation",
            _f(lc.saturation / nb),
            _delta_arrow(lc.saturation, prev_lc.saturation if prev_lc else None),
            "",
            f"sc={act.sat_scale_mean:.0f} c={act.sat_contrib_ratio:.0%} pen={act.sat_pen_global:.2f}/{act.sat_pen_pos:.2f} LM={act.pmax_logit_mean_global:.1f}/{act.pmax_logit_mean_pos:.1f} P={act.frac_has_pos:.0%}>{act.frac_over_thr_pos:.0%}",
        )
        table_loss.add_row(
            "coverage",
            _f(lc.coverage * w.coverage / nb),
            _delta_arrow(lc.coverage, prev_lc.coverage if prev_lc else None),
            "",
            f"Used={summary.kc_logits_used_count} ({summary.kc_logits_used_percent:.1f}%)",
        )
        console.print(table_loss)

        # INVARIANT: total_loss = struct + div + lb + collapse + sparsity + saturation + coverage
        # All components are on the same scale (per-batch sums), so they add to epoch loss.
        loss_sum = (
            lc.struct * w.struct / nb
            # Prior KC losses (formality, gender, register) removed - handled by style classifier
            + lc.div * w.div / nb
            + lc.lb * w.lb / nb
            + lc.collapse * w.collapse / nb
            + lc.sparsity * w.sparsity / nb
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

        # BLOCK 1: Dynamic Sizing
        # Compute aggregates from stored bins
        table_sizing = Table(
            show_header=True, header_style="bold magenta", box=None, padding=(0, 1)
        )
        table_sizing.add_column("Bin")
        table_sizing.add_column("N")
        table_sizing.add_column("Len")
        table_sizing.add_column("K(Avg|P10/50/90)")
        table_sizing.add_column("K/Len")
        table_sizing.add_column("TailMask")
        table_sizing.add_column("Keff")
        table_sizing.add_column("Diff")
        table_sizing.add_column("Spill")  # Prob of (k+1)th KC

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

            stats = KcDynSizingBinStats(
                bin_label=label,
                count=n,
                len_mean=self.bin_len_sum[label] / n,
                k_budget_mean=self.bin_k_budget_sum[label] / n,
                k_budget_p10=float(kp10),
                k_budget_p50=float(kp50),
                k_budget_p90=float(kp90),
                budget_ratio_mean=self.bin_budget_ratio_sum[label] / n,
                masked_tail_rate=self.bin_masked_tail_sum[label] / n,
                keff_mean=self.bin_keff_sum[label] / n,
                keff_minus_budget_mean=self.bin_keff_minus_budget_sum[label] / n,
                spill_prob_mean=self.bin_spill_prob_sum[label] / n,
            )
            summary.sizing_stats.append(stats)

        # Render
        for s in summary.sizing_stats:
            # Colors
            c_mask = (
                "red"
                if s.masked_tail_rate < 0.05 and s.k_budget_mean < 9.9
                else "green"
            )
            c_diff = "red" if abs(s.keff_minus_budget_mean) > 0.2 else "dim"

            # Color for spill: red if high (>0.75), green if low (<0.25), dim otherwise
            c_spill = (
                "red"
                if s.spill_prob_mean > 0.75
                else ("green" if s.spill_prob_mean < 0.25 else "dim")
            )

            # Spill trajectory arrow (lower is better)
            spill_arrow = ""
            prev_spill = self.prev_bin_spill.get(s.bin_label)
            if prev_spill is not None:
                delta = s.spill_prob_mean - prev_spill
                if delta < -0.01:
                    spill_arrow = "[green]↓[/green]"
                elif delta > 0.01:
                    spill_arrow = "[red]↑[/red]"

            table_sizing.add_row(
                s.bin_label,
                str(s.count),
                f"{s.len_mean:.1f}",
                f"{s.k_budget_mean:.1f}|{s.k_budget_p10:.0f}/{s.k_budget_p50:.0f}/{s.k_budget_p90:.0f}",
                f"{s.budget_ratio_mean:.2f}",
                f"[{c_mask}]{s.masked_tail_rate:.3f}[/{c_mask}]",
                f"{s.keff_mean:.1f}",
                f"[{c_diff}]{s.keff_minus_budget_mean:.2f}[/{c_diff}]",
                f"[{c_spill}]{s.spill_prob_mean:.3f}[/{c_spill}]{spill_arrow}",
            )

        # Store current spill values for next epoch trajectory arrows
        self.prev_bin_spill = {
            s.bin_label: s.spill_prob_mean for s in summary.sizing_stats
        }

        console.print(table_sizing)

        # SPARSE flag detection: high spill across most bins indicates sparsity penalty too low
        # Per-bin ↑K? indicator: if specific bins have high spill but SPARSE isn't triggered
        high_spill_bins = [s for s in summary.sizing_stats if s.spill_prob_mean > 0.2]
        total_bins = len(summary.sizing_stats)
        sparse_triggered = total_bins > 0 and len(high_spill_bins) / total_bins >= 0.7

        # If SPARSE not triggered but some bins have high spill, those bins may need more K
        if not sparse_triggered and high_spill_bins:
            bin_labels = [s.bin_label for s in high_spill_bins]
            console.print(
                f"[yellow]↑K? Bins [{', '.join(bin_labels)}] may need more k_budget "
                f"(Spill>{0.2:.1f})[/yellow]"
            )

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

        topk_p50 = _res_p(self.topk_sum_reservoir, 0.5)
        topk_p90 = _res_p(self.topk_sum_reservoir, 0.9)
        topk_p99 = _res_p(self.topk_sum_reservoir, 0.99)

        act_stats = summary.activation_stats

        # Populate act_stats with reservoirs for correctness
        act_stats.pmax_p50 = pmax_p50
        act_stats.pmax_p90 = pmax_p90
        act_stats.pmax_p99 = pmax_p99
        act_stats.topk_sum_p50 = topk_p50
        act_stats.topk_sum_p90 = topk_p90
        act_stats.topk_sum_p99 = topk_p99

        # BLOCK 3a: MSE Families (Regression Diagnostics)
        if summary.diagnostics.mse_families:
            table_mse = Table(
                show_header=True, header_style="bold magenta", box=None, padding=(0, 1)
            )
            table_mse.add_column("MSE Family")
            table_mse.add_column("Loss")
            table_mse.add_column("Acc±")  # Within ±0.1
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
                    delta = mse.accuracy_01 - prev_acc
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
                    if mse.accuracy_01 > 0.7
                    else ("red" if mse.accuracy_01 < 0.3 else "dim")
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
                    name,
                    f"{mse.loss_mean:.4f}{loss_arrow}",
                    f"[{c_acc}]{mse.accuracy_01 * 100:.1f}%[/{c_acc}]{acc_arrow}",
                    f"[{c_corr}]{mse.correlation:.3f}[/{c_corr}]{corr_arrow}",
                    f"[{c_bias}]{mse.mean_bias:.3f}[/{c_bias}]",
                    f"[{c_std}]{mse.pred_std:.3f}[/{c_std}]",
                    f"[{c_bdelta}]{mse.bias_delta:.3f}[/{c_bdelta}]",
                )

                # Store for next epoch
                self.prev_mse_stats[name] = {
                    "loss": mse.loss_mean,
                    "acc": mse.accuracy_01,
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
        table_fam.add_column("UnlabFP")  # unlabeled predicted positive rate
        table_fam.add_column("Logit(+/-)")
        table_fam.add_column("Gap")
        table_fam.add_column("Msk%")
        table_fam.add_column("Keys")
        table_fam.add_column("BΔ")  # bias delta for gradient flow check

        # Diagnosis Flag Accumulators
        flag_allneg05_count = 0
        flag_allneg01_count = 0
        flag_mask_triggered = False
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
            # If less than half the examples have any valid supervision, flag it.
            if fam.mask_coverage < 0.5:
                flag_mask_triggered = True

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
                if delta < -0.01:
                    loss_arrow = "[green]↓[/green]"
                elif delta > 0.01:
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
                if delta > 0.05:  # 5% threshold for significance
                    posp_arrow = "[green]↑[/green]"
                elif delta < -0.05:
                    posp_arrow = "[red]↓[/red]"

            # Accuracy arrow: higher is better (green ↑, red ↓)
            acc_arrow = ""
            if prev_acc is not None:
                delta = fam.accuracy - prev_acc
                if delta > 0.02:  # 2% threshold for significance
                    acc_arrow = "[green]↑[/green]"
                elif delta < -0.02:
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

            # Compute unlabeled FP rate from accumulator stats
            unlab_fp_rate_str = "-"
            if hasattr(summary, "accumulators") and name in summary.accumulators:
                acc = summary.accumulators[name]
                if acc.cnt_unlabeled > 0:
                    unlab_fp_rate = acc.cnt_unlabeled_pred_pos / acc.cnt_unlabeled
                    # Color: green if low (<10%), yellow if medium, red if high (>30%)
                    c_unlab = (
                        "green"
                        if unlab_fp_rate < 0.1
                        else ("yellow" if unlab_fp_rate < 0.3 else "red")
                    )
                    unlab_fp_rate_str = (
                        f"[{c_unlab}]{unlab_fp_rate * 100:.1f}%[/{c_unlab}]"
                    )

            # Display true per-batch loss contribution (no scaling)
            table_fam.add_row(
                name,
                f"{fam.loss_mean:.4f}{loss_arrow}",
                f"[{c_pos}]{fam.pos_ex_frac * 100:.2f}%[/{c_pos}]",
                f"{fam.pos_label_density:.3f}",
                f"[{c_posp}]{fam.prob_pos_mean * 100:.0f}%[/{c_posp}]{posp_arrow}",
                f"[{c_acc}]{fam.accuracy * 100:.0f}%[/{c_acc}]{acc_arrow}",
                unlab_fp_rate_str,
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
            for fam_name, sample in sorted(summary.worst_samples.items()):
                # Truncate long sentences for display
                sentence = sample.sentence
                if len(sentence) > 60:
                    sentence = sentence[:57] + "..."
                # Color-code based on loss magnitude
                loss_color = (
                    "red"
                    if sample.loss > 1.0
                    else ("yellow" if sample.loss > 0.25 else "dim")
                )
                # Build label display - show labels if either is non-empty
                label_info = ""
                if sample.target_labels and sample.pred_labels:
                    label_info = f" \\[{sample.target_labels}→{sample.pred_labels}]"
                elif sample.target_labels:
                    label_info = f" \\[{sample.target_labels}]"
                elif sample.pred_labels:
                    label_info = f" \\[?→{sample.pred_labels}]"
                console.print(
                    f"  [cyan]{fam_name}[/cyan]: "
                    f"[{loss_color}]loss={sample.loss:.4f}[/{loss_color}] "
                    f"tgt={sample.target:.2f} pred={sample.prediction:.2f}"
                    f"{label_info} "
                    f'[dim]idx={sample.sample_idx} "{sentence}"[/dim]'
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

        # UNDERK: Long bins (16+) have K/Len < 0.25 (or diff > 1.0)
        # Check sizing stats
        underk = False
        for s in summary.sizing_stats:
            if s.bin_label in ("16-31", "32+"):
                # budget_ratio_mean is avg(k/len)
                if s.budget_ratio_mean < 0.25:
                    underk = True
        if underk:
            flags.append("UNDERK")

        # MASK: Any family > 10% mask hit
        if flag_mask_triggered:
            flags.append("MASK")

        # COLL: EntNorm low, KL high (relaxed threshold)
        if act_stats.ent_norm < 0.5 and act_stats.kl_u_norm > 0.3:
            flags.append("COLL")

        # SPARSE: High spill across most bins - sparsity penalty too low
        if sparse_triggered:
            flags.append("SPARSE")

        if flags:
            console.print(f"[bold red]Flags: {' '.join(flags)}[/bold red]")


# Explicitly reference unused methods for static analysis tools
# pylint: disable=pointless-statement
