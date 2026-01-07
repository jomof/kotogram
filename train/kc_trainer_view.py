# pylint: disable=duplicate-code
import math
import random
from collections import defaultdict
from typing import Dict, List, Protocol

import torch
from rich.table import Table

from train.display import console, print_phase_header
from train.types import (
    KcDynSizingBinStats,
    KcEpochSummary,
    KCTrainingHistory,
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

    def on_kc_checkpoint_restored(
        self, path: str, epoch: int, batch_idx: int, global_step: int
    ) -> None: ...

    def on_kc_checkpoint_saved(
        self, path: str, epoch: int, global_step: int, filename: str
    ) -> None: ...

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

    # New Hooks
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
    ) -> None:
        pass

    def on_kc_epoch_summary(self, epoch: int, summary: KcEpochSummary) -> None:
        pass


class KCTrainerDiagnosticsView(KCTrainerView):
    """Default implementation of KCTrainerView that prints diagnostic tables."""

    def __init__(self) -> None:  # pylint: disable=attribute-defined-outside-init
        self.reset_epoch_stats()

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

        # Reservoirs for percentiles
        self.bin_k_reservoirs: Dict[str, List[float]] = defaultdict(list)

        # Activation Stats
        self.pmax_reservoir: List[float] = []
        self.topk_sum_reservoir: List[float] = []
        self.sat95_count: int = 0
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

    def on_kc_checkpoint_restored(
        self, path: str, epoch: int, batch_idx: int, global_step: int
    ) -> None:
        _ = path
        _ = epoch
        _ = batch_idx
        _ = global_step

    def on_kc_checkpoint_saved(
        self, path: str, epoch: int, global_step: int, filename: str
    ) -> None:
        _ = path
        _ = epoch
        _ = global_step
        _ = filename

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
            if val >= 0.95:
                self.sat95_count += 1
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

            # K Budget Reservoir
            if len(self.bin_k_reservoirs[label]) < 1000:
                self.bin_k_reservoirs[label].append(k)
            elif random.random() < 0.1:  # simple sub-sampling
                idx = random.randint(0, 999)
                self.bin_k_reservoirs[label][idx] = k

    # pylint: disable=too-many-locals
    def on_kc_epoch_summary(self, epoch: int, summary: KcEpochSummary) -> None:
        # BLOCK 0: Header
        lc = summary.loss_components
        f_state = "Frozen" if summary.frozen else "Thawed"

        # Format metrics
        def _f(v: float) -> str:
            return f"{v:.4f}"

        header = (
            f"[bold]KC EP{summary.epoch_idx + 1}[/bold] {f_state} | "
            f"Loss: [cyan]{_f(lc.base)}[/cyan]+[magenta]{_f(lc.struct)}[/magenta]+"
            f"[blue]{_f(lc.label)}[/blue]+div{_f(lc.div)}+lb{_f(lc.lb)}+"
            f"col{_f(lc.collapse)}+sp{_f(lc.sparsity)} | "
            f"dStep={summary.global_step_delta} Batch={summary.n_batches}/{summary.total_batches}"
        )
        console.print(header)

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

            table_sizing.add_row(
                s.bin_label,
                str(s.count),
                f"{s.len_mean:.1f}",
                f"{s.k_budget_mean:.1f}|{s.k_budget_p10:.0f}/{s.k_budget_p50:.0f}/{s.k_budget_p90:.0f}",
                f"{s.budget_ratio_mean:.2f}",
                f"[{c_mask}]{s.masked_tail_rate:.3f}[/{c_mask}]",
                f"{s.keff_mean:.1f}",
                f"[{c_diff}]{s.keff_minus_budget_mean:.2f}[/{c_diff}]",
            )
        console.print(table_sizing)

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

        # Saturation rates
        n_ex = max(1, self.total_ex_count)
        sat95_rate = self.sat95_count / n_ex
        sat99_rate = self.sat99_count / n_ex

        act_stats = summary.activation_stats

        # Populate act_stats with reservoirs for correctness
        act_stats.pmax_p50 = pmax_p50
        act_stats.pmax_p90 = pmax_p90
        act_stats.pmax_p99 = pmax_p99
        act_stats.topk_sum_p50 = topk_p50
        act_stats.topk_sum_p90 = topk_p90
        act_stats.topk_sum_p99 = topk_p99

        # Outlier Logic
        c_pmax = (
            "red" if act_stats.pmax_global_max > 0.995 or pmax_p99 > 0.99 else "green"
        )
        c_ent = "red" if act_stats.ent_norm < 0.2 else "cyan"  # heuristic
        c_sat = "red" if sat99_rate > 0.5 else "dim"

        console.print(
            f"Act: AvgProb={act_stats.kc_probs_mean:.4f} Dens={act_stats.act_dens_mean:.3f} | "
            f"PMax: [{c_pmax}]Glb={act_stats.pmax_global_max:.3f} P99={act_stats.pmax_p99:.3f}[/{c_pmax}] "
            f"P50={act_stats.pmax_p50:.3f} | "
            f"Sat95={sat95_rate:.2f} [{c_sat}]Sat99={sat99_rate:.2f}[/{c_sat}] | "
            f"Sat: w={act_stats.sat_w:.2f} alpha={act_stats.sat_alpha:.3f} scaleμ={act_stats.sat_scale_mean:.1f} contribμ={act_stats.sat_contrib_mean:.2f} contrib/prim={act_stats.sat_contrib_ratio:.1%} | "
            f"Pos({act_stats.frac_has_pos:.0%}): pen={act_stats.sat_pen_pos:.2f} LM={act_stats.pmax_logit_mean_pos:.2f} >Thr={act_stats.frac_over_thr_pos:.1%} | "
            f"Glb: pen={act_stats.sat_pen_global:.2f} LM={act_stats.pmax_logit_mean_global:.1f} "
            f"SumK: P50={act_stats.topk_sum_p50:.1f} P90={act_stats.topk_sum_p90:.1f} P99={act_stats.topk_sum_p99:.1f} | "
            f"Ent: [{c_ent}]{act_stats.ent_norm:.3f}[/{c_ent}] KL={act_stats.kl_u_norm:.3f}"
        )

        # BLOCK 3: Families
        # Pivot: High density separation metrics
        table_fam = Table(
            show_header=True, header_style="bold blue", box=None, padding=(0, 1)
        )
        table_fam.add_column("Family")
        table_fam.add_column("Loss")
        table_fam.add_column("Pos%")
        table_fam.add_column("P(+/-)")
        table_fam.add_column("ΔP")
        table_fam.add_column("Logit(+/-)")
        table_fam.add_column("R@.1/.5")
        table_fam.add_column("FPR@.5")
        table_fam.add_column("Msk%")
        table_fam.add_column("Supp")

        # Diagnosis Flag Accumulators
        flag_allneg05_count = 0
        flag_allneg01_count = 0
        flag_mask_triggered = False
        num_fams = 0

        # Sort by name
        fam_names = sorted(summary.diagnostics.families.keys())
        for name in fam_names:
            fam = summary.diagnostics.families[name]
            num_fams += 1

            # Checks for Flags
            if fam.recall_05 == 0.0 and fam.fp_rate == 0.0:
                flag_allneg05_count += 1
            if fam.recall_01 < 0.05:
                flag_allneg01_count += 1
            if fam.mask_pct > 0.10:
                flag_mask_triggered = True

            # Colors
            # Delta P
            c_dp = "red" if fam.delta_p < 0.02 else "green"

            # Recall@0.1
            c_r01 = "red" if fam.recall_01 < 0.05 else "green"

            # Mask
            c_msk = "red" if fam.mask_pct > 0.1 else "dim"

            table_fam.add_row(
                name,
                f"{fam.loss_mean:.4f}",
                f"{fam.rate * 100:.1f}%",
                f"{fam.prob_pos_mean:.2f}/{fam.prob_neg_mean:.2f}",
                f"[{c_dp}]{fam.delta_p:.2f}[/{c_dp}]",
                f"{fam.logit_pos_mean:.1f}/{fam.logit_neg_mean:.1f}",
                f"[{c_r01}]{fam.recall_01:.2f}[/{c_r01}]/{fam.recall_05:.2f}",
                f"{fam.fp_rate:.2f}",
                f"[{c_msk}]{fam.mask_pct * 100:.1f}%[/{c_msk}]",
                f"{fam.support:.2f}",
            )
        console.print(table_fam)

        # BLOCK 4: Label Heads
        # One line
        items = []
        for k, v in summary.label_losses.items():
            c = "red" if not math.isfinite(v) or v == 0.0 else "white"
            items.append(f"{k}=[{c}]{v:.4f}[/{c}]")
        if items:
            console.print("Labels: " + " ".join(items))

        # BLOCK 5: Diagnosis Flags
        flags = []

        # ALLNEG05: >70% families have 0 recall/0 fpr at .5
        if num_fams > 0 and (flag_allneg05_count / num_fams) > 0.7:
            flags.append("ALLNEG05")

        # ALLNEG01: >70% families have <5% recall at .1
        if num_fams > 0 and (flag_allneg01_count / num_fams) > 0.7:
            flags.append("ALLNEG01")

        # SAT: Sat99 > 0.5 or AvgProb > 0.7
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

        # COLL: EntNorm low, KL high
        if act_stats.ent_norm < 0.6 and act_stats.kl_u_norm > 0.3:
            flags.append("COLL")

        if flags:
            console.print(f"[bold red]Flags: {' '.join(flags)}[/bold red]")


# Explicitly reference unused methods for static analysis tools
# pylint: disable=pointless-statement
KCTrainerView.on_kc_checkpoint_saved
KCTrainerDiagnosticsView.on_kc_checkpoint_saved
