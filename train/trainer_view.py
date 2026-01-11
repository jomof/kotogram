import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

from train.display import print_best_model_saved, print_phase_header
from train.types import EvaluationMetrics, TrainingHistory

# ============================================================================
# Semantic dataclasses for type-safe epoch statistics
# ============================================================================


@dataclass(frozen=True)
class ClassPopulation:
    """Population counts for a binary classification task."""

    class0_count: int
    class1_count: int

    @property
    def total(self) -> int:
        return self.class0_count + self.class1_count


@dataclass(frozen=True)
class BinaryMetric:
    """Metric for a binary classification head with per-class recall."""

    loss: float
    accuracy: float
    class0_accuracy: float  # Negative class recall
    class1_accuracy: float  # Positive class recall
    population: ClassPopulation


@dataclass(frozen=True)
class GrammaticalityMetric:
    """Metric for grammaticality with per-class recall breakdown."""

    loss: float
    accuracy: float
    class0_accuracy: float  # Agrammatical recall
    class1_accuracy: float  # Grammatical recall
    class0_count: int
    class1_count: int


@dataclass(frozen=True)
class MseMetric:
    """Mean squared error metric with sample count."""

    value: float
    sample_count: int


@dataclass(frozen=True)
class RegisterMetric:
    """Metric for multi-label register classification."""

    loss: float
    accuracy: float
    sample_count: int


@dataclass(frozen=True)
class GradientNorms:
    """Average gradient L2 norms per model component."""

    formality: float = 0.0
    gender: float = 0.0
    grammaticality: float = 0.0
    register: float = 0.0
    encoder: float = 0.0
    pooler: float = 0.0


@dataclass(frozen=True)
class StyleEpochStats:
    """Semantic epoch statistics for style training."""

    formality: BinaryMetric
    formality_mse: MseMetric
    gender: BinaryMetric
    gender_mse: MseMetric
    grammaticality: GrammaticalityMetric
    register: RegisterMetric
    total_loss: float
    avg_accuracy: float
    grad_norms: Optional[GradientNorms] = None


class TrainerView(Protocol):
    """Interface for training visualization and logging."""

    def on_train_start(
        self, epochs: int, start_epoch: int, start_batch: int
    ) -> None: ...

    def on_epoch_start(self, epoch: int, total_epochs: int) -> None: ...

    # pylint: disable=too-many-positional-arguments
    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: Tuple[float, float, float, float, float],
        eval_metrics: EvaluationMetrics,
        avg_acc: float,
        is_best: bool,
        patience_counter: int,
    ) -> None:
        _ = train_metrics
        _ = eval_metrics

    def on_train_end(self, history: TrainingHistory) -> None: ...

    def on_progress_init(self, desc: str, total_steps: int) -> None: ...

    def on_progress_update(
        self, batch_idx: int, loss: float, total_steps: int
    ) -> None: ...

    def on_progress_stop(self) -> None: ...

    def on_progress_log(self, message: str) -> None: ...

    def on_checkpoint_saved(
        self, path: str, epoch: int, global_step: int, filename: str
    ) -> None: ...

    def on_checkpoint_restored(
        self, path: str, epoch: int, batch_idx: int, global_step: int
    ) -> None: ...

    def on_best_model_saved(self, model_path: str, best_val_loss: float) -> None: ...

    def on_timing_summary(
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

    # pylint: disable=unused-argument
    def on_lr_adjusted(self, reason: str, new_lrs: List[float]) -> None:
        _ = reason
        _ = new_lrs

    def on_warning(self, message: str) -> None: ...

    def on_early_stopping(self, epoch: int) -> None: ...

    def on_line_flush(self) -> None: ...

    def on_style_epoch_eval_stats(self, epoch: int, stats: StyleEpochStats) -> None: ...

    def on_auto_batch_size(self, batch_size: int, device: Any) -> None: ...


class TrainerDiagnosticsView(TrainerView):
    """Default implementation of TrainerView that does nothing (for now)."""

    def __init__(self) -> None:
        self.last_eval_stats: Dict[str, Dict[str, Any]] = {}

    def on_train_start(self, epochs: int, start_epoch: int, start_batch: int) -> None:
        _ = epochs
        _ = start_epoch
        _ = start_batch

    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        print_phase_header("Style", epoch=epoch + 1, total_epochs=total_epochs)

    # pylint: disable=too-many-positional-arguments
    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: Tuple[float, float, float, float, float],
        eval_metrics: EvaluationMetrics,
        avg_acc: float,
        is_best: bool,
        patience_counter: int,
    ) -> None:
        _ = epoch
        _ = train_metrics
        _ = eval_metrics
        _ = avg_acc
        _ = is_best
        _ = patience_counter

    def on_train_end(self, history: TrainingHistory) -> None:
        _ = history

    def on_progress_init(self, desc: str, total_steps: int) -> None:
        _ = desc
        _ = total_steps

    def on_progress_update(self, batch_idx: int, loss: float, total_steps: int) -> None:
        _ = batch_idx
        _ = loss
        _ = total_steps

    def on_progress_stop(self) -> None:
        pass

    def on_progress_log(self, message: str) -> None:
        _ = message

    def on_checkpoint_saved(
        self, path: str, epoch: int, global_step: int, filename: str
    ) -> None:
        _ = path
        _ = epoch
        _ = global_step
        _ = filename

    def on_checkpoint_restored(
        self, path: str, epoch: int, batch_idx: int, global_step: int
    ) -> None:
        _ = path
        _ = epoch
        _ = batch_idx
        _ = global_step

    def on_best_model_saved(self, model_path: str, best_val_loss: float) -> None:
        print_best_model_saved(model_path, best_val_loss)

    def on_early_stopping(self, epoch: int) -> None:
        print(f"Early stopping at epoch {epoch}")

    def on_timing_summary(
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

    def on_lr_adjusted(self, reason: str, new_lrs: List[float]) -> None:
        _ = reason
        _ = new_lrs

    def on_warning(self, message: str) -> None:
        print(f"[Style Warning] {message}")

    def on_line_flush(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.write("\n")
        sys.stdout.flush()

    def on_auto_batch_size(self, batch_size: int, device: Any) -> None:
        print(
            f"Auto-tuning batch size: Detected device memory on {device}, selected batch size {batch_size}"
        )

    # pylint: disable=too-many-locals
    def on_style_epoch_eval_stats(self, epoch: int, stats: StyleEpochStats) -> None:
        """Print detailed evaluation stats using rich Table for proper alignment."""
        from rich import box
        from rich.table import Table

        from train.display import console

        table = Table(
            show_header=True,
            header_style="bold",
            box=box.HORIZONTALS,  # Horizontal lines under header and between sections
            padding=(0, 1),
            collapse_padding=True,
        )

        # Add columns with right-alignment for numeric values
        table.add_column("Metric", style="", justify="left")
        table.add_column("Loss", justify="right")
        table.add_column("Acc", justify="right")
        table.add_column("Neg%", justify="right")
        table.add_column("Pos%", justify="right")
        table.add_column("Neg n", justify="right")
        table.add_column("Pos n", justify="right")

        # Formality
        self._add_metric_row(
            table,
            "Formality",
            stats.formality.loss,
            stats.formality.accuracy,
            stats.formality.class0_accuracy,
            stats.formality.class1_accuracy,
            stats.formality.population.class0_count,
            stats.formality.population.class1_count,
            "formality",
        )
        self._add_mse_row(table, stats.formality_mse.value, "formality_mse")

        # Gender
        self._add_metric_row(
            table,
            "Gender",
            stats.gender.loss,
            stats.gender.accuracy,
            stats.gender.class0_accuracy,
            stats.gender.class1_accuracy,
            stats.gender.population.class0_count,
            stats.gender.population.class1_count,
            "gender",
        )
        self._add_mse_row(table, stats.gender_mse.value, "gender_mse")

        # Grammaticality
        self._add_metric_row(
            table,
            "Grammaticality",
            stats.grammaticality.loss,
            stats.grammaticality.accuracy,
            stats.grammaticality.class0_accuracy,
            stats.grammaticality.class1_accuracy,
            stats.grammaticality.class0_count,
            stats.grammaticality.class1_count,
            "grammaticality",
        )

        # Register (no per-class breakdown)
        self._add_register_row(
            table,
            "Register",
            stats.register.loss,
            stats.register.accuracy,
            stats.register.sample_count,
            "register",
        )

        # Add section end row
        table.add_section()

        # Total row
        total_key = "total"
        loss_arrow = self._get_loss_arrow(total_key, stats.total_loss)
        acc_arrow = self._get_acc_arrow(total_key, stats.avg_accuracy)
        table.add_row(
            "[bold]Total[/bold]",
            f"{stats.total_loss:.4f}{loss_arrow}",
            f"{stats.avg_accuracy * 100:.1f}%{acc_arrow}",
            "",
            "",
            "",
            "",
        )
        self.last_eval_stats[total_key] = {
            "loss": stats.total_loss,
            "value": stats.avg_accuracy,
        }

        console.print(table)

        # Gradient norms (separate from main table)
        if stats.grad_norms:
            console.print("\n  [bold]Gradient Norms (avg)[/bold]")
            gn = stats.grad_norms
            for name in [
                "formality",
                "gender",
                "grammaticality",
                "register",
                "encoder",
                "pooler",
            ]:
                val = getattr(gn, name, 0.0)
                if val > 0:
                    console.print(f"    [grey62]{name:<20} {val:>8.4f}[/grey62]")

    def _add_metric_row(
        self,
        table: Any,
        label: str,
        loss: float,
        acc: float,
        neg_acc: float,
        pos_acc: float,
        neg_n: int,
        pos_n: int,
        key: str,
    ) -> None:
        """Add a metric row with per-class breakdown to the table."""
        loss_arrow = self._get_loss_arrow(key, loss)
        acc_arrow = self._get_acc_arrow(key, acc)
        neg_arrow = self._get_class_acc_arrow(key, "neg_acc", neg_acc)
        pos_arrow = self._get_class_acc_arrow(key, "pos_acc", pos_acc)

        table.add_row(
            label,
            f"{loss:.4f}{loss_arrow}",
            f"{acc * 100:.1f}%{acc_arrow}",
            f"{neg_acc * 100:.1f}%{neg_arrow}",
            f"{pos_acc * 100:.1f}%{pos_arrow}",
            str(neg_n),
            str(pos_n),
        )
        self.last_eval_stats[key] = {
            "loss": loss,
            "value": acc,
            "neg_acc": neg_acc,
            "pos_acc": pos_acc,
        }

    def _add_mse_row(self, table: Any, value: float, key: str) -> None:
        """Add an MSE row (indented, spanning fewer columns)."""
        val_arrow = self._get_mse_arrow(key, value)
        table.add_row(
            "[grey62]  MSE[/grey62]",
            "",
            f"[grey62]{value:.4f}{val_arrow}[/grey62]",
            "",
            "",
            "",
            "",
        )
        self.last_eval_stats[key] = {"value": value}

    def _add_register_row(
        self, table: Any, label: str, loss: float, acc: float, n: int, key: str
    ) -> None:
        """Add register row (no per-class breakdown, just count)."""
        loss_arrow = self._get_loss_arrow(key, loss)
        acc_arrow = self._get_acc_arrow(key, acc)
        table.add_row(
            label,
            f"{loss:.4f}{loss_arrow}",
            f"{acc * 100:.1f}%{acc_arrow}",
            "",
            "",
            "",
            str(n),
        )
        self.last_eval_stats[key] = {"loss": loss, "value": acc}

    def _get_loss_arrow(self, key: str, loss: float) -> str:
        """Get arrow indicator for loss change."""
        last = self.last_eval_stats.get(key)
        if last and "loss" in last:
            prev = last["loss"]
            if loss < prev:
                return "[green]↓[/green]"
            if loss > prev:
                return "[red]↑[/red]"
        return " "

    def _get_acc_arrow(self, key: str, acc: float) -> str:
        """Get arrow indicator for accuracy change."""
        last = self.last_eval_stats.get(key)
        if last and "value" in last:
            prev = last["value"]
            if acc > prev:
                return "[green]↑[/green]"
            if acc < prev:
                return "[red]↓[/red]"
        return " "

    def _get_mse_arrow(self, key: str, value: float) -> str:
        """Get arrow indicator for MSE change (lower is better)."""
        last = self.last_eval_stats.get(key)
        if last and "value" in last:
            prev = last["value"]
            if value < prev:
                return "[green]↓[/green]"
            if value > prev:
                return "[red]↑[/red]"
        return " "

    def _get_class_acc_arrow(self, key: str, field: str, value: float) -> str:
        """Get arrow indicator for per-class accuracy change (higher is better)."""
        last = self.last_eval_stats.get(key)
        if last and field in last:
            prev = last[field]
            if value > prev:
                return "[green]↑[/green]"
            if value < prev:
                return "[red]↓[/red]"
        return " "


# Explicitly reference unused methods for static analysis tools
# pylint: disable=pointless-statement
TrainerView.on_progress_log
TrainerView.on_lr_adjusted
TrainerView.on_warning
TrainerDiagnosticsView.on_progress_log
TrainerDiagnosticsView.on_lr_adjusted
TrainerDiagnosticsView.on_warning
# Dynamically accessed via getattr() in _build_display_items
GradientNorms.pooler
