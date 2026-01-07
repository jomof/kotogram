import sys
from typing import Any, Dict, List, Protocol, Tuple

from train.display import print_best_model_saved, print_phase_header
from train.types import EvaluationMetrics, TrainingHistory


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

    def on_style_epoch_eval_stats(
        self, epoch: int, stats: List[Dict[str, Any]]
    ) -> None: ...

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
    def on_style_epoch_eval_stats(
        self, epoch: int, stats: List[Dict[str, Any]]
    ) -> None:
        """Print detailed evaluation stats with formatting."""
        from train.display import console

        for item in stats:
            label = item["label"]
            loss = item.get("loss")
            value = item["value"]
            is_mse = item.get("is_mse", False)
            is_percent = item.get("is_percent", False)
            is_total = item.get("is_total", False)

            if is_total:
                # Add separator
                # Width calculation: 45 + 1 + 10 + 1 + 1 + 1 + 10 + 1 + 1 = 71
                console.print(f"  {'-' * 71}")

            # Indentation
            # Base indent 2. MSE adds 2 more to the label only.
            indent = "    " if is_mse else "  "

            # Format label
            # We want columns to align.
            # Total label width reserved: 45 chars?
            # Existing was 40.
            # "  Label..." (40 chars total) -> Loss (8) -> Value (8)

            label_display = f"{indent}{label}"

            # Format Loss
            loss_str = f"{loss:.4f}" if loss is not None else ""

            # Format Value
            if is_percent:
                val_str = f"{value * 100.0:.1f}%"
            else:
                val_str = f"{value:.3f}"

            # Determine arrows
            loss_arrow = " "
            val_arrow = " "
            last_item = self.last_eval_stats.get(label)

            if last_item:
                # Loss Arrow
                if loss is not None:
                    prev_loss = last_item.get("loss")
                    if prev_loss is not None:
                        if loss < prev_loss:
                            loss_arrow = "[green]↓[/green]"
                        elif loss > prev_loss:
                            loss_arrow = "[red]↑[/red]"

                # Value Arrow
                prev_val = last_item["value"]
                # For MSE (is_mse=True), lower is better.
                # For Accuracy (is_percent=True), higher is better.
                if is_mse:
                    if value < prev_val:
                        val_arrow = "[green]↓[/green]"
                    elif value > prev_val:
                        val_arrow = "[red]↑[/red]"
                else:
                    if value > prev_val:
                        val_arrow = "[green]↑[/green]"
                    elif value < prev_val:
                        val_arrow = "[red]↓[/red]"

            # Construct line
            # Label: <left 45> Loss: <right 10> Arr: <1> Val: <right 10> Arr: <1>
            # Separator length: 45 + 1 + 10 + 1 + 1 + 1 + 10 + 1 + 1 = 71
            line = f"{label_display:<45} {loss_str:>10} {loss_arrow} {val_str:>10} {val_arrow}"

            if is_mse:
                # Light gray for MSE lines
                # Using rich styling
                console.print(f"[grey62]{line}[/grey62]")
            else:
                console.print(line)

            self.last_eval_stats[label] = item


# Explicitly reference unused methods for static analysis tools
# pylint: disable=pointless-statement
TrainerView.on_progress_log
TrainerView.on_lr_adjusted
TrainerView.on_warning
TrainerDiagnosticsView.on_progress_log
TrainerDiagnosticsView.on_lr_adjusted
TrainerDiagnosticsView.on_warning
