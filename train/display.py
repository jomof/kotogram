"""Display logic for training progress reporting."""

import os
from typing import Any, Dict, Optional, Tuple, cast

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Allow disabling force_terminal for tests/logging
_force_term = os.getenv("KOTOGRAM_FORCE_TERMINAL", "True").lower() == "true"
console = Console(force_terminal=_force_term)


def format_worst_sample_display(
    sentence: str, loss: float, max_len: int = 60
) -> Tuple[str, str]:
    """Return truncated sentence and loss color for worst-sample display."""
    if len(sentence) > max_len:
        sentence = sentence[: max_len - 3] + "..."
    loss_color = "red" if loss > 1.0 else ("yellow" if loss > 0.25 else "dim")
    return sentence, loss_color


class RichTrainerProgressBar:
    """Stateful progress bar for training loops using Rich."""

    def __init__(
        self,
        desc: str,
        total_steps: int,
        batch_size: int,
    ):
        # Use provided console or fall back to global forced-terminal console
        self.console = console
        self.batch_size = batch_size
        # Display format: elapsed/remaining time (e.g., "1:30/2:00")
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("/"),
            TimeRemainingColumn(),
            TextColumn("{task.fields[status]}"),
            TextColumn("{task.fields[throughput]}"),
            TextColumn("{task.fields[total_els]}"),
            console=self.console,
            transient=False,
        )
        self.task_id = self.progress.add_task(
            desc,
            total=total_steps,
            status="Initializing...",
            throughput="",
            total_els="",
        )
        self.progress.start()
        self.total_elements = 0

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

        # Accumulate elements (cast to int for test compatibility with mocks)
        self.total_elements += int(self.batch_size)

        # Calculate throughput and total elements display
        task = self.progress.tasks[int(self.task_id)]
        if task.speed is not None and task.speed > 0:
            samples_per_sec = task.speed * self.batch_size
            fields["throughput"] = f"{samples_per_sec:.1f} el/s"

        # Format total elements with K/M suffix
        if self.total_elements >= 1_000_000:
            fields["total_els"] = f"[{self.total_elements / 1_000_000:.1f}M els]"
        elif self.total_elements >= 1_000:
            fields["total_els"] = f"[{self.total_elements / 1_000:.0f}K els]"
        else:
            fields["total_els"] = f"[{self.total_elements} els]"

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


def print_best_model_saved(path: str, val_loss: float) -> None:
    """Print success message when a new best model is saved.

    Args:
        path: Path where the model was saved.
        val_loss: The validation loss of this best model.
    """
    console.print(
        f"[bold green]New best model matched! Saving to {path} (Loss: {val_loss:.4f})[/bold green]"
    )
