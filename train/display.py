"""Display logic for training progress reporting."""

from typing import Any, Dict, List, Optional

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

console = Console()


def print_epoch_summary(
    epoch: int,
    total_epochs: int,
    primary_metrics: Dict[str, float],
    secondary_metrics: Optional[Dict[str, Any]] = None,
    phase: Optional[str] = None,
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
