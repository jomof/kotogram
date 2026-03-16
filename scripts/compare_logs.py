#!/usr/bin/env python3
"""Compare two training logs epoch-for-epoch.

Usage:
    scripts/compare_logs.py <log_a> <log_b> [--epochs N]

Displays key training metrics side-by-side for each epoch.
"""

import argparse
import re
import sys
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class EpochMetrics:
    """Metrics extracted from a single epoch."""

    epoch: int = 0
    struct: float = 0.0
    s1: int = 0
    s0: int = 0
    fuzzy: int = 0
    ortho: float = 0.0
    posp: int = 0
    k_avg: float = 0.0
    alive: int = 0

    def as_row(self) -> Tuple[object, ...]:
        """Return metric values in METRIC_DEFS order (static access)."""
        return (
            self.struct,
            self.s1,
            self.s0,
            self.fuzzy,
            self.posp,
            self.k_avg,
            self.alive,
            self.ortho,
        )


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


# (display_label, lower_is_better) — order must match EpochMetrics.as_row()
METRIC_DEFS: list[Tuple[str, Optional[bool]]] = [
    ("struct", True),
    ("S1%", True),
    ("S0%", False),
    ("Fzy%", True),
    ("PosP%", False),
    ("K(avg)", True),
    ("Alive", False),
    ("ortho", True),
]


def _parse_line(current: EpochMetrics, line: str) -> None:
    """Extract metric values from a single log line into current."""
    struct_match = re.match(r"\s+struct\s+([\d.]+)", line)
    if struct_match:
        current.struct = float(struct_match.group(1))
        return

    if "sparsity" in line:
        sparse_match = re.search(r"S1=(\d+)%\s+S0=(\d+)%\s+Fuzzy=(\d+)%", line)
        if sparse_match:
            current.s1 = int(sparse_match.group(1))
            current.s0 = int(sparse_match.group(2))
            current.fuzzy = int(sparse_match.group(3))
        return

    ortho_match = re.match(r"\s+orthogonality\s+([\d.]+)", line)
    if ortho_match:
        current.ortho = float(ortho_match.group(1))
        return

    if "grammar_point" in line and "validmask" in line:
        posp_match = re.search(r"\s(\d+)%", line)
        if posp_match:
            current.posp = int(posp_match.group(1))
        return

    total_match = re.match(r"\s*Total\s+\d+\s+[\d.]+\s+([\d.]+)\|", line)
    if total_match:
        current.k_avg = float(total_match.group(1))
        return

    alive_match = re.search(r"alive=(\d+)", line)
    if alive_match:
        current.alive = int(alive_match.group(1))


def parse_log(path: str) -> list[EpochMetrics]:
    """Parse a training log file and extract per-epoch metrics.

    The log contains multiple 'KC EP<N>' headers per epoch (one per batch
    plus the epoch-end summary). We keep the last occurrence per epoch.
    When epoch numbers reset (drop to a lower value), we detect a new
    training run and discard data from the previous run.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()

    text = _strip_ansi(raw).replace("\r", "\n")
    lines = text.split("\n")

    epoch_map: dict[int, EpochMetrics] = {}
    current: Optional[EpochMetrics] = None
    max_epoch_seen = 0

    for line in lines:
        ep_match = re.search(r"KC EP(\d+) Thawed Loss Breakdown:", line)
        if ep_match:
            if current is not None:
                epoch_map[current.epoch] = current
            ep_num = int(ep_match.group(1))
            # Detect run boundary: epoch number dropped, new run started
            if ep_num < max_epoch_seen:
                epoch_map.clear()
                max_epoch_seen = 0
            max_epoch_seen = max(max_epoch_seen, ep_num)
            current = EpochMetrics(epoch=ep_num)
            continue

        if current is not None:
            _parse_line(current, line)

    if current is not None:
        epoch_map[current.epoch] = current

    return [epoch_map[k] for k in sorted(epoch_map)]


def format_val(val: object) -> str:
    """Format a metric value for display."""
    if isinstance(val, float):
        if val == 0.0:
            return "—"
        if val >= 1.0:
            return f"{val:.3f}"
        return f"{val:.4f}"
    if isinstance(val, int):
        if val == 0:
            return "—"
        return str(val)
    return str(val)


def delta_style(a_val: object, b_val: object, lower_is_better: bool) -> str:
    """Return a rich style string for the B value based on comparison."""
    if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
        if a_val == 0 and b_val == 0:
            return "dim"
        if b_val < a_val:  # type: ignore[operator]
            return "green" if lower_is_better else "red"
        if b_val > a_val:  # type: ignore[operator]
            return "red" if lower_is_better else "green"
    return ""


# Sentinel for missing epoch data (all zeros)
_ZERO_ROW: Tuple[object, ...] = (0.0, 0, 0, 0, 0, 0.0, 0, 0.0)


def _build_row(
    ep: int,
    a: Optional[EpochMetrics],
    b: Optional[EpochMetrics],
) -> list[str]:
    """Build a single table row for one epoch."""
    row: list[str] = [str(ep)]
    vals_a = a.as_row() if a else _ZERO_ROW
    vals_b = b.as_row() if b else _ZERO_ROW
    for i, (_, lower_better) in enumerate(METRIC_DEFS):
        val_a, val_b = vals_a[i], vals_b[i]
        cell_a = format_val(val_a)
        cell_b = format_val(val_b)
        if lower_better is not None:
            style = delta_style(val_a, val_b, lower_better)
            if style:
                cell_b = f"[{style}]{cell_b}[/{style}]"
        row.append(cell_a)
        row.append(cell_b)
    return row


def print_comparison(
    epochs_a: list[EpochMetrics],
    epochs_b: list[EpochMetrics],
    label_a: str,
    label_b: str,
    max_epochs: int,
) -> None:
    """Print side-by-side comparison table using rich."""
    from rich.console import Console
    from rich.table import Table

    map_a = {e.epoch: e for e in epochs_a}
    map_b = {e.epoch: e for e in epochs_b}

    all_epochs = sorted(set(map_a.keys()) | set(map_b.keys()))
    if max_epochs > 0:
        all_epochs = all_epochs[:max_epochs]

    table = Table(
        title=f"[bold]{label_a}[/] vs [bold]{label_b}[/]",
        show_lines=False,
        pad_edge=True,
        padding=(0, 1),
    )

    table.add_column("EP", justify="right", style="bold", no_wrap=True)
    for label, _ in METRIC_DEFS:
        table.add_column(f"{label}\nA", justify="right", style="dim", no_wrap=True)
        table.add_column(f"{label}\nB", justify="right", no_wrap=True)

    for ep in all_epochs:
        table.add_row(*_build_row(ep, map_a.get(ep), map_b.get(ep)))

    console = Console(width=200)
    console.print()
    console.print(f"  A = {label_a}")
    console.print(f"  B = {label_b}")
    console.print()
    console.print(table)


def label_from_path(path: str) -> str:
    """Extract a short label from a log filename."""
    name = path.rsplit("/", maxsplit=1)[-1]
    name = name.replace("training-", "").replace(".log", "")
    name = name.replace("gp05-", "")
    return name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two training logs epoch-for-epoch."
    )
    parser.add_argument("log_a", help="Path to first log file")
    parser.add_argument("log_b", help="Path to second log file")
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Max epochs to show (0 = all)",
    )
    parser.add_argument("--label-a", default="", help="Label for first log")
    parser.add_argument("--label-b", default="", help="Label for second log")
    args = parser.parse_args()

    label_a = args.label_a or label_from_path(args.log_a)
    label_b = args.label_b or label_from_path(args.log_b)

    sys.stderr.write(f"Parsing {args.log_a}...\n")
    epochs_a = parse_log(args.log_a)
    sys.stderr.write(f"  Found {len(epochs_a)} epochs\n")

    sys.stderr.write(f"Parsing {args.log_b}...\n")
    epochs_b = parse_log(args.log_b)
    sys.stderr.write(f"  Found {len(epochs_b)} epochs\n")

    print_comparison(epochs_a, epochs_b, label_a, label_b, args.epochs)


if __name__ == "__main__":
    main()
