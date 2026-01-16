"""Shared progress bar utilities for scripts."""

from contextlib import contextmanager
from typing import Iterator

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)


@contextmanager
def create_progress(console: Console) -> Iterator[Progress]:
    """Create a standard progress bar with common columns.

    Args:
        console: Rich console for output

    Yields:
        Progress instance configured with standard columns
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        yield progress
