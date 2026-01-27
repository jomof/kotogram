"""
Clone the exact schema of a SQLite database reflectively.
"""

import os
import sqlite3
import sys

from rich.console import Console

console = Console()


def curate_clone_empty(
    source_db_path: str, target_db_path: str, overwrite: bool = False
) -> None:
    """
    Clone the schema of source_db_path to target_db_path without copying data.

    Args:
        source_db_path: Path to the source database (must exist).
        target_db_path: Path to the target database (created).
        overwrite: If True, overwrite target if it exists.
    """
    if not os.path.exists(source_db_path):
        console.print(
            f"[red]Error: Source database not found at {source_db_path}[/red]"
        )
        sys.exit(1)

    if os.path.exists(target_db_path):
        if overwrite:
            console.print(
                f"[yellow]Removing existing database at {target_db_path}[/yellow]"
            )
            os.remove(target_db_path)
        else:
            console.print(
                f"[red]Error: Target database already exists at {target_db_path}. Use --overwrite to replace.[/red]"
            )
            sys.exit(1)

    # Ensure directory exists
    target_dir = os.path.dirname(os.path.abspath(target_db_path))
    os.makedirs(target_dir, exist_ok=True)

    console.print(
        f"[blue]Cloning schema from {source_db_path} to {target_db_path}...[/blue]"
    )

    src_conn = sqlite3.connect(source_db_path)
    tgt_conn = sqlite3.connect(target_db_path)

    try:
        src_c = src_conn.cursor()
        tgt_c = tgt_conn.cursor()

        # Get all objects from sqlite_master
        # Order matters: Tables, then Indexes, then Triggers, then Views.
        # But sqlite_master doesn't guarantee dependency order for views/triggers.
        # Simple tables usually come first or can be unordered if no FKs?
        # If we enable FKs later, order matters.
        # Let's get them by type.

        # 1. Tables (excluding sqlite_sequence)
        src_c.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence' AND sql IS NOT NULL"
        )
        tables = [row[0] for row in src_c.fetchall()]

        # 2. Indexes
        src_c.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
        )
        indexes = [row[0] for row in src_c.fetchall()]

        # 3. Triggers
        src_c.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' AND sql IS NOT NULL"
        )
        triggers = [row[0] for row in src_c.fetchall()]

        # 4. Views
        src_c.execute(
            "SELECT sql FROM sqlite_master WHERE type='view' AND sql IS NOT NULL"
        )
        views = [row[0] for row in src_c.fetchall()]

        console.print(
            f"  Found {len(tables)} tables, {len(indexes)} indexes, {len(triggers)} triggers, {len(views)} views."
        )

        # Execute
        for sql in tables:
            tgt_c.execute(sql)

        for sql in indexes:
            tgt_c.execute(sql)

        for sql in triggers:
            tgt_c.execute(sql)

        for sql in views:
            tgt_c.execute(sql)

        tgt_conn.commit()

        console.print(f"[green]Successfully created {target_db_path} (Empty)[/green]")

    finally:
        src_conn.close()
        tgt_conn.close()
