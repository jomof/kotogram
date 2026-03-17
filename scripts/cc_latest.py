"""scripts/cc latest -- Show Japanese content stats for the latest Common Crawl.

Queries the Common Crawl columnar index (Parquet) to estimate how much
Japanese content is available, without downloading WARC/WET files.

Downloaded Parquet files and metadata are cached in .cc/<crawl-id>/
so subsequent runs are fast.

Note: Parquet files are sorted by url_surtkey, so random samples give
skewed TLD distributions. Aggregate counts (total pages, bytes) are
reliable; per-TLD breakdowns are approximate.
"""

from __future__ import annotations

import argparse
import gzip
import time
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.cc_common import (
    CC_DATA_BASE,
    _cache_path,
    console,
    download_to_cache,
    fetch_bytes,
    format_bytes,
    get_crawl_info,
    get_crawl_sizes,
    get_latest_crawl_id,
)

if TYPE_CHECKING:
    import duckdb


def _get_warc_parquet_paths(crawl_id: str) -> list[str]:
    """Fetch WARC-subset columnar index Parquet file paths for a crawl (cached)."""
    cached = _cache_path(crawl_id, "warc-parquet-paths.txt")
    if cached.exists():
        console.print(f"  Parquet listing loaded from [dim]{cached}[/dim]")
        return [
            ln for ln in cached.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]

    url = f"{CC_DATA_BASE}/crawl-data/{crawl_id}/cc-index-table.paths.gz"
    console.print(f"  Fetching Parquet file listing from [dim]{url}[/dim]")

    raw = fetch_bytes(url)
    text = gzip.decompress(raw).decode("utf-8")
    paths = [
        line.strip()
        for line in text.strip().split("\n")
        if line.strip() and "subset=warc" in line
    ]
    cached.write_text("\n".join(paths) + "\n", encoding="utf-8")
    return paths


def _ensure_duckdb() -> duckdb.DuckDBPyConnection:
    import duckdb

    con = duckdb.connect()
    con.execute("SET enable_http_metadata_cache = true")
    con.execute("SET http_timeout = 120")
    return con


def _sample_paths(
    parquet_paths: list[str], sample_size: int
) -> tuple[list[str], float]:
    """Pick a deterministic random sample. Returns (sampled_paths, scale_factor)."""
    import random

    random.seed(42)
    n = min(sample_size, len(parquet_paths))
    sampled = random.sample(parquet_paths, n)
    return sampled, len(parquet_paths) / n


def _ensure_cached_parquet(crawl_id: str, sampled_paths: list[str]) -> list[str]:
    """Download sampled Parquet files to .cc/ if not cached. Returns local paths."""
    local_paths: list[str] = []
    to_download: list[tuple[str, str]] = []

    for s3_path in sampled_paths:
        filename = Path(s3_path).name
        local = _cache_path(crawl_id, "parquet", filename)
        local_paths.append(str(local))
        if not local.exists():
            to_download.append((f"{CC_DATA_BASE}/{s3_path}", filename))

    if to_download:
        console.print(
            f"  Downloading [bold]{len(to_download)}[/bold] Parquet files "
            f"(~600 MB each, {len(sampled_paths) - len(to_download)} already cached)..."
        )
        for i, (url, filename) in enumerate(to_download, 1):
            console.print(f"  [{i}/{len(to_download)}]")
            download_to_cache(url, crawl_id, "parquet", filename)
    else:
        console.print(
            f"  All {len(sampled_paths)} Parquet files cached in "
            f"[dim].cc/{crawl_id}/parquet/[/dim]"
        )

    return local_paths


def _run_report(  # pylint: disable=too-many-locals
    crawl_id: str,
    parquet_paths: list[str],
    sample_size: int,
    wet_warc_ratio: float,
) -> None:
    """Run all queries and print the report."""
    from rich.table import Table

    con = _ensure_duckdb()
    sampled, scale = _sample_paths(parquet_paths, sample_size)
    n_sampled = len(sampled)

    console.print(
        f"  Sampling [bold]{n_sampled}[/bold] / {len(parquet_paths)} Parquet files..."
    )
    local_paths = _ensure_cached_parquet(crawl_id, sampled)
    path_list = ", ".join(f"'{p}'" for p in local_paths)
    src = f"read_parquet([{path_list}], hive_partitioning=false)"

    t0 = time.perf_counter()

    overview = con.execute(f"""
        SELECT
            count(*)                                                  AS total_pages,
            count(*) FILTER (content_languages LIKE 'jpn%')           AS jpn_primary,
            count(*) FILTER (
                content_languages LIKE '%jpn%'
                AND content_languages NOT LIKE 'jpn%'
            )                                                         AS jpn_secondary,
            sum(warc_record_length)                                   AS total_bytes,
            sum(warc_record_length)
                FILTER (content_languages LIKE 'jpn%')                AS jpn_primary_bytes,
            sum(warc_record_length)
                FILTER (content_languages LIKE '%jpn%')               AS jpn_any_bytes,
            count(DISTINCT url_host_registered_domain)
                FILTER (content_languages LIKE 'jpn%')                AS jpn_domains,
            min(fetch_time) FILTER (content_languages LIKE 'jpn%')    AS jpn_earliest,
            max(fetch_time) FILTER (content_languages LIKE 'jpn%')    AS jpn_latest
        FROM {src}
    """).fetchone()

    tld_rows = con.execute(f"""
        SELECT url_host_tld, count(*) AS cnt
        FROM {src}
        WHERE content_languages LIKE 'jpn%'
        GROUP BY url_host_tld
        ORDER BY cnt DESC
        LIMIT 20
    """).fetchall()

    mime_rows = con.execute(f"""
        SELECT content_mime_detected, count(*) AS cnt
        FROM {src}
        WHERE content_languages LIKE 'jpn%'
        GROUP BY content_mime_detected
        ORDER BY cnt DESC
        LIMIT 10
    """).fetchall()

    status_rows = con.execute(f"""
        SELECT fetch_status, count(*) AS cnt
        FROM {src}
        WHERE content_languages LIKE 'jpn%'
        GROUP BY fetch_status
        ORDER BY cnt DESC
        LIMIT 10
    """).fetchall()

    elapsed = time.perf_counter() - t0
    console.print(f"  Queries completed in {elapsed:.1f}s")

    assert overview is not None
    (
        total_pages,
        jpn_primary,
        jpn_secondary,
        total_bytes,
        jpn_primary_bytes,
        jpn_any_bytes,
        jpn_domains,
        jpn_earliest,
        jpn_latest,
    ) = overview

    jpn_any = jpn_primary + jpn_secondary
    total_bytes = total_bytes or 0
    jpn_primary_bytes = jpn_primary_bytes or 0
    jpn_any_bytes = jpn_any_bytes or 0
    primary_pct = jpn_primary / total_pages * 100 if total_pages else 0
    any_pct = jpn_any / total_pages * 100 if total_pages else 0

    # --- Overview table ---
    console.print()
    console.rule("[bold]Overview[/bold]")

    t = Table(show_header=True, title=f"{crawl_id} — Japanese Content")
    t.add_column("Metric", style="bold")
    t.add_column("Sample", justify="right")
    t.add_column(f"Extrapolated (×{scale:.0f})", justify="right")

    t.add_row(
        "Total pages in crawl",
        f"{total_pages:,}",
        f"{int(total_pages * scale):,}",
    )
    t.add_row("", "", "")
    t.add_row(
        "Japanese (primary)",
        f"{jpn_primary:,}  ({primary_pct:.2f}%)",
        f"~{int(jpn_primary * scale):,}",
    )
    t.add_row(
        "Japanese (secondary)",
        f"{jpn_secondary:,}",
        f"~{int(jpn_secondary * scale):,}",
    )
    t.add_row(
        "Japanese (any)",
        f"{jpn_any:,}  ({any_pct:.2f}%)",
        f"~{int(jpn_any * scale):,}",
    )
    t.add_row("", "", "")
    jpn_wet = jpn_primary_bytes * wet_warc_ratio
    total_wet = total_bytes * wet_warc_ratio
    t.add_row(
        "Primary JPN text (est.)",
        format_bytes(jpn_wet),
        f"~{format_bytes(jpn_wet * scale)}",
    )
    t.add_row(
        "Primary JPN WARC (raw)",
        format_bytes(jpn_primary_bytes),
        f"~{format_bytes(jpn_primary_bytes * scale)}",
    )
    t.add_row(
        "Total text (est.)",
        format_bytes(total_wet),
        f"~{format_bytes(total_wet * scale)}",
    )
    t.add_row("", "", "")
    t.add_row(
        "Unique JPN domains",
        f"{jpn_domains:,}",
        f"~{int(jpn_domains * scale):,}",
    )
    if jpn_earliest:
        t.add_row("Earliest fetch", str(jpn_earliest)[:19], "")
    if jpn_latest:
        t.add_row("Latest fetch", str(jpn_latest)[:19], "")

    console.print(t)

    # --- TLD breakdown ---
    console.print()
    console.rule("[bold]Top TLDs (primary Japanese)[/bold]")
    console.print(
        "[dim]Note: Parquet files are sorted by URL surtkey, so TLD distribution "
        "from random samples is approximate. Use --sample 300 for full accuracy.[/dim]"
    )

    tld_table = Table(show_header=True)
    tld_table.add_column("TLD", style="bold")
    tld_table.add_column("Pages", justify="right")
    tld_table.add_column("%", justify="right")
    for tld, cnt in tld_rows:
        pct = cnt / jpn_primary * 100 if jpn_primary else 0
        tld_table.add_row(tld or "(none)", f"{cnt:,}", f"{pct:.1f}%")
    console.print(tld_table)

    # --- MIME type breakdown ---
    console.print()
    console.rule("[bold]Content Types (primary Japanese)[/bold]")

    mime_table = Table(show_header=True)
    mime_table.add_column("MIME Type", style="bold")
    mime_table.add_column("Pages", justify="right")
    mime_table.add_column("%", justify="right")
    for mime, cnt in mime_rows:
        pct = cnt / jpn_primary * 100 if jpn_primary else 0
        mime_table.add_row(mime or "(unknown)", f"{cnt:,}", f"{pct:.1f}%")
    console.print(mime_table)

    # --- HTTP status breakdown ---
    console.print()
    console.rule("[bold]HTTP Status (primary Japanese)[/bold]")

    status_table = Table(show_header=True)
    status_table.add_column("Status", style="bold")
    status_table.add_column("Pages", justify="right")
    status_table.add_column("%", justify="right")
    for status, cnt in status_rows:
        pct = cnt / jpn_primary * 100 if jpn_primary else 0
        status_table.add_row(str(status), f"{cnt:,}", f"{pct:.1f}%")
    console.print(status_table)

    console.print()
    console.print(
        f"[dim]Based on {n_sampled}/{len(parquet_paths)} Parquet files. "
        f"Use --sample N for higher accuracy.[/dim]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show Japanese content stats for the latest Common Crawl."
    )
    parser.add_argument(
        "--crawl",
        default=None,
        help="Crawl ID to query (default: latest). Example: CC-MAIN-2026-08",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="Number of Parquet files to sample (default: 10)",
    )
    args = parser.parse_args()

    console.rule("[bold]Common Crawl — Japanese Content Report[/bold]")

    if args.crawl:
        crawl_id = args.crawl
    else:
        console.print("Fetching latest crawl info...")
        crawl_id = get_latest_crawl_id()

    info = get_crawl_info(crawl_id)
    console.print()
    console.print(f"  Crawl:  [bold]{crawl_id}[/bold]  ({info['name']})")
    console.print(f"  Period: {info['from'][:10]}  →  {info['to'][:10]}")
    console.print(f"  CDX:    {info['cdx-api']}")
    console.print()

    parquet_paths = _get_warc_parquet_paths(crawl_id)
    console.print(
        f"  Columnar index: [bold]{len(parquet_paths)}[/bold] Parquet files (WARC subset)"
    )

    sizes = get_crawl_sizes(crawl_id)
    wet_warc_ratio = sizes["wet"] / sizes["warc"] if sizes.get("warc") else 0.075
    console.print(
        f"  Crawl totals: WARC {sizes.get('warc', '?')} TiB, "
        f"WET {sizes.get('wet', '?')} TiB  "
        f"(text/raw ratio: {wet_warc_ratio:.1%})"
    )
    console.print()

    console.rule("[bold]Sampling[/bold]")
    _run_report(crawl_id, parquet_paths, args.sample, wet_warc_ratio)


if __name__ == "__main__":
    main()
