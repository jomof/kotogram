"""scripts/cc fetch-jp-wet -- Download WET files and extract Japanese text.

Downloads Common Crawl WET files, filters for Japanese-primary pages
(using URLs from the Parquet index), and saves only the Japanese text.
Raw WET files are deleted after processing; only the extracted text is kept.

Processed output is cached in .cc/<crawl-id>/wet-jp/ as JSONL files
(one JSON object per page: {"url": ..., "text": ...}).
"""

from __future__ import annotations

import argparse
import collections
import gzip
import http.client
import io
import json
import sys
import time
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.cc_common import (
    CC_DATA_BASE,
    _cache_path,
    console,
    download_progress,
    format_bytes,
    get_crawl_info,
    get_latest_crawl_id,
)

if TYPE_CHECKING:
    import duckdb


def _ensure_duckdb() -> duckdb.DuckDBPyConnection:
    import duckdb

    con = duckdb.connect()
    return con


def _get_cached_parquet_files(crawl_id: str) -> list[str]:
    """Return list of locally cached Parquet files for a crawl."""
    parquet_dir = _cache_path(crawl_id, "parquet", ".placeholder").parent
    if not parquet_dir.exists():
        return []
    return sorted(str(p) for p in parquet_dir.glob("*.parquet"))


def _index_cache_path(crawl_id: str) -> Path:
    return _cache_path(crawl_id, "wet-jp-index.json.gz")


def _build_jpn_url_index(
    crawl_id: str,
    con: duckdb.DuckDBPyConnection,
    parquet_files: list[str],
) -> dict[str, list[str]]:
    """Query Parquet files for Japanese-primary pages.

    Returns {wet_filename: [url1, url2, ...]}, cached to disk.
    """
    cache_file = _index_cache_path(crawl_id)
    if cache_file.exists():
        console.print(f"  Loading Japanese URL index from [dim]{cache_file}[/dim]...")
        t0 = time.perf_counter()
        with gzip.open(cache_file, "rt", encoding="utf-8") as f:
            index: dict[str, list[str]] = json.load(f)
        total_urls = sum(len(v) for v in index.values())
        elapsed = time.perf_counter() - t0
        console.print(
            f"  {total_urls:,} Japanese URLs across "
            f"{len(index):,} WET files ({elapsed:.1f}s)"
        )
        return index

    file_list = ", ".join(f"'{f}'" for f in parquet_files)
    src = f"read_parquet([{file_list}], hive_partitioning=false)"

    console.print("  Building Japanese URL index from cached Parquet files...")
    t0 = time.perf_counter()

    rows = con.execute(f"""
        SELECT warc_filename, url
        FROM {src}
        WHERE content_languages LIKE 'jpn%'
    """).fetchall()

    index = {}
    for warc_fn, url in rows:
        wet_fn = warc_fn.replace("/warc/", "/wet/").replace(".warc.gz", ".warc.wet.gz")
        index.setdefault(wet_fn, []).append(url)

    elapsed = time.perf_counter() - t0
    total_urls = sum(len(v) for v in index.values())
    console.print(
        f"  Found {total_urls:,} Japanese URLs across "
        f"{len(index):,} WET files ({elapsed:.1f}s)"
    )

    console.print(f"  Caching index to [dim]{cache_file}[/dim]...")
    with gzip.open(cache_file, "wt", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)

    return index


def _processed_path(crawl_id: str, wet_filename: str) -> Path:
    """Return the output path for a processed WET file."""
    stem = Path(wet_filename).name.replace(".warc.wet.gz", "")
    return _cache_path(crawl_id, "wet-jp", f"{stem}.jsonl.gz")


def _process_wet_stream(
    stream: io.BufferedIOBase,
    jpn_urls: frozenset[str],
) -> list[dict[str, str]]:
    """Parse a WET stream and extract records matching the Japanese URL set."""
    from warcio.archiveiterator import ArchiveIterator  # type: ignore[import-untyped]

    results: list[dict[str, str]] = []
    for record in ArchiveIterator(stream):
        if record.rec_type != "conversion":
            continue
        uri = record.rec_headers.get_header("WARC-Target-URI")
        if uri not in jpn_urls:
            continue
        text = record.content_stream().read().decode("utf-8", errors="replace")
        text = text.strip()
        if text:
            results.append({"url": uri, "text": text})
    return results


_MAX_RETRIES = 8


def _urlopen_with_retry(
    url: str,
    label: str,
) -> http.client.HTTPResponse | None:
    """Open *url* with exponential-backoff retries on 503 (S3 backpressure).

    Uses http.client directly so 503 is just a status code, not an exception.
    Returns ``None`` when all retries are exhausted so the caller can skip.
    """
    parsed = urllib.parse.urlsplit(url)
    assert parsed.hostname is not None
    headers = {"User-Agent": "kotogram-cc/1.0"}

    for attempt in range(1, _MAX_RETRIES + 1):
        conn = http.client.HTTPSConnection(parsed.hostname, timeout=60)
        conn.request("GET", parsed.path, headers=headers)
        resp = conn.getresponse()
        if resp.status != 503:
            return resp
        resp.read()
        conn.close()
        if attempt == _MAX_RETRIES:
            console.print(
                f"    [red]503 persisted after {_MAX_RETRIES} retries,"
                f" requeueing {label}[/red]"
            )
            return None
        wait = min(2 ** (attempt - 1), 30)
        console.print(
            f"    [yellow]503 on {label}, retry {attempt}"
            f"/{_MAX_RETRIES} in {wait}s[/yellow]"
        )
        time.sleep(wait)
    return None  # pragma: no cover


def _download_and_process_wet(  # pylint: disable=too-many-locals
    crawl_id: str,
    wet_path: str,
    jpn_urls: frozenset[str],
) -> tuple[int, int] | None:
    """Download a WET file, extract Japanese text, save, delete raw.

    Returns (pages_extracted, bytes_written), or ``None`` if the download
    failed after all retries (503 backpressure).
    """
    url = f"{CC_DATA_BASE}/{wet_path}"
    wet_name = Path(wet_path).name
    resp = _urlopen_with_retry(url, wet_name)
    if resp is None:
        return None

    with resp:
        total = int(resp.headers.get("Content-Length") or 0) or None
        chunks: list[bytes] = []

        with download_progress() as progress:
            task = progress.add_task(f"    {wet_name}", total=total)
            while True:
                chunk = resp.read(256 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
                progress.advance(task, len(chunk))

    raw_data = b"".join(chunks)
    results = _process_wet_stream(io.BytesIO(raw_data), jpn_urls)

    out_path = _processed_path(crawl_id, wet_path)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(results), out_path.stat().st_size


def main() -> None:  # pylint: disable=too-many-locals
    parser = argparse.ArgumentParser(
        description="Download WET files and extract Japanese text."
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of WET files to download and process.",
    )
    parser.add_argument(
        "--crawl",
        default=None,
        help="Crawl ID (default: latest). Example: CC-MAIN-2026-08",
    )
    args = parser.parse_args()

    console.rule("[bold]Common Crawl — Fetch Japanese WET[/bold]")

    crawl_id = args.crawl or get_latest_crawl_id()
    info = get_crawl_info(crawl_id)
    console.print(f"  Crawl: [bold]{crawl_id}[/bold]  ({info['name']})")
    console.print()

    parquet_files = _get_cached_parquet_files(crawl_id)
    if not parquet_files:
        console.print(
            "[red]No cached Parquet files found. Run 'scripts/cc latest' first.[/red]"
        )
        sys.exit(1)
    console.print(f"  Using {len(parquet_files)} cached Parquet files")

    con = _ensure_duckdb()
    wet_index = _build_jpn_url_index(crawl_id, con, parquet_files)

    already_done = 0
    to_process: list[tuple[str, list[str]]] = []
    for wet_path, urls in wet_index.items():
        if _processed_path(crawl_id, wet_path).exists():
            already_done += 1
        else:
            to_process.append((wet_path, urls))

    console.print(
        f"  {already_done:,} already processed, "
        f"{len(to_process):,} remaining (--count max: {len(to_process)})"
    )

    remaining = min(args.count, len(to_process))
    if remaining == 0:
        console.print("[green]Nothing to do.[/green]")
        return

    # Sort by descending Japanese page count for best yield first
    to_process.sort(key=lambda x: len(x[1]), reverse=True)
    batch = to_process[:remaining]

    est_download = remaining * 62 * 1024 * 1024
    console.print(
        f"\n  Will process [bold]{remaining}[/bold] WET files "
        f"(~{format_bytes(est_download)} download)"
    )
    console.print()
    console.rule("[bold]Downloading & Processing[/bold]")

    total_pages = 0
    total_bytes = 0
    completed = 0
    skipped = 0
    queue: collections.deque[tuple[str, list[str]]] = collections.deque(batch)
    requeued: set[str] = set()

    while queue:
        wet_path, urls = queue.popleft()
        completed += 1
        tag = " [yellow](retry)[/yellow]" if wet_path in requeued else ""
        console.print(
            f"  [{completed}/{remaining + len(requeued)}]"
            f" {len(urls)} expected Japanese pages{tag}"
        )
        result = _download_and_process_wet(crawl_id, wet_path, frozenset(urls))
        if result is None:
            if wet_path not in requeued:
                requeued.add(wet_path)
                queue.append((wet_path, urls))
                console.print(
                    f"    [yellow]Requeued to end ({len(queue)} remaining)[/yellow]"
                )
            else:
                skipped += 1
                console.print("    [red]Skipped (failed twice)[/red]")
            continue
        pages, nbytes = result
        total_pages += pages
        total_bytes += nbytes
        console.print(
            f"    Extracted {pages} pages → {format_bytes(nbytes)} compressed"
        )

    console.print()
    console.rule("[bold]Summary[/bold]")
    console.print(f"  Processed:  {completed - skipped} WET files")
    if skipped:
        console.print(f"  [red]Skipped:    {skipped} (persistent 503)[/red]")
    console.print(f"  Extracted:  {total_pages:,} Japanese pages")
    console.print(
        f"  Saved:      {format_bytes(total_bytes)} in .cc/{crawl_id}/wet-jp/"
    )


if __name__ == "__main__":
    main()
