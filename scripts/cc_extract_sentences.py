"""scripts/cc extract-sentences -- Extract and select Japanese sentences from Common Crawl.

Reads the JSONL.gz files produced by 'cc fetch-jp-wet', extracts individual
Japanese sentences using heuristic splitting and filtering, then scores them
by embedding diversity and prediction uncertainty to select the most impactful
sentences for training.

Incremental: tracks which JSONL files have been processed. Re-running
after fetching more WET files only processes the new ones.  Re-running
with no new files skips extraction and re-runs selection (e.g. with a
different --top-pct).

Output:
  .cc/<crawl-id>/sentences.txt.gz           All extracted sentences
  .cc/selected-sentences.txt                  Top-scored subset (stable path)
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import re
import sqlite3
import struct
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

from scripts.cc_common import (
    _CORPUS_EMBED_META,
    CC_CACHE_DIR,
    CHAR_FILTER,
    CORPUS_DB,
    EmbedStore,
    GRAMMATIC_SOFT_MIN,
    MAX_SENTENCE_LEN,
    _cache_path,
    clean_sentence,
    console,
    content_ok,
    diversity_scores,
    format_bytes,
    get_cc_scores,
    get_corpus_embeddings,
    get_crawl_info,
    get_latest_crawl_id,
    model_hash,
    perf_flush,
    perf_log,
    perf_log_dist,
    perf_start_run,
    set_tokenizer_path,
)

_HIRAGANA_RE = re.compile(r"[\u3040-\u309F]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?])")

MIN_LENGTH = 10
MAX_LENGTH = MAX_SENTENCE_LEN

_JUNK_PATTERNS = [
    re.compile(r"【ダミー"),
    re.compile(r"\|intent:"),
    re.compile(r"\|target_len:"),
    re.compile(r"ブランド:.*ジャンル:"),
    re.compile(r"^\d+\s[「『]"),
    re.compile(r"^(\d{2,4}[/\-年]){1,2}"),
    re.compile(r"(好評)?発売中です"),
    re.compile(r"^\(\d{2}/\d{2}\)$"),
    re.compile(r"https?://"),
    re.compile(r"[a-zA-Z]{10,}"),
    re.compile(r"cookie|Cookie|COOKIE"),
    re.compile(r"プライバシーポリシー|利用規約|著作権"),
]

_BLOOM_CACHE = Path(".cc/corpus-bloom.bin")
_BLOOM_FPR = 0.001


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Light normalization: NFKC, strip invisible formatting, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")
    text = text.replace("\u200d", "")
    text = text.replace("\u200e", "")
    text = re.sub(r"[\s\u3000]+", " ", text).strip()
    return text


def _is_candidate(s: str, min_len: int, max_len: int) -> bool:
    """Structural check: right length, has hiragana, ends with sentence punctuation."""
    if len(s) < min_len or len(s) > max_len:
        return False
    if not _HIRAGANA_RE.search(s):
        return False
    if s[-1] not in "。！？!?":
        return False
    return True


def _passes_char_filter(s: str) -> bool:
    """Return True if the sentence contains no banned characters or junk patterns."""
    if CHAR_FILTER.search(s):
        return False
    for pat in _JUNK_PATTERNS:
        if pat.search(s):
            return False
    return True


def _extract_sentences_from_text(
    text: str, min_len: int, max_len: int
) -> tuple[int, list[str]]:
    """Split page text into individual sentences and filter.

    Returns (candidates_before_char_filter, filtered_sentences).
    """
    candidates = 0
    sentences: list[str] = []
    for line in text.split("\n"):
        line = _normalize(line)
        if not line:
            continue
        parts = _SENTENCE_SPLIT_RE.split(line)
        for part in parts:
            part = part.strip()
            if _is_candidate(part, min_len, max_len):
                candidates += 1
                if _passes_char_filter(part):
                    sentences.append(part)
    return candidates, sentences


# ---------------------------------------------------------------------------
# File / manifest helpers
# ---------------------------------------------------------------------------


def _wet_jp_dir(crawl_id: str) -> Path:
    return _cache_path(crawl_id, "wet-jp", ".placeholder").parent


def _manifest_path(crawl_id: str) -> Path:
    return _cache_path(crawl_id, "sentences-manifest.json")


def _sentences_path(crawl_id: str) -> Path:
    return _cache_path(crawl_id, "sentences.txt.gz")


# ---------------------------------------------------------------------------
# Bloom filter for corpus deduplication
# ---------------------------------------------------------------------------


class BloomFilter:
    """Simple bloom filter backed by a bytearray."""

    def __init__(self, capacity: int, fpr: float = _BLOOM_FPR):
        self.num_hashes = max(1, round(-math.log2(fpr)))
        self.num_bits = max(8, round(-capacity * math.log(fpr) / (math.log(2) ** 2)))
        self.bits = bytearray(self.num_bits // 8 + 1)
        self.count = 0

    def _hashes(self, key: str) -> list[int]:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        h1, h2 = struct.unpack_from("<QQ", digest)
        return [(h1 + i * h2) % self.num_bits for i in range(self.num_hashes)]

    def add(self, key: str) -> None:
        for h in self._hashes(key):
            self.bits[h >> 3] |= 1 << (h & 7)
        self.count += 1

    def might_contain(self, key: str) -> bool:
        return all(self.bits[h >> 3] & (1 << (h & 7)) for h in self._hashes(key))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(struct.pack("<QII", self.num_bits, self.num_hashes, self.count))
            f.write(self.bits)

    @classmethod
    def load(cls, path: Path) -> BloomFilter:
        with open(path, "rb") as f:
            num_bits, num_hashes, count = struct.unpack("<QII", f.read(16))
            bits = bytearray(f.read())
        bf = cls.__new__(cls)
        bf.num_bits = num_bits
        bf.num_hashes = num_hashes
        bf.count = count
        bf.bits = bits
        return bf


def _corpus_mtime() -> float:
    return CORPUS_DB.stat().st_mtime if CORPUS_DB.exists() else 0.0


def _bloom_is_fresh() -> bool:
    if not _BLOOM_CACHE.exists():
        return False
    return _BLOOM_CACHE.stat().st_mtime >= _corpus_mtime()


def _build_corpus_bloom() -> BloomFilter:
    """Build a bloom filter from all sentences in corpus.db and cache it."""
    console.print("  Building corpus bloom filter...")
    con = sqlite3.connect(str(CORPUS_DB))
    n = con.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]
    bf = BloomFilter(max(n, 1))
    cur = con.execute("SELECT sentence FROM sentences")
    loaded = 0
    while True:
        rows = cur.fetchmany(10_000)
        if not rows:
            break
        for (s,) in rows:
            bf.add(clean_sentence(s))
            loaded += 1
    con.close()
    bf.save(_BLOOM_CACHE)
    console.print(
        f"  Bloom filter: {loaded:,} sentences, {len(bf.bits):,} bytes cached"
    )
    return bf


def _load_or_build_bloom() -> BloomFilter:
    if _bloom_is_fresh():
        bf = BloomFilter.load(_BLOOM_CACHE)
        console.print(f"  Corpus bloom filter: {bf.count:,} sentences (cached)")
        return bf
    if not CORPUS_DB.exists():
        console.print("  [yellow]No corpus.db found, skipping corpus dedup[/yellow]")
        return BloomFilter(1)
    return _build_corpus_bloom()


def _corpus_dedup(sentences: list[str], bloom: BloomFilter) -> tuple[list[str], int]:
    """Remove sentences already in corpus.db using bloom + SQL verification.

    Returns (filtered_sentences, num_removed).
    """
    if bloom.count == 0:
        return sentences, 0

    candidates = [s for s in sentences if bloom.might_contain(s)]
    if not candidates:
        return sentences, 0

    con = sqlite3.connect(str(CORPUS_DB))
    corpus_cleaned: set[str] = set()
    cur = con.execute("SELECT sentence FROM sentences")
    while True:
        rows = cur.fetchmany(10_000)
        if not rows:
            break
        for (s,) in rows:
            corpus_cleaned.add(clean_sentence(s))
    con.close()

    in_corpus = {s for s in candidates if s in corpus_cleaned}
    if not in_corpus:
        return sentences, 0

    filtered = [s for s in sentences if s not in in_corpus]
    return filtered, len(in_corpus)


def _load_manifest(crawl_id: str) -> set[str]:
    """Load the set of already-processed JSONL filenames."""
    path = _manifest_path(crawl_id)
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(json.load(f))


def _save_manifest(crawl_id: str, processed: set[str]) -> None:
    with open(_manifest_path(crawl_id), "w", encoding="utf-8") as f:
        json.dump(sorted(processed), f)


def _load_existing_sentences(crawl_id: str) -> tuple[list[str], set[str]]:
    """Load previously extracted sentences. Returns (ordered_list, dedup_set)."""
    path = _sentences_path(crawl_id)
    if not path.exists():
        return [], set()
    sentences: list[str] = []
    seen: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n")
            if s and s not in seen:
                seen.add(s)
                sentences.append(s)
    return sentences, seen


# ---------------------------------------------------------------------------
# Parallel WET extraction
# ---------------------------------------------------------------------------


def _process_one_file(
    jsonl_path: str, min_l: int, max_l: int
) -> tuple[str, int, int, list[str]]:
    """Worker: extract sentences from a single JSONL.gz file."""
    pages = 0
    cands = 0
    sents: list[str] = []
    with gzip.open(jsonl_path, "rt", encoding="utf-8") as f:
        for raw_line in f:
            pages += 1
            rec = json.loads(raw_line)
            text = rec.get("text", "")
            c, s = _extract_sentences_from_text(text, min_l, max_l)
            cands += c
            sents.extend(s)
    return Path(jsonl_path).name, pages, cands, sents


def _run_extraction(  # pylint: disable=too-many-locals
    new_files: list[Path],
    min_len: int,
    max_len: int,
    seen: set[str],
    already_processed: set[str],
) -> dict[str, Any]:
    """Run parallel extraction across JSONL files, dedup into seen set."""
    from rich.progress import Progress

    new_sentences: list[str] = []
    pages = 0
    candidates = 0
    extracted = 0

    num_workers = min(mp.cpu_count() or 1, len(new_files))
    worker_args = [(str(f), min_len, max_len) for f in new_files]

    with Progress(console=console) as progress:
        task = progress.add_task("  Extracting...", total=len(new_files))
        with mp.Pool(num_workers) as pool:
            for fname, file_pages, cands, sents in pool.starmap(
                _process_one_file, worker_args, chunksize=4
            ):
                pages += file_pages
                candidates += cands
                for s in sents:
                    extracted += 1
                    if s not in seen:
                        seen.add(s)
                        new_sentences.append(s)
                already_processed.add(fname)
                progress.advance(task)

    return {
        "new_sentences": new_sentences,
        "pages": pages,
        "candidates": candidates,
        "extracted": extracted,
    }


# ---------------------------------------------------------------------------
# Diversity cache (nearest-neighbour distances against corpus)
# ---------------------------------------------------------------------------


_DIV_META = "diversity-meta.json"


def _corpus_prefix_ok() -> tuple[bool, str]:
    """Check whether the old corpus is an exact prefix of the current one.

    ``get_corpus_embeddings`` stores *prefix_fp* (fingerprint of the reused
    rows) and *full_fp* (fingerprint of the entire array).  If the current
    ``prefix_fp`` equals the ``corpus_fp`` stored by the last diversity run,
    the old corpus rows occupy indices ``[:old_corpus_n]`` unchanged and new
    rows sit at ``[old_corpus_n:]``.
    """
    corpus_meta_path = CC_CACHE_DIR / _CORPUS_EMBED_META
    if not corpus_meta_path.exists():
        return False, ""
    meta = json.loads(corpus_meta_path.read_text(encoding="utf-8"))
    return True, meta.get("prefix_fp", "")


def _get_diversity(
    crawl_id: str,
    cc_emb: np.ndarray,
    corpus_emb: np.ndarray,
    model_md5: str,
    device: Any = None,
) -> np.ndarray:
    """Load or compute NN diversity scores, with incremental extension.

    Tracks ``(cc_n, corpus_n, corpus_fp)`` so incremental updates are safe:

    * **Corpus grew (additions only)**: old corpus is a verified prefix
      → compute distances only against the new corpus rows and take the
      element-wise minimum with cached scores.
    * **Corpus changed non-trivially** (removals / reordering): detected
      via ``prefix_fp`` mismatch → full recompute.
    * **CC grew**: extend with new CC rows scored against full corpus.
    """
    _t0_div = time.monotonic()
    _div_stale = ""
    cache_dir = _cache_path(crawl_id, "inference")
    cache_dir.mkdir(parents=True, exist_ok=True)
    div_path = cache_dir / "diversity.npy"
    meta_path = cache_dir / _DIV_META

    n_cc = cc_emb.shape[0]
    n_corpus = corpus_emb.shape[0]

    old_cc_n = 0
    old_corpus_n = 0
    old_corpus_fp = ""
    cached_div: np.ndarray | None = None

    if div_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("model_hash") == model_md5:
            old_cc_n = meta.get("cc_n", 0)
            old_corpus_n = meta.get("corpus_n", 0)
            old_corpus_fp = meta.get("corpus_fp", "")
            existing: np.ndarray = np.load(str(div_path))
            if existing.shape[0] == old_cc_n and old_cc_n <= n_cc:
                cached_div = existing
            else:
                old_cc_n = 0
                old_corpus_n = 0

    # Verify the old corpus is a true prefix of the current one
    prefix_ok, current_prefix_fp = _corpus_prefix_ok()
    if cached_div is not None and old_corpus_fp and prefix_ok:
        if current_prefix_fp != old_corpus_fp:
            console.print(
                "  Diversity cache stale (corpus rows changed), recomputing..."
            )
            _div_stale = "stale_corpus_prefix"
            cached_div = None
            old_cc_n = 0
            old_corpus_n = 0

    # Read the current full fingerprint to store in our metadata
    corpus_meta_path = CC_CACHE_DIR / _CORPUS_EMBED_META
    current_corpus_fp = ""
    if corpus_meta_path.exists():
        cmeta = json.loads(corpus_meta_path.read_text(encoding="utf-8"))
        current_corpus_fp = cmeta.get("full_fp", "")

    if cached_div is not None and old_cc_n == n_cc and old_corpus_n == n_corpus:
        perf_log(
            "diversity",
            cache="hit",
            cc=n_cc,
            corpus=n_corpus,
            time_s=time.monotonic() - _t0_div,
        )
        console.print("  Diversity scores loaded from cache")
        return cached_div

    if cached_div is None:
        console.print("  Computing diversity scores (no valid cache)...")
        result = diversity_scores(cc_emb, corpus_emb, device=device)
    else:
        parts: list[np.ndarray] = []

        if old_cc_n > 0:
            old_div = cached_div
            if old_corpus_n < n_corpus:
                new_corpus_rows = n_corpus - old_corpus_n
                console.print(
                    f"  Diversity: updating {old_cc_n:,} cached scores"
                    f" against {new_corpus_rows:,} new corpus rows"
                )
                update_dists = diversity_scores(
                    cc_emb[:old_cc_n],
                    corpus_emb[old_corpus_n:],
                    device=device,
                )
                old_div = np.minimum(old_div, update_dists)
            parts.append(old_div)

        if old_cc_n < n_cc:
            new_cc_rows = n_cc - old_cc_n
            console.print(
                f"  Diversity: scoring {new_cc_rows:,} new CC rows"
                f" against {n_corpus:,} corpus rows"
            )
            parts.append(diversity_scores(cc_emb[old_cc_n:], corpus_emb, device=device))

        result = np.concatenate(parts) if len(parts) > 1 else parts[0]

    if cached_div is None:
        _div_kind = _div_stale or "miss"
    elif old_cc_n < n_cc and old_corpus_n < n_corpus:
        _div_kind = "partial_both"
    elif old_cc_n < n_cc:
        _div_kind = "partial_cc"
    elif old_corpus_n < n_corpus:
        _div_kind = "partial_corpus"
    else:
        _div_kind = "recompute"
    perf_log(
        "diversity",
        cache=_div_kind,
        cc=n_cc,
        corpus=n_corpus,
        old_cc=old_cc_n,
        old_corpus=old_corpus_n,
        time_s=time.monotonic() - _t0_div,
    )
    np.save(str(div_path), result)
    div_meta = {
        "cc_n": n_cc,
        "corpus_n": n_corpus,
        "corpus_fp": current_corpus_fp,
        "model_hash": model_md5,
    }
    meta_path.write_text(json.dumps(div_meta), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Ranking and selection
# ---------------------------------------------------------------------------


def _rank_percentiles(values: np.ndarray) -> np.ndarray:
    """Convert raw values to rank percentiles in [0, 1]."""
    order = values.argsort()
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    pcts: np.ndarray = ranks / max(len(values) - 1, 1)
    return pcts


def _select(
    impact: np.ndarray,
    top_pct: float | None,
    min_impact: float | None,
) -> tuple[np.ndarray, float]:
    """Return (selected_indices, cutoff)."""
    if min_impact is not None:
        sel = np.where(impact >= min_impact)[0]
        cutoff = min_impact
    else:
        pct = top_pct if top_pct is not None else 1.0
        k = max(1, int(len(impact) * pct / 100.0))
        top_idx = np.argpartition(impact, -k)[-k:]
        sel = top_idx[np.argsort(impact[top_idx])[::-1]]
        cutoff = float(impact[sel[-1]]) if len(sel) > 0 else 0.0
    return sel, cutoff


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_extraction_summary(
    *,
    new_files: int,
    new_pages: int,
    new_candidates: int,
    new_extracted: int,
    new_unique: int,
    corpus_dupes: int,
    total: int,
    out_path: Path,
) -> None:
    from rich.table import Table

    console.print()
    console.rule("[bold]Extraction Summary[/bold]")
    tbl = Table(show_header=False)
    tbl.add_column("Metric", style="bold")
    tbl.add_column("Value", justify="right")
    filtered_pct = (
        f"  ({100 * (new_candidates - new_extracted) / new_candidates:.1f}% filtered)"
        if new_candidates > 0
        else ""
    )
    tbl.add_row("New files processed", f"{new_files}")
    tbl.add_row("New pages", f"{new_pages:,}")
    tbl.add_row("New candidate sentences", f"{new_candidates:,}")
    tbl.add_row("After filtering", f"{new_extracted:,}{filtered_pct}")
    tbl.add_row("New unique sentences", f"{new_unique:,}")
    tbl.add_row("Already in corpus.db", f"-{corpus_dupes:,}" if corpus_dupes else "0")
    tbl.add_row("", "")
    tbl.add_row("Total unique sentences", f"{total:,}")
    tbl.add_row("Output", str(out_path))
    tbl.add_row("Output size", format_bytes(out_path.stat().st_size))
    console.print(tbl)


def _print_length_histogram(sentences: list[str]) -> None:  # pylint: disable=too-many-locals
    from rich.table import Table

    console.print()
    console.rule("[bold]Sentence Length Distribution[/bold]")

    lengths = [len(s) for s in sentences]
    shortest = min(lengths)
    longest = max(lengths)
    avg = sum(lengths) / len(lengths)

    buckets = [
        (10, 14),
        (15, 19),
        (20, 29),
        (30, 39),
        (40, 49),
        (50, 59),
        (60, 79),
        (80, 99),
        (100, 149),
        (150, 199),
        (200, 299),
        (300, 500),
    ]
    max_count = 0
    bucket_data: list[tuple[str, int]] = []
    for lo, hi in buckets:
        count = sum(1 for ln in lengths if lo <= ln <= hi)
        bucket_data.append((f"{lo}–{hi}", count))
        max_count = max(max_count, count)

    bar_width = 30
    hist_table = Table(show_header=True)
    hist_table.add_column("Chars", style="bold", justify="right")
    hist_table.add_column("Count", justify="right")
    hist_table.add_column("", min_width=bar_width)
    for label, count in bucket_data:
        filled = int(count / max_count * bar_width) if max_count else 0
        hist_table.add_row(label, f"{count:,}", "█" * filled)
    console.print(hist_table)

    console.print()
    console.print(f"  Shortest: {shortest} chars")
    shortest_ex = next(s for s in sentences if len(s) == shortest)
    console.print(f"    {shortest_ex}")
    console.print(f"  Longest:  {longest} chars")
    longest_ex = next(s for s in sentences if len(s) == longest)
    console.print(f"    {longest_ex[:120]}{'...' if len(longest_ex) > 120 else ''}")
    console.print(f"  Average:  {avg:.0f} chars")


_REPORT_SKIP_NAMES = (
    "CJK UNIFIED IDEOGRAPH",
    "HIRAGANA",
    "KATAKANA",
    "YEN SIGN",
    "DIGIT",
    "THAI",
)


def _print_rare_characters(sentences: list[str]) -> None:
    from collections import Counter

    from rich.table import Table

    console.print()
    console.rule("[bold]Least Frequent Characters (bottom 40)[/bold]")

    char_counts: Counter[str] = Counter()
    for s in sentences:
        char_counts.update(set(s))

    filtered_counts = [
        (ch, n)
        for ch, n in char_counts.most_common()
        if not any(k in unicodedata.name(ch, "") for k in _REPORT_SKIP_NAMES)
    ]
    rare = filtered_counts[:-41:-1]

    rare_table = Table(show_header=True)
    rare_table.add_column("Char", justify="center")
    rare_table.add_column("Code", style="dim")
    rare_table.add_column("Name", style="dim")
    rare_table.add_column("Count", justify="right")
    rare_table.add_column("Example sentence")
    for ch, count in rare:
        name = unicodedata.name(ch, "?")
        example = next((s for s in sentences if ch in s), "")
        if len(example) > 60:
            idx = example.index(ch)
            start = max(0, idx - 25)
            end = min(len(example), idx + 25)
            example = (
                ("..." if start > 0 else "")
                + example[start:end]
                + ("..." if end < len(example) else "")
            )
        rare_table.add_row(ch, f"U+{ord(ch):04X}", name, f"{count:,}", example)
    console.print(rare_table)


def _print_selection_summary(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    total_cc: int,
    n_filtered: int,
    n_selected: int,
    cutoff: float,
    diversity: np.ndarray,
    unc: np.ndarray,
    out_path: Path,
) -> None:
    from rich.table import Table

    console.rule("Selection Summary")
    tbl = Table(show_header=False)
    tbl.add_column("Metric", style="bold")
    tbl.add_column("Value", justify="right")
    tbl.add_row("CC sentences (total)", f"{total_cc:,}")
    tbl.add_row("Filtered (low-grammatic)", f"{n_filtered:,}")
    tbl.add_row("Scored", f"{total_cc - n_filtered:,}")
    tbl.add_row("Selected", f"{n_selected:,}")
    tbl.add_row("Impact cutoff", f"{cutoff:.6f}")
    tbl.add_row("", "")
    tbl.add_row("Diversity (mean)", f"{diversity.mean():.4f}")
    tbl.add_row("Diversity (median)", f"{float(np.median(diversity)):.4f}")
    tbl.add_row("Uncertainty (mean)", f"{unc.mean():.4f}")
    tbl.add_row("Uncertainty (median)", f"{float(np.median(unc)):.4f}")
    tbl.add_row("", "")
    tbl.add_row("Output", str(out_path))
    console.print(tbl)

    console.print(
        f"\n  [bold]Impact cutoff: {cutoff:.6f}[/bold]"
        "  (reuse with --min-impact for future batches)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # pylint: disable=too-many-locals
    import torch

    from scripts.recon_bpd.inference import load_model_from_checkpoint

    parser = argparse.ArgumentParser(
        description="Extract and select Japanese sentences from Common Crawl."
    )
    parser.add_argument("--crawl", default=None, help="Crawl ID (default: latest).")
    parser.add_argument(
        "--min-length",
        type=int,
        default=MIN_LENGTH,
        help=f"Minimum sentence length in characters (default: {MIN_LENGTH}).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_LENGTH,
        help=f"Maximum sentence length in characters (default: {MAX_LENGTH}).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Reprocess all files from scratch, ignoring the manifest.",
    )
    parser.add_argument(
        "--top-pct",
        type=float,
        default=1.0,
        help="Select top N%% of sentences (default: 1.0)",
    )
    parser.add_argument(
        "--min-impact",
        type=float,
        default=None,
        help="Select all sentences with impact >= this value (overrides --top-pct)",
    )
    args = parser.parse_args()

    min_len = args.min_length
    max_len = args.max_length

    console.rule("[bold]Common Crawl — Extract & Select Japanese Sentences[/bold]")

    crawl_id = args.crawl or get_latest_crawl_id()
    perf_start_run("extract-sentences", crawl=crawl_id)

    _t0_bloom = time.monotonic()
    bloom = _load_or_build_bloom()
    perf_log("bloom_filter", time_s=time.monotonic() - _t0_bloom)

    info = get_crawl_info(crawl_id)
    console.print(f"  Crawl: [bold]{crawl_id}[/bold]  ({info['name']})")

    jp_dir = _wet_jp_dir(crawl_id)
    all_jsonl = sorted(jp_dir.glob("*.jsonl.gz"))
    if not all_jsonl:
        console.print(
            "[red]No fetched WET files found. Run 'cc fetch-jp-wet' first.[/red]"
        )
        perf_log("early_exit", reason="no_wet_files")
        perf_flush()
        return

    if args.rebuild:
        already_processed: set[str] = set()
        existing_sentences: list[str] = []
        seen: set[str] = set()
    else:
        already_processed = _load_manifest(crawl_id)
        existing_sentences, seen = _load_existing_sentences(crawl_id)

    new_files = [f for f in all_jsonl if f.name not in already_processed]

    console.print(f"  JSONL files: {len(all_jsonl)} total, {len(new_files)} new")
    console.print(f"  Existing sentences: {len(existing_sentences):,}")
    console.print(f"  Sentence length: {min_len}–{max_len} chars")

    _t0_extract = time.monotonic()
    # -- Extraction phase (skip if nothing new) --
    if new_files:
        console.print()
        stats = _run_extraction(new_files, min_len, max_len, seen, already_processed)

        new_batch = [clean_sentence(s) for s in stats["new_sentences"]]
        new_batch = [s for s in new_batch if content_ok(s)]
        new_batch, corpus_dupes = _corpus_dedup(new_batch, bloom)
        merged = list(dict.fromkeys(existing_sentences + new_batch))

        out_path = _sentences_path(crawl_id)
        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            for s in merged:
                f.write(s + "\n")

        _save_manifest(crawl_id, already_processed)

        _print_extraction_summary(
            new_files=len(new_files),
            new_pages=stats["pages"],
            new_candidates=stats["candidates"],
            new_extracted=stats["extracted"],
            new_unique=len(stats["new_sentences"]),
            corpus_dupes=corpus_dupes,
            total=len(merged),
            out_path=out_path,
        )

        if merged:
            _print_length_histogram(merged)
            _print_rare_characters(merged)
    else:
        console.print("\n  [green]No new files to extract.[/green]")

    perf_log(
        "extraction", new_files=len(new_files), time_s=time.monotonic() - _t0_extract
    )

    # -- Load all sentences for selection --
    sentences_path = _sentences_path(crawl_id)
    if not sentences_path.exists():
        console.print("[red]No sentences.txt.gz found.[/red]")
        perf_log("early_exit", reason="no_sentences_gz")
        perf_flush()
        return

    with gzip.open(sentences_path, "rt", encoding="utf-8") as fh:
        cc_sentences = [ln.rstrip("\n") for ln in fh]

    console.print(f"\n  Total CC sentences: {len(cc_sentences):,}")

    # -- Selection phase --
    console.rule("Scoring & Selection")

    model, tokenizer_path, checkpoint_id = load_model_from_checkpoint(
        drop_layers=7,
        output_rank=16,
    )
    set_tokenizer_path(tokenizer_path)
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    model.to(device)

    model_md5 = model_hash()
    if getattr(model, "_distilled", False):
        variant = "fp16 -7L r16"
    else:
        variant = "full fp32"
    console.print(f"  Checkpoint: {checkpoint_id}  ({variant}, cache key: {model_md5})")

    _t0_scoring = time.monotonic()
    embed_store = EmbedStore(model_md5, d_model=model.cfg.d_model)
    console.print(f"  Shared embedding store: {embed_store.count:,} sentences")

    corpus_emb = get_corpus_embeddings(
        model, device, model_md5, embed_store=embed_store
    )
    console.print(
        f"  Corpus embeddings: {corpus_emb.shape[0]:,} x {corpus_emb.shape[1]}"
    )

    cc_emb, cc_uncertainty, cc_gram_probs = get_cc_scores(
        crawl_id, cc_sentences, model, device, model_md5,
        embed_store=embed_store,
    )

    # Grammaticality filter -- currently a no-op (all gram_probs=1.0) because
    # the recon_bpd model has no grammaticality head yet.  When one is added,
    # this gate will start filtering again automatically.
    keep_mask = cc_gram_probs >= GRAMMATIC_SOFT_MIN
    keep_idx = np.where(keep_mask)[0]
    n_gram = int((~keep_mask).sum())
    console.print(f"  Filtered: {n_gram:,} low-grammatic")
    console.print(
        f"  Candidates after filtering: {len(keep_idx):,} / {len(cc_sentences):,}"
    )

    if len(keep_idx) == 0:
        console.print("[red]No sentences survived filtering.[/red]")
        perf_log("early_exit", reason="all_filtered")
        perf_flush()
        return

    perf_log(
        "model_scoring",
        n_cc=len(cc_sentences),
        n_corpus=corpus_emb.shape[0],
        time_s=time.monotonic() - _t0_scoring,
    )

    # -- Score filtered subset --
    _t0_sel = time.monotonic()
    all_diversity = _get_diversity(crawl_id, cc_emb, corpus_emb, model_md5, device)
    diversity = all_diversity[keep_idx]
    div_pct = _rank_percentiles(diversity)
    unc_pct = _rank_percentiles(cc_uncertainty[keep_idx])
    impact = div_pct * unc_pct**2

    perf_log_dist("diversity_all", all_diversity)
    perf_log_dist("diversity_kept", diversity)
    perf_log_dist("uncertainty_kept", cc_uncertainty[keep_idx])
    perf_log_dist("gram_probs", cc_gram_probs)
    perf_log_dist("impact", impact)

    # -- Select --
    sel_local, cutoff = _select(impact, args.top_pct, args.min_impact)
    sel_global = keep_idx[sel_local]
    selected_sentences = [cc_sentences[i] for i in sel_global]

    # -- Write selected output --
    sel_path = Path(".cc/selected-sentences.txt")
    with open(sel_path, "w", encoding="utf-8") as fh:
        for sent in selected_sentences:
            fh.write(sent + "\n")

    perf_log_dist("sel_diversity", diversity[sel_local], indent=1)
    perf_log_dist("sel_uncertainty", cc_uncertainty[sel_global], indent=1)
    perf_log_dist("sel_impact", impact[sel_local], indent=1)
    perf_log(
        "selection",
        n_selected=len(selected_sentences),
        n_candidates=len(keep_idx),
        time_s=time.monotonic() - _t0_sel,
    )

    _print_selection_summary(
        total_cc=len(cc_sentences),
        n_filtered=len(cc_sentences) - len(keep_idx),
        n_selected=len(selected_sentences),
        cutoff=cutoff,
        diversity=diversity[sel_local],
        unc=cc_uncertainty[sel_global],
        out_path=sel_path,
    )

    perf_flush()


if __name__ == "__main__":
    main()
