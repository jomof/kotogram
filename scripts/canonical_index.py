"""Reusable canonical-form dedup index backed by bloom filter + SQLite side table.

Accepts either a SQLite database (.db) or text file (.txt / .txt.gz) as its
sentence source.  Caches a bloom filter (keyed on canonical forms) and a SQLite
side table for bloom-hit confirmation.

Usage::

    # Well-known corpus.db index (shared by CC extraction, upsert, dataset build)
    idx = CanonicalIndex.corpus().load_or_build()

    if idx.might_contain(sentence):
        existing = idx.get_existing(sentence)
        if existing:
            ...  # confirmed duplicate
"""

from __future__ import annotations

import gzip
import hashlib
import math
import multiprocessing as mp
import sqlite3
import struct
from pathlib import Path
from typing import Iterator

from rich.console import Console

from kotogram.masking import canonicalize_sentence

console = Console()

CORPUS_DB = Path("data/corpus.db")
_DEFAULT_FPR = 0.001
_CANON_WORKERS = max(1, mp.cpu_count() - 1)
_CANON_CHUNK = 5_000


# ---------------------------------------------------------------------------
# Bloom filter (self-contained, no external deps)
# ---------------------------------------------------------------------------


class BloomFilter:
    """Simple bloom filter backed by a bytearray."""

    def __init__(self, capacity: int, fpr: float = _DEFAULT_FPR):
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


# ---------------------------------------------------------------------------
# Parallel canonicalization helper
# ---------------------------------------------------------------------------

_worker_parser = None  # pylint: disable=invalid-name  # module-level for pickling across processes


def _canon_worker_init() -> None:
    """Each pool worker creates its own SudachiPy parser once."""
    global _worker_parser  # noqa: PLW0603  # pylint: disable=global-statement
    from kotogram.sudachi_japanese_parser import SudachiJapaneseParser

    _worker_parser = SudachiJapaneseParser(validate=False)


def _canon_worker(sentences: list[str]) -> list[str]:
    return [canonicalize_sentence(s, _parser=_worker_parser) for s in sentences]


def parallel_canonicalize(sentences: list[str]) -> list[str]:
    """Canonicalize a list of sentences using a process pool."""
    if len(sentences) <= _CANON_CHUNK:
        from kotogram.sudachi_japanese_parser import SudachiJapaneseParser

        parser = SudachiJapaneseParser(validate=False)
        return [canonicalize_sentence(s, _parser=parser) for s in sentences]

    chunks = [
        sentences[i : i + _CANON_CHUNK] for i in range(0, len(sentences), _CANON_CHUNK)
    ]
    with mp.Pool(_CANON_WORKERS, initializer=_canon_worker_init) as pool:
        results = pool.map(_canon_worker, chunks)
    return [s for batch in results for s in batch]


# ---------------------------------------------------------------------------
# CanonicalIndex
# ---------------------------------------------------------------------------


class CanonicalIndex:
    """Canonical-form dedup index: bloom filter + SQLite confirmation table.

    Parameters
    ----------
    source : Path
        Sentence source -- either a ``.db`` (SQLite with a ``sentences`` table)
        or a ``.txt`` / ``.txt.gz`` file (one sentence per line).
    cache_dir : Path
        Directory for the cached bloom and side-table files.
    prefix : str
        Filename prefix for cache files (e.g. ``"corpus-canonical"``).
    """

    def __init__(self, source: Path, cache_dir: Path, prefix: str) -> None:
        self.source = source
        self.cache_dir = cache_dir
        self._bloom_path = cache_dir / f"{prefix}-bloom.bin"
        self._side_path = cache_dir / f"{prefix}-index.db"
        self._is_db_source = source.suffix == ".db"
        self._bloom: BloomFilter | None = None
        self._side_conn: sqlite3.Connection | None = None
        self._lazy_parser: object | None = None
        self._dirty = False

    # -- Factory methods ---------------------------------------------------

    @classmethod
    def corpus(cls) -> CanonicalIndex:
        """Well-known shared index for corpus.db."""
        return cls(
            source=CORPUS_DB,
            cache_dir=Path(".cc"),
            prefix="corpus-canonical",
        )

    # -- Lifecycle ---------------------------------------------------------

    def load_or_build(self) -> CanonicalIndex:
        """Load cached bloom + side table if fresh, else rebuild from source."""
        if self._is_fresh():
            self._bloom = BloomFilter.load(self._bloom_path)
            self._open_side_table()
            console.print(
                f"  Canonical index loaded: {self._bloom.count:,} entries "
                f"(source: {self.source.name})"
            )
        else:
            self._build()
        return self

    def _is_fresh(self) -> bool:
        if not self._bloom_path.exists() or not self._side_path.exists():
            return False
        if not self.source.exists():
            return False
        return self._bloom_path.stat().st_mtime >= self.source.stat().st_mtime

    def _build(self) -> None:
        """Full rebuild from source."""
        console.print(f"  Building canonical index from {self.source.name}...")
        sentences = list(self._iter_source())
        n = len(sentences)
        console.print(f"    {n:,} source sentences, canonicalizing...")

        canonical = parallel_canonicalize(sentences)

        self._bloom = BloomFilter(max(n, 1))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self._side_path.exists():
            self._side_path.unlink()
        self._open_side_table()
        assert self._side_conn is not None

        cur = self._side_conn.cursor()
        cur.execute("BEGIN")
        for orig, canon in zip(sentences, canonical):
            self._bloom.add(canon)
            if not self._is_db_source or canon != orig:
                cur.execute(
                    "INSERT OR IGNORE INTO canon_map (canonical, original) VALUES (?, ?)",
                    (canon, orig),
                )
        self._side_conn.commit()
        self._bloom.save(self._bloom_path)
        console.print(f"    Canonical index built: {self._bloom.count:,} entries")

    def _open_side_table(self) -> None:
        self._side_conn = sqlite3.connect(str(self._side_path))
        self._side_conn.execute(
            "CREATE TABLE IF NOT EXISTS canon_map "
            "(canonical TEXT NOT NULL, original TEXT NOT NULL)"
        )
        self._side_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_canon ON canon_map (canonical)"
        )

    def _iter_source(self) -> Iterator[str]:
        if self._is_db_source:
            conn = sqlite3.connect(str(self.source))
            cur = conn.execute("SELECT sentence FROM sentences")
            while True:
                rows = cur.fetchmany(10_000)
                if not rows:
                    break
                for (s,) in rows:
                    yield s
            conn.close()
        elif self.source.suffix == ".gz":
            with gzip.open(self.source, "rt", encoding="utf-8") as f:
                for line in f:
                    s = line.rstrip("\n")
                    if s:
                        yield s
        else:
            with open(self.source, encoding="utf-8") as f:
                for line in f:
                    s = line.rstrip("\n")
                    if s:
                        yield s

    # -- Lazy parser for per-call canonicalization ---------------------------

    @property
    def _parser(self) -> object:
        if self._lazy_parser is None:
            from kotogram.sudachi_japanese_parser import SudachiJapaneseParser

            self._lazy_parser = SudachiJapaneseParser(validate=False)
        return self._lazy_parser

    # -- Query API ---------------------------------------------------------

    def might_contain(self, sentence: str) -> bool:
        """Canonicalize *sentence* and check the bloom filter."""
        assert self._bloom is not None, "call load_or_build() first"
        canon = canonicalize_sentence(sentence, _parser=self._parser)
        return self._bloom.might_contain(canon)

    def get_existing(self, sentence: str) -> list[str] | None:
        """Confirm a bloom hit and return the existing original sentence(s).

        Returns None on bloom false-positive (no actual match found).
        """
        assert self._bloom is not None and self._side_conn is not None
        canon = canonicalize_sentence(sentence, _parser=self._parser)
        if not self._bloom.might_contain(canon):
            return None

        # Check side table
        rows = self._side_conn.execute(
            "SELECT original FROM canon_map WHERE canonical = ?", (canon,)
        ).fetchall()
        if rows:
            return [r[0] for r in rows]

        # For .db sources: canonical == original entries aren't in the side table
        if self._is_db_source:
            conn = sqlite3.connect(str(self.source))
            row = conn.execute(
                "SELECT sentence FROM sentences WHERE sentence = ? LIMIT 1",
                (canon,),
            ).fetchone()
            conn.close()
            if row:
                return [row[0]]

        return None

    def batch_might_contain(self, sentences: list[str]) -> tuple[list[bool], list[str]]:
        """Batch-canonicalize *sentences* in parallel, then check the bloom filter.

        Returns (mask, canonicals) where mask[i] is True if sentences[i]
        might be in the index.
        """
        assert self._bloom is not None, "call load_or_build() first"
        if sentences:
            console.print(
                f"  Canonicalizing {len(sentences):,} sentences for bloom check..."
            )
        canonicals = parallel_canonicalize(sentences)
        mask = [self._bloom.might_contain(c) for c in canonicals]
        return mask, canonicals

    def filter_duplicates(self, sentences: list[str]) -> tuple[list[str], int]:
        """Remove sentences whose canonical form already exists in the index.

        Batch-canonicalizes in parallel, then confirms bloom hits against
        the side table.  Returns (kept_sentences, num_removed).
        """
        assert self._bloom is not None and self._side_conn is not None
        if not sentences:
            return [], 0

        canonicals = parallel_canonicalize(sentences)

        kept: list[str] = []
        removed = 0
        for s, canon in zip(sentences, canonicals):
            if not self._bloom.might_contain(canon):
                kept.append(s)
                continue
            # Confirm via side table / DB
            rows = self._side_conn.execute(
                "SELECT 1 FROM canon_map WHERE canonical = ? LIMIT 1",
                (canon,),
            ).fetchone()
            if rows:
                removed += 1
                continue
            if self._is_db_source:
                conn = sqlite3.connect(str(self.source))
                row = conn.execute(
                    "SELECT 1 FROM sentences WHERE sentence = ? LIMIT 1",
                    (canon,),
                ).fetchone()
                conn.close()
                if row:
                    removed += 1
                    continue
            kept.append(s)
        return kept, removed

    # -- Incremental update API --------------------------------------------

    def add(self, original_sentence: str) -> None:
        """Incrementally register a new sentence (call after a successful insert)."""
        assert self._bloom is not None and self._side_conn is not None
        canon = canonicalize_sentence(original_sentence, _parser=self._parser)
        self._bloom.add(canon)
        if not self._is_db_source or canon != original_sentence:
            self._side_conn.execute(
                "INSERT OR IGNORE INTO canon_map (canonical, original) VALUES (?, ?)",
                (canon, original_sentence),
            )
        self._dirty = True

    def save(self) -> None:
        """Persist incremental updates to disk."""
        if not self._dirty:
            return
        assert self._bloom is not None and self._side_conn is not None
        self._bloom.save(self._bloom_path)
        self._side_conn.commit()
        self._dirty = False

    def close(self) -> None:
        """Close the side-table connection."""
        if self._side_conn:
            self._side_conn.close()
            self._side_conn = None
