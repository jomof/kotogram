"""Durable sharded cache for Japanese → kotogram + label conversions.

This module provides a sharded SQLite-based cache to store processing results.
It's designed to be shared between labeling scripts and training scripts.
"""

import hashlib
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from kotogram import locations

CacheEntryType = Tuple[
    str,
    str,
    Optional[int],
    Optional[float],
    Optional[int],
    Optional[List[int]],
    Optional[int],
    Optional[Dict[str, List[int]]],
]


class ShardedKotogramCache:
    """Durable sharded cache for Japanese → kotogram + label conversions."""

    SHARD_PREFIX_LEN = 2  # 2 hex chars = 256 shards

    def __init__(self) -> None:
        self.shards_dir = locations.get_shards_cache_dir()
        os.makedirs(self.shards_dir, exist_ok=True)

    def _get_shard_path(self, sentence_hash: str) -> str:
        shard_key = sentence_hash[: self.SHARD_PREFIX_LEN]
        return os.path.join(self.shards_dir, f"{shard_key}.db")

    def _init_shard(self, shard_path: str) -> None:
        conn = sqlite3.connect(shard_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    sentence_hash TEXT PRIMARY KEY,
                    sentence TEXT NOT NULL,
                    kotogram TEXT NOT NULL,
                    formality_label INTEGER,
                    gender_value REAL,
                    gender_pragmatic INTEGER,
                    register_labels TEXT,
                    grammaticality_label INTEGER,
                    feature_ids TEXT
                )
            """)
            # Check for column existence to avoid OperationalError on duplicate column
            cursor = conn.execute("PRAGMA table_info(cache_entries)")
            columns = [info[1] for info in cursor.fetchall()]
            if "feature_ids" not in columns:
                conn.execute("ALTER TABLE cache_entries ADD COLUMN feature_ids TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_hash ON cache_entries(sentence_hash)"
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _parse_cache_row(
        k: str,
        f_lbl: Optional[int],
        g_val: Optional[float],
        g_prag: Optional[int],
        r_lbls_json: Optional[str],
        gram_lbl: Optional[int],
        f_ids_json: Optional[str],
    ) -> Tuple[
        str,
        Optional[int],
        Optional[float],
        Optional[int],
        Optional[List[int]],
        Optional[int],
        Optional[Dict[str, List[int]]],
    ]:
        """Helper to parse a row from the cache."""
        # pylint: disable=too-many-positional-arguments
        r_lbls = json.loads(r_lbls_json) if r_lbls_json else None
        f_ids = json.loads(f_ids_json) if f_ids_json else None
        return (k, f_lbl, g_val, g_prag, r_lbls, gram_lbl, f_ids)

    @staticmethod
    def _prepare_cache_row(
        s: str,
        k: str,
        f_lbl: Optional[int],
        g_val: Optional[float],
        g_prag: Optional[int],
        r_lbls: Optional[List[int]],
        gram_lbl: Optional[int],
        f_ids: Optional[Dict[str, List[int]]],
    ) -> Tuple:
        """Helper to prepare a row for insertion."""
        # pylint: disable=too-many-positional-arguments
        r_lbls_json = json.dumps(r_lbls) if r_lbls is not None else None
        f_ids_json = json.dumps(f_ids) if f_ids is not None else None
        return (
            ShardedKotogramCache._hash_sentence(s),
            s,
            k,
            f_lbl,
            g_val,
            g_prag,
            r_lbls_json,
            gram_lbl,
            f_ids_json,
        )

    @staticmethod
    def _hash_sentence(sentence: str) -> str:
        return hashlib.sha256(sentence.encode("utf-8")).hexdigest()

    def get_batch(
        self, sentences: List[str]
    ) -> Dict[
        str,
        Optional[
            Tuple[
                str,
                Optional[int],
                Optional[float],
                Optional[int],
                Optional[List[int]],
                Optional[int],
                Optional[Dict[str, List[int]]],
            ]
        ],
    ]:
        # pylint: disable=too-many-locals
        if not sentences:
            return {}

        shard_to_hashes: Dict[str, List[Tuple[str, str]]] = {}
        results: Dict[str, Any] = {s: None for s in sentences}

        for s in sentences:
            h = self._hash_sentence(s)
            path = self._get_shard_path(h)
            if path not in shard_to_hashes:
                shard_to_hashes[path] = []
            shard_to_hashes[path].append((h, s))

        for shard_path, items in shard_to_hashes.items():
            if not os.path.exists(shard_path):
                continue

            hash_to_sentence = dict(items)
            hashes = list(hash_to_sentence.keys())

            conn = sqlite3.connect(shard_path)
            placeholders = ",".join("?" * len(hashes))
            cursor = conn.execute(
                f"SELECT sentence_hash, kotogram, formality_label, gender_value, gender_pragmatic, register_labels, grammaticality_label, feature_ids FROM cache_entries WHERE sentence_hash IN ({placeholders})",
                hashes,
            )

            for row in cursor:
                h, k, f_lbl, g_val, g_prag, r_lbls_json, gram_lbl, f_ids_json = row
                results[hash_to_sentence[h]] = self._parse_cache_row(
                    k, f_lbl, g_val, g_prag, r_lbls_json, gram_lbl, f_ids_json
                )
            conn.close()

        return results

    def put_batch(
        self,
        items: List[CacheEntryType],
    ) -> None:
        # pylint: disable=too-many-locals
        if not items:
            return

        shard_to_data: Dict[str, List[Tuple]] = {}

        for s, k, f_lbl, g_val, g_prag, r_lbls, gram_lbl, f_ids in items:
            h = self._hash_sentence(s)
            path = self._get_shard_path(h)
            if path not in shard_to_data:
                shard_to_data[path] = []

            shard_to_data[path].append(
                self._prepare_cache_row(
                    s, k, f_lbl, g_val, g_prag, r_lbls, gram_lbl, f_ids
                )
            )

        for shard_path, data in shard_to_data.items():
            self._init_shard(shard_path)
            conn = sqlite3.connect(shard_path)
            conn.executemany(
                """INSERT OR REPLACE INTO cache_entries 
                   (sentence_hash, sentence, kotogram, formality_label, gender_value, gender_pragmatic, register_labels, grammaticality_label, feature_ids) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                data,
            )
            conn.commit()
            conn.close()


_KOTOGRAM_CACHE: Optional[ShardedKotogramCache] = None


def get_kotogram_cache() -> ShardedKotogramCache:
    # pylint: disable=global-statement
    global _KOTOGRAM_CACHE
    expected_shards_dir = locations.get_shards_cache_dir()
    if _KOTOGRAM_CACHE is None or _KOTOGRAM_CACHE.shards_dir != expected_shards_dir:
        _KOTOGRAM_CACHE = ShardedKotogramCache()
    return _KOTOGRAM_CACHE
