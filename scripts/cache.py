"""Durable sharded cache for Japanese → kotogram + label conversions.

This module provides a sharded SQLite-based cache to store processing results.
It's designed to be shared between labeling scripts and training scripts.
"""

import os
import sqlite3
import hashlib
import json
from typing import Dict, List, Optional, Tuple




class ShardedKotogramCache:
    """Durable sharded cache for Japanese → kotogram + label conversions.

    This cache stores processing results in multiple small SQLite databases (shards)
    to keep file sizes manageable (~1MB) and avoid lock contention.
    
    Keyed by sentence hash.
    Schema: (sentence, kotogram, formality_label, gender_value, gender_pragmatic, register_labels, grammaticality_label)
    """

    DEFAULT_SHARDS_DIR = ".cache/kotogram_shards"
    SHARD_PREFIX_LEN = 3 # 3 hex chars = 4096 shards

    def __init__(self, shards_dir: str = DEFAULT_SHARDS_DIR):
        """Initialize the sharded cache.
        
        Args:
            shards_dir: Directory to store shard database files
        """
        self.shards_dir = shards_dir
        os.makedirs(shards_dir, exist_ok=True)

    def _get_shard_path(self, sentence_hash: str) -> str:
        """Get path to the shard file for a given hash."""
        shard_key = sentence_hash[:self.SHARD_PREFIX_LEN]
        return os.path.join(self.shards_dir, f"{shard_key}.db")

    def _init_shard(self, shard_path: str) -> None:
        """Initialize a single shard database (if not exists)."""
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
                    grammaticality_label INTEGER
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON cache_entries(sentence_hash)")
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _hash_sentence(sentence: str) -> str:
        """Create a hash key for a sentence."""
        return hashlib.sha256(sentence.encode('utf-8')).hexdigest()

    def get_batch(self, sentences: List[str]) -> Dict[str, Optional[Tuple[str, Optional[int], Optional[float], Optional[int], Optional[List[int]], Optional[int]]]]:
        """Get cached entries for multiple sentences.

        Returns:
            Dict mapping sentence → (kotogram, formality, gender_val, gender_prag, register, gram_label) OR None
        """
        if not sentences:
            return {}

        # Group by shard
        shard_to_hashes: Dict[str, List[Tuple[str, str]]] = {} # shard_path -> [(hash, sentence)]
        results: Dict[str, Optional[Tuple[str, Optional[int], Optional[float], Optional[int], Optional[List[int]], Optional[int]]]] = {s: None for s in sentences}
        
        for s in sentences:
            h = self._hash_sentence(s)
            path = self._get_shard_path(h)
            if path not in shard_to_hashes:
                shard_to_hashes[path] = []
            shard_to_hashes[path].append((h, s))

        # Query each shard
        for shard_path, items in shard_to_hashes.items():
            if not os.path.exists(shard_path):
                continue
                
            hash_to_sentence = {h: s for h, s in items}
            hashes = list(hash_to_sentence.keys())
            
            conn = sqlite3.connect(shard_path)
            placeholders = ",".join("?" * len(hashes))
            # We select all 7 columns (excluding sentence_hash which we already know)
            cursor = conn.execute(
                f"SELECT sentence_hash, kotogram, formality_label, gender_value, gender_pragmatic, register_labels, grammaticality_label FROM cache_entries WHERE sentence_hash IN ({placeholders})",
                hashes
            )
            
            for row in cursor:
                h, k, f_lbl, g_val, g_prag, r_lbls_json, gram_lbl = row
                r_lbls = json.loads(r_lbls_json) if r_lbls_json else None
                
                sent = hash_to_sentence[h]
                results[sent] = (k, f_lbl, g_val, g_prag, r_lbls, gram_lbl)
            conn.close()

        return results

    def put_batch(
        self, 
        items: List[Tuple[str, str, Optional[int], Optional[float], Optional[int], Optional[List[int]], Optional[int]]],
        verbose: bool = False
    ) -> None:
        """Cache multiple entries.
        
        Args:
            items: List of (sentence, kotogram, formality_label, gender_value, gender_pragmatic, register_labels, grammaticality_label)
        """
        if not items:
            return

        # Group by shard
        shard_to_data: Dict[str, List[Tuple[str, str, str, Optional[int], Optional[float], Optional[int], Optional[str], Optional[int]]]] = {}
        
        for s, k, f_lbl, g_val, g_prag, r_lbls, gram_lbl in items:
            h = self._hash_sentence(s)
            path = self._get_shard_path(h)
            if path not in shard_to_data:
                shard_to_data[path] = []
            
            r_lbls_json = json.dumps(r_lbls) if r_lbls is not None else None
            shard_to_data[path].append((h, s, k, f_lbl, g_val, g_prag, r_lbls_json, gram_lbl))

        # Write to each shard
        for shard_path, data in shard_to_data.items():
            self._init_shard(shard_path) # Ensure exists
            
            conn = sqlite3.connect(shard_path)
            conn.executemany(
                """INSERT OR REPLACE INTO cache_entries 
                   (sentence_hash, sentence, kotogram, formality_label, gender_value, gender_pragmatic, register_labels, grammaticality_label) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                data
            )
            conn.commit()
            conn.close()

    def __len__(self) -> int:
        """Return approximate number of cached entries (expensive to count all)."""
        total = 0
        if not os.path.exists(self.shards_dir):
            return 0
        for fname in os.listdir(self.shards_dir):
            if fname.endswith(".db"):
                conn = sqlite3.connect(os.path.join(self.shards_dir, fname))
                cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                total += int(cursor.fetchone()[0])
                conn.close()
        return total

_kotogram_cache: Optional[ShardedKotogramCache] = None

def get_kotogram_cache(shards_dir: Optional[str] = None) -> ShardedKotogramCache:
    """Get the global sharded kotogram cache instance."""
    global _kotogram_cache
    if shards_dir is None:
        shards_dir = ShardedKotogramCache.DEFAULT_SHARDS_DIR
    if _kotogram_cache is None or _kotogram_cache.shards_dir != shards_dir:
        _kotogram_cache = ShardedKotogramCache(shards_dir)
    return _kotogram_cache
