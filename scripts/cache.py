"""Durable sharded cache for Japanese → kotogram + label conversions.

This module provides a sharded SQLite-based cache to store processing results.
It's designed to be shared between labeling scripts and training scripts.
"""

import os
import sqlite3
import hashlib
import json
from typing import Dict, List, Optional, Tuple, cast




class ShardedKotogramCache:
    """Durable sharded cache for Japanese → kotogram + label conversions.

    This cache stores processing results in multiple small SQLite databases (shards)
    to keep file sizes manageable (~1MB) and avoid lock contention.
    
    It accepts a legacy monolithic database path for migration purposes.
    
    Keyed by sentence hash.
    Schema: (sentence, kotogram, formality_label, gender_label, gender_pragmatic, register_labels)
    """

    DEFAULT_SHARDS_DIR = ".cache/kotogram_shards"
    LEGACY_DB_PATH = ".cache/kotogram.db"
    SHARD_PREFIX_LEN = 3 # 3 hex chars = 4096 shards

    def __init__(self, shards_dir: str = DEFAULT_SHARDS_DIR):
        """Initialize the sharded cache.
        
        Args:
            shards_dir: Directory to store shard database files
        """
        self.shards_dir = shards_dir
        os.makedirs(shards_dir, exist_ok=True)
        
        # Check for legacy DB and migrate if needed
        if os.path.exists(self.LEGACY_DB_PATH):
            print(f"Found legacy cache at {self.LEGACY_DB_PATH}. Migrating to shards...")
            self._migrate_legacy_cache(self.LEGACY_DB_PATH)

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
                    register_labels TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON cache_entries(sentence_hash)")
            conn.commit()
        finally:
            conn.close()

    def _migrate_legacy_cache(self, legacy_path: str) -> None:
        """Migrate entries from legacy monolithic DB to shards."""
        try:
            conn = sqlite3.connect(legacy_path)
            cursor = conn.execute("SELECT sentence, kotogram FROM kotogram_cache")
            
            # Batch read to avoid memory issues (though 1.2GB might fit in RAM, safer to stream)
            count = 0
            while True:
                rows = cursor.fetchmany(10000)
                if not rows:
                    break
                
                # Convert to format expected by put_batch (sentence, kotogram, formality, gender, register)
                # Legacy cache has no labels, so None
                items = [(r[0], r[1], None, None, None, None) for r in rows]
                self.put_batch(cast(List[Tuple[str, str, Optional[int], Optional[float], Optional[int], Optional[List[int]]]], items), verbose=False) # Verbose False to avoid spam
                count += len(rows)
                print(f"  Migrated {count} entries...")
            
            conn.close()
            
            # Rename legacy file to prevent re-migration
            os.rename(legacy_path, legacy_path + ".bak")
            print(f"Migration complete. Legacy cache moved to {legacy_path}.bak")
            
        except sqlite3.Error as e:
            print(f"Error migrating legacy cache: {e}")
            # Don't crash, just continue with empty shards

    @staticmethod
    def _hash_sentence(sentence: str) -> str:
        """Create a hash key for a sentence."""
        return hashlib.sha256(sentence.encode('utf-8')).hexdigest()

    def get_batch(self, sentences: List[str]) -> Dict[str, Optional[Tuple[str, Optional[int], Optional[float], Optional[int], Optional[List[int]]]]]:
        """Get cached entries for multiple sentences.

        Returns:
            Dict mapping sentence → (kotogram, formality, gender_val, gender_prag, register) OR None
        """
        if not sentences:
            return {}

        # Group by shard
        shard_to_hashes: Dict[str, List[Tuple[str, str]]] = {} # shard_path -> [(hash, sentence)]
        results: Dict[str, Optional[Tuple[str, Optional[int], Optional[float], Optional[int], Optional[List[int]]]]] = {s: None for s in sentences}
        
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
            cursor = conn.execute(
                f"SELECT sentence_hash, kotogram, formality_label, gender_value, gender_pragmatic, register_labels FROM cache_entries WHERE sentence_hash IN ({placeholders})",
                hashes
            )
            
            for row in cursor:
                if len(row) == 6:
                        h, k, f_lbl, g_val, g_prag, r_lbls_json = row
                        r_lbls = json.loads(r_lbls_json) if r_lbls_json else None
                else:
                        continue # Should not happen with current schema
                
                sent = hash_to_sentence[h]
                results[sent] = (k, f_lbl, g_val, g_prag, r_lbls)
            conn.close()

        return results

    def put_batch(
        self, 
        items: List[Tuple[str, str, Optional[int], Optional[float], Optional[int], Optional[List[int]]]],
        verbose: bool = False
    ) -> None:
        """Cache multiple entries.
        
        Args:
            items: List of (sentence, kotogram, formality_label, gender_value, gender_pragmatic, register_labels)
        """
        if not items:
            return

        # Group by shard
        shard_to_data: Dict[str, List[Tuple[str, str, str, Optional[int], Optional[float], Optional[int], Optional[str]]]] = {}
        
        for s, k, f_lbl, g_val, g_prag, r_lbls in items:
            h = self._hash_sentence(s)
            path = self._get_shard_path(h)
            if path not in shard_to_data:
                shard_to_data[path] = []
            
            r_lbls_json = json.dumps(r_lbls) if r_lbls is not None else None
            shard_to_data[path].append((h, s, k, f_lbl, g_val, g_prag, r_lbls_json))

        # Write to each shard
        for shard_path, data in shard_to_data.items():
            self._init_shard(shard_path) # Ensure exists
            
            conn = sqlite3.connect(shard_path)
            conn.executemany(
                """INSERT OR REPLACE INTO cache_entries 
                   (sentence_hash, sentence, kotogram, formality_label, gender_value, gender_pragmatic, register_labels) 
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
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
