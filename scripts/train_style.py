"""Supervised style classifier for Japanese sentences using Kotogram representations.

This module provides a neural sequence classifier that predicts both formality and
gender-associated speech style labels for Japanese sentences based on their Kotogram
representation. It uses a pretrain-then-finetune approach with a small transformer encoder
and multi-task classification heads.

Architecture:
- Token Embedding: Multi-field embeddings for morphological features (pos, pos_detail1,
  pos_detail2, conjugated_type, conjugated_form, lemma) are concatenated and projected
  to d_model.
- Encoder: Small transformer encoder (2-4 layers) with multi-head self-attention.
- Pretraining: Multi-field Masked Language Modeling (MLM) that predicts all morphological
  features at masked positions, not just POS tags. This provides richer supervision.
- Fine-tuning: Sentence-level multi-task classification using [CLS] token representation,
  with separate heads for formality and gender prediction.

Pipeline:
1. Load Japanese sentences from TSV corpus (unlabeled for pretraining)
2. Convert sentences to Kotogram strings using japanese_to_kotogram()
3. Extract token features using extract_token_features()
4. Build vocabulary for each categorical field
5. Pretrain encoder with multi-field MLM on unlabeled data
6. Reinitialize classifier heads, then fine-tune with formality and gender labels

Usage:
    from kotogram.style_classifier import (
        StyleDataset, Tokenizer, StyleClassifier,
        StyleClassifierWithMLM, MLMTrainer, Trainer, predict_style
    )

    # Build vocabulary with unlabeled data
    tokenizer = Tokenizer()
    unlabeled = StyleDataset.from_tsv("data/sentences.tsv", tokenizer, labeled=False)

    # Pretrain with multi-field MLM
    model = StyleClassifierWithMLM(tokenizer.get_model_config())
    mlm_trainer = MLMTrainer(model, unlabeled)
    mlm_trainer.train(epochs=5)

    # Reset classifier and load labeled data
    model.reset_classifier()
    labeled = StyleDataset.from_tsv("data/sentences.tsv", tokenizer, labeled=True)
    train_data, val_data, test_data = labeled.split()

    # Fine-tune for classification
    trainer = Trainer(model, train_data, val_data)
    trainer.train(epochs=10)

    # Inference
    formality_label, gender_label, probs = predict_style("何かしてみましょう。", model, tokenizer)
"""

import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import sqlite3
import sys
import time  # For timing
import yaml  # For timing output

# Start timing immediately
script_start_time = time.time()
timings = {}

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, cast

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


import sudachipy
import sudachidict_full

from kotogram.kotogram import split_kotogram
from kotogram.analysis import FormalityLevel, GenderLevel
from kotogram.kotogram import extract_token_features

# Import rule-based analysis functions
try:
    from rule_based_analysis import analyze_formality, analyze_gender
except ImportError:
    from scripts.rule_based_analysis import analyze_formality, analyze_gender

from kotogram.japanese_parser import JapaneseParser

from kotogram.model import (
    StyleClassifier, Tokenizer, ModelConfig,
    FEATURE_FIELDS, ALL_FEATURE_FIELDS, set_excluded_features,
    NUM_FORMALITY_CLASSES, NUM_GENDER_CLASSES, NUM_GRAMMATICALITY_CLASSES,
    FORMALITY_LABEL_TO_ID, FORMALITY_ID_TO_LABEL,
    GENDER_LABEL_TO_ID, GENDER_ID_TO_LABEL,
    PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, MASK_TOKEN,
    load_model
)

# Record import time
timings['imports'] = time.time() - script_start_time
# Note: load_model is imported but we might need to adjust logic if training script uses custom loading.
# Actually, the training script has its own save_model function (if I recall).



class ShardedKotogramCache:
    """Durable sharded cache for Japanese → kotogram + label conversions.

    This cache stores processing results in multiple small SQLite databases (shards)
    to keep file sizes manageable (~1MB) and avoid lock contention.
    
    It accepts a legacy monolithic database path for migration purposes.
    
    Keyed by sentence hash.
    Schema: (sentence, kotogram, formality_label, gender_label)
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
                    gender_label INTEGER
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
            batch = []
            count = 0
            while True:
                rows = cursor.fetchmany(10000)
                if not rows:
                    break
                
                # Convert to format expected by put_batch (sentence, kotogram, formality, gender)
                # Legacy cache has no labels, so None
                items = [(r[0], r[1], None, None) for r in rows]
                self.put_batch(items, verbose=False) # Verbose False to avoid spam
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

    def get_batch(self, sentences: List[str]) -> Dict[str, Optional[Tuple[str, Optional[int], Optional[int]]]]:
        """Get cached entries for multiple sentences.

        Returns:
            Dict mapping sentence → (kotogram, formality, gender) OR None
        """
        if not sentences:
            return {}

        # Group by shard
        shard_to_hashes: Dict[str, List[Tuple[str, str]]] = {} # shard_path -> [(hash, sentence)]
        results: Dict[str, Optional[Tuple[str, Optional[int], Optional[int]]]] = {s: None for s in sentences}
        
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
            
            try:
                conn = sqlite3.connect(shard_path)
                placeholders = ",".join("?" * len(hashes))
                cursor = conn.execute(
                    f"SELECT sentence_hash, kotogram, formality_label, gender_label FROM cache_entries WHERE sentence_hash IN ({placeholders})",
                    hashes
                )
                
                for row in cursor:
                    h, k, f_lbl, g_lbl = row
                    sent = hash_to_sentence[h]
                    results[sent] = (k, f_lbl, g_lbl)
                conn.close()
            except sqlite3.Error:
                # If shard is corrupt or locked, just treat as miss
                pass

        return results

    def put_batch(
        self, 
        items: List[Tuple[str, str, Optional[int], Optional[int]]],
        verbose: bool = False
    ) -> None:
        """Cache multiple entries.
        
        Args:
            items: List of (sentence, kotogram, formality_label, gender_label)
        """
        if not items:
            return

        # Group by shard
        shard_to_data: Dict[str, List[Tuple[str, str, str, Optional[int], Optional[int]]]] = {}
        
        for s, k, f_lbl, g_lbl in items:
            h = self._hash_sentence(s)
            path = self._get_shard_path(h)
            if path not in shard_to_data:
                shard_to_data[path] = []
            shard_to_data[path].append((h, s, k, f_lbl, g_lbl))

        # Write to each shard
        for shard_path, data in shard_to_data.items():
            self._init_shard(shard_path) # Ensure exists
            
            conn = sqlite3.connect(shard_path)
            try:
                conn.executemany(
                    """INSERT OR REPLACE INTO cache_entries 
                       (sentence_hash, sentence, kotogram, formality_label, gender_label) 
                       VALUES (?, ?, ?, ?, ?)""",
                    data
                )
                conn.commit()
            except sqlite3.Error as e:
                if verbose:
                    print(f"Error writing to cache shard {shard_path}: {e}")
            finally:
                conn.close()

    def __len__(self) -> int:
        """Return approximate number of cached entries (expensive to count all)."""
        # For speed, we might just estimate or skip. 
        # But if needed, we iterate all shards.
        total = 0
        for fname in os.listdir(self.shards_dir):
            if fname.endswith(".db"):
                try:
                    conn = sqlite3.connect(os.path.join(self.shards_dir, fname))
                    cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                    total += int(cursor.fetchone()[0])
                    conn.close()
                except: pass
        return total


# Global kotogram cache instance
# Global kotogram cache instance
_kotogram_cache: Optional[ShardedKotogramCache] = None


def get_kotogram_cache(shards_dir: str = ShardedKotogramCache.DEFAULT_SHARDS_DIR) -> ShardedKotogramCache:
    """Get the global sharded kotogram cache instance."""
    global _kotogram_cache
    if _kotogram_cache is None or _kotogram_cache.shards_dir != shards_dir:
        _kotogram_cache = ShardedKotogramCache(shards_dir)
    return _kotogram_cache


def setup_distributed() -> Tuple[int, int, int]:
    """Initialize distributed training if available.

    Returns:
        Tuple of (rank, world_size, local_rank)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        try:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))

            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                dist.init_process_group(backend="nccl", init_method="env://")
                print(f"Distributed init: Rank {rank}/{world_size} (Local {local_rank})")
                return rank, world_size, local_rank
        except ValueError:
            pass

    # Default to single process
    return 0, 1, 0


def is_main_process() -> bool:
    """Check if we are on the main process (rank 0)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _compute_labels_batch(batch: List[Tuple[str, str, int]]) -> List[Tuple[str, str, int, int, int, int]]:
    """Compute labels for a batch of sentences."""
    results = []
    
    formality_to_id = {
        FormalityLevel.VERY_FORMAL: 0,
        FormalityLevel.FORMAL: 1,
        FormalityLevel.NEUTRAL: 2,
        FormalityLevel.CASUAL: 3,
        FormalityLevel.VERY_CASUAL: 4,
        FormalityLevel.UNPRAGMATIC_FORMALITY: 5,
    }
    gender_to_id = {
        GenderLevel.MASCULINE: 0,
        GenderLevel.FEMININE: 1,
        GenderLevel.NEUTRAL: 2,
        GenderLevel.UNPRAGMATIC_GENDER: 3,
    }
    
    for sentence, kotogram, gram_label in batch:
        try:
            formality_enum = analyze_formality(kotogram)
            gender_enum = analyze_gender(kotogram)
            formality_id = formality_to_id[formality_enum]
            gender_id = gender_to_id[gender_enum]
            results.append((sentence, kotogram, formality_id, gender_id, gram_label, 1))
        except Exception:
            pass # Skip failed
            
    return results


def _collect_tokens_batch(kotograms: List[str]) -> Dict[str, Counter]:
    """Collect token counts from a batch of kotograms in parallel.
    
    Returns:
        Dict mapping field_name -> Counter of token values
    """
    from kotogram.kotogram import extract_token_features, split_kotogram
    from kotogram.model import FEATURE_FIELDS
    
    counters = {f: Counter() for f in FEATURE_FIELDS}
    
    for k in kotograms:
        tokens = split_kotogram(k)
        for token in tokens:
            token_feat = extract_token_features(token)
            for field, value in token_feat.items():
                if field in counters: # Only track active features
                    counters[field][value] += 1
                    
    return counters


def _encode_samples_batch(
    items: List[Tuple[str, str, int, int, int]], # (sentence, kotogram, f_id, g_id, gram_label)
    tokenizer_state: Dict[str, Any], # Serialization of tokenizer
) -> List[Any]: # List[Sample]
    """Encode samples using a frozen tokenizer state."""
    from kotogram.model import Tokenizer
    # Sample is defined in this module (train_style.py), so it's available in global scope
    
    # Reconstruct tokenizer
    tokenizer = Tokenizer()
    tokenizer.field_vocabs = tokenizer_state['field_vocabs']
    tokenizer._frozen = True
    
    samples = []
    pad_id = tokenizer.pad_id
    unk_id = tokenizer.unk_id
    cls_id = tokenizer.cls_id
    
    for sentence, kotogram, f_id, g_id, gram_label in items:
        # Manually encode to avoid tokenizer overhead if possible, 
        # or just use tokenizer.encode. tokenizer.encode is fast if frozen.
        feature_ids = tokenizer.encode(kotogram, add_cls=True, add_to_vocab=False)
        
        sample = Sample(
            feature_ids=feature_ids,
            formality_label=f_id,
            gender_label=g_id,
            grammaticality_label=gram_label,
            original_sentence=sentence,
            kotogram=kotogram,
        )
        samples.append(sample)
        
    return samples


def _process_sentence_batch(
    batch: List[Tuple[str, str, int]],  # (sentence, sentence_id, gram_label)
) -> List[Tuple[str, str, str, int, int, int, int]]:
    """Process a batch of sentences in a worker process.

    Returns list of (sentence, sentence_id, kotogram, formality_id, gender_id, gram_label, success)
    where success=1 if processed successfully, 0 if failed.
    """
    # Import parser in worker process to avoid pickling issues
    from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
    from kotogram.analysis import FormalityLevel, GenderLevel
    # Try importing from local directory first, then package
    try:
        from rule_based_analysis import analyze_formality, analyze_gender
    except ImportError:
        from scripts.rule_based_analysis import analyze_formality, analyze_gender

    parser = SudachiJapaneseParser()
    results = []

    # Local copies of label mappings
    formality_to_id = {
        FormalityLevel.VERY_FORMAL: 0,
        FormalityLevel.FORMAL: 1,
        FormalityLevel.NEUTRAL: 2,
        FormalityLevel.CASUAL: 3,
        FormalityLevel.VERY_CASUAL: 4,
        FormalityLevel.UNPRAGMATIC_FORMALITY: 5,
    }
    gender_to_id = {
        GenderLevel.MASCULINE: 0,
        GenderLevel.FEMININE: 1,
        GenderLevel.NEUTRAL: 2,
        GenderLevel.UNPRAGMATIC_GENDER: 3,
    }

    for sentence, sentence_id, gram_label in batch:
        try:
            kotogram = parser.japanese_to_kotogram(sentence)
            formality_enum = analyze_formality(kotogram)
            gender_enum = analyze_gender(kotogram)
            formality_id = formality_to_id[formality_enum]
            gender_id = gender_to_id[gender_enum]
            results.append((sentence, sentence_id, kotogram, formality_id, gender_id, gram_label, 1))
        except Exception:
            results.append((sentence, sentence_id, "", 0, 0, gram_label, 0))

    return results





@dataclass
class Sample:
    """Single data sample with per-field feature IDs and labels for all tasks."""
    feature_ids: Dict[str, List[int]]  # field -> list of token IDs
    formality_label: int
    gender_label: int
    grammaticality_label: int = 1  # 1 = grammatic (default), 0 = agrammatic
    original_sentence: str = ""
    kotogram: str = ""

    @property
    def seq_len(self) -> int:
        """Get sequence length (same for all fields)."""
        first_field = next(iter(self.feature_ids.keys()))
        return len(self.feature_ids[first_field])





# Cache version - bump this when cache format changes to invalidate old caches
# v1: Initial version
# v2: Removed lemma pruning (lemma_min_freq, max_lemma_vocab)
# v3: Added parallel processing
CACHE_VERSION = 5  # v5: Removed source_id from Sample


class StyleDataset(Dataset[Sample]):  # type: ignore[misc]
    """PyTorch Dataset for style classification (formality + gender) using feature-based tokenization.

    Each sample contains per-field feature IDs rather than a single token ID sequence.
    This allows the model to learn from individual morphological features.
    """

    def __init__(
        self,
        samples: List[Sample],
        tokenizer: Tokenizer,
    ):
        """Initialize dataset with preprocessed samples.

        Args:
            samples: List of Sample objects
            tokenizer: Tokenizer used to encode samples
        """
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]

    @staticmethod
    def _get_cache_path(
        tsv_paths: List[str],
        max_samples: Optional[int],
        labeled: bool,
        grammaticality_labels: Optional[List[int]],
        cache_dir: str = ".cache/style_dataset",
    ) -> str:
        """Generate a cache file path based on input parameters.

        The cache key includes file paths, modification times, and processing options
        to ensure the cache is invalidated when inputs change.
        """
        # Build a hash from all relevant parameters
        hash_parts = [f"v{CACHE_VERSION}"]  # Include cache version

        for tsv_path in tsv_paths:
            hash_parts.append(tsv_path)
            # Include file modification time
            if os.path.exists(tsv_path):
                mtime = os.path.getmtime(tsv_path)
                hash_parts.append(str(mtime))

        hash_parts.append(f"max_samples={max_samples}")
        hash_parts.append(f"labeled={labeled}")
        hash_parts.append(f"gram_labels={grammaticality_labels}")

        # Create hash
        hash_str = hashlib.sha256("|".join(hash_parts).encode()).hexdigest()[:16]
        return os.path.join(cache_dir, f"dataset_v{CACHE_VERSION}_{hash_str}.pkl")

    @staticmethod
    def _process_parallel(
        rows: List[Tuple[str, str, int]],  # (sentence, sentence_id, gram_label)
        num_workers: Optional[int] = None,
        batch_size: int = 1000,
        verbose: bool = True,
        use_kotogram_cache: bool = True,
    ) -> List[Tuple[str, str, int, int, int, int]]:
        """Process sentences in parallel using multiprocessing.

        Uses a durable kotogram cache to avoid re-parsing sentences that have
        already been processed in previous runs. The cache is keyed only by
        sentence content, so it persists across changes to model architecture,
        training parameters, etc.

        Args:
            rows: List of (sentence, sentence_id, gram_label, source_id) tuples
            num_workers: Number of worker processes (default: CPU count)
            batch_size: Sentences per batch sent to workers
            verbose: Print progress
            use_kotogram_cache: If True, use durable kotogram cache

        Returns:
            List of (sentence, kotogram, formality_id, gender_id, gram_label, success) tuples
        """
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)

        if len(rows) == 0:
            return []

        # Check kotogram cache for already-parsed sentences
        # Check kotogram cache for already-parsed sentences
        cache = get_kotogram_cache() if use_kotogram_cache else None
        
        # Results storage
        # We need final list of (sentence, kotogram, f_id, g_id, gram_label, success)
        # We can store intermediate results in a dict or build list at the end
        final_data: Dict[str, Tuple[str, int, int]] = {} # sentence -> (kotogram, f, g)
        
        uncached_rows: List[Tuple[str, str, int]] = [] # Need parsing
        unlabeled_rows: List[Tuple[str, str, int]] = [] # Have kotogram, need labels

        if cache is not None:
            sentences = [r[0] for r in rows]
            cached_batch = cache.get_batch(sentences)
            
            kotogram_hits = 0
            label_hits = 0

            for row in rows:
                s = row[0]
                entry = cached_batch.get(s)
                
                if entry is not None:
                    k, f, g = entry
                    kotogram_hits += 1
                    
                    if f is not None and g is not None:
                        final_data[s] = (k, f, g)
                        label_hits += 1
                    else:
                        # Partial hit (kotogram only)
                        unlabeled_rows.append((s, k, row[2]))
                else:
                    uncached_rows.append(row)

            if verbose:
                print(f"Kotogram cache: {kotogram_hits}/{len(rows)} hits ({100*kotogram_hits/len(rows):.1f}%)")
                print(f"Label cache:    {label_hits}/{len(rows)} hits ({100*label_hits/len(rows):.1f}%)")
        else:
            uncached_rows = list(rows)


        # 1. Process Uncached (Parse + Label)
        # -----------------------------------
        new_cache_entries: List[Tuple[str, str, Optional[int], Optional[int]]] = []
        
        # Initialize multiprocessing context early
        ctx = mp.get_context('spawn')

        if uncached_rows:
            batches = [uncached_rows[i:i + batch_size] for i in range(0, len(uncached_rows), batch_size)]

            if verbose:
                print(f"Parsing {len(uncached_rows)} uncached sentences with {num_workers} workers...")

            processed = 0

            with ctx.Pool(num_workers) as pool:
                for batch_results in pool.imap(_process_sentence_batch, batches):
                    # batch_results: (sentence, _sid, kotogram, f_id, g_id, gram_label, success)
                    for sentence, _sid, kotogram, f_id, g_id, _gram_label, success in batch_results:
                        if success:
                            final_data[sentence] = (kotogram, f_id, g_id)
                            new_cache_entries.append((sentence, kotogram, f_id, g_id))
                    
                    processed += len(batch_results)
                    if verbose and processed % 10000 < batch_size:
                        print(f"  Parsed {processed}/{len(uncached_rows)} sentences...")

            if verbose:
                print(f"  Completed: {len(new_cache_entries)} new kotograms parsed")


        # 2. Process Unlabeled (Label only)
        # ---------------------------------
        # These are partial cache hits (legacy cache data)
        if unlabeled_rows:
            if verbose:
                print(f"Computing labels for {len(unlabeled_rows)} partial-hit sentences...")
            
            # Batching
            label_batches = [unlabeled_rows[i:i + batch_size] for i in range(0, len(unlabeled_rows), batch_size)]
            processed_labels = 0
            
            with ctx.Pool(num_workers) as pool:
                for batch_results in pool.imap(_compute_labels_batch, label_batches):
                    # batch_results: (sentence, kotogram, f_id, g_id, gram_label, success)
                    for sentence, kotogram, f_id, g_id, _gram_label, success in batch_results:
                        if success:
                            final_data[sentence] = (kotogram, f_id, g_id)
                            new_cache_entries.append((sentence, kotogram, f_id, g_id))
                    
                    processed_labels += len(batch_results)
                    if verbose and processed_labels % 100000 < batch_size:
                         print(f"  Labeled {processed_labels}/{len(unlabeled_rows)} sentences...")

        # 3. Update Cache
        # ---------------
        if cache is not None and new_cache_entries:
            cache.put_batch(new_cache_entries, verbose=verbose)
            if verbose:
                print(f"  Saved {len(new_cache_entries)} new entries to sharded cache")


        # 4. Assemble Final Results
        # -------------------------
        # Match original rows to preserve order/grammar label
        results: List[Tuple[str, str, int, int, int, int]] = []
        
        for sentence, _sid, gram_label in rows:
            if sentence in final_data:
                k, f, g = final_data[sentence]
                results.append((sentence, k, f, g, gram_label, 1))
            else:
                pass # Failed to parse or label

        if verbose:
            print(f"  Total: {len(results)} sentences ready for training")

        return results

    @staticmethod
    def _save_cache(
        cache_path: str,
        samples: List[Sample],
        tokenizer: Tokenizer,
    ) -> None:
        """Save preprocessed samples and tokenizer to cache."""
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cache_data = {
            'version': CACHE_VERSION,
            'samples': samples,
            'field_vocabs': tokenizer.field_vocabs,
            'frozen': tokenizer._frozen,
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

    @staticmethod
    def _load_cache(
        cache_path: str,
        tokenizer: Tokenizer,
    ) -> Optional[List[Sample]]:
        """Load preprocessed samples from cache and restore tokenizer state.

        Returns None if cache doesn't exist, has wrong version, or is invalid.
        """
        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, 'rb') as f:
                cache_data: Dict[str, Any] = pickle.load(f)

            # Check cache version
            cached_version = cache_data.get('version', 1)
            if cached_version != CACHE_VERSION:
                print(f"  Cache version mismatch (found v{cached_version}, need v{CACHE_VERSION}), rebuilding...")
                return None

            # Restore tokenizer state
            tokenizer.field_vocabs = cache_data['field_vocabs']
            tokenizer._frozen = cache_data.get('frozen', False)

            samples: List[Sample] = cache_data['samples']
            return samples
        except Exception:
            # Cache corrupted or incompatible, ignore it
            return None

    @classmethod
    def from_tsv(
        cls,
        tsv_path: str,
        tokenizer: Tokenizer,
        parser: Optional[JapaneseParser] = None,
        max_samples: Optional[int] = None,
        verbose: bool = True,
        labeled: bool = True,
        use_cache: bool = True,
        cache_dir: str = ".cache/style_dataset",
        sample_ratio: float = 1.0,
    ) -> 'StyleDataset':
        """Load dataset from TSV file of Japanese sentences.

        Args:
            tsv_path: Path to TSV file with Japanese sentences
            tokenizer: Tokenizer to build vocabulary
            parser: JapaneseParser instance (defaults to SudachiJapaneseParser)
            max_samples: Optional limit on number of samples
            verbose: If True, print progress
            labeled: If True, compute formality and gender labels. If False, use dummy labels
                    (for pretraining on unlabeled data).
            use_cache: If True, cache preprocessed data to disk for faster subsequent loads
            cache_dir: Directory for cache files
            sample_ratio: Ratio of data to use (0.0 to 1.0)

        Returns:
            StyleDataset with encoded samples
        """
        # Try to load from cache
        if use_cache:
            # Note: sample_ratio is NOT part of cache key - we cache full dataset
            cache_path = cls._get_cache_path([tsv_path], max_samples, labeled, None, cache_dir)
            cached_samples = cls._load_cache(cache_path, tokenizer)
            if cached_samples is not None:
                if verbose:
                    print(f"Loaded {len(cached_samples)} samples from cache")
                    print(f"Vocabulary sizes: {tokenizer.get_vocab_sizes()}")
                
                # Apply subsampling after loading from cache
                if sample_ratio < 1.0:
                    if verbose:
                        print(f"  Subsampling {sample_ratio:.1%} of {len(cached_samples)} cached samples...")
                    content_str = "".join(s.original_sentence for s in cached_samples[:100])
                    seed = int(hashlib.md5(content_str.encode()).hexdigest(), 16) % 100000
                    rng = random.Random(seed)
                    cached_samples = rng.sample(cached_samples, int(len(cached_samples) * sample_ratio))
                    if verbose:
                        print(f"  Using {len(cached_samples)} samples after subsampling")

                return cls(cached_samples, tokenizer)

        # Phase 1: Read all rows from TSV file (fast I/O)
        # Tuple: (sentence, sentence_id, gram_label)
        all_rows: List[Tuple[str, str, int]] = []
        gram_label = 1  # Single-file load assumes grammatic

        if verbose:
            print(f"Reading {tsv_path}...")

        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) < 3:
                    continue
                # Simply ignore rows that don't have enough columns, but otherwise be permissive
                # Column 1 = ID, Column 2 = Lang (ignored), Column 3 = Sentence
                sentence_id, _lang, sentence = row[0], row[1], row[2]
                
                # Removed: if lang != 'jpn': continue
                # Removed: source_id logic

                all_rows.append((sentence, sentence_id, gram_label))
                if max_samples and len(all_rows) >= max_samples:
                    break
        
        # Subsampling moved to after caching to ensure cache consistency
        # We read all rows here

        if verbose:
            print(f"  Read {len(all_rows)} sentences")

        # Phase 2: Process sentences in parallel (kotogram conversion + label computation)
        processed_results = cls._process_parallel(all_rows, verbose=verbose)

        # Phase 3: Build vocabulary and create samples (must be sequential)
        if verbose:
            print("\nBuilding vocabulary...")

        samples: List[Sample] = []
        formality_counts: Counter[FormalityLevel] = Counter()
        gender_counts: Counter[GenderLevel] = Counter()
        grammaticality_counts: Counter[int] = Counter()

        for sentence, kotogram, formality_id, gender_id, gram_label, _success in processed_results:
            # Encode to feature IDs (builds vocabulary - must be sequential)
            feature_ids = tokenizer.encode(kotogram, add_cls=True, add_to_vocab=True)

            sample = Sample(
                feature_ids=feature_ids,
                formality_label=formality_id,
                gender_label=gender_id,
                grammaticality_label=gram_label,
                original_sentence=sentence,
                kotogram=kotogram,
            )
            samples.append(sample)

            # Track counts using IDs (enums were converted in workers)
            formality_counts[FORMALITY_ID_TO_LABEL[formality_id]] += 1
            gender_counts[GENDER_ID_TO_LABEL[gender_id]] += 1
            grammaticality_counts[gram_label] += 1

        if verbose:
            print(f"\nDataset loaded: {len(samples)} samples")
            print(f"Vocabulary sizes: {tokenizer.get_vocab_sizes()}")
            print("Formality distribution:")
            for f_label, f_count in sorted(formality_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {f_label.value}: {f_count} ({100*f_count/len(samples):.1f}%)")
            print("Gender distribution:")
            for g_label, g_count in sorted(gender_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {g_label.value}: {g_count} ({100*g_count/len(samples):.1f}%)")
            print("Grammaticality distribution:")
            gram_labels_map = {1: "grammatic", 0: "agrammatic"}
            for g_id in [1, 0]:
                g_count = grammaticality_counts.get(g_id, 0)
                print(f"  {gram_labels_map[g_id]}: {g_count} ({100*g_count/len(samples):.1f}%)")

        # Freeze vocabulary after building (this finalizes lemma vocab)
        tokenizer.freeze()

        if verbose:
            final_sizes = tokenizer.get_vocab_sizes()
            print(f"Final vocabulary sizes: {final_sizes}")
            # Show detailed stats for key fields
            print(f"  surface: {final_sizes['surface']:,}, lemma: {final_sizes['lemma']:,}")
            print(f"  pos: {final_sizes['pos']}, conjugated_type: {final_sizes['conjugated_type']}, conjugated_form: {final_sizes['conjugated_form']}")

        # Save to cache (full dataset)
        if use_cache:
            cls._save_cache(cache_path, samples, tokenizer)
            if verbose:
                print(f"Saved {len(samples)} preprocessed samples to cache")

        # Apply subsampling after processing/caching
        if sample_ratio < 1.0:
            if verbose:
                print(f"  Subsampling {sample_ratio:.1%} of {len(samples)} samples...")
            content_str = "".join(s.original_sentence for s in samples[:100])
            seed = int(hashlib.md5(content_str.encode()).hexdigest(), 16) % 100000
            rng = random.Random(seed)
            samples = rng.sample(samples, int(len(samples) * sample_ratio))
            if verbose:
                print(f"  Using {len(samples)} samples after subsampling")

        return cls(samples, tokenizer)

    @classmethod
    def from_multiple_tsv(
        cls,
        tsv_paths: List[str],
        tokenizer: Tokenizer,
        parser: Optional[JapaneseParser] = None,
        max_samples: Optional[int] = None,
        verbose: bool = True,
        labeled: bool = True,
        grammaticality_labels: Optional[List[int]] = None,
        use_cache: bool = True,
        cache_dir: str = ".cache/style_dataset",
        sample_ratio: float = 1.0,
    ) -> 'StyleDataset':
        """Load dataset from multiple TSV files of Japanese sentences.

        This method loads samples from multiple TSV files, combining them into
        a single dataset. Useful for augmenting training data with additional
        examples (e.g., unpragmatic sentences, agrammatic sentences).

        Args:
            tsv_paths: List of paths to TSV files with Japanese sentences
            tokenizer: Tokenizer to build vocabulary
            parser: JapaneseParser instance (defaults to SudachiJapaneseParser)
            max_samples: Optional limit on total number of samples across all files
            verbose: If True, print progress
            labeled: If True, compute formality and gender labels
            grammaticality_labels: Optional list of grammaticality labels (0 or 1) for each
                                  TSV file. If provided, must have same length as tsv_paths.
                                  1 = grammatic (default), 0 = agrammatic.
            use_cache: If True, cache preprocessed data to disk for faster subsequent loads
            cache_dir: Directory for cache files
            sample_ratio: Ratio of data to use (0.0 to 1.0)

        Returns:
            StyleDataset with encoded samples from all files
        """
        # Default all files to grammatic if not specified
        if grammaticality_labels is None:
            grammaticality_labels = [1] * len(tsv_paths)
        elif len(grammaticality_labels) != len(tsv_paths):
            raise ValueError(
                f"grammaticality_labels length ({len(grammaticality_labels)}) "
                f"must match tsv_paths length ({len(tsv_paths)})"
            )

        # Try to load from cache
        if use_cache:
            # Note: sample_ratio is NOT part of cache key - we cache full dataset
            cache_path = cls._get_cache_path(tsv_paths, max_samples, labeled, grammaticality_labels, cache_dir)
            cached_samples = cls._load_cache(cache_path, tokenizer)
            if cached_samples is not None:
                if verbose:
                    print(f"Loaded {len(cached_samples)} samples from cache")
                    print(f"Vocabulary sizes: {tokenizer.get_vocab_sizes()}")
                
                # Apply subsampling after loading from cache
                if sample_ratio < 1.0:
                    if verbose:
                        print(f"  Subsampling {sample_ratio:.1%} of {len(cached_samples)} cached samples...")
                    content_str = "".join(s.original_sentence for s in cached_samples[:100])
                    seed = int(hashlib.md5(content_str.encode()).hexdigest(), 16) % 100000
                    rng = random.Random(seed)
                    cached_samples = rng.sample(cached_samples, int(len(cached_samples) * sample_ratio))
                    if verbose:
                        print(f"  Using {len(cached_samples)} samples after subsampling")

                return cls(cached_samples, tokenizer)

        # Phase 1: Read all rows from TSV files (fast I/O)
        # Tuple: (sentence, sentence_id, gram_label)
        all_rows: List[Tuple[str, str, int]] = []
        for tsv_path, gram_label in zip(tsv_paths, grammaticality_labels):
            if verbose:
                gram_str = "grammatic" if gram_label == 1 else "agrammatic"
                print(f"Reading {tsv_path} ({gram_str})...")

            file_count = 0
            file_rows: List[Tuple[str, str, int]] = []
            with open(tsv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if len(row) < 3:
                        continue
                    sentence_id, _lang, sentence = row[0], row[1], row[2]
                    
                    # Removed lang check and source_id logic
                    
                    file_rows.append((sentence, sentence_id, gram_label))
                    
                    if max_samples and len(all_rows) + len(file_rows) >= max_samples:
                        break
            
            # Subsampling now happens after all data is loaded and cached
            # We read all rows here

            all_rows.extend(file_rows)
            file_count = len(file_rows)

            if verbose:
                print(f"  Read {file_count} sentences from {tsv_path}")

            if max_samples and len(all_rows) >= max_samples:
                break

        if verbose:
            print(f"\nTotal sentences to process: {len(all_rows)}")

        # Phase 2: Process sentences in parallel (kotogram conversion + label computation)
        processed_results = cls._process_parallel(all_rows, verbose=verbose)

        # Phase 3: Build vocabulary and create samples (Parallelized)
        if verbose:
            print("\nBuilding vocabulary and encoding samples (Phase 3)...")

        # 3a. Parallel Token Collection (map-reduce)
        # ------------------------------------------
        kotograms = [r[1] for r in processed_results]
        
        # Batching for token collection
        # Larger batches are fine for token collection as it's CPU bound string processing
        token_batches = [kotograms[i:i + 5000] for i in range(0, len(kotograms), 5000)]
        
        ctx = mp.get_context('spawn')
        num_workers = max(1, mp.cpu_count() - 1)

        if verbose:
            print(f"  Collecting tokens from {len(kotograms)} sentences with {num_workers} workers...")
            
        merged_counters = {f: Counter() for f in tokenizer.field_vocabs.keys()}
        
        with ctx.Pool(num_workers) as pool:
            for batch_counters in pool.imap(_collect_tokens_batch, token_batches):
                for f, counter in batch_counters.items():
                    if f in merged_counters:
                        merged_counters[f].update(counter)

        # Update tokenizer sequentially (fast dict updates)
        if verbose:
            print("  Updating tokenizer vocabulary...")
            
        for f, counter in merged_counters.items():
            # Add tokens sorted by frequency (optional, but good for stability)
            for token, _count in counter.most_common():
                tokenizer._add_value(f, token)
        
        # Freeze vocabulary
        tokenizer.freeze()
        
        # 3b. Parallel Sample Encoding
        # ----------------------------
        if verbose:
            print("  Encoding samples with frozen tokenizer...")
            
        # Serialize tokenizer state for workers
        tokenizer_state = {
            'field_vocabs': tokenizer.field_vocabs,
        }
        
        # Prepare inputs: (sentence, kotogram, f_id, g_id, gram_label)
        encoding_inputs = []
        for p in processed_results:
            # p: (sentence, kotogram, f_id, g_id, gram_label, success)
            if p[5]: # success
                encoding_inputs.append((p[0], p[1], p[2], p[3], p[4]))

        batches = [encoding_inputs[i:i + 5000] for i in range(0, len(encoding_inputs), 5000)]
        
        samples: List[Sample] = []
        # Re-initialize counters for stats
        formality_counts: Counter[FormalityLevel] = Counter()
        gender_counts: Counter[GenderLevel] = Counter()
        grammaticality_counts: Counter[int] = Counter()

        processed_encodings = 0
        with ctx.Pool(num_workers) as pool:
            # Use starmap equivalent via a wrapper lambda or just pass tuple to helper
            # We defined helper to take (batch, state)
            # But imap takes one arg. Let's use partial or tuple.
            # Helper `_encode_samples_batch` takes (batch, state).
            # We can't use partial easily with spawn context pickling sometimes.
            # Better to use a generator or starmap.
            # Let's wrap calling logic or use `starmap` (but starmap is eager? No, starmap is eager).
            # `imap` is lazy.
            # We will use a list of args: [(batch, state), ...]
            
            pool_args = [(b, tokenizer_state) for b in batches]
            
            # We need a wrapper function because `_encode_samples_batch` takes 2 args, 
            # but we can make it take 1 tuple arg? No, let's just make it take 2 args in definition 
            # and use `starmap`. `starmap` supports multiple args.
            
            for batch_samples in pool.starmap(_encode_samples_batch, pool_args):
                samples.extend(batch_samples)
                processed_encodings += len(batch_samples)
                
                # Update stats locally (fast)
                for s in batch_samples:
                    formality_counts[FORMALITY_ID_TO_LABEL[s.formality_label]] += 1
                    gender_counts[GENDER_ID_TO_LABEL[s.gender_label]] += 1
                    grammaticality_counts[s.grammaticality_label] += 1
                    
                if verbose and processed_encodings % 100000 < 5000:
                     print(f"  Encoded {processed_encodings}/{len(encoding_inputs)} samples...")



        if verbose:
            print(f"\nDataset loaded: {len(samples)} samples from {len(tsv_paths)} files")
            print(f"Vocabulary sizes: {tokenizer.get_vocab_sizes()}")
            print("Formality distribution:")
            for f_label, f_count in sorted(formality_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {f_label.value}: {f_count} ({100*f_count/len(samples):.1f}%)")
            print("Gender distribution:")
            for g_label, g_count in sorted(gender_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {g_label.value}: {g_count} ({100*g_count/len(samples):.1f}%)")
            print("Grammaticality distribution:")
            gram_labels_map = {1: "grammatic", 0: "agrammatic"}
            for g_id in [1, 0]:
                g_count = grammaticality_counts.get(g_id, 0)
                print(f"  {gram_labels_map[g_id]}: {g_count} ({100*g_count/len(samples):.1f}%)")

        # Freeze vocabulary after building (this finalizes lemma vocab)
        tokenizer.freeze()

        if verbose:
            final_sizes = tokenizer.get_vocab_sizes()
            print(f"Final vocabulary sizes: {final_sizes}")
            # Show detailed stats for key fields
            print(f"  surface: {final_sizes['surface']:,}, lemma: {final_sizes['lemma']:,}")
            print(f"  pos: {final_sizes['pos']}, conjugated_type: {final_sizes['conjugated_type']}, conjugated_form: {final_sizes['conjugated_form']}")

        # Save to cache (full dataset)
        if use_cache:
            cls._save_cache(cache_path, samples, tokenizer)
            if verbose:
                print(f"Saved {len(samples)} preprocessed samples to cache")

        # Apply subsampling after processing/caching
        if sample_ratio < 1.0:
            if verbose:
                print(f"  Subsampling {sample_ratio:.1%} of {len(samples)} samples...")
            content_str = "".join(s.original_sentence for s in samples[:100])
            seed = int(hashlib.md5(content_str.encode()).hexdigest(), 16) % 100000
            rng = random.Random(seed)
            samples = rng.sample(samples, int(len(samples) * sample_ratio))
            if verbose:
                print(f"  Using {len(samples)} samples after subsampling")

        return cls(samples, tokenizer)

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        stratify: bool = True,
    ) -> Tuple['StyleDataset', 'StyleDataset', 'StyleDataset']:
        """Split dataset into train/validation/test sets.

        Args:
            train_ratio: Fraction of data for training (default 0.8)
            val_ratio: Fraction of data for validation (default 0.1)
            seed: Random seed for reproducibility
            stratify: If True, use stratified splitting to ensure proportional
                     representation of all class combinations in each split.
                     Uses combined (formality, gender, grammaticality) labels for stratification.

        Returns:
            Tuple of (train, validation, test) StyleDataset instances
        """
        random.seed(seed)

        train_indices: List[int] = []
        val_indices: List[int] = []
        test_indices: List[int] = []

        if not stratify:
            # Original random splitting
            indices = list(range(len(self.samples)))
            random.shuffle(indices)

            n_train = int(len(indices) * train_ratio)
            n_val = int(len(indices) * val_ratio)

            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            test_indices = indices[n_train + n_val:]
            # Original random splitting
            indices = list(range(len(self.samples)))
            random.shuffle(indices)

            n_train = int(len(indices) * train_ratio)
            n_val = int(len(indices) * val_ratio)

            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            test_indices = indices[n_train + n_val:]
        else:
            # Stratified splitting using combined (formality, gender, grammaticality) labels
            # Group samples by combined label
            label_to_indices: Dict[Tuple[int, int, int], List[int]] = {}
            for i, sample in enumerate(self.samples):
                combined_label = (sample.formality_label, sample.gender_label, sample.grammaticality_label)
                if combined_label not in label_to_indices:
                    label_to_indices[combined_label] = []
                label_to_indices[combined_label].append(i)

            # Split each group proportionally
            for combined_label, group_indices in label_to_indices.items():
                random.shuffle(group_indices)
                n = len(group_indices)
                n_train = max(1, int(n * train_ratio)) if n > 0 else 0
                n_val = max(1, int(n * val_ratio)) if n > 1 else 0

                # Ensure we have at least 1 sample in test if possible
                if n > 2 and n_train + n_val >= n:
                    # Reduce train to make room for test
                    n_train = max(1, n - n_val - 1)

                train_indices.extend(group_indices[:n_train])
                val_indices.extend(group_indices[n_train:n_train + n_val])
                test_indices.extend(group_indices[n_train + n_val:])

            # Shuffle the combined indices
            random.shuffle(train_indices)
            random.shuffle(val_indices)
            random.shuffle(test_indices)

        train_samples = [self.samples[i] for i in train_indices]
        val_samples = [self.samples[i] for i in val_indices]
        test_samples = [self.samples[i] for i in test_indices]

        return (
            StyleDataset(train_samples, self.tokenizer),
            StyleDataset(val_samples, self.tokenizer),
            StyleDataset(test_samples, self.tokenizer),
        )

    def get_formality_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights for imbalanced formality data."""
        counts = Counter(s.formality_label for s in self.samples)
        total = len(self.samples)
        weights = torch.zeros(NUM_FORMALITY_CLASSES)

        for label_id, count in counts.items():
            weights[label_id] = total / (NUM_FORMALITY_CLASSES * count) if count > 0 else 0.0

        return weights

    def get_gender_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights for imbalanced gender data."""
        counts = Counter(s.gender_label for s in self.samples)
        total = len(self.samples)
        weights = torch.zeros(NUM_GENDER_CLASSES)

        for label_id, count in counts.items():
            weights[label_id] = total / (NUM_GENDER_CLASSES * count) if count > 0 else 0.0

        return weights

    def get_grammaticality_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights for imbalanced grammaticality data."""
        counts = Counter(s.grammaticality_label for s in self.samples)
        total = len(self.samples)
        weights = torch.zeros(NUM_GRAMMATICALITY_CLASSES)

        for label_id, count in counts.items():
            weights[label_id] = total / (NUM_GRAMMATICALITY_CLASSES * count) if count > 0 else 0.0

        return weights


def collate_fn(
    batch: List[Sample],
    pad_id: int = 0,
    max_seq_len: Optional[int] = None,
) -> Dict[str, Any]:
    """Collate samples into padded batches.

    Args:
        batch: List of Sample objects
        pad_id: Padding token ID
        max_seq_len: Maximum sequence length. Sequences longer than this will be
                    truncated. If None, uses the maximum length in the batch.

    Returns:
        Dictionary with per-field 'input_ids_<field>', 'attention_mask',
        'formality_labels', 'gender_labels', 'grammaticality_labels' tensors,
        and 'original_sentence', 'kotogram' lists.
    """
    batch_max_len = max(s.seq_len for s in batch)
    # Apply truncation if max_seq_len is specified
    max_len = min(batch_max_len, max_seq_len) if max_seq_len else batch_max_len

    # Initialize per-field lists
    field_ids: Dict[str, List[List[int]]] = {f: [] for f in FEATURE_FIELDS}
    attention_mask = []
    formality_labels = []
    gender_labels = []
    grammaticality_labels = []

    for sample in batch:
        # Truncate sequence if needed
        seq_len = min(sample.seq_len, max_len)
        padding_len = max_len - seq_len

        for field in FEATURE_FIELDS:
            # Truncate and pad
            truncated = sample.feature_ids[field][:seq_len]
            padded = truncated + [pad_id] * padding_len
            field_ids[field].append(padded)

        attention_mask.append([1] * seq_len + [0] * padding_len)
        formality_labels.append(sample.formality_label)
        gender_labels.append(sample.gender_label)
        grammaticality_labels.append(sample.grammaticality_label)

    result = {
        f'input_ids_{field}': torch.tensor(field_ids[field], dtype=torch.long)
        for field in FEATURE_FIELDS
    }
    result['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
    result['formality_labels'] = torch.tensor(formality_labels, dtype=torch.long)
    result['gender_labels'] = torch.tensor(gender_labels, dtype=torch.long)
    result['grammaticality_labels'] = torch.tensor(grammaticality_labels, dtype=torch.long)

    # Pass through metadata
    # Pass through metadata
    result['original_sentence'] = [s.original_sentence for s in batch]
    result['kotogram'] = [s.kotogram for s in batch]

    return result





class MLMHead(nn.Module):  # type: ignore[misc]
    """Masked language modeling head for feature-based tokens.

    For MLM pretraining, we predict the original token's features at masked positions.
    This head predicts all feature fields (pos, pos_detail1, pos_detail2, conjugated_type,
    conjugated_form, lemma) to learn richer representations.
    """

    def __init__(self, config: ModelConfig):
        """Initialize MLM head.

        Args:
            config: ModelConfig with model dimensions
        """
        super().__init__()
        self.config = config

        # Shared transformation layer
        self.shared_dense = nn.Linear(config.d_model, config.d_model)
        self.shared_norm = nn.LayerNorm(config.d_model)

        # Per-field decoders
        self.decoders = nn.ModuleDict()
        for field_name in FEATURE_FIELDS:
            vocab_size = config.vocab_sizes.get(field_name, 100)
            self.decoders[field_name] = nn.Linear(config.d_model, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Project hidden states to per-field vocabulary logits.

        Args:
            hidden_states: Encoder output of shape (batch, seq_len, d_model)

        Returns:
            Dict mapping field name to logits of shape (batch, seq_len, field_vocab_size)
        """
        x = self.shared_dense(hidden_states)
        x = F.gelu(x)
        x = self.shared_norm(x)
        return {field: decoder(x) for field, decoder in self.decoders.items()}


class StyleClassifierWithMLM(StyleClassifier):
    """Multi-task style classifier with MLM pretraining head.

    This model can be:
    1. Pre-trained with masked token prediction (self-supervised)
    2. Fine-tuned for multi-task style classification (supervised)
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.mlm_head = MLMHead(config)

    def forward(
        self,
        *args,
        mode: str = "classification",
        **kwargs,
    ) -> Any:
        """Forward pass dispatch.
        
        Args:
            *args: Positional arguments for the specific forward method
            mode: 'classification' (default) or 'mlm'
            **kwargs: Keyword arguments for the specific forward method
        """
        if mode == "mlm":
            return self.forward_mlm(*args, **kwargs)
        return super().forward(*args, **kwargs)

    def forward_mlm(
        self,
        field_inputs: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for masked language modeling.

        Args:
            field_inputs: Dict with masked 'input_ids_<field>' tensors
            attention_mask: Binary mask for padding

        Returns:
            Dict mapping field name to logits of shape (batch, seq_len, field_vocab_size)
        """
        encoder_output = self.get_encoder_output(field_inputs, attention_mask)
        return cast(Dict[str, torch.Tensor], self.mlm_head(encoder_output))

    def reset_classifier(self) -> None:
        """Reinitialize all classifier head weights.

        Call this after MLM pretraining and before supervised fine-tuning
        to start the classification heads from a fresh state while keeping
        the pretrained encoder weights.
        """
        for classifier in [self.formality_classifier, self.gender_classifier, self.grammaticality_classifier]:
            for module in classifier.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)


@dataclass
class TrainerConfig:
    """Configuration for model training."""
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    patience: int = 5  # Early stopping patience
    lr_scheduler_patience: int = 2
    lr_scheduler_factor: float = 0.5
    gradient_clip: float = 1.0
    use_class_weights: bool = True
    formality_loss_weight: float = 1.0  # Weight for formality loss in multi-task
    gender_loss_weight: float = 1.0  # Weight for gender loss in multi-task
    grammaticality_loss_weight: float = 1.0  # Weight for grammaticality loss in multi-task
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    use_amp: bool = False  # Mixed precision training
    grad_accum_steps: int = 1  # Gradient accumulation steps
    local_rank: int = 0  # Local rank for distributed training
    world_size: int = 1  # World size for distributed training



def create_mlm_batch(
    batch: Dict[str, torch.Tensor],
    mask_prob: float = 0.15,
    mask_token_id: int = 3,
    vocab_sizes: Optional[Dict[str, int]] = None,
    special_token_ids: Optional[List[int]] = None,
) -> Dict[str, torch.Tensor]:
    """Create masked language modeling batch for feature-based tokens.

    Masks positions across all feature fields and creates labels for all fields.
    This enables richer MLM pretraining that learns to predict all morphological
    features, not just POS tags.

    Args:
        batch: Batch with 'input_ids_<field>' for each field, attention_mask
        mask_prob: Probability of masking a token position
        mask_token_id: ID of MASK token
        vocab_sizes: Dict mapping field name to vocabulary size (for random replacement)
        special_token_ids: IDs to never mask

    Returns:
        Batch with masked input_ids_<field> and mlm_labels_<field> for each field
    """
    special_token_ids = special_token_ids or [0, 1, 2, 3]  # PAD, UNK, CLS, MASK
    vocab_sizes = vocab_sizes or {}

    # Define fields that should be completely HIDDEN (100% masked, no prediction)
    # This forces the model to learn grammar structure without lexical cues.
    HIDDEN_FIELDS = ['surface', 'lemma']

    # Use 'pos' field as the primary for determining mask positions for NON-HIDDEN fields
    primary_field = 'pos'
    primary_ids = batch[f'input_ids_{primary_field}'].clone()

    # Create mask for tokens that can be masked
    maskable = batch['attention_mask'].bool()
    for special_id in special_token_ids:
        maskable &= (primary_ids != special_id)

    # Random mask for standard MLM fields
    probs = torch.rand_like(primary_ids.float())
    mask = maskable & (probs < mask_prob)

    # 80% MASK, 10% random, 10% unchanged
    mask_token_positions = mask & (probs < mask_prob * 0.8)
    random_token_positions = mask & (probs >= mask_prob * 0.8) & (probs < mask_prob * 0.9)

    # Clone all field IDs and apply masking, create labels for each field
    result = {'attention_mask': batch['attention_mask']}

    for field in FEATURE_FIELDS:
        field_ids = batch[f'input_ids_{field}'].clone()
        mlm_labels = torch.full_like(field_ids, -100)

        if field in HIDDEN_FIELDS:
            # For hidden fields:
            # 1. Mask ALL non-padding tokens (100% hidden)
            # 2. Labels remain -100 (never predict them)
            
            # Apply MASK token to all attended positions
            # We respect the attention mask (0 = padding)
            active_tokens = batch['attention_mask'].bool()
            field_ids[active_tokens] = mask_token_id
            
        else:
            # For standard fields (Grammar):
            # Apply standard MLM logic
            
            # Create labels for this field (ignore non-masked positions)
            mlm_labels[mask] = field_ids[mask]
            
            # Apply MASK token
            field_ids[mask_token_positions] = mask_token_id

            # Apply random replacement for this field using its own vocabulary
            field_vocab_size = vocab_sizes.get(field)
            if field_vocab_size:
                num_random = int(random_token_positions.sum().item())
                if num_random > 0:
                    field_ids[random_token_positions] = torch.randint(
                        len(special_token_ids), field_vocab_size, (num_random,)
                    )
        
        result[f'mlm_labels_{field}'] = mlm_labels
        result[f'input_ids_{field}'] = field_ids

    return result


class MLMTrainer:
    """Trainer for self-supervised MLM pretraining with feature-based tokens.

    This trainer predicts all morphological feature fields (pos, pos_detail1,
    pos_detail2, conjugated_type, conjugated_form, lemma) at masked positions,
    providing richer supervision than POS-only MLM.
    """

    def __init__(
        self,
        model: StyleClassifierWithMLM,
        dataset: StyleDataset,
        config: Optional[TrainerConfig] = None,
        mask_prob: float = 0.15,
        field_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize MLM trainer.

        Args:
            model: StyleClassifierWithMLM model
            dataset: Dataset for pretraining (can be unlabeled)
            config: TrainerConfig with hyperparameters
            mask_prob: Probability of masking each token position
            field_weights: Optional weights for each field's loss contribution.
                          Defaults to equal weights for all fields.
        """
        self.model = model
        self.dataset = dataset
        self.dataset = dataset
        self.config = config or TrainerConfig()
        self.mask_prob = mask_prob

        # Setup device and distributed
        if self.config.world_size > 1:
            self.device = torch.device('cuda', self.config.local_rank)
            self.is_distributed = True
        else:
            self.device = torch.device(self.config.device)
            self.is_distributed = False

        self.model.to(self.device)

        # Wrap in DDP if distributed
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
                find_unused_parameters=True # MLM might have unused params depending on implementation
            )  # type: ignore

        # Mixed precision scaler
        # Mixed precision scaler
        if torch.cuda.is_available():
            scaler_device = 'cuda'
        elif torch.backends.mps.is_available():
            scaler_device = 'mps'
        else:
            scaler_device = 'cpu'
            
        self.scaler = GradScaler(device=scaler_device, enabled=self.config.use_amp)

        pad_id = dataset.tokenizer.pad_id
        max_seq_len = self.model.module.config.max_seq_len if self.is_distributed else self.model.config.max_seq_len

        # Data sampler
        if self.is_distributed:
            self.sampler = DistributedSampler(
                dataset,
                num_replicas=self.config.world_size,
                rank=dist.get_rank(),
                shuffle=True,  # Handle shuffling in sampler
            )
            shuffle = False
        else:
            self.sampler = None
            shuffle = True

        self.data_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle, # False if using sampler
            sampler=self.sampler,
            collate_fn=lambda b: collate_fn(b, pad_id, max_seq_len),
            pin_memory=True if self.config.device == "cuda" else False,
            num_workers=4 if self.config.device == "cuda" else 0,
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)

        # Get vocabulary sizes for all fields
        self.vocab_sizes = dataset.tokenizer.get_vocab_sizes()

        self.history: Dict[str, Any] = {'mlm_loss': [], 'field_losses': {f: [] for f in FEATURE_FIELDS}}

    def train_epoch(self, verbose: bool = True) -> Tuple[float, Dict[str, float]]:
        """Run one MLM pretraining epoch.

        Returns:
            Tuple of (average total loss, dict of average per-field losses)
        """
        self.model.train()
        total_loss = 0.0
        field_losses = {field: 0.0 for field in FEATURE_FIELDS}
        n_batches = 0
        total_batches = len(self.data_loader)

        for batch_idx, batch in enumerate(self.data_loader):
            # Create MLM batch with labels for all fields
            mlm_batch = create_mlm_batch(
                batch,
                mask_prob=self.mask_prob,
                mask_token_id=self.dataset.tokenizer.mask_id,
                vocab_sizes=self.vocab_sizes,
            )

            # Move to device
            field_inputs = {
                k: v.to(self.device) for k, v in mlm_batch.items()
                if k.startswith('input_ids_')
            }
            attention_mask = mlm_batch['attention_mask'].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            # Mixed precision context
            # Determine device type string for autocast (e.g., 'cuda', 'mps', 'cpu')
            device_type = 'cuda' if 'cuda' in str(self.device) else ('mps' if 'mps' in str(self.device) else 'cpu')
            
            with autocast(device_type=device_type, enabled=self.config.use_amp):
                # Get logits for all fields

                # Note: if DDP, model is wrapped, so we call directly or check hierarchy
                # mlm_logits_dict = self.model.forward_mlm(field_inputs, attention_mask) if not self.is_distributed else self.model.module.forward_mlm(field_inputs, attention_mask)
                # FIX: Call forward() with mode='mlm' so DDP wrapper works
                mlm_logits_dict = self.model(field_inputs, attention_mask=attention_mask, mode='mlm')

                # Compute weighted sum of losses across all fields
                batch_loss: torch.Tensor = torch.tensor(0.0, device=self.device)
                valid_fields_count = 0

                for f in FEATURE_FIELDS:
                    logits = mlm_logits_dict[f]
                    labels = mlm_batch[f'mlm_labels_{f}'].to(self.device)
                    
                    # Skip if all labels are ignore_index (-100)
                    if (labels != -100).sum() == 0:
                        continue

                    field_loss = self.criterion(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                    )
                    
                    if torch.isnan(field_loss):
                        continue

                    weighted_loss = self.field_weights[f] * field_loss
                    batch_loss = batch_loss + weighted_loss
                    field_losses[f] += field_loss.item()
                    valid_fields_count += 1

                # Average across fields
                if valid_fields_count > 0:
                    loss = batch_loss / valid_fields_count
                else:
                    loss = torch.tensor(0.0, device=self.device, requires_grad=True)

                # Normalize loss for gradient accumulation
                loss = loss / self.config.grad_accum_steps

            # Backward pass with scaler
            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * self.config.grad_accum_steps # Scale back up for reporting
            n_batches += 1

            # Progress display (only on rank 0)
            if verbose and is_main_process():
                # Avoid division by zero if n_batches is 0 (shouldn't happen here but good practice)
                avg_loss_so_far = total_loss / max(1, n_batches)
                progress = (batch_idx + 1) / total_batches
                bar_len = 30
                filled = int(bar_len * progress)
                bar = '=' * filled + '>' + '.' * (bar_len - filled - 1)
                sys.stdout.write(f'\r  [{bar}] {batch_idx+1}/{total_batches} loss={avg_loss_so_far:.4f}')
                sys.stdout.flush()

        if verbose and is_main_process():
            sys.stdout.write('\n')
            sys.stdout.flush()

        avg_loss = total_loss / n_batches
        avg_field_losses = {field: loss / n_batches for field, loss in field_losses.items()}
        return avg_loss, avg_field_losses

    def train(self, epochs: Optional[int] = None, verbose: bool = True) -> Dict[str, Any]:
        """Run MLM pretraining.

        Args:
            epochs: Number of epochs (defaults to config.epochs)
            verbose: If True, print progress

        Returns:
            Training history with 'mlm_loss' and per-field losses
        """
        actual_epochs = epochs or self.config.epochs

        for epoch in range(actual_epochs):
            # Update sampler epoch for shuffling
            if self.dataset and self.is_distributed:
                 cast(DistributedSampler, self.sampler).set_epoch(epoch)

            if verbose and is_main_process():
                print(f"Epoch {epoch+1}/{actual_epochs}")
            mlm_loss, field_loss_dict = self.train_epoch(verbose=verbose)
            self.history['mlm_loss'].append(mlm_loss)
            for f, loss_val in field_loss_dict.items():
                self.history['field_losses'][f].append(loss_val)

            if verbose and is_main_process():
                print(f"  MLM Loss: {mlm_loss:.4f}")
                field_str = ", ".join(f"{f}={l:.3f}" for f, l in field_loss_dict.items())
                print(f"  Field losses: {field_str}")

        return self.history


class Trainer:
    """Training loop for multi-task style classifier with differential learning rates."""

    def __init__(
        self,
        model: StyleClassifier,
        train_dataset: StyleDataset,
        val_dataset: StyleDataset,
        config: Optional[TrainerConfig] = None,
        encoder_lr_factor: float = 0.1,
    ):
        """Initialize trainer.

        Args:
            model: StyleClassifier model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: TrainerConfig with hyperparameters
            encoder_lr_factor: Learning rate multiplier for encoder (vs classifier head).
                              Set < 1.0 to use smaller LR for pretrained encoder.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or TrainerConfig()
        self.encoder_lr_factor = encoder_lr_factor

        self.config = config or TrainerConfig()
        self.encoder_lr_factor = encoder_lr_factor

        # Setup device and distributed
        if self.config.world_size > 1:
            self.device = torch.device('cuda', self.config.local_rank)
            self.is_distributed = True
        else:
            self.device = torch.device(self.config.device)
            self.is_distributed = False

        self.model.to(self.device)

        # Wrap in DDP if distributed
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
                find_unused_parameters=True
            ) # type: ignore

        # Mixed precision scaler
        # Mixed precision scaler
        if torch.cuda.is_available():
            scaler_device = 'cuda'
        elif torch.backends.mps.is_available():
            scaler_device = 'mps'
        else:
            scaler_device = 'cpu'
            
        self.scaler = GradScaler(device=scaler_device, enabled=self.config.use_amp)

        # Data loaders with max_seq_len truncation

        pad_id = train_dataset.tokenizer.pad_id
        max_seq_len = self.model.module.config.max_seq_len if self.is_distributed else self.model.config.max_seq_len

        # Data samplers
        if self.is_distributed:
            self.train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.config.world_size,
                rank=dist.get_rank(),
                shuffle=True,
            )
            self.val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.config.world_size,
                rank=dist.get_rank(),
                shuffle=False, # Validation doesn't need shuffle, but distributing it speeds it up
            )
            train_shuffle = False
            val_shuffle = False
        else:
            self.train_sampler = None
            self.val_sampler = None
            train_shuffle = True
            val_shuffle = False

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=train_shuffle,
            sampler=self.train_sampler,
            collate_fn=lambda b: collate_fn(b, pad_id, max_seq_len),
            pin_memory=True if self.config.device == "cuda" else False,
            num_workers=4 if self.config.device == "cuda" else 0,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=val_shuffle,
            sampler=self.val_sampler,
            collate_fn=lambda b: collate_fn(b, pad_id, max_seq_len),
            pin_memory=True if self.config.device == "cuda" else False,
            num_workers=4 if self.config.device == "cuda" else 0,
        )

        # Loss functions with optional class weights
        if self.config.use_class_weights:
            formality_weights = train_dataset.get_formality_class_weights().to(self.device)
            gender_weights = train_dataset.get_gender_class_weights().to(self.device)
            grammaticality_weights = train_dataset.get_grammaticality_class_weights().to(self.device)
            self.formality_criterion = nn.CrossEntropyLoss(weight=formality_weights)
            self.gender_criterion = nn.CrossEntropyLoss(weight=gender_weights)
            self.grammaticality_criterion = nn.CrossEntropyLoss(weight=grammaticality_weights)
        else:
            self.formality_criterion = nn.CrossEntropyLoss()
            self.gender_criterion = nn.CrossEntropyLoss()
            self.grammaticality_criterion = nn.CrossEntropyLoss()

        # Optimizer with differential learning rates
        # Handle wrappped model
        model_module = self.model.module if self.is_distributed else self.model
        
        encoder_params = list(model_module.embedding.parameters()) + list(model_module.encoder.parameters())
        classifier_params = (
            list(model_module.formality_classifier.parameters()) +
            list(model_module.gender_classifier.parameters()) +
            list(model_module.grammaticality_classifier.parameters())
        )

        self.optimizer = Adam([
            {'params': encoder_params, 'lr': self.config.learning_rate * encoder_lr_factor},
            {'params': classifier_params, 'lr': self.config.learning_rate},
        ])

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.lr_scheduler_factor,
            patience=self.config.lr_scheduler_patience,
        )

        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_formality_loss': [],
            'train_gender_loss': [],
            'train_grammaticality_loss': [],
            'val_loss': [],
            'val_formality_loss': [],
            'val_gender_loss': [],
            'val_grammaticality_loss': [],
            'val_formality_accuracy': [],
            'val_gender_accuracy': [],
            'val_grammaticality_accuracy': [],
        }
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.start_epoch = 0  # For resumption

    def _batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move batch tensors to device and split into inputs/mask/labels."""
        field_inputs = {
            k: v.to(self.device) for k, v in batch.items()
            if k.startswith('input_ids_')
        }
        attention_mask = batch['attention_mask'].to(self.device)
        formality_labels = batch['formality_labels'].to(self.device)
        gender_labels = batch['gender_labels'].to(self.device)
        grammaticality_labels = batch['grammaticality_labels'].to(self.device)
        return field_inputs, attention_mask, formality_labels, gender_labels, grammaticality_labels

    def train_epoch(self, verbose: bool = True) -> Tuple[float, float, float, float]:
        """Run one training epoch.

        Returns:
            Tuple of (total_loss, formality_loss, gender_loss, grammaticality_loss)
        """
        self.model.train()
        total_loss = 0
        total_formality_loss = 0
        total_gender_loss = 0
        total_grammaticality_loss = 0
        n_batches = 0
        total_batches = len(self.train_loader)

        if total_batches == 0:
            print("  WARNING: No batches in train_loader!")
            return 0.0, 0.0, 0.0, 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            field_inputs, attention_mask, formality_labels, gender_labels, grammaticality_labels = self._batch_to_device(batch)

            self.optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision context
            device_type = 'cuda' if 'cuda' in str(self.device) else ('mps' if 'mps' in str(self.device) else 'cpu')
            
            with autocast(device_type=device_type, enabled=self.config.use_amp):
                formality_logits, gender_logits, grammaticality_logits = self.model(field_inputs, attention_mask)

                formality_loss = self.formality_criterion(formality_logits, formality_labels)
                gender_loss = self.gender_criterion(gender_logits, gender_labels)
                grammaticality_loss = self.grammaticality_criterion(grammaticality_logits, grammaticality_labels)

                # Weighted multi-task loss
                loss = (
                    self.config.formality_loss_weight * formality_loss +
                    self.config.gender_loss_weight * gender_loss +
                    self.config.grammaticality_loss_weight * grammaticality_loss
                )
                
                loss = loss / self.config.grad_accum_steps

            # Backward pass with scaler
            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * self.config.grad_accum_steps
            total_formality_loss += formality_loss.item()
            total_gender_loss += gender_loss.item()
            total_grammaticality_loss += grammaticality_loss.item()
            n_batches += 1

            # Progress display (only on main process)
            if verbose and is_main_process():
                avg_loss_so_far = total_loss / n_batches
                progress = (batch_idx + 1) / total_batches
                bar_len = 30
                filled = int(bar_len * progress)
                bar = '=' * filled + '>' + '.' * (bar_len - filled - 1)
                sys.stdout.write(f'\r  [{bar}] {batch_idx+1}/{total_batches} loss={avg_loss_so_far:.4f}')
                sys.stdout.flush()

        if verbose and is_main_process():
            sys.stdout.write('\n')

        return total_loss / n_batches, total_formality_loss / n_batches, total_gender_loss / n_batches, total_grammaticality_loss / n_batches

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def evaluate(self, return_mismatches: bool = False) -> Dict[str, Any]:
        """Evaluate on validation set.

        Args:
            return_mismatches: If True, return lists of misclassified examples

        Returns:
            Dictionary with losses, accuracies, and confusion matrices.
            If return_mismatches is True, also includes 'mismatches' dict.
        """
        self.model.eval()
        total_loss = 0
        total_formality_loss = 0
        total_gender_loss = 0
        total_grammaticality_loss = 0
        n_batches = 0

        all_formality_preds = []
        all_formality_labels = []
        all_gender_preds = []
        all_gender_labels = []
        all_grammaticality_preds: List[int] = []
        all_grammaticality_labels = []
        all_sentences = []
        all_kotograms = []

        for batch in self.val_loader:
            field_inputs, attention_mask, formality_labels, gender_labels, grammaticality_labels = self._batch_to_device(batch)

            formality_logits, gender_logits, grammaticality_logits = self.model(field_inputs, attention_mask)

            formality_loss = self.formality_criterion(formality_logits, formality_labels)
            gender_loss = self.gender_criterion(gender_logits, gender_labels)
            grammaticality_loss = self.grammaticality_criterion(grammaticality_logits, grammaticality_labels)
            loss = (
                self.config.formality_loss_weight * formality_loss +
                self.config.gender_loss_weight * gender_loss +
                self.config.grammaticality_loss_weight * grammaticality_loss
            )

            formality_preds = formality_logits.argmax(dim=-1)
            gender_preds = gender_logits.argmax(dim=-1)
            grammaticality_preds = grammaticality_logits.argmax(dim=-1)

            all_formality_preds.extend(formality_preds.cpu().tolist())
            all_formality_labels.extend(formality_labels.cpu().tolist())
            all_gender_preds.extend(gender_preds.cpu().tolist())
            all_gender_labels.extend(gender_labels.cpu().tolist())
            all_grammaticality_preds.extend(grammaticality_preds.cpu().tolist())
            all_grammaticality_labels.extend(grammaticality_labels.cpu().tolist())

            if return_mismatches:
                all_sentences.extend(batch.get('original_sentence', []))
                all_kotograms.extend(batch.get('kotogram', []))

            total_loss += loss.item()
            total_formality_loss += formality_loss.item()
            total_gender_loss += gender_loss.item()
            total_grammaticality_loss += grammaticality_loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_formality_loss = total_formality_loss / n_batches
        avg_gender_loss = total_gender_loss / n_batches
        avg_grammaticality_loss = total_grammaticality_loss / n_batches

        formality_accuracy = sum(
            p == l for p, l in zip(all_formality_preds, all_formality_labels)
        ) / len(all_formality_preds)
        gender_accuracy = sum(
            p == l for p, l in zip(all_gender_preds, all_gender_labels)
        ) / len(all_gender_preds)
        grammaticality_accuracy = sum(
            p == l for p, l in zip(all_grammaticality_preds, all_grammaticality_labels)
        ) / len(all_grammaticality_preds)

        # Build confusion matrices
        formality_confusion = [[0] * NUM_FORMALITY_CLASSES for _ in range(NUM_FORMALITY_CLASSES)]
        for pred, label in zip(all_formality_preds, all_formality_labels):
            formality_confusion[label][pred] += 1

        gender_confusion = [[0] * NUM_GENDER_CLASSES for _ in range(NUM_GENDER_CLASSES)]
        for pred, label in zip(all_gender_preds, all_gender_labels):
            gender_confusion[label][pred] += 1

        grammaticality_confusion = [[0] * NUM_GRAMMATICALITY_CLASSES for _ in range(NUM_GRAMMATICALITY_CLASSES)]
        for pred, label in zip(all_grammaticality_preds, all_grammaticality_labels):
            grammaticality_confusion[label][pred] += 1

        results = {
            'loss': avg_loss,
            'formality_loss': avg_formality_loss,
            'gender_loss': avg_gender_loss,
            'grammaticality_loss': avg_grammaticality_loss,
            'formality_accuracy': formality_accuracy,
            'gender_accuracy': gender_accuracy,
            'grammaticality_accuracy': grammaticality_accuracy,
            'formality_confusion': formality_confusion,
            'gender_confusion': gender_confusion,
            'grammaticality_confusion': grammaticality_confusion,
        }

        if return_mismatches:
            mismatches: Dict[str, List[Dict[str, Any]]] = {
                'formality': [],
                'gender': [],
                'grammaticality': []
            }

            for i, sent in enumerate(all_sentences):
                # Formality
                if all_formality_preds[i] != all_formality_labels[i]:
                    mismatches['formality'].append({
                        'sentence': sent,
                        'predicted': FORMALITY_ID_TO_LABEL[all_formality_preds[i]].value,
                        'actual': FORMALITY_ID_TO_LABEL[all_formality_labels[i]].value,
                        'kotogram': all_kotograms[i] if i < len(all_kotograms) else '',
                    })
                
                # Gender
                if all_gender_preds[i] != all_gender_labels[i]:
                    mismatches['gender'].append({
                        'sentence': sent,
                        'predicted': GENDER_ID_TO_LABEL[all_gender_preds[i]].value,
                        'actual': GENDER_ID_TO_LABEL[all_gender_labels[i]].value,
                        'kotogram': all_kotograms[i] if i < len(all_kotograms) else '',
                    })

                # Grammaticality
                if all_grammaticality_preds[i] != all_grammaticality_labels[i]:
                    # Map 0/1 to labels
                    pred_label = "grammatic" if all_grammaticality_preds[i] == 1 else "agrammatic"
                    actual_label = "grammatic" if all_grammaticality_labels[i] == 1 else "agrammatic"
                    mismatches['grammaticality'].append({
                        'sentence': sent,
                        'predicted': pred_label,
                        'actual': actual_label,
                        'kotogram': all_kotograms[i] if i < len(all_kotograms) else '',
                    })
            
            results['mismatches'] = mismatches

        return results

    def train(
        self,
        verbose: bool = True,
        checkpoint_dir: Optional[str] = None,
        checkpoint_args: Optional[Any] = None,
        model_config: Optional[ModelConfig] = None,
    ) -> Dict[str, List[float]]:
        """Run full training loop.

        Args:
            verbose: Print progress
            checkpoint_dir: Directory to save checkpoints (if provided)
            checkpoint_args: Args object to save in checkpoint
            model_config: Model config to save in checkpoint
        """
        for epoch in range(self.start_epoch, self.config.epochs):
            if verbose:
                print(f"Epoch {epoch+1}/{self.config.epochs}")
            train_loss, train_formality_loss, train_gender_loss, train_grammaticality_loss = self.train_epoch(verbose=verbose)
            eval_results = self.evaluate()

            self.scheduler.step(eval_results['loss'])

            self.history['train_loss'].append(train_loss)
            self.history['train_formality_loss'].append(train_formality_loss)
            self.history['train_gender_loss'].append(train_gender_loss)
            self.history['train_grammaticality_loss'].append(train_grammaticality_loss)
            self.history['val_loss'].append(eval_results['loss'])
            self.history['val_formality_loss'].append(eval_results['formality_loss'])
            self.history['val_gender_loss'].append(eval_results['gender_loss'])
            self.history['val_grammaticality_loss'].append(eval_results['grammaticality_loss'])
            self.history['val_formality_accuracy'].append(eval_results['formality_accuracy'])
            self.history['val_gender_accuracy'].append(eval_results['gender_accuracy'])
            self.history['val_grammaticality_accuracy'].append(eval_results['grammaticality_accuracy'])

            if verbose:
                print(f"  Train Loss: {train_loss:.4f} (formality={train_formality_loss:.4f}, gender={train_gender_loss:.4f}, gram={train_grammaticality_loss:.4f})")
                print(f"  Val Loss: {eval_results['loss']:.4f} (formality={eval_results['formality_loss']:.4f}, gender={eval_results['gender_loss']:.4f}, gram={eval_results['grammaticality_loss']:.4f})")
                print(f"  Val Acc: formality={eval_results['formality_accuracy']:.4f}, gender={eval_results['gender_accuracy']:.4f}, gram={eval_results['grammaticality_accuracy']:.4f}")
                enc_lr = self.optimizer.param_groups[0]['lr']
                cls_lr = self.optimizer.param_groups[1]['lr']
                print(f"  LR: encoder={enc_lr:.2e}, classifier={cls_lr:.2e}")

            # Early stopping
            if eval_results['loss'] < self.best_val_loss:
                self.best_val_loss = eval_results['loss']
                self.patience_counter = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

            # Save checkpoint after each epoch
            if checkpoint_dir and checkpoint_args and model_config:
                save_checkpoint(
                    checkpoint_dir,
                    self.model,
                    self.train_dataset.tokenizer,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    self.history,
                    self.best_val_loss,
                    self.patience_counter,
                    self.best_state,
                    checkpoint_args,
                    model_config,
                )

        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            self.model.to(self.device)

        return self.history

    def print_confusion_matrices(self, save_dir: Optional[str] = None) -> None:
        """Print confusion matrices for all tasks and optionally save mismatches."""
        eval_results = self.evaluate(return_mismatches=True)

        # Formality confusion matrix
        formality_labels = [FORMALITY_ID_TO_LABEL[i].value for i in range(NUM_FORMALITY_CLASSES)]
        print("\nFormality Confusion Matrix:")
        header = "True\\Pred".ljust(25) + " ".join(l[:8].ljust(10) for l in formality_labels)
        print(header)
        print("-" * len(header))
        for i, row in enumerate(eval_results['formality_confusion']):
            row_label = formality_labels[i].ljust(25)
            row_values = " ".join(str(v).ljust(10) for v in row)
            print(f"{row_label}{row_values}")

        # Gender confusion matrix
        gender_labels = [GENDER_ID_TO_LABEL[i].value for i in range(NUM_GENDER_CLASSES)]
        print("\nGender Confusion Matrix:")
        header = "True\\Pred".ljust(25) + " ".join(l[:8].ljust(10) for l in gender_labels)
        print(header)
        print("-" * len(header))
        for i, row in enumerate(eval_results['gender_confusion']):
            row_label = gender_labels[i].ljust(25)
            row_values = " ".join(str(v).ljust(10) for v in row)
            print(f"{row_label}{row_values}")

        # Grammaticality confusion matrix
        grammaticality_labels = ["agrammatic", "grammatic"]
        print("\nGrammaticality Confusion Matrix:")
        header = "True\\Pred".ljust(25) + " ".join(l[:10].ljust(12) for l in grammaticality_labels)
        print(header)
        print("-" * len(header))
        for i, row in enumerate(eval_results['grammaticality_confusion']):
            row_label = grammaticality_labels[i].ljust(25)
            row_values = " ".join(str(v).ljust(12) for v in row)
            print(f"{row_label}{row_values}")

        # Save mismatches if requested
        if save_dir and 'mismatches' in eval_results:
            import csv
            import os
            
            # Save Grammaticality Mismatches
            mismatches = eval_results['mismatches']
            if mismatches['grammaticality']:
                # Sort by sentence for reproducible order
                mismatches['grammaticality'].sort(key=lambda x: x.get('sentence', ''))
                
                out_path = os.path.join(save_dir, 'grammaticality_confusion.csv')
                with open(out_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=['sentence', 'predicted', 'actual', 'kotogram'], delimiter='\t')
                    writer.writeheader()
                    writer.writerows(mismatches['grammaticality'])
                print(f"\nSaved {len(mismatches['grammaticality'])} grammaticality mismatches to {out_path}")

            # Save Other Mismatches (optional, maybe just formality/gender if they have many errors)
            if mismatches['formality']:
                # Sort by sentence for reproducible order
                mismatches['formality'].sort(key=lambda x: x.get('sentence', ''))
                
                out_path = os.path.join(save_dir, 'formality_confusion.csv')
                with open(out_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=['sentence', 'predicted', 'actual', 'kotogram'], delimiter='\t')
                    writer.writeheader()
                    writer.writerows(mismatches['formality'])
                print(f"Saved {len(mismatches['formality'])} formality mismatches to {out_path}")
                
            if mismatches['gender']:
                # Sort by sentence for reproducible order
                mismatches['gender'].sort(key=lambda x: x.get('sentence', ''))

                out_path = os.path.join(save_dir, 'gender_confusion.csv')
                with open(out_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=['sentence', 'predicted', 'actual', 'kotogram'], delimiter='\t')
                    writer.writeheader()
                    writer.writerows(mismatches['gender'])
                print(f"Saved {len(mismatches['gender'])} gender mismatches to {out_path}")

    def restore_from_checkpoint(self, checkpoint: Dict[str, Any], reset_optimizer: bool = False) -> None:
        """Restore training state from checkpoint.

        Args:
            checkpoint: Checkpoint dict from load_checkpoint()
            reset_optimizer: If True, do not load optimizer/scheduler state (useful if model architecture changed)
        """
        if not reset_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print("  Note: Resetting optimizer and scheduler state due to model architecture change (vocabulary expansion)")
        
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        self.patience_counter = checkpoint['patience_counter']
        self.best_state = checkpoint['best_state']
        self.start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch

        print(f"Restored training state from epoch {checkpoint['epoch'] + 1}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  Patience counter: {self.patience_counter}")


def save_model(
    model: StyleClassifier,
    tokenizer: Tokenizer,
    path: str,
    config: Optional[ModelConfig] = None,
    fp16: bool = False,
    fp8: bool = False,
) -> None:
    """Save trained model, tokenizer, and config.

    Args:
        model: The trained model
        tokenizer: The tokenizer used for encoding
        path: Directory to save to
        config: Optional model config (uses model.config if not provided)
        fp16: If True, convert model weights to float16 for smaller size
        fp8: If True, convert model weights to float8 for even smaller size
             (requires PyTorch 2.1+, experimental)
    """
    import os
    os.makedirs(path, exist_ok=True)

    # Save model weights
    if fp8:
        # Convert to float8 for smallest model size
        if not hasattr(torch, 'float8_e4m3fn'):
            raise RuntimeError("FP8 requires PyTorch 2.1+. Use --fp16 instead.")
        # MPS doesn not support float8, so move to CPU first
        state_dict = {k: v.cpu().to(torch.float8_e4m3fn) if v.dtype == torch.float32 else v.cpu()
                      for k, v in model.state_dict().items()}
        torch.save(state_dict, os.path.join(path, 'model.pt'))
    elif fp16:
        # Convert to float16 for smaller model size
        state_dict = {k: v.cpu().half() if v.dtype == torch.float32 else v.cpu()
                      for k, v in model.state_dict().items()}
        torch.save(state_dict, os.path.join(path, 'model.pt'))
    else:
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(state_dict, os.path.join(path, 'model.pt'))

    # Save tokenizer
    tokenizer.save(os.path.join(path, 'tokenizer.json'))

    # Save config
    config = config or model.config
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    # Save label mappings
    formality_label_map = {k.value: v for k, v in FORMALITY_LABEL_TO_ID.items()}
    gender_label_map = {k.value: v for k, v in GENDER_LABEL_TO_ID.items()}
    grammaticality_label_map = {'agrammatic': 0, 'grammatic': 1}
    with open(os.path.join(path, 'labels.json'), 'w') as f:
        json.dump({
            'formality': formality_label_map,
            'gender': gender_label_map,
            'grammaticality': grammaticality_label_map,
        }, f, indent=2)

    # Mark as feature-based multi-task model
    with open(os.path.join(path, 'model_type.txt'), 'w') as f:
        f.write('style-multitask')


def save_checkpoint(
    path: str,
    model: StyleClassifier,
    tokenizer: Tokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    history: Dict[str, List[float]],
    best_val_loss: float,
    patience_counter: int,
    best_state: Optional[Dict[str, torch.Tensor]],
    args: Any,
    model_config: ModelConfig,
) -> None:
    """Save training checkpoint for resumption.

    Args:
        path: Directory to save checkpoint
        model: Current model state
        tokenizer: Tokenizer
        optimizer: Optimizer state
        scheduler: LR scheduler state
        epoch: Current epoch number (0-indexed, completed epochs)
        history: Training history dict
        best_val_loss: Best validation loss seen
        patience_counter: Current patience counter for early stopping
        best_state: Best model state dict
        args: Command line arguments (for reproducing settings)
        model_config: Model configuration
    """
    import os
    os.makedirs(path, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
        'best_state': best_state,
        'args': {
            'data': args.data,
            'extra_data': args.extra_data,
            'agrammatic_data': args.agrammatic_data,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'embed_dim': args.embed_dim,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
            'learning_rate': args.learning_rate,
            'encoder_lr_factor': args.encoder_lr_factor,
            'formality_weight': args.formality_weight,
            'gender_weight': args.gender_weight,
            'grammaticality_weight': args.grammaticality_weight,
            'fp16': args.fp16,
            'fp8': args.fp8,
            'exclude_features': args.exclude_features,
            'percent': args.percent,
        },
    }
    torch.save(checkpoint, os.path.join(path, 'checkpoint.pt'))

    # Also save tokenizer and config (needed to reconstruct model)
    tokenizer.save(os.path.join(path, 'tokenizer.json'))
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump(model_config.to_dict(), f, indent=2)

    print(f"  Checkpoint saved at epoch {epoch + 1}")


def load_checkpoint(
    path: str,
    device: Optional[str] = None,
) -> Tuple[StyleClassifier, Tokenizer, Dict[str, Any]]:
    """Load training checkpoint for resumption.

    Args:
        path: Directory containing checkpoint
        device: Device to load model to

    Returns:
        Tuple of (model, tokenizer, checkpoint_dict)
        checkpoint_dict contains: epoch, optimizer_state_dict, scheduler_state_dict,
                                  history, best_val_loss, patience_counter, best_state, args
    """
    import os

    checkpoint_path = os.path.join(path, 'checkpoint.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    # Load config and tokenizer
    with open(os.path.join(path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    config = ModelConfig.from_dict(config_dict)
    tokenizer = Tokenizer.load(os.path.join(path, 'tokenizer.json'))

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device or 'cpu')

    # Reconstruct model
    model = StyleClassifier(config)

    # Filter out MLM head weights (present when model was trained with --pretrain-mlm)
    model_state = checkpoint['model_state_dict']
    model_state = {k: v for k, v in model_state.items() if not k.startswith('mlm_head.')}
    model.load_state_dict(model_state)

    if device:
        model.to(device)

    return model, tokenizer, checkpoint





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train style classifier (formality + gender)")
    parser.add_argument("--data", type=str, default="data/jpn_sentences.tsv",
                        help="Path to TSV file with Japanese sentences")
    parser.add_argument("--output", type=str, default="models/style",
                        help="Output directory for trained model")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to use (for testing)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--embed-dim", type=int, default=192,
                        help="Model dimension (d_model)")
    parser.add_argument("--hidden-dim", type=int, default=384,
                        help="Hidden layer dimension")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="Number of encoder layers")
    parser.add_argument("--num-heads", type=int, default=6,
                        help="Number of attention heads")
    parser.add_argument("--pretrain-mlm", action="store_true",
                        help="Pre-train with masked language modeling")
    parser.add_argument("--pretrain-epochs", type=int, default=5,
                        help="MLM pretraining epochs")
    parser.add_argument("--encoder-lr-factor", type=float, default=0.1,
                        help="Learning rate factor for encoder during fine-tuning")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Base learning rate")
    parser.add_argument("--formality-weight", type=float, default=1.0,
                        help="Loss weight for formality task")
    parser.add_argument("--gender-weight", type=float, default=1.0,
                        help="Loss weight for gender task")
    parser.add_argument("--grammaticality-weight", type=float, default=1.0,
                        help="Loss weight for grammaticality task")
    parser.add_argument("--extra-data", type=str, default=None,
                        help="Path to additional TSV file with training data (e.g., unpragmatic examples)")
    parser.add_argument("--agrammatic-data", type=str, default=None,
                        help="Path to TSV file with agrammatic sentences (for grammaticality training)")
    parser.add_argument("--exclude-features", type=str, default="",
                        help="Comma-separated list of features to exclude (for ablation study). "
                             f"Valid: {','.join(ALL_FEATURE_FIELDS)}")
    parser.add_argument("--fp16", action="store_true", default=None,
                        help="Save model in float16 precision (half size, minimal accuracy loss)")
    parser.add_argument("--fp8", action="store_true", default=None,
                        help="Save model in float8 precision (quarter size, requires PyTorch 2.1+)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint in output directory")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain from scratch using parameters from existing checkpoint")
    parser.add_argument("--confusion", action="store_true",
                        help="Print confusion matrices for existing model and exit")
    parser.add_argument("--percent", type=float, default=None,
                        help="Percentage of data to use for training (1-100)")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Gradient accumulation steps for larger effective batch size")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of data loader workers")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank for distributed training (usually passed by torchrun)")
    parser.add_argument("--preprocess-only", action="store_true",
                        help="Exit after loading and caching data (for multi-stage pipelines)")



    args = parser.parse_args()
    timings['args_parsing'] = time.time() - script_start_time


    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    # If args.local_rank is set by argument (torchrun often sets this), prefer it, 
    # but our helper checks env vars which torchrun also sets.
    
    # Log device information
    if is_main_process():
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            print(f"\nDevice:         CUDA ({count} devices available)")
            print(f"  Name:         {name}")
            if world_size > 1:
                print(f"  Distributed:  d_model=DDP, {world_size} gpus, global_batch={args.batch_size * world_size * args.grad_accum_steps}")
                print(f"  Mixed Prec:   {'On' if args.fp16 else 'Off'}")
        elif torch.backends.mps.is_available():
             print(f"\nDevice:         MPS (Apple Silicon)")
        else:
             print(f"\nDevice:         CPU")
             print(f"  Info:         Training will be slow. CUDA or MPS not found.")


    # Handle resume from checkpoint or retrain or confusion logic
    checkpoint = None
    if args.resume or args.retrain or args.confusion:
        import os
        checkpoint_path = os.path.join(args.output, 'checkpoint.pt')
        import os
        checkpoint_path = os.path.join(args.output, 'checkpoint.pt')
        if os.path.exists(checkpoint_path):
            # First, peek at saved args to restore feature exclusion before loading model
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            saved_args = checkpoint_data['args']

            # Restore feature exclusion BEFORE loading model
            saved_exclude = saved_args.get('exclude_features', '')
            if saved_exclude:
                excluded = [f.strip() for f in saved_exclude.split(',') if f.strip()]
                set_excluded_features(excluded)
                if is_main_process():
                    print(f"Restored feature exclusion: {excluded}")
                    print(f"Active features: {FEATURE_FIELDS}")

            if args.resume:
                if is_main_process():
                    print(f"Resuming from checkpoint in {args.output}...")
                model, tokenizer, checkpoint = load_checkpoint(args.output)
            elif args.confusion:
                print(f"Loading best model for evaluation from {args.output}...")
                model, tokenizer = load_model(args.output)
                checkpoint = checkpoint_data
            else:
                if is_main_process():
                    print(f"Retraining from scratch using parameters from {args.output}...")
                pass

            # Override args with saved args (but keep epochs from command line to allow extending)
            if is_main_process():
                print(f"  Using saved parameters:")
                print(f"    data: {saved_args['data']}")
            # ... skipping full print for brevity in DDP

            print(f"    embed_dim: {saved_args['embed_dim']}")
            print(f"    hidden_dim: {saved_args['hidden_dim']}")
            print(f"    num_layers: {saved_args['num_layers']}")
            print(f"    num_heads: {saved_args['num_heads']}")
            print(f"    learning_rate: {saved_args['learning_rate']}")
            if args.resume:
                assert checkpoint is not None
                print(f"  Resuming from epoch {checkpoint['epoch'] + 1}, training to epoch {args.epochs}")
            elif args.confusion:
                print(f"  Evaluating best model from {args.output}")
            else:
                print(f"  Retraining from epoch 0 to {args.epochs}")

            # Update args with saved values (except epochs which can be extended)
            # Note: We do NOT restore data paths (data, extra_data, agrammatic_data) to allow
            # resuming training even if data files have moved or been reorganized, provided
            # valid paths are passed via command line.
            args.embed_dim = saved_args['embed_dim']
            args.hidden_dim = saved_args['hidden_dim']
            args.num_layers = saved_args['num_layers']
            args.num_heads = saved_args['num_heads']
            args.learning_rate = saved_args['learning_rate']
            args.encoder_lr_factor = saved_args['encoder_lr_factor']
            args.formality_weight = saved_args['formality_weight']
            args.gender_weight = saved_args['gender_weight']
            args.grammaticality_weight = saved_args['grammaticality_weight']
            args.grammaticality_weight = saved_args['grammaticality_weight']
            args.exclude_features = saved_exclude
            
            # Sticky flags: restore only if not explicitly set on command line
            if args.percent is None:
                args.percent = saved_args.get('percent', None)
                if args.percent is not None:
                    print(f"  Restored flag: --percent {args.percent}")
            
            # Sticky flags: restore only if not explicitly set on command line (i.e. None)
            if args.fp16 is None:
                args.fp16 = saved_args.get('fp16', False)
                if args.fp16:
                    print(f"  Restored flag: --fp16")
            elif args.fp16 is False:
                 # argparse 'store_true' with default=None sets False if not provided?
                 # No, 'store_true' only stores True if present.
                 # If default is None, and flag is absent, it remains None.
                 # Wait, let's verify argparse behavior.
                 # parser.add_argument('--foo', action='store_true', default=None)
                 # args = parser.parse_args([]) -> args.foo is None
                 # args = parser.parse_args(['--foo']) -> args.foo is True
                 pass

            if args.fp8 is None:
                args.fp8 = saved_args.get('fp8', False)
                if args.fp8:
                    print(f"  Restored flag: --fp8")

            # Ensure they are boolean False if still None (and not restored to True)
            if args.fp16 is None: args.fp16 = False
            if args.fp8 is None: args.fp8 = False
        else:
            if args.confusion:
                print(f"Error: No checkpoint found at {checkpoint_path}, cannot print confusion matrix.")
                import sys
                sys.exit(1)

            print(f"No checkpoint found at {checkpoint_path}, starting fresh training")
            args.resume = False
            # args.retrain = False # If no checkpoint, retrain just means train normally


    # Handle feature exclusion (for new training, not resume)
    if args.exclude_features and not checkpoint:
        excluded = [f.strip() for f in args.exclude_features.split(',') if f.strip()]
        set_excluded_features(excluded)
        if is_main_process():
            print(f"Feature ablation: excluding {excluded}")
            print(f"Active features: {FEATURE_FIELDS}")

    # Build list of data files and their grammaticality labels
    # grammatic (1) = normal sentences, agrammatic (0) = ungrammatical sentences
    data_files = [args.data]
    grammaticality_labels = [1]  # jpn_sentences.tsv is grammatic
    if args.extra_data:
        data_files.append(args.extra_data)
        grammaticality_labels.append(0)  # unpragmatic_sentences.tsv is agrammatic (unpragmatic = ungrammatical)
    if args.agrammatic_data:
        data_files.append(args.agrammatic_data)
        grammaticality_labels.append(0)  # agrammatic sentences

    # Track if vocabulary grew during data loading/resume
    vocab_grew = False

    # Check for sentence overlap between grammatic and agrammatic data
    # (Optimized: skips check if files haven't changed since last successful validation)
    t_check_start = time.time()
    
    validation_cache_path = os.path.join(os.path.dirname(args.output) if args.output else ".", ".cache", "data_validation_state.json")
    os.makedirs(os.path.dirname(validation_cache_path), exist_ok=True)

    def get_file_fingerprint(path: str) -> Optional[Dict[str, Any]]:
        if not path or not os.path.exists(path):
            return None
        stat = os.stat(path)
        return {'mtime': stat.st_mtime, 'size': stat.st_size}

    # Helper to clean read sentences
    def read_sentences_to_set(path: str, target_set: set):
        if not path or not os.path.exists(path):
            return
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 3 and row[1] == 'jpn':
                    target_set.add(row[2])

    # Current state
    current_state = {
        'data': get_file_fingerprint(args.data),
        'extra_data': get_file_fingerprint(args.extra_data) if args.extra_data else None,
        'agrammatic_data': get_file_fingerprint(args.agrammatic_data) if args.agrammatic_data else None,
    }

    # Check cache
    files_changed = True
    if os.path.exists(validation_cache_path):
        try:
            with open(validation_cache_path, 'r') as f:
                cached_state = json.load(f)
            if cached_state == current_state:
                files_changed = False
        except Exception:
            pass  # Ignore cache read errors

    if not files_changed:
        if is_main_process():
            print("Data validation: Files unchanged, skipping overlap check.")
    else:
        # Synchronization check: Only rank 0 does the check, others wait
        # This prevents race conditions on the cache file and printing
        if is_main_process():
            print("Data validation: Checking for sentence overlap...")
            grammatic_sentences = set()
            agrammatic_sentences = set()

            # Read grammatic
            read_sentences_to_set(args.data, grammatic_sentences)
            
            # Read agrammatic
            if args.extra_data:
                read_sentences_to_set(args.extra_data, agrammatic_sentences)
            if args.agrammatic_data:
                read_sentences_to_set(args.agrammatic_data, agrammatic_sentences)
                
            # Check intersection
            overlap = grammatic_sentences.intersection(agrammatic_sentences)
            if overlap:
                print(f"\nERROR: Found {len(overlap)} sentences appearing in both grammatic and agrammatic datasets.")
                print("This contamination invalidates the training assumption.")
                print("Examples:")
                for i, sent in enumerate(list(overlap)[:5]):
                    print(f"  {i+1}. {sent}")
                sys.exit(1) # This will kill rank 0. Other ranks will likely timeout or die.
            
            # Save state if successful
            try:
                with open(validation_cache_path, 'w') as f:
                    json.dump(current_state, f)
                print("Data validation: Passed and cached.")
            except Exception as e:
                print(f"Warning: Failed to cache validation state: {e}")
        
        # Wait for rank 0 to complete validation
        if torch.distributed.is_available() and torch.distributed.is_initialized():
             torch.distributed.barrier()

        
    timings['overlap_check'] = time.time() - t_check_start

    # Load data: if doing MLM pretraining, first load unlabeled data for pretraining,
    # then load labeled data for fine-tuning

    # Skip model creation if resuming (model already loaded)
    t_data_start = time.time()
    if (args.resume or args.confusion) and checkpoint is not None:
        # Load datasets with existing tokenizer
        # Unfreeze tokenizer to allow new vocabulary
        old_vocab_sizes = tokenizer.get_vocab_sizes()
        tokenizer._frozen = False

        print("\nLoading data (tokenizer unfrozen for new vocabulary)...")
        if len(data_files) > 1:
            if is_main_process():
                dataset = StyleDataset.from_multiple_tsv(
                    data_files,
                    tokenizer,
                    labeled=True,
                    grammaticality_labels=grammaticality_labels,
                    max_samples=args.max_samples,
                    sample_ratio=args.percent / 100.0 if args.percent else 1.0,
                )
            
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
                
            if not is_main_process():
                dataset = StyleDataset.from_multiple_tsv(
                    data_files,
                    tokenizer,
                    labeled=True,
                    grammaticality_labels=grammaticality_labels,
                    max_samples=args.max_samples,
                    sample_ratio=args.percent / 100.0 if args.percent else 1.0,
                    verbose=False,
                )
        else:
            if is_main_process():
                dataset = StyleDataset.from_tsv(
                    data_files[0],
                    tokenizer,
                    labeled=True,
                    max_samples=args.max_samples,
                    sample_ratio=args.percent / 100.0 if args.percent else 1.0,
                )
            
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
                
            if not is_main_process():
                dataset = StyleDataset.from_tsv(
                    data_files[0],
                    tokenizer,
                    labeled=True,
                    max_samples=args.max_samples,
                    sample_ratio=args.percent / 100.0 if args.percent else 1.0,
                    verbose=False,
                )

        train_data, val_data, test_data = dataset.split()

        # Check if vocabulary grew and resize embeddings if needed
        new_vocab_sizes = tokenizer.get_vocab_sizes()
        vocab_grew = any(new_vocab_sizes[f] > old_vocab_sizes[f] for f in FEATURE_FIELDS)

        if vocab_grew:
            print("\nResizing embeddings for new vocabulary...")
            resized = model.resize_embeddings(new_vocab_sizes)
            for f_name, count in resized.items():
                if count > 0:
                    print(f"  {f_name}: +{count} tokens ({old_vocab_sizes[f_name]} -> {new_vocab_sizes[f_name]})")

            # Update model config with new vocab sizes
            model_config = model.config
        else:
            print("\nNo new vocabulary tokens found.")
            model_config = model.config
        
        timings['data_loading'] = time.time() - t_data_start

    elif args.pretrain_mlm:
        # MLM pretraining should only use grammatical sentences
        # The encoder should learn what correct Japanese looks like
        # Agrammatic data is only used during fine-tuning for classification
        grammatic_files = [args.data]  # Only primary data file (grammatical)
        print("Loading grammatical data for MLM pretraining...")
        print(f"  (agrammatic data excluded from pretraining, will be used in fine-tuning)")
        tokenizer = Tokenizer()
        if is_main_process():
            unlabeled_dataset = StyleDataset.from_tsv(
                grammatic_files[0],
                tokenizer,
                max_samples=args.max_samples,
                verbose=True,
                labeled=False,  # No labels needed for pretraining
                sample_ratio=args.percent / 100.0 if args.percent else 1.0,
            )
        
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            
        if not is_main_process():
            unlabeled_dataset = StyleDataset.from_tsv(
                grammatic_files[0],
                tokenizer,
                max_samples=args.max_samples,
                verbose=False,
                labeled=False,  # No labels needed for pretraining
                sample_ratio=args.percent / 100.0 if args.percent else 1.0,
            )
        # Note: tokenizer is frozen after from_tsv

        # Model config (vocab is now fixed)
        excluded = [f.strip() for f in args.exclude_features.split(',') if f.strip()] if args.exclude_features else []
        model_config = ModelConfig(
            vocab_sizes=tokenizer.get_vocab_sizes(),
            num_formality_classes=NUM_FORMALITY_CLASSES,
            num_gender_classes=NUM_GENDER_CLASSES,
            num_grammaticality_classes=NUM_GRAMMATICALITY_CLASSES,
            d_model=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            excluded_features=excluded,
        )

        print("\nCreating model with MLM head...")
        model = StyleClassifierWithMLM(model_config)

        # MLM pretraining on unlabeled data
        if is_main_process():
            print("\nStarting MLM pretraining on unlabeled data...")
        
        pretrain_config = TrainerConfig(
            epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
            use_amp=args.fp16 if args.fp16 is not None else False,
            local_rank=local_rank,
            world_size=world_size,
            grad_accum_steps=args.grad_accum_steps
        )
        mlm_trainer = MLMTrainer(model, unlabeled_dataset, pretrain_config)
        mlm_trainer.train(epochs=args.pretrain_epochs, verbose=is_main_process())

        # Reset classifier heads for fine-tuning
        print("\nReinitializing classifier heads for fine-tuning...")
        model.reset_classifier()

        # Now load labeled dataset - may include new vocabulary from agrammatic data
        print("\nLoading labeled data for fine-tuning...")
        # Save old vocab sizes before loading new data
        old_vocab_sizes = tokenizer.get_vocab_sizes()
        # Unfreeze tokenizer to allow vocabulary expansion
        tokenizer._frozen = False
        if len(data_files) > 1:
            if is_main_process():
                labeled_dataset = StyleDataset.from_multiple_tsv(
                    data_files,
                    tokenizer,
                    max_samples=args.max_samples,
                    verbose=True,
                    labeled=True,
                    grammaticality_labels=grammaticality_labels,
                    sample_ratio=args.percent / 100.0 if args.percent else 1.0,
                )
            
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
                
            if not is_main_process():
                labeled_dataset = StyleDataset.from_multiple_tsv(
                    data_files,
                    tokenizer,
                    max_samples=args.max_samples,
                    verbose=False,
                    labeled=True,
                    grammaticality_labels=grammaticality_labels,
                    sample_ratio=args.percent / 100.0 if args.percent else 1.0,
                )
        else:
            if is_main_process():
                labeled_dataset = StyleDataset.from_tsv(
                    args.data,
                    tokenizer,
                    max_samples=args.max_samples,
                    verbose=True,
                    labeled=True,
                    sample_ratio=args.percent / 100.0 if args.percent else 1.0,
                )
            
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
                
            if not is_main_process():
                labeled_dataset = StyleDataset.from_tsv(
                    args.data,
                    tokenizer,
                    max_samples=args.max_samples,
                    verbose=False,
                    labeled=True,
                    sample_ratio=args.percent / 100.0 if args.percent else 1.0,
                )
        train_data, val_data, test_data = labeled_dataset.split()

        # Check if vocabulary grew and resize embeddings if needed
        new_vocab_sizes = tokenizer.get_vocab_sizes()
        vocab_grew = any(new_vocab_sizes[f] > old_vocab_sizes[f] for f in FEATURE_FIELDS)

        if vocab_grew:
            print("\nResizing embeddings for expanded vocabulary...")
            resized = model.resize_embeddings(new_vocab_sizes)
            for field_name, count in resized.items():
                if count > 0:
                    print(f"  {field_name}: +{count} tokens ({old_vocab_sizes[field_name]} -> {new_vocab_sizes[field_name]})")
            # Update model config
            model_config = ModelConfig(
                vocab_sizes=new_vocab_sizes,
                num_formality_classes=NUM_FORMALITY_CLASSES,
                num_gender_classes=NUM_GENDER_CLASSES,
                num_grammaticality_classes=NUM_GRAMMATICALITY_CLASSES,
                d_model=args.embed_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                excluded_features=excluded,
            )
    else:
        print("Loading data...")
        tokenizer = Tokenizer()
        if len(data_files) > 1:
            if is_main_process():
                dataset = StyleDataset.from_multiple_tsv(
                    data_files,
                    tokenizer,
                    max_samples=args.max_samples,
                    verbose=True,
                    grammaticality_labels=grammaticality_labels,
                    sample_ratio=args.percent / 100.0 if args.percent else 1.0,
                )
            
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            
            if not is_main_process():
                dataset = StyleDataset.from_multiple_tsv(
                    data_files,
                    tokenizer,
                    max_samples=args.max_samples,
                    verbose=False,
                    grammaticality_labels=grammaticality_labels,
                    sample_ratio=args.percent / 100.0 if args.percent else 1.0,
                )
        else:
            if is_main_process():
                dataset = StyleDataset.from_tsv(
                    args.data,
                    tokenizer,
                    max_samples=args.max_samples,
                    verbose=True,
                    sample_ratio=args.percent / 100.0 if args.percent else 1.0,
                )
            
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
                
            if not is_main_process():
                dataset = StyleDataset.from_tsv(
                    args.data,
                    tokenizer,
                    max_samples=args.max_samples,
                    verbose=False,
                    sample_ratio=args.percent / 100.0 if args.percent else 1.0,
                )
        train_data, val_data, test_data = dataset.split()
        timings['data_loading'] = time.time() - t_data_start

        # Model config
        excluded = [f.strip() for f in args.exclude_features.split(',') if f.strip()] if args.exclude_features else []
        model_config = ModelConfig(
            vocab_sizes=tokenizer.get_vocab_sizes(),
            num_formality_classes=NUM_FORMALITY_CLASSES,
            num_gender_classes=NUM_GENDER_CLASSES,
            num_grammaticality_classes=NUM_GRAMMATICALITY_CLASSES,
            d_model=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            excluded_features=excluded,
        )

        print("\nCreating model...")
        t_model_start = time.time()
        model = StyleClassifier(model_config)
        model = StyleClassifier(model_config)
        timings['model_creation'] = time.time() - t_model_start

    # Preprocessing only mode: Exit after data is loaded and cached
    if args.preprocess_only:
        if is_main_process():
            print("\nPreprocessing complete. Data is cached.")
        sys.exit(0)

    if args.confusion:
        print("\nLoading model for confusion matrix evaluation...")
        # Since we are not training, we can just load the model state we already set up?
        # But we need to make sure the model architecture matches.
        # If we came from 'resume' path (checkpoint detected), model is already loaded with correct config.
        # If we are here, we MUST have loaded from checkpoint because of the check above.

        print("\nEvaluating on validation set...")
        # Need trainer to run evaluation easily
        trainer_config = TrainerConfig(
            batch_size=args.batch_size,
            device="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        trainer = Trainer(model, train_data, val_data, trainer_config)

        trainer.print_confusion_matrices(save_dir=args.output)
        print("\nConfusion matrix evaluation complete.")
        import sys
        sys.exit(0)

    print(f"\nSplit: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # Supervised training with differential learning rates
    print("\nStarting supervised training...")
    
    # Save timings
    timings['total_startup'] = time.time() - script_start_time
    try:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, "timing.yml"), "w") as f:
            yaml.dump(timings, f)
        print(f"Startup timings saved to {os.path.join(args.output, 'timing.yml')}")
    except Exception as e:
        print(f"Warning: Failed to save timings: {e}")

    trainer_config = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        formality_loss_weight=args.formality_weight,
        gender_loss_weight=args.gender_weight,
        grammaticality_loss_weight=args.grammaticality_weight,
        device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        use_amp=args.fp16 if args.fp16 is not None else False,
        local_rank=local_rank,
        world_size=world_size,
        grad_accum_steps=args.grad_accum_steps
    )
    # Use smaller LR for encoder if pretrained
    encoder_lr_factor = args.encoder_lr_factor if args.pretrain_mlm else 1.0
    trainer = Trainer(
        model, train_data, val_data, trainer_config,
        encoder_lr_factor=encoder_lr_factor,
    )

    # Restore training state if resuming
    if args.resume and checkpoint is not None:
        trainer.restore_from_checkpoint(checkpoint, reset_optimizer=vocab_grew)

    history = trainer.train(
        checkpoint_dir=args.output,
        checkpoint_args=args,
        model_config=model_config,
    )

    # Print final metrics
    trainer.print_confusion_matrices()

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id, model_config.max_seq_len),
    )

    model.eval()
    device = torch.device(trainer_config.device)
    formality_correct = 0
    gender_correct = 0
    grammaticality_correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            field_inputs = {
                k: v.to(device) for k, v in batch.items()
                if k.startswith('input_ids_')
            }
            attention_mask = batch['attention_mask'].to(device)
            formality_labels = batch['formality_labels'].to(device)
            gender_labels = batch['gender_labels'].to(device)
            grammaticality_labels = batch['grammaticality_labels'].to(device)

            formality_logits, gender_logits, grammaticality_logits = model(field_inputs, attention_mask)
            formality_preds = formality_logits.argmax(dim=-1)
            gender_preds = gender_logits.argmax(dim=-1)
            grammaticality_preds = grammaticality_logits.argmax(dim=-1)

            formality_correct += (formality_preds == formality_labels).sum().item()
            gender_correct += (gender_preds == gender_labels).sum().item()
            grammaticality_correct += (grammaticality_preds == grammaticality_labels).sum().item()
            total += formality_labels.size(0)

    print(f"Test Accuracy (float32): formality={formality_correct/total:.4f}, gender={gender_correct/total:.4f}, gram={grammaticality_correct/total:.4f}")
    f32_accuracy = (formality_correct/total, gender_correct/total, grammaticality_correct/total)

    # Save model
    if is_main_process():
        print(f"\nSaving model to {args.output}...")
        if args.fp8:
            print("  (converting to float8 for smallest size)")
        elif args.fp16:
            print("  (converting to float16 for smaller size)")
        
        # Unwrap model if distributed
        model_to_save = model.module if hasattr(model, 'module') else model
        save_model(model_to_save, tokenizer, args.output, model_config, fp16=args.fp16, fp8=args.fp8)
        print("Done!")

    # Verify reduced precision model accuracy if applicable
    if args.fp16 or args.fp8:
        precision_name = "fp8" if args.fp8 else "fp16"
        print(f"\nVerifying loaded {precision_name} model accuracy...")
        loaded_model, _ = load_model(args.output, device=device)

        formality_correct = 0
        gender_correct = 0
        grammaticality_correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                field_inputs = {
                    k: v.to(device) for k, v in batch.items()
                    if k.startswith('input_ids_')
                }
                attention_mask = batch['attention_mask'].to(device)
                formality_labels = batch['formality_labels'].to(device)
                gender_labels = batch['gender_labels'].to(device)
                grammaticality_labels = batch['grammaticality_labels'].to(device)

                formality_logits, gender_logits, grammaticality_logits = loaded_model(field_inputs, attention_mask)
                formality_preds = formality_logits.argmax(dim=-1)
                gender_preds = gender_logits.argmax(dim=-1)
                grammaticality_preds = grammaticality_logits.argmax(dim=-1)

                formality_correct += (formality_preds == formality_labels).sum().item()
                gender_correct += (gender_preds == gender_labels).sum().item()
                grammaticality_correct += (grammaticality_preds == grammaticality_labels).sum().item()
                total += formality_labels.size(0)

        reduced_accuracy = (formality_correct/total, gender_correct/total, grammaticality_correct/total)
        print(f"Test Accuracy ({precision_name}):    formality={reduced_accuracy[0]:.4f}, gender={reduced_accuracy[1]:.4f}, gram={reduced_accuracy[2]:.4f}")

        # Show difference
        diff = tuple(reduced - f32 for reduced, f32 in zip(reduced_accuracy, f32_accuracy))
        print(f"Difference ({precision_name}-f32):   formality={diff[0]:+.4f}, gender={diff[1]:+.4f}, gram={diff[2]:+.4f}")
