"""Dataset and processing logic for style classification."""

import json
import multiprocessing as mp
import os
import random
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
from torch.utils.data import Dataset

from kotogram import locations
from kotogram.japanese_parser import JapaneseParser
from kotogram.model import (
    NUM_FORMALITY_PRAGMATIC_CLASSES,
    NUM_GENDER_PRAGMATIC_CLASSES,
    NUM_GRAMMATICALITY_CLASSES,
    NUM_REGISTER_CLASSES,
)
from kotogram.tokenizer import FEATURE_FIELDS, Tokenizer
from train.cache import get_kotogram_cache
from train.tsv import parse_tsv  # Re-exported for backward compatibility
from train.types import ProcessedSample, Sample
from train.worker import _encode_samples_batch, init_worker

# Cache version - bump this when cache format changes to invalidate old caches
CACHE_VERSION = 11

# KC Configuration
KC_HASH_BUCKETS = 16384
KC_NGRAM_ORDER = 3
KC_POS_BIASED_WINDOW = 5


# Types moved to types.py


@dataclass
class DatasetConfig:
    """Configuration for StyleDataset loading and processing."""

    parser: Optional[JapaneseParser] = None
    verbose: bool = True
    grammaticality_labels: Optional[List[int]] = None
    use_cache: bool = True
    cache_name: Optional[str] = "vocab.json"
    sample_ratio: float = 1.0


class StyleDataset(Dataset[Sample]):
    """PyTorch Dataset for style classification using feature-based tokenization."""

    def __init__(
        self,
        samples: List[Sample],
        tokenizer: Tokenizer,
    ):
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]

    @staticmethod
    def _load_vocab(cache_path: str, tokenizer: Tokenizer) -> None:
        """Load tokenizer vocabulary from cache."""
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Vocabulary not found at {cache_path}")

        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("version") != CACHE_VERSION:
            raise ValueError(f"Cache version mismatch. Expected {CACHE_VERSION}")

        if "field_vocabs" not in data:
            raise ValueError(
                f"Failed to load vocabulary from {cache_path}: 'field_vocabs'"
            )

        # pylint: disable=protected-access
        tokenizer.field_vocabs = data["field_vocabs"]
        tokenizer._frozen = bool(data.get("frozen", False))

    @classmethod
    def _process_parallel(
        cls,
        rows: List[Tuple[str, int]],
        verbose: bool = True,
    ) -> List[ProcessedSample]:
        # Note: batch_size and num_workers removed as unused
        cache = get_kotogram_cache()
        chunk_size = 10000
        chunks = [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]

        final_results: List[ProcessedSample] = []
        total_processed = 0

        if verbose:
            print(f"Loading {len(rows)} samples from cache...")

        for chunk in chunks:
            chunk_sentences = [r[0] for r in chunk]
            cached_data_map = cache.get_batch(chunk_sentences)

            for sentence, gram_label in chunk:
                cached_tuple = cached_data_map.get(sentence)
                if cached_tuple is not None:
                    final_results.append(
                        cls._process_cache_hit(sentence, gram_label, cached_tuple)
                    )
                else:
                    raise ValueError(f"Sentence not found in cache: {sentence[:30]}...")

            total_processed += len(chunk)
            if verbose and total_processed % 50000 == 0:
                print(
                    f"\r  Loaded {total_processed}/{len(rows)}...", end="", flush=True
                )

        if verbose:
            print()
        return final_results

    @staticmethod
    def _process_cache_hit(
        sentence: str, gram_label: int, cached_tuple: Tuple
    ) -> ProcessedSample:
        """Create ProcessedSample from cache tuple."""
        k, f_lbl, g_val, g_prag, r_lbls, _, f_ids = cached_tuple
        return ProcessedSample(
            sentence=sentence,
            kotogram=k,
            formality_id=cast(int, f_lbl) if f_lbl is not None else 2,
            gender_value=cast(float, g_val) if g_val is not None else 0.0,
            gender_pragmatic=cast(int, g_prag) if g_prag is not None else 0,
            register_ids=cast(List[int], r_lbls) if r_lbls is not None else [0],
            gram_label=gram_label,
            success=1,
            feature_ids=f_ids,
        )

    @classmethod
    def from_tsv(
        cls,
        tsv_path: str,
        tokenizer: Tokenizer,
        config: Optional[DatasetConfig] = None,
    ) -> "StyleDataset":
        if config is None:
            config = DatasetConfig()
        return cls.from_multiple_tsv(
            tsv_paths=[tsv_path],
            tokenizer=tokenizer,
            config=config,
        )

    @staticmethod
    def _load_raw_rows(
        tsv_paths: List[str], gram_labels: List[int], config: DatasetConfig
    ) -> List[Tuple[str, int]]:
        """Load and sample raw rows from TSVs."""
        all_rows: List[Tuple[str, int]] = []
        for tsv_path, gram_label in zip(tsv_paths, gram_labels):
            if config.verbose:
                print(f"Reading {tsv_path}...")
            file_rows: List[Tuple[str, int]] = []
            with open(tsv_path, "r", encoding="utf-8") as f:
                for line in f:
                    sentence = parse_tsv(line)
                    file_rows.append((sentence, gram_label))
            all_rows.extend(file_rows)

        if config.sample_ratio < 1.0:
            random.seed(42)
            n_samples = max(1, int(len(all_rows) * config.sample_ratio))
            all_rows = random.sample(all_rows, n_samples)

        return all_rows

    @classmethod
    def from_multiple_tsv(
        cls,
        tsv_paths: List[str],
        tokenizer: Tokenizer,
        config: Optional[DatasetConfig] = None,
    ) -> "StyleDataset":
        if config is None:
            config = DatasetConfig()

        cache_dir = locations.get_style_dataset_cache_dir()

        gram_labels = config.grammaticality_labels
        if gram_labels is None:
            gram_labels = [1] * len(tsv_paths)

        vocab_path = ""
        vocab_path = ""
        if config.use_cache and config.cache_name:
            vocab_path = os.path.join(cache_dir, config.cache_name)
            metadata_path = os.path.join(cache_dir, "label_metadata.json")
            if os.path.exists(vocab_path) and os.path.exists(metadata_path):
                # Simple presence check here for now, full validation could be added back
                cls._load_vocab(vocab_path, tokenizer)
                if config.verbose:
                    print(f"  Loaded vocabulary from cache: {vocab_path}")

        if len(tokenizer.field_vocabs["surface"]) <= 4:
            raise ValueError(
                "Vocabulary not loaded. Ensure label.py finished successfully."
            )

        preprocessing_start = time.time()
        phase1_start = time.time()
        all_rows = cls._load_raw_rows(tsv_paths, gram_labels, config)

        phase1_duration = time.time() - phase1_start
        phase2_start = time.time()
        processed_results = cls._process_parallel(all_rows, verbose=config.verbose)
        phase2_duration = time.time() - phase2_start

        return cls.from_processed_samples(
            processed_results=processed_results,
            tokenizer=tokenizer,
            config=config,
            timing_info={
                "preprocessing_start": preprocessing_start,
                "phase1_duration": phase1_duration,
                "phase2_duration": phase2_duration,
            },
        )

    @staticmethod
    def _compute_kc_targets(feature_ids: Dict[str, List[int]]) -> Dict[str, Any]:
        """Compute KC targets from feature IDs."""
        targets: Dict[str, Any] = {}

        # 1. Token bags (Multi-hot)
        # Target = unique token IDs appearing in the sentence
        for field in ["lemma", "pos", "conjugated_form"]:
            if field in feature_ids:
                targets[f"bag_{field}"] = list(set(feature_ids[field]))

        # 2. Position-biased token bags
        # Target = unique token IDs appearing in the last N tokens
        for field in ["surface", "lemma", "pos", "conjugated_form"]:
            if field in feature_ids:
                ids = feature_ids[field]
                tail_ids = ids[-KC_POS_BIASED_WINDOW:] if len(ids) > 0 else []
                targets[f"tail_{field}"] = list(set(tail_ids))

        # 3. N-gram hash targets
        # Target = hashed IDs for bigrams/trigrams
        for field in ["pos", "conjugated_form"]:
            if field in feature_ids:
                ids = feature_ids[field]
                hashes = set()
                # Unigrams, Bigrams, Trigrams
                # Actually user asked for Bigrams/Trigrams.
                # Let's include unigrams too? User said: "bigrams/trigrams of pos"
                # "Token bags" (Priority 1A) covers unigrams basically.
                # So let's stick to n=2..Order.
                for n_val in range(2, KC_NGRAM_ORDER + 1):
                    if len(ids) >= n_val:
                        for i in range(len(ids) - n_val + 1):
                            ngram = tuple(ids[i : i + n_val])
                            # Simple hash: polynomial rolling hash or python hash
                            # Python hash is randomized per process, strictly we might want stable
                            # but "Stable KC IDs across runs" is a non-goal.
                            # Start with python hash for simplicity and speed.
                            h = hash(ngram) % KC_HASH_BUCKETS
                            hashes.add(h)
                targets[f"ngram_{field}"] = list(hashes)

        # 3b. (pos, conjugated_form) pairs
        if "pos" in feature_ids and "conjugated_form" in feature_ids:
            p_ids = feature_ids["pos"]
            c_ids = feature_ids["conjugated_form"]
            if len(p_ids) == len(c_ids):
                pair_hashes = set()
                for i, p_id in enumerate(p_ids):
                    pair = (p_id, c_ids[i])
                    h = hash(pair) % KC_HASH_BUCKETS
                    pair_hashes.add(h)
                targets["pair_pos_conj"] = list(pair_hashes)

        return targets

    @classmethod
    def from_processed_samples(
        cls,
        processed_results: List[ProcessedSample],
        tokenizer: Tokenizer,
        config: Optional[DatasetConfig] = None,
        timing_info: Optional[Dict[str, float]] = None,
    ) -> "StyleDataset":
        if config is None:
            config = DatasetConfig()
        if timing_info is None:
            timing_info = {}

        # vocab_path = "" # Not strictly needed if handled in caller, but keeping for logic flow if needed

        precomputed_results = [
            p for p in processed_results if p.success and p.feature_ids is not None
        ]
        missing_results = [
            p for p in processed_results if p.success and p.feature_ids is None
        ]

        samples: List[Sample] = []

        # 1. Map precomputed
        for p in precomputed_results:
            samples.append(cls._map_processed_to_sample(p))

        # 2. Encode missing
        if missing_results:
            newly_encoded = cls._encode_infos_parallel(
                missing_results, tokenizer, config
            )

            # Cache updates
            cache = get_kotogram_cache()
            update_items = [
                p.to_cache_tuple(s.feature_ids)
                for p, s in zip(missing_results, newly_encoded)
            ]
            cache.put_batch(cast(List[Any], update_items))

            # KC Targets
            for s in newly_encoded:
                s.kc_targets = cls._compute_kc_targets(s.feature_ids)

            samples.extend(newly_encoded)

        # pylint: disable=protected-access
        tokenizer.freeze()
        return cls(samples, tokenizer)

    @staticmethod
    def _map_processed_to_sample(p: ProcessedSample) -> Sample:
        """Convert ProcessedSample to Sample."""
        f_id = p.formality_id
        if f_id == 5:
            f_val, f_prag = 0.0, 0
        else:
            f_val = {0: 1.0, 1: 0.5, 2: 0.0, 3: -0.5, 4: -1.0}.get(f_id, 0.0)
            f_prag = 1

        return Sample(
            feature_ids=cast(Dict[str, List[int]], p.feature_ids),
            formality_value=f_val,
            formality_pragmatic=f_prag,
            gender_value=p.gender_value,
            gender_pragmatic=p.gender_pragmatic,
            register_labels=p.register_ids,
            grammaticality_label=p.gram_label,
            original_sentence=p.sentence,
            kotogram=p.kotogram,
            kc_targets=StyleDataset._compute_kc_targets(
                cast(Dict[str, List[int]], p.feature_ids)
            ),
        )

    @staticmethod
    def _encode_infos_parallel(
        missing_results: List[ProcessedSample],
        tokenizer: Tokenizer,
        config: DatasetConfig,
    ) -> List[Sample]:
        """Encode missing samples in parallel."""
        ctx = mp.get_context("spawn")
        num_workers = max(1, mp.cpu_count() - 1)

        tokenizer_state = {"field_vocabs": tokenizer.field_vocabs}
        small_batch_threshold = 1000

        batches = [
            missing_results[i : i + 5000] for i in range(0, len(missing_results), 5000)
        ]

        results: List[Sample] = []

        if len(missing_results) < small_batch_threshold:
            if config.verbose:
                print(f"Encoding {len(missing_results)} samples sequentially...")

            init_worker(tokenizer_state)
            for batch in batches:
                results.extend(_encode_samples_batch(batch))
        else:
            pool = ctx.Pool(
                num_workers, initializer=init_worker, initargs=(tokenizer_state,)
            )
            try:
                for batch_encoded in pool.imap(_encode_samples_batch, batches):
                    results.extend(batch_encoded)
                pool.close()
                pool.join()
            finally:
                pool.terminate()
                pool.join()

        return results

    def split(
        self,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> Tuple["StyleDataset", "StyleDataset"]:
        """Split dataset into train and validation sets."""
        random.seed(seed)
        indices = list(range(len(self.samples)))
        random.shuffle(indices)

        n_train = int(len(self.samples) * train_ratio)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        return (
            StyleDataset([self.samples[i] for i in train_indices], self.tokenizer),
            StyleDataset([self.samples[i] for i in val_indices], self.tokenizer),
        )

    def get_formality_class_weights(self) -> torch.Tensor:
        """Calculate inverse frequency weights for formality classes."""
        counts = Counter(s.formality_pragmatic for s in self.samples)
        total = sum(counts.values())
        weights = torch.zeros(NUM_FORMALITY_PRAGMATIC_CLASSES)
        for i in range(NUM_FORMALITY_PRAGMATIC_CLASSES):
            weights[i] = total / (counts[i] + 1e-5)
        return weights / weights.sum() * NUM_FORMALITY_PRAGMATIC_CLASSES

    def get_gender_class_weights(self) -> torch.Tensor:
        """Calculate inverse frequency weights for gender classes."""
        counts = Counter(s.gender_pragmatic for s in self.samples)
        total = sum(counts.values())
        weights = torch.zeros(NUM_GENDER_PRAGMATIC_CLASSES)
        for i in range(NUM_GENDER_PRAGMATIC_CLASSES):
            weights[i] = total / (counts[i] + 1e-5)
        return weights / weights.sum() * NUM_GENDER_PRAGMATIC_CLASSES

    def get_grammaticality_class_weights(self) -> torch.Tensor:
        """Calculate inverse frequency weights for grammaticality classes."""
        counts = Counter(s.grammaticality_label for s in self.samples)
        total = sum(counts.values())
        weights = torch.zeros(NUM_GRAMMATICALITY_CLASSES)
        for i in range(NUM_GRAMMATICALITY_CLASSES):
            weights[i] = total / (counts[i] + 1e-5)
        return weights / weights.sum() * NUM_GRAMMATICALITY_CLASSES


def collate_fn(
    batch: List[Sample],
    pad_id: int = 0,
    max_seq_len: Optional[int] = None,
    vocab_sizes: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Collate samples into padded batches."""
    batch_size = len(batch)
    if max_seq_len is None:
        max_seq_len = max(s.seq_len for s in batch)

    attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.float)
    for i, sample in enumerate(batch):
        seq_len = min(sample.seq_len, max_seq_len)
        attention_mask[i, :seq_len] = 1.0

    result = {
        "attention_mask": attention_mask,
        "formality_value": torch.tensor(
            [s.formality_value for s in batch], dtype=torch.float
        ),
        "formality_pragmatic": torch.tensor(
            [s.formality_pragmatic for s in batch], dtype=torch.long
        ),
        "gender_value": torch.tensor(
            [s.gender_value for s in batch], dtype=torch.float
        ),
        "gender_pragmatic": torch.tensor(
            [s.gender_pragmatic for s in batch], dtype=torch.long
        ),
        "grammaticality_labels": torch.tensor(
            [s.grammaticality_label for s in batch], dtype=torch.long
        ),
        "original_sentence": [s.original_sentence for s in batch],
        "kotogram": [s.kotogram for s in batch],
    }

    register_targets = torch.zeros(
        (batch_size, NUM_REGISTER_CLASSES), dtype=torch.float
    )
    for i, sample in enumerate(batch):
        for reg_id in sample.register_labels:
            if reg_id < NUM_REGISTER_CLASSES:
                register_targets[i, reg_id] = 1.0
    result["register_labels"] = register_targets

    for field_name in FEATURE_FIELDS:
        tensor = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.long)
        for i, sample in enumerate(batch):
            ids = sample.feature_ids[field_name]
            seq_len = min(len(ids), max_seq_len)
            tensor[i, :seq_len] = torch.tensor(ids[:seq_len], dtype=torch.long)
        result[f"input_ids_{field_name}"] = tensor

    if batch[0].kc_targets:
        result.update(_collate_kc_targets(batch, batch_size, vocab_sizes))

    return result


def _collate_kc_targets(
    batch: List[Sample],
    batch_size: int,
    vocab_sizes: Optional[Dict[str, int]] = None,
) -> Dict[str, torch.Tensor]:
    """Helper to collate KC targets."""
    result: Dict[str, torch.Tensor] = {}

    # Determine all keys present in the batch
    all_keys = set().union(*(s.kc_targets.keys() for s in batch))

    for key in all_keys:
        # Determine vocab size for this target
        v_size = 0
        if key.startswith("ngram_") or key.startswith("pair_"):
            v_size = KC_HASH_BUCKETS
        elif key.startswith("bag_") or key.startswith("tail_"):
            # Extract field name
            field_name = key.split("_", 1)[1]
            if vocab_sizes and field_name in vocab_sizes:
                v_size = vocab_sizes[field_name]
            else:
                # Fallback if vocab_size not known
                pass

        if v_size > 0:
            target_tensor = torch.zeros((batch_size, v_size), dtype=torch.float)
            for i, sample in enumerate(batch):
                indices = sample.kc_targets.get(key, [])
                # Filter valid indices
                valid_indices = [idx for idx in indices if idx < v_size]
                if valid_indices:
                    target_tensor[i, valid_indices] = 1.0
            result[f"kc_targets_{key}"] = target_tensor

    return result
