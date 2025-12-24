"""Dataset and processing logic for style classification."""

import json
import multiprocessing as mp
import os
import random
import time
from collections import Counter
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


# Types moved to types.py


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

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("version") != CACHE_VERSION:
                raise ValueError(f"Cache version mismatch. Expected {CACHE_VERSION}")

            tokenizer.field_vocabs = data["field_vocabs"]
            tokenizer._frozen = bool(data.get("frozen", False))
        except Exception as e:
            raise ValueError(f"Failed to load vocabulary from {cache_path}: {e}")

    @classmethod
    def _process_parallel(
        cls,
        rows: List[Tuple[str, int]],
        batch_size: int = 1000,
        num_workers: Optional[int] = None,
        verbose: bool = True,
    ) -> List[ProcessedSample]:
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
                    k, f_lbl, g_val, g_prag, r_lbls, _, f_ids = cached_tuple
                    processed_sample = ProcessedSample(
                        sentence=sentence,
                        kotogram=k,
                        formality_id=cast(int, f_lbl) if f_lbl is not None else 2,
                        gender_value=cast(float, g_val) if g_val is not None else 0.0,
                        gender_pragmatic=cast(int, g_prag) if g_prag is not None else 0,
                        register_ids=cast(List[int], r_lbls)
                        if r_lbls is not None
                        else [0],
                        gram_label=gram_label,
                        success=1,
                        feature_ids=f_ids,
                    )
                    final_results.append(processed_sample)
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

    @classmethod
    def from_tsv(
        cls,
        tsv_path: str,
        tokenizer: Tokenizer,
        parser: Optional[JapaneseParser] = None,
        verbose: bool = True,
        labeled: bool = True,
        use_cache: bool = True,
        sample_ratio: float = 1.0,
    ) -> "StyleDataset":
        return cls.from_multiple_tsv(
            tsv_paths=[tsv_path],
            tokenizer=tokenizer,
            parser=parser,
            verbose=verbose,
            labeled=labeled,
            use_cache=use_cache,
            sample_ratio=sample_ratio,
        )

    @classmethod
    def from_multiple_tsv(
        cls,
        tsv_paths: List[str],
        tokenizer: Tokenizer,
        parser: Optional[JapaneseParser] = None,
        verbose: bool = True,
        labeled: bool = True,
        grammaticality_labels: Optional[List[int]] = None,
        use_cache: bool = True,
        cache_name: Optional[str] = "vocab.json",
        sample_ratio: float = 1.0,
    ) -> "StyleDataset":
        cache_dir = locations.get_style_dataset_cache_dir()

        if grammaticality_labels is None:
            grammaticality_labels = [1] * len(tsv_paths)

        vocab_path = ""
        if use_cache and cache_name:
            vocab_path = os.path.join(cache_dir, cache_name)
            metadata_path = os.path.join(cache_dir, "label_metadata.json")
            if os.path.exists(vocab_path) and os.path.exists(metadata_path):
                # Simple presence check here for now, full validation could be added back
                cls._load_vocab(vocab_path, tokenizer)
                if verbose:
                    print(f"  Loaded vocabulary from cache: {vocab_path}")

        if len(tokenizer.field_vocabs["surface"]) <= 4:
            raise ValueError(
                "Vocabulary not loaded. Ensure label.py finished successfully."
            )

        preprocessing_start = time.time()
        all_rows: List[Tuple[str, int]] = []
        phase1_start = time.time()

        for tsv_path, gram_label in zip(tsv_paths, grammaticality_labels):
            if verbose:
                print(f"Reading {tsv_path}...")
            file_rows: List[Tuple[str, int]] = []
            with open(tsv_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        sentence = parse_tsv(line)
                        file_rows.append((sentence, gram_label))
                    except ValueError:
                        continue
            all_rows.extend(file_rows)

        if sample_ratio < 1.0:
            random.seed(42)
            n_samples = max(1, int(len(all_rows) * sample_ratio))
            all_rows = random.sample(all_rows, n_samples)

        phase1_duration = time.time() - phase1_start
        phase2_start = time.time()
        processed_results = cls._process_parallel(all_rows, verbose=verbose)
        phase2_duration = time.time() - phase2_start

        return cls.from_processed_samples(
            processed_results=processed_results,
            tokenizer=tokenizer,
            verbose=verbose,
            use_cache=use_cache,
            cache_name=cache_name,
            sample_ratio=sample_ratio,
            preprocessing_start=preprocessing_start,
            phase1_duration=phase1_duration,
            phase2_duration=phase2_duration,
            vocab_path=vocab_path,
        )

    @classmethod
    def from_processed_samples(
        cls,
        processed_results: List[ProcessedSample],
        tokenizer: Tokenizer,
        verbose: bool = True,
        use_cache: bool = True,
        cache_name: Optional[str] = "vocab.json",
        sample_ratio: float = 1.0,
        preprocessing_start: Optional[float] = None,
        phase1_duration: float = 0.0,
        phase2_duration: float = 0.0,
        vocab_path: Optional[str] = None,
    ) -> "StyleDataset":
        if preprocessing_start is None:
            preprocessing_start = time.time()

        ctx = mp.get_context("spawn")
        num_workers = max(1, mp.cpu_count() - 1)

        tokenizer.freeze()

        precomputed_results = [
            p for p in processed_results if p.success and p.feature_ids is not None
        ]
        missing_results = [
            p for p in processed_results if p.success and p.feature_ids is None
        ]

        samples: List[Sample] = []

        if precomputed_results:
            for p in precomputed_results:
                f_id = p.formality_id
                if f_id == 5:
                    f_val, f_prag = 0.0, 0
                else:
                    f_val = {0: 1.0, 1: 0.5, 2: 0.0, 3: -0.5, 4: -1.0}.get(f_id, 0.0)
                    f_prag = 1

                samples.append(
                    Sample(
                        feature_ids=cast(Dict[str, List[int]], p.feature_ids),
                        formality_value=f_val,
                        formality_pragmatic=f_prag,
                        gender_value=p.gender_value,
                        gender_pragmatic=p.gender_pragmatic,
                        register_labels=p.register_ids,
                        grammaticality_label=p.gram_label,
                        original_sentence=p.sentence,
                        kotogram=p.kotogram,
                    )
                )

        if missing_results:
            tokenizer_state = {"field_vocabs": tokenizer.field_vocabs}

            # Optimization: Run sequentially for small batches to avoid spawn overhead
            SMALL_BATCH_THRESHOLD = 1000

            batches = [
                missing_results[i : i + 5000]
                for i in range(0, len(missing_results), 5000)
            ]

            newly_encoded_samples: List[Sample] = []

            if len(missing_results) < SMALL_BATCH_THRESHOLD:
                if verbose:
                    print(f"Encoding {len(missing_results)} samples sequentially...")

                # Initialize worker state in main process
                # Note: This modifies train.worker._tokenizer global in the main process
                init_worker(tokenizer_state)

                for batch_samples in batches:
                    newly_encoded_samples.extend(_encode_samples_batch(batch_samples))
            else:
                pool = ctx.Pool(
                    num_workers, initializer=init_worker, initargs=(tokenizer_state,)
                )
                try:
                    for batch_encoded in pool.imap(_encode_samples_batch, batches):
                        newly_encoded_samples.extend(batch_encoded)
                    pool.close()
                    pool.join()
                except Exception:
                    pool.terminate()
                    pool.join()
                    raise

            if newly_encoded_samples:
                cache = get_kotogram_cache()
                update_items = []
                for p, s in zip(missing_results, newly_encoded_samples):
                    # Re-map formality_id to formality_id for cache (bit annoying)
                    update_items.append(
                        (
                            p.sentence,
                            p.kotogram,
                            p.formality_id,
                            p.gender_value,
                            p.gender_pragmatic,
                            p.register_ids,
                            p.gram_label,
                            s.feature_ids,
                        )
                    )
                cache.put_batch(cast(List[Any], update_items))

            samples.extend(newly_encoded_samples)

        tokenizer.freeze()
        return cls(samples, tokenizer)

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

    return result
