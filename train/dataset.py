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
        samples: Optional[List[Sample]],
        tokenizer: Tokenizer,
        tensor_data: Optional[Dict[str, Any]] = None,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.tensor_data = tensor_data

        if self.tensor_data is not None:
            # Check length from offsets
            if "offsets" in self.tensor_data:
                self._len = len(self.tensor_data["offsets"]) - 1
            else:
                self._len = 0

            # Ensure safe access for Mypy
            self.tensor_data = cast(Dict[str, Any], self.tensor_data)
        elif self.samples is not None:
            self._len = len(self.samples)
        else:
            self._len = 0

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Sample:
        if self.tensor_data is not None:
            return self._get_item_from_tensors(idx)
        # Fallback to list-based access with safety check
        if self.samples is None:
            raise ValueError("StyleDataset not initialized with any data.")
        return self.samples[idx]

    def _get_item_from_tensors(self, idx: int) -> Sample:
        """Construct Sample upon retrieval."""
        # Mypy safety
        if self.tensor_data is None:
            raise ValueError("Tensor data is None")

        offsets = self.tensor_data["offsets"]
        start = offsets[idx].item()
        end = offsets[idx + 1].item()

        feature_ids: Dict[str, List[int]] = {}
        # Iterate over tokenizer fields.
        # If tensor_data doesn't fail on missing key, check existence.
        # FEATURE_FIELDS imported or use keys from tensor excluding metadata?
        # Let's iterate keys and check if they match fields.
        # But efficiently: we iterate known fields.
        for field in FEATURE_FIELDS:
            if field in self.tensor_data:
                field_tensor = self.tensor_data[field]
                feature_ids[field] = field_tensor[start:end].tolist()

        # Labels
        labels = self.tensor_data["labels"]

        # Register labels: List[int]
        # Reconstruct from reg_ids + reg_offsets
        reg_ids_tensor = labels["reg_ids"]
        reg_offsets = labels["reg_offsets"]
        r_start = reg_offsets[idx].item()
        r_end = reg_offsets[idx + 1].item()
        reg_list = reg_ids_tensor[r_start:r_end].tolist()
        if not reg_list:
            reg_list = [0]  # Default

        # Mapping for formality value (replicate _map_processed_to_sample logic or use stored value)
        # We stored f_val directly.

        return Sample(
            feature_ids=feature_ids,
            formality_value=labels["f_val"][idx].item(),
            formality_pragmatic=labels["f_prag"][idx].item(),
            gender_value=labels["g_val"][idx].item(),
            gender_pragmatic=labels["g_prag"][idx].item(),
            grammaticality_label=labels["gram"][idx].item(),
            register_labels=reg_list,
            original_sentence="",  # Missing in binary mode
            kotogram="",  # Missing
            # KC targets computed on demand or stored?
            # We don't store KC targets in binary mode yet to save space/time.
            # If they are needed, we can compute them from feature_ids.
            kc_targets=(
                self._compute_kc_targets(feature_ids)
                if DatasetConfig().parser is None
                else {}
            ),
        )

    @staticmethod
    def _subset_register_labels(
        labels: Dict[str, Any], indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper to slice register ragged tensors."""
        old_offsets = labels["reg_offsets"]
        starts = old_offsets[indices]
        ends = old_offsets[indices + 1]

        # New Offsets
        new_offsets = torch.zeros(len(indices) + 1, dtype=torch.int32)
        torch.cumsum(ends - starts, dim=0, out=new_offsets[1:])

        # New IDs
        slices = [labels["reg_ids"][s:e] for s, e in zip(starts, ends)]
        new_ids = torch.cat(slices) if slices else torch.tensor([], dtype=torch.long)
        return new_offsets, new_ids

    def _subset_from_tensors(self, indices: torch.Tensor) -> "StyleDataset":
        """Create a new StyleDataset backed by a subset of tensors."""
        if self.tensor_data is None:
            raise ValueError("No tensor data")

        labels = self.tensor_data["labels"]
        indices = indices.long()  # Ensure long for indexing

        # 1. New Offsets
        offsets = self.tensor_data["offsets"]
        new_offsets = torch.zeros(len(indices) + 1, dtype=torch.int32)
        # Combine starts/ends access
        torch.cumsum(
            offsets[indices + 1] - offsets[indices], dim=0, out=new_offsets[1:]
        )

        new_data: Dict[str, Any] = {
            "offsets": new_offsets,
            "labels": {},
            "version": self.tensor_data.get("version", 2),
        }

        # 2. Slice Labels
        for k, v in labels.items():
            if k == "reg_offsets":
                # Handle register offsets
                (
                    new_data["labels"]["reg_offsets"],
                    new_data["labels"]["reg_ids"],
                ) = self._subset_register_labels(labels, indices)
            elif k != "reg_ids":
                new_data["labels"][k] = v[indices]

        # 3. Slice Features
        starts = offsets[indices]
        ends = offsets[indices + 1]
        for field in FEATURE_FIELDS:
            if field in self.tensor_data:
                field_slices = [
                    self.tensor_data[field][s:e] for s, e in zip(starts, ends)
                ]
                new_data[field] = (
                    torch.cat(field_slices)
                    if field_slices
                    else torch.tensor([], dtype=torch.int32)
                )

        return StyleDataset(None, self.tokenizer, tensor_data=new_data)

    def filter_by_grammaticality(self, valid_label: int = 1) -> "StyleDataset":
        """Return a new dataset containing only samples with the given grammaticality label."""
        if self.tensor_data is not None:
            mask = self.tensor_data["labels"]["gram"] == valid_label
            return self._subset_from_tensors(torch.nonzero(mask).squeeze(-1))

        if self.samples is not None:
            return StyleDataset(
                [s for s in self.samples if s.grammaticality_label == valid_label],
                self.tokenizer,
            )

        return StyleDataset([], self.tokenizer)

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
    def _try_load_from_binary_cache(
        cls, cache_dir: str, config: DatasetConfig
    ) -> Optional[List[Tuple[str, int]]]:
        """Attempt to load dataset from binary cache."""
        index_path = os.path.join(cache_dir, "dataset_index.pt")
        if config.use_cache and os.path.exists(index_path):
            if config.verbose:
                print(f"  Loading dataset index from binary cache: {index_path}")
            all_rows = cast(List[Tuple[str, int]], torch.load(index_path))
            # Apply downsampling if needed
            if config.sample_ratio < 1.0:
                random.seed(42)
                n_samples = max(1, int(len(all_rows) * config.sample_ratio))
                all_rows = random.sample(all_rows, n_samples)
            return all_rows
        return None

    @classmethod
    def _try_load_tensor_cache(
        cls, cache_dir: str, config: DatasetConfig
    ) -> Optional[Dict[str, Any]]:
        """Attempt to load dataset from binary tensor cache."""
        tensor_path = os.path.join(cache_dir, "dataset_tensors.pt")
        if not (config.use_cache and os.path.exists(tensor_path)):
            return None

        if config.verbose:
            print(f"  Loading binary tensors from: {tensor_path}")
        tensor_data = torch.load(tensor_path)

        if config.sample_ratio < 1.0:
            # Simple random sampling logic for tensors
            random.seed(42)
            num_samples = len(tensor_data["offsets"]) - 1
            keep_count = max(1, int(num_samples * config.sample_ratio))

            if config.verbose:
                print(
                    f"  Subsampling binary cache: using first {config.sample_ratio:.1%} of data (shuffled)."
                )

            # Slice offsets: we need keep_count + 1 offsets
            new_offsets = tensor_data["offsets"][: keep_count + 1]
            end_token_idx = new_offsets[-1].item()
            tensor_data["offsets"] = new_offsets

            # Slice features
            for field in FEATURE_FIELDS:
                if field in tensor_data:
                    tensor_data[field] = tensor_data[field][:end_token_idx]

            # Slice labels
            # Register offsets need special slicing
            reg_offsets = tensor_data["labels"]["reg_offsets"][: keep_count + 1]
            end_reg_idx = reg_offsets[-1].item()
            tensor_data["labels"]["reg_offsets"] = reg_offsets
            tensor_data["labels"]["reg_ids"] = tensor_data["labels"]["reg_ids"][
                :end_reg_idx
            ]

            for k, v in tensor_data["labels"].items():
                if k not in ("reg_ids", "reg_offsets"):
                    tensor_data["labels"][k] = v[:keep_count]

        return cast(Dict[str, Any], tensor_data)

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
        if config.use_cache and config.cache_name:
            vocab_path = os.path.join(cache_dir, config.cache_name)
            if os.path.exists(vocab_path) and os.path.exists(
                os.path.join(cache_dir, "label_metadata.json")
            ):
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

        # Priority 1: Binary Tensor Cache (dataset_tensors.pt) - RAM Optimal
        tensor_data = cls._try_load_tensor_cache(cache_dir, config)
        if tensor_data is not None:
            return cls(samples=None, tokenizer=tokenizer, tensor_data=tensor_data)

        # Priority 2: Old Binary Index (dataset_index.pt) - Legacy
        all_rows = cls._try_load_from_binary_cache(cache_dir, config)
        if all_rows is None:
            # Priority 3: Raw TSV - Legacy Slow Path
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
        torch.manual_seed(seed)
        total_len = len(self)
        indices = torch.randperm(total_len)

        n_train = int(total_len * train_ratio)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        if self.tensor_data is not None:
            return (
                self._subset_from_tensors(train_indices),
                self._subset_from_tensors(val_indices),
            )

        if self.samples is not None:
            # Convert indices to list for list indexing
            train_idx_list = train_indices.tolist()
            val_idx_list = val_indices.tolist()
            return (
                StyleDataset([self.samples[i] for i in train_idx_list], self.tokenizer),
                StyleDataset([self.samples[i] for i in val_idx_list], self.tokenizer),
            )

        return (StyleDataset([], self.tokenizer), StyleDataset([], self.tokenizer))

    def get_formality_class_weights(self) -> torch.Tensor:
        """Calculate inverse frequency weights for formality classes."""
        if self.tensor_data is not None:
            # Revert to float if needed for validation logging etc
            labels = self.tensor_data["labels"]["f_prag"]
            t_counts = torch.bincount(
                labels, minlength=NUM_FORMALITY_PRAGMATIC_CLASSES
            ).float()
            total = t_counts.sum()
            weights = total / (t_counts + 1e-5)
            return weights / weights.sum() * NUM_FORMALITY_PRAGMATIC_CLASSES

        if self.samples is None:
            return torch.ones(NUM_FORMALITY_PRAGMATIC_CLASSES)

        counts = Counter(s.formality_pragmatic for s in self.samples)
        total_val = sum(counts.values())
        l_weights = torch.zeros(NUM_FORMALITY_PRAGMATIC_CLASSES)
        for i in range(NUM_FORMALITY_PRAGMATIC_CLASSES):
            l_weights[i] = total_val / (counts[i] + 1e-5)
        return l_weights / l_weights.sum() * NUM_FORMALITY_PRAGMATIC_CLASSES

    def get_gender_class_weights(self) -> torch.Tensor:
        """Calculate inverse frequency weights for gender classes."""
        if self.tensor_data is not None:
            # Use tensor operations
            g_prags = self.tensor_data["labels"]["g_prag"]
            # Bin count logic
            t_counts = torch.bincount(
                g_prags, minlength=NUM_GENDER_PRAGMATIC_CLASSES
            ).float()
            total = t_counts.sum()
            weights = total / (t_counts + 1e-5)
            return weights / weights.sum() * NUM_GENDER_PRAGMATIC_CLASSES

        if self.samples is None:
            return torch.ones(NUM_GENDER_PRAGMATIC_CLASSES)

        counts = Counter(s.gender_pragmatic for s in self.samples)
        total_val = sum(counts.values())
        l_weights = torch.zeros(NUM_GENDER_PRAGMATIC_CLASSES)
        for i in range(NUM_GENDER_PRAGMATIC_CLASSES):
            l_weights[i] = total_val / (counts[i] + 1e-5)
        return l_weights / l_weights.sum() * NUM_GENDER_PRAGMATIC_CLASSES

    def get_grammaticality_class_weights(self) -> torch.Tensor:
        """Calculate inverse frequency weights for grammaticality classes."""
        if self.tensor_data is not None:
            probs = self.tensor_data["labels"]["gram"]
            t_counts = torch.bincount(
                probs, minlength=NUM_GRAMMATICALITY_CLASSES
            ).float()
            total = t_counts.sum()
            weights = total / (t_counts + 1e-5)
            return weights / weights.sum() * NUM_GRAMMATICALITY_CLASSES

        if self.samples is None:
            return torch.ones(NUM_GRAMMATICALITY_CLASSES)

        counts = Counter(s.grammaticality_label for s in self.samples)
        total_val = sum(counts.values())
        l_weights = torch.zeros(NUM_GRAMMATICALITY_CLASSES)
        for i in range(NUM_GRAMMATICALITY_CLASSES):
            l_weights[i] = total_val / (counts[i] + 1e-5)
        return l_weights / l_weights.sum() * NUM_GRAMMATICALITY_CLASSES


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


# pylint: disable=too-many-locals
def create_mlm_batch(
    batch: Dict[str, torch.Tensor],
    mask_prob: float = 0.15,
    mask_token_id: int = 3,
    vocab_sizes: Optional[Dict[str, int]] = None,
    special_token_ids: Optional[List[int]] = None,
) -> Dict[str, torch.Tensor]:
    """Create masked language modeling batch for feature-based tokens."""
    special_token_ids = special_token_ids or [0, 1, 2, 3]
    vocab_sizes = vocab_sizes or {}
    hidden_fields = ["surface", "lemma"]
    primary_field = "pos"
    primary_ids = batch[f"input_ids_{primary_field}"].clone()

    maskable = batch["attention_mask"].bool()
    for special_id in special_token_ids:
        maskable &= primary_ids != special_id

    probs = torch.rand_like(primary_ids.float())
    mask = maskable & (probs < mask_prob)
    mask_token_positions = mask & (probs < mask_prob * 0.8)
    random_token_positions = (
        mask & (probs >= mask_prob * 0.8) & (probs < mask_prob * 0.9)
    )

    result = {"attention_mask": batch["attention_mask"]}
    for field in FEATURE_FIELDS:
        field_ids = batch[f"input_ids_{field}"].clone()
        mlm_labels = torch.full_like(field_ids, -100)
        if field in hidden_fields:
            active_tokens = batch["attention_mask"].bool()
            field_ids[active_tokens] = mask_token_id
        else:
            mlm_labels[mask] = field_ids[mask]
            field_ids[mask_token_positions] = mask_token_id
            field_vocab_size = vocab_sizes.get(field)
            if field_vocab_size:
                num_random = int(random_token_positions.sum().item())
                low, high = len(special_token_ids), field_vocab_size
                if num_random > 0 and high > low:
                    field_ids[random_token_positions] = torch.randint(
                        low, high, (num_random,)
                    )
        result[f"mlm_labels_{field}"] = mlm_labels
        result[f"input_ids_{field}"] = field_ids
    return result


# pylint: disable=too-many-locals
def create_kc_batch(
    batch: Dict[str, torch.Tensor],
    _tokenizer: Tokenizer,
    target_specs: Dict[str, int],
    *,
    large_head_threshold: int = 4096,
    max_pos_per_sample: int = 64,
) -> Dict[str, torch.Tensor]:
    """Create target batches for Knowledge Component (KC) training.

    Round 13: Hybrid dense/sparse approach:
    - For small heads (vocab_size <= large_head_threshold): dense multi-hot (B, V)
    - For large heads: sparse positive indices (B, P) with mask
    """
    # pylint: disable=too-many-locals
    result: Dict[str, torch.Tensor] = {}
    attn = batch["attention_mask"].bool()
    batch_size = int(attn.size(0))

    for name, vocab_size in target_specs.items():
        # Strip prefixes for multi-task heads (bag_lemma -> lemma)
        field_name = name
        if "_" in name:
            parts = name.split("_")
            if parts[0] in ["bag", "tail", "ngram", "prefix"]:
                field_name = "_".join(parts[1:])

        input_key = f"input_ids_{field_name}"
        if input_key not in batch:
            continue

        ids = batch[input_key]  # (B, T)

        if vocab_size <= large_head_threshold:
            # Dense path for small heads
            multi_hot = torch.zeros((batch_size, vocab_size), device=ids.device)
            for i in range(batch_size):
                tok = ids[i, attn[i]]
                tok = tok[tok >= 4]  # skip specials
                if tok.numel() == 0:
                    continue
                uniq = torch.unique(tok)
                uniq = uniq[uniq < vocab_size]
                if uniq.numel() > 0:
                    multi_hot[i, uniq] = 1.0
            result[f"kc_targets_{name}"] = multi_hot
        else:
            # Sparse path for large heads
            pos_inds = torch.full(
                (batch_size, max_pos_per_sample),
                -1,
                dtype=torch.long,
                device=ids.device,
            )
            pos_mask = torch.zeros(
                (batch_size, max_pos_per_sample), dtype=torch.bool, device=ids.device
            )
            for i in range(batch_size):
                tok = ids[i, attn[i]]
                tok = tok[tok >= 4]
                if tok.numel() == 0:
                    continue
                uniq = torch.unique(tok)
                uniq = uniq[uniq < vocab_size]
                if uniq.numel() == 0:
                    continue
                if uniq.numel() > max_pos_per_sample:
                    uniq = uniq[:max_pos_per_sample]
                n = int(uniq.numel())
                pos_inds[i, :n] = uniq
                pos_mask[i, :n] = True
            result[f"kc_pos_inds_{name}"] = pos_inds
            result[f"kc_pos_mask_{name}"] = pos_mask

    return result
