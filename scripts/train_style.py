# ruff: noqa: E402
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
        StyleClassifierWithMLM, MLMTrainer, Trainer

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
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter
from datetime import timedelta
from dataclasses import dataclass

from typing import Dict, List, Optional, Tuple, Any, cast

import yaml

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast

# Early logging to show progress during slow imports
_is_main_rank0 = os.environ.get("RANK", "0") == "0" and mp.current_process().name == 'MainProcess'
# Also suppress if we are being imported by the label script (which does its own logging)
_is_labeling = "scripts.label" in sys.modules or (len(sys.argv) > 0 and "label" in sys.argv[0])

if _is_main_rank0 and not _is_labeling:
    print("Loading PyTorch...", flush=True)
import torch  # noqa: E402
if _is_main_rank0 and not _is_labeling:
    print("Loading neural network modules...", flush=True)

import torch.nn as nn  # noqa: E402
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from kotogram.kotogram import split_kotogram, extract_token_features
from kotogram.japanese_parser import JapaneseParser
from kotogram import locations

from kotogram.model import (
    StyleClassifier, Tokenizer, ModelConfig,
    FEATURE_FIELDS, ALL_FEATURE_FIELDS, set_excluded_features,
    NUM_FORMALITY_PRAGMATIC_CLASSES, NUM_GENDER_PRAGMATIC_CLASSES, NUM_GRAMMATICALITY_CLASSES, NUM_REGISTER_CLASSES,
    FORMALITY_LABEL_TO_ID,
    GENDER_LABEL_TO_ID, load_model
)

from scripts.style_data import Sample, ProcessedSample  # noqa: E402



# Start timing immediately
script_start_time = time.time()
timings = {}







GENDER_LOSS_WEIGHT = 10.0


def setup_distributed() -> Tuple[int, int, int]:
    """Initialize distributed training if available.

    Returns:
        Tuple of (rank, world_size, local_rank)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend="nccl", 
                init_method="env://", 
                device_id=torch.device(f"cuda:{local_rank}"),
                timeout=timedelta(minutes=60)
            )
            print(f"Distributed init: Rank {rank}/{world_size} (Local {local_rank})")
            return rank, world_size, local_rank

    # Default to single process
    return 0, 1, 0


def is_main_process() -> bool:
    """Check if we are on the main process (rank 0)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _collect_tokens_batch(kotograms: List[str]) -> Dict[str, Counter]:
    """Collect token counts from a batch of kotograms in parallel.
    
    Returns:
        Dict mapping field_name -> Counter of token values
    """
    from kotogram.model import FEATURE_FIELDS
    
    counters: Dict[str, Counter[Any]] = {f: Counter() for f in FEATURE_FIELDS}
    
    for k in kotograms:
        tokens = split_kotogram(k)
        for token in tokens:
            token_feat = extract_token_features(token)
            for field in FEATURE_FIELDS:
                value = getattr(token_feat, field)
                if field in counters: # Only track active features
                    counters[field][value] += 1
                    
    return counters


def _encode_samples_batch(
    items: List['ProcessedSample'], 
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
    
    for item in items:
        # Tuple validation handled by NamedTuple
        
        # Manually encode to avoid tokenizer overhead if possible, 
        # or just use tokenizer.encode. tokenizer.encode is fast if frozen.
        feature_ids = tokenizer.encode(item.kotogram, add_cls=True, add_to_vocab=False)
        
        # Map formality_id to value/pragmatic
        # 0=VeryFormal(1.0), 1=Formal(0.5), 2=Neutral(0.0), 3=Casual(-0.5), 4=VeryCasual(-1.0)
        # 5=Unpragmatic -> Value=0.0, Pragmatic=0
        f_id = item.formality_id
        if f_id == 5: # UNPRAGMATIC_FORMALITY
            f_val = 0.0
            f_prag = 0
        else:
            f_val = {0: 1.0, 1: 0.5, 2: 0.0, 3: -0.5, 4: -1.0}.get(f_id, 0.0)
            f_prag = 1

        sample = Sample(
            feature_ids=feature_ids,
            formality_value=f_val,
            formality_pragmatic=f_prag,
            gender_value=item.gender_value,
            gender_pragmatic=item.gender_pragmatic,
            register_labels=item.register_ids,
            grammaticality_label=item.gram_label,
            # original_sentence=item.sentence_id, # Actually using sentence_id not sentence string in some cases? 
            # Wait, item.sentence is the text. item.sentence_id is ID. Sample.original_sentence should probably be sentence.
            # Checking previous code: for sentence, kotogram... -> original_sentence=sentence
            # So item.sentence is correct.
            original_sentence=item.sentence,
            kotogram=item.kotogram,
        )
        samples.append(sample)
        
    return samples










# Cache version - bump this when cache format changes to invalidate old caches
# v1: Initial version
# v2: Removed lemma pruning (lemma_min_freq, max_lemma_vocab)
# v3: Added parallel processing
# v4: Added register labeling
# v6: Added register_label to Sample
# v7: Changed register_label to register_labels (multi-label)
CACHE_VERSION = 10


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
    def _get_vocab_cache_path(
        tsv_paths: List[str],
        labeled: bool,
        grammaticality_labels: Optional[List[int]],
    ) -> str:
        """Generate a cache file path for the vocabulary configuration."""
        # Build a hash from all relevant parameters
        hash_parts = [f"v{CACHE_VERSION}"] 

        for tsv_path in tsv_paths:
            hash_parts.append(tsv_path)
            # Include file modification time
            if os.path.exists(tsv_path):
                mtime = os.path.getmtime(tsv_path)
                hash_parts.append(str(mtime))
        
        # Include row count hash if available (to detect content changes without full re-read)
        # For now, mtime + path is a reasonable proxy for "same input file"
        
        hash_parts.append(f"labeled={labeled}")
        hash_parts.append(f"gram_labels={grammaticality_labels}")

        # Create hash
        hash_str = hashlib.sha256("|".join(hash_parts).encode()).hexdigest()[:16]
        
        from kotogram import locations
        cache_dir = locations.get_style_dataset_cache_dir()
        return os.path.join(cache_dir, f"vocab_v{CACHE_VERSION}_{hash_str}.json")

    @staticmethod
    def _load_vocab(
        cache_path: str,
        tokenizer: Tokenizer,
    ) -> None:
        """Load tokenizer vocabulary from cache. Raises error if missing."""
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Vocabulary not found at {cache_path}. Run label.py first.")

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check version
            if data.get('version') != CACHE_VERSION:
                raise ValueError(f"Cache version mismatch. Expected {CACHE_VERSION}, got {data.get('version')}")

            # Restore tokenizer state
            tokenizer.field_vocabs = data['field_vocabs']
            tokenizer._frozen = bool(data.get('frozen', False))
            
        except Exception as e:
            raise ValueError(f"Failed to load vocabulary from {cache_path}: {e}")





    @classmethod
    def _process_parallel(
        cls,
        rows: List[Tuple[str, str, int]],
        batch_size: int = 1000,
        num_workers: Optional[int] = None,
        verbose: bool = True,
    ) -> List[ProcessedSample]:
        """Process rows by retrieving pre-computed labels from cache."""
        from scripts.cache import get_kotogram_cache
        
        cache = get_kotogram_cache()
        
        # We process in chunks to avoid huge memory/IO overhead for very large datasets
        chunk_size = 10000
        chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
        
        final_results: List[ProcessedSample] = []
        total_processed = 0
        
        if verbose:
             print(f"Loading {len(rows)} samples from cache...")

        for i, chunk in enumerate(chunks):
            chunk_sentences = [r[0] for r in chunk]
            cached_data_map = cache.get_batch(chunk_sentences)
            
            for sentence, sent_id, gram_label in chunk:
                cached_tuple = cached_data_map.get(sentence)
                if cached_tuple is not None:
                    # Hit
                    k, f_lbl, g_val, g_prag, r_lbls, cached_gram_lbl = cached_tuple
                    
                    processed_sample = ProcessedSample(
                        sentence=sentence,
                        sentence_id=sent_id,
                        kotogram=k,
                        formality_id=cast(int, f_lbl) if f_lbl is not None else 2,
                        gender_value=cast(float, g_val) if g_val is not None else 0.0,
                        gender_pragmatic=cast(int, g_prag) if g_prag is not None else 0,
                        register_ids=cast(List[int], r_lbls) if r_lbls is not None else [0],
                        gram_label=gram_label,
                        success=1
                    )
                    final_results.append(processed_sample)
                else:
                    # Strict mode: If not in cache, we fail.
                    # This assumes label.py has been run on the dataset.
                    raise ValueError(f"Sentence not found in cache: {sentence[:30]}... Run label.py first.")
            
            total_processed += len(chunk)
            if verbose and total_processed % 50000 == 0:
                print(f"\r  Loaded {total_processed}/{len(rows)}...", end="", flush=True)
        
        if verbose:
            print()  # Final newline after progress

        return final_results

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
        sample_ratio: float = 1.0,
    ) -> 'StyleDataset':
        """Load dataset from a single TSV file."""
        return cls.from_multiple_tsv(
            tsv_paths=[tsv_path],
            tokenizer=tokenizer,
            parser=parser,
            max_samples=max_samples,
            verbose=verbose,
            labeled=labeled,
            use_cache=use_cache,
            sample_ratio=sample_ratio
        )




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
        cache_name: Optional[str] = "vocab.json",
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
            sample_ratio: Ratio of data to use (0.0 to 1.0)

        Returns:
            StyleDataset with encoded samples from all files
        """
        # Derive cache directory
        from kotogram import locations
        cache_dir = locations.get_style_dataset_cache_dir()

        # Default all files to grammatic if not specified
        if grammaticality_labels is None:
            grammaticality_labels = [1] * len(tsv_paths)
        elif len(grammaticality_labels) != len(tsv_paths):
            raise ValueError(
                f"grammaticality_labels length ({len(grammaticality_labels)}) "
                f"must match tsv_paths length ({len(tsv_paths)})"
            )

        # Check Vocabulary Cache
        vocab_path = ""
        
        if use_cache:
            if cache_name:
                vocab_path = os.path.join(cache_dir, cache_name)
                # Check for label_metadata.json validation
                metadata_path = os.path.join(cache_dir, "label_metadata.json")
                if os.path.exists(vocab_path) and os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        # Check validity using either output fingerprints (preferred) or source fingerprints
                        is_valid = False
                        
                        # extracting current paths
                        g_path = next((p for p, lbl in zip(tsv_paths, grammaticality_labels) if lbl == 1), None) if grammaticality_labels else tsv_paths[0]
                        a_path = next((p for p, lbl in zip(tsv_paths, grammaticality_labels) if lbl == 0), None) if grammaticality_labels else None

                        if 'output_fingerprints' in metadata:
                            # Direct artifact validation: check if the provided tsv_paths match the ones produced by label.py
                            from scripts.label import get_file_fingerprint
                            outs = metadata['output_fingerprints']
                            is_valid = True
                            
                            # Check grammatic file
                            if g_path and 'grammatic' in outs:
                                fp_current = get_file_fingerprint(g_path)
                                fp_stored = outs['grammatic']
                                # Robust comparison for mtime (JSON float precision issues)
                                if fp_current and fp_stored and (fp_current['size'] != fp_stored['size'] or 
                                    abs(fp_current['mtime'] - fp_stored['mtime']) > 0.001):
                                    if verbose:
                                        print(f"  Fingerprint mismatch (Grammatic): Current {fp_current} != Stored {fp_stored}")
                                    is_valid = False
                                elif not fp_current:
                                    is_valid = False
                            elif g_path or 'grammatic' in outs:
                                # One exists but not the other -> mismatch configuration
                                is_valid = False

                            # Check agrammatic file
                            if is_valid:
                                if a_path and 'agrammatic' in outs:
                                    fp_current = get_file_fingerprint(a_path)
                                    fp_stored = outs['agrammatic']
                                    if fp_current and fp_stored and (fp_current['size'] != fp_stored['size'] or 
                                        abs(fp_current['mtime'] - fp_stored['mtime']) > 0.001):
                                        if verbose:
                                            print(f"  Fingerprint mismatch (Arawmatic): Current {fp_current} != Stored {fp_stored}")
                                        is_valid = False
                                    elif not fp_current:
                                        is_valid = False
                                elif a_path or 'agrammatic' in outs:
                                    # Mismatch in configuration
                                    is_valid = False 
            
                        # else: Strict validation requires output_fingerprints.
                        # If missing, is_valid remains False (initialized at start of loop).

                        if is_valid:
                            cls._load_vocab(vocab_path, tokenizer)
                            if verbose:
                                print(f"  Loaded vocabulary from cache: {vocab_path}")
                                print(f"  Vocabulary. Sizes: {tokenizer.get_vocab_sizes()}")
                        else:
                            if verbose:
                                print("  Cache validation failed or mismatch.")
                            # Strict mode: If cache is invalid, we CANNOT proceed.
                            if 'label_metadata.json' in os.listdir(cache_dir):
                                 # Metadata exists but invalid?
                                 raise ValueError("Cache invalid or modified. Please re-run: ./train_style.sh --label --force-relabel")
                            else:
                                 raise FileNotFoundError("Cache metadata not found. Please run: ./train_style.sh --label")

                    except Exception as e:
                        if verbose:
                             print(f"  Cache validation error: {e}")
                        raise 

        # Strictly require vocabulary
        if len(tokenizer.field_vocabs['surface']) <= 4: # Only special tokens
             # Check if we should try loading explicitly?
             # from_multiple_tsv should have loaded it if is_valid.
             # If strict, we just fail if not loaded.
             raise ValueError("Vocabulary not loaded. Ensure label.py finished successfully.")

        preprocessing_start = time.time()
        
        # Phase 1: Read rows from TSV files (fast I/O)
        # If sample_ratio < 1.0, we subsample during reading to avoid loading entire files.
        # Tuple: (sentence, sentence_id, gram_label)
        all_rows: List[Tuple[str, str, int]] = []
        phase1_start = time.time()
        
        import random
        random.seed(42)  # Deterministic sampling
        
        for tsv_path, gram_label in zip(tsv_paths, grammaticality_labels):
            if verbose:
                gram_str = "grammatic" if gram_label == 1 else "agrammatic"
                print(f"Reading {tsv_path} ({gram_str})...")

            file_count = 0
            file_rows: List[Tuple[str, str, int]] = []
            with open(tsv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if not row:
                        continue
                    
                    # Support both 3-column (ID, Lang, Sentence) and 1-column (Sentence only)
                    if len(row) >= 3:
                        sentence_id, _lang, sentence = row[0], row[1], row[2]
                    elif len(row) == 1:
                        sentence_id = ""
                        sentence = row[0]
                    else:
                        continue
                        
                    # Early subsampling: skip rows based on sample_ratio
                    if sample_ratio < 1.0 and random.random() >= sample_ratio:
                        continue

                    file_rows.append((sentence, sentence_id, gram_label))
                    if max_samples and len(all_rows) + len(file_rows) >= max_samples:
                        break
            
            all_rows.extend(file_rows)
            file_count = len(file_rows)

            if verbose:
                print(f"  Read {file_count} sentences from {tsv_path}")

            if max_samples and len(all_rows) >= max_samples:
                break

        if verbose:
            print(f"\nTotal sentences to load: {len(all_rows)}")
            
        phase1_duration = time.time() - phase1_start

        # Phase 2: Process sentences in parallel (kotogram conversion + label computation)
        phase2_start = time.time()
        # Phase 2: Process sentences in parallel (kotogram conversion + label computation)
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
            vocab_path=vocab_path
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
    ) -> 'StyleDataset':
        """Initialize dataset from pre-processed samples (already containing kotograms and labels).
        
        This method handles:
        1. Token collection (building vocabulary) if add_to_vocab=True
        2. Sample encoding using the tokenizer
        3. Saving the vocabulary cache
        4. Subsampling
        """
        if preprocessing_start is None:
            preprocessing_start = time.time()
            
        phase3a_duration = 0.0
        ctx = mp.get_context('spawn')
        num_workers = max(1, mp.cpu_count() - 1)

        # Phase 3: Build samples (Parallelized)
        if verbose:
            print("\nFinalizing dataset and building vocabulary...")

        # 3a. Token Collection Removed (Handled by label.py)
        # Vocabulary must be frozen already
        tokenizer.freeze()
        phase3a_duration = 0.0
        
        # Freeze vocabulary
        tokenizer.freeze()
        
        # 3b. Parallel Sample Encoding
        phase3b_start = time.time()
        if verbose:
            print("  Encoding samples with frozen tokenizer...")
            
        tokenizer_state = {
            'field_vocabs': tokenizer.field_vocabs,
        }
        
        encoding_inputs = [p for p in processed_results if p.success]
        batches = [encoding_inputs[i:i + 5000] for i in range(0, len(encoding_inputs), 5000)]
        
        samples: List[Sample] = []
        gender_counts: Counter[int] = Counter()
        grammaticality_counts: Counter[int] = Counter()

        processed_encodings = 0
        with ctx.Pool(num_workers) as pool:
            pool_args = [(b, tokenizer_state) for b in batches]
            
            for batch_samples in pool.starmap(_encode_samples_batch, pool_args):
                samples.extend(batch_samples)
                processed_encodings += len(batch_samples)
                
                # Update stats
                for s in batch_samples:
                    gender_counts[s.gender_pragmatic] += 1
                    grammaticality_counts[s.grammaticality_label] += 1
                    
                if verbose and processed_encodings % 100000 < 5000:
                     print(f"  Encoded {processed_encodings}/{len(encoding_inputs)} samples...")

        if verbose:
            print(f"\nDataset loaded: {len(samples)} samples")
            print(f"Vocabulary sizes: {tokenizer.get_vocab_sizes()}")
            print("Gender distribution (Pragmatic=1, Unpragmatic=0):")
            for g_prag, g_count in sorted(gender_counts.items(), key=lambda x: x[1], reverse=True):
                label = "Pragmatic" if g_prag == 1 else "Unpragmatic"
                print(f"  {label}: {g_count} ({100*g_count/len(samples):.1f}%)")
            print("Grammaticality distribution:")
            gram_labels_map = {1: "grammatic", 0: "agrammatic"}
            for g_id in [1, 0]:
                g_count = grammaticality_counts.get(g_id, 0)
                print(f"  {gram_labels_map[g_id]}: {g_count} ({100*g_count/len(samples):.1f}%)")

        # Freeze vocabulary again to finalize lemma vocab
        tokenizer.freeze()

        phase3b_duration = time.time() - phase3b_start
        total_preprocessing_duration = time.time() - preprocessing_start
        
        # Write timing metrics to timing.yml
        if is_main_process():
            timing_path = os.path.join(locations.get_style_support_dir(), "timing.yml")
            os.makedirs(os.path.dirname(timing_path), exist_ok=True)
            
            # Use yaml from global scope
            if 'yaml' in globals() and yaml:
                current_timing: Dict[str, float] = {}
                try:
                    if os.path.exists(timing_path):
                        with open(timing_path, 'r') as f:
                            current_timing = yaml.safe_load(f) or {}
                    
                    current_timing.update({
                        'preprocessing_phase1_io': phase1_duration,
                        'preprocessing_phase2_parsing': phase2_duration,
                        'preprocessing_phase3a_token_collection': phase3a_duration,
                        'preprocessing_phase3b_encoding': phase3b_duration,
                        'preprocessing_total': total_preprocessing_duration
                    })
                    
                    with open(timing_path, 'w') as f:
                        yaml.dump(current_timing, f, default_flow_style=False)
                    if verbose:
                        print(f"Detailed preprocessing timing saved to {timing_path}")
                except Exception:
                    pass

        # No need to save vocab anymore. Handled by label.py.

        # Note: Subsampling now happens during Phase 1 (TSV reading) for efficiency.

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
                combined_label = (sample.formality_pragmatic, sample.gender_pragmatic, sample.grammaticality_label)
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
        counts = Counter(s.formality_pragmatic for s in self.samples)
        total = len(self.samples)
        weights = torch.zeros(NUM_FORMALITY_PRAGMATIC_CLASSES)

        for label_id, count in counts.items():
            weights[label_id] = total / (NUM_FORMALITY_PRAGMATIC_CLASSES * count) if count > 0 else 0.0

        return weights

    def get_gender_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights for imbalanced gender pragmatic data."""
        counts = Counter(s.gender_pragmatic for s in self.samples)
        total = len(self.samples)
        weights = torch.zeros(NUM_GENDER_PRAGMATIC_CLASSES)

        for label_id, count in counts.items():
            weights[label_id] = total / (NUM_GENDER_PRAGMATIC_CLASSES * count) if count > 0 else 0.0

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
        'formality_labels', 'gender_value', 'gender_pragmatic', 'grammaticality_labels' tensors,
        and 'original_sentence', 'kotogram' lists.
    """
    batch_max_len = max(s.seq_len for s in batch)
    # Apply truncation if max_seq_len is specified
    max_len = min(batch_max_len, max_seq_len) if max_seq_len else batch_max_len

    # Initialize per-field lists
    field_ids: Dict[str, List[List[int]]] = {f: [] for f in FEATURE_FIELDS}
    attention_mask = []
    formality_values = []
    formality_pragmatics = []
    gender_values = []
    gender_pragmatics = []
    grammaticality_labels = []
    register_labels = []

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
        formality_values.append(sample.formality_value)
        formality_pragmatics.append(sample.formality_pragmatic)
        gender_values.append(sample.gender_value)
        gender_pragmatics.append(sample.gender_pragmatic)
        grammaticality_labels.append(sample.grammaticality_label)
        
        # Multi-hot encoding for register labels
        reg_vec = torch.zeros(NUM_REGISTER_CLASSES)
        for lbl in sample.register_labels:
            if 0 <= lbl < NUM_REGISTER_CLASSES:
                reg_vec[lbl] = 1.0
        register_labels.append(reg_vec)

    result: Dict[str, Any] = {
        f'input_ids_{field}': torch.tensor(field_ids[field], dtype=torch.long)
        for field in FEATURE_FIELDS
    }
    result['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
    result['formality_value'] = torch.tensor(formality_values, dtype=torch.float)
    result['formality_pragmatic'] = torch.tensor(formality_pragmatics, dtype=torch.long)
    result['gender_value'] = torch.tensor(gender_values, dtype=torch.float)
    result['gender_pragmatic'] = torch.tensor(gender_pragmatics, dtype=torch.long)
    result['grammaticality_labels'] = torch.tensor(grammaticality_labels, dtype=torch.long)
    result['register_labels'] = torch.stack(register_labels) # Stack vectors -> (B, NumClasses)

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
        *args: Any,
        mode: str = "classification",
        **kwargs: Any,
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
        for classifier in [self.formality_value_head, self.formality_pragmatic_head, self.gender_value_head, self.gender_pragmatic_head, self.grammaticality_classifier, self.register_classifier]:
            if isinstance(classifier, nn.Module):
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
    register_loss_weight: float = 1.0  # Weight for register loss in multi-task
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
        max_seq_len = getattr(self.model, "module", self.model).config.max_seq_len

        # Data sampler
        sampler: Optional[DistributedSampler] = None
        if self.is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.config.world_size,
                rank=dist.get_rank(),
                shuffle=True,  # Handle shuffling in sampler
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        self.sampler = sampler

        self.data_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle, # False if using sampler
            sampler=self.sampler,
            collate_fn=lambda b: collate_fn(b, pad_id, cast(Optional[int], max_seq_len)),
            pin_memory=True if self.config.device == "cuda" else False,
            num_workers=4 if self.config.device == "cuda" else 0,
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)

        # Get vocabulary sizes for all fields
        self.vocab_sizes = dataset.tokenizer.get_vocab_sizes()

        # Initialize field weights (default to 1.0 for all fields during pre-training)
        self.field_weights = {f: 1.0 for f in FEATURE_FIELDS}

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
                field_str = ", ".join(f"{f}={loss:.3f}" for f, loss in field_loss_dict.items())
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
        support_dir: Optional[str] = None,
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
        self.support_dir = support_dir or locations.get_style_support_dir()

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
        max_seq_len = getattr(self.model, "module", self.model).config.max_seq_len

        # Data samplers
        if self.is_distributed:
            self.train_sampler: Optional[DistributedSampler[Any]] = DistributedSampler(
                train_dataset,
                num_replicas=self.config.world_size,
                rank=dist.get_rank(),
                shuffle=True,
            )
            self.val_sampler: Optional[DistributedSampler[Any]] = DistributedSampler(
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
            collate_fn=lambda b: collate_fn(b, pad_id, cast(Optional[int], max_seq_len)),
            pin_memory=True if self.config.device == "cuda" else False,
            num_workers=4 if self.config.device == "cuda" else 0,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=val_shuffle,
            sampler=self.val_sampler,
            collate_fn=lambda b: collate_fn(b, pad_id, cast(Optional[int], max_seq_len)),
            pin_memory=True if self.config.device == "cuda" else False,
            num_workers=4 if self.config.device == "cuda" else 0,
        )

        # Loss functions with optional class weights
        if self.config.use_class_weights:
            formality_weights = train_dataset.get_formality_class_weights().to(self.device)
            gender_weights = train_dataset.get_gender_class_weights().to(self.device)
            grammaticality_weights = train_dataset.get_grammaticality_class_weights().to(self.device)
            self.formality_criterion = nn.CrossEntropyLoss(weight=formality_weights)
            self.gender_pragmatic_criterion = nn.CrossEntropyLoss(weight=gender_weights)
            self.grammaticality_criterion = nn.CrossEntropyLoss(weight=grammaticality_weights)
            # Register is multi-label, uses BCE (weights handled internally if needed, or pos_weight)
            # For now, no class weights for register to keep it simple
            self.register_criterion = nn.BCEWithLogitsLoss()
        else:
            self.formality_criterion = nn.CrossEntropyLoss()
            self.gender_pragmatic_criterion = nn.CrossEntropyLoss()
            self.grammaticality_criterion = nn.CrossEntropyLoss()
            self.register_criterion = nn.BCEWithLogitsLoss()

        # Optimizer with differential learning rates
        # Handle wrappped model
        model_module = cast(StyleClassifier, self.model.module if self.is_distributed else self.model)
        
        encoder_params = list(model_module.embedding.parameters()) + list(model_module.encoder.parameters())
        classifier_params = (
            list(model_module.formality_value_head.parameters()) +
            list(model_module.formality_pragmatic_head.parameters()) +
            list(model_module.gender_value_head.parameters()) +
            list(model_module.gender_pragmatic_head.parameters()) +
            list(model_module.grammaticality_classifier.parameters()) +
            list(model_module.register_classifier.parameters())
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
            'train_register_loss': [],
            'val_loss': [],
            'val_formality_loss': [],
            'val_gender_loss': [],
            'val_grammaticality_loss': [],
            'val_register_loss': [],
            'val_formality_accuracy': [], # Pragmatic accuracy
            'val_formality_mse': [], # Value MSE
            'val_gender_pragmatic_accuracy': [],
            'val_gender_value_mse': [],
            'val_grammaticality_accuracy': [],
            'val_register_accuracy': [],
        }
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.start_epoch = 0  # For resumption



    def train_epoch(self, verbose: bool = True) -> Tuple[float, float, float, float, float]:
        """Run one training epoch.

        Returns:
            Tuple of (total_loss, formality_loss, gender_loss, grammaticality_loss, register_loss)
        """
        self.model.train()
        total_loss = 0
        total_formality_loss = 0
        total_gender_loss = 0
        total_grammaticality_loss = 0
        total_register_loss = 0
        n_batches = 0
        total_batches = len(self.train_loader)

        if total_batches == 0:
            print("  WARNING: No batches in train_loader!")
            return 0.0, 0.0, 0.0, 0.0, 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            # Reconstruct field_inputs from flat batch keys
            field_inputs = {
                f'input_ids_{f}': batch[f'input_ids_{f}'].to(self.device) 
                for f in FEATURE_FIELDS
            }
            attention_mask = batch['attention_mask'].to(self.device)
            formality_value_targets = batch['formality_value'].to(self.device)
            formality_prag_targets = batch['formality_pragmatic'].to(self.device)
            gender_value_targets = batch['gender_value'].to(self.device)
            gender_prag_targets = batch['gender_pragmatic'].to(self.device)
            grammaticality_targets = batch['grammaticality_labels'].to(self.device)
            register_targets = batch['register_labels'].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision context
            device_type = 'cuda' if 'cuda' in str(self.device) else ('mps' if 'mps' in str(self.device) else 'cpu')
            
            with autocast(device_type=device_type, enabled=self.config.use_amp):
                # Forward pass
                # formality_value, formality_pragmatic, gender_value, gender_pragmatic, grammaticality, register
                formality_val_logits, formality_prag_logits, gender_val, gender_prag_logits, grammaticality_logits, register_logits = self.model(
                    field_inputs, attention_mask
                )

                # --- HYBRID TRAINING LOGIC ---
                # Masks
                is_grammatic = (grammaticality_targets == 1)
                is_formality_pragmatic = (formality_prag_targets == 1)
                is_gender_pragmatic = (gender_prag_targets == 1)
                
                # Valid style mask: fully valid sentence for training specific style values
                is_valid_style = is_grammatic & is_formality_pragmatic & is_gender_pragmatic

                # 1. Formality Loss (Value MSE + Pragmatic CrossEntropy)
                formality_prag_loss = self.formality_criterion(formality_prag_logits, formality_prag_targets)
                
                formality_val_squeezed = formality_val_logits.squeeze(-1)
                if is_valid_style.sum() > 0:
                     formality_val_loss = F.mse_loss(
                         formality_val_squeezed[is_valid_style], 
                         formality_value_targets[is_valid_style]
                     )
                else:
                     formality_val_loss = torch.tensor(0.0, device=self.device)

                formality_loss = formality_val_loss + formality_prag_loss # Equal weight 1.0 each inside formality

                # 2. Gender Loss (Value MSE + Pragmatic CrossEntropy)
                gender_prag_loss = self.gender_pragmatic_criterion(gender_prag_logits, gender_prag_targets)
                
                gender_val_squeezed = gender_val.squeeze(-1)
                if is_valid_style.sum() > 0:
                     gender_val_loss = F.mse_loss(
                         gender_val_squeezed[is_valid_style], 
                         gender_value_targets[is_valid_style]
                     )
                else:
                     gender_val_loss = torch.tensor(0.0, device=self.device)
                
                gender_loss = (gender_val_loss * GENDER_LOSS_WEIGHT) + gender_prag_loss

                # 3. Grammaticality Loss (All samples)
                grammaticality_loss = self.grammaticality_criterion(grammaticality_logits, grammaticality_targets)
                
                # 4. Register Loss (Only valid style samples)
                # We interpret agrammatic/unpragmatic as noise for register training
                if is_valid_style.sum() > 0:
                    register_loss = self.register_criterion(register_logits[is_valid_style], register_targets[is_valid_style])
                else:
                    register_loss = torch.tensor(0.0, device=self.device)

                # Weighted multi-task loss
                loss = (
                    self.config.formality_loss_weight * formality_loss +
                    self.config.gender_loss_weight * gender_loss +
                    self.config.grammaticality_loss_weight * grammaticality_loss +
                    self.config.register_loss_weight * register_loss
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
            total_register_loss += register_loss.item()
            n_batches += 1

            # Progress display (only on main process, updated every 100 batches to reduce I/O overhead)
            if verbose and is_main_process() and ((batch_idx + 1) % 100 == 0 or (batch_idx + 1) == total_batches):
                avg_loss_so_far = total_loss / n_batches
                progress = (batch_idx + 1) / total_batches
                bar_len = 30
                filled = int(bar_len * progress)
                bar = '=' * filled + '>' + '.' * (bar_len - filled - 1)
                sys.stdout.write(f'\r  [{bar}] {batch_idx+1}/{total_batches} loss={avg_loss_so_far:.4f}')
                sys.stdout.flush()

        if verbose and is_main_process():
            sys.stdout.write('\n')

        return total_loss / n_batches, total_formality_loss / n_batches, total_gender_loss / n_batches, total_grammaticality_loss / n_batches, total_register_loss / n_batches

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
        total_register_loss = 0
        n_batches = 0

        # Storage for metrics
        all_formality_val_preds = []
        all_formality_val_labels = []
        all_formality_prag_preds = []
        all_formality_prag_labels = []
        
        all_gender_val_preds = []
        all_gender_val_labels = []
        all_gender_prag_preds = []
        all_gender_prag_labels = []
        
        all_grammaticality_preds = []
        all_grammaticality_labels = []
        
        all_register_preds = []
        all_register_labels = []
        
        all_valid_style_mask = [] # To filter specific metrics later

        all_sentences = []
        all_kotograms = []

        for batch in self.val_loader:
             # Move batch to device
            field_inputs = {
                f'input_ids_{f}': batch[f'input_ids_{f}'].to(self.device) 
                for f in FEATURE_FIELDS
            }
            attention_mask = batch['attention_mask'].to(self.device)
            formality_value_targets = batch['formality_value'].to(self.device)
            formality_prag_targets = batch['formality_pragmatic'].to(self.device)
            gender_value_targets = batch['gender_value'].to(self.device)
            gender_prag_targets = batch['gender_pragmatic'].to(self.device)
            grammaticality_targets = batch['grammaticality_labels'].to(self.device)
            register_targets = batch['register_labels'].to(self.device)

            formality_val_logits, formality_prag_logits, gender_val, gender_prag_logits, grammaticality_logits, register_logits = self.model(
                field_inputs, attention_mask
            )

            # Masks
            is_grammatic = (grammaticality_targets == 1)
            is_formality_pragmatic = (formality_prag_targets == 1)
            is_gender_pragmatic = (gender_prag_targets == 1)
            is_valid_style = is_grammatic & is_formality_pragmatic & is_gender_pragmatic

            # 1. Formality Loss
            formality_prag_loss = self.formality_criterion(formality_prag_logits, formality_prag_targets)
            formality_val_squeezed = formality_val_logits.squeeze(-1)
            if is_valid_style.sum() > 0:
                 formality_val_loss = F.mse_loss(
                     formality_val_squeezed[is_valid_style], 
                     formality_value_targets[is_valid_style]
                 )
            else:
                 formality_val_loss = torch.tensor(0.0, device=self.device)
            formality_loss = formality_val_loss + formality_prag_loss

            # 2. Gender Loss
            gender_prag_loss = self.gender_pragmatic_criterion(gender_prag_logits, gender_prag_targets)
            gender_val_squeezed = gender_val.squeeze(-1)
            if is_valid_style.sum() > 0:
                 gender_val_loss = F.mse_loss(
                     gender_val_squeezed[is_valid_style], 
                     gender_value_targets[is_valid_style]
                 )
            else:
                 gender_val_loss = torch.tensor(0.0, device=self.device)
            gender_loss = (gender_val_loss * GENDER_LOSS_WEIGHT) + gender_prag_loss

            # 3. Grammaticality Loss
            grammaticality_loss = self.grammaticality_criterion(grammaticality_logits, grammaticality_targets)
            
            # 4. Register Loss
            if is_valid_style.sum() > 0:
                register_loss = self.register_criterion(register_logits[is_valid_style], register_targets[is_valid_style])
            else:
                register_loss = torch.tensor(0.0, device=self.device)
            
            loss = (
                self.config.formality_loss_weight * formality_loss +
                self.config.gender_loss_weight * gender_loss +
                self.config.grammaticality_loss_weight * grammaticality_loss +
                self.config.register_loss_weight * register_loss
            )

            total_loss += loss.item()
            total_formality_loss += formality_loss.item()
            total_gender_loss += gender_loss.item()
            total_grammaticality_loss += grammaticality_loss.item()
            total_register_loss += register_loss.item()
            n_batches += 1

            formality_prag_preds = formality_prag_logits.argmax(dim=-1)
            gender_prag_preds = gender_prag_logits.argmax(dim=-1)
            grammaticality_preds = grammaticality_logits.argmax(dim=-1)
            
            # Multi-label prediction
            register_probs = torch.sigmoid(register_logits)
            register_preds = (register_probs > 0.5).long()
            register_labels_long = register_targets.long()

            # Store predictions
            all_formality_val_preds.extend(formality_val_squeezed.detach().cpu().tolist())
            all_formality_val_labels.extend(formality_value_targets.cpu().tolist())
            all_formality_prag_preds.extend(formality_prag_preds.cpu().tolist())
            all_formality_prag_labels.extend(formality_prag_targets.cpu().tolist())
            
            all_gender_val_preds.extend(gender_val_squeezed.detach().cpu().tolist())
            all_gender_val_labels.extend(gender_value_targets.cpu().tolist())
            
            all_gender_prag_preds.extend(gender_prag_preds.cpu().tolist())
            all_gender_prag_labels.extend(gender_prag_targets.cpu().tolist())

            all_grammaticality_preds.extend(grammaticality_preds.cpu().tolist())
            all_grammaticality_labels.extend(grammaticality_targets.cpu().tolist())
            all_register_preds.extend(register_preds.cpu().tolist())
            all_register_labels.extend(register_labels_long.cpu().tolist())
            
            all_valid_style_mask.extend(is_valid_style.cpu().tolist())

            if return_mismatches or True: # Always collect sentences for confusion matrices
                all_sentences.extend(batch.get('original_sentence', []))
                all_kotograms.extend(batch.get('kotogram', []))

        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        avg_formality_loss = total_formality_loss / n_batches if n_batches > 0 else 0
        avg_gender_loss = total_gender_loss / n_batches if n_batches > 0 else 0
        avg_grammaticality_loss = total_grammaticality_loss / n_batches if n_batches > 0 else 0
        avg_register_loss = total_register_loss / n_batches if n_batches > 0 else 0


        # Calculate metrics (local)
        # Formality Pragmatic Accuracy (All)
        formality_acc = sum(p == label for p, label in zip(all_formality_prag_preds, all_formality_prag_labels)) / len(all_formality_prag_labels) if all_formality_prag_labels else 0
        
        # Gender Pragmatic Accuracy (All)
        gender_prag_acc = sum(p == label for p, label in zip(all_gender_prag_preds, all_gender_prag_labels)) / len(all_gender_prag_labels) if all_gender_prag_labels else 0
        
        # Grammaticality Accuracy (All)
        grammaticality_acc = sum(p == label for p, label in zip(all_grammaticality_preds, all_grammaticality_labels)) / len(all_grammaticality_labels) if all_grammaticality_labels else 0

        # Filtered Metrics (Only valid style)
        valid_indices = [i for i, m in enumerate(all_valid_style_mask) if m]
        
        # Formality Value MSE (Filtered)
        if valid_indices:
            f_mse = sum((all_formality_val_preds[i] - all_formality_val_labels[i]) ** 2 for i in valid_indices) / len(valid_indices)
            g_mse = sum((all_gender_val_preds[i] - all_gender_val_labels[i]) ** 2 for i in valid_indices) / len(valid_indices)
            
            valid_reg_preds = [all_register_preds[i] for i in valid_indices]
            valid_reg_labels = [all_register_labels[i] for i in valid_indices]
            
            # Register Exact Match (Filtered)
            register_acc = sum([1 for idx in range(len(valid_reg_preds)) if all(valid_reg_preds[idx][i] == valid_reg_labels[idx][i] for i in range(len(valid_reg_preds[idx])))]) / len(valid_reg_preds)
        else:
            f_mse = 0.0
            g_mse = 0.0
            register_acc = 0.0

        # Output Confusion Matrices (TSV)
        if is_main_process():
            # Create a dedicated directory for confusion matrices in style-support
            confusion_dir = os.path.join(self.support_dir, 'confusion_matrices')
            os.makedirs(confusion_dir, exist_ok=True)
            
            # 1. Formality Pragmatic Confusion (TSV)
            # Row: Actual, Col: Predicted
            with open(os.path.join(confusion_dir, 'formality_confusion.tsv'), 'w') as f:
                f.write('sentence\tactual_pragmatic\tpredicted_pragmatic\tactual_value\tpredicted_value\tis_valid_style\n')
                for i in range(len(all_sentences)):
                    f.write(f"{all_sentences[i]}\t{all_formality_prag_labels[i]}\t{all_formality_prag_preds[i]}\t{all_formality_val_labels[i]:.4f}\t{all_formality_val_preds[i]:.4f}\t{all_valid_style_mask[i]}\n")
            
            # 2. Gender Value MSE Confusion (TSV) - Top errors
            # Filtered by valid style
            mse_errors = []
            for i in valid_indices:
                error = (all_gender_val_preds[i] - all_gender_val_labels[i]) ** 2
                mse_errors.append((error, all_sentences[i], all_gender_val_preds[i], all_gender_val_labels[i]))
            
            mse_errors.sort(key=lambda x: x[0], reverse=True)
            
            with open(os.path.join(confusion_dir, 'gender_mse_confusion.tsv'), 'w') as f:
                f.write('sentence\terror\tpredicted_value\tactual_value\n')
                for err, sent, pred, act in mse_errors: # Dump all, let user filter top N
                    f.write(f"{sent}\t{err:.6f}\t{pred:.4f}\t{act:.4f}\n")
            
            # 3. Register Confusion (TSV) - Mismatches only, filtered by valid style
            with open(os.path.join(confusion_dir, 'register_confusion.tsv'), 'w') as f:
                f.write('sentence\tactual_labels\tpredicted_labels\n')
                for i in valid_indices:
                    # Check for mismatch
                    if any(all_register_preds[i][j] != all_register_labels[i][j] for j in range(NUM_REGISTER_CLASSES)):
                         # Convert to strings
                         act_str = ",".join(str(j) for j, val in enumerate(all_register_labels[i]) if val == 1)
                         pred_str = ",".join(str(j) for j, val in enumerate(all_register_preds[i]) if val == 1)
                         f.write(f"{all_sentences[i]}\t{act_str}\t{pred_str}\n")


        return {
            'loss': avg_loss,
            'formality_loss': avg_formality_loss,
            'gender_loss': avg_gender_loss,
            'grammaticality_loss': avg_grammaticality_loss,
            'register_loss': avg_register_loss,
            'formality_accuracy': formality_acc,
            'formality_value_mse': f_mse,
            'gender_pragmatic_accuracy': gender_prag_acc,
            'gender_value_mse': g_mse,
            'grammaticality_accuracy': grammaticality_acc,
            'register_accuracy': register_acc,
        }

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
            train_loss, train_formality_loss, train_gender_loss, train_grammaticality_loss, train_register_loss = self.train_epoch(verbose=verbose)
            eval_results = self.evaluate()

            self.scheduler.step(eval_results['loss'])

            self.history['train_loss'].append(train_loss)
            self.history['train_formality_loss'].append(train_formality_loss)
            self.history['train_gender_loss'].append(train_gender_loss)
            self.history['train_grammaticality_loss'].append(train_grammaticality_loss)
            self.history['train_register_loss'].append(train_register_loss)
            self.history['val_loss'].append(eval_results['loss'])
            self.history['val_formality_loss'].append(eval_results['formality_loss'])
            self.history['val_gender_loss'].append(eval_results['gender_loss'])
            self.history['val_grammaticality_loss'].append(eval_results['grammaticality_loss'])
            self.history['val_register_loss'].append(eval_results['register_loss'])
            self.history['val_formality_accuracy'].append(eval_results['formality_accuracy'])
            self.history['val_formality_mse'].append(eval_results['formality_value_mse'])
            self.history['val_gender_pragmatic_accuracy'].append(eval_results['gender_pragmatic_accuracy'])
            self.history['val_gender_value_mse'].append(eval_results['gender_value_mse'])
            self.history['val_grammaticality_accuracy'].append(eval_results['grammaticality_accuracy'])
            self.history['val_register_accuracy'].append(eval_results['register_accuracy'])

            if verbose:
                # Beautiful epoch summary with better formatting
                print(f"\n{'='*80}")
                print(f"📊 EPOCH {epoch+1}/{self.config.epochs} SUMMARY")
                print(f"{'='*80}")
                
                # Training metrics
                print("\n🔥 Training:")
                print(f"   Overall Loss:         {train_loss:.4f}")
                print(f"   ├─ Formality:         {train_formality_loss:.4f}")
                print(f"   ├─ Gender:            {train_gender_loss:.4f}")
                print(f"   ├─ Grammaticality:    {train_grammaticality_loss:.4f}")
                print(f"   └─ Register:          {train_register_loss:.4f}")
                
                # Validation metrics
                print("\n✅ Validation:")
                print(f"   Overall Loss:         {eval_results['loss']:.4f}")
                print(f"   ├─ Formality:         {eval_results['formality_loss']:.4f}")
                print(f"   ├─ Gender:            {eval_results['gender_loss']:.4f}")
                print(f"   ├─ Grammaticality:    {eval_results['grammaticality_loss']:.4f}")
                print(f"   └─ Register:          {eval_results['register_loss']:.4f}")
                
                # Accuracy metrics (as percentages)
                print("\n🎯 Accuracy:")
                print(f"   Formality (Pragmatic): {eval_results['formality_accuracy']*100:6.2f}%")
                print(f"   Formality (Value MSE): {eval_results['formality_value_mse']:.4f}")
                print(f"   Gender (Pragmatic):   {eval_results['gender_pragmatic_accuracy']*100:6.2f}%")
                print(f"   Gender (Value MSE):   {eval_results['gender_value_mse']:.4f}")
                print(f"   Grammaticality:       {eval_results['grammaticality_accuracy']*100:6.2f}%")
                print(f"   Register:             {eval_results['register_accuracy']*100:6.2f}%")
                
                # Learning rates
                enc_lr = self.optimizer.param_groups[0]['lr']
                cls_lr = self.optimizer.param_groups[1]['lr']
                print("\n📉 Learning Rates:")
                print(f"   Encoder:              {enc_lr:.2e}")
                print(f"   Classifier:           {cls_lr:.2e}")

            # Early stopping
            is_best_epoch = eval_results['loss'] < self.best_val_loss
            if is_best_epoch:
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
                    is_best=is_best_epoch,
                )

            # Synchronize all ranks after checkpoint saving to prevent desync
            # (Rank 0 might take time to save, causing others to timeout on next backward)
            if self.is_distributed:
                dist.barrier()

        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            self.model.to(self.device)

        return self.history


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
        if reset_optimizer:
            self.best_state = None
            print("  Note: Cleared best_state due to architecture change")
        else:
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
    is_best: bool = False,
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
    if not is_main_process():
        return

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

    if is_best:
        print(f"\n⭐ NEW BEST MODEL! Checkpoint saved at epoch {epoch + 1} (val_loss={best_val_loss:.4f})")
    else:
        print(f"\n💾 Checkpoint saved at epoch {epoch + 1}")
    print(f"{'='*80}\n")


def load_checkpoint(
    path: str,
    device: Optional[str] = None,
) -> Tuple[StyleClassifier, Tokenizer, Dict[str, Any], bool]:
    """Load training checkpoint for resumption.

    Args:
        path: Directory containing checkpoint
        device: Device to load model to

    Returns:
        Tuple of (model, tokenizer, checkpoint_dict, strict_load_failed)
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
    # Strip 'module.' prefix if present (from DDP training)
    model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
    
    model_state = {k: v for k, v in model_state.items() if not k.startswith('mlm_head.')}
    
    try:
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print("WARNING: Size mismatch detected (likely due to feature set changes). Reloading with loose matching.")
            # Filter out incompatible keys by attempting to load one by one or filtering based on logic?
            # Easiest is to filter out keys that cause size mismatch.
            # But load_state_dict doesn't return the specific mismatching keys in a nice way unless we catch the error.
            # Actually, the best way: iterate over model_state, check shape against model.state_dict().
            
            compatible_state = {}
            current_state = model.state_dict()
            mismatched_keys = []
            
            for k, v in model_state.items():
                if k in current_state:
                    if v.shape == current_state[k].shape:
                        compatible_state[k] = v
                    else:
                        mismatched_keys.append(k)
                else:
                    # Unexpected keys will be handled by strict=False anyway
                    compatible_state[k] = v 
            
            print(f"Skipping mismatched keys: {mismatched_keys}")
            missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)
            
            # Since we dropped keys, they will be in missing_keys now.
        else:
            raise e
    
    if missing_keys:
        print(f"WARNING: Missing keys in state_dict: {missing_keys}")
    if unexpected_keys:
        print(f"WARNING: Unexpected keys in state_dict: {unexpected_keys}")

    if device:
        model.to(device)

    return model, tokenizer, checkpoint, bool(missing_keys or unexpected_keys)





if __name__ == "__main__":
    if os.environ.get("RANK", "0") == "0":
        print("Starting training script...", flush=True)
    import argparse

    parser = argparse.ArgumentParser(description="Train style classifier (formality + gender)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to use (for testing)")
    parser.add_argument("--epochs", type=int, default=None,
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
    parser.add_argument("--exclude-features", type=str, default="",
                        help="Comma-separated list of features to exclude (for ablation study). "
                             f"Valid: {','.join(ALL_FEATURE_FIELDS)}")
    parser.add_argument("--fp16", action="store_true", default=None,
                        help="Save model in float16 precision (half size, minimal accuracy loss)")
    parser.add_argument("--fp8", action="store_true", default=None,
                        help="Save model in float8 precision (quarter size, requires PyTorch 2.1+). Default is True unless --fp16 is specified.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint in output directory")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain from scratch using parameters from existing checkpoint")
    parser.add_argument("--percent", type=float, default=None,
                        help="Percentage of data to use for training (1-100)")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Gradient accumulation steps for larger effective batch size")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of data loader workers")
    parser.add_argument("--local-rank", type=int, default=0,
                        help="Local rank for distributed training (usually passed by torchrun)")
    parser.add_argument("--preprocess-only", action="store_true",
                        help="Exit after loading and caching data (for multi-stage pipelines)")
    # parser.add_argument("--cache-dir", type=str, default=".cache", help="Base directory for cache (default: .cache)") # Removed



    args = parser.parse_args()
    
    # Resolve and inject paths from locations.py into args namespace
    cache_dir = locations.get_cache_dir()
    args.data = os.path.join(cache_dir, "grammatic_combined.tsv")
    args.agrammatic_data = os.path.join(cache_dir, "agrammatic_combined.tsv")
    args.output = locations.get_style_output_dir()
    args.support_dir = locations.get_style_support_dir()

    timings['args_parsing'] = time.time() - script_start_time

    if is_main_process():
        print("Setting up distributed training...", flush=True)
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    # If args.local_rank is set by argument (torchrun often sets this), prefer it, 
    # but our helper checks env vars which torchrun also sets.
    
    if is_main_process():
        print("Device initialization...", flush=True)
    
    # Log device information
    if is_main_process():
        print("  Probing CUDA...", flush=True)
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            print(f"\nDevice:         CUDA ({count} devices available)")
            print(f"  Name:         {name}")
            if world_size > 1:
                print(f"  Distributed:  d_model=DDP, {world_size} gpus, global_batch={args.batch_size * world_size * args.grad_accum_steps}")
                print(f"  Mixed Prec:   {'On' if args.fp16 else 'Off'}")
        else:
            print("  Probing MPS...", flush=True)
            if torch.backends.mps.is_available():
                print("  MPS found, initializing hardware...", flush=True)
                # Force initialization to happen while we are logging
                _ = torch.zeros(1, device="mps")
                print("\nDevice:         MPS (Apple Silicon)")
            else:
                print("\nDevice:         CPU")
                print("  Info:         Training will be slow. CUDA or MPS not found.")
    else:
        # Non-main processes should still print that they started if in debug mode
        if os.environ.get("DEBUG"):
            print(f"Process started: Rank {rank}/{world_size}")


    # Handle resume from checkpoint or retrain logic
    checkpoint = None
    if args.resume or args.retrain:
        checkpoint_path = os.path.join(args.support_dir, 'checkpoint.pt')
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

            strict_load_failed = False
            if args.resume:
                if is_main_process():
                    print(f"Resuming from checkpoint in {args.support_dir}...")
                model, tokenizer, checkpoint, strict_load_failed = load_checkpoint(args.support_dir)
            else:
                if is_main_process():
                    print(f"Retraining from scratch using parameters from {args.support_dir}...")
                pass

            # Override args with saved args (but keep epochs from command line to allow extending)
            if is_main_process():
                print("  Using saved parameters:")
                print(f"    data: {saved_args['data']}")
            # ... skipping full print for brevity in DDP

            print(f"    embed_dim: {saved_args['embed_dim']}")
            print(f"    hidden_dim: {saved_args['hidden_dim']}")
            print(f"    num_layers: {saved_args['num_layers']}")
            print(f"    num_heads: {saved_args['num_heads']}")
            print(f"    learning_rate: {saved_args['learning_rate']}")
            if args.resume:
                assert checkpoint is not None
                # Note: epochs count print is deferred until after restore
            else:
                # Note: epochs count print is deferred until after restore
                pass

            # Update args with saved values (except epochs which can be extended)
            # Note: We do NOT restore data paths (data, agrammatic_data) to allow
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

            # Restore epochs if not explicitly set on command line
            if args.epochs is None:
                args.epochs = saved_args.get('epochs', 20)
                print(f"  Restored flag: --epochs {args.epochs}")
            
            # Now print the resume/retrain info with correct epochs
            if args.resume and checkpoint is not None:
                print(f"  Resuming from epoch {checkpoint['epoch'] + 1}, training to epoch {args.epochs}")
            else:
                print(f"  Training from epoch 0 to {args.epochs}")
            
            # Sticky flags: restore only if not explicitly set on command line (i.e. None)
            if args.fp16 is None:
                args.fp16 = saved_args.get('fp16', False)
                if args.fp16:
                    print("  Restored flag: --fp16")
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
                    print("  Restored flag: --fp8")

            # Ensure they are boolean False if still None (and not restored to True from checkpoint)
            if args.fp16 is None:
                args.fp16 = False
            if args.fp8 is None:
                # If neither was specified on CLI nor found in checkpoint, default to FP8
                if not args.fp16:
                    args.fp8 = True
                else:
                    args.fp8 = False

        else:

            print(f"No checkpoint found at {checkpoint_path}, starting fresh training")
            args.resume = False
            # args.retrain = False # If no checkpoint, retrain just means train normally

    # Handle defaults for precision if not set via CLI or restored from checkpoint
    if args.fp16 is None:
        args.fp16 = False
    if args.fp8 is None:
        if not args.fp16:
            args.fp8 = True
        else:
            args.fp8 = False
    
    # Handle epochs default if not set via CLI or restored from checkpoint
    if args.epochs is None:
        args.epochs = 20


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
    if os.path.exists(args.agrammatic_data):
        data_files.append(args.agrammatic_data)
        grammaticality_labels.append(0)  # agrammatic sentences

    # Track if vocabulary grew during data loading/resume
    vocab_grew = False

    # Data validation (overlap check) removed as per optimization request.
    # We assume label.py outputs are correct and disjoint.
    pass

    # Load data: if doing MLM pretraining, first load unlabeled data for pretraining,
    # then load labeled data for fine-tuning

    # Skip model creation if resuming (model already loaded)
    t_data_start = time.time()
    if args.resume and checkpoint is not None:
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
        print("  (agrammatic data excluded from pretraining, will be used in fine-tuning)")
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
            dist.barrier(device_ids=[local_rank])
            
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
            num_formality_pragmatic_classes=NUM_FORMALITY_PRAGMATIC_CLASSES,
            num_gender_pragmatic_classes=NUM_GENDER_PRAGMATIC_CLASSES,
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
        if not args.preprocess_only:
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
                dist.barrier(device_ids=[local_rank])
                
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
                dist.barrier(device_ids=[local_rank])
                
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
                num_formality_pragmatic_classes=NUM_FORMALITY_PRAGMATIC_CLASSES,
                num_gender_pragmatic_classes=NUM_GENDER_PRAGMATIC_CLASSES,
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
                dist.barrier(device_ids=[local_rank])
            
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
                dist.barrier(device_ids=[local_rank])
                
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
            num_formality_pragmatic_classes=NUM_FORMALITY_PRAGMATIC_CLASSES,
            num_gender_pragmatic_classes=NUM_GENDER_PRAGMATIC_CLASSES,
            num_grammaticality_classes=NUM_GRAMMATICALITY_CLASSES,
            d_model=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            excluded_features=excluded,
        )

        if is_main_process():
            print("\nCreating model...")
        t_model_start = time.time()
        model = StyleClassifier(model_config)
        timings['model_creation'] = time.time() - t_model_start

    # Preprocessing only mode: Exit after data is loaded and cached
    if args.preprocess_only:
        if is_main_process():
            print("\nPreprocessing complete. Data is cached.")
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(0)


    if is_main_process():
        print(f"\nSplit: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # Supervised training with differential learning rates
    if is_main_process():
        print("\nStarting supervised training...")
    
    # Save timings
    timings['total_startup'] = time.time() - script_start_time
    if is_main_process():
        os.makedirs(".cache/style", exist_ok=True)
        timing_path = ".cache/style/timing.yml"
        existing_timings: Dict[str, float] = {}
        if os.path.exists(timing_path):
            with open(timing_path, "r") as f:
                existing_timings = yaml.safe_load(f) or {}
        
        existing_timings.update(timings)
        
        with open(timing_path, "w") as f:
            yaml.dump(existing_timings, f)
        print(f"Startup timings saved to {timing_path}")

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
        support_dir=args.support_dir,
    )

    # Restore training state if resuming
    if args.resume and checkpoint is not None:
        trainer.restore_from_checkpoint(checkpoint, reset_optimizer=vocab_grew or strict_load_failed)

    history = trainer.train(
        checkpoint_dir=args.support_dir,
        checkpoint_args=args,
        model_config=model_config,
        verbose=is_main_process(),
    )


    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id, model_config.max_seq_len),
    )

    model.eval()
    device = torch.device(trainer_config.device)
    total_formality_prag_acc = 0.0
    total_formality_val_sq_err = 0.0
    total_formality_prag_samples = 0
    total_gender_prag_acc = 0.0
    total_gender_val_sq_err = 0.0
    total_pragmatic_samples = 0
    total_gram_acc = 0.0
    total_register_acc = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            field_inputs = {k: v.to(device) for k, v in batch.items() if k.startswith('input_ids_')}
            attention_mask = batch['attention_mask'].to(device)
            formality_val_targets = batch['formality_value'].to(device)
            formality_prag_targets = batch['formality_pragmatic'].to(device)
            gender_value_targets = batch['gender_value'].to(device)
            gender_prag_targets = batch['gender_pragmatic'].to(device)
            grammaticality_labels = batch['grammaticality_labels'].to(device)
            register_labels = batch['register_labels'].to(device)
            
            formality_val_preds, formality_prag_logits, gender_val_preds, gender_prag_logits, grammaticality_logits, register_logits = model(field_inputs, attention_mask)
            
            register_preds = (torch.sigmoid(register_logits) > 0.5).long()

            # Formality Pragmatic Acc
            total_formality_prag_acc += (formality_prag_logits.argmax(-1) == formality_prag_targets).float().mean().item()
            
            # Formality Value MSE (masked by pragmatic)
            # Actually we use is_valid_style which includes gender_prag/gram too, but simpler here: 
            # just match how we report gender?
            # Or use the same logic as Trainer.evaluate?
            # Let's use simple logic: calculate MSE on ALL pragmatic samples for reporting here.
            # Mask: where formality_prag_targets == 1
            f_prag_mask = (formality_prag_targets == 1)
            if f_prag_mask.sum() > 0:
                 sq_err = F.mse_loss(formality_val_preds.squeeze(-1)[f_prag_mask], formality_val_targets[f_prag_mask], reduction='sum').item()
                 total_formality_val_sq_err += sq_err
                 total_formality_prag_samples += f_prag_mask.sum().item()
            
            # Gender Pragmatic Acc
            total_gender_prag_acc += (gender_prag_logits.argmax(-1) == gender_prag_targets).float().mean().item()
            
            # Gender Value MSE (masked)
            prag_mask = (gender_prag_targets == 1)
            if prag_mask.sum() > 0:
                 sq_err = F.mse_loss(gender_val_preds.squeeze(-1)[prag_mask], gender_value_targets[prag_mask], reduction='sum').item()
                 total_gender_val_sq_err += sq_err
                 total_pragmatic_samples += prag_mask.sum().item()

            total_gram_acc += (grammaticality_logits.argmax(-1) == grammaticality_labels).float().mean().item()
            total_register_acc += (register_preds == register_labels.long()).all(dim=1).float().mean().item()
            num_batches += 1
            
    f32_accuracy = (
        total_formality_prag_acc / num_batches,
        total_formality_val_sq_err / total_formality_prag_samples if total_formality_prag_samples > 0 else 0.0,
        total_gender_prag_acc / num_batches,
        total_gender_val_sq_err / total_pragmatic_samples if total_pragmatic_samples > 0 else 0.0,
        total_gram_acc / num_batches,
        total_register_acc / num_batches,
    )
    print(f"Test Accuracy (float32): form_prag={f32_accuracy[0]:.4f}, form_mse={f32_accuracy[1]:.4f}, gender_prag={f32_accuracy[2]:.4f}, gender_mse={f32_accuracy[3]:.4f}, gram={f32_accuracy[4]:.4f}, register={f32_accuracy[5]:.4f}")

    # Save model
    if is_main_process():
        print(f"\nSaving model to {args.output}...")
        if args.fp8:
            print("  (converting to float8 for smallest size)")
        elif args.fp16:
            print("  (converting to float16 for smaller size)")
        
        # Unwrap model if distributed
        model_to_save = model.module if hasattr(model, 'module') else model
        save_model(cast(StyleClassifier, model_to_save), tokenizer, args.output, model_config, fp16=args.fp16, fp8=args.fp8)
        print("Done!")

    # Verify reduced precision model accuracy if applicable
    if (args.fp16 or args.fp8) and is_main_process():
        precision_name = "fp8" if args.fp8 else "fp16"
        print(f"\nVerifying loaded {precision_name} model accuracy...")
        loaded_model, *unused = load_model(args.output, device=device.type)

        formality_prag_correct = 0
        formality_val_sq_err = 0.0
        form_pragmatic_samples = 0
        register_correct = 0
        gender_prag_correct = 0
        gender_val_sq_err = 0.0
        gender_pragmatic_samples = 0
        grammaticality_correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                field_inputs = {
                    k: v.to(device) for k, v in batch.items()
                    if k.startswith('input_ids_')
                }
                attention_mask = batch['attention_mask'].to(device)
                
                # Targets
                formality_value_targets = batch['formality_value'].to(device)
                formality_prag_targets = batch['formality_pragmatic'].to(device)
                gender_value_targets = batch['gender_value'].to(device)
                gender_prag_targets = batch['gender_pragmatic'].to(device)
                grammaticality_labels = batch['grammaticality_labels'].to(device)
                register_labels = batch['register_labels'].to(device)

                prediction = loaded_model.predict(field_inputs, attention_mask)
                
                # Predictions
                formality_prag_preds = prediction.formality_pragmatic_probs.argmax(dim=-1)
                gender_prag_preds = prediction.gender_pragmatic_probs.argmax(dim=-1)
                grammaticality_preds = prediction.grammaticality_probs.argmax(dim=-1)
                register_preds = (torch.sigmoid(prediction.register_probs) > 0.5).long() # Multi-label


                # Formality Metrics
                formality_prag_correct += (formality_prag_preds == formality_prag_targets).sum().item()
                f_prag_mask = (formality_prag_targets == 1)
                if f_prag_mask.size(0) > 0 and f_prag_mask.sum() > 0:
                     sq_err = F.mse_loss(prediction.formality_value.squeeze(-1)[f_prag_mask], formality_value_targets[f_prag_mask], reduction='sum').item()
                     formality_val_sq_err += sq_err
                     form_pragmatic_samples += f_prag_mask.sum().item()

                # Gender Metrics
                gender_prag_correct += (gender_prag_preds == gender_prag_targets).sum().item()
                g_prag_mask = (gender_prag_targets == 1)
                if g_prag_mask.size(0) > 0 and g_prag_mask.sum() > 0:
                     sq_err = F.mse_loss(prediction.gender_value.squeeze(-1)[g_prag_mask], gender_value_targets[g_prag_mask], reduction='sum').item()
                     gender_val_sq_err += sq_err
                     gender_pragmatic_samples += g_prag_mask.sum().item()

                grammaticality_correct += (grammaticality_preds == grammaticality_labels).sum().item()
                register_correct += (register_preds == register_labels.long()).all(dim=1).int().sum().item()
                total += formality_prag_targets.size(0)

        print(f"Test Accuracy ({precision_name}):")
        print(f"  Formality Prag: {formality_prag_correct/total:.4f}")
        if form_pragmatic_samples > 0:
             print(f"  Formality MSE: {formality_val_sq_err/form_pragmatic_samples:.4f}")
        print(f"  Gender Prag: {gender_prag_correct/total:.4f}")
        if gender_pragmatic_samples > 0:
            print(f"  Gender MSE: {gender_val_sq_err/gender_pragmatic_samples:.4f}")
        print(f"  Grammaticality: {grammaticality_correct/total:.4f}")
        print(f"  Register: {register_correct/total:.4f}")



    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
