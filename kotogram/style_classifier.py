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
import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from kotogram.kotogram import split_kotogram
from kotogram.analysis import formality, FormalityLevel, gender, GenderLevel
from kotogram.analysis import extract_token_features  # type: ignore[attr-defined]
from kotogram.japanese_parser import JapaneseParser


# Special token values for vocabulary
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
CLS_TOKEN = "<CLS>"
MASK_TOKEN = "<MASK>"  # For self-supervised pretraining

# Feature fields used for token embedding
# NOTE: 'surface' is critical for gender detection (pronouns like 僕, 俺, あたし)
FEATURE_FIELDS = ['surface', 'pos', 'pos_detail1', 'pos_detail2', 'conjugated_type', 'conjugated_form', 'lemma']

# Number of classes for each task
NUM_FORMALITY_CLASSES = 6
NUM_GENDER_CLASSES = 4
NUM_GRAMMATICALITY_CLASSES = 2  # grammatic (1) vs agrammatic (0)


class Tokenizer:
    """Tokenizer that extracts morphological features from Kotogram tokens.

    Instead of treating each token as a single vocabulary item, this tokenizer
    extracts categorical features (pos, pos_detail1, conjugated_type, conjugated_form,
    lemma) and maintains separate vocabularies for each field.

    This allows the model to generalize better across tokens with similar
    grammatical properties, even if the exact surface form is unseen.

    Attributes:
        field_vocabs: Dict mapping field name to {value: id} mapping
        field_vocab_sizes: Dict mapping field name to vocabulary size
        lemma_min_freq: Minimum frequency for lemma to be in vocabulary
    """

    def __init__(self, lemma_min_freq: int = 5, max_lemma_vocab: int = 10000):
        """Initialize feature tokenizer.

        Args:
            lemma_min_freq: Minimum frequency for a lemma to be included in vocabulary.
                           Less frequent lemmas map to UNK.
            max_lemma_vocab: Maximum vocabulary size for lemma field.
        """
        self.lemma_min_freq = lemma_min_freq
        self.max_lemma_vocab = max_lemma_vocab

        # Initialize vocabularies for each field with special tokens
        self.field_vocabs: Dict[str, Dict[str, int]] = {}
        self._field_counters: Dict[str, Counter[str]] = {}
        for f in FEATURE_FIELDS:
            self.field_vocabs[f] = {
                PAD_TOKEN: 0,
                UNK_TOKEN: 1,
                CLS_TOKEN: 2,
                MASK_TOKEN: 3,
            }
            self._field_counters[f] = Counter()

        self._frozen = False
        self._lemma_counts: Counter[str] = Counter()

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    @property
    def cls_id(self) -> int:
        return 2

    @property
    def mask_id(self) -> int:
        return 3

    def get_vocab_size(self, field: str) -> int:
        """Get vocabulary size for a specific field."""
        return len(self.field_vocabs[field])

    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary sizes for all fields."""
        return {field: len(vocab) for field, vocab in self.field_vocabs.items()}

    def _add_value(self, field: str, value: str) -> int:
        """Add a value to field vocabulary and return its ID."""
        if not value:
            value = UNK_TOKEN

        vocab = self.field_vocabs[field]
        if value in vocab:
            return vocab[value]

        if self._frozen:
            return self.unk_id

        # For lemma field, track counts for later pruning
        if field == 'lemma':
            self._lemma_counts[value] += 1
            # Don't add to vocab until finalize_vocab is called
            return self.unk_id

        new_id = len(vocab)
        vocab[value] = new_id
        return new_id

    def extract_features(self, kotogram: str) -> List[Dict[str, str]]:
        """Extract features from each token in a Kotogram string.

        Args:
            kotogram: Kotogram string to process

        Returns:
            List of feature dictionaries, one per token
        """
        tokens = split_kotogram(kotogram)
        features_list = []

        for token in tokens:
            features = extract_token_features(token)
            # Only keep the fields we use
            filtered = {field: features.get(field, '') for field in FEATURE_FIELDS}
            features_list.append(filtered)

        return features_list

    def encode_features(
        self,
        features_list: List[Dict[str, str]],
        add_cls: bool = True,
        add_to_vocab: bool = True,
    ) -> Dict[str, List[int]]:
        """Convert list of feature dicts to sequences of field IDs.

        Args:
            features_list: List of feature dictionaries from extract_features
            add_cls: If True, prepend CLS token IDs
            add_to_vocab: If True, add new values to vocabulary

        Returns:
            Dict mapping field name to list of token IDs for that field
        """
        result: Dict[str, List[int]] = {f: [] for f in FEATURE_FIELDS}

        if add_cls:
            for field in FEATURE_FIELDS:
                result[field].append(self.cls_id)

        for features in features_list:
            for field in FEATURE_FIELDS:
                value = features.get(field, '')
                if add_to_vocab and not self._frozen:
                    self._field_counters[field][value] += 1
                    token_id = self._add_value(field, value)
                else:
                    vocab = self.field_vocabs[field]
                    token_id = vocab.get(value, self.unk_id)
                result[field].append(token_id)

        return result

    def encode(
        self,
        kotogram: str,
        add_cls: bool = True,
        add_to_vocab: bool = True,
    ) -> Dict[str, List[int]]:
        """Encode a Kotogram string to feature ID sequences.

        Args:
            kotogram: Kotogram string to encode
            add_cls: If True, prepend CLS token
            add_to_vocab: If True, add new values to vocabulary

        Returns:
            Dict mapping field name to list of token IDs
        """
        features_list = self.extract_features(kotogram)
        return self.encode_features(features_list, add_cls, add_to_vocab)

    def finalize_vocab(self) -> None:
        """Finalize vocabulary, pruning rare lemmas.

        Should be called after processing all training data but before freezing.
        """
        # Prune lemma vocabulary to most frequent items
        vocab = self.field_vocabs['lemma']
        frequent_lemmas = self._lemma_counts.most_common(self.max_lemma_vocab)

        for lemma, count in frequent_lemmas:
            if count >= self.lemma_min_freq and lemma not in vocab:
                vocab[lemma] = len(vocab)

    def freeze(self) -> None:
        """Freeze vocabulary - new values will map to UNK."""
        self.finalize_vocab()
        self._frozen = True

    def unfreeze(self) -> None:
        """Unfreeze vocabulary."""
        self._frozen = False

    def get_model_config(self, **kwargs: Any) -> 'ModelConfig':
        """Create a ModelConfig with vocabulary sizes from this tokenizer.

        Args:
            **kwargs: Additional config parameters to override defaults

        Returns:
            ModelConfig instance
        """
        return ModelConfig(
            vocab_sizes=self.get_vocab_sizes(),
            **kwargs
        )

    def save(self, path: str) -> None:
        """Save tokenizer vocabularies to JSON file."""
        data = {
            'field_vocabs': self.field_vocabs,
            'lemma_min_freq': self.lemma_min_freq,
            'max_lemma_vocab': self.max_lemma_vocab,
            'frozen': self._frozen,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'Tokenizer':
        """Load tokenizer from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tokenizer = cls(
            lemma_min_freq=data.get('lemma_min_freq', 5),
            max_lemma_vocab=data.get('max_lemma_vocab', 10000),
        )
        tokenizer.field_vocabs = data['field_vocabs']
        tokenizer._frozen = data.get('frozen', False)
        return tokenizer


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


# Label mappings
FORMALITY_LABEL_TO_ID = {
    FormalityLevel.VERY_FORMAL: 0,
    FormalityLevel.FORMAL: 1,
    FormalityLevel.NEUTRAL: 2,
    FormalityLevel.CASUAL: 3,
    FormalityLevel.VERY_CASUAL: 4,
    FormalityLevel.UNPRAGMATIC_FORMALITY: 5,
}
FORMALITY_ID_TO_LABEL = {v: k for k, v in FORMALITY_LABEL_TO_ID.items()}

GENDER_LABEL_TO_ID = {
    GenderLevel.MASCULINE: 0,
    GenderLevel.FEMININE: 1,
    GenderLevel.NEUTRAL: 2,
    GenderLevel.UNPRAGMATIC_GENDER: 3,
}
GENDER_ID_TO_LABEL = {v: k for k, v in GENDER_LABEL_TO_ID.items()}


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

    @classmethod
    def from_tsv(
        cls,
        tsv_path: str,
        tokenizer: Tokenizer,
        parser: Optional[JapaneseParser] = None,
        max_samples: Optional[int] = None,
        verbose: bool = True,
        labeled: bool = True,
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

        Returns:
            StyleDataset with encoded samples
        """
        actual_parser: JapaneseParser
        if parser is None:
            from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
            actual_parser = SudachiJapaneseParser()
        else:
            actual_parser = parser

        samples: List[Sample] = []
        formality_counts: Counter[FormalityLevel] = Counter()
        gender_counts: Counter[GenderLevel] = Counter()

        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            total = 0

            for row in reader:
                if len(row) < 3:
                    continue

                sentence_id, lang, sentence = row[0], row[1], row[2]

                if lang != 'jpn':
                    continue

                try:
                    # Convert to Kotogram
                    kotogram = actual_parser.japanese_to_kotogram(sentence)

                    # Get labels (or dummy for pretraining)
                    if labeled:
                        formality_enum = formality(kotogram)
                        gender_enum = gender(kotogram)
                        formality_id = FORMALITY_LABEL_TO_ID[formality_enum]
                        gender_id = GENDER_LABEL_TO_ID[gender_enum]
                    else:
                        formality_enum = FormalityLevel.NEUTRAL
                        gender_enum = GenderLevel.NEUTRAL
                        formality_id = FORMALITY_LABEL_TO_ID[formality_enum]
                        gender_id = GENDER_LABEL_TO_ID[gender_enum]

                    # Encode to feature IDs (builds vocabulary)
                    feature_ids = tokenizer.encode(kotogram, add_cls=True, add_to_vocab=True)

                    sample = Sample(
                        feature_ids=feature_ids,
                        formality_label=formality_id,
                        gender_label=gender_id,
                        original_sentence=sentence,
                        kotogram=kotogram,
                    )
                    samples.append(sample)
                    formality_counts[formality_enum] += 1
                    gender_counts[gender_enum] += 1
                    total += 1

                    if verbose and total % 10000 == 0:
                        vocab_sizes = tokenizer.get_vocab_sizes()
                        print(f"Processed {total} sentences, vocab sizes: {vocab_sizes}")

                    if max_samples and total >= max_samples:
                        break

                except Exception as e:
                    if verbose:
                        print(f"Error processing sentence {sentence_id}: {e}")
                    continue

        if verbose:
            print(f"\nDataset loaded: {len(samples)} samples")
            print(f"Vocabulary sizes: {tokenizer.get_vocab_sizes()}")
            if labeled:
                print("Formality distribution:")
                for f_label, f_count in sorted(formality_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {f_label.value}: {f_count} ({100*f_count/len(samples):.1f}%)")
                print("Gender distribution:")
                for g_label, g_count in sorted(gender_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {g_label.value}: {g_count} ({100*g_count/len(samples):.1f}%)")

        # Freeze vocabulary after building
        tokenizer.freeze()

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

        Returns:
            StyleDataset with encoded samples from all files
        """
        actual_parser: JapaneseParser
        if parser is None:
            from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
            actual_parser = SudachiJapaneseParser()
        else:
            actual_parser = parser

        # Default all files to grammatic if not specified
        if grammaticality_labels is None:
            grammaticality_labels = [1] * len(tsv_paths)
        elif len(grammaticality_labels) != len(tsv_paths):
            raise ValueError(
                f"grammaticality_labels length ({len(grammaticality_labels)}) "
                f"must match tsv_paths length ({len(tsv_paths)})"
            )

        samples: List[Sample] = []
        formality_counts: Counter[FormalityLevel] = Counter()
        gender_counts: Counter[GenderLevel] = Counter()
        grammaticality_counts: Counter[int] = Counter()
        total = 0

        for tsv_path, gram_label in zip(tsv_paths, grammaticality_labels):
            if verbose:
                gram_str = "grammatic" if gram_label == 1 else "agrammatic"
                print(f"\nLoading from {tsv_path} ({gram_str})...")

            file_count = 0
            with open(tsv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')

                for row in reader:
                    if len(row) < 3:
                        continue

                    sentence_id, lang, sentence = row[0], row[1], row[2]

                    if lang != 'jpn':
                        continue

                    try:
                        # Convert to Kotogram
                        kotogram = actual_parser.japanese_to_kotogram(sentence)

                        # Get labels (or dummy for pretraining)
                        if labeled:
                            formality_enum = formality(kotogram)
                            gender_enum = gender(kotogram)
                            formality_id = FORMALITY_LABEL_TO_ID[formality_enum]
                            gender_id = GENDER_LABEL_TO_ID[gender_enum]
                        else:
                            formality_enum = FormalityLevel.NEUTRAL
                            gender_enum = GenderLevel.NEUTRAL
                            formality_id = FORMALITY_LABEL_TO_ID[formality_enum]
                            gender_id = GENDER_LABEL_TO_ID[gender_enum]

                        # Encode to feature IDs (builds vocabulary)
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
                        formality_counts[formality_enum] += 1
                        gender_counts[gender_enum] += 1
                        grammaticality_counts[gram_label] += 1
                        total += 1
                        file_count += 1

                        if verbose and total % 10000 == 0:
                            vocab_sizes = tokenizer.get_vocab_sizes()
                            print(f"Processed {total} sentences, vocab sizes: {vocab_sizes}")

                        if max_samples and total >= max_samples:
                            break

                    except Exception as e:
                        if verbose:
                            print(f"Error processing sentence {sentence_id}: {e}")
                        continue

            if verbose:
                print(f"  Loaded {file_count} samples from {tsv_path}")

            if max_samples and total >= max_samples:
                break

        if verbose:
            print(f"\nDataset loaded: {len(samples)} samples from {len(tsv_paths)} files")
            print(f"Vocabulary sizes: {tokenizer.get_vocab_sizes()}")
            if labeled:
                print("Formality distribution:")
                for f_label, f_count in sorted(formality_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {f_label.value}: {f_count} ({100*f_count/len(samples):.1f}%)")
                print("Gender distribution:")
                for g_label, g_count in sorted(gender_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {g_label.value}: {g_count} ({100*g_count/len(samples):.1f}%)")
            print("Grammaticality distribution:")
            gram_labels = {1: "grammatic", 0: "agrammatic"}
            for g_id in [1, 0]:
                g_count = grammaticality_counts.get(g_id, 0)
                print(f"  {gram_labels[g_id]}: {g_count} ({100*g_count/len(samples):.1f}%)")

        # Freeze vocabulary after building
        tokenizer.freeze()

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
) -> Dict[str, torch.Tensor]:
    """Collate samples into padded batches.

    Args:
        batch: List of Sample objects
        pad_id: Padding token ID
        max_seq_len: Maximum sequence length. Sequences longer than this will be
                    truncated. If None, uses the maximum length in the batch.

    Returns:
        Dictionary with per-field 'input_ids_<field>', 'attention_mask',
        'formality_labels', 'gender_labels', and 'grammaticality_labels' tensors
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

    return result


class PositionalEncoding(nn.Module):  # type: ignore[misc]
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        pe = cast(torch.Tensor, self.pe)
        x = x + pe[:, :x.size(1), :]
        return cast(torch.Tensor, self.dropout(x))


@dataclass
class ModelConfig:
    """Configuration for StyleClassifier model.

    This config supports multi-field embeddings where each morphological feature
    (pos, pos_detail1, conjugated_type, conjugated_form, lemma) has its own
    embedding table. The model has three classification heads: formality, gender,
    and grammaticality.
    """
    vocab_sizes: Dict[str, int]  # Field name -> vocabulary size
    num_formality_classes: int = NUM_FORMALITY_CLASSES
    num_gender_classes: int = NUM_GENDER_CLASSES
    num_grammaticality_classes: int = NUM_GRAMMATICALITY_CLASSES
    field_embed_dims: Dict[str, int] = field(default_factory=lambda: {
        'pos': 32,
        'pos_detail1': 32,
        'pos_detail2': 16,
        'conjugated_type': 32,
        'conjugated_form': 32,
        'lemma': 64,
    })
    d_model: int = 192  # Total model dimension after projection
    hidden_dim: int = 384
    num_layers: int = 3
    num_heads: int = 6
    dropout: float = 0.1
    max_seq_len: int = 512
    pooling: str = "cls"  # "cls", "mean", or "max"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'vocab_sizes': self.vocab_sizes,
            'num_formality_classes': self.num_formality_classes,
            'num_gender_classes': self.num_gender_classes,
            'num_grammaticality_classes': self.num_grammaticality_classes,
            'field_embed_dims': self.field_embed_dims,
            'd_model': self.d_model,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'max_seq_len': self.max_seq_len,
            'pooling': self.pooling,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModelConfig':
        return cls(**d)


class MultiFieldEmbedding(nn.Module):  # type: ignore[misc]
    """Embedding layer that combines multiple categorical feature embeddings.

    For each token position, this layer:
    1. Looks up embeddings for each feature field (pos, pos_detail1, etc.)
    2. Concatenates the field embeddings
    3. Projects to the model dimension

    This allows the model to learn from morphological features while
    maintaining a fixed-size representation for the transformer.
    """

    def __init__(self, config: ModelConfig):
        """Initialize multi-field embedding layer.

        Args:
            config: ModelConfig with vocabulary sizes and dimensions
        """
        super().__init__()
        self.config = config

        # Create embedding tables for each field
        self.embeddings = nn.ModuleDict()
        total_embed_dim = 0

        for field_name in FEATURE_FIELDS:
            vocab_size = config.vocab_sizes.get(field_name, 100)
            embed_dim = config.field_embed_dims.get(field_name, 32)
            self.embeddings[field_name] = nn.Embedding(
                vocab_size,
                embed_dim,
                padding_idx=0,  # PAD token
            )
            total_embed_dim += embed_dim

        # Projection to model dimension
        self.projection = nn.Linear(total_embed_dim, config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, field_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Embed and combine features from all fields.

        Args:
            field_inputs: Dict mapping field name to tensor of shape (batch, seq_len)

        Returns:
            Combined embeddings of shape (batch, seq_len, d_model)
        """
        # Look up embeddings for each field
        field_embeds = []
        for field_name in FEATURE_FIELDS:
            input_ids = field_inputs[f'input_ids_{field_name}']
            embed = self.embeddings[field_name](input_ids)
            field_embeds.append(embed)

        # Concatenate along embedding dimension
        concat = torch.cat(field_embeds, dim=-1)  # (batch, seq_len, total_embed_dim)

        # Project to model dimension
        projected = self.projection(concat)
        normalized = self.layer_norm(projected)
        return cast(torch.Tensor, self.dropout(normalized))


class StyleClassifier(nn.Module):  # type: ignore[misc]
    """Neural sequence classifier for multi-task style prediction (formality + gender + grammaticality).

    Architecture:
    1. Multi-field embedding: per-field embeddings concatenated and projected
    2. Positional encoding: learned or sinusoidal
    3. Transformer encoder: multi-layer self-attention
    4. Pooling: CLS token embedding for sentence representation
    5. Classification heads: Separate MLP heads for formality, gender, and grammaticality
    """

    def __init__(self, config: ModelConfig):
        """Initialize classifier with given configuration.

        Args:
            config: ModelConfig instance with hyperparameters
        """
        super().__init__()
        self.config = config

        # Multi-field embedding
        self.embedding = MultiFieldEmbedding(config)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.d_model,
            config.max_seq_len,
            config.dropout,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_layers)

        # Formality classification head
        self.formality_classifier = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_formality_classes),
        )

        # Gender classification head
        self.gender_classifier = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_gender_classes),
        )

        # Grammaticality classification head (binary: grammatic vs agrammatic)
        self.grammaticality_classifier = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_grammaticality_classes),
        )

    def _get_pooled_output(
        self,
        field_inputs: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get pooled sentence representation.

        Args:
            field_inputs: Dict with 'input_ids_<field>' tensors of shape (batch, seq_len)
            attention_mask: Binary mask of shape (batch, seq_len), 1 for real tokens

        Returns:
            Pooled representation of shape (batch, d_model)
        """
        # Embed tokens
        x = self.embedding(field_inputs)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)

        # Create attention mask for transformer (True = ignore)
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        # Encode
        x = cast(torch.Tensor, self.encoder(x, src_key_padding_mask=src_key_padding_mask))

        # Pooling
        if self.config.pooling == "cls":
            pooled = x[:, 0, :]
        elif self.config.pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = x.mean(dim=1)
        elif self.config.pooling == "max":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                x = x.masked_fill(mask == 0, float('-inf'))
            pooled = x.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling: {self.config.pooling}")

        return pooled

    def forward(
        self,
        field_inputs: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the model.

        Args:
            field_inputs: Dict with 'input_ids_<field>' tensors of shape (batch, seq_len)
            attention_mask: Binary mask of shape (batch, seq_len), 1 for real tokens

        Returns:
            Tuple of (formality_logits, gender_logits, grammaticality_logits),
            each of shape (batch, num_classes)
        """
        pooled = self._get_pooled_output(field_inputs, attention_mask)

        formality_logits = self.formality_classifier(pooled)
        gender_logits = self.gender_classifier(pooled)
        grammaticality_logits = self.grammaticality_classifier(pooled)

        return formality_logits, gender_logits, grammaticality_logits

    def predict(
        self,
        field_inputs: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get predicted class probabilities for all tasks."""
        formality_logits, gender_logits, grammaticality_logits = self.forward(field_inputs, attention_mask)
        return (
            F.softmax(formality_logits, dim=-1),
            F.softmax(gender_logits, dim=-1),
            F.softmax(grammaticality_logits, dim=-1),
        )

    def get_encoder_output(
        self,
        field_inputs: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get encoder output for all positions (for MLM).

        Returns:
            Tensor of shape (batch, seq_len, d_model)
        """
        x = self.embedding(field_inputs)
        x = self.pos_encoding(x)

        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        return cast(torch.Tensor, self.encoder(x, src_key_padding_mask=src_key_padding_mask))


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
    patience: int = 3  # Early stopping patience
    lr_scheduler_patience: int = 2
    lr_scheduler_factor: float = 0.5
    gradient_clip: float = 1.0
    use_class_weights: bool = True
    formality_loss_weight: float = 1.0  # Weight for formality loss in multi-task
    gender_loss_weight: float = 1.0  # Weight for gender loss in multi-task
    grammaticality_loss_weight: float = 1.0  # Weight for grammaticality loss in multi-task
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


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

    # Use 'pos' field as the primary for determining mask positions
    pos_ids = batch['input_ids_pos'].clone()

    # Create mask for tokens that can be masked
    maskable = batch['attention_mask'].bool()
    for special_id in special_token_ids:
        maskable &= (pos_ids != special_id)

    # Random mask
    probs = torch.rand_like(pos_ids.float())
    mask = maskable & (probs < mask_prob)

    # 80% MASK, 10% random, 10% unchanged
    mask_token_positions = mask & (probs < mask_prob * 0.8)
    random_token_positions = mask & (probs >= mask_prob * 0.8) & (probs < mask_prob * 0.9)

    # Clone all field IDs and apply masking, create labels for each field
    result = {'attention_mask': batch['attention_mask']}

    for field in FEATURE_FIELDS:
        field_ids = batch[f'input_ids_{field}'].clone()

        # Create labels for this field (ignore non-masked positions)
        mlm_labels = torch.full_like(field_ids, -100)
        mlm_labels[mask] = field_ids[mask]
        result[f'mlm_labels_{field}'] = mlm_labels

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
        self.config = config or TrainerConfig()
        self.mask_prob = mask_prob

        # Default to equal weights for all fields
        self.field_weights = field_weights or {field: 1.0 for field in FEATURE_FIELDS}

        self.device = torch.device(self.config.device)
        self.model.to(self.device)

        pad_id = dataset.tokenizer.pad_id
        max_seq_len = model.config.max_seq_len
        self.data_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, pad_id, max_seq_len),
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.optimizer = Adam(model.parameters(), lr=self.config.learning_rate)

        # Get vocabulary sizes for all fields
        self.vocab_sizes = dataset.tokenizer.get_vocab_sizes()

        self.history: Dict[str, Any] = {'mlm_loss': [], 'field_losses': {f: [] for f in FEATURE_FIELDS}}

    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Run one MLM pretraining epoch.

        Returns:
            Tuple of (average total loss, dict of average per-field losses)
        """
        self.model.train()
        total_loss = 0.0
        field_losses = {field: 0.0 for field in FEATURE_FIELDS}
        n_batches = 0

        for batch in self.data_loader:
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

            self.optimizer.zero_grad()

            # Get logits for all fields
            mlm_logits_dict = self.model.forward_mlm(field_inputs, attention_mask)

            # Compute weighted sum of losses across all fields
            batch_loss: torch.Tensor = torch.tensor(0.0, device=self.device)
            for f in FEATURE_FIELDS:
                logits = mlm_logits_dict[f]
                labels = mlm_batch[f'mlm_labels_{f}'].to(self.device)
                field_loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )
                weighted_loss = self.field_weights[f] * field_loss
                batch_loss = batch_loss + weighted_loss
                field_losses[f] += field_loss.item()

            # Average across fields
            loss = batch_loss / len(FEATURE_FIELDS)

            loss.backward()
            if self.config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

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
            mlm_loss, field_loss_dict = self.train_epoch()
            self.history['mlm_loss'].append(mlm_loss)
            for f, loss_val in field_loss_dict.items():
                self.history['field_losses'][f].append(loss_val)

            if verbose:
                print(f"Epoch {epoch+1}/{actual_epochs} - MLM Loss: {mlm_loss:.4f}")
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

        self.device = torch.device(self.config.device)
        self.model.to(self.device)

        # Data loaders with max_seq_len truncation
        pad_id = train_dataset.tokenizer.pad_id
        max_seq_len = model.config.max_seq_len
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, pad_id, max_seq_len),
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, pad_id, max_seq_len),
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
        encoder_params = list(model.embedding.parameters()) + list(model.encoder.parameters())
        classifier_params = (
            list(model.formality_classifier.parameters()) +
            list(model.gender_classifier.parameters()) +
            list(model.grammaticality_classifier.parameters())
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

    def train_epoch(self) -> Tuple[float, float, float, float]:
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

        for batch in self.train_loader:
            field_inputs, attention_mask, formality_labels, gender_labels, grammaticality_labels = self._batch_to_device(batch)

            self.optimizer.zero_grad()
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

            loss.backward()

            if self.config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

            self.optimizer.step()

            total_loss += loss.item()
            total_formality_loss += formality_loss.item()
            total_gender_loss += gender_loss.item()
            total_grammaticality_loss += grammaticality_loss.item()
            n_batches += 1

        return total_loss / n_batches, total_formality_loss / n_batches, total_gender_loss / n_batches, total_grammaticality_loss / n_batches

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate on validation set.

        Returns:
            Dictionary with losses, accuracies, and confusion matrices
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
        all_grammaticality_preds = []
        all_grammaticality_labels = []

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

        return {
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

    def train(self, verbose: bool = True) -> Dict[str, List[float]]:
        """Run full training loop."""
        for epoch in range(self.config.epochs):
            train_loss, train_formality_loss, train_gender_loss, train_grammaticality_loss = self.train_epoch()
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
                print(f"Epoch {epoch+1}/{self.config.epochs}")
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

        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            self.model.to(self.device)

        return self.history

    def print_confusion_matrices(self) -> None:
        """Print confusion matrices for all tasks."""
        eval_results = self.evaluate()

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


def save_model(
    model: StyleClassifier,
    tokenizer: Tokenizer,
    path: str,
    config: Optional[ModelConfig] = None,
) -> None:
    """Save trained model, tokenizer, and config."""
    import os
    os.makedirs(path, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(path, 'model.pt'))

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


def load_model(
    path: str,
    device: Optional[str] = None,
) -> Tuple[StyleClassifier, Tokenizer]:
    """Load trained model and tokenizer."""
    import os

    # Load config
    with open(os.path.join(path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    config = ModelConfig.from_dict(config_dict)

    # Load tokenizer
    tokenizer = Tokenizer.load(os.path.join(path, 'tokenizer.json'))

    # Load model
    model = StyleClassifier(config)
    model.load_state_dict(torch.load(os.path.join(path, 'model.pt'), map_location=device or 'cpu'))

    if device:
        model.to(device)

    model.eval()
    return model, tokenizer


def predict_style(
    sentence: str,
    model: StyleClassifier,
    tokenizer: Tokenizer,
    parser: Optional[JapaneseParser] = None,
    device: Optional[str] = None,
) -> Tuple[FormalityLevel, GenderLevel, bool, Dict[str, Dict[str, float]]]:
    """Predict formality, gender, and grammaticality for a Japanese sentence.

    Args:
        sentence: Japanese sentence text
        model: Trained StyleClassifier
        tokenizer: Tokenizer
        parser: JapaneseParser (defaults to SudachiJapaneseParser)
        device: Device to run inference on

    Returns:
        Tuple of (formality_label, gender_label, is_grammatic, probability_dicts)
        where probability_dicts has 'formality', 'gender', and 'grammaticality' keys
    """
    if parser is None:
        from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
        parser = SudachiJapaneseParser()

    # Convert to Kotogram
    kotogram = parser.japanese_to_kotogram(sentence)

    # Encode
    feature_ids = tokenizer.encode(kotogram, add_cls=True, add_to_vocab=False)

    # Create batch tensors
    field_inputs = {
        f'input_ids_{field}': torch.tensor([feature_ids[field]], dtype=torch.long)
        for field in FEATURE_FIELDS
    }
    attention_mask = torch.ones(1, len(feature_ids[FEATURE_FIELDS[0]]), dtype=torch.long)

    if device:
        field_inputs = {k: v.to(device) for k, v in field_inputs.items()}
        attention_mask = attention_mask.to(device)
        model.to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        formality_probs, gender_probs, grammaticality_probs = model.predict(field_inputs, attention_mask)
        formality_probs = formality_probs[0]
        gender_probs = gender_probs[0]
        grammaticality_probs = grammaticality_probs[0]

    formality_id = int(formality_probs.argmax().item())
    gender_id = int(gender_probs.argmax().item())
    grammaticality_id = int(grammaticality_probs.argmax().item())

    formality_label = FORMALITY_ID_TO_LABEL[formality_id]
    gender_label = GENDER_ID_TO_LABEL[gender_id]
    is_grammatic = grammaticality_id == 1  # 1 = grammatic, 0 = agrammatic

    prob_dicts = {
        'formality': {
            FORMALITY_ID_TO_LABEL[i].value: formality_probs[i].item()
            for i in range(NUM_FORMALITY_CLASSES)
        },
        'gender': {
            GENDER_ID_TO_LABEL[i].value: gender_probs[i].item()
            for i in range(NUM_GENDER_CLASSES)
        },
        'grammaticality': {
            'agrammatic': grammaticality_probs[0].item(),
            'grammatic': grammaticality_probs[1].item(),
        },
    }

    return formality_label, gender_label, is_grammatic, prob_dicts


if __name__ == "__main__":
    # Example usage
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

    args = parser.parse_args()

    # Build list of data files and their grammaticality labels
    # grammatic (1) = normal sentences, agrammatic (0) = generated bad grammar
    data_files = [args.data]
    grammaticality_labels = [1]  # jpn_sentences.tsv is grammatic
    if args.extra_data:
        data_files.append(args.extra_data)
        grammaticality_labels.append(1)  # unpragmatic_sentences.tsv is still grammatic (just unusual style)
    if args.agrammatic_data:
        data_files.append(args.agrammatic_data)
        grammaticality_labels.append(0)  # agrammatic sentences

    # Load data: if doing MLM pretraining, first load unlabeled data for pretraining,
    # then load labeled data for fine-tuning
    model: StyleClassifier  # Type annotation for both branches
    if args.pretrain_mlm:
        print("Loading unlabeled data for MLM pretraining...")
        tokenizer = Tokenizer()
        if len(data_files) > 1:
            unlabeled_dataset = StyleDataset.from_multiple_tsv(
                data_files,
                tokenizer,
                max_samples=args.max_samples,
                verbose=True,
                labeled=False,  # No labels needed for pretraining
                grammaticality_labels=grammaticality_labels,
            )
        else:
            unlabeled_dataset = StyleDataset.from_tsv(
                args.data,
                tokenizer,
                max_samples=args.max_samples,
                verbose=True,
                labeled=False,  # No labels needed for pretraining
            )
        # Note: tokenizer is frozen after from_tsv

        # Model config (vocab is now fixed)
        model_config = ModelConfig(
            vocab_sizes=tokenizer.get_vocab_sizes(),
            num_formality_classes=NUM_FORMALITY_CLASSES,
            num_gender_classes=NUM_GENDER_CLASSES,
            num_grammaticality_classes=NUM_GRAMMATICALITY_CLASSES,
            d_model=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )

        print("\nCreating model with MLM head...")
        model = StyleClassifierWithMLM(model_config)

        # MLM pretraining on unlabeled data
        print("\nStarting MLM pretraining on unlabeled data...")
        pretrain_config = TrainerConfig(
            epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        mlm_trainer = MLMTrainer(model, unlabeled_dataset, pretrain_config)
        mlm_trainer.train(epochs=args.pretrain_epochs)

        # Reset classifier heads for fine-tuning
        print("\nReinitializing classifier heads for fine-tuning...")
        model.reset_classifier()

        # Now load labeled dataset with the frozen tokenizer
        print("\nLoading labeled data for fine-tuning...")
        # For fine-tuning, we need to unfreeze temporarily to load new data
        tokenizer._frozen = False
        if len(data_files) > 1:
            labeled_dataset = StyleDataset.from_multiple_tsv(
                data_files,
                tokenizer,
                max_samples=args.max_samples,
                verbose=True,
                labeled=True,
                grammaticality_labels=grammaticality_labels,
            )
        else:
            labeled_dataset = StyleDataset.from_tsv(
                args.data,
                tokenizer,
                max_samples=args.max_samples,
                verbose=True,
                labeled=True,
            )
        train_data, val_data, test_data = labeled_dataset.split()
    else:
        print("Loading data...")
        tokenizer = Tokenizer()
        if len(data_files) > 1:
            dataset = StyleDataset.from_multiple_tsv(
                data_files,
                tokenizer,
                max_samples=args.max_samples,
                verbose=True,
                grammaticality_labels=grammaticality_labels,
            )
        else:
            dataset = StyleDataset.from_tsv(
                args.data,
                tokenizer,
                max_samples=args.max_samples,
                verbose=True,
            )
        train_data, val_data, test_data = dataset.split()

        # Model config
        model_config = ModelConfig(
            vocab_sizes=tokenizer.get_vocab_sizes(),
            num_formality_classes=NUM_FORMALITY_CLASSES,
            num_gender_classes=NUM_GENDER_CLASSES,
            num_grammaticality_classes=NUM_GRAMMATICALITY_CLASSES,
            d_model=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )

        print("\nCreating model...")
        model = StyleClassifier(model_config)

    print(f"\nSplit: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # Supervised training with differential learning rates
    print("\nStarting supervised training...")
    trainer_config = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        formality_loss_weight=args.formality_weight,
        gender_loss_weight=args.gender_weight,
        grammaticality_loss_weight=args.grammaticality_weight,
    )
    # Use smaller LR for encoder if pretrained
    encoder_lr_factor = args.encoder_lr_factor if args.pretrain_mlm else 1.0
    trainer = Trainer(
        model, train_data, val_data, trainer_config,
        encoder_lr_factor=encoder_lr_factor,
    )
    history = trainer.train()

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

    print(f"Test Accuracy: formality={formality_correct/total:.4f}, gender={gender_correct/total:.4f}, gram={grammaticality_correct/total:.4f}")

    # Save model
    print(f"\nSaving model to {args.output}...")
    save_model(model, tokenizer, args.output, model_config)
    print("Done!")
