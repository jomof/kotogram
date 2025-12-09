"""Supervised formality classifier for Japanese sentences using Kotogram representations.

This module provides a neural sequence classifier that predicts formality labels
for Japanese sentences based on their Kotogram representation. It uses a
pretrain-then-finetune approach with a small transformer encoder.

Architecture:
- Token Embedding: Multi-field embeddings for morphological features (pos, pos_detail1,
  pos_detail2, conjugated_type, conjugated_form, lemma) are concatenated and projected
  to d_model.
- Encoder: Small transformer encoder (2-4 layers) with multi-head self-attention.
- Pretraining: Multi-field Masked Language Modeling (MLM) that predicts all morphological
  features at masked positions, not just POS tags. This provides richer supervision.
- Fine-tuning: Sentence-level classification using [CLS] token representation.

Pipeline:
1. Load Japanese sentences from TSV corpus (unlabeled for pretraining)
2. Convert sentences to Kotogram strings using japanese_to_kotogram()
3. Extract token features using extract_token_features()
4. Build vocabulary for each categorical field
5. Pretrain encoder with multi-field MLM on unlabeled data
6. Reinitialize classifier head, then fine-tune with formality labels

Usage:
    from kotogram.formality_classifier import (
        FormalityDataset, Tokenizer, FormalityClassifier,
        FormalityClassifierWithMLM, MLMTrainer, Trainer, predict_formality
    )

    # Build vocabulary with unlabeled data
    tokenizer = Tokenizer()
    unlabeled = FormalityDataset.from_tsv("data/sentences.tsv", tokenizer, labeled=False)

    # Pretrain with multi-field MLM
    model = FormalityClassifierWithMLM(tokenizer.get_model_config())
    mlm_trainer = MLMTrainer(model, unlabeled)
    mlm_trainer.train(epochs=5)

    # Reset classifier and load labeled data
    model.reset_classifier()
    labeled = FormalityDataset.from_tsv("data/sentences.tsv", tokenizer, labeled=True)
    train_data, val_data, test_data = labeled.split()

    # Fine-tune for classification
    trainer = Trainer(model, train_data, val_data)
    trainer.train(epochs=10)

    # Inference
    label, probs = predict_formality("何かしてみましょう。", model, tokenizer)
"""

import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from kotogram.kotogram import split_kotogram
from kotogram.analysis import formality, FormalityLevel, extract_token_features


# Special token values for vocabulary
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
CLS_TOKEN = "<CLS>"
MASK_TOKEN = "<MASK>"  # For self-supervised pretraining

# Feature fields used for token embedding
FEATURE_FIELDS = ['pos', 'pos_detail1', 'pos_detail2', 'conjugated_type', 'conjugated_form', 'lemma']


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
        self._field_counters: Dict[str, Counter] = {}
        for field in FEATURE_FIELDS:
            self.field_vocabs[field] = {
                PAD_TOKEN: 0,
                UNK_TOKEN: 1,
                CLS_TOKEN: 2,
                MASK_TOKEN: 3,
            }
            self._field_counters[field] = Counter()

        self._frozen = False
        self._lemma_counts: Counter = Counter()

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
        result = {field: [] for field in FEATURE_FIELDS}

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

    def finalize_vocab(self):
        """Finalize vocabulary, pruning rare lemmas.

        Should be called after processing all training data but before freezing.
        """
        # Prune lemma vocabulary to most frequent items
        vocab = self.field_vocabs['lemma']
        frequent_lemmas = self._lemma_counts.most_common(self.max_lemma_vocab)

        for lemma, count in frequent_lemmas:
            if count >= self.lemma_min_freq and lemma not in vocab:
                vocab[lemma] = len(vocab)

    def freeze(self):
        """Freeze vocabulary - new values will map to UNK."""
        self.finalize_vocab()
        self._frozen = True

    def unfreeze(self):
        """Unfreeze vocabulary."""
        self._frozen = False

    def get_model_config(self, **kwargs) -> 'ModelConfig':
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

    def save(self, path: str):
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
    """Single data sample with per-field feature IDs and label."""
    feature_ids: Dict[str, List[int]]  # field -> list of token IDs
    label: int
    original_sentence: str = ""
    kotogram: str = ""

    @property
    def seq_len(self) -> int:
        """Get sequence length (same for all fields)."""
        first_field = next(iter(self.feature_ids.keys()))
        return len(self.feature_ids[first_field])


class FormalityDataset(Dataset):
    """PyTorch Dataset for formality classification using feature-based tokenization.

    Each sample contains per-field feature IDs rather than a single token ID sequence.
    This allows the model to learn from individual morphological features.
    """

    # Map FormalityLevel enum to integer class IDs
    LABEL_TO_ID = {
        FormalityLevel.VERY_FORMAL: 0,
        FormalityLevel.FORMAL: 1,
        FormalityLevel.NEUTRAL: 2,
        FormalityLevel.CASUAL: 3,
        FormalityLevel.VERY_CASUAL: 4,
        FormalityLevel.UNPRAGMATIC_FORMALITY: 5,
    }
    ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
    NUM_CLASSES = len(LABEL_TO_ID)

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
        parser=None,
        max_samples: Optional[int] = None,
        verbose: bool = True,
        labeled: bool = True,
    ) -> 'FormalityDataset':
        """Load dataset from TSV file of Japanese sentences.

        Args:
            tsv_path: Path to TSV file with Japanese sentences
            tokenizer: Tokenizer to build vocabulary
            parser: JapaneseParser instance (defaults to SudachiJapaneseParser)
            max_samples: Optional limit on number of samples
            verbose: If True, print progress
            labeled: If True, compute formality labels. If False, use dummy labels
                    (for pretraining on unlabeled data).

        Returns:
            FormalityDataset with encoded samples
        """
        if parser is None:
            from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
            parser = SudachiJapaneseParser()

        samples = []
        label_counts = Counter()

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
                    kotogram = parser.japanese_to_kotogram(sentence)

                    # Get formality label (or dummy for pretraining)
                    if labeled:
                        label_enum = formality(kotogram)
                        label_id = cls.LABEL_TO_ID[label_enum]
                    else:
                        label_enum = FormalityLevel.NEUTRAL
                        label_id = cls.LABEL_TO_ID[label_enum]

                    # Encode to feature IDs (builds vocabulary)
                    feature_ids = tokenizer.encode(kotogram, add_cls=True, add_to_vocab=True)

                    sample = Sample(
                        feature_ids=feature_ids,
                        label=label_id,
                        original_sentence=sentence,
                        kotogram=kotogram,
                    )
                    samples.append(sample)
                    label_counts[label_enum] += 1
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
                print("Label distribution:")
                for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {label.value}: {count} ({100*count/len(samples):.1f}%)")

        # Freeze vocabulary after building
        tokenizer.freeze()

        return cls(samples, tokenizer)

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple['FormalityDataset', 'FormalityDataset', 'FormalityDataset']:
        """Split dataset into train/validation/test sets."""
        random.seed(seed)
        indices = list(range(len(self.samples)))
        random.shuffle(indices)

        n_train = int(len(indices) * train_ratio)
        n_val = int(len(indices) * val_ratio)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        train_samples = [self.samples[i] for i in train_indices]
        val_samples = [self.samples[i] for i in val_indices]
        test_samples = [self.samples[i] for i in test_indices]

        return (
            FormalityDataset(train_samples, self.tokenizer),
            FormalityDataset(val_samples, self.tokenizer),
            FormalityDataset(test_samples, self.tokenizer),
        )

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights for imbalanced data."""
        counts = Counter(s.label for s in self.samples)
        total = len(self.samples)
        weights = torch.zeros(self.NUM_CLASSES)

        for label_id, count in counts.items():
            weights[label_id] = total / (self.NUM_CLASSES * count) if count > 0 else 0.0

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
        Dictionary with per-field 'input_ids_<field>', 'attention_mask', 'labels' tensors
    """
    batch_max_len = max(s.seq_len for s in batch)
    # Apply truncation if max_seq_len is specified
    max_len = min(batch_max_len, max_seq_len) if max_seq_len else batch_max_len

    # Initialize per-field lists
    field_ids = {field: [] for field in FEATURE_FIELDS}
    attention_mask = []
    labels = []

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
        labels.append(sample.label)

    result = {
        f'input_ids_{field}': torch.tensor(field_ids[field], dtype=torch.long)
        for field in FEATURE_FIELDS
    }
    result['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
    result['labels'] = torch.tensor(labels, dtype=torch.long)

    return result


class PositionalEncoding(nn.Module):
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
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


@dataclass
class ModelConfig:
    """Configuration for FormalityClassifier model.

    This config supports multi-field embeddings where each morphological feature
    (pos, pos_detail1, conjugated_type, conjugated_form, lemma) has its own
    embedding table.
    """
    vocab_sizes: Dict[str, int]  # Field name -> vocabulary size
    num_classes: int = 6
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
            'num_classes': self.num_classes,
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


class MultiFieldEmbedding(nn.Module):
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
        return self.dropout(normalized)


class FormalityClassifier(nn.Module):
    """Neural sequence classifier using multi-field morphological features.

    Architecture:
    1. Multi-field embedding: per-field embeddings concatenated and projected
    2. Positional encoding: learned or sinusoidal
    3. Transformer encoder: multi-layer self-attention
    4. Pooling: CLS token embedding for sentence representation
    5. Classification head: MLP to formality classes
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

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(
        self,
        field_inputs: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            field_inputs: Dict with 'input_ids_<field>' tensors of shape (batch, seq_len)
            attention_mask: Binary mask of shape (batch, seq_len), 1 for real tokens

        Returns:
            Logits of shape (batch, num_classes)
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
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

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

        # Classification
        logits = self.classifier(pooled)
        return logits

    def predict(
        self,
        field_inputs: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get predicted class probabilities."""
        logits = self.forward(field_inputs, attention_mask)
        return F.softmax(logits, dim=-1)

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

        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


class MLMHead(nn.Module):
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


class FormalityClassifierWithMLM(FormalityClassifier):
    """Feature-based formality classifier with MLM pretraining head.

    This model can be:
    1. Pre-trained with masked token prediction (self-supervised)
    2. Fine-tuned for formality classification (supervised)
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
        return self.mlm_head(encoder_output)

    def reset_classifier(self):
        """Reinitialize the classifier head weights.

        Call this after MLM pretraining and before supervised fine-tuning
        to start the classification head from a fresh state while keeping
        the pretrained encoder weights.
        """
        for module in self.classifier.modules():
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
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def create_mlm_batch(
    batch: Dict[str, torch.Tensor],
    mask_prob: float = 0.15,
    mask_token_id: int = 3,
    vocab_sizes: Dict[str, int] = None,
    special_token_ids: List[int] = None,
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
            num_random = random_token_positions.sum().item()
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
        model: FormalityClassifierWithMLM,
        dataset: FormalityDataset,
        config: TrainerConfig = None,
        mask_prob: float = 0.15,
        field_weights: Dict[str, float] = None,
    ):
        """Initialize MLM trainer.

        Args:
            model: FormalityClassifierWithMLM model
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

        self.history = {'mlm_loss': [], 'field_losses': {field: [] for field in FEATURE_FIELDS}}

    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Run one MLM pretraining epoch.

        Returns:
            Tuple of (average total loss, dict of average per-field losses)
        """
        self.model.train()
        total_loss = 0
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
            batch_loss = 0.0
            for field in FEATURE_FIELDS:
                logits = mlm_logits_dict[field]
                labels = mlm_batch[f'mlm_labels_{field}'].to(self.device)
                field_loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )
                weighted_loss = self.field_weights[field] * field_loss
                batch_loss = batch_loss + weighted_loss
                field_losses[field] += field_loss.item()

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

    def train(self, epochs: int = None, verbose: bool = True) -> Dict[str, List[float]]:
        """Run MLM pretraining.

        Args:
            epochs: Number of epochs (defaults to config.epochs)
            verbose: If True, print progress

        Returns:
            Training history with 'mlm_loss' and per-field losses
        """
        epochs = epochs or self.config.epochs

        for epoch in range(epochs):
            mlm_loss, field_losses = self.train_epoch()
            self.history['mlm_loss'].append(mlm_loss)
            for field, loss in field_losses.items():
                self.history['field_losses'][field].append(loss)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - MLM Loss: {mlm_loss:.4f}")
                field_str = ", ".join(f"{f}={l:.3f}" for f, l in field_losses.items())
                print(f"  Field losses: {field_str}")

        return self.history


class Trainer:
    """Training loop for feature-based formality classifier with differential learning rates."""

    def __init__(
        self,
        model: FormalityClassifier,
        train_dataset: FormalityDataset,
        val_dataset: FormalityDataset,
        config: TrainerConfig = None,
        encoder_lr_factor: float = 0.1,
    ):
        """Initialize trainer.

        Args:
            model: FormalityClassifier model
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

        # Loss function with optional class weights
        if self.config.use_class_weights:
            weights = train_dataset.get_class_weights().to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer with differential learning rates
        encoder_params = list(model.embedding.parameters()) + list(model.encoder.parameters())
        classifier_params = list(model.classifier.parameters())

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
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
        }

    def _batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Move batch tensors to device and split into inputs/mask/labels."""
        field_inputs = {
            k: v.to(self.device) for k, v in batch.items()
            if k.startswith('input_ids_')
        }
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        return field_inputs, attention_mask, labels

    def train_epoch(self) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0
        n_batches = 0

        for batch in self.train_loader:
            field_inputs, attention_mask, labels = self._batch_to_device(batch)

            self.optimizer.zero_grad()
            logits = self.model(field_inputs, attention_mask)
            loss = self.criterion(logits, labels)
            loss.backward()

            if self.config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float, List[List[int]]]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        n_batches = 0
        all_preds = []
        all_labels = []

        for batch in self.val_loader:
            field_inputs, attention_mask, labels = self._batch_to_device(batch)

            logits = self.model(field_inputs, attention_mask)
            loss = self.criterion(logits, labels)

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)

        # Build confusion matrix
        num_classes = FormalityDataset.NUM_CLASSES
        confusion = [[0] * num_classes for _ in range(num_classes)]
        for pred, label in zip(all_preds, all_labels):
            confusion[label][pred] += 1

        return avg_loss, accuracy, confusion

    def train(self, verbose: bool = True) -> Dict[str, List[float]]:
        """Run full training loop."""
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch()
            val_loss, val_acc, confusion = self.evaluate()

            self.scheduler.step(val_loss)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)

            if verbose:
                print(f"Epoch {epoch+1}/{self.config.epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                enc_lr = self.optimizer.param_groups[0]['lr']
                cls_lr = self.optimizer.param_groups[1]['lr']
                print(f"  LR: encoder={enc_lr:.2e}, classifier={cls_lr:.2e}")

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        # Restore best model
        if hasattr(self, 'best_state'):
            self.model.load_state_dict(self.best_state)
            self.model.to(self.device)

        return self.history

    def print_confusion_matrix(self):
        """Print confusion matrix with class labels."""
        _, _, confusion = self.evaluate()

        labels = [FormalityDataset.ID_TO_LABEL[i].value for i in range(FormalityDataset.NUM_CLASSES)]

        print("\nConfusion Matrix:")
        header = "True\\Pred".ljust(25) + " ".join(l[:8].ljust(10) for l in labels)
        print(header)
        print("-" * len(header))

        for i, row in enumerate(confusion):
            row_label = labels[i].ljust(25)
            row_values = " ".join(str(v).ljust(10) for v in row)
            print(f"{row_label}{row_values}")


def save_model(
    model: FormalityClassifier,
    tokenizer: Tokenizer,
    path: str,
    config: Optional[ModelConfig] = None,
):
    """Save feature-based trained model, tokenizer, and config."""
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

    # Save label mapping
    label_map = {k.value: v for k, v in FormalityDataset.LABEL_TO_ID.items()}
    with open(os.path.join(path, 'labels.json'), 'w') as f:
        json.dump(label_map, f, indent=2)

    # Mark as feature-based model
    with open(os.path.join(path, 'model_type.txt'), 'w') as f:
        f.write('feature')


def load_model(
    path: str,
    device: str = None,
) -> Tuple[FormalityClassifier, Tokenizer]:
    """Load feature-based trained model and tokenizer."""
    import os

    # Load config
    with open(os.path.join(path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    config = ModelConfig.from_dict(config_dict)

    # Load tokenizer
    tokenizer = Tokenizer.load(os.path.join(path, 'tokenizer.json'))

    # Load model
    model = FormalityClassifier(config)
    model.load_state_dict(torch.load(os.path.join(path, 'model.pt'), map_location=device or 'cpu'))

    if device:
        model.to(device)

    model.eval()
    return model, tokenizer


def predict_formality(
    sentence: str,
    model: FormalityClassifier,
    tokenizer: Tokenizer,
    parser=None,
    device: str = None,
) -> Tuple[FormalityLevel, Dict[str, float]]:
    """Predict formality level for a Japanese sentence using feature-based model.

    Args:
        sentence: Japanese sentence text
        model: Trained FormalityClassifier
        tokenizer: Tokenizer
        parser: JapaneseParser (defaults to SudachiJapaneseParser)
        device: Device to run inference on

    Returns:
        Tuple of (predicted_label, class_probabilities)
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
        probs = model.predict(field_inputs, attention_mask)[0]

    predicted_id = probs.argmax().item()
    predicted_label = FormalityDataset.ID_TO_LABEL[predicted_id]

    prob_dict = {
        FormalityDataset.ID_TO_LABEL[i].value: probs[i].item()
        for i in range(FormalityDataset.NUM_CLASSES)
    }

    return predicted_label, prob_dict


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Train formality classifier")
    parser.add_argument("--data", type=str, default="data/jpn_sentences.tsv",
                        help="Path to TSV file with Japanese sentences")
    parser.add_argument("--output", type=str, default="models/formality",
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

    args = parser.parse_args()

    # Load data: if doing MLM pretraining, first load unlabeled data for pretraining,
    # then load labeled data for fine-tuning
    if args.pretrain_mlm:
        print("Loading unlabeled data for MLM pretraining...")
        tokenizer = Tokenizer()
        unlabeled_dataset = FormalityDataset.from_tsv(
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
            num_classes=FormalityDataset.NUM_CLASSES,
            d_model=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )

        print("\nCreating model with MLM head...")
        model = FormalityClassifierWithMLM(model_config)

        # MLM pretraining on unlabeled data
        print("\nStarting MLM pretraining on unlabeled data...")
        pretrain_config = TrainerConfig(
            epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        mlm_trainer = MLMTrainer(model, unlabeled_dataset, pretrain_config)
        mlm_trainer.train(epochs=args.pretrain_epochs)

        # Reset classifier head for fine-tuning
        print("\nReinitializing classifier head for fine-tuning...")
        model.reset_classifier()

        # Now load labeled dataset with the frozen tokenizer
        print("\nLoading labeled data for fine-tuning...")
        labeled_dataset = FormalityDataset.from_tsv(
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
        dataset = FormalityDataset.from_tsv(
            args.data,
            tokenizer,
            max_samples=args.max_samples,
            verbose=True,
        )
        train_data, val_data, test_data = dataset.split()

        # Model config
        model_config = ModelConfig(
            vocab_sizes=tokenizer.get_vocab_sizes(),
            num_classes=FormalityDataset.NUM_CLASSES,
            d_model=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )

        print("\nCreating model...")
        model = FormalityClassifier(model_config)

    print(f"\nSplit: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # Supervised training with differential learning rates
    print("\nStarting supervised training...")
    trainer_config = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    # Use smaller LR for encoder if pretrained
    encoder_lr_factor = args.encoder_lr_factor if args.pretrain_mlm else 1.0
    trainer = Trainer(
        model, train_data, val_data, trainer_config,
        encoder_lr_factor=encoder_lr_factor,
    )
    history = trainer.train()

    # Print final metrics
    trainer.print_confusion_matrix()

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id, model_config.max_seq_len),
    )

    model.eval()
    device = torch.device(trainer_config.device)
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            field_inputs = {
                k: v.to(device) for k, v in batch.items()
                if k.startswith('input_ids_')
            }
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(field_inputs, attention_mask)
            preds = logits.argmax(dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {correct/total:.4f}")

    # Save model
    print(f"\nSaving model to {args.output}...")
    save_model(model, tokenizer, args.output, model_config)
    print("Done!")
