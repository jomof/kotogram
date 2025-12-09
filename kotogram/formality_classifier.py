"""Supervised formality classifier for Japanese sentences using Kotogram representations.

This module provides a neural sequence classifier that predicts formality labels
for Japanese sentences based on their Kotogram representation.

Pipeline:
1. Load Japanese sentences from TSV corpus
2. Convert sentences to Kotogram strings using japanese_to_kotogram()
3. Derive formality labels using formality(kotogram)
4. Tokenize Kotograms into sequences of token IDs
5. Train a neural model to predict formality from token sequences

Usage:
    from kotogram.formality_classifier import (
        FormalityDataset, KotogramTokenizer, FormalityClassifier, Trainer
    )

    # Load and preprocess data
    tokenizer = KotogramTokenizer()
    dataset = FormalityDataset.from_tsv("data/jpn_sentences.tsv", tokenizer)

    # Train model
    model = FormalityClassifier(vocab_size=tokenizer.vocab_size, num_classes=6)
    trainer = Trainer(model, dataset)
    trainer.train(epochs=10)
"""

import csv
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from kotogram.kotogram import split_kotogram
from kotogram.analysis import formality, FormalityLevel


# Special token IDs
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
CLS_TOKEN = "<CLS>"
MASK_TOKEN = "<MASK>"  # For self-supervised pretraining


def strip_surface_form(token: str) -> str:
    """Strip surface form (kanji/kana) from a Kotogram token, keeping only POS tags.

    Converts: ⌈ˢ食べᵖv:e-ichidan-ba:conjunctive⌉ -> ⌈ᵖv:e-ichidan-ba:conjunctive⌉

    This reduces vocabulary size by collapsing tokens that differ only by
    surface form into the same grammatical pattern.

    Args:
        token: A single Kotogram token string

    Returns:
        Token with surface form removed
    """
    import re
    # Remove surface form (ˢ...ᵖ) but keep the ᵖ marker
    return re.sub(r'ˢ[^ᵖ]*ᵖ', 'ᵖ', token)


class KotogramTokenizer:
    """Tokenizer that splits Kotogram strings into tokens and manages vocabulary.

    Each ⌈...⌉ block in a Kotogram string is treated as one token. The tokenizer
    builds a vocabulary mapping from token strings to integer IDs.

    Attributes:
        token_to_id: Mapping from token string to integer ID
        id_to_token: Mapping from integer ID to token string
        vocab_size: Total vocabulary size including special tokens
        strip_surface: If True, remove surface forms (kanji/kana) from tokens
    """

    def __init__(self, strip_surface: bool = False):
        """Initialize tokenizer with special tokens.

        Args:
            strip_surface: If True, strip surface forms from tokens to reduce
                          vocabulary size. Tokens will only contain POS/grammar info.
        """
        self.token_to_id: Dict[str, int] = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
            CLS_TOKEN: 2,
            MASK_TOKEN: 3,
        }
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.token_to_id.items()}
        self._next_id = len(self.token_to_id)
        self._frozen = False
        self.strip_surface = strip_surface

    @property
    def vocab_size(self) -> int:
        """Return total vocabulary size."""
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def cls_id(self) -> int:
        return self.token_to_id[CLS_TOKEN]

    @property
    def mask_id(self) -> int:
        return self.token_to_id[MASK_TOKEN]

    def tokenize(self, kotogram: str) -> List[str]:
        """Split a Kotogram string into token strings.

        Args:
            kotogram: Kotogram string with ⌈...⌉ token boundaries

        Returns:
            List of token strings (each ⌈...⌉ block), optionally with
            surface forms stripped if strip_surface=True
        """
        tokens = split_kotogram(kotogram)
        if self.strip_surface:
            tokens = [strip_surface_form(t) for t in tokens]
        return tokens

    def add_token(self, token: str) -> int:
        """Add a token to vocabulary and return its ID.

        If vocabulary is frozen, returns UNK ID for new tokens.

        Args:
            token: Token string to add

        Returns:
            Integer ID for the token
        """
        if token in self.token_to_id:
            return self.token_to_id[token]

        if self._frozen:
            return self.unk_id

        token_id = self._next_id
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        self._next_id += 1
        return token_id

    def encode(self, kotogram: str, add_cls: bool = True, add_to_vocab: bool = True) -> List[int]:
        """Convert Kotogram string to sequence of token IDs.

        Args:
            kotogram: Kotogram string to encode
            add_cls: If True, prepend CLS token ID
            add_to_vocab: If True, add new tokens to vocabulary

        Returns:
            List of integer token IDs
        """
        tokens = self.tokenize(kotogram)
        ids = []

        if add_cls:
            ids.append(self.cls_id)

        for token in tokens:
            if add_to_vocab:
                ids.append(self.add_token(token))
            else:
                ids.append(self.token_to_id.get(token, self.unk_id))

        return ids

    def decode(self, ids: List[int]) -> str:
        """Convert sequence of token IDs back to Kotogram string.

        Args:
            ids: List of integer token IDs

        Returns:
            Kotogram string (concatenated tokens, special tokens excluded)
        """
        tokens = []
        for id_ in ids:
            if id_ in (self.pad_id, self.cls_id, self.mask_id):
                continue
            token = self.id_to_token.get(id_, UNK_TOKEN)
            if token != UNK_TOKEN:
                tokens.append(token)
        return ''.join(tokens)

    def freeze(self):
        """Freeze vocabulary - new tokens will map to UNK."""
        self._frozen = True

    def unfreeze(self):
        """Unfreeze vocabulary - allow adding new tokens."""
        self._frozen = False

    def save(self, path: str):
        """Save tokenizer vocabulary to JSON file.

        Args:
            path: File path to save vocabulary
        """
        data = {
            'token_to_id': self.token_to_id,
            'frozen': self._frozen,
            'strip_surface': self.strip_surface,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'KotogramTokenizer':
        """Load tokenizer vocabulary from JSON file.

        Args:
            path: File path to load vocabulary from

        Returns:
            KotogramTokenizer instance with loaded vocabulary
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tokenizer = cls(strip_surface=data.get('strip_surface', False))
        tokenizer.token_to_id = data['token_to_id']
        tokenizer.id_to_token = {int(v): k for k, v in tokenizer.token_to_id.items()}
        tokenizer._next_id = max(tokenizer.token_to_id.values()) + 1
        tokenizer._frozen = data.get('frozen', False)
        return tokenizer


@dataclass
class FormalitySample:
    """Single data sample with encoded sequence and label."""
    token_ids: List[int]
    label: int
    original_sentence: str = ""
    kotogram: str = ""


class FormalityDataset(Dataset):
    """PyTorch Dataset for formality classification.

    Loads Japanese sentences, converts to Kotogram, extracts formality labels,
    and provides batched access to encoded sequences.
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

    def __init__(self, samples: List[FormalitySample], tokenizer: KotogramTokenizer):
        """Initialize dataset with preprocessed samples.

        Args:
            samples: List of FormalitySample objects
            tokenizer: KotogramTokenizer used to encode samples
        """
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> FormalitySample:
        return self.samples[idx]

    @classmethod
    def from_tsv(
        cls,
        tsv_path: str,
        tokenizer: KotogramTokenizer,
        parser=None,
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> 'FormalityDataset':
        """Load dataset from TSV file of Japanese sentences.

        Expects TSV format: id<tab>lang<tab>sentence

        Args:
            tsv_path: Path to TSV file with Japanese sentences
            tokenizer: KotogramTokenizer to build vocabulary
            parser: JapaneseParser instance (defaults to SudachiJapaneseParser)
            max_samples: Optional limit on number of samples
            verbose: If True, print progress

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

                    # Get formality label
                    label_enum = formality(kotogram)
                    label_id = cls.LABEL_TO_ID[label_enum]

                    # Encode to token IDs (builds vocabulary)
                    token_ids = tokenizer.encode(kotogram, add_cls=True, add_to_vocab=True)

                    sample = FormalitySample(
                        token_ids=token_ids,
                        label=label_id,
                        original_sentence=sentence,
                        kotogram=kotogram,
                    )
                    samples.append(sample)
                    label_counts[label_enum] += 1
                    total += 1

                    if verbose and total % 10000 == 0:
                        print(f"Processed {total} sentences, vocab size: {tokenizer.vocab_size}")

                    if max_samples and total >= max_samples:
                        break

                except Exception as e:
                    if verbose:
                        print(f"Error processing sentence {sentence_id}: {e}")
                    continue

        if verbose:
            print(f"\nDataset loaded: {len(samples)} samples")
            print(f"Vocabulary size: {tokenizer.vocab_size}")
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
        """Split dataset into train/validation/test sets.

        Args:
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
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
        """Compute inverse frequency class weights for imbalanced data.

        Returns:
            Tensor of shape (num_classes,) with class weights
        """
        counts = Counter(s.label for s in self.samples)
        total = len(self.samples)
        weights = torch.zeros(self.NUM_CLASSES)

        for label_id, count in counts.items():
            # Inverse frequency weighting
            weights[label_id] = total / (self.NUM_CLASSES * count) if count > 0 else 0.0

        return weights


def collate_fn(batch: List[FormalitySample], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """Collate samples into padded batches.

    Args:
        batch: List of FormalitySample objects
        pad_id: Padding token ID

    Returns:
        Dictionary with 'input_ids', 'attention_mask', 'labels' tensors
    """
    max_len = max(len(s.token_ids) for s in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for sample in batch:
        seq_len = len(sample.token_ids)
        padding_len = max_len - seq_len

        input_ids.append(sample.token_ids + [pad_id] * padding_len)
        attention_mask.append([1] * seq_len + [0] * padding_len)
        labels.append(sample.label)

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
    }


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
    """Configuration for FormalityClassifier model."""
    vocab_size: int
    num_classes: int = 6
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    max_seq_len: int = 512
    encoder_type: str = "transformer"  # "transformer" or "bilstm"
    pooling: str = "cls"  # "cls", "mean", or "max"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'vocab_size': self.vocab_size,
            'num_classes': self.num_classes,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'max_seq_len': self.max_seq_len,
            'encoder_type': self.encoder_type,
            'pooling': self.pooling,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModelConfig':
        return cls(**d)


class FormalityClassifier(nn.Module):
    """Neural sequence classifier for Kotogram formality prediction.

    Architecture:
    1. Embedding layer: token IDs -> dense vectors
    2. Positional encoding (for Transformer) or implicit order (for BiLSTM)
    3. Encoder: Transformer or BiLSTM stack
    4. Pooling: CLS token, mean, or max pooling
    5. Classification head: FC layers -> softmax over formality classes
    """

    def __init__(self, config: ModelConfig):
        """Initialize classifier with given configuration.

        Args:
            config: ModelConfig instance with hyperparameters
        """
        super().__init__()
        self.config = config

        # Embedding layer
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
            padding_idx=0,  # PAD token
        )

        # Encoder
        if config.encoder_type == "transformer":
            self.pos_encoding = PositionalEncoding(
                config.embedding_dim,
                config.max_seq_len,
                config.dropout,
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim,
                dropout=config.dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, config.num_layers)
            encoder_output_dim = config.embedding_dim

        elif config.encoder_type == "bilstm":
            self.encoder = nn.LSTM(
                config.embedding_dim,
                config.hidden_dim // 2,  # Bidirectional doubles output
                config.num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=config.dropout if config.num_layers > 1 else 0,
            )
            encoder_output_dim = config.hidden_dim
            self.pos_encoding = None
        else:
            raise ValueError(f"Unknown encoder_type: {config.encoder_type}")

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(encoder_output_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Binary mask of shape (batch, seq_len), 1 for real tokens

        Returns:
            Logits of shape (batch, num_classes)
        """
        # Embed tokens
        x = self.embedding(input_ids)  # (batch, seq_len, embed_dim)

        if self.config.encoder_type == "transformer":
            x = self.pos_encoding(x)

            # Create attention mask for transformer (True = ignore)
            if attention_mask is not None:
                # Transformer expects True for positions to mask
                src_key_padding_mask = attention_mask == 0
            else:
                src_key_padding_mask = None

            x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        elif self.config.encoder_type == "bilstm":
            if attention_mask is not None:
                # Pack sequences for efficient LSTM processing
                lengths = attention_mask.sum(dim=1).cpu()
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths, batch_first=True, enforce_sorted=False
                )
                packed_out, _ = self.encoder(packed)
                x, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            else:
                x, _ = self.encoder(x)

        # Pooling
        if self.config.pooling == "cls":
            # Use CLS token (first position)
            pooled = x[:, 0, :]
        elif self.config.pooling == "mean":
            # Mean pooling over non-padded positions
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = x.mean(dim=1)
        elif self.config.pooling == "max":
            # Max pooling over non-padded positions
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                x = x.masked_fill(mask == 0, float('-inf'))
            pooled = x.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling: {self.config.pooling}")

        # Classification
        logits = self.classifier(pooled)
        return logits

    def predict(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get predicted class probabilities.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Binary mask of shape (batch, seq_len)

        Returns:
            Probabilities of shape (batch, num_classes)
        """
        logits = self.forward(input_ids, attention_mask)
        return F.softmax(logits, dim=-1)


class MaskedLMHead(nn.Module):
    """Masked language modeling head for self-supervised pretraining."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dense = nn.Linear(config.embedding_dim if config.encoder_type == "transformer" else config.hidden_dim, config.embedding_dim)
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        self.decoder = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits.

        Args:
            hidden_states: Encoder output of shape (batch, seq_len, hidden_dim)

        Returns:
            Vocabulary logits of shape (batch, seq_len, vocab_size)
        """
        x = self.dense(hidden_states)
        x = F.gelu(x)
        x = self.layer_norm(x)
        logits = self.decoder(x)
        return logits


class FormalityClassifierWithMLM(FormalityClassifier):
    """Formality classifier with optional masked language modeling head.

    This model can be:
    1. Pre-trained with masked token prediction (self-supervised)
    2. Fine-tuned for formality classification (supervised)
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.mlm_head = MaskedLMHead(config)

    def forward_mlm(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for masked language modeling.

        Args:
            input_ids: Token IDs with some positions masked
            attention_mask: Binary mask for padding

        Returns:
            Vocabulary logits of shape (batch, seq_len, vocab_size)
        """
        # Embed and encode
        x = self.embedding(input_ids)

        if self.config.encoder_type == "transformer":
            x = self.pos_encoding(x)
            if attention_mask is not None:
                src_key_padding_mask = attention_mask == 0
            else:
                src_key_padding_mask = None
            x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        elif self.config.encoder_type == "bilstm":
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1).cpu()
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths, batch_first=True, enforce_sorted=False
                )
                packed_out, _ = self.encoder(packed)
                x, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            else:
                x, _ = self.encoder(x)

        # MLM prediction
        mlm_logits = self.mlm_head(x)
        return mlm_logits


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


class Trainer:
    """Training loop for formality classifier with validation metrics."""

    def __init__(
        self,
        model: FormalityClassifier,
        train_dataset: FormalityDataset,
        val_dataset: FormalityDataset,
        config: TrainerConfig = None,
    ):
        """Initialize trainer.

        Args:
            model: FormalityClassifier model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: TrainerConfig with hyperparameters
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or TrainerConfig()

        self.device = torch.device(self.config.device)
        self.model.to(self.device)

        # Data loaders
        pad_id = train_dataset.tokenizer.pad_id
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, pad_id),
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, pad_id),
        )

        # Loss function with optional class weights
        if self.config.use_class_weights:
            weights = train_dataset.get_class_weights().to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer and scheduler
        self.optimizer = Adam(model.parameters(), lr=self.config.learning_rate)
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

    def train_epoch(self) -> float:
        """Run one training epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        n_batches = 0

        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            loss.backward()

            if self.config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float, Dict[str, List[int]]]:
        """Evaluate on validation set.

        Returns:
            Tuple of (val_loss, accuracy, confusion_matrix)
        """
        self.model.eval()
        total_loss = 0
        n_batches = 0
        all_preds = []
        all_labels = []

        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            logits = self.model(input_ids, attention_mask)
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
        """Run full training loop.

        Args:
            verbose: If True, print progress

        Returns:
            Training history dictionary
        """
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
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")

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

        # Header
        print("\nConfusion Matrix:")
        header = "True\\Pred".ljust(25) + " ".join(l[:8].ljust(10) for l in labels)
        print(header)
        print("-" * len(header))

        # Rows
        for i, row in enumerate(confusion):
            row_label = labels[i].ljust(25)
            row_values = " ".join(str(v).ljust(10) for v in row)
            print(f"{row_label}{row_values}")


def save_model(
    model: FormalityClassifier,
    tokenizer: KotogramTokenizer,
    path: str,
    config: Optional[ModelConfig] = None,
):
    """Save trained model, tokenizer, and config.

    Args:
        model: Trained FormalityClassifier
        tokenizer: KotogramTokenizer with vocabulary
        path: Directory path to save files
    """
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


def load_model(path: str, device: str = None) -> Tuple[FormalityClassifier, KotogramTokenizer]:
    """Load trained model and tokenizer.

    Args:
        path: Directory path with saved model files
        device: Device to load model to

    Returns:
        Tuple of (model, tokenizer)
    """
    import os

    # Load config
    with open(os.path.join(path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    config = ModelConfig.from_dict(config_dict)

    # Load tokenizer
    tokenizer = KotogramTokenizer.load(os.path.join(path, 'tokenizer.json'))

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
    tokenizer: KotogramTokenizer,
    parser=None,
    device: str = None,
) -> Tuple[FormalityLevel, Dict[str, float]]:
    """Predict formality level for a Japanese sentence.

    Args:
        sentence: Japanese sentence text
        model: Trained FormalityClassifier
        tokenizer: KotogramTokenizer
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
    token_ids = tokenizer.encode(kotogram, add_cls=True, add_to_vocab=False)
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    if device:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        model.to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        probs = model.predict(input_ids, attention_mask)[0]

    predicted_id = probs.argmax().item()
    predicted_label = FormalityDataset.ID_TO_LABEL[predicted_id]

    prob_dict = {
        FormalityDataset.ID_TO_LABEL[i].value: probs[i].item()
        for i in range(FormalityDataset.NUM_CLASSES)
    }

    return predicted_label, prob_dict


# Self-supervised pretraining utilities

def create_mlm_batch(
    batch: Dict[str, torch.Tensor],
    mask_prob: float = 0.15,
    mask_token_id: int = 3,
    vocab_size: int = None,
    special_token_ids: List[int] = None,
) -> Dict[str, torch.Tensor]:
    """Create masked language modeling batch from regular batch.

    Args:
        batch: Batch with input_ids, attention_mask
        mask_prob: Probability of masking a token
        mask_token_id: ID of MASK token
        vocab_size: Vocabulary size for random token replacement
        special_token_ids: IDs to never mask

    Returns:
        Batch with masked input_ids and mlm_labels
    """
    special_token_ids = special_token_ids or [0, 1, 2, 3]  # PAD, UNK, CLS, MASK

    input_ids = batch['input_ids'].clone()
    mlm_labels = torch.full_like(input_ids, -100)  # Ignore index for loss

    # Create mask for tokens that can be masked
    maskable = batch['attention_mask'].bool()
    for special_id in special_token_ids:
        maskable &= (input_ids != special_id)

    # Random mask
    probs = torch.rand_like(input_ids.float())
    mask = maskable & (probs < mask_prob)

    # Set labels for masked positions
    mlm_labels[mask] = input_ids[mask]

    # 80% MASK, 10% random, 10% unchanged
    mask_token = mask & (probs < mask_prob * 0.8)
    random_token = mask & (probs >= mask_prob * 0.8) & (probs < mask_prob * 0.9)

    input_ids[mask_token] = mask_token_id
    if vocab_size:
        input_ids[random_token] = torch.randint(
            len(special_token_ids), vocab_size, (random_token.sum().item(),)
        )

    return {
        'input_ids': input_ids,
        'attention_mask': batch['attention_mask'],
        'mlm_labels': mlm_labels,
    }


class MLMTrainer:
    """Trainer for self-supervised masked language modeling pretraining."""

    def __init__(
        self,
        model: FormalityClassifierWithMLM,
        dataset: FormalityDataset,
        config: TrainerConfig = None,
        mask_prob: float = 0.15,
    ):
        self.model = model
        self.dataset = dataset
        self.config = config or TrainerConfig()
        self.mask_prob = mask_prob

        self.device = torch.device(self.config.device)
        self.model.to(self.device)

        pad_id = dataset.tokenizer.pad_id
        self.data_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, pad_id),
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.optimizer = Adam(model.parameters(), lr=self.config.learning_rate)

        self.history = {'mlm_loss': []}

    def train_epoch(self) -> float:
        """Run one MLM pretraining epoch."""
        self.model.train()
        total_loss = 0
        n_batches = 0

        for batch in self.data_loader:
            # Create MLM batch
            mlm_batch = create_mlm_batch(
                batch,
                mask_prob=self.mask_prob,
                mask_token_id=self.dataset.tokenizer.mask_id,
                vocab_size=self.dataset.tokenizer.vocab_size,
            )

            input_ids = mlm_batch['input_ids'].to(self.device)
            attention_mask = mlm_batch['attention_mask'].to(self.device)
            mlm_labels = mlm_batch['mlm_labels'].to(self.device)

            self.optimizer.zero_grad()
            mlm_logits = self.model.forward_mlm(input_ids, attention_mask)

            # Compute loss over masked positions
            loss = self.criterion(
                mlm_logits.view(-1, mlm_logits.size(-1)),
                mlm_labels.view(-1),
            )

            loss.backward()
            if self.config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def train(self, epochs: int = None, verbose: bool = True) -> Dict[str, List[float]]:
        """Run MLM pretraining.

        Args:
            epochs: Number of epochs (defaults to config.epochs)
            verbose: Print progress

        Returns:
            Training history
        """
        epochs = epochs or self.config.epochs

        for epoch in range(epochs):
            mlm_loss = self.train_epoch()
            self.history['mlm_loss'].append(mlm_loss)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - MLM Loss: {mlm_loss:.4f}")

        return self.history


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
    parser.add_argument("--encoder", type=str, default="transformer",
                        choices=["transformer", "bilstm"],
                        help="Encoder architecture")
    parser.add_argument("--embed-dim", type=int, default=128,
                        help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden layer dimension")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of encoder layers")
    parser.add_argument("--pretrain-mlm", action="store_true",
                        help="Pre-train with masked language modeling")
    parser.add_argument("--pretrain-epochs", type=int, default=5,
                        help="MLM pretraining epochs")
    parser.add_argument("--strip-surface", action="store_true",
                        help="Strip surface forms (kanji) from tokens to reduce vocab size")

    args = parser.parse_args()

    print("Loading data...")
    tokenizer = KotogramTokenizer(strip_surface=args.strip_surface)
    if args.strip_surface:
        print("Surface forms will be stripped (grammar-only tokens)")
    dataset = FormalityDataset.from_tsv(
        args.data,
        tokenizer,
        max_samples=args.max_samples,
        verbose=True,
    )

    train_data, val_data, test_data = dataset.split()
    print(f"\nSplit: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # Model config
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        num_classes=FormalityDataset.NUM_CLASSES,
        embedding_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        encoder_type=args.encoder,
    )

    # Create model
    if args.pretrain_mlm:
        print("\nCreating model with MLM head...")
        model = FormalityClassifierWithMLM(model_config)

        # MLM pretraining
        print("\nStarting MLM pretraining...")
        mlm_trainer = MLMTrainer(model, train_data)
        mlm_trainer.train(epochs=args.pretrain_epochs)
    else:
        print("\nCreating model...")
        model = FormalityClassifier(model_config)

    # Supervised training
    print("\nStarting supervised training...")
    trainer_config = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    trainer = Trainer(model, train_data, val_data, trainer_config)
    history = trainer.train()

    # Print final metrics
    trainer.print_confusion_matrix()

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id),
    )

    model.eval()
    device = torch.device(trainer_config.device)
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {correct/total:.4f}")

    # Save model
    print(f"\nSaving model to {args.output}...")
    save_model(model, tokenizer, args.output, model_config)
    print("Done!")
