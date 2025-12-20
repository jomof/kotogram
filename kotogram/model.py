"""Style classification model definition and inference utilities.

This module contains the model architecture, configuration, and inference functions
for the Japanese style classifier (formality + gender + grammaticality).
"""

import hashlib
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from kotogram.kotogram import split_kotogram
from kotogram.analysis import FormalityLevel, GenderLevel
from kotogram.kotogram import extract_token_features
from kotogram.japanese_parser import JapaneseParser
from kotogram.preprocess import preprocess

# Special token values for vocabulary
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
CLS_TOKEN = "<CLS>"
MASK_TOKEN = "<MASK>"  # For self-supervised pretraining

# Feature fields used for token embedding
# NOTE: 'surface' is critical for gender detection (pronouns like 僕, 俺, あたし)
ALL_FEATURE_FIELDS = ['surface', 'pos', 'pos_detail1', 'pos_detail2', 'conjugated_type', 'conjugated_form', 'lemma']
FEATURE_FIELDS = ALL_FEATURE_FIELDS  # Default: use all features

# Global variable to track excluded features (set via --exclude-features)
_EXCLUDED_FEATURES: List[str] = []


def get_active_features() -> List[str]:
    """Get the list of active feature fields (excluding any disabled ones)."""
    return [f for f in ALL_FEATURE_FIELDS if f not in _EXCLUDED_FEATURES]


def set_excluded_features(excluded: List[str]) -> None:
    """Set the list of features to exclude from training/inference."""
    global _EXCLUDED_FEATURES, FEATURE_FIELDS
    invalid = [f for f in excluded if f not in ALL_FEATURE_FIELDS]
    if invalid:
        raise ValueError(f"Invalid feature names: {invalid}. Valid: {ALL_FEATURE_FIELDS}")
    _EXCLUDED_FEATURES = excluded
    FEATURE_FIELDS = get_active_features()


from kotogram.analysis import RegisterLevel
from kotogram.constants import (
    NUM_FORMALITY_CLASSES as NUM_FORMALITY_CLASSES,
    NUM_GRAMMATICALITY_CLASSES as NUM_GRAMMATICALITY_CLASSES,
    NUM_GENDER_PRAGMATIC_CLASSES as NUM_GENDER_PRAGMATIC_CLASSES,
    NUM_REGISTER_CLASSES as NUM_REGISTER_CLASSES,
    FORMALITY_LABEL_TO_ID as FORMALITY_LABEL_TO_ID,
    FORMALITY_ID_TO_LABEL as FORMALITY_ID_TO_LABEL,
    GENDER_LABEL_TO_ID as GENDER_LABEL_TO_ID,
    GENDER_ID_TO_LABEL as GENDER_ID_TO_LABEL,
    REGISTER_LABEL_TO_ID as REGISTER_LABEL_TO_ID,
    REGISTER_ID_TO_LABEL as REGISTER_ID_TO_LABEL,
)


@dataclass
class Sample:
    """Single data sample with features and labels."""
    feature_ids: Dict[str, List[int]]
    formality_label: int
    gender_value: float
    gender_pragmatic: int
    register_labels: List[int] = field(default_factory=lambda: [0])
    grammaticality_label: int = 1
    original_sentence: str = ""
    kotogram: str = ""

    @property
    def seq_len(self) -> int:
        """Get sequence length."""
        first = next(iter(self.feature_ids.values()))
        return len(first)

class Tokenizer:
    """Tokenizer that extracts morphological features from Kotogram tokens.

    Instead of treating each token as a single vocabulary item, this tokenizer
    extracts categorical features (pos, pos_detail1, conjugated_type, conjugated_form,
    lemma) and maintains separate vocabularies for each field.

    Attributes:
        field_vocabs: Dict mapping field name to {value: id} mapping
        field_vocab_sizes: Dict mapping field name to vocabulary size
    """

    def __init__(self) -> None:
        """Initialize feature tokenizer."""
        # Initialize vocabularies for each field with special tokens
        self.field_vocabs: Dict[str, Dict[str, int]] = {}
        # self._field_counters: Dict[str, Counter[str]] = {} # Training only
        for f in FEATURE_FIELDS:
            self.field_vocabs[f] = {
                PAD_TOKEN: 0,
                UNK_TOKEN: 1,
                CLS_TOKEN: 2,
                MASK_TOKEN: 3,
            }
            # self._field_counters[f] = Counter() # Training only

        self._frozen = True # Inference tokenizer is always frozen

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

    # _add_value removed (training only)

    def extract_features(self, kotogram: str) -> List[Dict[str, str]]:
        """Extract features from each token in a Kotogram string."""
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
    ) -> Dict[str, List[int]]:
        """Convert list of feature dicts to sequences of field IDs."""
        result: Dict[str, List[int]] = {f: [] for f in FEATURE_FIELDS}

        if add_cls:
            for field in FEATURE_FIELDS:
                result[field].append(self.cls_id)

        for features in features_list:
            for field in FEATURE_FIELDS:
                value = features.get(field, '')
                # Inference only: just lookup
                vocab = self.field_vocabs[field]
                token_id = vocab.get(value, self.unk_id)
                result[field].append(token_id)

        return result

    def encode(
        self,
        kotogram: str,
        add_cls: bool = True,
    ) -> Dict[str, List[int]]:
        """Encode a Kotogram string to feature ID sequences."""
        features_list = self.extract_features(kotogram)
        return self.encode_features(features_list, add_cls)

    # freeze/unfreeze/save removed (training only)

    def get_model_config(self, **kwargs: Any) -> 'ModelConfig':
        """Create a ModelConfig with vocabulary sizes from this tokenizer."""
        return ModelConfig(
            vocab_sizes=self.get_vocab_sizes(),
            **kwargs
        )


    @classmethod
    def load(cls, path: str) -> 'Tokenizer':
        """Load tokenizer from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tokenizer = cls()
        tokenizer.field_vocabs = data['field_vocabs']
        tokenizer._frozen = data.get('frozen', False)
        return tokenizer


@dataclass
class ModelConfig:
    """Configuration for StyleClassifier model."""
    vocab_sizes: Dict[str, int]  # Field name -> vocabulary size
    num_formality_classes: int = NUM_FORMALITY_CLASSES
    num_gender_pragmatic_classes: int = NUM_GENDER_PRAGMATIC_CLASSES
    num_grammaticality_classes: int = NUM_GRAMMATICALITY_CLASSES
    num_register_classes: int = NUM_REGISTER_CLASSES
    field_embed_dims: Dict[str, int] = field(default_factory=lambda: {
        'surface': 64,
        'pos': 32,
        'pos_detail1': 32,
        'pos_detail2': 16,
        'conjugated_type': 32,
        'conjugated_form': 32,
        'lemma': 64,
    })
    d_model: int = 256
    hidden_dim: int = 512
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 512
    pooling: str = "cls"
    excluded_features: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'vocab_sizes': self.vocab_sizes,
            'num_formality_classes': self.num_formality_classes,
            'num_gender_pragmatic_classes': self.num_gender_pragmatic_classes,
            'num_grammaticality_classes': self.num_grammaticality_classes,
            'num_register_classes': self.num_register_classes,
            'field_embed_dims': self.field_embed_dims,
            'd_model': self.d_model,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'max_seq_len': self.max_seq_len,
            'pooling': self.pooling,
            'excluded_features': self.excluded_features,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModelConfig':
        d = dict(d)
        if 'excluded_features' not in d:
            d['excluded_features'] = []
        
        # Legacy compatibility: remove old fields
        if 'num_gender_classes' in d:
            d.pop('num_gender_classes')
            
        return cls(**d)


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
        pe = cast(torch.Tensor, self.pe)
        x = x + pe[:, :x.size(1), :]
        return cast(torch.Tensor, self.dropout(x))


class MultiFieldEmbedding(nn.Module):  # type: ignore[misc]
    """Embedding layer that combines multiple categorical feature embeddings."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embeddings = nn.ModuleDict()
        total_embed_dim = 0

        for field_name in FEATURE_FIELDS:
            vocab_size = config.vocab_sizes.get(field_name, 100)
            embed_dim = config.field_embed_dims.get(field_name, 32)
            self.embeddings[field_name] = nn.Embedding(
                vocab_size,
                embed_dim,
                padding_idx=0,
            )
            total_embed_dim += embed_dim

        self.projection = nn.Linear(total_embed_dim, config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, field_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        field_embeds = []
        for field_name in FEATURE_FIELDS:
            input_ids = field_inputs[f'input_ids_{field_name}']
            embed = self.embeddings[field_name](input_ids)
            field_embeds.append(embed)

        concat = torch.cat(field_embeds, dim=-1)
        projected = self.projection(concat)
        normalized = self.layer_norm(projected)
        return cast(torch.Tensor, self.dropout(normalized))

    # resize_embeddings removed (training only)


class StyleClassifier(nn.Module):  # type: ignore[misc]
    """Neural sequence classifier for multi-task style prediction."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embedding = MultiFieldEmbedding(config)
        self.pos_encoding = PositionalEncoding(
            config.d_model,
            config.max_seq_len,
            config.dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, config.num_layers, enable_nested_tensor=False
        )

        self.formality_classifier = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_formality_classes),
        )

        self.gender_value_head = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
            nn.Tanh(),
        )

        self.gender_pragmatic_head = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_gender_pragmatic_classes),
        )

        self.grammaticality_classifier = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_grammaticality_classes),
        )

        self.register_classifier = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_register_classes),
        )

    def get_encoder_output(
        self,
        field_inputs: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get the sequence of encoder hidden states."""
        x = self.embedding(field_inputs)
        x = self.pos_encoding(x)

        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        x = cast(torch.Tensor, self.encoder(x, src_key_padding_mask=src_key_padding_mask))
        return x

    def _get_pooled_output(
        self,
        field_inputs: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.get_encoder_output(field_inputs, attention_mask)

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = self._get_pooled_output(field_inputs, attention_mask)
        return (
            self.formality_classifier(pooled),
            self.gender_value_head(pooled),
            self.gender_pragmatic_head(pooled),
            self.grammaticality_classifier(pooled),
            self.register_classifier(pooled),
        )

    def predict(
        self,
        field_inputs: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        formality_logits, gender_val, gender_prag_logits, grammaticality_logits, register_logits = self(field_inputs, attention_mask)
        return (
            F.softmax(formality_logits, dim=-1),
            gender_val, # Already Tanh
            F.softmax(gender_prag_logits, dim=-1),
            F.softmax(grammaticality_logits, dim=-1),
            torch.sigmoid(register_logits),
        )

    # resize_embeddings removed (training only)


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

    # Restore excluded features BEFORE creating model or using tokenizer
    if config.excluded_features:
        set_excluded_features(config.excluded_features)

    # Load tokenizer
    tokenizer = Tokenizer.load(os.path.join(path, 'tokenizer.json'))

    # Load model
    model = StyleClassifier(config)
    # Always load to CPU first
    state_dict = torch.load(os.path.join(path, 'model.pt'), map_location='cpu')

    # Convert weights back to float32
    def to_float32(v: torch.Tensor) -> torch.Tensor:
        if v.dtype == torch.float16:
            return v.float()
        if hasattr(torch, 'float8_e4m3fn') and v.dtype == torch.float8_e4m3fn:
            return v.float()
        return v
    state_dict = {k: to_float32(v) for k, v in state_dict.items()}

    # Filter out MLM head weights if present
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('mlm_head.')}

    # Load with strict=False to allow architecture changes (e.g. gender head refactor)
    # We catch the error/warning to report relevant mismatches
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"WARNING: Missing keys in state_dict: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        print(f"WARNING: Unexpected keys in state_dict: {incompatible.unexpected_keys}")

    if device:
        model.to(device)

    model.eval()
    return model, tokenizer


def load_default_style_model(
    device: Optional[str] = None
) -> Tuple[StyleClassifier, Tokenizer]:
    """Load the default trained style classification model included in the package."""
    import importlib.resources
    import sys

    try:
        if sys.version_info >= (3, 9):
            from importlib.resources import files, as_file
            ref = files('kotogram.model_data').joinpath('model.pt')
            with as_file(ref) as model_file:
                 model_dir = os.path.dirname(model_file)
                 return load_model(model_dir, device=device)
        else:
            with importlib.resources.path('kotogram.model_data', 'model.pt') as model_file:
                 model_dir = os.path.dirname(model_file)
                 return load_model(model_dir, device=device)
    except (ImportError, ModuleNotFoundError):
        raise ImportError("Could not load default model. Ensure 'kotogram.model_data' package is installed and contains model files.")


def predict_style(
    sentence: str,
    model: StyleClassifier,
    tokenizer: Tokenizer,
    parser: Optional[JapaneseParser] = None,
    device: Optional[str] = None,
) -> Tuple[FormalityLevel, GenderLevel, bool, Dict[str, Any]]:
    """Predict formality, gender, and grammaticality for a Japanese sentence."""
    if parser is None:
        from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
        parser = SudachiJapaneseParser()

    # Preprocess the sentence before parsing
    sentence, _ = preprocess(sentence, parser=parser)
    
    kotogram = parser.japanese_to_kotogram(sentence)
    feature_ids = tokenizer.encode(kotogram, add_cls=True)

    field_inputs = {
        f'input_ids_{field}': torch.tensor([feature_ids[field]], dtype=torch.long)
        for field in FEATURE_FIELDS
    }
    attention_mask = torch.ones(1, len(feature_ids[FEATURE_FIELDS[0]]), dtype=torch.long)

    if device:
        field_inputs = {k: v.to(device) for k, v in field_inputs.items()}
        attention_mask = attention_mask.to(device)
        model.to(device)

    model.eval()
    with torch.no_grad():
        formality_probs, gender_val, gender_prag_probs, grammaticality_probs, _ = model.predict(field_inputs, attention_mask)
        formality_probs = formality_probs[0]
        gender_val = gender_val[0]
        gender_prag_probs = gender_prag_probs[0]
        grammaticality_probs = grammaticality_probs[0]

    formality_id = int(formality_probs.argmax().item())
    
    # Map continuous gender to label for legacy support logic
    is_pragmatic = gender_prag_probs[1].item() > 0.5
    if not is_pragmatic:
        gender_id = 3 # UNPRAGMATIC_GENDER
    else:
        val = gender_val.item()
        if val < -0.33:
            gender_id = 0 # MASCULINE
        elif val > 0.33:
            gender_id = 1 # FEMININE
        else:
            gender_id = 2 # NEUTRAL

    grammaticality_id = int(grammaticality_probs.argmax().item())

    formality_label = FORMALITY_ID_TO_LABEL[formality_id]
    gender_label = GENDER_ID_TO_LABEL[gender_id]
    is_grammatic = grammaticality_id == 1

    prob_dicts = {
        'formality': {
            FORMALITY_ID_TO_LABEL[i].value: formality_probs[i].item()
            for i in range(NUM_FORMALITY_CLASSES)
        },
        'gender': {
            'value': gender_val.item(),
            'pragmatic_prob': gender_prag_probs[1].item(),
            'label': GENDER_ID_TO_LABEL[gender_id].value
        },
        'grammaticality': {
            'grammatical': grammaticality_probs[1].item(),
            'agrammatical': grammaticality_probs[0].item(),
        }
    }

    return formality_label, gender_label, is_grammatic, prob_dicts
