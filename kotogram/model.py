"""Style classification model definition and inference utilities.

This module contains the model architecture, configuration, and inference functions
for the Japanese style classifier (formality + gender + grammaticality).
"""

import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, TypedDict, cast

import torch
import torch.nn.functional as F
from torch import nn

from kotogram.constants import RegisterLevel
from kotogram.tokenizer import (
    FEATURE_FIELDS,
    Tokenizer,
)

NUM_FORMALITY_PRAGMATIC_CLASSES = 2
NUM_GRAMMATICALITY_CLASSES = 2  # grammatic (1) vs agrammatic (0)
NUM_GENDER_PRAGMATIC_CLASSES = 2  # pragmatic (1) vs unpragmatic (0)

# Label mappings
# Register classes
NUM_REGISTER_CLASSES = 14
REGISTER_ID_TO_LABEL = {
    0: RegisterLevel.NEUTRAL,
    1: RegisterLevel.SONKEIGO,
    2: RegisterLevel.KENJOGO,
    3: RegisterLevel.KANSAIBEN,
    4: RegisterLevel.HAKATABEN,
    5: RegisterLevel.KYOSHIGO,
    6: RegisterLevel.NETSLANG,
    7: RegisterLevel.OJOUSAMA,
    8: RegisterLevel.GUNTAI,
    9: RegisterLevel.JOSEIGO,
    10: RegisterLevel.DANSEIGO,
    11: RegisterLevel.BURIKKO,
    12: RegisterLevel.TOHOKU,
    13: RegisterLevel.BUSHI,
}


class StylePrediction(NamedTuple):
    """Output prediction from the style classifier."""

    formality_value: torch.Tensor
    formality_pragmatic_probs: torch.Tensor
    gender_value: torch.Tensor
    gender_pragmatic_probs: torch.Tensor
    grammaticality_probs: torch.Tensor
    register_probs: torch.Tensor
    kcs: Optional[torch.Tensor] = None


class ModelConfigDict(TypedDict):
    """Serialization format for ModelConfig."""

    vocab_sizes: Dict[str, int]
    num_formality_pragmatic_classes: int
    num_gender_pragmatic_classes: int
    num_grammaticality_classes: int
    num_register_classes: int
    field_embed_dims: Dict[str, int]
    d_model: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    max_seq_len: int
    pooling: str
    kc_enabled: bool
    kc_vocab_size: int
    kc_topk: int
    kc_temperature: float
    kc_target_specs: Dict[str, int]


@dataclass
class ModelConfig:
    """Configuration for StyleClassifier model."""

    vocab_sizes: Dict[str, int]  # Field name -> vocabulary size
    num_formality_pragmatic_classes: int = NUM_FORMALITY_PRAGMATIC_CLASSES
    num_gender_pragmatic_classes: int = NUM_GENDER_PRAGMATIC_CLASSES
    num_grammaticality_classes: int = NUM_GRAMMATICALITY_CLASSES
    num_register_classes: int = NUM_REGISTER_CLASSES
    field_embed_dims: Dict[str, int] = field(
        default_factory=lambda: {
            "surface": 64,
            "pos": 32,
            "pos_detail_1": 32,
            "pos_detail_2": 16,
            "pos_detail_3": 16,
            "conjugated_type": 32,
            "conjugated_form": 32,
            "lemma": 64,
        }
    )
    d_model: int = 256
    hidden_dim: int = 512
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 512
    pooling: str = "cls"

    # KC Learning configuration
    kc_enabled: bool = False
    kc_vocab_size: int = 1024  # Size of the sparse concept, vocabulary
    kc_topk: int = 8  # Number of active concepts to retrieve
    kc_temperature: float = 1.0  # Sparsification temperature
    kc_target_specs: Dict[str, int] = field(
        default_factory=dict
    )  # Target head name -> vocab size

    def to_dict(self) -> ModelConfigDict:
        return {
            "vocab_sizes": self.vocab_sizes,
            "num_formality_pragmatic_classes": self.num_formality_pragmatic_classes,
            "num_gender_pragmatic_classes": self.num_gender_pragmatic_classes,
            "num_grammaticality_classes": self.num_grammaticality_classes,
            "num_register_classes": self.num_register_classes,
            "field_embed_dims": self.field_embed_dims,
            "d_model": self.d_model,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "max_seq_len": self.max_seq_len,
            "pooling": self.pooling,
            "kc_enabled": self.kc_enabled,
            "kc_vocab_size": self.kc_vocab_size,
            "kc_topk": self.kc_topk,
            "kc_temperature": self.kc_temperature,
            "kc_target_specs": self.kc_target_specs,
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any]
    ) -> "ModelConfig":  # Keep Dict[str, Any] for flexibility or strict?
        # Sticking to Dict[str, Any] for looser input, but validation happened if it came from to_dict
        # Ideally input should be Union[ModelConfigDict, Dict[str, Any]] but Dict[str, Any] covers both
        from dataclasses import fields

        valid_fields = {f.name for f in fields(cls)}

        # Migration logic for old pos_detail naming in field_embed_dims
        if "field_embed_dims" in d:
            dims = d["field_embed_dims"]
            for i in range(1, 4):
                old_key = f"pos_detail{i}"
                new_key = f"pos_detail_{i}"
                if old_key in dims and new_key not in dims:
                    dims[new_key] = dims.pop(old_key)

        return cls(**{k: v for k, v in d.items() if k in valid_fields})


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        max_len = 512

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = cast(torch.Tensor, self.pe)
        x = x + pe[:, : x.size(1), :]
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
            input_ids = field_inputs[f"input_ids_{field_name}"]
            embed = self.embeddings[field_name](input_ids)
            field_embeds.append(embed)

        concat = torch.cat(field_embeds, dim=-1)
        projected = self.projection(concat)
        normalized = self.layer_norm(projected)
        return cast(torch.Tensor, self.dropout(normalized))

    def resize_embeddings(self, new_vocab_sizes: Dict[str, int]) -> Dict[str, int]:
        resized = {}
        for field_name in FEATURE_FIELDS:
            embedding = self.embeddings[field_name]
            assert isinstance(embedding, nn.Embedding)  # Type hint for mypy
            old_size = embedding.num_embeddings
            new_size = new_vocab_sizes.get(field_name, old_size)

            if new_size > old_size:
                embed_dim = embedding.embedding_dim
                old_weight = embedding.weight.data

                new_embedding = nn.Embedding(new_size, embed_dim, padding_idx=0)
                new_embedding.weight.data[:old_size] = old_weight

                self.embeddings[field_name] = new_embedding
                resized[field_name] = new_size - old_size
                self.config.vocab_sizes[field_name] = new_size
            else:
                resized[field_name] = 0
        return resized


# Required for Mypy compatibility with torch.nn.Module
class KCHead(nn.Module):  # type: ignore[misc]
    """Head for predicting Knowledge Components (sparse concepts)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.d_model, config.kc_vocab_size)
        self.layer_norm = nn.LayerNorm(config.kc_vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.layer_norm(x)
        return cast(torch.Tensor, x)

    def forward_with_raw(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.linear(x)
        out = self.layer_norm(raw)
        return raw, cast(torch.Tensor, out)


# Required for Mypy compatibility with torch.nn.Module
class StyleClassifier(nn.Module):  # type: ignore[misc]
    """Neural sequence classifier for multi-task style prediction."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embedding = MultiFieldEmbedding(config)
        self.pos_encoding = PositionalEncoding(
            config.d_model,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, config.num_layers, enable_nested_tensor=False
        )

        self.formality_value_head = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
            nn.Tanh(),
        )

        self.formality_pragmatic_head = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_formality_pragmatic_classes),
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

        if config.kc_enabled:
            self.kc_head = KCHead(config)

    def get_encoder_output(
        self,
        field_inputs: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get the sequence of encoder hidden states."""
        x = self.embedding(field_inputs)
        x = self.pos_encoding(x)

        src_key_padding_mask = attention_mask == 0

        x = cast(
            torch.Tensor, self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        )
        return x

    def _get_pooled_output(
        self,
        field_inputs: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.get_encoder_output(field_inputs, attention_mask)

        if self.config.pooling == "cls":
            pooled = x[:, 0, :]
        elif self.config.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        elif self.config.pooling == "max":
            mask = attention_mask.unsqueeze(-1).float()
            x = x.masked_fill(mask == 0, float("-inf"))
            pooled = x.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling: {self.config.pooling}")

        return pooled

    def forward(
        self,
        field_inputs: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        pooled = self._get_pooled_output(field_inputs, attention_mask)
        return (
            self.formality_value_head(pooled),
            self.formality_pragmatic_head(pooled),
            self.gender_value_head(pooled),
            self.gender_pragmatic_head(pooled),
            self.grammaticality_classifier(pooled),
            self.register_classifier(pooled),
        )

    def predict(
        self,
        field_inputs: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> StylePrediction:
        formality_val, formality_prag, gender_val, gender_prag, gram, reg = self(
            field_inputs, attention_mask
        )
        kcs = None
        if self.config.kc_enabled:
            kcs = self.predict_kcs(field_inputs, attention_mask)

        return StylePrediction(
            formality_value=formality_val,  # Already Tanh
            formality_pragmatic_probs=F.softmax(formality_prag, dim=-1),
            gender_value=gender_val,  # Already Tanh
            gender_pragmatic_probs=F.softmax(gender_prag, dim=-1),
            grammaticality_probs=F.softmax(gram, dim=-1),
            register_probs=torch.sigmoid(reg),
            kcs=kcs,
        )

    def resize_embeddings(self, new_vocab_sizes: Dict[str, int]) -> Dict[str, int]:
        return self.embedding.resize_embeddings(new_vocab_sizes)

    def predict_kcs(
        self,
        field_inputs: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict Knowledge Component activations for input sentences.

        Args:
            field_inputs: Input features
            attention_mask: Attention mask

        Returns:
            activations: (B, K) tensor of sparse KC activations (or logits)
        """
        if not self.config.kc_enabled:
            raise ValueError("KC learning is not enabled for this model.")

        pooled = self._get_pooled_output(field_inputs, attention_mask)
        return cast(torch.Tensor, self.kc_head(pooled))

    def predict_kcs_top(
        self,
        field_inputs: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        topk: Optional[int] = None,
        min_prob: float = 0.0,
    ) -> List[List[Tuple[int, float]]]:
        # pylint: disable=too-many-locals
        """Predict top-K Knowledge Components with probabilities.

        Args:
            field_inputs: Input features
            attention_mask: Attention mask
            topk: Number of KCs to return (defaults to filtered by config.kc_topk)
            min_prob: Minimum probability threshold

        Returns:
            List of lists, where each inner list contains (kc_id, probability) tuples
            for a sample in the batch.
        """
        # Get dense logits
        if not self.config.kc_enabled:
            return []

        logits = self.predict_kcs(field_inputs, attention_mask)
        cur_temp = getattr(self.config, "kc_temperature", 1.0)
        probs = torch.sigmoid(logits / cur_temp)

        # Determine K
        k = topk if topk is not None else getattr(self.config, "kc_topk", 8)
        k = min(k, probs.size(-1))

        # Get top-k
        topk_vals, topk_inds = torch.topk(probs, k, dim=-1)

        results = []
        # Convert to python lists
        probs_list = topk_vals.tolist()
        inds_list = topk_inds.tolist()

        for prob_row, ind_row in zip(probs_list, inds_list):
            sample_res = []
            for j in range(k):
                p = prob_row[j]
                if p >= min_prob:
                    sample_res.append((int(ind_row[j]), float(p)))
            results.append(sample_res)

        return results


def load_model(
    path: str, device: Optional[torch.device] = None
) -> Tuple[StyleClassifier, Tokenizer]:
    """Load trained model and tokenizer."""
    # Load config
    with open(os.path.join(path, "model.json"), "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config = ModelConfig.from_dict(config_dict)

    # Load tokenizer
    tokenizer = Tokenizer.load(os.path.join(path, "tokenizer.json"))

    # Load model
    model = StyleClassifier(config)
    if device:
        model.to(device)

    # Always load to CPU first
    state_dict: Dict[str, Any] = torch.load(
        os.path.join(path, "model.pt"), map_location="cpu"
    )

    # Convert weights back to float32
    def to_float32(v: torch.Tensor) -> torch.Tensor:
        if v.dtype == torch.float16:
            return v.float()
        if hasattr(torch, "float8_e4m3fn") and v.dtype == torch.float8_e4m3fn:
            return v.float()
        return v

    state_dict = {k: to_float32(v).contiguous() for k, v in state_dict.items()}

    # Load with strict=False to allow architecture changes (e.g. gender head refactor)
    # We catch the error/warning to report relevant mismatches
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"WARNING: Missing keys in state_dict: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        print(f"WARNING: Unexpected keys in state_dict: {incompatible.unexpected_keys}")

    model.eval()
    return model, tokenizer


def load_default_style_model() -> Tuple[StyleClassifier, Tokenizer]:
    """Load the default trained style classification model included in the package."""
    import importlib.resources

    from kotogram import locations

    # Dev/Source mode: Check if we are running in a project with a trained model
    dev_model_dir = locations.get_style_output_dir()
    if os.path.exists(os.path.join(dev_model_dir, "model.pt")):
        return load_model(dev_model_dir)

    if sys.version_info >= (3, 9):
        from importlib.resources import as_file, files

        ref = files("kotogram.model_data").joinpath("model.pt")
        with as_file(ref) as model_file:
            model_dir = os.path.dirname(model_file)
            return load_model(model_dir)
    else:
        with importlib.resources.path("kotogram.model_data", "model.pt") as model_file:
            model_dir = os.path.dirname(model_file)
            return load_model(model_dir)


def is_default_style_model_available() -> bool:
    """Check if the default style model is available."""
    import importlib.util

    from kotogram import locations

    # Dev/Source mode: Check if we are running in a project with a trained model
    dev_model_dir = locations.get_style_output_dir()
    if os.path.exists(os.path.join(dev_model_dir, "model.pt")):
        return True

    # Package mode check
    if sys.version_info >= (3, 9):
        from importlib.resources import files  # type: ignore

        return files("kotogram.model_data").joinpath("model.pt").is_file()

    # Fallback for < 3.9
    import importlib.resources

    return importlib.resources.is_resource("kotogram.model_data", "model.pt")
