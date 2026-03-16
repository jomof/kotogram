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

from kotogram.tokenizer import (
    ENCODER_FEATURE_FIELDS,
    Tokenizer,
)

NUM_FORMALITY_PRAGMATIC_CLASSES = 2
NUM_GRAMMATICALITY_CLASSES = 2  # grammatic (1) vs agrammatic (0)
NUM_GENDER_PRAGMATIC_CLASSES = 2  # pragmatic (1) vs unpragmatic (0)


# Label mappings
# Register classes
NUM_REGISTER_CLASSES = 14
# REGISTER_ID_TO_LABEL is imported from constants.py (source of truth)


class StylePrediction(NamedTuple):
    """Output prediction from the style classifier."""

    formality_value: torch.Tensor
    formality_pragmatic_probs: torch.Tensor
    gender_value: torch.Tensor
    gender_pragmatic_probs: torch.Tensor
    grammaticality_probs: torch.Tensor
    register_probs: torch.Tensor


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

    kc_vocab_size: int
    kc_temperature: float
    kc_threshold: float


@dataclass
class ModelConfig:
    """Configuration for InferenceClassifier model."""

    vocab_sizes: Dict[str, int]  # Field name -> vocabulary size
    num_formality_pragmatic_classes: int = NUM_FORMALITY_PRAGMATIC_CLASSES
    num_gender_pragmatic_classes: int = NUM_GENDER_PRAGMATIC_CLASSES
    num_grammaticality_classes: int = NUM_GRAMMATICALITY_CLASSES
    num_register_classes: int = NUM_REGISTER_CLASSES
    field_embed_dims: Dict[str, int] = field(
        default_factory=lambda: {
            "surface": 300,  # Matches chiVe pretrained vector dimension
        }
    )
    d_model: int = 512
    hidden_dim: int = 2048
    num_layers: int = 4
    num_heads: int = 16
    dropout: float = 0.1
    max_seq_len: int = 512
    pooling: str = "cls"

    # KC Learning configuration (KC is always enabled)
    kc_vocab_size: int = 1024  # Size of the concept vocabulary
    kc_temperature: float = 1.0  # Sparsification temperature
    kc_threshold: float = 0.5  # Adaptive KC activation threshold

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
            "kc_vocab_size": self.kc_vocab_size,
            "kc_temperature": self.kc_temperature,
            "kc_threshold": self.kc_threshold,
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any]
    ) -> "ModelConfig":  # Keep Dict[str, Any] for flexibility or strict?
        # Sticking to Dict[str, Any] for looser input, but validation happened if it came from to_dict
        # Ideally input should be Union[ModelConfigDict, Dict[str, Any]] but Dict[str, Any] covers both
        from dataclasses import fields

        valid_fields = {f.name for f in fields(cls)}

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


class SurfaceEmbedding(nn.Module):  # type: ignore[misc]
    """Embedding layer for surface token inputs.

    Embeds ENCODER_FEATURE_FIELDS (currently surface-only) and projects to d_model.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embeddings = nn.ModuleDict()
        total_embed_dim = 0

        for field_name in ENCODER_FEATURE_FIELDS:
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
        for field_name in ENCODER_FEATURE_FIELDS:
            input_ids = field_inputs[f"input_ids_{field_name}"]
            embed = self.embeddings[field_name](input_ids)
            field_embeds.append(embed)

        concat = torch.cat(field_embeds, dim=-1)
        projected = self.projection(concat)
        normalized = self.layer_norm(projected)
        return cast(torch.Tensor, self.dropout(normalized))

    def resize_embeddings(self, new_vocab_sizes: Dict[str, int]) -> Dict[str, int]:
        resized = {}
        for field_name in ENCODER_FEATURE_FIELDS:
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
    """Head for predicting Knowledge Components (sparse concepts).

    Architecture: Multi-layer MLP for richer KC encoding.
    - Hidden layer 1: d_model -> d_model * 2 (expansion)
    - ReLU + Dropout
    - Hidden layer 2: d_model * 2 -> d_model
    - ReLU + Dropout
    - Output layer: d_model -> kc_vocab_size
    - LayerNorm on output

    Rationale: Deeper/wider head gives more capacity to encode subtle signals
    (like gender markers) into KC space alongside structural features.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        intermediate_dim = config.d_model * 2  # Expansion layer

        # Multi-layer projection
        self.hidden1 = nn.Linear(config.d_model, intermediate_dim)
        self.hidden2 = nn.Linear(intermediate_dim, config.d_model)
        self.output = nn.Linear(config.d_model, config.kc_vocab_size)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.kc_vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Through hidden layers
        x = self.dropout(self.activation(self.hidden1(x)))
        x = self.dropout(self.activation(self.hidden2(x)))
        # Output projection
        x = self.output(x)
        x = self.layer_norm(x)
        return cast(torch.Tensor, x)

    def forward_with_raw(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Through hidden layers
        x = self.dropout(self.activation(self.hidden1(x)))
        x = self.dropout(self.activation(self.hidden2(x)))
        # Output projection
        raw = self.output(x)
        out = self.layer_norm(raw)
        return raw, cast(torch.Tensor, out)


# Constants for inference-time KC decoder
KC_DECODER_HIDDEN_DIM = 256  # Must match train/models.py KCDecoder


# Required for Mypy compatibility with torch.nn.Module
class KCDecoderInference(nn.Module):  # type: ignore[misc]
    """Lightweight KC decoder for inference-time predictions.

    This mirrors the structure of train/models.py KCDecoder to load saved weights.
    Supports:
    - grammar_point: Multi-label classification via label pathway
    - formality/gender: Continuous regression via MSE pathway
    - register: Multi-label classification via label pathway (style family)

    Architecture matches training:
    - label_hidden1: kc_vocab_size -> hidden_dim (256)
    - label_hidden2: hidden_dim -> hidden_dim
    - mse_hidden1: kc_vocab_size -> hidden_dim (256)
    - mse_hidden2: hidden_dim -> hidden_dim
    - decoders.grammar_point: hidden_dim -> num_grammar_points
    - decoders.register: hidden_dim -> num_register_classes
    - mse_decoders.formality: hidden_dim -> 1
    - mse_decoders.gender: hidden_dim -> 1
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        config: ModelConfig,
        num_grammar_points: int,
        has_formality: bool = False,
        has_gender: bool = False,
        has_register: bool = False,
    ):
        super().__init__()
        kc_vocab_size = config.kc_vocab_size

        # Label pathway hidden layers (grammar_point is a label family)
        # Must match train/models.py KCDecoder architecture for weight loading
        self.label_hidden1 = nn.Linear(kc_vocab_size, KC_DECODER_HIDDEN_DIM)
        self.label_hidden2 = nn.Linear(KC_DECODER_HIDDEN_DIM, KC_DECODER_HIDDEN_DIM)
        self.activation = nn.ReLU()

        # MSE pathway (for formality/gender regression)
        self.mse_hidden1 = nn.Linear(kc_vocab_size, KC_DECODER_HIDDEN_DIM)
        self.mse_hidden2 = nn.Linear(KC_DECODER_HIDDEN_DIM, KC_DECODER_HIDDEN_DIM)
        self.tanh = nn.Tanh()

        # Per-family label decoders
        self.decoders = nn.ModuleDict(
            {"grammar_point": nn.Linear(KC_DECODER_HIDDEN_DIM, num_grammar_points)}
        )
        if has_register:
            self.decoders["register"] = nn.Linear(
                KC_DECODER_HIDDEN_DIM, config.num_register_classes
            )

        # MSE decoders for continuous style predictions
        self.mse_decoders = nn.ModuleDict()
        if has_formality:
            self.mse_decoders["formality"] = nn.Linear(KC_DECODER_HIDDEN_DIM, 1)
        if has_gender:
            self.mse_decoders["gender"] = nn.Linear(KC_DECODER_HIDDEN_DIM, 1)

        self.num_grammar_points = num_grammar_points
        self.has_register = has_register

    def forward(self, kc_activations: torch.Tensor) -> torch.Tensor:
        """Predict grammar point probabilities from KC probabilities.

        Args:
            kc_activations: KC probability vector [B, kc_vocab_size]

        Returns:
            grammar_point_probs: [B, num_grammar_points] probabilities
        """
        # Through label pathway (grammar_point is a label family)
        h = self.activation(self.label_hidden1(kc_activations))
        h = self.activation(self.label_hidden2(h))

        # Through grammar_point decoder
        logits = self.decoders["grammar_point"](h)
        probs = torch.sigmoid(logits)

        return cast(torch.Tensor, probs)

    def predict_style_values(
        self, kc_probs: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Predict formality and gender values from KC probabilities.

        Uses the MSE pathway which processes full KC probabilities (not sparse).

        Args:
            kc_probs: Full KC probabilities [B, kc_vocab_size]

        Returns:
            Tuple of (formality_value, gender_value), each [B, 1] or None if not available
        """
        formality_val = None
        gender_val = None

        if not self.mse_decoders:
            return formality_val, gender_val

        # Through MSE pathway
        h = self.activation(self.mse_hidden1(kc_probs))
        h = self.activation(self.mse_hidden2(h))

        if "formality" in self.mse_decoders:
            formality_val = self.tanh(self.mse_decoders["formality"](h))

        if "gender" in self.mse_decoders:
            gender_val = self.tanh(self.mse_decoders["gender"](h))

        return formality_val, gender_val

    def predict_register(self, kc_probs: torch.Tensor) -> Optional[torch.Tensor]:
        """Predict register probabilities from KC probabilities.

        Register uses the label pathway with full KC probabilities (diffuse signal)
        since it's a style feature, not a structural one.

        Args:
            kc_probs: Full KC probabilities [B, kc_vocab_size]

        Returns:
            register_probs: [B, num_register_classes] probabilities or None if not available
        """
        if "register" not in self.decoders:
            return None

        # Register uses full KC probs (style family = diffuse signal)
        h = self.activation(self.label_hidden1(kc_probs))
        h = self.activation(self.label_hidden2(h))
        logits = self.decoders["register"](h)
        return cast(torch.Tensor, torch.sigmoid(logits))

    @property
    def weight_loading_modules(self) -> List[nn.Module]:
        """Return all modules needed for weight loading from training checkpoints.

        These modules are defined to match the training KCDecoder structure,
        even though not all are used at inference time. This ensures that
        state_dict loading works correctly.
        """
        return [
            self.label_hidden1,
            self.label_hidden2,
            self.mse_hidden1,
            self.mse_hidden2,
            self.activation,
            self.tanh,
            self.decoders,
            self.mse_decoders,
        ]


# Required for Mypy compatibility with torch.nn.Module
class AttentionPooler(nn.Module):  # type: ignore[misc]
    """Attention-weighted pooling with learnable query.

    Uses a learnable query vector to compute attention weights over
    encoder hidden states, producing a single pooled representation.
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self, encoder_output: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool encoder output using attention-weighted mechanism.

        Args:
            encoder_output: (B, S, D) encoder hidden states
            attention_mask: (B, S) attention mask (1=valid, 0=padding)

        Returns:
            pooled: (B, D) pooled representation
        """
        batch_size = encoder_output.size(0)
        # Expand query to batch size: (1, 1, D) -> (B, 1, D)
        query = self.query.expand(batch_size, -1, -1)

        # Create key_padding_mask for attention (True=ignore)
        key_padding_mask = attention_mask == 0

        # Cross-attention: query attends to encoder_output
        attn_output, _ = self.attention(
            query=query,
            key=encoder_output,
            value=encoder_output,
            key_padding_mask=key_padding_mask,
        )

        # attn_output: (B, 1, D) -> squeeze to (B, D)
        pooled = attn_output.squeeze(1)
        pooled = self.layer_norm(pooled)
        return cast(torch.Tensor, pooled)


# Required for Mypy compatibility with torch.nn.Module
class InferenceClassifier(nn.Module):  # type: ignore[misc]
    """Neural sequence classifier for multi-task style prediction.

    Style values (formality_value, gender_value) are predicted via the KC decoder's
    MSE pathway. Classification heads (pragmatic, grammaticality, register) are
    predicted directly from the pooled encoder output.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embedding = SurfaceEmbedding(config)
        self.position_encoding = PositionalEncoding(
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

        # Classifier input dimension: d_model (using attention pooler)
        classifier_input_dim = config.d_model

        # Unified attention pooler for both KC and style classification
        self.pooler = AttentionPooler(config.d_model, config.num_heads, config.dropout)

        # Pragmatic classification heads (is the style dimension pragmatically relevant?)
        self.formality_pragmatic_head = nn.Sequential(
            nn.Linear(classifier_input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_formality_pragmatic_classes),
        )

        self.gender_pragmatic_head = nn.Sequential(
            nn.Linear(classifier_input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_gender_pragmatic_classes),
        )

        self.grammaticality_classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_grammaticality_classes),
        )

        # Note: Register predictions now come from the KC decoder pathway,
        # not a standalone classifier. See kc_decoders.predict_register().

        self.kc_head = KCHead(config)

        # KC decoders for inference-time predictions (grammar_point, formality, gender)
        # Initialized by load_model with decoder weights from saved model
        # Note: TrainingClassifier overrides this with KCDecoder during training
        self.kc_decoders: Optional[KCDecoderInference] = None

    def get_encoder_output(
        self,
        field_inputs: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get the sequence of encoder hidden states."""
        x = self.embedding(field_inputs)
        x = self.position_encoding(x)

        src_key_padding_mask = attention_mask == 0

        x = cast(
            torch.Tensor, self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        )
        return x

    def forward(
        self,
        field_inputs: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Forward pass for classification heads only.

        Returns:
            Tuple of (formality_prag, gender_prag, grammaticality) logits.
            Style values (formality_value, gender_value) and register come from KC decoder.
        """
        # Get encoder hidden states
        encoder_output = self.get_encoder_output(field_inputs, attention_mask)

        # Use unified attention pooler for style classification
        classifier_input = self.pooler(encoder_output, attention_mask)

        return (
            self.formality_pragmatic_head(classifier_input),
            self.gender_pragmatic_head(classifier_input),
            self.grammaticality_classifier(classifier_input),
        )

    def predict(
        self,
        field_inputs: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> StylePrediction:
        """Full prediction including style values from KC decoder."""
        formality_prag, gender_prag, gram = self(field_inputs, attention_mask)

        # Get KC probabilities for style value prediction
        kc_logits = self.predict_kcs(field_inputs, attention_mask)
        cur_temp = getattr(self.config, "kc_temperature", 1.0)
        kc_probs = torch.sigmoid(kc_logits / cur_temp)

        # Get style values and register from KC decoder
        formality_val = None
        gender_val = None
        register_probs = None

        if self.kc_decoders is not None:
            formality_val, gender_val = self.kc_decoders.predict_style_values(kc_probs)
            register_probs = self.kc_decoders.predict_register(kc_probs)

        # Handle None from predict_style_values
        if formality_val is None:
            batch_size = formality_prag.size(0)
            device = formality_prag.device
            formality_val = torch.zeros(batch_size, 1, device=device)
        if gender_val is None:
            batch_size = gender_prag.size(0)
            device = gender_prag.device
            gender_val = torch.zeros(batch_size, 1, device=device)
        if register_probs is None:
            batch_size = gram.size(0)
            device = gram.device
            register_probs = torch.zeros(
                batch_size, self.config.num_register_classes, device=device
            )

        return StylePrediction(
            formality_value=formality_val,
            formality_pragmatic_probs=F.softmax(formality_prag, dim=-1),
            gender_value=gender_val,
            gender_pragmatic_probs=F.softmax(gender_prag, dim=-1),
            grammaticality_probs=F.softmax(gram, dim=-1),
            register_probs=register_probs,
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
            activations: (B, kc_vocab_size) tensor of KC logits
        """

        pooled = self.pooler(
            self.get_encoder_output(field_inputs, attention_mask), attention_mask
        )
        # Use raw (pre-LayerNorm) logits to match training's probability path.
        # Training uses forward_with_raw() and computes sigmoid(raw / temp).
        raw, _ = self.kc_head.forward_with_raw(pooled)
        return cast(torch.Tensor, raw)

    def predict_kcs_top(
        self,
        field_inputs: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        topk: Optional[int] = None,
        min_prob: float = 0.0,
    ) -> List[List[Tuple[int, float]]]:
        # pylint: disable=too-many-locals
        """Predict top Knowledge Components with probabilities.

        Returns all KCs above min_prob threshold, sorted by probability descending.
        If topk is specified, returns at most topk results per sample.

        Args:
            field_inputs: Input features
            attention_mask: Attention mask
            topk: Optional maximum number of results per sample
            min_prob: Minimum probability threshold

        Returns:
            List of lists, where each inner list contains (kc_id, probability) tuples
            for a sample in the batch.
        """
        # Get dense logits
        logits = self.predict_kcs(field_inputs, attention_mask)
        cur_temp = getattr(self.config, "kc_temperature", 1.0)
        probs = torch.sigmoid(logits / cur_temp)

        batch_size = probs.size(0)
        kc_vocab_size = probs.size(-1)

        # Sort all KCs by probability
        k = min(topk, kc_vocab_size) if topk is not None else kc_vocab_size

        results = []
        for i in range(batch_size):
            sample_probs = probs[i]  # (kc_vocab_size,)

            topk_vals, topk_inds = torch.topk(sample_probs, k)
            sample_res = []
            for j in range(k):
                p = topk_vals[j].item()
                if p >= min_prob:
                    sample_res.append((int(topk_inds[j].item()), float(p)))
                else:
                    break  # Sorted descending, no more will pass
            results.append(sample_res)

        return results

    def predict_grammar_points(
        self,
        field_inputs: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Predict grammar point probabilities for input sentences.

        Args:
            field_inputs: Input features
            attention_mask: Attention mask

        Returns:
            grammar_point_probs: [B, num_grammar_points] tensor of probabilities,
                                 or None if kc_decoders not available
        """
        if self.kc_decoders is None:
            return None

        # Only KCDecoderInference is supported for inference-time grammar_point
        if not isinstance(self.kc_decoders, KCDecoderInference):
            return None

        # Get KC logits and probabilities
        logits = self.predict_kcs(field_inputs, attention_mask)
        cur_temp = getattr(self.config, "kc_temperature", 1.0)
        kc_probs = torch.sigmoid(logits / cur_temp)

        return cast(torch.Tensor, self.kc_decoders(kc_probs))


def load_model(  # pylint: disable=too-many-locals
    path: str, device: Optional[torch.device] = None
) -> Tuple[InferenceClassifier, Tokenizer]:
    """Load trained model and tokenizer."""
    # Load config
    with open(os.path.join(path, "model.json"), "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config = ModelConfig.from_dict(config_dict)

    # Load tokenizer
    tokenizer = Tokenizer.load(os.path.join(path, "tokenizer.json"))

    # Load model
    model = InferenceClassifier(config)
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

    # Check for KC decoder weights
    gp_weight_key = "kc_decoders.decoders.grammar_point.weight"
    formality_weight_key = "kc_decoders.mse_decoders.formality.weight"
    gender_weight_key = "kc_decoders.mse_decoders.gender.weight"
    register_weight_key = "kc_decoders.decoders.register.weight"

    has_grammar_point = gp_weight_key in state_dict
    has_formality = formality_weight_key in state_dict
    has_gender = gender_weight_key in state_dict
    has_register = register_weight_key in state_dict

    if has_grammar_point or has_formality or has_gender or has_register:
        # Infer num_grammar_points from weight shape (default to 1 if not present)
        num_grammar_points = (
            state_dict[gp_weight_key].shape[0] if has_grammar_point else 1
        )

        # Initialize kc_decoders module with appropriate families
        model.kc_decoders = KCDecoderInference(
            config,
            num_grammar_points,
            has_formality=has_formality,
            has_gender=has_gender,
            has_register=has_register,
        )
        if device:
            model.kc_decoders.to(device)

        # Verify all weight loading modules are initialized
        _ = model.kc_decoders.weight_loading_modules

    # Load with strict=False; some KC decoder keys may be present
    result = model.load_state_dict(state_dict, strict=False)

    # Validate that only expected extra keys are present
    # With kc_decoders initialized, those keys should load; only unexpected are errors
    unexpected = [k for k in result.unexpected_keys if not k.startswith("kc_decoders.")]
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys in state_dict (not kc_decoders): {unexpected}"
        )

    model.eval()
    return model, tokenizer


def load_default_style_model() -> Tuple[InferenceClassifier, Tokenizer]:
    """Load the default trained style classification model included in the package."""

    from kotogram import locations

    # Dev/Source mode: Check if we are running in a project with a trained model
    dev_model_dir = locations.get_style_output_dir()
    if os.path.exists(os.path.join(dev_model_dir, "model.pt")):
        return load_model(dev_model_dir)

    from importlib.resources import as_file, files

    ref = files("kotogram.model_data").joinpath("model.pt")
    with as_file(ref) as model_file:
        model_dir = os.path.dirname(model_file)
        return load_model(model_dir)


def is_default_style_model_available() -> bool:
    """Check if the default style model is available."""

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
