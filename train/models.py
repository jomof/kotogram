"""Model extensions and heads for style training."""

from typing import Any, Dict, Optional

import torch
from torch import nn

from kotogram.model import InferenceClassifier, ModelConfig
from train.kc import KC_FAMILIES, KcFamilyId


class KCDecoder(nn.Module):
    """Decoder for predicting sentence-level attributes from KC activations.

    Architecture (Separated Pathways):
    - For MSE families (gender/formality - style features):
      - Dedicated hidden layer 1: kc_vocab_size -> hidden_dim (default 256)
      - ReLU activation
      - Dedicated hidden layer 2: hidden_dim -> hidden_dim
      - ReLU activation
      - Per-family output heads: hidden_dim -> 1
      - Tanh activation (bounded to [-1, 1])

    - For label families (structural features - grammar_point, n-grams, etc.):
      - Dedicated hidden layer 1: kc_vocab_size -> hidden_dim (default 256)
      - ReLU activation
      - Dedicated hidden layer 2: hidden_dim -> hidden_dim
      - ReLU activation
      - Per-family output heads: hidden_dim -> vocab_size

    Rationale: Style features (continuous regression) and structural features
    (multi-label classification) benefit from separate representation pathways.
    """

    def __init__(
        self,
        kc_vocab_size: int,
        target_specs: Dict[KcFamilyId, int],
        hidden_dim: int = 256,
        pooled_dim: int = 512,
    ):
        super().__init__()
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()

        # Derive MSE families from registry
        from train.kc import KcBertFamily, KcMseFamily

        self._mse_families = frozenset(
            fid.name.lower()
            for fid, fam in KC_FAMILIES.items()
            if isinstance(fam, KcMseFamily) and fid in target_specs
        )

        self._bert_families = frozenset(
            fid.name.lower()
            for fid, fam in KC_FAMILIES.items()
            if isinstance(fam, KcBertFamily) and fid in target_specs
        )

        # Separate hidden layers for MSE families (style features)
        self.mse_hidden1 = nn.Linear(kc_vocab_size, hidden_dim)
        self.mse_hidden2 = nn.Linear(hidden_dim, hidden_dim)

        # Separate hidden layers for label families (structural features)
        self.label_hidden1 = nn.Linear(kc_vocab_size, hidden_dim)
        self.label_hidden2 = nn.Linear(hidden_dim, hidden_dim)

        # BERT cloze reads from pooler output (pre-KC), not KC logits
        self.bert_hidden1 = nn.Linear(pooled_dim, hidden_dim)
        self.bert_hidden2 = nn.Linear(hidden_dim, hidden_dim)

        # Per-family output heads
        self.decoders = nn.ModuleDict()
        self.mse_decoders = nn.ModuleDict()
        self.bert_decoders = nn.ModuleDict()

        for fid, vocab_size in target_specs.items():
            name = fid.name.lower()
            if name in self._mse_families:
                # MSE families: use MSE hidden pathway → output → Tanh
                self.mse_decoders[name] = nn.Linear(hidden_dim, vocab_size)
            elif name in self._bert_families:
                # BERT cloze: kc_probs → hidden → surface_vocab_size
                self.bert_decoders[name] = nn.Linear(hidden_dim, vocab_size)
            else:
                # Label families: use label hidden pathway → output
                self.decoders[name] = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        kc_probs: torch.Tensor,
        pooled: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Decode KC probabilities (and pooled output) to family outputs.

        Label/MSE families receive KC probabilities. BERT cloze receives the
        pooler output directly, bypassing the KC bottleneck so it acts as an
        encoder regularizer rather than a KC signal.

        Args:
            kc_probs: Full KC probabilities [B, kc_vocab_size]
            pooled: Attention pooler output [B, d_model] (required for BERT cloze)

        Returns:
            Dict mapping family name to output tensor.
        """
        result = {}

        # MSE pathway: Process style features (gender, formality)
        if self.mse_decoders:
            h_mse = self.activation(self.mse_hidden1(kc_probs))
            h_mse = self.activation(self.mse_hidden2(h_mse))

            for name, decoder in self.mse_decoders.items():
                out = decoder(h_mse)
                out = self.tanh(out)  # Bound to [-1, 1]
                result[name] = out

        # Label pathway: All families use full KC probs
        if self.decoders:
            h_label = self.activation(self.label_hidden1(kc_probs))
            h_label = self.activation(self.label_hidden2(h_label))

            for name, decoder in self.decoders.items():
                result[name] = decoder(h_label)

        # BERT cloze pathway: reads from pooler, not KC logits
        if self.bert_decoders and pooled is not None:
            h_bert = self.activation(self.bert_hidden1(pooled))
            h_bert = self.activation(self.bert_hidden2(h_bert))

            for name, decoder in self.bert_decoders.items():
                result[name] = decoder(h_bert)

        return result


class TrainingClassifier(InferenceClassifier):
    """Multi-task style classifier with KC pretraining support."""

    # Override type from base class - training uses KCDecoder, inference uses KCDecoderInference
    kc_decoders: KCDecoder  # type: ignore[assignment]

    def __init__(
        self,
        config: ModelConfig,
        kc_target_specs: Optional[Dict[KcFamilyId, int]] = None,
    ):
        super().__init__(config)
        # KC is always enabled; kc_target_specs defines decoders for pretraining

        if kc_target_specs is None:
            kc_target_specs = {}

        self.kc_decoders = KCDecoder(
            config.kc_vocab_size, kc_target_specs, pooled_dim=config.d_model
        )

    def forward(
        self,
        *args: Any,
        mode: str = "classification",
        **kwargs: Any,
    ) -> Any:
        if mode == "kc":
            return self.forward_kc(*args, **kwargs)
        return super().forward(*args, **kwargs)

    # pylint: disable=too-many-locals,too-many-positional-arguments,too-many-arguments
    def forward_kc(
        self,
        field_inputs: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        temperature: Optional[float] = None,
        gumbel_scale: Optional[float] = None,
        grad_cap: Optional[float] = None,
        pooled: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        # Use unified pooler for KC (shared with style classifier)
        if pooled is None:
            encoder_output = self.get_encoder_output(field_inputs, attention_mask)
            pooled = self.pooler(encoder_output, attention_mask)

        # Get raw and normalized logits
        if hasattr(self.kc_head, "forward_with_raw"):
            kc_logits_raw, kc_logits = self.kc_head.forward_with_raw(pooled)
        else:
            kc_logits = self.kc_head(pooled)
            kc_logits_raw = kc_logits

        # Apply Gumbel Noise (Training Only)
        # Noisy logits for exploration, clean logits for regularization
        if gumbel_scale is not None and gumbel_scale > 0 and self.training:
            # Round 12: Clamp u away from {0,1} to prevent inf/nan in log
            u = torch.rand_like(kc_logits_raw).clamp_(1e-6, 1 - 1e-6)
            g = -torch.log(-torch.log(u))
            logits_select = kc_logits_raw + gumbel_scale * g
        else:
            logits_select = kc_logits_raw

        # Round 14: Gradient capping via hook on primary logits Path
        if self.training and grad_cap is not None:
            if kc_logits_raw.requires_grad:
                kc_logits_raw.register_hook(
                    lambda grad: grad.clamp(min=-grad_cap, max=grad_cap)
                )

        # Round 14: Split clamping logic (Stability Hardening)
        # 1. Path for Selection/Probabilities (Sigmoid)
        # We want to prevent sigmoid saturation and large gradients.
        logits_select = logits_select.clamp(min=-12.0, max=12.0)

        # 2. Path for Diversity Regularizer (Usage)
        # We want to prevent rare large logits from dominating the softmax mean.
        # Note: this is stored in outputs now, to be retrieved in train_epoch.
        logits_usage = kc_logits_raw.clamp(min=-8.0, max=8.0)

        # Compute probs from (possibly noisy) logits
        cur_temp = float(
            temperature
            if temperature is not None
            else getattr(self.config, "kc_temperature", 1.0)
        )
        kc_logits_effective = logits_select / cur_temp
        kc_probs = torch.sigmoid(kc_logits_effective)

        # Clean probabilities (no Gumbel noise) for diagnostics.
        # Training loss uses noisy kc_probs; Bin report uses these.
        kc_probs_clean = torch.sigmoid(
            kc_logits_raw.clamp(min=-12.0, max=12.0) / cur_temp
        )

        # Round 12: Guard against any non-finite values
        kc_probs = torch.nan_to_num(kc_probs, nan=0.0, posinf=1.0, neginf=0.0)
        kc_probs_clean = torch.nan_to_num(
            kc_probs_clean, nan=0.0, posinf=1.0, neginf=0.0
        )

        target_logits = self.kc_decoders(kc_probs, pooled=pooled)

        return {
            "kc_logits": kc_logits,
            "kc_logits_raw": kc_logits_raw,
            "kc_logits_effective": kc_logits_effective,
            "logits_usage": logits_usage,  # Round 14: Passed for use in train_epoch
            "kc_probs": kc_probs,
            "kc_probs_clean": kc_probs_clean,
            "target_logits": target_logits,
        }
