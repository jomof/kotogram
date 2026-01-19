"""Model extensions and heads for style training."""

from typing import Any, Dict, Optional

import torch
from torch import nn

from kotogram.model import InferenceClassifier, ModelConfig
from train.kc import KC_FAMILIES, KcFamilyId, KcLogitMode


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
    ):
        super().__init__()
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()

        # Derive MSE families from registry
        from train.kc import KcMseFamily

        self._mse_families = frozenset(
            fid.name.lower()
            for fid, fam in KC_FAMILIES.items()
            if isinstance(fam, KcMseFamily) and fid in target_specs
        )

        # Derive ALL_LOGITS families from registry (use full KC probs, not sparse)
        self._all_logits_families = frozenset(
            fid.name.lower()
            for fid, fam in KC_FAMILIES.items()
            if fam.logit_mode == KcLogitMode.ALL_LOGITS and fid in target_specs
        )

        # Derive HOT_LOGITS families from registry (use only prob >= 0.5)
        self._hot_logits_families = frozenset(
            fid.name.lower()
            for fid, fam in KC_FAMILIES.items()
            if fam.logit_mode == KcLogitMode.HOT_LOGITS and fid in target_specs
        )

        # Derive SPARSE_LOGITS families from registry (use sparse top-k activations)
        self._sparse_logits_families = frozenset(
            fid.name.lower()
            for fid, fam in KC_FAMILIES.items()
            if fam.logit_mode == KcLogitMode.SPARSE_LOGITS and fid in target_specs
        )

        # Separate hidden layers for MSE families (style features)
        self.mse_hidden1 = nn.Linear(kc_vocab_size, hidden_dim)
        self.mse_hidden2 = nn.Linear(hidden_dim, hidden_dim)

        # Separate hidden layers for label families (structural features)
        self.label_hidden1 = nn.Linear(kc_vocab_size, hidden_dim)
        self.label_hidden2 = nn.Linear(hidden_dim, hidden_dim)

        # Per-family output heads
        self.decoders = nn.ModuleDict()
        self.mse_decoders = nn.ModuleDict()

        for fid, vocab_size in target_specs.items():
            name = fid.name.lower()
            if name in self._mse_families:
                # MSE families: use MSE hidden pathway → output → Tanh
                self.mse_decoders[name] = nn.Linear(hidden_dim, vocab_size)
            else:
                # Label families: use label hidden pathway → output
                self.decoders[name] = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        kc_activations: torch.Tensor,
        kc_probs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Decode KC activations to family outputs.

        Args:
            kc_activations: Sparse KC activations [B, kc_vocab_size] (top-k non-zero)
            kc_probs: Full KC probabilities [B, kc_vocab_size] (all values)

        Returns:
            Dict mapping family name to output tensor.

        Strategy (driven by KcLogitMode):
            - ALL_LOGITS families: Use kc_probs (full distribution)
            - HOT_LOGITS families: Use kc_probs masked to only values >= 0.5
            - SPARSE_LOGITS families: Use kc_activations (sparse top-k)
        """
        result = {}

        # Compute hot_probs lazily (only if needed)
        hot_probs = None

        # MSE pathway: Process style features (gender, formality)
        # Uses FULL KC probabilities (not sparse top-k) since style is diffuse
        if self.mse_decoders:
            h_mse = self.activation(self.mse_hidden1(kc_probs))
            h_mse = self.activation(self.mse_hidden2(h_mse))

            for name, decoder in self.mse_decoders.items():
                out = decoder(h_mse)
                out = self.tanh(out)  # Bound to [-1, 1]
                result[name] = out

        # Label pathway: Selection driven by logit_mode from family registry
        if self.decoders:
            for name, decoder in self.decoders.items():
                if name in self._all_logits_families:
                    # ALL_LOGITS: use full KC probs (diffuse signal)
                    h_label = self.activation(self.label_hidden1(kc_probs))
                    h_label = self.activation(self.label_hidden2(h_label))
                elif name in self._hot_logits_families:
                    # HOT_LOGITS: use only probs >= 0.5 (thresholded)
                    if hot_probs is None:
                        hot_probs = kc_probs * (kc_probs >= 0.5).float()
                    h_label = self.activation(self.label_hidden1(hot_probs))
                    h_label = self.activation(self.label_hidden2(h_label))
                elif name in self._sparse_logits_families:
                    # SPARSE_LOGITS: use sparse activations (localized signal)
                    h_label = self.activation(self.label_hidden1(kc_activations))
                    h_label = self.activation(self.label_hidden2(h_label))
                else:
                    raise ValueError(f"Unknown logit_mode for family '{name}'")

                result[name] = decoder(h_label)

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

        self.kc_decoders = KCDecoder(config.kc_vocab_size, kc_target_specs)

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
        k_budget: Optional[torch.Tensor] = None,
        long_sentence_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        # Use unified pooler for KC (shared with style classifier)
        encoder_output = self.get_encoder_output(field_inputs, attention_mask)
        pooled = self.pooler(encoder_output, attention_mask)

        # Get raw and normalized logits
        if hasattr(self.kc_head, "forward_with_raw"):
            kc_logits_raw, kc_logits = self.kc_head.forward_with_raw(pooled)
        else:
            kc_logits = self.kc_head(pooled)
            kc_logits_raw = kc_logits

        # Apply Gumbel Noise for Top-K Selection (Training Only)
        # We use noisy logits for selection, but return clean logits for regularization
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

        # Long-Sentence Coverage Protection (Training Only)
        if (
            self.training
            and long_sentence_mask is not None
            and long_sentence_mask.any()
        ):
            if long_sentence_mask.device != kc_probs.device:
                long_sentence_mask = long_sentence_mask.to(kc_probs.device)

            # Check top-1 dominance on long sentences
            # Check max prob
            max_probs = kc_probs.max(dim=-1)[0]
            violation_mask = (max_probs > 0.85) & long_sentence_mask

            if violation_mask.any():
                # Re-normalize with higher temperature for these rows
                # Factor 1.5x temperature boost for violated rows
                boosted_temp = cur_temp * 1.5
                improved_logits = logits_select[violation_mask]
                kc_logits_effective = kc_logits_effective.clone()
                kc_logits_effective[violation_mask] = improved_logits / boosted_temp
                kc_probs = kc_probs.clone()
                kc_probs[violation_mask] = torch.sigmoid(
                    kc_logits_effective[violation_mask]
                )

        # Round 12: Guard against any non-finite values before topk
        kc_probs = torch.nan_to_num(kc_probs, nan=0.0, posinf=1.0, neginf=0.0)

        # Determine K
        if k_budget is not None:
            # Variable k: take max required k for the batch
            k = int(k_budget.max().item())
        else:
            k = getattr(self.config, "kc_topk", 8)

        # Get top-k
        topk_vals, topk_inds = torch.topk(kc_probs, k, dim=-1)

        # Apply per-sample budget masking if variable k
        if k_budget is not None:
            # Create mask: (B, K)
            col_indices = torch.arange(k, device=topk_vals.device).unsqueeze(0)
            budget_mask = col_indices < k_budget.unsqueeze(1)

            # Zero out values beyond budget
            topk_vals = topk_vals * budget_mask.float()
            # Note: topk_inds beyond budget are technically invalid/pad.

        # Removed: topk_vals.clamp(max=0.80) to separate presence from strength
        # and allow natural activation range.

        # Create sparse activation (everything else zero)
        # We start with zeros and scatter the top-k values back
        sparse_activations = torch.zeros_like(kc_probs)
        sparse_activations.scatter_(1, topk_inds, topk_vals)

        target_logits = self.kc_decoders(sparse_activations, kc_probs)

        return {
            "kc_logits": kc_logits,
            "kc_logits_raw": kc_logits_raw,
            "kc_logits_effective": kc_logits_effective,
            "logits_usage": logits_usage,  # Round 14: Passed for use in train_epoch
            "kc_probs": kc_probs,
            "sparse_activations": sparse_activations,
            "topk_vals": topk_vals,
            "topk_inds": topk_inds,
            "target_logits": target_logits,
        }
