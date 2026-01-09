"""Model extensions and heads for style training."""

from typing import Any, Dict, Optional

import torch
from torch import nn

from kotogram.model import ModelConfig, StyleClassifier
from train.kc import KcFamilyId


class KCDecoder(nn.Module):
    """Decoder for predicting sentence-level attributes from KC activations.

    Architecture:
    - Shared hidden layer 1: kc_vocab_size -> hidden_dim (default 256)
    - ReLU activation
    - Shared hidden layer 2: hidden_dim -> hidden_dim
    - ReLU activation
    - Per-family output heads: hidden_dim -> vocab_size
    """

    def __init__(
        self,
        kc_vocab_size: int,
        target_specs: Dict[KcFamilyId, int],
        hidden_dim: int = 256,
    ):
        super().__init__()
        # Shared hidden layers for all families
        self.hidden1 = nn.Linear(kc_vocab_size, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()

        # Per-family output heads from shared hidden
        self.decoders = nn.ModuleDict()
        for fid, vocab_size in target_specs.items():
            # nn.ModuleDict requires string keys
            self.decoders[fid.name.lower()] = nn.Linear(hidden_dim, vocab_size)

    def forward(self, kc_activations: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.activation(self.hidden1(kc_activations))
        h = self.activation(self.hidden2(h))
        return {name: decoder(h) for name, decoder in self.decoders.items()}


class StyleClassifierWithKC(StyleClassifier):
    """Multi-task style classifier with KC pretraining support."""

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
        pooled = self._get_pooled_output(field_inputs, attention_mask)

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

        target_logits = self.kc_decoders(sparse_activations)

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

    def reset_classifier(self) -> None:
        """Reinitialize all classifier head weights."""
        for classifier in [
            self.formality_value_head,
            self.formality_pragmatic_head,
            self.gender_value_head,
            self.gender_pragmatic_head,
            self.grammaticality_classifier,
            self.register_classifier,
        ]:
            if isinstance(classifier, nn.Module):
                for module in classifier.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
