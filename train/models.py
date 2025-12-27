"""Model extensions and heads for style training."""

from typing import Any, Dict, Optional

import torch
from torch import nn

from kotogram.model import ModelConfig, StyleClassifier


class KCDecoder(nn.Module):
    """Decoder for predicting sentence-level attributes from KC activations."""

    def __init__(self, kc_vocab_size: int, target_specs: Dict[str, int]):
        super().__init__()
        self.decoders = nn.ModuleDict()
        for name, vocab_size in target_specs.items():
            self.decoders[name] = nn.Linear(kc_vocab_size, vocab_size)

    def forward(self, kc_activations: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            name: decoder(kc_activations) for name, decoder in self.decoders.items()
        }


class StyleClassifierWithKC(StyleClassifier):
    """Multi-task style classifier with KC pretraining support."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.kc_enabled:
            self.kc_decoders = KCDecoder(config.kc_vocab_size, config.kc_target_specs)

    def forward(
        self,
        *args: Any,
        mode: str = "classification",
        **kwargs: Any,
    ) -> Any:
        if mode == "kc":
            return self.forward_kc(*args, **kwargs)
        return super().forward(*args, **kwargs)

    # pylint: disable=too-many-locals,too-many-positional-arguments
    def forward_kc(
        self,
        field_inputs: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        gumbel_scale: Optional[float] = None,
        grad_cap: Optional[float] = None,
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
        cur_temp = (
            temperature
            if temperature is not None
            else getattr(self.config, "kc_temperature", 1.0)
        )
        kc_probs = torch.sigmoid(logits_select / cur_temp)

        # Round 12: Guard against any non-finite values before topk
        kc_probs = torch.nan_to_num(kc_probs, nan=0.0, posinf=1.0, neginf=0.0)

        # Get top-k
        k = getattr(self.config, "kc_topk", 8)
        topk_vals, topk_inds = torch.topk(kc_probs, k, dim=-1)

        # Round 14: Clamp topk_vals to 0.80 (Hard ceiling on confidence)
        # This prevents deterministic "locking" where a KC gets 1.0 and stops exploring.
        topk_vals = topk_vals.clamp(max=0.80)

        # Create sparse activation (everything else zero)
        # We start with zeros and scatter the top-k values back
        sparse_activations = torch.zeros_like(kc_probs)
        sparse_activations.scatter_(1, topk_inds, topk_vals)

        target_logits = self.kc_decoders(sparse_activations)

        return {
            "kc_logits": kc_logits,
            "kc_logits_raw": kc_logits_raw,
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
