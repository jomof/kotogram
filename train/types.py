"""Data types for Kotogram training."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class Sample:
    """A single training sample."""

    feature_ids: Dict[str, List[int]]
    formality_value: float = 0.5
    formality_pragmatic: int = 1
    gender_value: float = 0.5
    gender_pragmatic: int = 1
    grammaticality_label: int = 1
    register_labels: List[int] = field(default_factory=lambda: [0])
    original_sentence: str = ""
    kotogram: str = ""
    kc_targets: Dict[str, Any] = field(default_factory=dict)

    @property
    def seq_len(self) -> int:
        """Get sequence length."""
        if not self.feature_ids:
            return 0
        first = next(iter(self.feature_ids.values()))
        return len(first)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_ids": self.feature_ids,
            "formality_value": self.formality_value,
            "formality_pragmatic": self.formality_pragmatic,
            "gender_value": self.gender_value,
            "gender_pragmatic": self.gender_pragmatic,
            "grammaticality_label": self.grammaticality_label,
            "register_labels": self.register_labels,
            "original_sentence": self.original_sentence,
            "kotogram": self.kotogram,
        }


@dataclass
class ProcessedSample:
    """A sample with associated metadata before indexing."""

    sentence: str
    kotogram: str
    formality_id: int
    gender_value: float
    gender_pragmatic: int
    register_ids: List[int]
    gram_label: int
    success: int
    feature_ids: Optional[Dict[str, List[int]]] = None

    def to_cache_tuple(
        self, feature_ids_override: Optional[Dict[str, List[int]]] = None
    ) -> Any:
        """Convert to cache tuple format (used by dataset cache)."""
        return (
            self.sentence,
            self.kotogram,
            self.formality_id,
            self.gender_value,
            self.gender_pragmatic,
            self.register_ids,
            self.gram_label,
            feature_ids_override
            if feature_ids_override is not None
            else self.feature_ids,
        )


@dataclass
class TrainingMetrics:
    """Accumulate training metrics for an epoch."""

    total_loss: Any = 0.0  # Can be float or Tensor
    formality_loss: Any = 0.0
    gender_loss: Any = 0.0
    grammaticality_loss: Any = 0.0
    register_loss: Any = 0.0
    count: int = 0

    def update(self, loss_dict: Dict[str, Any], count: int = 1) -> None:
        """Update metrics with batch losses."""
        # Accumulate as is (Tensor or float)
        self.total_loss += loss_dict["loss"] * count
        self.formality_loss += loss_dict["formality_loss"] * count
        self.gender_loss += loss_dict["gender_loss"] * count
        self.grammaticality_loss += loss_dict["grammaticality_loss"] * count
        self.register_loss += loss_dict["register_loss"] * count
        self.count += count

    def average(self) -> Tuple[float, float, float, float, float]:
        """Return averaged metrics as floats."""
        n = max(1, self.count)

        def _to_float(val: Any) -> float:
            if isinstance(val, torch.Tensor):
                return val.item()
            return float(val)

        return (
            _to_float(self.total_loss) / n,
            _to_float(self.formality_loss) / n,
            _to_float(self.gender_loss) / n,
            _to_float(self.grammaticality_loss) / n,
            _to_float(self.register_loss) / n,
        )

    def get_avg_loss(self) -> float:
        """Return average total loss as float."""
        val = self.total_loss
        if isinstance(val, torch.Tensor):
            val = val.item()
        return float(val) / max(1, self.count)


@dataclass
class KCMetricsAccumulator:
    """Accumulate KC probe metrics."""

    n_samples: int = 0
    sum_entropy: float = 0.0
    sum_kl: float = 0.0
    sum_tv: float = 0.0
    sum_gap: float = 0.0
    sum_avg_prob: float = 0.0
    sum_act_dens: float = 0.0
    topk_hist: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    top1_hist: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    head_samples: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class KCProbeConfig:
    """Configuration for KC probe evaluation."""

    tau_usage: float
    vocab_size: int
    topk: int
    target_specs: Dict[str, Any]
    max_samples_per_head: int
