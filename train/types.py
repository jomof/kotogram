"""Data types for Kotogram training."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch


@dataclass
class Sample:
    """A single training sample."""

    feature_ids: Dict[str, Any]  # List[int] or torch.Tensor
    formality_value: float = 0.5
    formality_pragmatic: int = 1
    gender_value: float = 0.5
    gender_pragmatic: int = 1
    grammaticality_label: int = 1
    register_labels: List[int] = field(default_factory=lambda: [0])
    original_sentence: str = ""
    kotogram: str = ""
    kc_targets: Dict[str, Any] = field(default_factory=dict)
    idx: int = -1

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


@dataclass
class EvaluationMetrics:
    """Evaluation results for a validation pass."""

    loss: float = 0.0
    formality_loss: float = 0.0
    gender_loss: float = 0.0
    grammaticality_loss: float = 0.0
    register_loss: float = 0.0
    formality_accuracy: float = 0.0
    formality_mse: float = 0.0
    gender_accuracy: float = 0.0
    gender_mse: float = 0.0
    grammaticality_accuracy: float = 0.0
    register_accuracy: float = 0.0


@dataclass
class TrainingHistory:
    """Accumulated training history."""

    train_loss: List[float] = field(default_factory=list)
    train_formality_loss: List[float] = field(default_factory=list)
    train_gender_loss: List[float] = field(default_factory=list)
    train_grammaticality_loss: List[float] = field(default_factory=list)
    train_register_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_formality_loss: List[float] = field(default_factory=list)
    val_gender_loss: List[float] = field(default_factory=list)
    val_grammaticality_loss: List[float] = field(default_factory=list)
    val_register_loss: List[float] = field(default_factory=list)
    val_formality_accuracy: List[float] = field(default_factory=list)
    val_formality_mse: List[float] = field(default_factory=list)
    val_gender_pragmatic_accuracy: List[float] = field(default_factory=list)
    val_gender_value_mse: List[float] = field(default_factory=list)
    val_grammaticality_accuracy: List[float] = field(default_factory=list)
    val_register_accuracy: List[float] = field(default_factory=list)
    sentence_count: List[int] = field(default_factory=list)


@dataclass
class KCDiagnosticFamilyStats:
    """KC diagnostic statistics for a single family."""

    rate: float
    p50: int
    p90: int
    empty_pct: float
    dnll: float
    mask_pct: float


@dataclass
class KCDiagnosticReport:
    """Full KC diagnostic report for an epoch."""

    families: Dict[str, KCDiagnosticFamilyStats]  # UNDONE: Comment on what's in the key


@dataclass
class TrainEpochStats:
    """Statistics collected during a training epoch."""

    avg_struct_loss: float
    avg_label_loss: float
    num_struct_heads_processed: int
    num_label_heads_processed: int
    avg_sparsity: float
    avg_prob: float
    act_dens: float
    first_batch_separation: Dict[str, float]
    first_batch_grad_norms: Dict[str, float]
    avg_entropy_norm: float
    avg_kl_to_uniform: float
    uniq_kcs_epoch: int
    avg_p_max: float
    kc_diagnostics: KCDiagnosticReport


@dataclass
class TrainEpochResult:
    """Result of a training epoch."""

    total_loss: float
    kc_losses: Dict[str, float]  # Key is KC head name
    avg_sparsity: float
    epoch_stats: TrainEpochStats


@dataclass
class KCTrainingHistory(TrainingHistory):
    """Accumulated training history for KC training."""

    total_loss: List[float] = field(default_factory=list)
    kc_sparsity: List[float] = field(default_factory=list)
    kc_losses: Dict[str, List[float]] = field(default_factory=dict)
    avg_struct_loss: List[float] = field(default_factory=list)
    avg_label_loss: List[float] = field(default_factory=list)
    num_struct_heads_processed: List[float] = field(default_factory=list)
    num_label_heads_processed: List[float] = field(default_factory=list)
    avg_sparsity: List[float] = field(default_factory=list)
    first_batch_separation: List[Dict[str, float]] = field(default_factory=list)
    first_batch_grad_norms: List[Dict[str, float]] = field(default_factory=list)
    kc_diagnostics: List[KCDiagnosticReport] = field(default_factory=list)
