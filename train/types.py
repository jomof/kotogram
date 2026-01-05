"""Data types for Kotogram training."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from train.kc import KcFamilyId


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
    kc_targets: Dict[KcFamilyId, Any] = field(default_factory=dict)
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
class KCDiagnosticHeadStats:
    """Accumulator for KC head statistics."""

    pos_logits: List[float] = field(default_factory=list)
    neg_logits: List[float] = field(default_factory=list)
    p_sum: float = 0.0
    count: int = 0


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
    head_samples: Dict[KcFamilyId, KCDiagnosticHeadStats] = field(default_factory=dict)


@dataclass
class KCProbeConfig:
    """Configuration for KC probe evaluation."""

    tau_usage: float
    vocab_size: int
    topk: int
    target_specs: Dict[KcFamilyId, int]
    max_samples_per_head: int


@dataclass(frozen=True)
class TensorStats:
    """Statistics for tensor finite checks."""

    finite: bool
    n_nan: int
    n_inf: int
    min: float
    max: float


@dataclass(frozen=True)
class KCSnapshot:
    """Snapshot of KC model state for restoration."""

    kc_head: Dict[str, torch.Tensor]
    kc_decoders: Optional[Dict[str, torch.Tensor]] = None


@dataclass(frozen=True)
class KCCoverageCounts:
    """Counts for KC target coverage checks."""

    dense: int = 0
    sparse: int = 0
    label: int = 0
    missing: int = 0


@dataclass(frozen=True)
class KCStructuralBiases:
    """Accumulators for structural decoder bias initialization."""

    sums: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RunningLossComponents:
    """Running loss components for an epoch."""

    base: float = 0.0
    struct: float = 0.0
    label: float = 0.0
    div: float = 0.0
    lb: float = 0.0
    collapse: float = 0.0
    sparsity: float = 0.0

    def add(self, other: "RunningLossComponents") -> "RunningLossComponents":
        """Add another RunningLossComponents instance."""
        return RunningLossComponents(
            base=self.base + other.base,
            struct=self.struct + other.struct,
            label=self.label + other.label,
            div=self.div + other.div,
            lb=self.lb + other.lb,
            collapse=self.collapse + other.collapse,
            sparsity=self.sparsity + other.sparsity,
        )


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

    families: Dict[str, KCDiagnosticFamilyStats]  # Key is KC family name


@dataclass
class TrainingPredictions:
    """Predictions extracted from a training batch."""

    f_prag_p: List[int]
    f_prag_l: List[int]
    f_val_p: List[float]
    f_val_l: List[float]
    g_prag_p: List[int]
    g_prag_l: List[int]
    g_val_p: List[float]
    g_val_l: List[float]
    gram_p: List[int]
    gram_l: List[int]
    reg_p: List[int]
    reg_l: List[int]
    is_valid: List[bool]


@dataclass(frozen=True)
class KCLosses:
    """Immutable accumulator for KC losses."""

    _losses: Dict[str, float] = field(default_factory=dict)

    @property
    def losses(self) -> Dict[str, float]:
        """Return a copy of the losses."""
        return dict(self._losses)

    def add(self, key: str, value: float) -> "KCLosses":
        """Add a loss value to the accumulator."""
        new_losses = dict(self._losses)
        new_losses[key] = new_losses.get(key, 0.0) + value
        return KCLosses(_losses=new_losses)

    def get(self, key: str, default: float = 0.0) -> float:
        """Get a loss value."""
        return self._losses.get(key, default)

    def items(self) -> Any:
        """Return items iterator."""
        return self._losses.items()

    def keys(self) -> Any:
        """Return keys iterator."""
        return self._losses.keys()


@dataclass(frozen=True)
class FirstBatchSeparation:
    """Immutable accumulator for first batch separation."""

    _data: Dict[str, float] = field(default_factory=dict)

    @property
    def data(self) -> Dict[str, float]:
        """Return a copy of the data."""
        return dict(self._data)

    def with_entry(self, key: str, value: float) -> "FirstBatchSeparation":
        """Add an entry."""
        new_data = dict(self._data)
        new_data[key] = value
        return FirstBatchSeparation(_data=new_data)


@dataclass(frozen=True)
class FirstBatchGradNorms:
    """Immutable accumulator for first batch gradient norms."""

    _data: Dict[str, float] = field(default_factory=dict)

    @property
    def data(self) -> Dict[str, float]:
        """Return a copy of the data."""
        return dict(self._data)

    def with_entry(self, key: str, value: float) -> "FirstBatchGradNorms":
        """Add an entry."""
        new_data = dict(self._data)
        new_data[key] = value
        return FirstBatchGradNorms(_data=new_data)


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
    first_batch_separation: FirstBatchSeparation
    first_batch_grad_norms: FirstBatchGradNorms
    avg_entropy_norm: float
    avg_logit_gap: float
    avg_kl_to_uniform: float
    uniq_kcs_epoch: int
    avg_pmax_mean: float  # Mean of max(kc_probs) per example
    kc_diagnostics: KCDiagnosticReport


@dataclass
class TrainEpochResult:
    """Result of a training epoch."""

    total_loss: float
    kc_losses: KCLosses
    avg_sparsity: float
    epoch_stats: TrainEpochStats


@dataclass
class KCProbeEvaluationResult:
    """Result of KC probe evaluation."""

    n_samples: int
    uniq_kcs: int
    max_top1: float
    entropy_norm: float
    kl_to_uniform: float
    tv_mean: float
    gap_mean: float
    avg_prob: float
    act_dens: float
    kc_vocab_size: int
    head_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingBatch:
    """Typed batch for training."""

    feature_inputs: Dict[str, torch.Tensor]  # e.g. "kanji" -> tensor
    attention_mask: torch.Tensor
    formality_value: torch.Tensor
    formality_pragmatic: torch.Tensor
    gender_value: torch.Tensor
    gender_pragmatic: torch.Tensor
    grammaticality_labels: torch.Tensor
    register_labels: torch.Tensor
    indices: torch.Tensor
    original_sentence: List[str]
    kotogram: List[str]
    kc_targets: List[Dict[KcFamilyId, Any]] = field(default_factory=list)


@dataclass
class TrainingLosses:
    """Typed losses for training step."""

    loss: torch.Tensor
    f_loss: torch.Tensor
    g_loss: torch.Tensor
    gram_loss: torch.Tensor
    reg_loss: torch.Tensor


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
    active_kc_targets: List[str] = field(default_factory=list)
    kc_diagnostics: List[KCDiagnosticReport] = field(default_factory=list)
