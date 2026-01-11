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


@dataclass(frozen=True)
class TensorStats:
    """Statistics for tensor finite checks."""

    finite: bool
    n_nan: int
    n_inf: int
    min: float
    max: float


@dataclass(frozen=True)
class KCStructuralBiases:
    """Accumulators for structural decoder bias initialization."""

    sums: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class FamilyAccumulator:
    """Accumulator for per-family wakefulness diagnostics.

    Semantics:
    - n_ex: total examples accumulated
    - n_pos_ex: examples with ≥1 true positive label
    - n_pos_labels: total count of true positive labels
    - sum_valid_any: examples with ≥1 supervised/eligible entry (from valid_mask)
    - sum_logit_pos/cnt_logit_pos: sum and count for logits at true positive positions
    - sum_logit_neg/cnt_logit_neg: sum and count for logits at true negative positions
    - saw_dense: saw dense target updates (source="dense")
    - saw_sparse: saw sparse sampled target updates (source="sparse")
    - saw_valid_mask: saw any valid_mask provided
    """

    n_ex: int = 0
    n_batches: int = 0
    n_pos_ex: int = 0
    n_pos_labels: int = 0
    sum_valid_any: int = 0  # renamed from sum_mask_any for clarity
    sum_logit_pos: float = 0.0
    cnt_logit_pos: int = 0
    sum_logit_neg: float = 0.0
    cnt_logit_neg: int = 0
    saw_dense: bool = False
    saw_sparse: bool = False
    saw_valid_mask: bool = False

    # pylint: disable=too-many-positional-arguments
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pos_mask: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        source: str = "dense",
    ) -> None:
        """Update stats from a batch.

        Args:
            logits: [B, K] or [B, P+N] logits tensor (detached)
            targets: [B, K] or [B, P+N] target tensor (detached)
            pos_mask: boolean mask for true positives; if None, derived from targets > 0.5
            valid_mask: boolean mask for supervised/eligible entries; if None, all valid
            source: "dense" or "sparse" to track provenance
        """
        with torch.no_grad():
            batch_size = logits.size(0)
            self.n_ex += batch_size
            self.n_batches += 1

            # Track provenance
            if source == "dense":
                self.saw_dense = True
            else:
                self.saw_sparse = True

            if valid_mask is not None:
                self.saw_valid_mask = True

            # Derive pos_mask if not provided
            if pos_mask is None:
                pos_mask = targets > 0.5

            # Positive example detection: examples with ≥1 true positive
            has_pos = pos_mask.any(dim=1)
            self.n_pos_ex += int(has_pos.sum().item())
            self.n_pos_labels += int(pos_mask.sum().item())

            # Valid/supervised coverage: examples with ≥1 eligible entry
            if valid_mask is not None:
                has_valid = valid_mask.any(dim=1)
                self.sum_valid_any += int(has_valid.sum().item())
            else:
                # No valid_mask means all entries are valid (dense or sparse sampled)
                self.sum_valid_any += batch_size

            # Compute neg_mask respecting valid_mask
            if valid_mask is not None:
                # Only count valid entries as negatives
                neg_mask = valid_mask & ~pos_mask
                # Also restrict pos to valid (should already be, but defensive)
                effective_pos_mask = valid_mask & pos_mask
            else:
                neg_mask = ~pos_mask
                effective_pos_mask = pos_mask

            # Logit stats for positives
            if effective_pos_mask.any():
                l_pos = logits[effective_pos_mask]
                self.sum_logit_pos += float(l_pos.sum().item())
                self.cnt_logit_pos += int(l_pos.numel())

            # Logit stats for negatives
            if neg_mask.any():
                l_neg = logits[neg_mask]
                self.sum_logit_neg += float(l_neg.sum().item())
                self.cnt_logit_neg += int(l_neg.numel())


@dataclass(frozen=True)
class RunningLossComponents:
    """Running loss components for an epoch."""

    struct: float = 0.0
    gap: float = 0.0
    div: float = 0.0
    lb: float = 0.0
    collapse: float = 0.0
    sparsity: float = 0.0
    saturation: float = 0.0  # Anti-saturation penalty
    formality: float = 0.0  # Prior KC cross-entropy loss (KC0-3)
    gender: float = 0.0  # Prior KC cross-entropy loss (KC4-5)
    register: float = 0.0  # Prior KC BCE multi-label loss (KC6-18)
    # Accuracy tracking for prior KCs
    formality_correct: int = 0
    formality_total: int = 0
    gender_correct: int = 0
    gender_total: int = 0
    register_correct: int = 0
    register_total: int = 0

    def add(self, other: "RunningLossComponents") -> "RunningLossComponents":
        """Add another RunningLossComponents instance."""
        return RunningLossComponents(
            struct=self.struct + other.struct,
            gap=self.gap + other.gap,
            div=self.div + other.div,
            lb=self.lb + other.lb,
            collapse=self.collapse + other.collapse,
            sparsity=self.sparsity + other.sparsity,
            saturation=self.saturation + other.saturation,
            formality=self.formality + other.formality,
            gender=self.gender + other.gender,
            register=self.register + other.register,
            formality_correct=self.formality_correct + other.formality_correct,
            formality_total=self.formality_total + other.formality_total,
            gender_correct=self.gender_correct + other.gender_correct,
            gender_total=self.gender_total + other.gender_total,
            register_correct=self.register_correct + other.register_correct,
            register_total=self.register_total + other.register_total,
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
    formality_class0_accuracy: float = 0.0  # Non-pragmatic recall
    formality_class1_accuracy: float = 0.0  # Pragmatic recall
    formality_class0_count: int = 0  # Non-pragmatic formality samples
    formality_class1_count: int = 0  # Pragmatic formality samples
    gender_accuracy: float = 0.0
    gender_mse: float = 0.0
    gender_class0_accuracy: float = 0.0  # Non-pragmatic recall
    gender_class1_accuracy: float = 0.0  # Pragmatic recall
    gender_class0_count: int = 0  # Non-pragmatic gender samples
    gender_class1_count: int = 0  # Pragmatic gender samples
    grammaticality_accuracy: float = 0.0
    gram_class0_accuracy: float = 0.0  # Agrammatical recall
    gram_class1_accuracy: float = 0.0  # Grammatical recall
    gram_class0_count: int = 0  # Number of agrammatical samples
    gram_class1_count: int = 0  # Number of grammatical samples
    register_accuracy: float = 0.0
    register_count: int = 0  # Number of register samples evaluated


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
    # New fields with defaults to preserve compatibility
    loss_mean: float = 0.0
    prob_pos_mean: float = 0.0
    prob_neg_mean: float = 0.0
    auc_proxy: float = 0.0
    fp_rate: float = 0.0
    fn_rate: float = 0.0
    support: float = 0.0
    logit_pos_mean: float = 0.0
    logit_neg_mean: float = 0.0
    delta_p: float = 0.0
    recall_01: float = 0.0
    recall_05: float = 0.0
    # Wakefulness Diagnostics
    pos_ex_frac: float = 0.0
    pos_label_density: float = 0.0
    mask_coverage: float = 0.0
    keys_present: str = ""
    # Gradient flow diagnostic
    bias_delta: float = 0.0  # Mean bias change during epoch


@dataclass
class KcDynSizingBinStats:
    """Stats for a single content-length bin."""

    bin_label: str
    count: int
    len_mean: float
    k_budget_mean: float
    k_budget_p10: float
    k_budget_p50: float
    k_budget_p90: float
    budget_ratio_mean: float
    masked_tail_rate: float
    keff_mean: float
    keff_minus_budget_mean: float
    spill_prob_mean: float = 0.0  # Mean prob of (k+1)th KC (outside budget)


@dataclass
class KcEpochActivationStats:
    """Global activation stats for an epoch."""

    pmax_global_max: float
    pmax_p50: float
    pmax_p90: float
    pmax_p99: float
    topk_sum_p50: float
    topk_sum_p90: float
    topk_sum_p99: float
    ent_norm: float
    kl_u_norm: float
    act_dens_mean: float
    kc_probs_mean: float
    # Saturation Stats (Gated & Scaled)
    sat_w: float = 0.0
    sat_alpha: float = 0.0
    sat_scale_mean: float = 0.0
    sat_contrib_mean: float = 0.0
    sat_contrib_ratio: float = 0.0
    sat_pen_global: float = 0.0
    sat_pen_pos: float = 0.0
    pmax_logit_mean_global: float = 0.0
    pmax_logit_max_global: float = 0.0
    pmax_logit_mean_pos: float = 0.0
    pmax_logit_max_pos: float = 0.0
    frac_over_thr_pos: float = 0.0
    frac_has_pos: float = 0.0


@dataclass
class KCDiagnosticReport:
    """Full KC diagnostic report for an epoch."""

    families: Dict[str, KCDiagnosticFamilyStats]  # Key is KC family name


@dataclass(frozen=True)
class KcLossWeights:
    """Weights used for each loss component (for display scaling).

    Note: All losses except struct/gap are stored ALREADY WEIGHTED in
    RunningLossComponents, so their display weight is 1.0. Full formula:
        lc.<component> * w.<component> / n_batches
    """

    struct: float = 1.0  # Raw, normalized by num_struct
    gap: float = 1.0  # Raw, normalized by num_struct
    # These are stored ALREADY WEIGHTED, so display weight is 1.0:
    div: float = 1.0  # Already weighted in RunningLossComponents
    lb: float = 1.0  # Already weighted in RunningLossComponents
    collapse: float = 1.0  # Already weighted in RunningLossComponents
    sparsity: float = 1.0  # Already weighted in RunningLossComponents


@dataclass
class KcEpochSummary:
    """Full summary package for a KC epoch."""

    epoch_idx: int
    frozen: bool
    loss_components: RunningLossComponents
    sizing_stats: List[KcDynSizingBinStats]
    activation_stats: KcEpochActivationStats
    diagnostics: KCDiagnosticReport
    weights: KcLossWeights = field(default_factory=KcLossWeights)
    n_batches: int = 1  # Number of batches (for averaging loss components)
    total_loss: float = 0.0  # Epoch total loss for validation


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


@dataclass
class TrainEpochStats:
    """Statistics collected during a training epoch."""

    avg_struct_loss: float
    num_struct_heads_processed: int
    avg_sparsity: float
    avg_prob: float
    act_dens: float

    # Optional when metrics are skipped (skip_first_metrics flag)
    kc_diagnostics: Optional[KCDiagnosticReport] = None


@dataclass
class TrainEpochResult:
    """Result of a training epoch."""

    total_loss: float
    kc_losses: KCLosses
    avg_sparsity: float
    epoch_stats: TrainEpochStats


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
    num_struct_heads_processed: List[float] = field(default_factory=list)
    avg_sparsity: List[float] = field(default_factory=list)

    active_kc_targets: List[str] = field(default_factory=list)
    # List can contain None for epochs where metrics were skipped
    kc_diagnostics: List[Optional[KCDiagnosticReport]] = field(default_factory=list)
