"""Data types for Kotogram training."""

import heapq
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
    cnt_pred_pos_on_pos: int = 0  # Predicted positive labels on positive examples
    pos_pred_hist: Dict[int, int] = field(
        default_factory=dict
    )  # Histogram of predicted positives per positive example
    saw_dense: bool = False
    saw_sparse: bool = False
    saw_valid_mask: bool = False
    loss_by_label: Optional[torch.Tensor] = None  # Running sum of loss per label
    freq_by_label: Optional[torch.Tensor] = (
        None  # Running sum of predicted-positive count per label (sigmoid > 0.5)
    )

    # pylint: disable=too-many-positional-arguments,too-many-locals
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pos_mask: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        source: str = "dense",
        loss_by_label: Optional[torch.Tensor] = None,
    ) -> None:
        """Update stats from a batch.

        Args:
            logits: [B, K] or [B, P+N] logits tensor (detached)
            targets: [B, K] or [B, P+N] target tensor (detached)
            pos_mask: boolean mask for true positives; if None, derived from targets > 0.5
            valid_mask: boolean mask for supervised/eligible entries; if None, all valid
            source: "dense" or "sparse" to track provenance
            loss_by_label: [K] tensor of summed loss per label for this batch
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

            # Predictions (logit > 0 equivalent to sigmoid > 0.5)
            # Use no_grad for stats
            pred_pos = logits > 0
            # Count predictions only on positive examples
            if has_pos.any():
                self.cnt_pred_pos_on_pos += int(pred_pos[has_pos].sum().item())
                pos_counts = pred_pos[has_pos].sum(dim=1).to(torch.int64)
                uniq, cnts = torch.unique(pos_counts, return_counts=True)
                for val, cnt in zip(uniq.tolist(), cnts.tolist()):
                    self.pos_pred_hist[val] = self.pos_pred_hist.get(val, 0) + int(cnt)

            # Compute neg_mask respecting valid_mask
            if valid_mask is not None:
                has_valid = valid_mask.any(dim=1)
                neg_mask = valid_mask & ~pos_mask
                effective_pos_mask = valid_mask & pos_mask
            else:
                has_valid = None
                neg_mask = ~pos_mask
                effective_pos_mask = pos_mask

            # Logit stats - OPTIMIZED: use multiplication to avoid intermediate tensors
            pos_mask_float = effective_pos_mask.float()
            neg_mask_float = neg_mask.float()

            # OPTIMIZED: Compute sums directly without torch.stack to avoid allocations
            self.n_pos_ex += int(has_pos.sum().item())
            self.n_pos_labels += int(pos_mask.sum().item())
            n_pos = int(pos_mask_float.sum().item())
            n_neg = int(neg_mask_float.sum().item())

            if valid_mask is not None:
                self.sum_valid_any += int(has_valid.sum().item())  # type: ignore[union-attr]
            else:
                self.sum_valid_any += batch_size

            # Logit stats for positives
            if n_pos > 0:
                self.sum_logit_pos += (logits * pos_mask_float).sum().item()
                self.cnt_logit_pos += n_pos

            # Logit stats for negatives
            if n_neg > 0:
                self.sum_logit_neg += (logits * neg_mask_float).sum().item()
                self.cnt_logit_neg += n_neg

            # Accumulate per-label loss if provided
            if loss_by_label is not None:
                if self.loss_by_label is None:
                    self.loss_by_label = loss_by_label.detach().clone()
                else:
                    self.loss_by_label += loss_by_label.detach()

            # Accumulate per-label predicted-positive count.
            # freq_by_label = count of predictions with sigmoid(logit) > 0.5,
            # matching the prior's definition: fraction of sentences the model
            # predicts as positive.  Divide by n_ex in the view to get the rate.
            pred_pos_per_label = (logits > 0).float().sum(dim=0)
            if self.freq_by_label is None:
                self.freq_by_label = pred_pos_per_label.detach().clone()
            else:
                self.freq_by_label += pred_pos_per_label.detach()

    def median_pred_pos_on_pos(self) -> Optional[float]:
        """Median predicted positives per positive example (epoch-level)."""
        if self.n_pos_ex <= 0 or not self.pos_pred_hist:
            return None
        target = (self.n_pos_ex + 1) // 2
        running = 0
        for val in sorted(self.pos_pred_hist.keys()):
            running += self.pos_pred_hist[val]
            if running >= target:
                return float(val)
        return None


@dataclass(frozen=True)
class RunningLossComponents:
    """Running loss components for an epoch."""

    struct: float = 0.0
    div: float = 0.0
    entropy: float = 0.0  # Per-probability entropy penalty
    collapse: float = 0.0
    kl_sparse: float = 0.0
    cov_penalty: float = 0.0  # Off-diagonal covariance penalty
    saturation: float = 0.0  # Anti-saturation penalty
    coverage: float = 0.0  # Coverage loss (threshold or Zipf usage fit)
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
            div=self.div + other.div,
            entropy=self.entropy + other.entropy,
            collapse=self.collapse + other.collapse,
            kl_sparse=self.kl_sparse + other.kl_sparse,
            cov_penalty=self.cov_penalty + other.cov_penalty,
            saturation=self.saturation + other.saturation,
            coverage=self.coverage + other.coverage,
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
    batch_count: int = 0
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
    accuracy: float = 0.0  # (TP + TN) / Total at threshold 0.5
    # Wakefulness Diagnostics
    pos_ex_frac: float = 0.0
    pos_label_density: float = 0.0
    mask_coverage: float = 0.0
    keys_present: str = ""
    # Gradient flow diagnostic
    bias_delta: float = 0.0  # Mean bias change during epoch


@dataclass
class KcDynSizingBinStats:
    """Stats for a single content-length bin.

    K = count of KCs with prob > 0.5 (natural threshold).
    Kth = probability of the Kth KC (last above 0.5).
    Spill = probability of the (K+1)th KC (first below 0.5).
    Gap = Kth - Spill (decision boundary sharpness).
    """

    bin_label: str
    count: int
    len_mean: float
    k_mean: float  # Mean count of KCs with prob > 0.5
    k_p10: float
    k_p50: float
    k_p90: float
    kth_prob_mean: float = 0.0  # Mean prob of Kth KC (last above 0.5), K≥1 only
    spill_prob_mean: float = 0.0  # Mean prob of (K+1)th KC (first below 0.5), K≥1 only
    gap_mean: float = (
        0.0  # Mean of (Kth - Spill), decision boundary sharpness, K≥1 only
    )
    active_pct: float = 0.0  # Fraction of sentences in bin with K≥1


@dataclass
class KcEpochActivationStats:
    """Global activation stats for an epoch."""

    pmax_global_max: float
    pmax_p50: float
    pmax_p90: float
    pmax_p99: float

    ent_norm: float
    kl_u_norm: float

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
    avg_entropy: float = 0.0  # Mean per-KC-slot Bernoulli entropy


@dataclass
class KCMseFamilyStats:
    """KC diagnostic statistics for a single MSE (regression) family."""

    loss_mean: float  # MSE loss
    discrete_accuracy: float  # Fraction matching discrete label bucket
    correlation: float  # Pearson correlation
    mean_bias: float  # Mean(pred) - Mean(target)
    pred_std: float  # Std dev of predictions
    batch_count: int = 0
    bias_delta: float = 0.0  # Decoder bias change during epoch


@dataclass
class KCDiagnosticReport:
    """Full KC diagnostic report for an epoch."""

    families: Dict[str, KCDiagnosticFamilyStats]  # Label family stats
    mse_families: Dict[str, KCMseFamilyStats] = field(default_factory=dict)


@dataclass
class WorstSampleInfo:
    """Tracks the sample with highest loss for a family during an epoch.

    Used to identify problematic samples that the model struggles with.
    """

    sentence: str  # Original sentence text (or kotogram if sentence empty)
    loss: float  # Per-sample loss value
    target: float  # Target value (for MSE) or label count (for classification)
    prediction: float  # Model's prediction
    sample_idx: int = -1  # Dataset index of this sample
    target_labels: str = ""  # Label names/IDs for classification families
    pred_labels: str = ""  # Predicted labels for classification families


class WorstSamplesTracker:
    """Tracks the top-N worst (highest loss) samples using a min-heap.

    Deduplicates by sentence: only the highest-loss entry per unique
    sentence is kept.
    """

    def __init__(self, max_size: int = 50) -> None:
        self.max_size = max_size
        self._heap: List[Tuple[float, int, WorstSampleInfo]] = []
        self._counter = 0  # Tiebreaker for heap stability
        # Maps sentence -> (loss, counter) for dedup tracking
        self._seen: Dict[str, Tuple[float, int]] = {}
        self._dirty = False  # Set when entries are logically removed

    def push(self, sample: WorstSampleInfo) -> None:
        """Add a sample; deduplicates by sentence, keeping highest loss."""
        prev = self._seen.get(sample.sentence)
        if prev is not None:
            prev_loss, _ = prev
            if sample.loss <= prev_loss:
                return  # Already have a higher-loss entry for this sentence
            # Mark old entry as stale (will be filtered on read)
            self._dirty = True

        entry = (sample.loss, self._counter, sample)
        self._counter += 1
        self._seen[sample.sentence] = (sample.loss, entry[1])

        if len(self._heap) < self.max_size + len(self._seen):
            heapq.heappush(self._heap, entry)
        elif sample.loss > self._heap[0][0]:
            heapq.heapreplace(self._heap, entry)

    def _rebuild_if_dirty(self) -> None:
        """Remove stale entries and trim to max_size."""
        if not self._dirty:
            return
        # Keep only entries whose counter matches the current best for that sentence
        live = []
        for loss, counter, info in self._heap:
            best = self._seen.get(info.sentence)
            if best is not None and best[1] == counter:
                live.append((loss, counter, info))
        heapq.heapify(live)
        # Trim to max_size (keep highest-loss entries)
        while len(live) > self.max_size:
            heapq.heappop(live)
        self._heap = live
        self._dirty = False

    def top_n(self) -> List[WorstSampleInfo]:
        """Return all tracked samples sorted by loss descending."""
        self._rebuild_if_dirty()
        return [s for _, _, s in sorted(self._heap, reverse=True)]

    def worst(self) -> Optional[WorstSampleInfo]:
        """Return the single worst (highest loss) sample, or None."""
        self._rebuild_if_dirty()
        if not self._heap:
            return None
        return max(self._heap)[2]


@dataclass(frozen=True)
class KcLossWeights:
    """Weights used for each loss component (for display scaling).

    All loss components are stored as raw sums in RunningLossComponents.
    Display formula: lc.<component> * w.<component> / n_batches

    INVARIANTS (enforced by checksums):
    1. struct = sum(all family losses) - each family contributes its task_loss directly
    2. total_loss = struct + div + entropy + collapse + kl_sparse + cov_penalty + saturation + coverage
    """

    struct: float = 1.0  # Sum of all family task_losses
    # These are stored ALREADY WEIGHTED, so display weight is 1.0:
    div: float = 1.0  # Already weighted in RunningLossComponents
    entropy: float = 1.0  # Already weighted in RunningLossComponents
    collapse: float = 1.0  # Already weighted in RunningLossComponents
    kl_sparse: float = 1.0  # Already weighted in RunningLossComponents
    cov_penalty: float = 1.0  # Already weighted in RunningLossComponents
    coverage: float = 1.0  # Already weighted in RunningLossComponents


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
    kc_logits_used_count: int = 0  # Number of unique KC logits that fired
    kc_logits_used_percent: float = 0.0  # Percent of KC logits utilized
    zipf_kl: float = 0.0  # Epoch usage vs Zipf KL (lower is closer)
    worst_samples: Dict[str, "WorstSamplesTracker"] = field(
        default_factory=dict
    )  # Per-family top-N worst samples
    accumulators: Dict[str, "FamilyAccumulator"] = field(default_factory=dict)
    # Optional per-GP priors vector used for printing curate hints (NaN => unset/default).
    gp_priors: Optional[torch.Tensor] = None
    gp_default_prior: float = 1e-8
    total_samples: int = 0  # Total examples processed in epoch (for frequency calc)
    # Per-bin canary sentence evaluation text (bin_label -> summary string)
    canary_texts: Dict[str, str] = field(default_factory=dict)
    kc_threshold: float = 0.5  # Adaptive KC threshold for this epoch
    layer_health: Optional["LayerHealthStats"] = None
    # Populated by view: Dead KCs line (alive = KCs that have been both above/below 0.5)
    alive_kcs: Optional[int] = None
    # Populated by view: Total row K@threshold stats (mean and percentiles of KCs fired per sample)
    total_k_mean: Optional[float] = None
    total_k_p10: Optional[float] = None
    total_k_p50: Optional[float] = None
    total_k_p90: Optional[float] = None


@dataclass
class LayerHealthStats:
    """Per-layer health diagnostics for the transformer encoder.

    Adapts automatically to any number of layers.
    """

    delta_norm: List[float]  # ||output - input|| / ||input|| per layer
    cka_adjacent: List[float]  # CKA(layer_i, layer_{i+1}), length = num_layers - 1
    effective_rank: List[float]  # Effective rank (90% variance) per layer
    num_layers: int


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
    reg_p: List[List[int]]  # Multi-label: [B, num_classes]
    reg_l: List[List[int]]  # Multi-label: [B, num_classes]
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
    avg_kl_sparse: float
    avg_prob: float

    # Optional when metrics are skipped (skip_first_metrics flag)
    kc_diagnostics: Optional[KCDiagnosticReport] = None


@dataclass
class TrainEpochResult:
    """Result of a training epoch."""

    total_loss: float
    kc_losses: KCLosses
    avg_kl_sparse: float
    epoch_stats: TrainEpochStats
    # Sizing metrics from view (alive_kcs, total_k_mean, total_k_p10/p50/p90), None when skipped
    sizing_metrics: Optional[Dict[str, Any]] = None


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
    kc_kl_sparse: List[float] = field(default_factory=list)
    kc_losses: Dict[str, List[float]] = field(default_factory=dict)
    avg_struct_loss: List[float] = field(default_factory=list)
    num_struct_heads_processed: List[float] = field(default_factory=list)
    avg_kl_sparse: List[float] = field(default_factory=list)

    active_kc_targets: List[str] = field(default_factory=list)
    # List can contain None for epochs where metrics were skipped
    kc_diagnostics: List[Optional[KCDiagnosticReport]] = field(default_factory=list)
    # Sizing metrics (alive KCs, Total K mean/p10/p50/p90, kc_threshold); None when metrics skipped
    alive_kcs: List[Optional[int]] = field(default_factory=list)
    total_k_mean: List[Optional[float]] = field(default_factory=list)
    total_k_p10: List[Optional[float]] = field(default_factory=list)
    total_k_p50: List[Optional[float]] = field(default_factory=list)
    total_k_p90: List[Optional[float]] = field(default_factory=list)
    kc_threshold: List[Optional[float]] = field(default_factory=list)
