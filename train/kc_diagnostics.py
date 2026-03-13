"""Diagnostic tools for Knowledge Component (KC) training."""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch

from kotogram.constants import (
    FormalityThresholds,
    GenderThresholds,
    GrammaticalityThresholds,
)
from train.types import (
    KCBertFamilyStats,
    KCDiagnosticFamilyStats,
    KCDiagnosticReport,
    KCMseFamilyStats,
)


def discretize_mse(values: torch.Tensor, family_name: str) -> torch.Tensor:
    """Map continuous values to discrete bucket IDs matching inference thresholds.

    Uses the same thresholds as kotogram.constants so that epoch-report accuracy
    reflects what ``bin/kotogram`` would actually print.
    """
    if family_name == "formality":
        # 5 levels: very_casual < casual < neutral < formal < very_formal
        thresholds = torch.tensor(
            [
                FormalityThresholds.CASUAL_MIN,  # -0.75
                FormalityThresholds.NEUTRAL_MIN,  # -0.25
                FormalityThresholds.FORMAL_MIN,  # 0.25
                FormalityThresholds.VERY_FORMAL_MIN,  # 0.75
            ],
            device=values.device,
        )
    elif family_name == "gender":
        # 3 levels: masculine < neutral < feminine
        thresholds = torch.tensor(
            [
                GenderThresholds.MASCULINE_MAX,  # -0.5
                GenderThresholds.FEMININE_MIN,  # 0.5
            ],
            device=values.device,
        )
    elif family_name == "grammatic":
        # 2 levels: ungrammatical vs grammatical
        thresholds = torch.tensor(
            [GrammaticalityThresholds.GRAMMATIC_MIN],  # 0.5
            device=values.device,
        )
    else:
        # Unknown family: fall back to single bucket (always "correct")
        return torch.zeros_like(values, dtype=torch.long)

    # bucketize: value < t[0] → 0, t[0] <= value < t[1] → 1, etc.
    return torch.bucketize(values.contiguous(), thresholds)


# Pylint suppressions for diagnostic complexity
# pylint: disable=too-many-positional-arguments,too-many-locals,unused-argument,too-many-return-statements

# --- Types ---


# --- Canonical Gathering ---


# --- Existing Statistic Classes ---


@dataclass
class FamilyStats:
    """Accumulator for per-family statistics."""

    # Target Rate
    num_pos: int = 0
    num_total_labels: int = 0

    # Target Cardinality (positives per example)
    card_reservoir: List[int] = field(default_factory=list)
    num_examples: int = 0
    num_empty: int = 0

    # Loss (true contribution tracking)
    sum_loss: float = 0.0  # Sum of task_loss.item() across batches
    batch_count: int = 0  # Number of batches

    # Predictions (New)
    sum_prob_pos: float = 0.0
    count_prob_pos: int = 0
    sum_prob_neg: float = 0.0
    count_prob_neg: int = 0
    fp_count: int = 0
    fn_count: int = 0

    # Extended Separation Metrics
    sum_logit_pos: float = 0.0
    count_logit_pos: int = 0
    sum_logit_neg: float = 0.0
    count_logit_neg: int = 0

    # Recall Counters (TPs at thresholds)
    tp_01_count: int = 0  # pred >= 0.1
    tp_05_count: int = 0  # pred >= 0.5

    # Collisions / Uniqueness
    unique_ids: Set[int] = field(default_factory=set)
    unique_capped: bool = False
    max_unique_cap: int = (
        100_000  # Reduced from 2M - only need sample for collision detection
    )
    precomputed_unique_count: Optional[int] = (
        None  # Set from label phase for amortized tracking
    )

    # Masking (specific to reading_gram)
    mask_count: int = 0
    total_token_count: int = 0

    def add_cardinality(self, count: int) -> None:
        self.num_examples += 1
        if count == 0:
            self.num_empty += 1

        # Reservoir sampling for p50/p90
        if len(self.card_reservoir) < 4096:
            self.card_reservoir.append(count)
        else:
            j = random.randint(0, self.num_examples - 1)
            if j < 4096:
                self.card_reservoir[j] = count

    def add_unique_ids(self, ids: List[int]) -> None:
        if not self.unique_capped:
            self.unique_ids.update(ids)
            if len(self.unique_ids) > self.max_unique_cap:
                self.unique_capped = True
                self.unique_ids.clear()  # Free memory


@dataclass
class MseFamilyStats:
    """Accumulator for MSE (regression) family statistics."""

    # Accumulators for online statistics
    batch_count: int = 0  # Number of batches
    sample_count: int = 0  # Number of samples (for other metrics)
    sum_loss: float = 0.0  # Sum of task_loss.item() across batches
    sum_pred: float = 0.0
    sum_target: float = 0.0
    sum_pred_sq: float = 0.0  # For variance
    sum_target_sq: float = 0.0
    sum_cross: float = 0.0  # For correlation (sum of pred * target)
    correct_discrete: int = 0  # Count matching discrete label bucket


@dataclass
class BertFamilyStats:
    """Accumulator for BERT cloze (morpheme-prediction) family statistics."""

    batch_count: int = 0
    sum_loss: float = 0.0
    top1_correct: int = 0
    top5_correct: int = 0
    top1_pos_only: int = 0
    n_samples: int = 0


class KCEpochDiag:
    """Accumulates and reports per-epoch KC diagnostics."""

    def __init__(self) -> None:
        self.families: Dict[str, FamilyStats] = {}
        self.mse_families: Dict[str, MseFamilyStats] = {}
        self.bert_families: Dict[str, BertFamilyStats] = {}
        # We need a stable hash capability if we want to report collisions on hashed values,
        # but the inputs to update are typically already hashed or raw IDs.
        # We assume 'pos_ids' passed to update are the relevant IDs for collision checking.

    def load_precomputed_unique_counts(self, dataset_dir: str) -> None:
        """Load precomputed KC unique ID counts from the label phase.

        This skips the expensive live collision tracking during training.
        The counts file is generated by scripts/label.py Phase 2/3.
        """
        import json
        import os

        counts_path = os.path.join(dataset_dir, "kc_unique_counts.json")
        if not os.path.exists(counts_path):
            return  # No precomputed counts, will use live tracking

        with open(counts_path, "r", encoding="utf-8") as f:
            counts = json.load(f)

        # Pre-populate families with precomputed counts
        for family_name, count in counts.items():
            if family_name not in self.families:
                self.families[family_name] = FamilyStats()
            self.families[family_name].precomputed_unique_count = count

    def update_family(
        self,
        family_name: str,
        pos_ids: torch.Tensor,  # (B, P) or list of lists, flattened logic preferred
        pos_mask: torch.Tensor,  # (B, P) bool
        probs: torch.Tensor,  # (B, P_sample)
        targets: torch.Tensor,  # (B, P_sample) 0 or 1
        nll: float,
        mask_id: Optional[int] = None,  # For reading_gram masking check
        logits: Optional[torch.Tensor] = None,
    ) -> None:
        if family_name not in self.families:
            self.families[family_name] = FamilyStats()
        stats = self.families[family_name]

        # A4: Validate shapes
        if pos_ids.dim() != 2:
            raise ValueError(
                f"pos_ids must be 2D, got {pos_ids.shape} (family={family_name})"
            )

        if not pos_ids.shape == pos_mask.shape == probs.shape == targets.shape:
            raise ValueError(
                f"Shape mismatch in update_family({family_name}): "
                f"ids={pos_ids.shape} mask={pos_mask.shape} "
                f"probs={probs.shape} targets={targets.shape}"
            )

        # 1. Target Rate & Cardinality

        # Count actual positives (targets == 1 where pos_mask is True)
        targets_float = targets.float()
        actual_positives = (targets_float * pos_mask.float()).sum()
        total_pos = int(actual_positives.item())
        stats.num_pos += total_pos

        # Cardinality tracking: count positives per example
        pos_counts = (targets_float * pos_mask.float()).sum(dim=1)
        counts_list = pos_counts.cpu().tolist()
        for c in counts_list:
            stats.add_cardinality(int(c))

        # Count only valid labels (where pos_mask is True), not all tensor elements
        total_valid_labels = int(pos_mask.sum().item())
        stats.num_total_labels += total_valid_labels

        # 2. Collisions (Track usage of IDs) - OPTIMIZED
        # Only compute valid_pos_ids_cached if we need it for:
        # - Unique ID tracking (when no precomputed counts)
        # - Masking (when mask_id is provided, i.e., reading_gram family)
        valid_pos_ids_cached = None
        needs_unique_tracking = (
            not stats.unique_capped and stats.precomputed_unique_count is None
        )
        needs_masking = mask_id is not None

        if total_pos > 0 and (needs_unique_tracking or needs_masking):
            # Only compute once for both uses if needed
            valid_pos_ids_cached = pos_ids[pos_mask].detach()

            # Unique ID sampling (only if not precomputed)
            if needs_unique_tracking:
                sample_size = min(1000, valid_pos_ids_cached.numel())
                if sample_size < valid_pos_ids_cached.numel():
                    # Random sample indices - only if needed
                    indices = torch.randperm(
                        valid_pos_ids_cached.numel(), device=valid_pos_ids_cached.device
                    )[:sample_size]
                    sampled_ids = valid_pos_ids_cached[indices].cpu().tolist()
                else:
                    sampled_ids = valid_pos_ids_cached.cpu().tolist()
                stats.add_unique_ids(sampled_ids)

        # 3. Predictions - OPTIMIZED: avoid indexed tensor creation
        # Use multiplication with float targets instead of boolean indexing
        p_detach = probs.detach()
        targets_float = targets.float()

        # pos_indices and neg_indices as float masks (0.0 or 1.0)
        pos_mask_float = targets_float
        neg_mask_float = 1.0 - targets_float

        # OPTIMIZED: Compute sums directly without torch.stack to avoid allocations
        # Each sum is a scalar tensor, calling .item() extracts it
        num_pos_samples = int(targets_float.sum().item())
        num_neg_samples = int(neg_mask_float.sum().item())

        preds_float = (p_detach > 0.5).float()

        stats.sum_prob_pos += (p_detach * pos_mask_float).sum().item()
        stats.count_prob_pos += num_pos_samples
        stats.sum_prob_neg += (p_detach * neg_mask_float).sum().item()
        stats.count_prob_neg += num_neg_samples
        stats.fp_count += int((preds_float * neg_mask_float).sum().item())
        stats.fn_count += int(((1.0 - preds_float) * pos_mask_float).sum().item())
        stats.tp_01_count += int(
            ((p_detach >= 0.1).float() * pos_mask_float).sum().item()
        )
        stats.tp_05_count += int(
            ((p_detach >= 0.5).float() * pos_mask_float).sum().item()
        )

        # Logits stats
        if logits is not None:
            l_detach = logits.detach()
            stats.sum_logit_pos += (l_detach * pos_mask_float).sum().item()
            stats.count_logit_pos += num_pos_samples
            stats.sum_logit_neg += (l_detach * neg_mask_float).sum().item()
            stats.count_logit_neg += num_neg_samples

        # 4. Loss (true contribution per batch)
        stats.sum_loss += nll  # nll = task_loss.item() per batch
        stats.batch_count += 1

        # 5. Masking (specific to reading_gram) - OPTIMIZED: reuse cached valid_pos_ids
        if mask_id is not None and valid_pos_ids_cached is not None:
            stats.mask_count += int((valid_pos_ids_cached == mask_id).sum().item())
            stats.total_token_count += total_pos

    def update_mse_family(
        self,
        family_name: str,
        preds: torch.Tensor,  # [B] or [B, 1] predictions
        targets: torch.Tensor,  # [B] or [B, 1] targets
        loss: float,  # Batch MSE loss (already computed)
    ) -> None:
        """Update MSE family diagnostics with a batch of predictions."""
        if family_name not in self.mse_families:
            self.mse_families[family_name] = MseFamilyStats()
        stats = self.mse_families[family_name]

        # Flatten to 1D
        p = preds.squeeze().detach()
        t = targets.squeeze().detach()
        if p.dim() == 0:
            p = p.unsqueeze(0)
            t = t.unsqueeze(0)

        batch_size = p.numel()
        stats.batch_count += 1  # Count batches for loss averaging
        stats.sample_count += batch_size  # Count samples for other metrics
        stats.sum_loss += loss  # loss = task_loss.item() per batch

        # Accumulate for mean/variance/correlation
        stats.sum_pred += p.sum().item()
        stats.sum_target += t.sum().item()
        stats.sum_pred_sq += (p**2).sum().item()
        stats.sum_target_sq += (t**2).sum().item()
        stats.sum_cross += (p * t).sum().item()

        # Discrete bucket accuracy (matches inference thresholds)
        pred_buckets = discretize_mse(p, family_name)
        target_buckets = discretize_mse(t, family_name)
        correct = (pred_buckets == target_buckets).sum().item()
        stats.correct_discrete += int(correct)

    def update_bert_family(
        self,
        family_name: str,
        loss: float,
        top1_correct: int,
        top5_correct: int,
        n_samples: int,
        top1_pos_only_correct: int = 0,
    ) -> None:
        """Update BERT cloze family diagnostics with a batch of predictions."""
        if family_name not in self.bert_families:
            self.bert_families[family_name] = BertFamilyStats()
        stats = self.bert_families[family_name]
        stats.batch_count += 1
        stats.sum_loss += loss
        stats.top1_correct += top1_correct
        stats.top5_correct += top5_correct
        stats.top1_pos_only += top1_pos_only_correct
        stats.n_samples += n_samples

    def get_stats(self) -> KCDiagnosticReport:
        """Return structured statistics."""
        data: Dict[str, KCDiagnosticFamilyStats] = {}
        sorted_families = sorted(self.families.keys())

        for name in sorted_families:
            s = self.families[name]
            # Rate
            rate = s.num_pos / max(1, s.num_total_labels)

            # Cardinality
            if s.card_reservoir:
                s.card_reservoir.sort()
                n = len(s.card_reservoir)
                p50 = s.card_reservoir[n // 2]
                p90 = s.card_reservoir[int(n * 0.9)]
            else:
                p50, p90 = 0, 0

            empty_pct = s.num_empty / max(1, s.num_examples)

            # Loss per batch (true contribution)
            loss_per_batch = s.sum_loss / max(1, s.batch_count)
            p = max(1e-6, min(rate, 1 - 1e-6))
            bias_nll = -(rate * math.log(p) + (1 - rate) * math.log(1 - p))
            dnll = loss_per_batch - bias_nll

            # Mask
            mask_pct = 0.0
            if s.total_token_count > 0:
                mask_pct = s.mask_count / s.total_token_count

            # Extended Metrics
            prob_pos_mean = s.sum_prob_pos / max(1, s.count_prob_pos)
            prob_neg_mean = s.sum_prob_neg / max(1, s.count_prob_neg)
            auc_proxy = prob_pos_mean - prob_neg_mean

            fp_rate = s.fp_count / max(1, s.count_prob_neg)
            fn_rate = s.fn_count / max(1, s.count_prob_pos)
            support = s.num_pos / max(1, s.num_examples)

            # Extended metrics
            logit_pos_mean = s.sum_logit_pos / max(1, s.count_logit_pos)
            logit_neg_mean = s.sum_logit_neg / max(1, s.count_logit_neg)
            delta_p = prob_pos_mean - prob_neg_mean

            # Recall = TP / (TP + FN) = TP / TotalPos
            # We use count_prob_pos as the denominator (positives in the sampled set)
            recall_01 = s.tp_01_count / max(1, s.count_prob_pos)
            recall_05 = s.tp_05_count / max(1, s.count_prob_pos)

            # Accuracy = (TP + TN) / (TP + TN + FP + FN)
            # TN = total_neg - FP
            tp_count = s.tp_05_count  # Using 0.5 threshold
            tn_count = max(0, s.count_prob_neg - s.fp_count)
            total_count = s.count_prob_pos + s.count_prob_neg
            accuracy = (tp_count + tn_count) / max(1, total_count)

            family_stats = KCDiagnosticFamilyStats(
                rate=rate,
                p50=p50,
                p90=p90,
                empty_pct=empty_pct,
                dnll=dnll,
                mask_pct=mask_pct,
                batch_count=s.batch_count,
                loss_mean=loss_per_batch,
                prob_pos_mean=prob_pos_mean,
                prob_neg_mean=prob_neg_mean,
                auc_proxy=auc_proxy,
                fp_rate=fp_rate,
                fn_rate=fn_rate,
                support=support,
                logit_pos_mean=logit_pos_mean,
                logit_neg_mean=logit_neg_mean,
                delta_p=delta_p,
                recall_01=recall_01,
                recall_05=recall_05,
                accuracy=accuracy,
            )
            data[name] = family_stats

        # Compute MSE family stats
        mse_data: Dict[str, KCMseFamilyStats] = {}
        mse_name: str
        mse_s: MseFamilyStats
        for mse_name, mse_s in sorted(self.mse_families.items()):
            n_batches = max(1, mse_s.batch_count)
            n_samples = max(1, mse_s.sample_count)

            # Loss per batch (true contribution)
            loss_per_batch = mse_s.sum_loss / n_batches
            discrete_accuracy = mse_s.correct_discrete / n_samples

            # Mean values (per sample)
            mean_pred = mse_s.sum_pred / n_samples
            mean_target = mse_s.sum_target / n_samples
            mean_bias = mean_pred - mean_target

            # Variance of predictions: E[X^2] - E[X]^2
            var_pred = max(0, mse_s.sum_pred_sq / n_samples - mean_pred**2)
            pred_std = math.sqrt(var_pred)

            # Pearson correlation: cov(P,T) / (std_P * std_T)
            var_target = max(0, mse_s.sum_target_sq / n_samples - mean_target**2)
            cov_pt = mse_s.sum_cross / n_samples - mean_pred * mean_target
            std_target = math.sqrt(var_target)
            if pred_std > 1e-6 and std_target > 1e-6:
                correlation = cov_pt / (pred_std * std_target)
            else:
                correlation = 0.0

            mse_data[mse_name] = KCMseFamilyStats(
                loss_mean=loss_per_batch,
                discrete_accuracy=discrete_accuracy,
                correlation=correlation,
                mean_bias=mean_bias,
                pred_std=pred_std,
                batch_count=mse_s.batch_count,
            )

        # Compute BERT cloze family stats
        bert_data: Dict[str, KCBertFamilyStats] = {}
        for bert_name, bert_s in sorted(self.bert_families.items()):
            n_batches = max(1, bert_s.batch_count)
            n_samples = max(1, bert_s.n_samples)
            bert_data[bert_name] = KCBertFamilyStats(
                loss_mean=bert_s.sum_loss / n_batches,
                top1_accuracy=bert_s.top1_correct / n_samples,
                top5_accuracy=bert_s.top5_correct / n_samples,
                top1_pos_only_accuracy=bert_s.top1_pos_only / n_samples
                if bert_s.top1_pos_only
                else 0.0,
                batch_count=bert_s.batch_count,
            )

        return KCDiagnosticReport(
            families=data, mse_families=mse_data, bert_families=bert_data
        )


# --- Formatting ---
