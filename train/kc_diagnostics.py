"""Diagnostic tools for Knowledge Component (KC) training."""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch

from train.types import (
    KCDiagnosticFamilyStats,
    KCDiagnosticReport,
)

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

    # Loss
    sum_nll: float = 0.0
    count_nll: int = 0

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
    max_unique_cap: int = 2000000

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


class KCEpochDiag:
    """Accumulates and reports per-epoch KC diagnostics."""

    def __init__(self) -> None:
        self.families: Dict[str, FamilyStats] = {}
        # We need a stable hash capability if we want to report collisions on hashed values,
        # but the inputs to update are typically already hashed or raw IDs.
        # We assume 'pos_ids' passed to update are the relevant IDs for collision checking.

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
        # pos_ids/pos_mask represent the raw positive set per example
        batch_size = pos_mask.size(0)

        # Flattened valid positive IDs
        # Detach to ensure no graph retention in diagnostics
        valid_pos_ids = pos_ids[pos_mask].detach()

        # Num positives per example
        counts = pos_mask.sum(dim=1).cpu().tolist()
        for c in counts:
            stats.add_cardinality(c)

        stats.num_pos += int(valid_pos_ids.numel())
        # For rate, we use the training sample universe (positives + sampled negatives)
        # targets tensor shape is (B, P_sample)
        stats.num_total_labels += int(targets.numel())

        # 2. Collisions (Track usage of IDs)
        if valid_pos_ids.numel() > 0:
            stats.add_unique_ids(valid_pos_ids.cpu().tolist())

        # 3. Predictions
        # Probs and targets are aligned (B, P_sample)
        # Separate into pos and neg stats
        pos_indices = targets == 1
        neg_indices = targets == 0

        # Detach for stats to avoid graph retention
        p_detach = probs.detach()

        # Accumulate sums
        stats.sum_prob_pos += float(p_detach[pos_indices].sum())
        stats.count_prob_pos += int(pos_indices.sum())

        stats.sum_prob_neg += float(p_detach[neg_indices].sum())
        stats.count_prob_neg += int(neg_indices.sum())

        # FP/FN at 0.5
        preds = p_detach > 0.5
        # FP: preds=1, targets=0
        stats.fp_count += int((preds & neg_indices).sum())
        # FN: preds=0, targets=1
        stats.fn_count += int((~preds & pos_indices).sum())

        # Recall metrics (TP counts at thresholds)
        # Recall = TP / NumPos. We already track num_pos (total valid positives).
        # Here we count TPs for the sampled set. Note num_pos tracks *all* positives in the example,
        # but pos_indices tracks positives *in the sampled set*.
        # For precision/recall accuracy, we should normalize by the sampled set positives.

        stats.tp_01_count += int(((p_detach >= 0.1) & pos_indices).sum())
        stats.tp_05_count += int(((p_detach >= 0.5) & pos_indices).sum())

        # Logits stats
        if logits is not None:
            # Detach and CPU
            l_detach = logits.detach()
            stats.sum_logit_pos += float(l_detach[pos_indices].sum())
            stats.count_logit_pos += int(pos_indices.sum())

            stats.sum_logit_neg += float(l_detach[neg_indices].sum())
            stats.count_logit_neg += int(neg_indices.sum())
        # 4. Loss
        stats.sum_nll += nll * batch_size  # Weighted average accumulate
        stats.count_nll += batch_size

        # 5. Masking (specific to reading_gram)
        # If mask_id is provided, check how many pos_ids match it
        if mask_id is not None and valid_pos_ids.numel() > 0:
            # This is "post-derivation" masking check
            stats.mask_count += int((valid_pos_ids == mask_id).sum().item())
            stats.total_token_count += int(valid_pos_ids.numel())

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

            # Preds
            nll = s.sum_nll / max(1, s.count_nll)
            p = max(1e-6, min(rate, 1 - 1e-6))
            bias_nll = -(rate * math.log(p) + (1 - rate) * math.log(1 - p))
            dnll = nll - bias_nll

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

            family_stats = KCDiagnosticFamilyStats(
                rate=rate,
                p50=p50,
                p90=p90,
                empty_pct=empty_pct,
                dnll=dnll,
                mask_pct=mask_pct,
                loss_mean=nll,
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
            )
            data[name] = family_stats

        return KCDiagnosticReport(families=data)


# --- Formatting ---
