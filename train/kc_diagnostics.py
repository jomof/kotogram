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
# pylint: disable=too-many-positional-arguments,too-many-locals,unused-argument


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

    # Predictions
    sum_prob_pos: float = 0.0
    count_prob_pos: int = 0
    sum_prob_neg: float = 0.0
    count_prob_neg: int = 0

    # Loss
    sum_nll: float = 0.0
    count_nll: int = 0

    # Collisions / Uniqueness
    unique_ids: Set[int] = field(default_factory=set)
    unique_capped: bool = False
    max_unique_cap: int = 2000000
    total_targets_seen: int = 0

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
        self.total_targets_seen += len(ids)
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
    ) -> None:
        if family_name not in self.families:
            self.families[family_name] = FamilyStats()
        stats = self.families[family_name]

        # 1. Target Rate & Cardinality
        # pos_ids/pos_mask represent the raw positive set per example
        batch_size = pos_mask.size(0)

        # Flattened valid positive IDs
        valid_pos_ids = pos_ids[pos_mask]

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
        with torch.no_grad():
            t_bool = targets.bool()

            p_pos = probs[t_bool]
            if p_pos.numel() > 0:
                stats.sum_prob_pos += float(p_pos.sum().item())
                stats.count_prob_pos += int(p_pos.numel())

            p_neg = probs[~t_bool]
            if p_neg.numel() > 0:
                stats.sum_prob_neg += float(p_neg.sum().item())
                stats.count_prob_neg += int(p_neg.numel())

        # 4. Loss
        stats.sum_nll += nll * batch_size  # Weighted average accumulate
        stats.count_nll += batch_size

        # 5. Masking (specific to reading_gram)
        # If mask_id is provided, check how many pos_ids match it
        if mask_id is not None and valid_pos_ids.numel() > 0:
            # This is "post-derivation" masking check
            stats.mask_count += int((valid_pos_ids == mask_id).sum().item())
            stats.total_token_count += int(valid_pos_ids.numel())

    def finalize(self, epoch: int) -> List[str]:
        """Produce compact report strings."""
        lines = []

        # Sort by weight or name? Let's sort by name for stability, or custom order.
        # User requested sorting by effective_weighted_loss or name. Name is stable.
        sorted_families = sorted(self.families.keys())

        # Header
        # Reuse summary style or own? "KCdiag ep=3 ..."
        # Note: We don't have global stats here easily without aggregation, so we stick to per-family.
        # But we can produce a summary line first.

        # Global Density Estimate (weighted by family inputs?)
        total_pos = sum(f.num_pos for f in self.families.values())
        total_lbl = sum(f.num_total_labels for f in self.families.values())
        global_dens = total_pos / max(1, total_lbl)

        header = (
            f"KCdiag ep={epoch + 1} fam={len(sorted_families)} dens={global_dens:.4f}"
        )
        lines.append(header)

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
            pp = s.sum_prob_pos / max(1, s.count_prob_pos)
            pn = s.sum_prob_neg / max(1, s.count_prob_neg)
            sep = pp - pn

            # NLL & Bias
            nll = s.sum_nll / max(1, s.count_nll)

            # Bias NLL: - (p log p + (1-p) log (1-p)) roughly, but strictly BCE
            # We use the observed rate as the prior 'p'
            p = max(1e-6, min(rate, 1 - 1e-6))
            # If model predicts constant 'p', average BCE is exactly binary_entropy(p)
            # bias_nll = - (rate * log(p) + (1-rate) * log(1-p)) = binary_entropy(rate)
            # Actually we typically evaluate against targets 0 and 1.
            # So bias NLL is simply BCE(constant_p, targets).
            # Since avg target is 'rate', this simplifies to binary entropy of rate.
            bias_nll = -(rate * math.log(p) + (1 - rate) * math.log(1 - p))
            dnll = nll - bias_nll

            # Collisions
            if s.unique_capped:
                col_str = "CAP"
            else:
                uniq = len(s.unique_ids)
                # Collision rate within the positive targets we actually saw
                # If we saw T total targets and U unique, redundancy is 1 - U/T
                # But 'collisions' for hash usually means different inputs mapping to same bucket.
                # Here we are looking at output bucket reuse.
                # If the user means "hash collisions" (different inputs -> same bucket), we can't fully know without inputs.
                # The user spec says: "collisions% = 1 - uniq_targets/max(1,total_targets)"
                # This measures REPETITION of targets.
                total_t = s.total_targets_seen
                col_pct = 1.0 - (uniq / max(1, total_t))
                col_str = f"{col_pct:.2f}"

            # Mask
            mask_str = "NA"
            if s.total_token_count > 0 and name == "reading_gram":
                m_pct = s.mask_count / s.total_token_count
                mask_str = f"{m_pct:.2f}"

            # Format Line
            # "F bag_pos r=0.328 c50/90=1/3 e=0.02 pp/pn=0.61/0.49 s=0.12 dNLL=-0.004 col=NA m=NA"
            line = (
                f"F {name:<20} "
                f"r={rate:.3f} "
                f"c50/90={p50}/{p90} "
                f"e={empty_pct:.2f} "
                f"pp/pn={pp:.2f}/{pn:.2f} "
                f"s={sep:.2f} "
                f"dNLL={dnll:+.3f} "
                f"col={col_str} "
                f"m={mask_str}"
            )
            lines.append(line)

        return lines

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

            family_stats = KCDiagnosticFamilyStats(
                rate=rate,
                p50=p50,
                p90=p90,
                empty_pct=empty_pct,
                dnll=dnll,
                mask_pct=mask_pct,
            )
            data[name] = family_stats

        return KCDiagnosticReport(
            families=data,
        )
