# pylint: disable=too-many-lines,not-callable,too-many-nested-blocks,duplicate-code
import math
import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from kotogram.constants import REGISTER_ID_TO_LABEL
from kotogram.model import compute_k_budget
from train.config import (
    DataLoaderConfig,
    KCConfig,
    TrainerConfig,
    _safe_configure_threads,
    configure_runtime_thread_limits,
)
from train.dataset import StyleDataset, collate_fn, create_kc_batch
from train.display import (
    RichTrainerProgressBar,
)
from train.kc import get_family, is_family_db_sourced, is_family_sparse
from train.kc_diagnostics import (
    KCEpochDiag,
)
from train.kc_trainer_view import KCTrainerDiagnosticsView, KCTrainerView
from train.models import TrainingClassifier
from train.profile import Timer, get_profile_dir
from train.pytorch_utils import estimate_optimal_batch_size
from train.types import (
    FamilyAccumulator,
    KcEpochActivationStats,
    KcEpochSummary,
    KCLosses,
    KcLossWeights,
    KCStructuralBiases,
    KCTrainingHistory,
    RunningLossComponents,
    TensorStats,
    TrainEpochResult,
    TrainEpochStats,
    WorstSampleInfo,
)
from train.worker import _worker_init_fn


def tensor_finite_stats(x: Optional[torch.Tensor]) -> TensorStats:
    if x is None:
        return TensorStats(
            finite=True,
            n_nan=0,
            n_inf=0,
            min=float("nan"),
            max=float("nan"),
        )

    if x.isfinite().all():
        flat = x.detach().flatten().float()
        min_val = float(flat.min().item()) if flat.numel() else float("nan")
        max_val = float(flat.max().item()) if flat.numel() else float("nan")
        return TensorStats(
            finite=True,
            n_nan=0,
            n_inf=0,
            min=min_val,
            max=max_val,
        )

    flat = x.detach().flatten().float()
    is_finite = torch.isfinite(flat)
    n_nan = int(torch.isnan(flat).sum().item())
    n_inf = int(torch.isinf(flat).sum().item())

    finite_vals = flat[is_finite]
    if len(finite_vals) > 0:
        min_val = float(finite_vals.min().item())
        max_val = float(finite_vals.max().item())
    else:
        min_val = float("nan")
        max_val = float("nan")

    return TensorStats(
        finite=False,
        n_nan=n_nan,
        n_inf=n_inf,
        min=min_val,
        max=max_val,
    )


def _get_display_sentence(
    sentence: str,
    kotogram: str = "",
    fallback: str = "<binary>",
) -> str:
    """Get displayable sentence text, preferring sentence over kotogram."""
    if sentence:
        return sentence
    if kotogram:
        return kotogram
    return fallback


class KCTrainer:
    # pylint: disable=too-many-positional-arguments,too-many-locals
    def __init__(
        self,
        model: TrainingClassifier,
        dataset: StyleDataset,
        config: TrainerConfig,
        dl_config: DataLoaderConfig,
        kc_config: KCConfig,
        view: Optional[KCTrainerView] = None,
    ):
        dataset = dataset.filter_by_grammaticality(1)

        self.model = model
        self.dataset = dataset
        self.config = config
        self.view: KCTrainerView = (
            view if view is not None else KCTrainerDiagnosticsView()
        )

        _safe_configure_threads(self.config)

        configure_runtime_thread_limits(self.config)

        self.kc_config = kc_config
        self.kc_sparsity_weight = self.kc_config.sparsity_weight
        self.kc_sat_weight = self.kc_config.sat_weight
        self.freeze_encoder_epochs = self.kc_config.freeze_encoder_epochs

        self.device = torch.device(self.config.device)
        self.model.to(self.device)

        self.val_sampler = None
        self.sampler = None

        # Create style-aware sampler if enabled (oversamples non-neutral examples)
        if kc_config.style_oversample and hasattr(dataset, "create_style_oversampler"):
            self.sampler = dataset.create_style_oversampler(
                formality_boost=kc_config.formality_boost,
                gender_boost=kc_config.gender_boost,
            )
            self.view.on_style_oversampling_enabled(
                kc_config.formality_boost, kc_config.gender_boost
            )

        if dl_config is None:
            dl_config = self.config.resolve_dataloader_config(self.device, mode="train")

        batch_size = self.config.batch_size
        if batch_size == -1:
            batch_size = estimate_optimal_batch_size(
                self.device, self.model.config, is_kc=True
            )
            # Log this via view if possible, or print
            self.view.on_auto_batch_size(batch_size, self.device)

        self.data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(self.sampler is None),
            sampler=self.sampler,
            collate_fn=partial(
                collate_fn,
                # max_seq_len=max_seq_len, # Removed constant param
            ),
            num_workers=dl_config.num_workers,
            pin_memory=dl_config.pin_memory,
            persistent_workers=dl_config.persistent_workers,
            prefetch_factor=dl_config.prefetch_factor,
            worker_init_fn=_worker_init_fn,
        )
        self._create_optimizer(freeze_encoder=self.freeze_encoder_epochs > 0)

        self.mse_loss = nn.MSELoss()

        pid = os.getpid()
        profile_dir = get_profile_dir()
        data_log = (
            os.path.join(profile_dir, f"kc_data_{pid}.jsonl") if profile_dir else None
        )
        comp_log = (
            os.path.join(profile_dir, f"kc_compute_{pid}.jsonl")
            if profile_dir
            else None
        )

        self.train_timer_data = Timer("data_loading", output_path=data_log)
        self.train_timer_compute = Timer("compute", output_path=comp_log)

        self.kc_diversity_weight_frozen = self.kc_config.diversity_weight
        self.kc_diversity_weight_thawed = self.kc_config.diversity_weight_thawed

        self.kc_diversity_eps = self.kc_config.diversity_eps
        self.kc_diversity_warmup_epochs = self.kc_config.diversity_warmup_epochs
        self.kc_sparsity_mode = "target_density"

        self.kc_lb_weight_frozen = self.kc_config.lb_weight
        self.kc_lb_weight_thawed = self.kc_config.lb_weight_thawed

        self.kc_collapse_weight_thawed = self.kc_config.collapse_weight_thawed

        self.kc_temperature_frozen = float(self.model.config.kc_temperature)

        self.kc_temperature_thawed = self.kc_config.temperature_thawed

        self.kc_grad_cap = self.kc_config.kc_grad_cap

        self.kc_entropy_floor = self.kc_config.entropy_floor
        self.kc_kl_cap = self.kc_config.kl_cap

        self.history = KCTrainingHistory()
        profile_dir = get_profile_dir()
        pid = os.getpid()
        self.train_timer_data = Timer(
            "kc_data_loading",
            output_path=os.path.join(profile_dir, f"kc_data_{pid}.jsonl")
            if profile_dir
            else None,
        )
        self.start_epoch = 0
        self.session_start_epoch: Optional[int] = None
        self.start_batch = 0
        # Per-family positive densities for adaptive loss weighting
        self.family_pos_densities: Dict[str, float] = {}

    def _balanced_bce_loss(
        self,
        gathered_logits: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_count: int,
        valid: torch.Tensor,
        pos_density: float = 8.0,
    ) -> torch.Tensor:
        """Computes balanced BCE loss with adaptive pos/neg weighting based on pos_density."""
        # gathered_logits: (B, P + N)
        # pos_mask: (B, P)
        # valid: (B, P + N) - typically [pos_mask, ones(neg_count)]

        n_pos = pos_mask.size(1)
        assert neg_count > 0, "neg_count must be positive"

        # Split logits
        pos_logits = gathered_logits[:, :n_pos]
        neg_logits = gathered_logits[:, n_pos:]

        # Prevalence-Aware Loss
        if n_pos > 0:
            n_pos_total = float(pos_mask.sum().item())
            n_neg_total = float(neg_logits.numel())

            ratio = n_neg_total / max(1.0, n_pos_total)
            pos_weight_val = max(1.0, min(50.0, ratio))
        else:
            pos_weight_val = 1.0

        # Positive Loss (Weighted)
        if pos_mask.any():
            pos_weight_t = torch.tensor(pos_weight_val, device=gathered_logits.device)
            per_entry_pos_loss = F.binary_cross_entropy_with_logits(
                pos_logits,
                torch.ones_like(pos_logits),
                reduction="none",
                pos_weight=pos_weight_t,
            )
            # Mask out padding
            masked_pos_loss = per_entry_pos_loss * pos_mask.float()
            # Mean over POSITIVES only
            loss_pos = masked_pos_loss.sum() / pos_mask.float().sum().clamp_min(1.0)
        else:
            loss_pos = torch.tensor(0.0, device=gathered_logits.device)

        # Negative Loss (Standard)
        per_entry_neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits, torch.zeros_like(neg_logits), reduction="none"
        )
        neg_valid = valid[:, n_pos:]
        masked_neg_loss = per_entry_neg_loss * neg_valid.float()
        # Mean over NEGATIVES only
        loss_neg = masked_neg_loss.sum() / neg_valid.float().sum().clamp_min(1.0)

        # Adaptive pos/neg weighting based on pos_density
        # Lower pos_density -> higher pos_weight to compensate for fewer positives
        baseline_density = 8.0
        adaptive_pos_weight = min(0.8, 0.5 * baseline_density / max(1.0, pos_density))
        adaptive_neg_weight = 1.0 - adaptive_pos_weight

        bce = adaptive_pos_weight * loss_pos + adaptive_neg_weight * loss_neg
        return bce

    def _continuous_mse_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE loss for continuous targets.

        Uses Tanh output range [-1, 1] where:
        - 0 = neutral
        - -1 = one extreme (e.g., masculine/informal)
        - +1 = other extreme (e.g., feminine/formal)

        Args:
            logits: (B, 1) predictions from the KC decoder (Tanh-bounded to [-1, 1])
            targets: (B,) float targets in range [-1, 1]

        Returns:
            Scalar MSE loss tensor
        """
        # Squeeze logits from (B, 1) to (B,)
        preds = logits.squeeze(-1)
        return torch.nn.functional.mse_loss(preds, targets)

    def _multilabel_pnu_loss(
        self,
        logits: torch.Tensor,
        pos_ids: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_ids: torch.Tensor,
        neg_mask: torch.Tensor,
        vocab_size: int,
        unlabeled_weight: float = 0.001,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
    ) -> torch.Tensor:
        """Multi-label PNU (Positive-Negative-Unlabeled) loss for grammar points.

        This is adapted for multi-label classification with explicit negatives,
        following semi-supervised multi-label learning principles.

        Key differences from single-label PNU:
        1. We have 1,374 independent binary problems (one per grammar point)
        2. Each problem has its own class distribution
        3. Explicit negatives provide direct supervision
        4. Unlabeled positions treated as weak negatives (sparsity assumption)

        Loss components:
        1. Positive loss: Standard BCE on labeled positives (target=1)
        2. Negative loss: Standard BCE on labeled negatives (target=0)
        3. Unlabeled loss: Weak negative (small weight, encourages sparsity)

        The sparsity assumption: For any given sentence, most grammar points
        don't apply (true negatives). Unlabeled positions are likely negative,
        but we use low weight since they might contain hidden positives.

        This avoids the pitfalls of applying single-label nnPU to multi-label:
        - No single class prior (each GP has different base rate)
        - No need for correction term (we have explicit negatives)
        - Simple, stable, and aligned with multi-label SSL literature

        References:
        - Bucak et al. (2011): Multi-label learning with incomplete class assignments
        - Cabral et al. (2011): Matrix completion for multi-label classification
        - Durand et al. (2019): Learning with partial labels in multi-label classification

        Args:
            logits: (B, vocab_size) logits from KC decoder
            pos_ids: (B, max_pos) positive grammar point IDs
            pos_mask: (B, max_pos) mask for valid positive IDs
            neg_ids: (B, max_neg) negative grammar point IDs
            neg_mask: (B, max_neg) mask for valid negative IDs
            vocab_size: number of grammar points (1374)
            unlabeled_weight: weight for unlabeled risk (default: 0.001)
            pos_weight: weight for positive loss (default: 1.0)
            neg_weight: weight for negative loss (default: 1.0)

        Returns:
            Scalar loss tensor
        """
        batch_size = logits.size(0)
        device = logits.device

        # Build label masks: (B, vocab_size)
        labeled_pos = torch.zeros(batch_size, vocab_size, device=device)
        labeled_neg = torch.zeros(batch_size, vocab_size, device=device)

        # Scatter positive labels
        valid_pos = pos_ids.clamp(0, vocab_size - 1)
        labeled_pos.scatter_(1, valid_pos, pos_mask.float())

        # Scatter negative labels
        valid_neg = neg_ids.clamp(0, vocab_size - 1)
        labeled_neg.scatter_(1, valid_neg, neg_mask.float())

        # Unlabeled mask
        unlabeled_mask = 1.0 - labeled_pos - labeled_neg
        unlabeled_mask = unlabeled_mask.clamp(0, 1)

        # Component 1: Positive loss (BCE with target=1)
        pos_count = labeled_pos.sum()
        if pos_count > 0:
            pos_loss = F.binary_cross_entropy_with_logits(
                logits, torch.ones_like(logits), reduction="none"
            )
            risk_pos = pos_weight * (pos_loss * labeled_pos).sum() / pos_count
        else:
            risk_pos = torch.tensor(0.0, device=device)

        # Component 2: Negative loss (BCE with target=0)
        neg_count = labeled_neg.sum()
        if neg_count > 0:
            neg_loss = F.binary_cross_entropy_with_logits(
                logits, torch.zeros_like(logits), reduction="none"
            )
            risk_neg = neg_weight * (neg_loss * labeled_neg).sum() / neg_count
        else:
            risk_neg = torch.tensor(0.0, device=device)

        # Component 3: Unlabeled loss (weak negative, low weight)
        # Encourages sparsity: most GPs don't apply to most sentences
        unl_count = unlabeled_mask.sum()
        if unl_count > 0 and unlabeled_weight > 0:
            unl_loss = F.binary_cross_entropy_with_logits(
                logits, torch.zeros_like(logits), reduction="none"
            )
            risk_unl = unlabeled_weight * (unl_loss * unlabeled_mask).sum() / unl_count
        else:
            risk_unl = torch.tensor(0.0, device=device)

        total_loss: torch.Tensor = risk_pos + risk_neg + risk_unl
        return total_loss

    # pylint: disable=too-many-locals,too-many-positional-arguments
    def _bce_sampled_from_sparse(
        self,
        logits_f: torch.Tensor,
        pos_inds: torch.Tensor,
        pos_mask: torch.Tensor,
        vocab_size: int,
        neg_count: int = 128,
        seed: int = 0,
        diag: Optional[KCEpochDiag] = None,
        family_name: str = "",
        reading_mask_id: int = 0,
        accumulator: Optional[FamilyAccumulator] = None,
        loss_weight: float = 1.0,
    ) -> torch.Tensor:
        batch_size = int(logits_f.size(0))
        device = logits_f.device
        n_pos = int(pos_inds.size(1))
        neg_c = neg_count

        pos_i = pos_inds.clone()
        pos_i[~pos_mask] = -1

        g = torch.Generator(device=device)
        g.manual_seed(seed)
        neg_i = torch.randint(
            4, vocab_size, (batch_size, neg_c), device=device, generator=g
        )

        if n_pos > 0 and pos_mask.any():
            for _ in range(3):
                coll = (neg_i.unsqueeze(-1) == pos_i.unsqueeze(1)).any(dim=-1)
                if not coll.any():
                    break
                repl = torch.randint(
                    4, vocab_size, (int(coll.sum().item()),), device=device, generator=g
                )
                neg_i[coll] = repl

        idxs = torch.cat([pos_i, neg_i], dim=1)

        # --- INVARIANT CHECK I: Sampling ---
        if idxs.dtype not in (torch.int64, torch.int32):
            raise RuntimeError("idxs not int")
        if idxs.min().item() < -1:
            raise RuntimeError(f"idxs < -1: {idxs.min().item()}")

        idxs_safe_chk = idxs.clamp_min(0)
        if idxs_safe_chk.max().item() >= vocab_size:
            raise RuntimeError(
                f"idxs >= vocab_size: {idxs_safe_chk.max().item()} >= {vocab_size}"
            )

        t_pos = pos_mask.float()
        t_neg = torch.zeros((batch_size, neg_c), device=device, dtype=torch.float)
        t = torch.cat([t_pos, t_neg], dim=1)

        valid = torch.cat(
            [
                pos_mask,
                torch.ones((batch_size, neg_c), device=device, dtype=torch.bool),
            ],
            dim=1,
        )

        idxs_safe = idxs.clamp_min(0)
        gathered = logits_f.gather(1, idxs_safe)

        bce = self._balanced_bce_loss(
            gathered,
            pos_mask,
            neg_count,
            valid,
            pos_density=self.family_pos_densities.get(family_name, 8.0),
        )

        if not torch.isfinite(bce):
            raise RuntimeError(f"Non-finite KC loss for {family_name}")

        # Scale by per-family loss weight
        bce = bce * loss_weight

        if diag is not None and family_name:
            with torch.no_grad():
                if not family_name:
                    raise ValueError("Family name cannot be empty")

                # A3: Pass 2D tensors to update_family (B, P+N)
                diag_inds = idxs_safe
                diag_pos_mask = torch.cat(
                    [
                        pos_mask,
                        torch.zeros(
                            (batch_size, neg_c), dtype=torch.bool, device=device
                        ),
                    ],
                    dim=1,
                )
                diag_probs = torch.sigmoid(gathered)
                diag_targets = t

                assert (
                    diag_inds.shape
                    == diag_probs.shape
                    == diag_targets.shape
                    == diag_pos_mask.shape
                )

                diag.update_family(
                    family_name,
                    diag_inds.detach(),
                    diag_pos_mask.detach(),
                    diag_probs.detach(),
                    diag_targets.detach(),
                    bce.item(),
                    mask_id=reading_mask_id,
                    logits=gathered,
                )

        # Update accumulator for sparse path (no double-counting: this is the ONLY
        # place sparse sampled targets are accumulated)
        if accumulator is not None:
            with torch.no_grad():
                # t is [B, P+N]. Entries with 1.0 are pos.
                full_pos_mask = t > 0.5
                accumulator.update(
                    logits=gathered.detach(),
                    targets=t.detach(),
                    pos_mask=full_pos_mask,
                    valid_mask=None,  # all sampled entries are valid
                    source="sparse",
                )

        return bce

    # pylint: disable=too-many-locals
    def _init_structural_decoder_biases(self, num_batches: int = 10) -> None:
        m = self.model
        if not hasattr(m, "kc_decoders"):
            return

        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        biases = KCStructuralBiases(sums=sums, counts=counts)
        # Track positive densities (positives per example) for adaptive weighting
        pos_density_sums: Dict[str, float] = {}
        pos_density_counts: Dict[str, int] = {}

        for i, batch in enumerate(self.data_loader):
            if i >= num_batches:
                break

            kc_targets = create_kc_batch(
                batch=batch,
                tokenizer=self.dataset.tokenizer,
                target_specs=self.config.kc_target_specs,
            )

            for fid, vocab_size in self.config.kc_target_specs.items():
                # fid is KcFamilyId
                name = fid.name.lower()
                dense_key = f"kc_targets_{name}"
                mask_key = f"kc_pos_mask_{name}"

                if dense_key in kc_targets:
                    t = kc_targets[dense_key].float()
                    p = t.mean().item()
                    biases.sums[name] = biases.sums.get(name, 0.0) + p
                    biases.counts[name] = biases.counts.get(name, 0) + 1
                    # PosDen for dense: sum of positives per example
                    batch_size = t.size(0)
                    total_pos = t.sum().item()
                    pos_density_sums[name] = pos_density_sums.get(name, 0.0) + total_pos
                    pos_density_counts[name] = (
                        pos_density_counts.get(name, 0) + batch_size
                    )
                elif mask_key in kc_targets:
                    pos_mask_t = kc_targets[mask_key]
                    batch_size = pos_mask_t.size(0)
                    num_pos = pos_mask_t.sum().item()
                    p = num_pos / (batch_size * vocab_size)
                    # p = max(p, 1e-5)  # CLAMP FIX: Prevent initialization at -inf
                    biases.sums[name] = biases.sums.get(name, 0.0) + p
                    biases.counts[name] = biases.counts.get(name, 0) + 1
                    # PosDen for sparse: num positives / num examples
                    pos_density_sums[name] = pos_density_sums.get(name, 0.0) + num_pos
                    pos_density_counts[name] = (
                        pos_density_counts.get(name, 0) + batch_size
                    )
                else:
                    # DB-sourced families have different key patterns
                    gp_mask_key = f"kc_gp_pos_mask_{name}"
                    continuous_key = f"kc_continuous_{name}"
                    class_key = f"kc_class_{name}"

                    if gp_mask_key in kc_targets:
                        # PNU families (GRAMMAR_POINT)
                        pos_mask_t = kc_targets[gp_mask_key]
                        batch_size = pos_mask_t.size(0)
                        num_pos = pos_mask_t.sum().item()
                        # For DB-sourced families, use the actual positive rate
                        # (not divided by vocab_size like sparse families)
                        p = num_pos / max(1, batch_size * pos_mask_t.size(1))
                        biases.sums[name] = biases.sums.get(name, 0.0) + p
                        biases.counts[name] = biases.counts.get(name, 0) + 1
                        pos_density_sums[name] = (
                            pos_density_sums.get(name, 0.0) + num_pos
                        )
                        pos_density_counts[name] = (
                            pos_density_counts.get(name, 0) + batch_size
                        )
                    elif continuous_key in kc_targets:
                        # MSE families (GENDER, FORMALITY) - no bias init needed
                        # (Tanh output, bias doesn't affect learning much)
                        pass
                    elif class_key in kc_targets:
                        # Classification families (GENDER_CLASS, FORMALITY_CLASS)
                        # Compute class distribution for bias initialization
                        class_targets = kc_targets[class_key]
                        batch_size = class_targets.size(0)
                        # Average probability per class (1/num_classes for uniform init)
                        p = 1.0 / vocab_size  # vocab_size = num_classes
                        biases.sums[name] = biases.sums.get(name, 0.0) + p
                        biases.counts[name] = biases.counts.get(name, 0) + 1
                        # Density = 1 (every example has exactly 1 true class)
                        pos_density_sums[name] = (
                            pos_density_sums.get(name, 0.0) + batch_size
                        )
                        pos_density_counts[name] = (
                            pos_density_counts.get(name, 0) + batch_size
                        )
                    else:
                        # Check for multi-label families (register)
                        multilabel_key = f"kc_multilabel_{name}"
                        if multilabel_key in kc_targets:
                            # Multi-label families use actual positive rate
                            targets_ml = kc_targets[multilabel_key]
                            batch_size = targets_ml.size(0)
                            num_pos = targets_ml.sum().item()
                            # Positive rate across all positions
                            p = num_pos / max(1, batch_size * vocab_size)
                            biases.sums[name] = biases.sums.get(name, 0.0) + p
                            biases.counts[name] = biases.counts.get(name, 0) + 1
                            # PosDen for multi-label: average positives per example
                            pos_density_sums[name] = (
                                pos_density_sums.get(name, 0.0) + num_pos
                            )
                            pos_density_counts[name] = (
                                pos_density_counts.get(name, 0) + batch_size
                            )

        for name_id, vocab_size in self.config.kc_target_specs.items():
            name = name_id.name.lower()
            # Note: KCStructuralBiases implementation uses string keys currently, matching KCDecoder's ModuleDict.
            # We must use strings here.
            if name not in sums or counts.get(name, 0) == 0:
                continue

            p = sums[name] / counts[name]
            if vocab_size >= 4096:
                p = max(5e-4, p)  # Fix stuck negative logits for large vocab
            p = max(1e-6, min(1.0 - 1e-6, p))
            b = float(-torch.log(torch.tensor(1.0 / p - 1.0)).item())

            # Check if this is a label family or MSE family
            if name in m.kc_decoders.decoders:
                lin = m.kc_decoders.decoders[name]
                if lin.bias is not None:
                    nn.init.constant_(lin.bias, b)
            elif name in m.kc_decoders.mse_decoders:
                lin = m.kc_decoders.mse_decoders[name]
                if lin.bias is not None:
                    nn.init.constant_(lin.bias, b)
            else:
                continue  # Family not in either decoder dict

            # Store pos_density for adaptive loss weighting
            if name in pos_density_sums and pos_density_counts.get(name, 0) > 0:
                pos_den = pos_density_sums[name] / pos_density_counts[name]
                self.family_pos_densities[name] = pos_den

            self.view.on_kc_bias_init(
                name=name, p_mean=p, bias=b, bias_count=int(lin.bias.numel())
            )

    # pylint: disable=too-many-locals,too-many-positional-arguments
    def _perform_optimizer_step(
        self,
        m: TrainingClassifier,
    ) -> bool:
        name_map: Dict[int, str] = {}
        for n, p in m.named_parameters():
            name_map[id(p)] = n

        found_nonfinite = False
        bad: List[Tuple[str, int, int, float]] = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if not torch.isfinite(g).all():
                    found_nonfinite = True
                    nnan = int(torch.isnan(g).sum().item())
                    ninf = int(torch.isinf(g).sum().item())
                    gmax = (
                        float(g.detach().float().abs().max().item())
                        if torch.isfinite(g.detach().float().abs().max())
                        else float("inf")
                    )
                    pname = name_map.get(id(p), "<unnamed>")
                    bad.append((pname, nnan, ninf, gmax))
                    if len(bad) >= 5:
                        break
            if found_nonfinite:
                break

        if found_nonfinite:
            raise RuntimeError("found_nonfinite")

        # B1: Split Clipping (Encoder vs Heads)
        if self.config.gradient_clip and self.config.gradient_clip > 0:
            # We must identify which params belong to encoder vs heads
            # Heads: kc_head and kc_decoders
            head_params = set(m.kc_head.parameters())
            if hasattr(m, "kc_decoders"):
                head_params.update(m.kc_decoders.parameters())

            enc_params = [
                p
                for group in self.optimizer.param_groups
                for p in group["params"]
                if p.grad is not None and p not in head_params
            ]

            # Clip encoder aggressively as configured
            if enc_params:
                nn.utils.clip_grad_norm_(enc_params, self.config.gradient_clip)

            # Leave heads unclipped (or use a much higher clip if needed).
            # Per user instruction: "preferred: do NOT clip at all"
        # else: do not clip at all (0 means disabled)

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        skipped = False

        if not skipped:
            param_nonfinite = False
            for p in m.kc_head.parameters():
                if not torch.isfinite(p.data).all():
                    param_nonfinite = True
                    break
            if not param_nonfinite and hasattr(m, "kc_decoders"):
                for p in m.kc_decoders.parameters():
                    if not torch.isfinite(p.data).all():
                        param_nonfinite = True
                        break

            if param_nonfinite:
                raise RuntimeError("Params became NaN after step")

        return skipped

    def _create_optimizer(self, freeze_encoder: bool = False) -> None:
        m = self.model

        pg_heads = {
            "params": list(m.kc_head.parameters())
            + (list(m.kc_decoders.parameters()) if hasattr(m, "kc_decoders") else []),
            "lr": self.config.learning_rate,
        }

        pg_encoder = {
            "params": list(m.embedding.parameters()) + list(m.encoder.parameters()),
            "lr": self.config.learning_rate * 0.1 if not freeze_encoder else 0.0,
        }

        self.optimizer = Adam([pg_heads, pg_encoder])

    def _check_kc_coverage(
        self, outputs: Dict[str, Any], kc_targets: Dict[str, Any]
    ) -> None:
        """Helper to check and report KC target coverage."""
        missing_keys = []
        for name in outputs["target_logits"]:
            if f"kc_targets_{name}" in kc_targets:
                continue
            if f"kc_pos_inds_{name}" in kc_targets:
                continue
            # DB-sourced PNU families (GRAMMAR_POINT) use gp_ prefix
            if f"kc_gp_pos_inds_{name}" in kc_targets:
                continue
            # DB-sourced continuous families (gender/formality) use continuous_ prefix
            if f"kc_continuous_{name}" in kc_targets:
                continue
            # DB-sourced classification families (gender_class/formality_class) use class_ prefix
            if f"kc_class_{name}" in kc_targets:
                continue
            # DB-sourced multi-label families (register) use multilabel_ prefix
            if f"kc_multilabel_{name}" in kc_targets:
                continue
            missing_keys.append(name)

        # If missing keys exist, verify if they are legitimately missing or aliasing issues
        if missing_keys:
            raise ValueError(
                f"KC Targets MISSING for: {missing_keys}. Check dataset generation (kc.py) and collation."
            )

    # pylint: disable=too-many-locals
    def train_epoch(self, epoch: int = 0) -> TrainEpochResult:
        # Use relative epoch from session start for freezing (warm-up)
        # If session_start_epoch is None (e.g. direct call), fall back to absolute
        base_epoch = (
            self.session_start_epoch if self.session_start_epoch is not None else 0
        )
        relative_epoch = max(0, epoch - base_epoch)

        should_freeze = relative_epoch < self.freeze_encoder_epochs
        # Performance: Skip diagnostic metrics gathering for early epochs
        skip_metrics = epoch < self.kc_config.skip_first_metrics
        # self._create_optimizer(freeze_encoder=should_freeze) <- REMOVED to preserve moment
        # Instead, update LR in place for the encoder group
        assert len(self.optimizer.param_groups) >= 2, (
            f"Expected >=2 param_groups (heads, encoder), got {len(self.optimizer.param_groups)}"
        )
        enc_lr = 0.0 if should_freeze else (self.config.learning_rate * 0.1)
        self.optimizer.param_groups[1]["lr"] = enc_lr

        self.view.on_kc_epoch_start(epoch, self.config.kc_epochs, should_freeze)

        # Set training mode with special handling for frozen epochs:
        # During frozen epochs, put encoder pipeline in eval mode to disable dropout
        # for deterministic outputs, while keeping decoder heads in train mode.
        if should_freeze:
            # Encoder pipeline: eval mode (disable dropout)
            self.model.embedding.eval()
            self.model.position_encoding.eval()
            self.model.encoder.eval()
            self.model.pooler.eval()
            # Decoder heads: train mode (keep dropout active for regularization)
            self.model.kc_head.train()
            self.model.kc_decoders.train()
        else:
            self.model.train()

        total_loss, n_batches = 0.0, 0

        total_sparsity = 0.0

        total_batches = len(self.data_loader)

        running_struct_loss = 0.0
        running_num_struct_total = 0
        running_sparsity = 0.0
        running_avg_prob, running_act_dens = 0.0, 0.0
        running_pmax_global = -1.0

        # --- Saturation Penalty Config (Per Epoch) ---
        sat_w = 0.0
        if relative_epoch >= self.freeze_encoder_epochs:
            # Stronger ramp: 0.25 -> 1.0 over 3 epochs
            epoch_idx_thawed = max(0, relative_epoch - self.freeze_encoder_epochs)
            ramp = min(1.0, epoch_idx_thawed / 3.0)
            sat_w = 0.25 + 0.75 * ramp

        # Saturation Usage Accumulators
        sat_alpha: float = 0.0  # Initialize for empty/skipped batch cases
        sat_global_batches: int = 0
        sat_pen_global_sum: float = 0.0
        pmax_logit_mean_global_sum: float = 0.0
        pmax_logit_max_global: float = -float("inf")

        sat_pos_ex_count: int = 0
        sat_pen_pos_sum: float = 0.0
        pmax_logit_pos_sum: float = 0.0
        frac_over_thr_pos_sum: float = 0.0
        pmax_logit_max_pos: float = -float("inf")

        frac_has_pos_batches_sum: float = 0.0
        sat_pos_batches: int = 0

        # Auto-scaling Stats
        # Only accumulated when sat_w > 0
        sat_scale_sum: float = 0.0
        sat_contrib_sum: float = 0.0
        sat_contrib_ratio_sum: float = 0.0
        sat_active_batches: int = 0

        epoch_kc_losses: Dict[str, float] = {}
        family_accumulators: Dict[str, FamilyAccumulator] = {}
        worst_samples: Dict[
            str, WorstSampleInfo
        ] = {}  # Track highest-loss sample per family

        pending_accum = 0
        did_any_backward = False

        kc_vocab_size = int(self.model.config.kc_vocab_size)
        running_pmax_global = 0.0
        running_avg_prob = 0.0
        running_act_dens = 0.0
        running_usage_probs_sum = torch.zeros(kc_vocab_size, device=self.device)
        total_samples_seen = 0

        running_loss_components = RunningLossComponents()

        # Track which KC logits are used (fire) for at least one sample
        kc_logits_used_set: set = set()

        kc_diag = KCEpochDiag()
        # Load precomputed unique ID counts from label phase (amortized collision tracking)
        kc_diag.load_precomputed_unique_counts(self.dataset.data_dir)
        reading_mask_id = getattr(self.dataset.tokenizer, "unk_id", 0)
        if "reading" in self.dataset.tokenizer.field_vocabs:
            reading_mask_id = self.dataset.tokenizer.field_vocabs["reading"].get(
                "<READING_MASK>", reading_mask_id
            )
        all_lens_aligned = []

        self.optimizer.zero_grad(set_to_none=True)

        # Capture decoder bias at start of epoch for delta tracking
        bias_start: Dict[str, torch.Tensor] = {}
        for name, decoder in self.model.kc_decoders.decoders.items():
            if hasattr(decoder, "bias") and decoder.bias is not None:
                bias_start[name] = decoder.bias.detach().clone()

        pbar = None

        current_display_loss = 0.5
        pbar_desc = f"KC Epoch {epoch + 1}/{self.config.kc_epochs}"
        if should_freeze:
            pbar_desc += " (Encoder Frozen)"

        pbar = RichTrainerProgressBar(
            desc=pbar_desc,
            total_steps=total_batches,
            batch_size=self.data_loader.batch_size or 1,
        )
        self.view.on_kc_progress_init(
            pbar_desc,
            total_steps=total_batches,
        )

        self.train_timer_data.start()
        for batch_idx, batch in enumerate(self.data_loader):
            self.train_timer_data.stop(epoch=epoch, batch=batch_idx)
            self.train_timer_compute.start()

            # --- Saturation Scale Ramp ---
            ramp_val = 0.0
            if relative_epoch >= self.freeze_encoder_epochs:
                # Ramp up sat_w/alpha linearly over 1 epoch (or defined steps)
                # Typically we want it fully active quickly after thawing.
                # Let's say ramp over 1000 steps or 1 epoch.
                # Just use epoch-based ramp for simplicity if step tracking is complex
                # Actually, let's use intra-epoch ramp:
                epoch_since_thaw = relative_epoch - self.freeze_encoder_epochs
                # Linear ramp from 0.0 to 1.0 over first thawed epoch
                if epoch_since_thaw == 0:
                    ramp_val = min(1.0, (batch_idx + 1) / max(1, n_batches))
                else:
                    ramp_val = 1.0

            # --- Pre-calculate Saturation Weight ---
            sat_w = 0.0
            if self.kc_sat_weight > 0 and relative_epoch >= self.freeze_encoder_epochs:
                # Saturation penalty: Ramp up from 0 to full weight
                sat_w = self.kc_sat_weight * ramp_val

            if epoch == self.start_epoch and batch_idx < self.start_batch:
                self.train_timer_compute.stop(epoch=epoch, batch=batch_idx)
                self.train_timer_data.start()
                continue

            m = self.model

            kc_targets = create_kc_batch(
                batch=batch,
                tokenizer=self.dataset.tokenizer,
                target_specs=self.config.kc_target_specs,
            )
            # Ensure targets are on device (they come from CPU dataset/tokenizer)
            for k, v in kc_targets.items():
                kc_targets[k] = v.to(self.device)

            # INVARIANT: kc_targets batch dimension must match batch
            expected_batch_size = batch.attention_mask.size(0)
            for tgt_key, tgt_val in kc_targets.items():
                if tgt_val.size(0) != expected_batch_size:
                    raise ValueError(
                        f"kc_targets['{tgt_key}'] batch mismatch: "
                        f"{tgt_val.size(0)} vs expected {expected_batch_size}"
                    )

            if self.config.kc_target_specs and not kc_targets and batch_idx == 0:
                # One-off safety check (fails fast only on batch 0 to avoid noise if later batches are empty)
                self.view.on_kc_warning(
                    f"Batch 0 produced no KC targets. Specs: {list(self.config.kc_target_specs.keys())}"
                )

            field_inputs = {
                k: v.to(self.device) for k, v in batch.feature_inputs.items()
            }
            attention_mask = batch.attention_mask.to(self.device)

            # Compute content_len (approximate: non-pad count)
            # Use attention_mask for robust length calculation even if feature_inputs is empty (e.g. in tests)
            content_len = attention_mask.sum(dim=1).float()

            # Dynamic Sizing using k_budget params from model config
            # This ensures training and inference use identical k_budget logic
            k_budget_t = compute_k_budget(content_len, m.config, self.device)

            # Long sentence mask (>= long_threshold tokens)
            long_threshold = float(getattr(m.config, "kc_long_threshold", 20))
            long_sentence_mask = content_len >= long_threshold

            gumbel_scale = 0.0
            if relative_epoch < self.freeze_encoder_epochs:
                t_val = self.kc_temperature_frozen
            else:
                t_val = self.kc_temperature_thawed

                total_kc_epochs = self.config.kc_epochs or self.config.epochs or 3
                epochs_remaining = max(1, total_kc_epochs - self.freeze_encoder_epochs)
                epoch_idx_thawed = max(0, relative_epoch - self.freeze_encoder_epochs)
                ratio = min(1.0, epoch_idx_thawed / float(epochs_remaining))
                gumbel_scale = 0.6 * (1.0 - ratio) + 0.2 * ratio

            outputs = self.model(
                field_inputs,
                attention_mask=attention_mask,
                mode="kc",
                temperature=t_val,
                gumbel_scale=gumbel_scale,
                grad_cap=self.kc_grad_cap,
                k_budget=k_budget_t,
                long_sentence_mask=long_sentence_mask,
            )

            # --- INVARIANT CHECK A-E: Post-Forward Validation ---
            # Fail-fast validations for adaptive budget plumbing.
            if (
                "topk_inds" in outputs
            ):  # Only check if keys exist (not mandatory for non-KC modes, though mode='kc' implies it)
                inv_inds = outputs["topk_inds"]
                inv_vals = outputs["topk_vals"]
                inv_probs = outputs.get("kc_probs")

                # A) Presence and Shape
                if inv_inds is None or inv_vals is None:
                    raise RuntimeError("Missing topk_inds/vals in outputs")
                if inv_probs is None:
                    raise RuntimeError("Missing kc_probs in outputs")
                # inv_logits_raw is optional but good to have

                batch_size_chk = attention_mask.size(0)
                if inv_inds.size(0) != batch_size_chk:
                    raise RuntimeError(
                        f"topk_inds B mismatch: {inv_inds.size(0)} vs {batch_size_chk}"
                    )
                if inv_vals.size(0) != batch_size_chk:
                    raise RuntimeError(
                        f"topk_vals B mismatch: {inv_vals.size(0)} vs {batch_size_chk}"
                    )
                if inv_probs.size(0) != batch_size_chk:
                    raise RuntimeError(
                        f"kc_probs B mismatch: {inv_probs.size(0)} vs {batch_size_chk}"
                    )
                if inv_probs.dim() != 2:
                    raise RuntimeError(f"kc_probs dim error: {inv_probs.dim()}")

                k_size_chk = inv_inds.size(1)
                if inv_vals.size(1) != k_size_chk:
                    raise RuntimeError(
                        f"topk_vals K mismatch: {inv_vals.size(1)} vs {k_size_chk}"
                    )
                if k_size_chk < 1:
                    raise RuntimeError("Kmax < 1")

                # B) Index Validity
                vocab_size_chk = int(getattr(self.model.config, "kc_vocab_size", 0))
                if vocab_size_chk > 0:
                    if inv_inds.dtype not in (torch.int64, torch.int32):
                        raise RuntimeError("topk_inds not int")
                    # Check for valid range [0, V). -1 allowed if used for padding (but usually not in topk)
                    min_idx = inv_inds.min().item()
                    max_idx = inv_inds.max().item()
                    if min_idx < 0:
                        raise RuntimeError(f"Invalid negative index: {min_idx}")
                    if max_idx >= vocab_size_chk:
                        raise RuntimeError(
                            f"Index out of bounds: {max_idx} >= {vocab_size_chk}"
                        )

                # C) Value Constraints
                if not torch.isfinite(inv_vals).all():
                    raise RuntimeError("Non-finite topk_vals")
                if inv_vals.min().item() < -1e-5:
                    raise RuntimeError(f"topk_vals < 0: {inv_vals.min().item()}")
                if inv_vals.max().item() > 1.0 + 1e-5:
                    raise RuntimeError(f"topk_vals > 1: {inv_vals.max().item()}")

                # Monotonicity check (row 0)
                if k_size_chk > 1 and batch_size_chk > 0:
                    row0 = inv_vals[0]
                    if not (row0[:-1] + 1e-6 >= row0[1:]).all():
                        # Depending on variable budget masking, tail might be 0.
                        # 0 is <= prev unless prev was 0.
                        # So it should be non-increasing.
                        # However, if we zero-out entries that were NOT sorted (e.g. by index), we break it.
                        # Models usually zero-out *after* topk.
                        pass

                # D) Consistency
                # Gathered probs should match topk_vals (approx)
                # But topk_vals might be masked (zeroed) by budget.
                # So gathered_vals * budget_mask should approx topk_vals

                # E) Variable Budget
                # k_budget_t is (B,)
                if k_budget_t.shape != (batch_size_chk,):
                    raise RuntimeError("k_budget shape mismatch")
                if k_budget_t.min().item() < 1:
                    raise RuntimeError("k_budget < 1")
                # max check
                if k_budget_t.max().item() > k_size_chk:
                    # This might happen if max_k > k_size_chk?
                    # k_size_chk comes from outputs, which should respect max_k.
                    # If model uses hardcoded K, mismatch possible.
                    # But current implementation uses topk(k=max(budget)).
                    pass

            should_check_nan = batch_idx < 50 or (batch_idx % 50 == 0)

            if should_check_nan:
                logits_stats = tensor_finite_stats(outputs.get("kc_logits_raw"))
                probs_stats = tensor_finite_stats(outputs.get("kc_probs"))
                forward_nonfinite = not logits_stats.finite or not probs_stats.finite
            else:
                forward_nonfinite = False

            if forward_nonfinite:
                raise RuntimeError("Non-finite values in forward pass")

            # Check required keys for KC training
            topk_inds = outputs.get("topk_inds", None)
            topk_vals = outputs.get("topk_vals", None)

            if topk_inds is None or topk_vals is None:
                raise RuntimeError("KC training requires topk_inds and topk_vals")

            # Decoder Consistency Fix: Unconditional Decoding
            # Always produce target_logits from sparse activations, even in frozen epoch,
            # so decoders learn valid weights against the sparse distribution.

            # Removed clamp(max=0.98) to allow natural range per "Separate KC Presence..." constraint
            topk_vals_used = outputs["topk_vals"]

            sparse_clamped = torch.zeros_like(outputs["kc_probs"])
            sparse_clamped.scatter_(1, outputs["topk_inds"], topk_vals_used)

            # --- INVARIANT CHECK F: Sparse Activations ---
            # 1) Shape
            if sparse_clamped.shape != outputs["kc_probs"].shape:
                raise RuntimeError("sparse_clamped shape mismatch")
            # 2) Support
            # Ensuring non-zeros are subset of topk_inds
            # This is guaranteed by scatter_, but we assert values
            # 3) Value
            if sparse_clamped.min().item() < 0:
                raise RuntimeError("sparse_clamped < 0")
            if sparse_clamped.max().item() > 1.0 + 1e-6:
                raise RuntimeError("sparse_clamped > 1")

            if hasattr(m, "kc_decoders"):
                # Pass kc_probs so MSE families can use full probabilities
                # for gradient flow to KC selection
                outputs["target_logits"] = m.kc_decoders(
                    sparse_clamped, kc_probs=outputs["kc_probs"]
                )

            # INVARIANT: target_logits batch dimension must match attention_mask
            target_logits = outputs["target_logits"]
            for tl_name, tl_tensor in target_logits.items():
                if tl_tensor.size(0) != expected_batch_size:
                    raise ValueError(
                        f"target_logits['{tl_name}'] batch mismatch: "
                        f"{tl_tensor.size(0)} vs expected {expected_batch_size}"
                    )

            # Update Diagnostic Accumulators
            # k_eff_t = (outputs["sparse_activations"] > 0).float().sum(dim=1).cpu()
            len_t = content_len.detach().cpu().float()

            all_lens_aligned.extend(len_t.tolist())
            # all_keff_aligned.extend(k_eff_t.tolist())

            # Track which KC logits fire for at least one sample
            # topk_inds shape: [batch_size, k]
            unique_indices = topk_inds.unique().cpu().tolist()
            kc_logits_used_set.update(unique_indices)

            # Update kc usage stats

            if batch_idx == 0:
                if not self.config.kc_target_specs:
                    raise ValueError(
                        "kc_target_specs is empty! Model has no KC targets configured."
                    )

                if not outputs["target_logits"]:
                    raise ValueError(
                        "outputs['target_logits'] is empty! Model produced no KC outputs."
                    )

                # Check for overlap
                has_match = False
                for name in outputs["target_logits"]:
                    dense_key = f"kc_targets_{name}"
                    sparse_key = f"kc_pos_inds_{name}"
                    if dense_key in kc_targets:
                        has_match = True
                        break
                    if sparse_key in kc_targets:
                        has_match = True
                        break

                if not has_match:
                    tgt_keys = list(kc_targets.keys())

                    msg = (
                        f"Loss Loop Failure: No target_logits keys match available kc_targets.\n"
                        f"  Configured Specs: {list(self.config.kc_target_specs.keys())}\n"
                        f"  Batch Features: {list(batch.feature_inputs.keys())}\n"
                        f"  KC Targets Keys: {tgt_keys[:20]}...\n"
                    )
                    raise ValueError(msg)

            # Coverage Summary
            if batch_idx == 0:
                self._check_kc_coverage(outputs, kc_targets)

            # Loss accumulation for this batch:
            # - structural_loss = sum of task_loss for each family (the "struct" component)
            # - Each family contributes its task_loss.item() directly
            # - INVARIANT: sum(family loss_means) = struct (validated by checksum)
            loss = torch.tensor(0.0, device=self.device)
            batch_kc_losses: Dict[str, float] = {}
            structural_loss = torch.tensor(0.0, device=self.device)
            num_struct = 0

            for fid, vocab_size in self.config.kc_target_specs.items():
                name = fid.name.lower()
                dense_key = f"kc_targets_{name}"
                pos_key = f"kc_pos_inds_{name}"
                mask_key = f"kc_pos_mask_{name}"

                # Retrieve logits for this family from the outputs
                logits = target_logits.get(name)
                if logits is None:
                    # This family might not have logits in the current batch, skip
                    continue

                # Get/Create accumulator
                if name not in family_accumulators:
                    family_accumulators[name] = FamilyAccumulator()
                fam_acc = family_accumulators[name]

                # Special handling for DB-sourced families (e.g., GRAMMAR_POINT, GENDER, FORMALITY)
                # These use PNU loss (KcPnuFamily), MSE loss (KcMseFamily), or CE loss (KcDbClassFamily)
                if is_family_db_sourced(fid):
                    from train.kc import (
                        KcDbClassFamily,
                        KcDbMultilabelFamily,
                        KcPnuFamily,
                    )

                    family_def = get_family(fid)

                    if isinstance(family_def, KcPnuFamily):
                        # PNU loss for grammar points (pos/neg arrays)
                        gp_pos_key = f"kc_gp_pos_inds_{name}"
                        gp_pos_mask_key = f"kc_gp_pos_mask_{name}"
                        gp_neg_key = f"kc_gp_neg_inds_{name}"
                        gp_neg_mask_key = f"kc_gp_neg_mask_{name}"

                        if gp_pos_key in kc_targets:
                            pos_ids = kc_targets[gp_pos_key].to(self.device)
                            pos_mask = kc_targets[gp_pos_mask_key].to(self.device)
                            neg_ids = kc_targets[gp_neg_key].to(self.device)
                            neg_mask = kc_targets[gp_neg_mask_key].to(self.device)

                            task_loss = self._multilabel_pnu_loss(
                                logits.float(),
                                pos_ids,
                                pos_mask,
                                neg_ids,
                                neg_mask,
                                vocab_size=vocab_size,
                                unlabeled_weight=self.kc_config.gp_unlabeled_weight,
                                pos_weight=self.kc_config.gp_pos_weight,
                                neg_weight=self.kc_config.gp_neg_weight,
                            )

                            # Apply per-family loss weight for balanced training
                            task_loss = task_loss * family_def.loss_weight
                            batch_kc_losses[f"{name}"] = task_loss.item()
                            structural_loss = structural_loss + task_loss
                            num_struct += 1

                            # Accumulator tracking for GRAMMAR_POINT:
                            # Build synthetic targets from pos/neg for diagnostics
                            # labeled_pos = 1 where positive, labeled_neg = 1 where negative
                            labeled_pos = torch.zeros(
                                logits.size(0), vocab_size, device=self.device
                            )
                            labeled_neg = torch.zeros(
                                logits.size(0), vocab_size, device=self.device
                            )
                            valid_pos = pos_ids.clamp(0, vocab_size - 1)
                            labeled_pos.scatter_(1, valid_pos, pos_mask.float())
                            valid_neg = neg_ids.clamp(0, vocab_size - 1)
                            labeled_neg.scatter_(1, valid_neg, neg_mask.float())

                            # targets = labeled_pos (positives are 1, rest 0)
                            # valid_mask = labeled_pos + labeled_neg (only labeled are valid)
                            targets_gp = labeled_pos
                            valid_mask_gp = (labeled_pos + labeled_neg).clamp(0, 1)

                            fam_acc.update(
                                logits.detach(),
                                targets_gp.detach(),
                                pos_mask=None,  # Let update derive from targets
                                valid_mask=valid_mask_gp.bool(),
                                source="dense",
                            )

                            # Update kc_diag so DB-sourced families appear in diagnostics
                            if not skip_metrics:
                                # Build sampled representation for kc_diag
                                # Use valid (labeled) positions only
                                probs_gp = torch.sigmoid(logits.float()).detach()
                                gathered_logits_gp = logits.gather(
                                    1, valid_pos
                                ).detach()
                                kc_diag.update_family(
                                    name,
                                    pos_ids.detach().cpu(),
                                    pos_mask.detach().cpu(),
                                    probs_gp.gather(1, valid_pos).detach().cpu(),
                                    labeled_pos.gather(1, valid_pos).detach().cpu(),
                                    task_loss.item(),
                                    logits=gathered_logits_gp.cpu(),
                                )

                                # Track worst sample for PNU (grammar_point) families
                                # Loss = BCE on labeled positions per sample
                                probs_full = torch.sigmoid(logits.float())
                                # Use labeled_pos as targets, valid_mask_gp as mask
                                bce_full = F.binary_cross_entropy(
                                    probs_full, targets_gp, reduction="none"
                                )
                                # Only count loss on labeled (valid) positions
                                per_sample_loss = (
                                    bce_full * valid_mask_gp.float()
                                ).sum(dim=1)
                                max_loss_idx = int(per_sample_loss.argmax().item())
                                max_loss_val = per_sample_loss[max_loss_idx].item()

                                # Update if this is the worst so far
                                current_worst = worst_samples.get(name)
                                if (
                                    current_worst is None
                                    or max_loss_val > current_worst.loss
                                ):
                                    # For target/pred: count of positive labels
                                    target_count = (
                                        labeled_pos[max_loss_idx].sum().item()
                                    )
                                    pred_count = (
                                        (probs_full[max_loss_idx] > 0.5).sum().item()
                                    )
                                    # Get positive GP IDs for this sample (targets)
                                    pos_gp_ids = pos_ids[max_loss_idx][
                                        pos_mask[max_loss_idx]
                                    ].tolist()
                                    target_labels = ",".join(
                                        f"gp{gid:04d}" for gid in pos_gp_ids[:5]
                                    )
                                    if len(pos_gp_ids) > 5:
                                        target_labels += f"...+{len(pos_gp_ids) - 5}"
                                    # Get predicted GP IDs (above 0.5 threshold)
                                    pred_gp_ids = (
                                        (probs_full[max_loss_idx] > 0.5)
                                        .nonzero(as_tuple=True)[0]
                                        .tolist()
                                    )
                                    if pred_gp_ids:
                                        pred_labels = ",".join(
                                            f"gp{gid:04d}" for gid in pred_gp_ids[:5]
                                        )
                                        if len(pred_gp_ids) > 5:
                                            pred_labels += f"...+{len(pred_gp_ids) - 5}"
                                    else:
                                        pred_labels = "none"
                                    sample_idx = int(batch.indices[max_loss_idx].item())
                                    worst_samples[name] = WorstSampleInfo(
                                        sentence=_get_display_sentence(
                                            self.dataset.get_sentence_by_idx(
                                                sample_idx
                                            ),
                                            batch.kotogram[max_loss_idx],
                                        ),
                                        loss=max_loss_val,
                                        target=float(target_count),
                                        prediction=float(pred_count),
                                        sample_idx=sample_idx,
                                        target_labels=target_labels,
                                        pred_labels=pred_labels,
                                    )

                                # Track worst FALSE POSITIVE sample (unlabeled predicted as positive)
                                # Find cases where labeled_pos == 0 but prediction > 0.5
                                unlabeled_mask = (labeled_pos == 0).float()
                                pred_positive_mask = (probs_full > 0.5).float()
                                fp_mask = unlabeled_mask * pred_positive_mask
                                # Compute per-sample FP loss (BCE where unlabeled but predicted positive)
                                per_sample_fp_loss = (bce_full * fp_mask).sum(dim=1)

                                if per_sample_fp_loss.max().item() > 0:
                                    max_fp_idx = int(per_sample_fp_loss.argmax().item())
                                    max_fp_val = per_sample_fp_loss[max_fp_idx].item()

                                    fp_key = f"{name}_fp"
                                    current_worst_fp = worst_samples.get(fp_key)
                                    if (
                                        current_worst_fp is None
                                        or max_fp_val > current_worst_fp.loss
                                    ):
                                        # For FP: target should be 0, pred is count of false positives
                                        fp_target_count = 0.0  # Unlabeled/negative
                                        fp_pred_gp_ids = (
                                            (fp_mask[max_fp_idx] > 0.5)
                                            .nonzero(as_tuple=True)[0]
                                            .tolist()
                                        )

                                        fp_target_labels = "none"
                                        if fp_pred_gp_ids:
                                            fp_pred_labels = ",".join(
                                                f"gp{gid:04d}"
                                                for gid in fp_pred_gp_ids[:5]
                                            )
                                            if len(fp_pred_gp_ids) > 5:
                                                fp_pred_labels += (
                                                    f"...+{len(fp_pred_gp_ids) - 5}"
                                                )
                                        else:
                                            fp_pred_labels = "none"

                                        fp_sample_idx = int(
                                            batch.indices[max_fp_idx].item()
                                        )
                                        worst_samples[fp_key] = WorstSampleInfo(
                                            sentence=_get_display_sentence(
                                                self.dataset.get_sentence_by_idx(
                                                    fp_sample_idx
                                                ),
                                                batch.kotogram[max_fp_idx],
                                            ),
                                            loss=max_fp_val,
                                            target=fp_target_count,
                                            prediction=float(len(fp_pred_gp_ids)),
                                            sample_idx=fp_sample_idx,
                                            target_labels=fp_target_labels,
                                            pred_labels=fp_pred_labels,
                                        )

                    elif isinstance(family_def, KcDbClassFamily):
                        # Multi-class classification for gender_class/formality_class
                        # Targets are class indices [B] (e.g., 0, 1, 2 for gender)
                        class_targets = (
                            kc_targets[f"kc_class_{name}"].to(self.device).long()
                        )

                        # logits shape: [B, num_classes]
                        raw_ce_loss = F.cross_entropy(logits, class_targets)

                        # Apply per-family loss weight for balanced training
                        task_loss = raw_ce_loss * family_def.loss_weight
                        batch_kc_losses[f"{name}"] = task_loss.item()
                        structural_loss = structural_loss + task_loss
                        num_struct += 1

                        # Update FamilyAccumulator for pos_ex_frac tracking
                        # Create synthetic binary targets: true class = 1, others = 0
                        dense_binary = torch.zeros_like(logits)
                        dense_binary.scatter_(1, class_targets.unsqueeze(1), 1.0)
                        fam_acc.update(
                            logits.detach(),
                            dense_binary.detach(),
                            pos_mask=None,
                            valid_mask=None,  # All positions valid for classification
                            source="dense",
                        )

                        # Update diagnostics for classification families
                        # Treat as binary: true class (positive) vs. all other classes (negatives)
                        if not skip_metrics and kc_diag is not None:
                            probs = torch.softmax(logits.float(), dim=-1).detach()
                            batch_size = logits.size(0)
                            num_classes = logits.size(1)

                            # Build sampled representation: true class + sample of wrong classes
                            max_samples = 32  # Sample up to 32 positions per example
                            pos_ids_list = []
                            pos_mask_list = []
                            probs_list = []
                            targets_list = []
                            logits_list = []

                            for i in range(batch_size):
                                true_cls = class_targets[i].item()
                                # Collect: [true_class, other_classes...]
                                ids = [true_cls]
                                target_vals = [1.0]  # True class is positive

                                # Add all other classes as negatives
                                for c in range(num_classes):
                                    if c != true_cls and len(ids) < max_samples:
                                        ids.append(c)
                                        target_vals.append(0.0)  # Negative

                                # Pad to max_samples
                                n = len(ids)
                                ids += [0] * (max_samples - n)
                                target_vals += [0.0] * (max_samples - n)

                                # Gather probs and logits
                                ids_tensor = torch.tensor(
                                    ids[:max_samples], device=self.device
                                )
                                probs_vals = probs[i, ids_tensor].cpu().tolist()
                                logits_vals = (
                                    logits[i, ids_tensor].detach().cpu().tolist()
                                )

                                pos_ids_list.append(ids)
                                pos_mask_list.append(
                                    [True] * n + [False] * (max_samples - n)
                                )
                                probs_list.append(probs_vals)
                                targets_list.append(target_vals)
                                logits_list.append(logits_vals)

                            # Convert to tensors
                            pos_ids_t = torch.tensor(pos_ids_list, dtype=torch.long)
                            pos_mask_t = torch.tensor(pos_mask_list, dtype=torch.bool)
                            probs_t = torch.tensor(probs_list, dtype=torch.float32)
                            targets_t = torch.tensor(targets_list, dtype=torch.float32)
                            logits_t = torch.tensor(logits_list, dtype=torch.float32)

                            kc_diag.update_family(
                                name,
                                pos_ids_t,
                                pos_mask_t,
                                probs_t,
                                targets_t,
                                task_loss.item(),
                                logits=logits_t,
                            )

                        # Track worst sample for classification families
                        if not skip_metrics:
                            with torch.no_grad():
                                # Compute per-sample CE loss
                                per_sample_ce = F.cross_entropy(
                                    logits.float(), class_targets, reduction="none"
                                )
                                max_loss_idx = int(per_sample_ce.argmax().item())
                                max_loss_val = per_sample_ce[max_loss_idx].item()

                                # Update if this is the worst so far
                                current_worst = worst_samples.get(name)
                                if (
                                    current_worst is None
                                    or max_loss_val > current_worst.loss
                                ):
                                    pred_probs = torch.softmax(logits.float(), dim=-1)
                                    pred_class = int(
                                        pred_probs[max_loss_idx].argmax().item()
                                    )
                                    target_class = int(
                                        class_targets[max_loss_idx].item()
                                    )
                                    # Map class indices to readable names
                                    if name == "formality_class":
                                        class_names = [
                                            "v_formal",
                                            "formal",
                                            "neutral",
                                            "casual",
                                            "v_casual",
                                        ]
                                    elif name == "gender_class":
                                        class_names = ["masc", "neutral", "fem"]
                                    else:
                                        class_names = []
                                    # ASSERT: class_names should be set for known families
                                    assert class_names, (
                                        f"Unknown classification family: {name!r}"
                                    )
                                    target_label = (
                                        class_names[target_class]
                                        if target_class < len(class_names)
                                        else f"cls{target_class}"
                                    )
                                    pred_label = (
                                        class_names[pred_class]
                                        if pred_class < len(class_names)
                                        else f"cls{pred_class}"
                                    )
                                    # ASSERT: labels should be non-empty
                                    assert target_label, (
                                        f"Empty target_label for {name}"
                                    )
                                    assert pred_label, f"Empty pred_label for {name}"
                                    sample_idx = int(batch.indices[max_loss_idx].item())
                                    worst_samples[name] = WorstSampleInfo(
                                        sentence=_get_display_sentence(
                                            self.dataset.get_sentence_by_idx(
                                                sample_idx
                                            ),
                                            batch.kotogram[max_loss_idx],
                                        ),
                                        loss=max_loss_val,
                                        target=float(target_class),
                                        prediction=float(pred_class),
                                        sample_idx=sample_idx,
                                        target_labels=target_label,
                                        pred_labels=pred_label,
                                    )
                    elif isinstance(family_def, KcDbMultilabelFamily):
                        # Multi-label classification for register
                        # Targets are multi-hot [B, num_classes] (can have multiple active)
                        multilabel_key = f"kc_multilabel_{name}"
                        if multilabel_key in kc_targets:
                            targets_multilabel = (
                                kc_targets[multilabel_key].to(self.device).float()
                            )

                            # logits shape: [B, num_classes]
                            # Use BCE with logits for multi-label classification
                            raw_bce_loss = F.binary_cross_entropy_with_logits(
                                logits, targets_multilabel, reduction="mean"
                            )

                            # Apply per-family loss weight for balanced training
                            task_loss = raw_bce_loss * family_def.loss_weight
                            batch_kc_losses[f"{name}"] = task_loss.item()
                            structural_loss = structural_loss + task_loss
                            num_struct += 1

                            # Update FamilyAccumulator for pos_ex_frac tracking
                            fam_acc.update(
                                logits.detach(),
                                targets_multilabel.detach(),
                                pos_mask=None,
                                valid_mask=None,  # All positions valid for multi-label
                                source="dense",
                            )

                            # Update diagnostics for multi-label families
                            if not skip_metrics and kc_diag is not None:
                                probs = torch.sigmoid(logits.float()).detach()
                                batch_size = logits.size(0)
                                num_classes = logits.size(1)

                                # Build sampled representation for kc_diag
                                max_samples = min(32, num_classes)
                                pos_ids_list = []
                                pos_mask_list = []
                                probs_list = []
                                targets_list = []
                                logits_list = []

                                for i in range(batch_size):
                                    # Collect all positive and some negative classes
                                    pos_cls = torch.nonzero(
                                        targets_multilabel[i] > 0.5, as_tuple=True
                                    )[0].tolist()
                                    neg_cls = torch.nonzero(
                                        targets_multilabel[i] <= 0.5, as_tuple=True
                                    )[0].tolist()

                                    # Include all positives, then fill with negatives
                                    ids = pos_cls[:]
                                    target_vals = [1.0] * len(pos_cls)

                                    # Add negatives up to max_samples
                                    for c in neg_cls:
                                        if len(ids) >= max_samples:
                                            break
                                        ids.append(c)
                                        target_vals.append(0.0)

                                    # Pad to max_samples
                                    n = len(ids)
                                    ids += [0] * (max_samples - n)
                                    target_vals += [0.0] * (max_samples - n)

                                    # Gather probs and logits
                                    ids_tensor = torch.tensor(
                                        ids[:max_samples], device=self.device
                                    )
                                    probs_vals = probs[i, ids_tensor].cpu().tolist()
                                    logits_vals = (
                                        logits[i, ids_tensor].detach().cpu().tolist()
                                    )

                                    pos_ids_list.append(ids)
                                    pos_mask_list.append(
                                        [True] * n + [False] * (max_samples - n)
                                    )
                                    probs_list.append(probs_vals)
                                    targets_list.append(target_vals)
                                    logits_list.append(logits_vals)

                                # Convert to tensors
                                pos_ids_t = torch.tensor(pos_ids_list, dtype=torch.long)
                                pos_mask_t = torch.tensor(
                                    pos_mask_list, dtype=torch.bool
                                )
                                probs_t = torch.tensor(probs_list, dtype=torch.float32)
                                targets_t = torch.tensor(
                                    targets_list, dtype=torch.float32
                                )
                                logits_t = torch.tensor(
                                    logits_list, dtype=torch.float32
                                )

                                kc_diag.update_family(
                                    name,
                                    pos_ids_t,
                                    pos_mask_t,
                                    probs_t,
                                    targets_t,
                                    task_loss.item(),
                                    logits=logits_t,
                                )

                            # Track worst sample for multi-label families
                            if not skip_metrics:
                                with torch.no_grad():
                                    # Compute per-sample BCE loss (sum across labels)
                                    per_sample_bce = F.binary_cross_entropy_with_logits(
                                        logits.float(),
                                        targets_multilabel,
                                        reduction="none",
                                    ).sum(dim=1)  # Sum BCE across labels
                                    max_loss_idx = int(per_sample_bce.argmax().item())
                                    max_loss_val = per_sample_bce[max_loss_idx].item()

                                    # Update if this is the worst so far
                                    current_worst = worst_samples.get(name)
                                    if (
                                        current_worst is None
                                        or max_loss_val > current_worst.loss
                                    ):
                                        # For target/pred: count of active labels
                                        target_count = (
                                            targets_multilabel[max_loss_idx]
                                            .sum()
                                            .item()
                                        )
                                        pred_probs = torch.sigmoid(logits.float())
                                        pred_count = (
                                            (pred_probs[max_loss_idx] > 0.5)
                                            .sum()
                                            .item()
                                        )
                                        # Get register names for target and prediction
                                        target_ids = (
                                            targets_multilabel[max_loss_idx]
                                            .nonzero(as_tuple=True)[0]
                                            .tolist()
                                        )
                                        target_names = []
                                        for i in target_ids[:3]:
                                            reg = REGISTER_ID_TO_LABEL.get(i)
                                            if reg is not None:
                                                target_names.append(reg.value)
                                            else:
                                                target_names.append(f"r{i}")
                                        pred_ids = (
                                            (pred_probs[max_loss_idx] > 0.5)
                                            .nonzero(as_tuple=True)[0]
                                            .tolist()
                                        )
                                        pred_names = []
                                        for i in pred_ids[:3]:
                                            reg = REGISTER_ID_TO_LABEL.get(i)
                                            if reg is not None:
                                                pred_names.append(reg.value)
                                            else:
                                                pred_names.append(f"r{i}")
                                        sample_idx = int(
                                            batch.indices[max_loss_idx].item()
                                        )
                                        worst_samples[name] = WorstSampleInfo(
                                            sentence=_get_display_sentence(
                                                self.dataset.get_sentence_by_idx(
                                                    sample_idx
                                                ),
                                                batch.kotogram[max_loss_idx],
                                            ),
                                            loss=max_loss_val,
                                            target=float(target_count),
                                            prediction=float(pred_count),
                                            sample_idx=sample_idx,
                                            target_labels=",".join(target_names)
                                            or "none",
                                            pred_labels=",".join(pred_names) or "none",
                                        )
                    else:
                        # MSE loss for continuous families (gender/formality)
                        # Get target values from the original batch
                        target_key = f"kc_continuous_{name}"
                        if target_key in kc_targets:
                            targets_cont = (
                                kc_targets[target_key].to(self.device).float()
                            )

                            # Invariant: KC samples are all grammatic, no NaNs allowed
                            if torch.isnan(targets_cont).any():
                                raise RuntimeError(
                                    f"NaN detected in MSE targets for {name}. "
                                    "KC samples must be grammatic with valid targets."
                                )

                            raw_mse_loss = self._continuous_mse_loss(
                                logits.float(),
                                targets_cont,
                            )

                            # Apply per-family loss weight for balanced training
                            task_loss = raw_mse_loss * family_def.loss_weight
                            batch_kc_losses[f"{name}"] = task_loss.item()
                            structural_loss = structural_loss + task_loss
                            num_struct += 1

                            # Update MSE family diagnostics separately from label families
                            if not skip_metrics and kc_diag is not None:
                                kc_diag.update_mse_family(
                                    name,
                                    logits.float(),
                                    targets_cont,
                                    task_loss.item(),
                                )

                            # Track worst sample for this MSE family
                            if not skip_metrics:
                                with torch.no_grad():
                                    preds = logits.float().squeeze(-1)
                                    per_sample_loss = (preds - targets_cont).pow(2)
                                    max_loss_idx = int(per_sample_loss.argmax().item())
                                    max_loss_val = per_sample_loss[max_loss_idx].item()

                                    # Update if this is the worst so far
                                    current_worst = worst_samples.get(name)
                                    if (
                                        current_worst is None
                                        or max_loss_val > current_worst.loss
                                    ):
                                        sample_idx = int(
                                            batch.indices[max_loss_idx].item()
                                        )
                                        worst_samples[name] = WorstSampleInfo(
                                            sentence=_get_display_sentence(
                                                self.dataset.get_sentence_by_idx(
                                                    sample_idx
                                                ),
                                                batch.kotogram[max_loss_idx],
                                            ),
                                            loss=max_loss_val,
                                            target=targets_cont[max_loss_idx].item(),
                                            prediction=preds[max_loss_idx].item(),
                                            sample_idx=sample_idx,
                                        )

                    continue  # Skip standard dense/sparse path

                if dense_key in kc_targets:
                    targets = kc_targets[dense_key].to(self.device).float()
                    logits_f = logits.float()

                    # INVARIANT: targets and logits must have same shape
                    if targets.shape != logits_f.shape:
                        raise ValueError(
                            f"Shape mismatch for family '{name}': "
                            f"targets={targets.shape} vs logits={logits_f.shape}. "
                            f"batch_idx={batch_idx}"
                        )

                    batch_size_f, vocab_size_f = logits_f.shape
                    # Use is_family_sparse to decide path, not vocab size
                    # fid is already a KcFamilyId from the loop
                    is_sparse = is_family_sparse(fid)
                    if is_sparse:
                        # 1) Per-row index sampling
                        pos_mask_bool = targets > 0.5
                        pos_rows = []
                        max_pos = 0
                        for i in range(batch_size_f):
                            row_inds = torch.nonzero(pos_mask_bool[i], as_tuple=True)[0]
                            pos_rows.append(row_inds)
                            max_pos = max(max_pos, row_inds.numel())

                        max_pos = max(max_pos, 1)
                        pos_ids_sampled = torch.zeros(
                            (batch_size_f, max_pos),
                            device=self.device,
                            dtype=torch.long,
                        )
                        pos_mask_sampled = torch.zeros(
                            (batch_size_f, max_pos),
                            device=self.device,
                            dtype=torch.bool,
                        )

                        for i in range(batch_size_f):
                            n = pos_rows[i].numel()
                            if n > 0:
                                pos_ids_sampled[i, :n] = pos_rows[i]
                                pos_mask_sampled[i, :n] = True

                        neg_count = 128
                        neg_ids = torch.randint(
                            4,
                            vocab_size_f,
                            (batch_size_f, neg_count),
                            device=self.device,
                        )

                        idxs = torch.cat([pos_ids_sampled, neg_ids], dim=1)  # (B, P+N)
                        # C1: Validate sampled indices range
                        assert idxs.min().item() >= 0
                        assert idxs.max().item() < vocab_size_f

                        # 2) Gather Logits
                        gathered_logits = logits_f.gather(1, idxs)

                        # 3) Build Targets
                        targets_sampled = torch.cat(
                            [
                                torch.ones(
                                    (batch_size_f, max_pos),
                                    device=self.device,
                                    dtype=torch.float,
                                )
                                * pos_mask_sampled.float(),
                                torch.zeros(
                                    (batch_size_f, neg_count),
                                    device=self.device,
                                    dtype=torch.float,
                                ),
                            ],
                            dim=1,
                        )

                        # 4) Compute Loss (Prevalence-Aware + Separation)
                        valid = torch.cat(
                            [
                                pos_mask_sampled,
                                torch.ones(
                                    (batch_size_f, neg_count),
                                    device=self.device,
                                    dtype=torch.bool,
                                ),
                            ],
                            dim=1,
                        )

                        task_bce = self._balanced_bce_loss(
                            gathered_logits,
                            pos_mask_sampled,
                            neg_count,
                            valid,
                            pos_density=self.family_pos_densities.get(name, 8.0),
                        )

                        if not torch.isfinite(task_bce):
                            raise RuntimeError(f"Non-finite KC loss for {name} (dense)")

                        # Apply per-family loss weight for balanced training
                        family_def_sparse_block = get_family(fid)
                        w = family_def_sparse_block.loss_weight
                        task_bce = task_bce * w

                        if kc_diag is not None:
                            # Construct full-width mask for diagnostics to match idxs/logits
                            diag_pos_mask = torch.cat(
                                [
                                    pos_mask_sampled,
                                    torch.zeros(
                                        (batch_size_f, neg_count),
                                        device=self.device,
                                        dtype=torch.bool,
                                    ),
                                ],
                                dim=1,
                            )
                            if not skip_metrics:
                                kc_diag.update_family(
                                    name,
                                    idxs.detach().cpu(),
                                    diag_pos_mask.detach().cpu(),
                                    torch.sigmoid(gathered_logits).detach().cpu(),
                                    targets_sampled.detach().cpu(),
                                    task_bce.item(),
                                    logits=gathered_logits.detach(),
                                )

                        if fam_acc is not None and not skip_metrics:
                            with torch.no_grad():
                                fam_acc.update(
                                    logits=gathered_logits.detach(),
                                    targets=targets_sampled.detach(),
                                    pos_mask=(targets_sampled > 0.5).detach(),
                                    valid_mask=None,  # all sampled entries valid
                                    source="sparse",  # sampled P+N entries, not full-K
                                )

                        structural_loss += task_bce
                        num_struct += 1
                        batch_kc_losses[name] = task_bce.item()
                    else:
                        if not name:
                            raise ValueError("Family name cannot be empty")

                        # Balanced Dense Loss (computed WITH gradients)
                        # Shape already validated above when logits_f was assigned
                        pos_mask_d = targets > 0.5
                        if pos_mask_d.any():
                            loss_pos_d = F.binary_cross_entropy_with_logits(
                                logits_f[pos_mask_d],
                                torch.ones_like(logits_f[pos_mask_d]),
                                reduction="mean",
                            )
                        else:
                            loss_pos_d = torch.tensor(0.0, device=self.device)

                        neg_mask_d = targets < 0.5
                        if neg_mask_d.any():
                            loss_neg_d = F.binary_cross_entropy_with_logits(
                                logits_f[neg_mask_d],
                                torch.zeros_like(logits_f[neg_mask_d]),
                                reduction="mean",
                            )
                        else:
                            loss_neg_d = torch.tensor(0.0, device=self.device)

                        task_loss = 0.5 * loss_pos_d + 0.5 * loss_neg_d

                        # Apply per-family loss weight for balanced training
                        family_def_dense_block = get_family(fid)
                        w = family_def_dense_block.loss_weight
                        task_loss = task_loss * w

                        if kc_diag is not None:
                            with torch.no_grad():
                                # A1: Keep tensors 2D and aligned (Restore scaffolds)
                                probs_2d = torch.sigmoid(logits_f)
                                targets_2d = targets
                                v_ids_2d = (
                                    torch.arange(vocab_size_f, device=self.device)
                                    .unsqueeze(0)
                                    .expand(batch_size_f, -1)
                                )
                                pos_mask_2d = targets_2d > 0.5

                                if not skip_metrics:
                                    kc_diag.update_family(
                                        name,
                                        v_ids_2d.detach(),
                                        pos_mask_2d.detach(),
                                        probs_2d.detach(),
                                        targets_2d.detach(),
                                        task_loss.item(),
                                        mask_id=reading_mask_id,
                                        logits=logits_f,
                                    )

                            if fam_acc is not None and not skip_metrics:
                                with torch.no_grad():
                                    fam_acc.update(
                                        logits=logits_f.detach(),
                                        targets=targets.detach(),
                                        pos_mask=(targets > 0.5).detach(),
                                        valid_mask=None,  # dense targets, all valid
                                        source="dense",
                                    )

                    structural_loss += task_loss
                    num_struct += 1
                    batch_kc_losses[name] = task_loss.item()

                elif pos_key in kc_targets and mask_key in kc_targets:
                    pos_inds = kc_targets[pos_key].to(self.device)
                    pos_mask_t = kc_targets[mask_key].to(self.device)
                    logits_f = logits.float()

                    # Get family loss weight
                    family_def_sparse = get_family(fid)
                    task_bce = self._bce_sampled_from_sparse(
                        logits_f=logits_f,
                        pos_inds=pos_inds,
                        pos_mask=pos_mask_t,
                        vocab_size=vocab_size,
                        neg_count=128,
                        seed=(epoch * 100000 + batch_idx),
                        diag=None if skip_metrics else kc_diag,
                        family_name=name,
                        reading_mask_id=reading_mask_id,
                        accumulator=None if skip_metrics else fam_acc,
                        loss_weight=family_def_sparse.loss_weight,
                    )

                    structural_loss += task_bce
                    num_struct += 1
                    batch_kc_losses[name] = task_bce.item()

            if num_struct > 0:
                running_struct_loss += structural_loss.item()
                running_num_struct_total += 1
            # Build combined_loss from components (clone to avoid aliasing)
            loss_primary_val = structural_loss.item()
            combined_loss = structural_loss.clone()

            if relative_epoch < self.freeze_encoder_epochs:
                div_weight = self.kc_diversity_weight_frozen
                lb_weight = self.kc_lb_weight_frozen
            else:
                div_weight = self.kc_diversity_weight_thawed
                lb_weight = self.kc_lb_weight_thawed

            loss_div_val = 0.0
            loss_lb_val = 0.0
            loss_coll_val = 0.0
            loss_coverage_val = 0.0

            # p_max removed

            if epoch >= self.kc_diversity_warmup_epochs:
                logits_usage = outputs.get("logits_usage", outputs["kc_logits_raw"])
                tau_usage = 1.0 if relative_epoch < self.freeze_encoder_epochs else 2.0

                logit_ref = logits_usage
                # Adaptive Divergence / Collapse Logic
                # Split batch into short (<=3) and normal (>3)
                # Short: 50% entropy floor, NO collapse penalty
                # Normal: Full entropy floor, Full collapse penalty

                is_short = content_len <= 3

                # We compute weighted metrics
                div_accum = torch.tensor(0.0, device=self.device)
                # coll_accum removed

                splits = []
                if is_short.any():
                    splits.append(
                        {
                            "mask": is_short,
                            "floor": 0.5 * self.kc_entropy_floor,
                            "apply_collapse": False,
                        }
                    )
                if (~is_short).any():
                    splits.append(
                        {
                            "mask": ~is_short,
                            "floor": self.kc_entropy_floor,
                            "apply_collapse": True,
                        }
                    )

                total_n = logit_ref.size(0)

                for s in splits:
                    mask = s["mask"]
                    sub_logits = logit_ref[mask]
                    weight = mask.sum().float() / total_n

                    scale = tau_usage  # Use same tau for simplification
                    q_sub = torch.softmax(sub_logits / scale, dim=-1)
                    p = q_sub.mean(dim=0)

                    p_sum = p.sum().clamp_min(self.kc_diversity_eps)
                    p = p / p_sum

                    # --- INVARIANT CHECK H: Diagnostic ---
                    if abs(p.sum().item() - 1.0) > 1e-3:
                        raise RuntimeError(f"p sum != 1: {p.sum().item()}")
                    if p.min().item() < -1e-9:
                        raise RuntimeError(f"p < 0: {p.min().item()}")

                    log_p = (p + self.kc_diversity_eps).log()
                    entropy = -(p * log_p).sum()
                    ent_n = entropy / math.log(kc_vocab_size)

                    # Diversity (unhinge: always active, targeting entropy_floor)
                    d_loss = s["floor"] - ent_n
                    if div_weight > 0:
                        div_accum += weight * (div_weight * d_loss)

                    # KL to Uniform (Load Balance)
                    kl_val = (p * (p.clamp_min(1e-9) * kc_vocab_size).log()).sum()
                    lb_val = kl_val / math.log(kc_vocab_size)

                    if lb_weight > 0:
                        # Load balance (unhinge: always active, targeting kl_cap)
                        lb_l = lb_val - self.kc_kl_cap
                        combined_loss += weight * (lb_weight * lb_l)
                        loss_lb_val += (weight * lb_weight * lb_l).item()

                    # Collapse
                    softmax_peak = p.max()

                    if s["apply_collapse"]:
                        if (
                            relative_epoch >= self.freeze_encoder_epochs
                            and self.kc_collapse_weight_thawed > 0
                        ):
                            thr = max(1.5 / max(1, kc_vocab_size), 0.001)
                            diff = (softmax_peak - thr).clamp_min(0.0)
                            c_pen = diff
                            c_loss = self.kc_collapse_weight_thawed * c_pen
                            combined_loss += weight * c_loss
                            loss_coll_val += (weight * c_loss).item()

                        # (Secondary Collapse Penalty removed per instruction)

                combined_loss += div_accum
                loss_div_val = div_accum.item()

            # Coverage Loss: Encourage all KC logits to be used
            # For each KC logit, find its max probability across the batch
            # Penalize if many logits have low max probability
            coverage_weight = (
                self.kc_config.coverage_weight
                if relative_epoch < self.freeze_encoder_epochs
                else self.kc_config.coverage_weight_thawed
            )

            kc_probs = torch.sigmoid(outputs["kc_logits_effective"])

            if coverage_weight > 0:
                # Max probability each KC achieves across the batch
                # Shape: kc_probs is [batch_size, vocab_size]
                max_probs_per_kc = kc_probs.max(dim=0)[0]  # [vocab_size]

                # Penalize KCs that don't reach minimum threshold
                min_threshold = self.kc_config.coverage_min_prob
                coverage_violations = torch.nn.functional.relu(
                    min_threshold - max_probs_per_kc
                )
                coverage_loss = coverage_violations.mean()

                weighted_coverage_loss = coverage_weight * coverage_loss
                combined_loss += weighted_coverage_loss
                loss_coverage_val += weighted_coverage_loss.item()

            # Fail-Fast: Bounds
            if not (kc_probs.min() >= -1e-6 and kc_probs.max() <= 1 + 1e-6):
                raise ValueError(
                    f"kc_probs out of bounds: min={kc_probs.min()} max={kc_probs.max()}"
                )

            pmax_per_ex = kc_probs.max(dim=1).values
            batch_pmax_mean = pmax_per_ex.mean().item()
            batch_pmax_global = kc_probs.max().item()
            batch_probs_mean = kc_probs.mean().item()

            # Fail-Fast: Logic
            if batch_pmax_global < batch_probs_mean - 1e-3:
                raise ValueError(
                    f"pmax_global ({batch_pmax_global}) < probs_mean ({batch_probs_mean})"
                )
            if batch_pmax_mean < batch_probs_mean - 1e-3:
                raise ValueError(
                    f"pmax_mean ({batch_pmax_mean}) < probs_mean ({batch_probs_mean})"
                )

            running_pmax_global = max(running_pmax_global, batch_pmax_global)

            if (
                self.kc_sparsity_weight > 0
                and self.kc_sparsity_mode == "target_density"
            ):
                avg_prob = outputs["kc_probs"].mean()
                act_dens = (outputs["sparse_activations"] > 0).float().mean()

                # Adaptive Sparsity: sum(topk_vals) / k_i
                # Weighted by sqrt(content_len / mean_len)
                topk_vals = outputs["topk_vals"]  # (B, K)
                sum_vals_per_row = topk_vals.sum(dim=1)  # (B,)

                # We reuse k_budget_t from earlier (B,)
                sparsity_per_row = sum_vals_per_row / k_budget_t.float().clamp_min(1.0)

                mean_len = content_len.mean().clamp_min(1.0)
                len_scaling = (content_len / mean_len).sqrt()

                weighted_sparsity = sparsity_per_row * len_scaling
                sparsity_term = weighted_sparsity.mean()

                if not torch.isfinite(sparsity_term):
                    raise RuntimeError("Non-finite sparsity_term")
                st_val = sparsity_term.item()
                if st_val < 0.0:
                    raise RuntimeError(f"sparsity_term < 0: {st_val}")
            else:
                avg_prob = outputs["kc_probs"].mean()
                act_dens = outputs["sparse_activations"].mean()
                sparsity_term = act_dens

            running_avg_prob += avg_prob.item()
            running_act_dens += act_dens.item()

            total_sparsity += float(sparsity_term.detach().item())
            running_sparsity += sparsity_term.item()

            # --- Anti-Saturation Penalty (Gated & Auto-Scaled) ---
            # Penalize logit magnitude for pmax > 0.95 (logit > 3.0)
            # ONLY for examples with at least one positive target.

            # 1. Retrieve efficient has_pos_mask [B]
            # 1. Retrieve efficient has_pos_mask [B]
            has_pos_mask = kc_targets.get(
                "kc_has_pos_effective",
                torch.zeros(
                    batch.attention_mask.size(0), dtype=torch.bool, device=self.device
                ),
            )

            # 2. Compute Penalty from RAW LOGITS
            raw_logits = outputs["kc_logits_effective"]
            pmax_logit_per_ex = raw_logits.max(dim=1).values

            logit_thr = 3.0
            sat_excess = (pmax_logit_per_ex - logit_thr).clamp_min(0.0)

            # Global (ungated) penalty vector
            sat_pen_per_ex = sat_excess * sat_excess

            # Gated Penalty Loss
            if has_pos_mask.any():
                sat_pen_loss = sat_pen_per_ex[has_pos_mask].mean()
            else:
                # Ensure grad path but zero value
                sat_pen_loss = raw_logits.sum() * 0.0

            # Auto-Scaling Ratio (Alpha)
            # Ramp from 0.5% to 2.0% over the same window as sat_w
            sat_alpha = 0.005 + 0.015 * ramp_val

            if sat_w > 0:
                # Calculate Auto-Scale Factor (Detached)
                # We want: sat_scale * sat_pen_loss ≈ alpha * primary_loss
                # So: sat_scale ≈ alpha * (primary_loss / sat_pen_loss)
                eps = 1e-8
                loss_prim_det = loss_primary_val  # scalar float from earlier
                sat_pen_det = sat_pen_loss.detach().item()

                # Only scale if we have a non-trivial penalty to scale against
                if sat_pen_det > eps:
                    sat_scale = sat_alpha * (max(eps, loss_prim_det) / sat_pen_det)
                else:
                    sat_scale = 0.0

                # Apply scaled penalty
                loss_sat_term = sat_w * sat_scale * sat_pen_loss
                combined_loss = combined_loss + loss_sat_term

                # Safety Assertion: First batch of first thawed epoch
                if (
                    relative_epoch == self.freeze_encoder_epochs
                    and batch_idx == 0
                    and has_pos_mask.any()
                ):
                    assert sat_pen_loss.requires_grad, "sat_pen_loss must require grad"

                # Accumulate Stats
                # 1) Global (Batch-Weighted)
                sat_global_batches += 1
                sat_pen_global_sum += sat_pen_per_ex.detach().mean().item()
                pmax_logit_mean_global_sum += pmax_logit_per_ex.detach().mean().item()
                pmax_logit_max_global = max(
                    pmax_logit_max_global, pmax_logit_per_ex.detach().max().item()
                )

                # 2) Pos-Only (Example-Weighted)
                avg_has_pos = has_pos_mask.float().mean().item()
                frac_has_pos_batches_sum += avg_has_pos
                sat_pos_batches += 1

                if has_pos_mask.any():
                    pos_logits = pmax_logit_per_ex.detach()[has_pos_mask]
                    pos_pen = sat_pen_per_ex.detach()[has_pos_mask]
                    pos_over = (pos_logits > logit_thr).float()

                    n_pos = int(has_pos_mask.sum().item())
                    sat_pos_ex_count += n_pos

                    sat_pen_pos_sum += pos_pen.sum().item()
                    pmax_logit_pos_sum += pos_logits.sum().item()
                    frac_over_thr_pos_sum += pos_over.sum().item()
                    pmax_logit_max_pos = max(
                        pmax_logit_max_pos, pos_logits.max().item()
                    )

                # 3) Scaling Stats
                sat_active_batches += 1
                sat_scale_sum += sat_scale

                contrib_val = loss_sat_term.detach().item()
                sat_contrib_sum += contrib_val
                loss_sat_val = contrib_val  # Track for loss breakdown

                ratio_val = contrib_val / max(eps, loss_prim_det)
                sat_contrib_ratio_sum += ratio_val
            else:
                # Keep alpha for logging even if w=0 (metrics placeholder)
                loss_sat_val = 0.0

            spar_w = self.kc_sparsity_weight
            if relative_epoch >= self.freeze_encoder_epochs:
                epoch_idx_thawed = max(0, relative_epoch - self.freeze_encoder_epochs)
                if epoch_idx_thawed < 3:
                    spar_w = 0.5 * self.kc_sparsity_weight

            # --- STRUCTURAL LOSS (compute first, then average) ---
            loss = (
                combined_loss + spar_w * sparsity_term
            ) / self.config.grad_accum_steps

            # --- PRIOR KC LOSSES: REMOVED ---
            # Formality (KC0-3), Gender (KC4-5), and Register (KC6-18) supervision
            # is now handled by the style classifier to prevent interference
            loss_formality_val = 0.0
            form_correct = 0
            form_total = 0
            loss_gender_val = 0.0
            gend_correct = 0
            gend_total = 0
            loss_register_val = 0.0
            reg_correct = 0
            reg_total = 0

            loss_spar_val = (spar_w * sparsity_term).item()

            # Build loss components for display (all values are raw sums per batch)
            gas = self.config.grad_accum_steps
            current_epoch_comp = {
                "struct": structural_loss.item(),
                "formality": loss_formality_val * gas,
                "gender": loss_gender_val * gas,
                "register": loss_register_val * gas,
                "div": loss_div_val,
                "lb": loss_lb_val,
                "collapse": loss_coll_val,
                "sparsity": loss_spar_val,
                "saturation": loss_sat_val,  # Part of combined_loss, no gas scaling
                "coverage": loss_coverage_val,  # Part of combined_loss, no gas scaling
            }

            # Accumulate for epoch summary (values match what goes into loss)
            current_comp = RunningLossComponents(
                struct=current_epoch_comp["struct"],
                formality=current_epoch_comp["formality"],
                gender=current_epoch_comp["gender"],
                register=current_epoch_comp["register"],
                div=current_epoch_comp["div"],
                lb=current_epoch_comp["lb"],
                collapse=current_epoch_comp["collapse"],
                sparsity=current_epoch_comp["sparsity"],
                saturation=current_epoch_comp["saturation"],
                coverage=current_epoch_comp["coverage"],
                formality_correct=form_correct,
                formality_total=form_total,
                gender_correct=gend_correct,
                gender_total=gend_total,
                register_correct=reg_correct,
                register_total=reg_total,
            )
            running_loss_components = running_loss_components.add(current_comp)

            if loss.item() == 0.0 and loss.requires_grad:
                pass

            if torch.isfinite(loss):
                loss.backward()
                did_any_backward = True
                pending_accum += 1

                if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                    self._perform_optimizer_step(m)

                    pending_accum = 0

                total_loss += loss.item() * self.config.grad_accum_steps
                for k_loss, v_loss in batch_kc_losses.items():
                    epoch_kc_losses[k_loss] = epoch_kc_losses.get(k_loss, 0.0) + v_loss  # type: ignore
                n_batches += 1

            else:
                raise RuntimeError("Non-finite loss detected")

            if pbar:
                current_display_loss = total_loss / max(1, n_batches)
                pbar.update(batch_idx, current_display_loss)

            # Update running usage for Entropy calc (skip for early epochs)
            # We use logits_usage if available (for consistency with diversity), else raw
            # Note: logits_usage depends on epoch >= warmup.
            # We want global coverage.
            if not skip_metrics:
                if "logits_usage" in outputs:
                    l_usage = outputs["logits_usage"]
                else:
                    l_usage = outputs["kc_logits_raw"]

                # Accumulate sum of softmax probabilities
                # Detach to save memory
                usage_probs = F.softmax(l_usage.detach(), dim=1)
                running_usage_probs_sum += usage_probs.sum(dim=0)
                total_samples_seen += usage_probs.size(0)

            # View Batch Stats (skip for early epochs if configured)
            if "topk_vals" in outputs and not skip_metrics:
                topk_v = outputs["topk_vals"].detach()
                topk_s = topk_v.sum(dim=1)

                # pmax_per_ex calculated earlier at line 1210
                # But we need access to it. It is local variable 'pmax_per_ex'.
                # We assume it is available here.

                self.view.on_kc_batch_stats(
                    epoch=epoch,
                    batch_idx=batch_idx,
                    content_len=content_len.detach(),
                    k_budget_t=k_budget_t.detach(),
                    topk_vals=topk_v,
                    pmax_per_ex=pmax_per_ex.detach(),
                    topk_sum_per_ex=topk_s,
                    kc_probs=outputs["kc_probs"].detach(),
                )

            self.view.on_kc_progress_update(
                batch_idx,
                current_display_loss if current_display_loss is not None else 0.0,
                total_batches,
            )

            # Explicitly clear loop variables to prevent graph retention across batches
            del loss, combined_loss, structural_loss, outputs

            self.train_timer_compute.stop(epoch=epoch, batch=batch_idx)
            self.train_timer_data.start()

        self.start_batch = 0

        if did_any_backward and pending_accum > 0:
            self._perform_optimizer_step(self.model)

        self.view.on_line_flush()

        epoch_stats = TrainEpochStats(
            avg_struct_loss=running_struct_loss / max(1, running_num_struct_total),
            num_struct_heads_processed=running_num_struct_total,
            avg_sparsity=running_sparsity / max(1, n_batches),
            avg_prob=running_avg_prob / max(1, n_batches),
            act_dens=running_act_dens / max(1, n_batches),
            # Only include diagnostics when metrics are not skipped
            kc_diagnostics=None if skip_metrics else kc_diag.get_stats(),
        )

        pbar.stop()

        self.view.on_kc_progress_stop()

        # --- Epoch Summary Usage Stats ---
        # Normalize usage distribution
        if total_samples_seen > 0:
            p_mean = running_usage_probs_sum / total_samples_seen
            # Entropy
            p_mean = p_mean / p_mean.sum().clamp_min(1e-9)  # Re-normalize
            log_p = (p_mean + 1e-9).log()
            ent_val_final = float(-(p_mean * log_p).sum().item())
            ent_norm_final = ent_val_final / math.log(max(1, kc_vocab_size))

            # KL to Uniform
            # KL(p || u) = sum(p * log(p/u)) = sum(p * log(p)) - sum(p * log(u))
            #            = -Entropy(p) - log(1/V) = -H(p) + log(V)
            # kl_val = -ent_val + math.log(kc_vocab_size)
            # Or explicit:
            kl_val_final = float((p_mean * (p_mean * kc_vocab_size).log()).sum().item())
            kl_u_norm_final = kl_val_final / math.log(max(1, kc_vocab_size))
        else:
            ent_norm_final = 0.0
            kl_u_norm_final = 0.0

        activation_stats = KcEpochActivationStats(
            pmax_global_max=running_pmax_global,
            pmax_p50=0.0,  # Filled by View
            pmax_p90=0.0,
            pmax_p99=0.0,
            topk_sum_p50=0.0,
            topk_sum_p90=0.0,
            topk_sum_p99=0.0,
            ent_norm=ent_norm_final,
            kl_u_norm=kl_u_norm_final,
            act_dens_mean=running_act_dens / max(1, n_batches),
            kc_probs_mean=running_avg_prob / max(1, n_batches),
            # Saturation Stats (Gated & Scaled)
            sat_w=sat_w,
            sat_alpha=sat_alpha if sat_w > 0 else 0.0,
            sat_scale_mean=(
                (sat_scale_sum / sat_active_batches) if sat_active_batches > 0 else 0.0
            ),
            sat_contrib_mean=(
                (sat_contrib_sum / sat_active_batches)
                if sat_active_batches > 0
                else 0.0
            ),
            sat_contrib_ratio=(
                (sat_contrib_ratio_sum / sat_active_batches)
                if sat_active_batches > 0
                else 0.0
            ),
            sat_pen_global=(
                (sat_pen_global_sum / sat_global_batches)
                if sat_global_batches > 0
                else 0.0
            ),
            sat_pen_pos=(
                (sat_pen_pos_sum / sat_pos_ex_count) if sat_pos_ex_count > 0 else 0.0
            ),
            pmax_logit_mean_global=(
                (pmax_logit_mean_global_sum / sat_global_batches)
                if sat_global_batches > 0
                else 0.0
            ),
            pmax_logit_max_global=(
                pmax_logit_max_global if sat_global_batches > 0 else 0.0
            ),
            pmax_logit_mean_pos=(
                (pmax_logit_pos_sum / sat_pos_ex_count) if sat_pos_ex_count > 0 else 0.0
            ),
            pmax_logit_max_pos=(pmax_logit_max_pos if sat_pos_ex_count > 0 else 0.0),
            frac_over_thr_pos=(
                (frac_over_thr_pos_sum / sat_pos_ex_count)
                if sat_pos_ex_count > 0
                else 0.0
            ),
            frac_has_pos=(
                (frac_has_pos_batches_sum / sat_pos_batches)
                if sat_pos_batches > 0
                else 0.0
            ),
        )

        diag_report = kc_diag.get_stats()

        # Update diag_report with FamilyAccumulator stats
        for name, fam_acc in family_accumulators.items():
            if name in diag_report.families:
                fam_diag = diag_report.families[name]

                if fam_acc.n_ex > 0:
                    fam_diag.pos_ex_frac = fam_acc.n_pos_ex / fam_acc.n_ex
                    # pos_label_density = avg pos labels per example
                    fam_diag.pos_label_density = fam_acc.n_pos_labels / fam_acc.n_ex
                    # mask_coverage: fraction of examples with valid supervision
                    # If no valid_mask was ever seen, interpret as 1.0 (all valid)
                    if fam_acc.saw_valid_mask:
                        fam_diag.mask_coverage = fam_acc.sum_valid_any / fam_acc.n_ex
                    else:
                        fam_diag.mask_coverage = 1.0

                if fam_acc.cnt_logit_pos > 0:
                    fam_diag.logit_pos_mean = (
                        fam_acc.sum_logit_pos / fam_acc.cnt_logit_pos
                    )
                else:
                    fam_diag.logit_pos_mean = float("nan")

                if fam_acc.cnt_logit_neg > 0:
                    fam_diag.logit_neg_mean = (
                        fam_acc.sum_logit_neg / fam_acc.cnt_logit_neg
                    )
                else:
                    fam_diag.logit_neg_mean = float("nan")

                # Build keys_present to reflect actual sources used
                keys = []
                if fam_acc.saw_dense:
                    keys.append("dense")
                if fam_acc.saw_sparse:
                    keys.append("sparse")
                if fam_acc.saw_valid_mask:
                    keys.append("validmask")
                fam_diag.keys_present = ",".join(keys)

                # Compute bias delta for gradient flow diagnostic
                if name in bias_start and name in m.kc_decoders.decoders:
                    decoder_lin = m.kc_decoders.decoders[name]
                    if hasattr(decoder_lin, "bias") and decoder_lin.bias is not None:
                        # Use absolute sum of changes to detect learning
                        # (mean would cancel out push-up vs push-down)
                        bias_change = (decoder_lin.bias - bias_start[name]).abs().sum()
                        fam_diag.bias_delta = float(bias_change.item())

        # Compute bias_delta for MSE families
        for mse_name, mse_diag in diag_report.mse_families.items():
            if mse_name in bias_start and mse_name in m.kc_decoders.decoders:
                mse_decoder = m.kc_decoders.decoders[mse_name]
                if hasattr(mse_decoder, "bias") and mse_decoder.bias is not None:
                    mse_bias_change = (
                        (mse_decoder.bias - bias_start[mse_name]).abs().sum()
                    )
                    mse_diag.bias_delta = float(mse_bias_change.item())

        # KcLossWeights for display - most losses are already weighted in storage,
        # only prior losses (formality/gender/register) need their weight applied.
        # Use defaults: struct=1.0, prior=0.2, others=1.0 (already weighted)
        loss_weights = KcLossWeights()

        # Calculate KC logit utilization
        kc_logits_used_count = len(kc_logits_used_set)
        kc_logits_used_percent = (
            (100.0 * kc_logits_used_count / kc_vocab_size) if kc_vocab_size > 0 else 0.0
        )

        summary = KcEpochSummary(
            epoch_idx=epoch,
            frozen=should_freeze,
            loss_components=running_loss_components,
            sizing_stats=[],  # Filled by View
            activation_stats=activation_stats,
            diagnostics=diag_report,
            weights=loss_weights,
            n_batches=n_batches,
            total_loss=total_loss
            / max(1, n_batches),  # Per-batch average (guard div-by-zero)
            accumulators=family_accumulators,
            kc_logits_used_count=kc_logits_used_count,
            kc_logits_used_percent=kc_logits_used_percent,
            worst_samples=worst_samples,
        )

        # Skip full diagnostics for early epochs (performance optimization)
        if skip_metrics:
            self.view.on_kc_epoch_metrics_skipped(epoch, total_loss)
        else:
            self.view.on_kc_epoch_summary(epoch, summary)

        self.view.on_kc_epoch_end(
            epoch,
            epoch_result=TrainEpochResult(
                total_loss=total_loss,
                kc_losses=KCLosses(_losses=epoch_kc_losses),
                avg_sparsity=total_sparsity / max(1, len(self.data_loader)),
                epoch_stats=epoch_stats,
            ),
        )

        return TrainEpochResult(
            total_loss=total_loss,
            kc_losses=KCLosses(_losses=epoch_kc_losses),
            avg_sparsity=total_sparsity / max(1, len(self.data_loader)),
            epoch_stats=epoch_stats,
        )

    def _log_training_progress(self) -> None:
        data_avg = self.train_timer_data.avg()
        compute_avg = self.train_timer_compute.avg()
        total = data_avg + compute_avg
        if total > 0:
            self.view.on_kc_timing_summary(
                total * 1000, data_avg * 1000, compute_avg * 1000, data_avg / total
            )
        self.train_timer_data.reset()
        self.train_timer_compute.reset()

    def train(
        self,
        epochs: int,
        on_epoch_end: Callable[[KCTrainingHistory], None],
        start_epoch: Optional[int] = None,
    ) -> KCTrainingHistory:
        # Use explicit start_epoch if provided
        effective_start = start_epoch if start_epoch is not None else self.start_epoch

        # Record when this session started to support relative freezing/warmups
        if self.session_start_epoch is None:
            self.session_start_epoch = effective_start

        if effective_start == 0 and self.start_batch == 0:
            self._init_structural_decoder_biases()

        self.view.on_kc_train_start(epochs, effective_start, self.start_batch)

        for epoch in range(effective_start, epochs):
            epoch_res = self.train_epoch(epoch=epoch)
            total_loss = epoch_res.total_loss
            kc_losses = epoch_res.kc_losses
            avg_sparsity = epoch_res.avg_sparsity
            epoch_stats = epoch_res.epoch_stats

            self._log_training_progress()

            self.history.total_loss.append(total_loss / max(1, len(self.data_loader)))
            self.history.kc_sparsity.append(avg_sparsity)
            self.history.avg_struct_loss.append(epoch_stats.avg_struct_loss)
            self.history.num_struct_heads_processed.append(
                float(epoch_stats.num_struct_heads_processed)
            )
            self.history.avg_sparsity.append(epoch_stats.avg_sparsity)

            # Always append to keep list aligned with epoch indices (None if skipped)
            self.history.kc_diagnostics.append(epoch_stats.kc_diagnostics)

            # Record active KC targets
            active_targets = sorted(list(kc_losses.keys()))
            self.history.active_kc_targets.append(",".join(active_targets))

            for k, v in kc_losses.items():
                if k not in self.history.kc_losses:
                    self.history.kc_losses[k] = []
                self.history.kc_losses[k].append(v)

            self.history.sentence_count.append(len(self.dataset))

            on_epoch_end(self.history)

        self.view.on_kc_train_end(self.history)

        return self.history
