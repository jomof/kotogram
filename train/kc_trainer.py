# pylint: disable=too-many-lines,not-callable,too-many-nested-blocks,duplicate-code
import math
import os
import random
from collections.abc import Iterable, Sized
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from kotogram.constants import REGISTER_ID_TO_LABEL
from kotogram.tokenizer import ENCODER_FEATURE_FIELDS, MASK_ID
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
from train.kc import KcFamilyId, get_family, is_family_db_sourced, is_family_sparse
from train.kc_diagnostics import (
    KCEpochDiag,
    discretize_mse,
)
from train.kc_trainer_view import KCTrainerDiagnosticsView, KCTrainerView
from train.models import TrainingClassifier
from train.profile import Timer, get_profile_dir
from train.types import (
    FamilyAccumulator,
    KcEpochActivationStats,
    KcEpochSummary,
    KCLosses,
    KcLossWeights,
    KCStructuralBiases,
    KCTrainingHistory,
    KcValResult,
    LayerHealthStats,
    RunningLossComponents,
    TensorStats,
    TrainEpochResult,
    TrainEpochStats,
    TrainingBatch,
    WorstSampleInfo,
    WorstSamplesTracker,
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


def _linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA between two (n, d) representation matrices."""
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cross = float(torch.norm(y.T @ x, p="fro").item() ** 2)
    xx = float(torch.norm(x.T @ x, p="fro").item())
    yy = float(torch.norm(y.T @ y, p="fro").item())
    denom = xx * yy
    if denom < 1e-12:
        return 0.0
    return cross / denom


def _effective_rank_90(x: torch.Tensor) -> float:
    """Effective rank: min k such that top-k singular values capture >= 90% variance."""
    x = (x - x.mean(dim=0, keepdim=True)).cpu()
    s = torch.linalg.svdvals(x)
    var_cumsum = torch.cumsum(s**2, dim=0)
    total = var_cumsum[-1].item()
    if total < 1e-12:
        return 0.0
    hits = (var_cumsum >= 0.9 * total).nonzero(as_tuple=True)[0]
    return float(hits[0].item() + 1) if hits[0].numel() > 0 else float(s.numel())


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
        self.model = model
        self.dataset = dataset
        self.config = config
        self.view: KCTrainerView = (
            view if view is not None else KCTrainerDiagnosticsView()
        )

        _safe_configure_threads(self.config)

        configure_runtime_thread_limits(self.config)

        self.kc_config = kc_config
        self.kl_sparse_weight = self.kc_config.kl_sparse_weight
        self.kl_target_rho = self.kc_config.kl_target_rho
        self.rho_length_scale = self.kc_config.rho_length_scale
        self.cov_penalty_weight = self.kc_config.cov_penalty_weight
        self.median_content_len: float = 12.0  # EMA estimate, updated per batch
        self.kc_sat_weight = self.kc_config.sat_weight
        self.freeze_encoder_epochs = self.kc_config.freeze_encoder_epochs

        # Cloze K: number of random positions to mask per sentence
        from train.kc import KcBertFamily

        bert_fam = next(
            (
                get_family(fid)
                for fid in (config.kc_target_specs or {})
                if isinstance(get_family(fid), KcBertFamily)
            ),
            None,
        )
        self._cloze_k: int = (
            bert_fam.cloze_k if isinstance(bert_fam, KcBertFamily) else 2
        )

        # Per-grammar-point priors -> per-label loss weights
        self._gp_prior_tensor: torch.Tensor
        self._gp_computed_default_prior: float = 1e-8  # Overwritten by median in _init
        self._init_gp_prior_weights()

        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self._init_recon_freq_weights()

        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        self._ramp_step = config.ramp_step
        self._ramp_threshold = config.ramp_posp_threshold
        self._current_ratio = config.sample_ratio
        self._surface_unfrozen_by_ramp = False

        self.val_sampler = None
        self.gram_sampler = None
        self._surface_id_to_token: Dict[int, str] = {}
        self.data_loader: Optional[Iterable[Any]] = None
        self.ungram_loader: Iterable[Any] = []
        self.gram_loader: Iterable[Any] = []

        use_dataset_loaders = isinstance(dataset, StyleDataset) or (
            isinstance(getattr(dataset, "indices", None), torch.Tensor)
            and isinstance(getattr(dataset, "labels", None), dict)
        )
        if use_dataset_loaders:
            self.ungram_dataset = dataset.filter_by_grammaticality(0)
            self.gram_dataset = dataset.filter_by_grammaticality(1)

            # Create style-aware sampler if enabled (oversamples non-neutral examples)
            if kc_config.style_oversample and hasattr(
                self.gram_dataset, "create_style_oversampler"
            ):
                self.gram_sampler = self.gram_dataset.create_style_oversampler(
                    formality_boost=kc_config.formality_boost,
                    gender_boost=kc_config.gender_boost,
                    length_reweight=kc_config.length_reweight,
                )
                self.view.on_style_oversampling_enabled(
                    kc_config.formality_boost, kc_config.gender_boost
                )

            if dl_config is None:
                dl_config = self.config.resolve_dataloader_config(
                    self.device, mode="train"
                )
            self._dl_config = dl_config

            self.ungram_loader = DataLoader(
                self.ungram_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                sampler=None,
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
            self.gram_loader = DataLoader(
                self.gram_dataset,
                batch_size=self.config.batch_size,
                shuffle=(self.gram_sampler is None),
                sampler=self.gram_sampler,
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
        else:
            self._dl_config = dl_config
            self.ungram_dataset = dataset
            self.gram_dataset = dataset
            self.ungram_loader = []
            self.gram_loader = []
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

        self.entropy_weight = self.kc_config.entropy_weight

        self.kc_collapse_weight_thawed = self.kc_config.collapse_weight_thawed

        self.kc_temperature_frozen = float(self.model.config.kc_temperature)

        self.kc_temperature_thawed = self.kc_config.temperature_thawed

        self.kc_grad_cap = self.kc_config.kc_grad_cap

        self.kc_entropy_floor = self.kc_config.entropy_floor

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

    @torch.no_grad()
    def compute_validation_loss(self, val_dataset: StyleDataset) -> KcValResult:
        """Compute KC validation loss for grammar_point, gender, formality.

        Creates a temporary DataLoader over grammatical sentences,
        runs forward passes in KC mode (eval), and computes losses
        for the three key families only.
        """
        val_families = {
            KcFamilyId.GRAMMAR_POINT,
            KcFamilyId.GENDER,
            KcFamilyId.FORMALITY,
        }

        self.model.eval()
        gram_val = val_dataset.filter_by_grammaticality(1)
        val_loader = DataLoader(
            gram_val,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=partial(collate_fn),
            num_workers=0,
            pin_memory=False,
        )

        total_loss = 0.0
        n_batches = 0
        t_val = self.kc_temperature_thawed
        family_loss_sums: Dict[str, float] = {}
        family_loss_counts: Dict[str, int] = {}

        for batch_data in val_loader:
            batch: TrainingBatch = batch_data

            field_inputs = {
                k: v.to(self.device, non_blocking=True)
                for k, v in batch.feature_inputs.items()
            }
            attention_mask = batch.attention_mask.to(self.device, non_blocking=True)

            outputs = self.model(
                field_inputs,
                attention_mask=attention_mask,
                mode="kc",
                temperature=t_val,
                gumbel_scale=0.0,
            )

            target_logits = outputs["target_logits"]

            kc_targets = create_kc_batch(
                batch=batch,
                tokenizer=self.dataset.tokenizer,
                target_specs=self.config.kc_target_specs,
            )
            for k, v in kc_targets.items():
                kc_targets[k] = v.to(self.device, non_blocking=True)

            batch_loss = 0.0
            n_families = 0
            for fid, vocab_size in self.config.kc_target_specs.items():
                if fid not in val_families:
                    continue
                name = fid.name.lower()
                logits_f = target_logits.get(name)
                if logits_f is None:
                    continue

                if fid == KcFamilyId.GRAMMAR_POINT:
                    # PNU multilabel loss
                    pos_key = f"kc_gp_pos_inds_{name}"
                    pos_mask_key = f"kc_gp_pos_mask_{name}"
                    neg_key = f"kc_gp_neg_inds_{name}"
                    neg_mask_key = f"kc_gp_neg_mask_{name}"
                    if pos_key in kc_targets:
                        loss, _ = self._multilabel_pnu_loss(
                            logits_f,
                            kc_targets[pos_key],
                            kc_targets[pos_mask_key],
                            kc_targets[neg_key],
                            kc_targets[neg_mask_key],
                            vocab_size,
                            priors=self._gp_prior_tensor,
                            unlabeled_weight=self.kc_config.gp_unlabeled_weight,
                        )
                        loss_val = loss.item()
                        batch_loss += loss_val
                        n_families += 1
                        family_loss_sums[name] = (
                            family_loss_sums.get(name, 0.0) + loss_val
                        )
                        family_loss_counts[name] = family_loss_counts.get(name, 0) + 1
                else:
                    # Continuous MSE (gender, formality)
                    continuous_key = f"kc_continuous_{name}"
                    if continuous_key in kc_targets:
                        targets_cont = kc_targets[continuous_key].float()
                        if not torch.isnan(targets_cont).any():
                            loss = self._continuous_mse_loss(
                                logits_f.float(), targets_cont
                            )
                            loss_val = loss.item()
                            batch_loss += loss_val
                            n_families += 1
                            family_loss_sums[name] = (
                                family_loss_sums.get(name, 0.0) + loss_val
                            )
                            family_loss_counts[name] = (
                                family_loss_counts.get(name, 0) + 1
                            )

            if n_families > 0:
                total_loss += batch_loss / n_families
                n_batches += 1

        self.model.train()
        avg_total = total_loss / n_batches if n_batches > 0 else 0.0
        avg_families = {
            name: total / family_loss_counts[name]
            for name, total in family_loss_sums.items()
        }
        return KcValResult(total_loss=avg_total, family_losses=avg_families)

    def _rebuild_dataloaders(self) -> None:
        """Re-split dataset into gram/ungram and recreate DataLoaders."""
        self.ungram_dataset = self.dataset.filter_by_grammaticality(0)
        self.gram_dataset = self.dataset.filter_by_grammaticality(1)

        if self.kc_config.style_oversample and hasattr(
            self.gram_dataset, "create_style_oversampler"
        ):
            self.gram_sampler = self.gram_dataset.create_style_oversampler(
                formality_boost=self.kc_config.formality_boost,
                gender_boost=self.kc_config.gender_boost,
                length_reweight=self.kc_config.length_reweight,
            )

        dl = self._dl_config
        self.ungram_loader = DataLoader(
            self.ungram_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            sampler=None,
            collate_fn=partial(collate_fn),
            num_workers=dl.num_workers,
            pin_memory=dl.pin_memory,
            persistent_workers=dl.persistent_workers,
            prefetch_factor=dl.prefetch_factor,
            worker_init_fn=_worker_init_fn,
        )
        self.gram_loader = DataLoader(
            self.gram_dataset,
            batch_size=self.config.batch_size,
            shuffle=(self.gram_sampler is None),
            sampler=self.gram_sampler,
            collate_fn=partial(collate_fn),
            num_workers=dl.num_workers,
            pin_memory=dl.pin_memory,
            persistent_workers=dl.persistent_workers,
            prefetch_factor=dl.prefetch_factor,
            worker_init_fn=_worker_init_fn,
        )

    @staticmethod
    def _format_kc_pbar_desc(pbar_desc: str, batch_label: str) -> str:
        if batch_label == "gram":
            return f"[white]🩷 {pbar_desc}[/white]"
        if batch_label == "ungram":
            return f"[white]💓 {pbar_desc}[/white]"
        return f"  {pbar_desc}"

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

    @staticmethod
    def _filter_batch_by_mask(
        batch: TrainingBatch, mask: torch.Tensor
    ) -> TrainingBatch:
        """Return a new batch containing only samples where mask is True."""
        if mask.dim() != 1:
            raise ValueError("Mask must be 1D for batch filtering")
        mask = mask.bool()
        mask_list = mask.tolist()

        def _filter_list(values: List[Any]) -> List[Any]:
            return [v for v, keep in zip(values, mask_list) if keep]

        feature_inputs = {k: v[mask] for k, v in batch.feature_inputs.items()}
        return TrainingBatch(
            feature_inputs=feature_inputs,
            attention_mask=batch.attention_mask[mask],
            formality_value=batch.formality_value[mask],
            formality_pragmatic=batch.formality_pragmatic[mask],
            gender_value=batch.gender_value[mask],
            gender_pragmatic=batch.gender_pragmatic[mask],
            grammaticality_labels=batch.grammaticality_labels[mask],
            register_labels=batch.register_labels[mask],
            indices=batch.indices[mask],
            original_sentence=_filter_list(batch.original_sentence),
            kotogram=_filter_list(batch.kotogram),
            kc_targets=_filter_list(batch.kc_targets),
        )

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
        priors: torch.Tensor,
        unlabeled_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-label PNU loss with stochastic pseudo-labeling.

        Labeled positions get hard targets (pos=1, neg=0).  Unlabeled
        positions get stochastic hard targets sampled from Bernoulli(prior)
        -- a fresh draw every forward pass.  Labeled negatives the model
        predicts as positive are upweighted by 1/prior (hard-negative
        scaling).

        Because all targets are hard 0/1, the loss achieves exactly zero
        when the model perfectly predicts the current random assignment.
        Over many batches the stochastic assignments average out: ~prior
        fraction are labeled 1, teaching the model the correct base rate
        without an irreducible entropy floor.

        Args:
            logits: (B, vocab_size) logits from KC decoder
            pos_ids: (B, max_pos) positive grammar point IDs
            pos_mask: (B, max_pos) mask for valid positive IDs
            neg_ids: (B, max_neg) negative grammar point IDs
            neg_mask: (B, max_neg) mask for valid negative IDs
            vocab_size: number of grammar points (1374)
            unlabeled_weight: weight for unlabeled loss relative to labeled
            priors: (vocab_size,) per-GP prior probabilities

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (total_loss, loss_by_label)
        """
        batch_size = logits.size(0)
        device = logits.device

        # Build label masks: (B, vocab_size)
        labeled_pos = torch.zeros(batch_size, vocab_size, device=device)
        labeled_neg = torch.zeros(batch_size, vocab_size, device=device)

        valid_pos = pos_ids.clamp(0, vocab_size - 1)
        labeled_pos.scatter_(1, valid_pos, pos_mask.float())

        valid_neg = neg_ids.clamp(0, vocab_size - 1)
        labeled_neg.scatter_(1, valid_neg, neg_mask.float())

        labeled_mask = labeled_pos + labeled_neg
        unlabeled_mask = (1.0 - labeled_mask).clamp(0, 1)

        # Build targets: hard labels everywhere
        #   labeled_pos → 1.0, labeled_neg → 0.0
        #   unlabeled → Bernoulli(prior) fresh each forward pass
        prior_probs = priors.to(device=device, dtype=logits.dtype).unsqueeze(0)
        pseudo_labels = torch.bernoulli(prior_probs.expand(batch_size, vocab_size))
        targets = labeled_pos + unlabeled_mask * pseudo_labels

        # Single BCE pass -- all targets are hard 0/1
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Hard-negative upweighting: scale loss by ln(1/prior) for labeled
        # negatives the model predicts as positive (rare GP false positives
        # are penalised more heavily, log-scaled to stay tractable).
        hard_neg = labeled_neg * (logits.detach() > 0).float()
        ln_inv_prior = torch.log(1.0 / prior_probs.clamp_min(1e-6))
        bce = bce * (1.0 + hard_neg * (ln_inv_prior - 1.0))

        # Normalize labeled and unlabeled separately
        labeled_bce = (bce * labeled_mask).sum(dim=0)
        labeled_count = labeled_mask.sum(dim=0).clamp_min(1.0)
        labeled_loss = labeled_bce / labeled_count

        unlabeled_bce = (bce * unlabeled_mask).sum(dim=0)
        unlabeled_count = unlabeled_mask.sum(dim=0).clamp_min(1.0)
        unlabeled_loss = unlabeled_bce / unlabeled_count

        loss_per_gp = labeled_loss + unlabeled_weight * unlabeled_loss

        loss_by_label = loss_per_gp.clone()
        total_loss = loss_per_gp.mean()

        return total_loss, loss_by_label

    def _init_gp_prior_weights(self) -> None:
        """Initialize the GP prior tensor from dataset priors.

        Builds a (gp_vocab_size,) prior tensor where each entry is the
        expected population frequency for that GP.  NaN/unset priors are
        filled with the configured gp_default_prior.  The tensor is stored
        on CPU and moved to device per-batch inside the loss function.
        """
        gp_vocab_size = int(
            self.config.kc_target_specs.get(KcFamilyId.GRAMMAR_POINT, 0)
        )
        if gp_vocab_size <= 0:
            return

        gp_priors = self.dataset.gp_priors
        if gp_priors.numel() < gp_vocab_size:
            raise ValueError(
                f"gp_priors vector too small: {gp_priors.numel()} < {gp_vocab_size}. "
                "Re-run scripts/label.py so gp_priors.bin covers the grammar_point vocab."
            )

        pri = gp_priors.detach().float().cpu()[:gp_vocab_size]

        pri_ref = max(self.kc_config.gp_default_prior, 1e-6)
        self._gp_computed_default_prior = pri_ref

        # Fill NaN / out-of-range entries with the default prior.
        finite = torch.isfinite(pri) & (pri >= 0.0) & (pri <= 1.0)
        if (~finite).any():
            pri = pri.clone()
            pri[~finite] = pri_ref

        self._gp_prior_tensor = pri

    def _init_recon_freq_weights(self) -> None:
        """Compute inverse-frequency sampling weights from dataset token counts."""
        alpha = self.kc_config.recon_freq_alpha
        if alpha <= 0:
            return

        surface_data = getattr(self.dataset, "features", {}).get("surface")
        if surface_data is None:
            return

        vocab_size = self.model.config.vocab_sizes.get("surface", 0)
        if vocab_size <= 0:
            return

        counts = torch.bincount(surface_data.long(), minlength=vocab_size).float()
        counts.clamp_(min=1)
        weights = (1.0 / counts) ** alpha
        weights /= weights.mean()
        self.model.kc_decoders.recon_freq_weights = weights.to(self.device)

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

    def _get_surface_id_to_token(self) -> Dict[int, str]:
        """Lazily build and cache reverse surface vocab mapping."""
        if not self._surface_id_to_token:
            tok = self.dataset.tokenizer
            vocab = tok.field_vocabs.get("surface", {})
            self._surface_id_to_token = {v: k for k, v in vocab.items()}
        return self._surface_id_to_token

    def _evaluate_canary(
        self, sentence: str, expected_gp: Optional[int] = None
    ) -> Tuple[str, Optional[float]]:
        """Evaluate a canary sentence and return (summary_string, expected_gp_prob).

        Args:
            sentence: Japanese sentence to evaluate.
            expected_gp: If set, extract sigmoid(logit) for this GP index.

        Returns:
            (display_text, gp_prob) where gp_prob is the sigmoid probability
            for expected_gp, or None if not requested / unavailable.
        """
        tok = self.dataset.tokenizer
        encoded = tok.encode(sentence)
        seq_len = len(encoded[ENCODER_FEATURE_FIELDS[0]])
        field_inputs = {
            f"input_ids_{f}": torch.tensor(
                [encoded[f]], dtype=torch.long, device=self.device
            )
            for f in ENCODER_FEATURE_FIELDS
            if f in encoded
        }
        attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=self.device)

        was_training = self.model.training
        self.model.eval()
        expected_gp_prob: Optional[float] = None
        with torch.no_grad():
            outputs = self.model(
                field_inputs,
                attention_mask=attention_mask,
                mode="kc",
                temperature=self.kc_temperature_thawed,
                gumbel_scale=0.0,
            )
            kc_probs = outputs["kc_probs_clean"]
            thresh = self.view.kc_threshold
            kc_count = int((kc_probs > thresh).sum().item())

            target_logits = outputs["target_logits"]

            # Gender: MSE output in [-1, 1], negative=masculine, positive=feminine
            gender_str = "neutral"
            if "gender" in target_logits:
                g_val = float(target_logits["gender"][0].item())
                if g_val < -0.3:
                    gender_str = "masc"
                elif g_val > 0.3:
                    gender_str = "fem"

            # Grammaticality
            gram_str = "gram"
            if "grammatic" in target_logits:
                gram_val = float(target_logits["grammatic"][0].item())
                if gram_val < 0.0:
                    gram_str = "ungram"

            # Grammar points: indices with sigmoid > 0.5
            gp_list: list[str] = []
            if "grammar_point" in target_logits:
                gp_logits = target_logits["grammar_point"][0]
                gp_probs = torch.sigmoid(gp_logits)
                for idx in torch.where(gp_probs > 0.5)[0].tolist():
                    gp_list.append(f"gp{int(idx):04d}")
                if expected_gp is not None and expected_gp < len(gp_probs):
                    expected_gp_prob = float(gp_probs[expected_gp].item())

            # Reconstruction: decode predicted tokens back to text
            recon_str = ""
            if "recon" in target_logits:
                recon_logits = target_logits["recon"][0]  # [S, vocab]
                pred_ids = recon_logits.argmax(dim=-1).tolist()  # [S]
                id_to_tok = self._get_surface_id_to_token()
                tokens = []
                for pid in pred_ids:
                    t = id_to_tok.get(pid, f"#{pid}")
                    if t.startswith("<"):
                        continue
                    tokens.append(t)
                recon_str = "".join(tokens)

        if was_training:
            self.model.train()

        # Format: truncate gp list to avoid wrapping
        gps_str = ",".join(gp_list[:8])
        if len(gp_list) > 8:
            gps_str += f"..+{len(gp_list) - 8}"
        base = f"{sentence} kcs={kc_count} {gender_str} {gram_str} gps={gps_str}"
        if recon_str:
            base += f" → {recon_str}"
        return base, expected_gp_prob

    # (bin_label, sentence, expected_gp_index)
    _CANARIES: List[Tuple[str, str, int]] = [
        ("1-3", "食べます", 1007),  # gp1007 = Verb[ます]
        ("4-7", "ああ、もう無理だ。", 1),  # gp0001 = だ (copula)
        (
            "8-15",
            "昨日の夜に降った雪はとても綺麗だったわ",
            466,
        ),  # gp0466 = わ (feminine)
    ]

    def _build_canary_fields(self, skip: bool) -> Dict[str, Any]:
        """Evaluate all canary sentences and return fields for KcEpochSummary."""
        if skip:
            return {"canary_texts": {}, "canary_gp_probs": {}, "canary_gp_labels": {}}
        texts: Dict[str, str] = {}
        gp_probs: Dict[str, float] = {}
        gp_labels: Dict[str, str] = {}
        for bin_label, sentence, expected_gp in self._CANARIES:
            text, prob = self._evaluate_canary(sentence, expected_gp=expected_gp)
            texts[bin_label] = text
            gp_labels[bin_label] = f"gp{expected_gp:04d}"
            if prob is not None:
                gp_probs[bin_label] = prob
        return {
            "canary_texts": texts,
            "canary_gp_probs": gp_probs,
            "canary_gp_labels": gp_labels,
        }

    def _iter_layer_health_batches(self, num_batches: int) -> Any:
        """Yield batches for layer health, interleaving gram/ungram to match training."""
        gram_len = self._loader_len(self.gram_loader)
        ungram_len = self._loader_len(self.ungram_loader)
        if ungram_len == 0 or not isinstance(self.ungram_loader, DataLoader):
            for i, batch in enumerate(self.gram_loader):
                if i >= num_batches:
                    break
                yield batch
            return
        if gram_len == 0:
            for i, batch in enumerate(self.ungram_loader):
                if i >= num_batches:
                    break
                yield batch
            return
        gram_iter = iter(self.gram_loader)
        ungram_iter = iter(self.ungram_loader)
        remaining_gram = gram_len
        remaining_ungram = ungram_len
        yielded = 0
        while yielded < num_batches and (remaining_gram > 0 or remaining_ungram > 0):
            total = remaining_gram + remaining_ungram
            pick_gram = (
                random.random() < (remaining_gram / total) if total > 0 else False
            )
            if pick_gram and remaining_gram > 0:
                batch = next(gram_iter, None)
                if batch is None:
                    remaining_gram = 0
                    continue
                yield batch
                remaining_gram -= 1
                yielded += 1
            elif remaining_ungram > 0:
                batch = next(ungram_iter, None)
                if batch is None:
                    remaining_ungram = 0
                    continue
                yield batch
                remaining_ungram -= 1
                yielded += 1

    def _compute_layer_health(
        self,
        num_batches: int = 16,
    ) -> Optional[LayerHealthStats]:
        """Compute layer health diagnostics, averaged over multiple batches.

        Samples from gram and ungram loaders in the same proportion as training,
        so CKA/delta_norm/rank reflect the encoder's behavior on the full input
        distribution.  Hooks each TransformerEncoderLayer to capture intermediate
        representations, then computes residual contribution norms, adjacent-layer
        CKA, and effective rank.  Adapts to any number of layers.
        """
        encoder = self.model.encoder
        if not hasattr(encoder, "layers"):
            return None
        layers = list(encoder.layers)
        n_layers = len(layers)
        if n_layers == 0:
            return None

        # Accumulate over batches for stable estimates
        delta_norm_sums: List[float] = [0.0] * n_layers
        delta_norm_counts: List[int] = [0] * n_layers
        pooled_concat: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]
        batch_count = 0

        def _hook(
            _mod: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor
        ) -> None:
            layer_inputs.append(inp[0].detach())
            layer_outputs.append(out.detach())

        was_training = self.model.training
        self.model.eval()
        try:
            for batch in self._iter_layer_health_batches(num_batches):
                if batch_count >= num_batches:
                    break
                enc_keys = {f"input_ids_{f}" for f in ENCODER_FEATURE_FIELDS}
                field_inputs = {
                    k: v.to(self.device)
                    for k, v in batch.feature_inputs.items()
                    if k in enc_keys
                }
                attention_mask = batch.attention_mask.to(self.device)

                layer_inputs: List[torch.Tensor] = []
                layer_outputs: List[torch.Tensor] = []
                hooks = [layer.register_forward_hook(_hook) for layer in layers]

                try:
                    with torch.no_grad():
                        self.model.get_encoder_output(field_inputs, attention_mask)
                finally:
                    for h in hooks:
                        h.remove()

                if len(layer_inputs) != n_layers:
                    continue

                # 1. Residual contribution norms per layer
                for i in range(n_layers):
                    inp = layer_inputs[i]
                    out = layer_outputs[i]
                    delta = (out - inp).norm(dim=-1).mean().item()
                    in_norm = inp.norm(dim=-1).mean().item()
                    delta_norm_sums[i] += delta / max(in_norm, 1e-8)
                    delta_norm_counts[i] += 1

                # 2. Mean-pool over sequence -> (batch, d_model) for CKA and rank
                for i, out in enumerate(layer_outputs):
                    pooled_concat[i].append(out.mean(dim=1))

                batch_count += 1
        finally:
            if was_training:
                self.model.train()

        if batch_count == 0:
            return None

        # 1. Average delta norms
        delta_norms = [
            s / max(c, 1) for s, c in zip(delta_norm_sums, delta_norm_counts)
        ]

        # 2. Concatenate pooled across batches for stable CKA and rank
        pooled = [
            torch.cat(tensors, dim=0) if tensors else None for tensors in pooled_concat
        ]
        valid_pooled: List[torch.Tensor] = [p for p in pooled if p is not None]
        if len(valid_pooled) != n_layers or any(p.shape[0] == 0 for p in valid_pooled):
            return None

        # 3. Adjacent-layer CKA on concatenated data
        cka_adj: List[float] = []
        for i in range(n_layers - 1):
            cka_adj.append(_linear_cka(valid_pooled[i], valid_pooled[i + 1]))

        # 4. Effective rank per layer
        ranks: List[float] = []
        for p in valid_pooled:
            ranks.append(_effective_rank_90(p))

        return LayerHealthStats(
            delta_norm=delta_norms,
            cka_adjacent=cka_adj,
            effective_rank=ranks,
            num_layers=n_layers,
        )

    # pylint: disable=too-many-locals
    def _init_structural_decoder_biases(self, num_batches: int = 10) -> None:
        m = self.model
        if not hasattr(m, "kc_decoders"):
            return
        if not self.config.kc_target_specs:
            return

        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        biases = KCStructuralBiases(sums=sums, counts=counts)
        # Track positive densities (positives per example) for adaptive weighting
        pos_density_sums: Dict[str, float] = {}
        pos_density_counts: Dict[str, int] = {}

        if self._loader_len(self.gram_loader) == 0:
            return
        for i, batch in enumerate(self.gram_loader):
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

                    if gp_mask_key in kc_targets:
                        # PNU families (GRAMMAR_POINT)
                        pos_mask_t = kc_targets[gp_mask_key]
                        batch_size = pos_mask_t.size(0)
                        num_pos = pos_mask_t.sum().item()
                        # IMPORTANT:
                        # For multi-label families like grammar_point, bias init should
                        # reflect the *per-label* base rate, not the fraction of filled
                        # slots in the fixed-width ragged buffer (pos_mask_t.size(1) is
                        # max_pos, typically 64).
                        #
                        # We want: p ≈ E[#pos labels per example] / vocab_size.
                        # This keeps initial logits appropriately negative and prevents
                        # the decoder from starting (and staying) overly "positive",
                        # which otherwise yields hundreds of predicted GPs per sentence.
                        p = num_pos / max(1, batch_size * vocab_size)
                        biases.sums[name] = biases.sums.get(name, 0.0) + p
                        biases.counts[name] = biases.counts.get(name, 0) + 1
                        pos_density_sums[name] = (
                            pos_density_sums.get(name, 0.0) + num_pos
                        )
                        pos_density_counts[name] = (
                            pos_density_counts.get(name, 0) + batch_size
                        )
                    elif continuous_key not in kc_targets:
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

            # Check if this is the GP decoder, a label family, or MSE family
            if name == "grammar_point" and hasattr(m.kc_decoders, "gp_decoder"):
                lin = m.kc_decoders.gp_decoder
                if lin.bias is not None:
                    nn.init.constant_(lin.bias, b)
            elif name in m.kc_decoders.decoders:
                lin = m.kc_decoders.decoders[name]
                if lin.bias is not None:
                    nn.init.constant_(lin.bias, b)
            elif name in m.kc_decoders.mse_decoders:
                lin = m.kc_decoders.mse_decoders[name]
                if lin.bias is not None:
                    nn.init.constant_(lin.bias, b)
            else:
                continue  # Family not in any decoder dict

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
        self.scaler.unscale_(self.optimizer)

        if not self.use_amp:
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
            head_params = set(m.kc_head.parameters())
            if hasattr(m, "kc_decoders"):
                head_params.update(m.kc_decoders.parameters())

            enc_params = [
                p
                for group in self.optimizer.param_groups
                for p in group["params"]
                if p.grad is not None and p not in head_params
            ]

            if enc_params:
                nn.utils.clip_grad_norm_(enc_params, self.config.gradient_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        if not self.use_amp:
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

        return False

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

    def _estimate_total_batches(self) -> int:
        total = self._loader_len(self.ungram_loader) + self._loader_len(
            self.gram_loader
        )
        if total == 0 and isinstance(self.data_loader, Sized):
            return len(self.data_loader)
        return total

    @staticmethod
    def _loader_len(loader: Iterable[Any]) -> int:
        return len(loader) if isinstance(loader, Sized) else 0

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
            # DB-sourced multi-label families (register) use multilabel_ prefix
            if f"kc_multilabel_{name}" in kc_targets:
                continue
            # BERT cloze targets are generated dynamically (not in kc_targets)
            if name == "bert":
                continue
            # Recon targets are generated dynamically (snapshot of surface IDs)
            if name == "recon":
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

        self.view.on_kc_epoch_start(
            epoch,
            self.config.kc_epochs,
            should_freeze,
            batch_size=self.config.batch_size,
        )

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

        total_kl_sparse = 0.0

        total_batches = self._estimate_total_batches()

        running_struct_loss = 0.0
        running_num_struct_total = 0
        running_kl_sparse = 0.0
        running_avg_prob = 0.0
        running_avg_entropy = 0.0
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
            str, WorstSamplesTracker
        ] = {}  # Track highest-loss samples per family

        pending_accum = 0
        did_any_backward = False

        kc_vocab_size = int(self.model.config.kc_vocab_size)
        running_pmax_global = 0.0
        running_avg_prob = 0.0

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
        if hasattr(self.model.kc_decoders, "gp_decoder"):
            gp_dec = self.model.kc_decoders.gp_decoder
            if hasattr(gp_dec, "bias") and gp_dec.bias is not None:
                bias_start["grammar_point"] = gp_dec.bias.detach().clone()
        for name, decoder in self.model.kc_decoders.decoders.items():
            if hasattr(decoder, "bias") and decoder.bias is not None:
                bias_start[name] = decoder.bias.detach().clone()

        pbar = None

        current_display_loss = 0.5
        pbar_desc = f"KC Epoch {epoch + 1}/{self.config.kc_epochs}"
        if should_freeze:
            pbar_desc += " (Encoder Frozen)"

        pbar_batch_size = getattr(self.ungram_loader, "batch_size", None)
        if pbar_batch_size is None:
            pbar_batch_size = getattr(self.gram_loader, "batch_size", None)
        if pbar_batch_size is None:
            pbar_batch_size = self.config.batch_size
        total_elements_target = total_batches * (pbar_batch_size or 1)
        pbar = RichTrainerProgressBar(
            desc=pbar_desc,
            total_steps=total_batches,
            batch_size=pbar_batch_size or 1,
            total_elements_target=total_elements_target,
        )
        self.view.on_kc_progress_init(
            pbar_desc,
            total_steps=total_batches,
        )

        self.train_timer_data.start()

        def _iter_interleaved_batches() -> Any:
            ungram_batches = self._loader_len(self.ungram_loader)
            gram_batches = self._loader_len(self.gram_loader)
            if (
                ungram_batches == 0
                and gram_batches == 0
                and isinstance(self.data_loader, Iterable)
            ):
                yield from self.data_loader  # pylint: disable=not-an-iterable
                return
            if ungram_batches == 0:
                yield from self.gram_loader
                return
            if gram_batches == 0:
                yield from self.ungram_loader
                return

            gram_iter = iter(self.gram_loader)
            ungram_iter = iter(self.ungram_loader)
            remaining_gram = gram_batches
            remaining_ungram = ungram_batches

            while remaining_gram > 0 or remaining_ungram > 0:
                total_remaining = remaining_gram + remaining_ungram
                pick_gram = (
                    random.random() < (remaining_gram / total_remaining)
                    if total_remaining > 0
                    else False
                )

                if pick_gram and remaining_gram > 0:
                    gram_batch = next(gram_iter, None)
                    if gram_batch is None:
                        remaining_gram = 0
                        continue
                    yield gram_batch
                    remaining_gram -= 1
                    continue

                if remaining_ungram > 0:
                    ungram_batch = next(ungram_iter, None)
                    if ungram_batch is None:
                        remaining_ungram = 0
                        continue
                    yield ungram_batch
                    remaining_ungram -= 1
                    continue

                if remaining_gram > 0:
                    gram_batch = next(gram_iter, None)
                    if gram_batch is None:
                        remaining_gram = 0
                        continue
                    yield gram_batch
                    remaining_gram -= 1

        for batch_idx, batch in enumerate(_iter_interleaved_batches()):
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

            track_worst = batch_idx >= total_batches - 12

            full_batch = batch
            nb = self.use_amp
            encoder_keys = {f"input_ids_{f}" for f in ENCODER_FEATURE_FIELDS}
            full_field_inputs = {
                k: v.to(self.device, non_blocking=nb)
                for k, v in full_batch.feature_inputs.items()
                if k in encoder_keys
            }
            full_attention_mask = full_batch.attention_mask.to(
                self.device, non_blocking=nb
            )
            gram_labels = full_batch.grammaticality_labels
            if not torch.is_tensor(gram_labels):
                gram_labels = torch.as_tensor(gram_labels)
            if gram_labels.numel() != full_attention_mask.size(0):
                gram_labels = torch.ones(full_attention_mask.size(0), dtype=torch.long)
            gram_mask_cpu = gram_labels == 1
            has_grammatic = bool(gram_mask_cpu.any().item())
            if not isinstance(self.model, TrainingClassifier):
                gram_mask_cpu = torch.ones_like(gram_mask_cpu, dtype=torch.bool)
                has_grammatic = True
            if has_grammatic and gram_mask_cpu.all():
                batch_label = "gram"
            elif not has_grammatic:
                batch_label = "ungram"
            else:
                batch_label = "mixed"

            # Grammaticality MSE uses pooled classifier (all sentences)
            pooled = None
            gram_probs = None
            gram_targets = None
            gram_loss = torch.tensor(0.0, device=self.device)
            if isinstance(self.model, TrainingClassifier):
                with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                    encoder_output = self.model.get_encoder_output(
                        full_field_inputs, full_attention_mask
                    )
                    pooled = self.model.pooler(encoder_output, full_attention_mask)
                    gram_logits = self.model.grammaticality_classifier(pooled)
                gram_probs = F.softmax(gram_logits.float(), dim=-1)[:, 1]
                gram_targets = full_batch.grammaticality_labels.to(
                    self.device, non_blocking=nb
                ).float()
                gram_loss = F.mse_loss(gram_probs, gram_targets)
                gram_loss = gram_loss * self.config.grammaticality_loss_weight

                if not skip_metrics and kc_diag is not None:
                    kc_diag.update_mse_family(
                        "grammatic", gram_probs, gram_targets, gram_loss.item()
                    )

            if track_worst and gram_probs is not None and gram_targets is not None:
                with torch.no_grad():
                    per_sample_loss = (gram_probs - gram_targets).pow(2)
                    # Skip samples that already match the correct discrete label
                    pred_buckets = discretize_mse(gram_probs, "grammatic")
                    tgt_buckets = discretize_mse(gram_targets, "grammatic")
                    mismatch = pred_buckets != tgt_buckets
                    if mismatch.any():
                        mis_loss = per_sample_loss[mismatch]
                        mis_idxs = mismatch.nonzero(as_tuple=True)[0]
                        tracker = worst_samples.setdefault(
                            "grammatic", WorstSamplesTracker()
                        )
                        k = min(50, mis_loss.size(0))
                        top_vals, top_idxs = torch.topk(mis_loss, k)
                        for ti in range(k):
                            idx_in_batch = int(mis_idxs[top_idxs[ti]].item())
                            loss_val = top_vals[ti].item()
                            sample_idx = int(full_batch.indices[idx_in_batch].item())
                            tracker.push(
                                WorstSampleInfo(
                                    sentence=_get_display_sentence(
                                        self.dataset.get_sentence_by_idx(sample_idx),
                                        full_batch.kotogram[idx_in_batch],
                                    ),
                                    loss=loss_val,
                                    target=gram_targets[idx_in_batch].item(),
                                    prediction=gram_probs[idx_in_batch].item(),
                                    sample_idx=sample_idx,
                                )
                            )

            if not has_grammatic:
                loss = gram_loss / self.config.grad_accum_steps
                running_struct_loss += gram_loss.item()
                running_num_struct_total += 1
                running_loss_components = running_loss_components.add(
                    RunningLossComponents(struct=gram_loss.item())
                )
                if torch.isfinite(loss):
                    self.scaler.scale(loss).backward()
                    did_any_backward = True
                    pending_accum += 1

                    if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                        self._perform_optimizer_step(self.model)
                        pending_accum = 0

                    total_loss += loss.item() * self.config.grad_accum_steps
                    epoch_kc_losses["grammatic"] = (
                        epoch_kc_losses.get("grammatic", 0.0) + gram_loss.item()
                    )
                    n_batches += 1
                else:
                    raise RuntimeError("Non-finite loss detected")

                if pbar:
                    current_display_loss = total_loss / max(1, n_batches)
                    desc = self._format_kc_pbar_desc(pbar_desc, batch_label)
                    pbar.update(batch_idx, current_display_loss, desc=desc)

                self.train_timer_compute.stop(epoch=epoch, batch=batch_idx)
                self.train_timer_data.start()
                continue

            if not gram_mask_cpu.all():
                batch = self._filter_batch_by_mask(full_batch, gram_mask_cpu)

            m = self.model

            kc_targets = create_kc_batch(
                batch=batch,
                tokenizer=self.dataset.tokenizer,
                target_specs=self.config.kc_target_specs,
            )
            for k, v in kc_targets.items():
                kc_targets[k] = v.to(self.device, non_blocking=nb)

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

            gram_mask = gram_mask_cpu.to(self.device, non_blocking=nb)
            attention_mask = full_attention_mask[gram_mask]
            field_inputs = {k: v[gram_mask] for k, v in full_field_inputs.items()}

            # Snapshot original surface IDs for reconstruction target (before any masking)
            recon_targets: Optional[torch.Tensor] = None
            surface_key_recon = "input_ids_surface"
            if self.model.training and surface_key_recon in field_inputs:
                recon_targets = field_inputs[surface_key_recon].clone()

            # Morpheme-cloze target selection: pick K random positions per sample
            # BEFORE any masking, so we have the original token IDs.
            cloze_targets: Optional[torch.Tensor] = None
            cloze_valid: Optional[torch.Tensor] = None
            cloze_k = self._cloze_k
            if self.model.training:
                surface_key = "input_ids_surface"
                if surface_key in field_inputs:
                    surface_ids = field_inputs[surface_key]
                    bs_cloze = surface_ids.size(0)
                    cloze_targets = torch.zeros(
                        bs_cloze, cloze_k, dtype=torch.long, device=self.device
                    )
                    cloze_valid = torch.zeros(
                        bs_cloze, cloze_k, dtype=torch.bool, device=self.device
                    )
                    for i in range(bs_cloze):
                        content_mask = attention_mask[i].bool()
                        token_ids = surface_ids[i]
                        content_positions = (
                            content_mask & (token_ids > MASK_ID)
                        ).nonzero(as_tuple=True)[0]
                        if content_positions.numel() > 0:
                            n_pick = min(cloze_k, content_positions.numel())
                            perm = torch.randperm(
                                content_positions.numel(), device=self.device
                            )[:n_pick]
                            chosen_pos = content_positions[perm]
                            cloze_targets[i, :n_pick] = token_ids[chosen_pos]
                            cloze_valid[i, :n_pick] = True
                            surface_ids[i, chosen_pos] = MASK_ID

            # BERT-style input masking: randomly replace tokens with pad_id=0
            # (zero embedding) while keeping attention_mask=1 so the model
            # knows a token exists but can't see its identity.
            mask_ratio = self.kc_config.input_mask_ratio
            if self.model.training and 0.0 < mask_ratio < 1.0:
                maskable = attention_mask.bool()
                for key in list(field_inputs):
                    ids = field_inputs[key]
                    rand_mask = (torch.rand_like(ids.float()) < mask_ratio) & maskable
                    field_inputs[key] = ids.masked_fill(rand_mask, 0)

            # Compute content_len (approximate: non-pad count)
            # Use attention_mask for robust length calculation even if feature_inputs is empty (e.g. in tests)
            content_len = attention_mask.sum(dim=1).float()

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

            pooled_kc = pooled[gram_mask] if pooled is not None else None
            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                outputs = self.model(
                    field_inputs,
                    attention_mask=attention_mask,
                    mode="kc",
                    temperature=t_val,
                    gumbel_scale=gumbel_scale,
                    grad_cap=self.kc_grad_cap,
                    pooled=pooled_kc,
                )

            should_check_nan = batch_idx < 50 or (batch_idx % 50 == 0)

            if should_check_nan:
                logits_stats = tensor_finite_stats(outputs.get("kc_logits_raw"))
                probs_stats = tensor_finite_stats(outputs.get("kc_probs"))
                forward_nonfinite = not logits_stats.finite or not probs_stats.finite
            else:
                forward_nonfinite = False

            if forward_nonfinite:
                raise RuntimeError("Non-finite values in forward pass")

            # INVARIANT: target_logits batch dimension must match attention_mask
            target_logits = outputs["target_logits"]
            for tl_name, tl_tensor in target_logits.items():
                if tl_tensor.size(0) != expected_batch_size:
                    raise ValueError(
                        f"target_logits['{tl_name}'] batch mismatch: "
                        f"{tl_tensor.size(0)} vs expected {expected_batch_size}"
                    )

            # Update Diagnostic Accumulators
            len_t = content_len.detach().cpu().float()

            all_lens_aligned.extend(len_t.tolist())

            # Track which KC logits fire (prob > 0.5) for utilization reporting
            hot_mask = (outputs["kc_probs"] > 0.5).detach()
            hot_indices = hot_mask.nonzero(as_tuple=False)[:, 1].unique().cpu().tolist()
            kc_logits_used_set.update(hot_indices)

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
                    # BERT family uses cloze_targets, not kc_targets
                    if name == "bert" and cloze_targets is not None:
                        has_match = True
                        break
                    # Recon family uses recon_targets, not kc_targets
                    if name == "recon" and recon_targets is not None:
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
                if is_family_db_sourced(fid):
                    from train.kc import (
                        KcBertFamily,
                        KcDbMultilabelFamily,
                        KcPnuFamily,
                        KcReconFamily,
                    )

                    family_def = get_family(fid)

                    if isinstance(family_def, KcBertFamily):
                        # Morpheme-cloze loss: K cross-entropy terms per sentence
                        if (
                            cloze_targets is not None
                            and cloze_valid is not None
                            and cloze_valid.any()
                        ):
                            logits_bert = logits.float()  # [B, vocab]
                            # Expand logits to match K targets: [B, K, vocab]
                            # cloze_targets: [B, K], cloze_valid: [B, K]
                            any_valid = cloze_valid.any(dim=1)  # [B]
                            flat_logits = logits_bert[any_valid]  # [B', vocab]
                            flat_targets = cloze_targets[any_valid]  # [B', K]
                            flat_valid = cloze_valid[any_valid]  # [B', K]
                            k_dim = flat_targets.size(1)

                            total_ce = torch.tensor(0.0, device=self.device)
                            total_t1 = 0
                            total_t5 = 0
                            total_n = 0
                            for ki in range(k_dim):
                                slot_valid = flat_valid[:, ki]
                                if not slot_valid.any():
                                    continue
                                sl = flat_logits[slot_valid]
                                st = flat_targets[slot_valid, ki]
                                total_ce = total_ce + F.cross_entropy(
                                    sl, st, reduction="sum"
                                )
                                total_n += int(slot_valid.sum().item())
                                if not skip_metrics and kc_diag is not None:
                                    with torch.no_grad():
                                        total_t1 += int(
                                            (sl.argmax(dim=1) == st).sum().item()
                                        )
                                        _, t5 = sl.topk(5, dim=1)
                                        total_t5 += int(
                                            (t5 == st.unsqueeze(1))
                                            .any(dim=1)
                                            .sum()
                                            .item()
                                        )

                            if total_n > 0:
                                raw_ce_loss = total_ce / total_n
                                task_loss = raw_ce_loss * family_def.loss_weight
                                batch_kc_losses[name] = task_loss.item()
                                structural_loss = structural_loss + task_loss
                                num_struct += 1

                                if not skip_metrics and kc_diag is not None:
                                    kc_diag.update_bert_family(
                                        name,
                                        loss=task_loss.item(),
                                        top1_correct=total_t1,
                                        top5_correct=total_t5,
                                        n_samples=total_n,
                                    )
                        continue

                    if isinstance(family_def, KcReconFamily):
                        # Reconstruction loss: CE against original surface IDs
                        if recon_targets is not None:
                            recon_logits = logits.float()
                            decoder = self.model.kc_decoders
                            positions = getattr(decoder, "_last_recon_positions", None)
                            valid = getattr(decoder, "_last_recon_valid", None)

                            if positions is not None and valid is not None:
                                # Training: sampled K positions per sentence
                                # recon_logits: (B, K, V), positions: (B, K), valid: (B, K)
                                tgt = torch.gather(recon_targets, 1, positions)
                                flat_logits = recon_logits[valid]
                                flat_targets = tgt[valid]
                            else:
                                # Eval: full sequence
                                mask = attention_mask.bool()
                                flat_logits = recon_logits[mask]
                                flat_targets = recon_targets[mask]

                            total_n_recon = int(flat_targets.numel())

                            if total_n_recon > 0:
                                raw_ce = F.cross_entropy(
                                    flat_logits, flat_targets, reduction="mean"
                                )
                                task_loss = raw_ce * family_def.loss_weight
                                batch_kc_losses[name] = task_loss.item()
                                structural_loss = structural_loss + task_loss
                                num_struct += 1

                                if not skip_metrics and kc_diag is not None:
                                    with torch.no_grad():
                                        preds = flat_logits.argmax(dim=1)
                                        t1 = int((preds == flat_targets).sum().item())
                                        _, t5 = flat_logits.topk(
                                            min(5, flat_logits.size(1)), dim=1
                                        )
                                        t5_correct = int(
                                            (t5 == flat_targets.unsqueeze(1))
                                            .any(dim=1)
                                            .sum()
                                            .item()
                                        )
                                        t1_pos_only = 0
                                        if positions is not None and valid is not None:
                                            baseline = decoder.recon_position_only(
                                                positions
                                            )
                                            bl_logits = baseline[name][valid]
                                            t1_pos_only = int(
                                                (
                                                    bl_logits.argmax(dim=1)
                                                    == flat_targets
                                                )
                                                .sum()
                                                .item()
                                            )
                                    kc_diag.update_bert_family(
                                        name,
                                        loss=task_loss.item(),
                                        top1_correct=t1,
                                        top5_correct=t5_correct,
                                        n_samples=total_n_recon,
                                        top1_pos_only_correct=t1_pos_only,
                                    )
                        continue

                    if isinstance(family_def, KcPnuFamily):
                        # PNU loss for grammar points (pos/neg arrays)
                        gp_pos_key = f"kc_gp_pos_inds_{name}"
                        gp_pos_mask_key = f"kc_gp_pos_mask_{name}"
                        gp_neg_key = f"kc_gp_neg_inds_{name}"
                        gp_neg_mask_key = f"kc_gp_neg_mask_{name}"

                        if gp_pos_key in kc_targets:
                            pos_ids = kc_targets[gp_pos_key]
                            pos_mask = kc_targets[gp_pos_mask_key]
                            neg_ids = kc_targets[gp_neg_key]
                            neg_mask = kc_targets[gp_neg_mask_key]

                            task_loss, loss_by_label = self._multilabel_pnu_loss(
                                logits.float(),
                                pos_ids,
                                pos_mask,
                                neg_ids,
                                neg_mask,
                                vocab_size=vocab_size,
                                priors=self._gp_prior_tensor,
                                unlabeled_weight=self.kc_config.gp_unlabeled_weight,
                            )

                            # Apply per-family loss weight for balanced training
                            task_loss = task_loss * family_def.loss_weight
                            loss_by_label = loss_by_label * family_def.loss_weight

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
                                loss_by_label=loss_by_label,
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

                            if track_worst:
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

                                def _fmt_gp_plain(gid: int) -> str:
                                    return f"gp{gid:04d}"

                                # Track top-K worst samples for grammar_point
                                tracker = worst_samples.setdefault(
                                    name, WorstSamplesTracker()
                                )
                                k = min(50, per_sample_loss.size(0))
                                top_vals, top_idxs = torch.topk(per_sample_loss, k)
                                for ti in range(k):
                                    idx_b = int(top_idxs[ti].item())
                                    loss_val = top_vals[ti].item()
                                    if loss_val <= 0:
                                        break
                                    target_count = labeled_pos[idx_b].sum().item()
                                    pred_count = (probs_full[idx_b] > 0.5).sum().item()
                                    pos_gp_ids = pos_ids[idx_b][
                                        pos_mask[idx_b]
                                    ].tolist()
                                    if pos_gp_ids:
                                        target_labels = ",".join(
                                            _fmt_gp_plain(gid) for gid in pos_gp_ids[:5]
                                        )
                                        if len(pos_gp_ids) > 5:
                                            target_labels += (
                                                f"...+{len(pos_gp_ids) - 5}"
                                            )
                                    else:
                                        target_labels = "none"
                                    sample_probs = probs_full[idx_b]
                                    pred_mask = sample_probs > 0.5
                                    pred_indices = pred_mask.nonzero(as_tuple=True)[0]
                                    pred_gp_probs = sample_probs[pred_indices]
                                    sorted_order = pred_gp_probs.argsort(
                                        descending=True
                                    )
                                    pred_gp_ids = pred_indices[sorted_order].tolist()
                                    if pred_gp_ids:
                                        pred_labels = ",".join(
                                            _fmt_gp_plain(gid) for gid in pred_gp_ids
                                        )
                                    else:
                                        pred_labels = "none"
                                    sample_idx = int(batch.indices[idx_b].item())
                                    tracker.push(
                                        WorstSampleInfo(
                                            sentence=_get_display_sentence(
                                                self.dataset.get_sentence_by_idx(
                                                    sample_idx
                                                ),
                                                batch.kotogram[idx_b],
                                            ),
                                            loss=loss_val,
                                            target=float(target_count),
                                            prediction=float(pred_count),
                                            sample_idx=sample_idx,
                                            target_labels=target_labels,
                                            pred_labels=pred_labels,
                                        )
                                    )

                                # Also push FALSE POSITIVE samples into same tracker
                                unlabeled_mask = (labeled_pos == 0).float()
                                pred_positive_mask = (probs_full > 0.5).float()
                                fp_mask = unlabeled_mask * pred_positive_mask
                                per_sample_fp_loss = (bce_full * fp_mask).sum(dim=1)

                                if per_sample_fp_loss.max().item() > 0:
                                    k_fp = min(50, per_sample_fp_loss.size(0))
                                    fp_vals, fp_idxs = torch.topk(
                                        per_sample_fp_loss, k_fp
                                    )
                                    for ti in range(k_fp):
                                        fp_idx_b = int(fp_idxs[ti].item())
                                        fp_loss_val = fp_vals[ti].item()
                                        if fp_loss_val <= 0:
                                            break
                                        fp_sample_probs = probs_full[fp_idx_b]
                                        fp_pred_mask = fp_mask[fp_idx_b] > 0.5
                                        fp_pred_indices = fp_pred_mask.nonzero(
                                            as_tuple=True
                                        )[0]
                                        fp_gp_probs = fp_sample_probs[fp_pred_indices]
                                        fp_sorted = fp_gp_probs.argsort(descending=True)
                                        fp_pred_gp_ids = fp_pred_indices[
                                            fp_sorted
                                        ].tolist()
                                        fp_target_labels = "none"
                                        if fp_pred_gp_ids:
                                            fp_pred_labels = ",".join(
                                                _fmt_gp_plain(gid)
                                                for gid in fp_pred_gp_ids
                                            )
                                        else:
                                            fp_pred_labels = "none"
                                        fp_sample_idx = int(
                                            batch.indices[fp_idx_b].item()
                                        )
                                        tracker.push(
                                            WorstSampleInfo(
                                                sentence=_get_display_sentence(
                                                    self.dataset.get_sentence_by_idx(
                                                        fp_sample_idx
                                                    ),
                                                    batch.kotogram[fp_idx_b],
                                                ),
                                                loss=fp_loss_val,
                                                target=0.0,
                                                prediction=float(len(fp_pred_gp_ids)),
                                                sample_idx=fp_sample_idx,
                                                target_labels=fp_target_labels,
                                                pred_labels=fp_pred_labels,
                                            )
                                        )

                    elif isinstance(family_def, KcDbMultilabelFamily):
                        # Multi-label classification for register
                        # Targets are multi-hot [B, num_classes] (can have multiple active)
                        multilabel_key = f"kc_multilabel_{name}"
                        if multilabel_key in kc_targets:
                            targets_multilabel = kc_targets[multilabel_key].float()

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
                            if track_worst:
                                with torch.no_grad():
                                    # Compute per-sample BCE loss (sum across labels)
                                    per_sample_bce = F.binary_cross_entropy_with_logits(
                                        logits.float(),
                                        targets_multilabel,
                                        reduction="none",
                                    ).sum(dim=1)  # Sum BCE across labels
                                    pred_probs = torch.sigmoid(logits.float())
                                    tracker = worst_samples.setdefault(
                                        name, WorstSamplesTracker()
                                    )
                                    k = min(50, per_sample_bce.size(0))
                                    top_vals, top_idxs = torch.topk(per_sample_bce, k)
                                    for ti in range(k):
                                        idx_b = int(top_idxs[ti].item())
                                        loss_val = top_vals[ti].item()
                                        # For target/pred: count of active labels
                                        target_count = (
                                            targets_multilabel[idx_b].sum().item()
                                        )
                                        pred_count = (
                                            (pred_probs[idx_b] > 0.5).sum().item()
                                        )
                                        # Get register names for target and prediction
                                        target_ids = (
                                            targets_multilabel[idx_b]
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
                                            (pred_probs[idx_b] > 0.5)
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
                                        sample_idx = int(batch.indices[idx_b].item())
                                        tracker.push(
                                            WorstSampleInfo(
                                                sentence=_get_display_sentence(
                                                    self.dataset.get_sentence_by_idx(
                                                        sample_idx
                                                    ),
                                                    batch.kotogram[idx_b],
                                                ),
                                                loss=loss_val,
                                                target=float(target_count),
                                                prediction=float(pred_count),
                                                sample_idx=sample_idx,
                                                target_labels=",".join(target_names)
                                                or "none",
                                                pred_labels=",".join(pred_names)
                                                or "none",
                                            )
                                        )
                    else:
                        # MSE loss for continuous families (gender/formality)
                        # Get target values from the original batch
                        target_key = f"kc_continuous_{name}"
                        if target_key in kc_targets:
                            targets_cont = kc_targets[target_key].float()

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
                            if track_worst:
                                with torch.no_grad():
                                    preds = logits.float().squeeze(-1)
                                    per_sample_loss = (preds - targets_cont).pow(2)
                                    # Skip samples that already match the correct discrete label
                                    pred_buckets = discretize_mse(preds, name)
                                    tgt_buckets = discretize_mse(targets_cont, name)
                                    mismatch = pred_buckets != tgt_buckets
                                    if mismatch.any():
                                        mis_loss = per_sample_loss[mismatch]
                                        mis_idxs = mismatch.nonzero(as_tuple=True)[0]
                                        tracker = worst_samples.setdefault(
                                            name, WorstSamplesTracker()
                                        )
                                        k = min(50, mis_loss.size(0))
                                        top_vals, top_idxs = torch.topk(mis_loss, k)
                                        for ti in range(k):
                                            idx_b = int(mis_idxs[top_idxs[ti]].item())
                                            loss_val = top_vals[ti].item()
                                            sample_idx = int(
                                                batch.indices[idx_b].item()
                                            )
                                            tracker.push(
                                                WorstSampleInfo(
                                                    sentence=_get_display_sentence(
                                                        self.dataset.get_sentence_by_idx(
                                                            sample_idx
                                                        ),
                                                        batch.kotogram[idx_b],
                                                    ),
                                                    loss=loss_val,
                                                    target=targets_cont[idx_b].item(),
                                                    prediction=preds[idx_b].item(),
                                                    sample_idx=sample_idx,
                                                )
                                            )

                    continue  # Skip standard dense/sparse path

                if dense_key in kc_targets:
                    targets = kc_targets[dense_key].float()
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

                        # Balanced Dense Loss on full targets.
                        bce_all = F.binary_cross_entropy_with_logits(
                            logits_f, targets, reduction="none"
                        )
                        pos_mask_d = (targets > 0.5).float()
                        neg_mask_d = (targets < 0.5).float()
                        n_pos_d = pos_mask_d.sum().clamp_min(1.0)
                        n_neg_d = neg_mask_d.sum().clamp_min(1.0)
                        loss_pos_d = (bce_all * pos_mask_d).sum() / n_pos_d
                        loss_neg_d = (bce_all * neg_mask_d).sum() / n_neg_d

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
                    pos_inds = kc_targets[pos_key]
                    pos_mask_t = kc_targets[mask_key]
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

            # Add grammaticality MSE (pooler-based) to structural loss
            structural_loss = structural_loss + gram_loss
            num_struct += 1
            batch_kc_losses["grammatic"] = gram_loss.item()

            if num_struct > 0:
                running_struct_loss += structural_loss.item()
                running_num_struct_total += 1
            # Build combined_loss from components (clone to avoid aliasing)
            loss_primary_val = structural_loss.item()
            combined_loss = structural_loss.clone()

            if relative_epoch < self.freeze_encoder_epochs:
                div_weight = self.kc_diversity_weight_frozen
            else:
                div_weight = self.kc_diversity_weight_thawed

            loss_div_val = 0.0
            loss_entropy_val = 0.0
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

                # Precompute softmax over all rows (per-row op, safe to share)
                q_all = torch.softmax(logit_ref / tau_usage, dim=-1)

                for s in splits:
                    mask = s["mask"]
                    mask_f = mask.float().unsqueeze(1)  # (B, 1)
                    mask_count = mask.sum().clamp_min(1).float()
                    weight = mask_count / total_n

                    # Mask-weighted mean avoids MPS boolean indexing bugs
                    p = (q_all * mask_f).sum(dim=0) / mask_count

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

            # Coverage Loss: Encourage all KC logits to be used / follow Zipf
            # Modes:
            # - threshold: penalize logits whose max prob stays below threshold
            # - zipf: fit per-batch usage distribution to Zipf's law
            coverage_weight = (
                self.kc_config.coverage_weight
                if relative_epoch < self.freeze_encoder_epochs
                else self.kc_config.coverage_weight_thawed
            )

            kc_probs = torch.sigmoid(outputs["kc_logits_effective"])

            if coverage_weight > 0:
                min_threshold = self.kc_config.coverage_min_prob
                if self.kc_config.coverage_mode == "zipf":
                    # Soft indicator for "usage" to keep gradients smooth
                    tau = max(self.kc_config.coverage_zipf_tau, 1e-6)
                    eps = self.kc_config.coverage_zipf_eps
                    zipf_s = self.kc_config.coverage_zipf_s

                    usage_soft = torch.sigmoid((kc_probs - min_threshold) / tau)
                    usage = usage_soft.mean(dim=0)  # [vocab_size], fraction per logit
                    usage_sorted, _ = usage.sort(descending=True)

                    # Observed distribution
                    obs = usage_sorted + eps
                    obs = obs / obs.sum()

                    # Zipf target distribution
                    ranks = torch.arange(
                        1, obs.numel() + 1, device=obs.device, dtype=obs.dtype
                    )
                    target = torch.pow(ranks, -zipf_s)
                    target = target / target.sum()
                    target = target + eps
                    target = target / target.sum()

                    # KL divergence: obs || target
                    coverage_loss = torch.sum(obs * (obs.log() - target.log()))
                else:
                    # Max probability each KC achieves across the batch
                    # Shape: kc_probs is [batch_size, vocab_size]
                    max_probs_per_kc = kc_probs.max(dim=0)[0]  # [vocab_size]

                    # Penalize KCs that don't reach minimum threshold
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

            kc_probs = outputs["kc_probs"]
            avg_prob = kc_probs.mean()
            running_avg_prob += avg_prob.item()

            # --- KL-Sparse: per-slot Bernoulli KL with length-adaptive target ρ ---
            if self.kl_sparse_weight > 0:
                # Update median length EMA
                batch_median_len = float(content_len.median().item())
                self.median_content_len = (
                    0.95 * self.median_content_len + 0.05 * batch_median_len
                )

                # Compute per-example target ρ(L)
                med_len = max(1.0, self.median_content_len)
                if self.rho_length_scale == "sqrt":
                    rho_per_ex = self.kl_target_rho * torch.sqrt(content_len / med_len)
                elif self.rho_length_scale == "log":
                    rho_per_ex = self.kl_target_rho * (
                        torch.log1p(content_len) / math.log(1.0 + med_len)
                    )
                else:  # "none"
                    rho_per_ex = torch.full_like(content_len, self.kl_target_rho)
                rho_per_ex = rho_per_ex.clamp(0.005, 0.20)
                # Batch-average target (scalar)
                rho = rho_per_ex.mean()

                # Per-slot batch-average activation
                rho_hat = kc_probs.mean(dim=0)  # [vocab_size]
                # Clamp for numerical stability in log
                rho_hat = rho_hat.clamp(1e-7, 1.0 - 1e-7)
                rho_c = rho.clamp(1e-7, 1.0 - 1e-7)

                # Bernoulli KL: sum over slots, gives scalar
                kl_term = (
                    rho_hat * torch.log(rho_hat / rho_c)
                    + (1.0 - rho_hat) * torch.log((1.0 - rho_hat) / (1.0 - rho_c))
                ).sum()

                if not torch.isfinite(kl_term):
                    raise RuntimeError("Non-finite kl_term")
            else:
                kl_term = torch.tensor(0.0, device=self.device)

            total_kl_sparse += float(kl_term.detach().item())
            running_kl_sparse += kl_term.item()

            # --- Per-probability entropy penalty: push each p_i toward 0 or 1 ---
            if self.entropy_weight > 0:
                p_clamped = kc_probs.clamp(1e-7, 1.0 - 1e-7)
                per_prob_entropy = (
                    -p_clamped * p_clamped.log()
                    - (1.0 - p_clamped) * (1.0 - p_clamped).log()
                )
                # Mean over alive KCs and batch elements
                entropy_term = per_prob_entropy.mean()
                combined_loss_for_entropy = self.entropy_weight * entropy_term
                loss_entropy_val = combined_loss_for_entropy.item()
                # Track mean per-prob entropy for diagnostic
                running_avg_entropy += entropy_term.item()
            else:
                entropy_term = torch.tensor(0.0, device=self.device)
                loss_entropy_val = 0.0
                # Compute entropy for diagnostic even when weight is 0
                p_clamped_diag = kc_probs.clamp(1e-7, 1.0 - 1e-7)
                per_prob_h = (
                    -p_clamped_diag * p_clamped_diag.log()
                    - (1.0 - p_clamped_diag) * (1.0 - p_clamped_diag).log()
                )
                running_avg_entropy += per_prob_h.mean().item()

            # --- Covariance Penalty: off-diagonal correlations for orthogonality ---
            if self.cov_penalty_weight > 0:
                centered = kc_probs - kc_probs.mean(dim=0)  # [B, V]
                cov = (centered.T @ centered) / max(1, kc_probs.size(0))  # [V, V]
                cov.fill_diagonal_(0.0)
                cov_term = (cov**2).mean()
            else:
                cov_term = torch.tensor(0.0, device=self.device)

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

            kl_w = self.kl_sparse_weight
            if relative_epoch >= self.freeze_encoder_epochs:
                epoch_idx_thawed = max(0, relative_epoch - self.freeze_encoder_epochs)
                if epoch_idx_thawed < 3:
                    kl_w = 0.5 * self.kl_sparse_weight

            cov_w = self.cov_penalty_weight
            loss = (
                combined_loss
                + kl_w * kl_term
                + cov_w * cov_term
                + self.entropy_weight * entropy_term
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

            loss_kl_val = (kl_w * kl_term).item()
            loss_cov_val = (cov_w * cov_term).item()

            # Build loss components for display (all values are raw sums per batch)
            gas = self.config.grad_accum_steps
            current_epoch_comp = {
                "struct": structural_loss.item(),
                "formality": loss_formality_val * gas,
                "gender": loss_gender_val * gas,
                "register": loss_register_val * gas,
                "div": loss_div_val,
                "entropy": loss_entropy_val,
                "collapse": loss_coll_val,
                "kl_sparse": loss_kl_val,
                "cov_penalty": loss_cov_val,
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
                entropy=current_epoch_comp["entropy"],
                collapse=current_epoch_comp["collapse"],
                kl_sparse=current_epoch_comp["kl_sparse"],
                cov_penalty=current_epoch_comp["cov_penalty"],
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
                self.scaler.scale(loss).backward()
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
                desc = self._format_kc_pbar_desc(pbar_desc, batch_label)
                pbar.update(batch_idx, current_display_loss, desc=desc)

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
            if not skip_metrics:
                self.view.on_kc_batch_stats(
                    epoch=epoch,
                    batch_idx=batch_idx,
                    content_len=content_len.detach(),
                    pmax_per_ex=pmax_per_ex.detach(),
                    kc_probs=outputs["kc_probs_clean"].detach(),
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
            avg_kl_sparse=running_kl_sparse / max(1, n_batches),
            avg_prob=running_avg_prob / max(1, n_batches),
            # Only include diagnostics when metrics are not skipped
            kc_diagnostics=None if skip_metrics else kc_diag.get_stats(),
        )

        pbar.stop()

        self.view.on_kc_progress_stop()

        # --- Epoch Summary Usage Stats ---
        zipf_kl_final: float = 0.0
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

            # Zipf KL (lower is closer to Zipf)
            zipf_s = self.kc_config.coverage_zipf_s
            zipf_eps = self.kc_config.coverage_zipf_eps
            p_sorted, _ = p_mean.sort(descending=True)
            obs = p_sorted + zipf_eps
            obs = obs / obs.sum()
            ranks = torch.arange(1, obs.numel() + 1, device=obs.device, dtype=obs.dtype)
            target = torch.pow(ranks, -zipf_s)
            target = target / target.sum()
            target = target + zipf_eps
            target = target / target.sum()
            zipf_kl_final = float((obs * (obs.log() - target.log())).sum().item())
        else:
            ent_norm_final = 0.0
            kl_u_norm_final = 0.0

        activation_stats = KcEpochActivationStats(
            pmax_global_max=running_pmax_global,
            pmax_p50=0.0,  # Filled by View
            pmax_p90=0.0,
            pmax_p99=0.0,
            ent_norm=ent_norm_final,
            kl_u_norm=kl_u_norm_final,
            kc_probs_mean=running_avg_prob / max(1, n_batches),
            avg_entropy=running_avg_entropy / max(1, n_batches),
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

                # AvgPos / MedPos from accumulator
                if fam_acc.n_pos_ex > 0:
                    fam_diag.avg_pos = fam_acc.cnt_pred_pos_on_pos / fam_acc.n_pos_ex
                    fam_diag.med_pos = fam_acc.median_pred_pos_on_pos()

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
                decoder_lin = None
                if name == "grammar_point" and hasattr(m.kc_decoders, "gp_decoder"):
                    decoder_lin = m.kc_decoders.gp_decoder
                elif name in m.kc_decoders.decoders:
                    decoder_lin = m.kc_decoders.decoders[name]
                if (
                    name in bias_start
                    and decoder_lin is not None
                    and hasattr(decoder_lin, "bias")
                    and decoder_lin.bias is not None
                ):
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

        gp_priors = getattr(self.dataset, "gp_priors", None)
        if torch.is_tensor(gp_priors) and gp_priors.numel() > 0:
            gp_priors_summary = gp_priors.detach().float().cpu()
        else:
            gp_priors_summary = None

        layer_health: Optional[LayerHealthStats] = None
        if not skip_metrics and isinstance(self.gram_loader, DataLoader):
            layer_health = self._compute_layer_health()

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
            kc_logits_used_count=kc_logits_used_count,
            kc_logits_used_percent=kc_logits_used_percent,
            zipf_kl=zipf_kl_final,
            worst_samples=worst_samples,
            accumulators=family_accumulators,
            gp_priors=gp_priors_summary,
            gp_default_prior=self._gp_computed_default_prior,
            total_samples=total_samples_seen,
            **self._build_canary_fields(skip_metrics),
            layer_health=layer_health,
        )

        # Skip full diagnostics for early epochs (performance optimization)
        sizing_metrics = None
        if skip_metrics:
            self.view.on_kc_epoch_metrics_skipped(epoch, total_loss)
        else:
            self.view.on_kc_epoch_summary(epoch, summary)
            # Propagate adaptive threshold to model config for inference
            self.model.config.kc_threshold = summary.kc_threshold
            # Extract sizing metrics for history / MLflow
            if summary.alive_kcs is not None or summary.total_k_mean is not None:
                sizing_metrics = {
                    k: v
                    for k, v in [
                        ("alive_kcs", summary.alive_kcs),
                        ("total_k_mean", summary.total_k_mean),
                        ("total_k_p10", summary.total_k_p10),
                        ("total_k_p50", summary.total_k_p50),
                        ("total_k_p90", summary.total_k_p90),
                        ("kc_threshold", summary.kc_threshold),
                    ]
                    if v is not None
                }
            apc = max(1, self.view.alive_prob_count)
            sizing_metrics = sizing_metrics or {}
            sizing_metrics["s1_pct"] = self.view.sharp1_count / apc
            sizing_metrics["s0_pct"] = self.view.sharp0_count / apc
            sizing_metrics["fuzzy_pct"] = self.view.fuzzy_count / apc
            for bin_label, prob in summary.canary_gp_probs.items():
                sizing_metrics[f"canary_gp_{bin_label}"] = prob

        epoch_result = TrainEpochResult(
            total_loss=total_loss,
            kc_losses=KCLosses(_losses=epoch_kc_losses),
            avg_kl_sparse=total_kl_sparse / max(1, total_batches),
            epoch_stats=epoch_stats,
            sizing_metrics=sizing_metrics,
        )
        self.view.on_kc_epoch_end(epoch, epoch_result=epoch_result)
        return epoch_result

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

    def _maybe_unfreeze_surface(self, epoch_stats: TrainEpochStats) -> None:
        """Unfreeze surface embedding layers when grammar_point PosP crosses threshold."""
        if self._surface_unfrozen_by_ramp:
            return
        diag = epoch_stats.kc_diagnostics
        if diag is None:
            return
        gp_stats = diag.families.get("grammar_point")
        if gp_stats is None:
            return
        posp = gp_stats.prob_pos_mean
        if round(posp * 100) < round(self._ramp_threshold * 100):
            return
        emb_dict = self.model.embedding.embeddings
        if "surface" not in emb_dict:
            return
        surface_emb = emb_dict["surface"]
        if (
            not isinstance(surface_emb, nn.Embedding)
            or surface_emb.weight.requires_grad
        ):
            return
        surface_emb.weight.requires_grad = True
        self._surface_unfrozen_by_ramp = True
        self.view.on_surface_unfrozen_by_ramp()

    def _maybe_ramp(self, epoch_stats: TrainEpochStats) -> None:
        """Bump data ratio when grammar_point PosP crosses the threshold."""
        if self._ramp_step <= 0 or self._current_ratio >= 1.0:
            return
        diag = epoch_stats.kc_diagnostics
        if diag is None:
            return
        gp_stats = diag.families.get("grammar_point")
        if gp_stats is None:
            return
        posp = gp_stats.prob_pos_mean
        if round(posp * 100) < round(self._ramp_threshold * 100):
            return

        old_ratio = self._current_ratio
        new_ratio = min(1.0, self._current_ratio + self._ramp_step)
        self._current_ratio = new_ratio

        self.dataset.resample(new_ratio)
        self._rebuild_dataloaders()
        self.view.on_ramp(old_ratio, new_ratio, posp, self._ramp_threshold)

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
            avg_kl_sparse = epoch_res.avg_kl_sparse
            epoch_stats = epoch_res.epoch_stats

            self._log_training_progress()

            total_batches = max(1, self._estimate_total_batches())
            self.history.total_loss.append(total_loss / total_batches)
            self.history.kc_kl_sparse.append(avg_kl_sparse)
            self.history.avg_struct_loss.append(epoch_stats.avg_struct_loss)
            self.history.num_struct_heads_processed.append(
                float(epoch_stats.num_struct_heads_processed)
            )
            self.history.avg_kl_sparse.append(epoch_stats.avg_kl_sparse)

            # Always append to keep list aligned with epoch indices (None if skipped)
            self.history.kc_diagnostics.append(epoch_stats.kc_diagnostics)

            # Sizing metrics (alive KCs, Total K mean/p10/p50/p90, kc_threshold)
            sm = epoch_res.sizing_metrics
            self.history.alive_kcs.append(sm.get("alive_kcs") if sm else None)
            self.history.total_k_mean.append(sm.get("total_k_mean") if sm else None)
            self.history.total_k_p10.append(sm.get("total_k_p10") if sm else None)
            self.history.total_k_p50.append(sm.get("total_k_p50") if sm else None)
            self.history.total_k_p90.append(sm.get("total_k_p90") if sm else None)
            self.history.kc_threshold.append(sm.get("kc_threshold") if sm else None)
            self.history.s1_pct.append(sm.get("s1_pct") if sm else None)
            self.history.s0_pct.append(sm.get("s0_pct") if sm else None)
            self.history.fuzzy_pct.append(sm.get("fuzzy_pct") if sm else None)
            self.history.canary_gp_1_3.append(sm.get("canary_gp_1-3") if sm else None)
            self.history.canary_gp_4_7.append(sm.get("canary_gp_4-7") if sm else None)
            self.history.canary_gp_8_15.append(sm.get("canary_gp_8-15") if sm else None)

            # Record active KC targets
            active_targets = sorted(list(kc_losses.keys()))
            self.history.active_kc_targets.append(",".join(active_targets))

            for k, v in kc_losses.items():
                if k not in self.history.kc_losses:
                    self.history.kc_losses[k] = []
                self.history.kc_losses[k].append(v)

            sentence_count = (
                len(self.gram_dataset)
                if hasattr(self, "gram_dataset") and self.gram_dataset is not None
                else len(self.dataset)
            )
            self.history.sentence_count.append(sentence_count)

            on_epoch_end(self.history)

            self._maybe_unfreeze_surface(epoch_stats)
            self._maybe_ramp(epoch_stats)

        self.view.on_kc_train_end(self.history)

        return self.history
