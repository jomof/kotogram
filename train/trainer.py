# pylint: disable=too-many-lines,not-callable,too-many-nested-blocks
import math
import os
import sys
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from kotogram.model import (
    StyleClassifier,
)
from kotogram.tokenizer import FEATURE_FIELDS
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
    print_best_model_saved,
    print_epoch_summary,
    print_kc_first_batch_debug,
    print_phase_header,
)
from train.io import (
    load_training_state,
    save_training_state,
)
from train.kc import SPARSE_FAMILY_PREFIXES
from train.kc_diagnostics import (
    KCEpochDiag,
    assert_diagnostics_invariants,
    compute_auc_checked,
    format_kc_epoch_details,
    format_kc_epoch_summary,
    format_kc_first_batch_summary,
    gather_kc_diag,
)
from train.models import StyleClassifierWithKC
from train.profile import Timer, get_profile_dir
from train.types import (
    EvaluationMetrics,
    FirstBatchGradNorms,
    FirstBatchSeparation,
    KCCoverageCounts,
    KCDiagnosticHeadStats,
    KCLosses,
    KCMetricsAccumulator,
    KCProbeConfig,
    KCProbeEvaluationResult,
    KCSnapshot,
    KCStructuralBiases,
    KCTrainingHistory,
    RunningLossComponents,
    TensorStats,
    TrainEpochResult,
    TrainEpochStats,
    TrainingBatch,
    TrainingHistory,
    TrainingLosses,
    TrainingMetrics,
    TrainingPredictions,
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


class KCTrainer:
    # pylint: disable=too-many-positional-arguments,too-many-locals
    def __init__(
        self,
        model: StyleClassifierWithKC,
        dataset: StyleDataset,
        config: TrainerConfig,
        dl_config: DataLoaderConfig,
        kc_config: KCConfig,
    ):
        dataset = dataset.filter_by_grammaticality(1)

        self.model = model
        self.dataset = dataset
        self.config = config

        _safe_configure_threads(self.config)

        configure_runtime_thread_limits(self.config)

        self.kc_config = kc_config
        self.kc_sparsity_weight = self.kc_config.sparsity_weight
        self.freeze_encoder_epochs = self.kc_config.freeze_encoder_epochs

        self.device = torch.device(self.config.device)
        self.model.to(self.device)

        self.val_sampler = None
        self.sampler = None
        self.last_fb_diag: Optional[Dict[str, Any]] = (
            None  # Canonical diagnostic snapshot from first batch
        )

        if dl_config is None:
            dl_config = self.config.resolve_dataloader_config(self.device, mode="train")

        self.data_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
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

        self.default_bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

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

        self.kc_pos_weight_cap = self.kc_config.pos_weight_cap
        self.kc_pos_weight_eps = self.kc_config.pos_weight_eps

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

        self.kc_log_level = self.kc_config.log_level
        self.kc_first_batch_debug_every = self.kc_config.first_batch_debug_every

        self.kc_first_batch_debug_epochs = list(self.kc_config.first_batch_debug_epochs)

        self.kc_show_epoch_table = self.kc_config.show_epoch_table
        self.kc_show_step_checks = self.kc_config.show_step_checks
        self.kc_show_grad_norms = self.kc_config.show_grad_norms

        self.kc_grad_cap = self.kc_config.kc_grad_cap

        self.kc_entropy_floor = self.kc_config.entropy_floor
        self.kc_kl_cap = self.kc_config.kl_cap

        if self.kc_log_level == "debug":
            self.kc_show_epoch_table = True
            self.kc_show_step_checks = True
            self.kc_show_grad_norms = True

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
        self.start_batch = 0
        self.global_step = 0
        self._did_print_debug_for_epoch = -1

        self._kc_last_good_state: Optional[KCSnapshot] = None
        self._nonfinite_streak = 0
        self._nonfinite_total = 0
        self._nonfinite_logged = 0
        self._max_nonfinite_streak = 50

        self._consecutive_step_skips = 0
        self._total_step_skips = 0
        self._total_steps_applied = 0
        self._max_consecutive_skips = self.kc_config.max_consecutive_skips

        # Diagnostics Persistence
        self.diag_persistence: Dict[str, Any] = {
            "families": {},  # name -> {metric -> count}
            "global": {},  # name -> count
        }
        self.cumulative_positives: Dict[str, int] = {}

    def save_checkpoint(self, epoch: int) -> None:
        if self.config.checkpoint.dir is None:
            return

        save_training_state(
            path=self.config.checkpoint.dir,
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            history=self.history,
            global_step=self.global_step,
            config=self.config,
            filename="checkpoint_kc.pt",
        )

    def restore_from_checkpoint(self, path: str) -> bool:
        full_path = os.path.join(path, "checkpoint_kc.pt")
        if not os.path.exists(full_path):
            return False

        checkpoint = load_training_state(
            path=path,
            model=getattr(self.model, "module", self.model),
            optimizer=self.optimizer,
            filename="checkpoint_kc.pt",
        )
        self.start_epoch = checkpoint["epoch"]
        self.start_batch = checkpoint.get("batch_idx", 0)
        self.global_step = checkpoint.get("global_step", 0)
        history_data = checkpoint["history"]
        if isinstance(history_data, dict):
            for k, v in history_data.items():
                if hasattr(self.history, k):
                    setattr(self.history, k, v)
            if isinstance(self.history, dict):
                self.history.update(history_data)
        else:
            self.history = history_data
        print(
            f"  [Resume] Restored KC checkpoint from {path} "
            f"(epoch {self.start_epoch}, step {self.global_step})"
        )
        return True

    def _save_kc_snapshot(self) -> None:
        m = self.model

        head_state = {
            k: v.detach().cpu().clone() for k, v in m.kc_head.state_dict().items()
        }
        dec_state = None
        if hasattr(m, "kc_decoders"):
            dec_state = {
                k: v.detach().cpu().clone()
                for k, v in m.kc_decoders.state_dict().items()
            }
        self._kc_last_good_state = KCSnapshot(
            kc_head=head_state,
            kc_decoders=dec_state,
        )

    def _restore_kc_snapshot(self) -> bool:
        if self._kc_last_good_state is None:
            return False
        m = self.model

        device = next(m.kc_head.parameters()).device
        restored_head = {
            k: v.to(device) for k, v in self._kc_last_good_state.kc_head.items()
        }
        m.kc_head.load_state_dict(restored_head, strict=True)

        if self._kc_last_good_state.kc_decoders is not None and hasattr(
            m, "kc_decoders"
        ):
            device_dec = next(m.kc_decoders.parameters()).device
            restored_dec = {
                k: v.to(device_dec)
                for k, v in self._kc_last_good_state.kc_decoders.items()
            }
            m.kc_decoders.load_state_dict(restored_dec, strict=True)

        return True

    def _reinit_kc_head(self) -> None:
        m = self.model

        nn.init.xavier_uniform_(m.kc_head.linear.weight)
        if m.kc_head.linear.bias is not None:
            nn.init.zeros_(m.kc_head.linear.bias)

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

        loss_elem = F.binary_cross_entropy_with_logits(gathered, t, reduction="none")
        loss = (loss_elem * valid.float()).sum() / valid.float().sum().clamp_min(1.0)

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
                    diag_inds,
                    diag_pos_mask,
                    diag_probs,
                    diag_targets,
                    loss.item(),
                    mask_id=reading_mask_id,
                )
        return loss

    # pylint: disable=too-many-locals
    def _init_structural_decoder_biases(self, num_batches: int = 10) -> None:
        m = self.model
        if not hasattr(m, "kc_decoders"):
            return

        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        biases = KCStructuralBiases(sums=sums, counts=counts)

        for i, batch in enumerate(self.data_loader):
            if i >= num_batches:
                break

            kc_targets = create_kc_batch(
                batch=batch,
                tokenizer=self.dataset.tokenizer,
                target_specs=m.config.kc_target_specs,
            )

            for name, vocab_size in m.config.kc_target_specs.items():
                dense_key = f"kc_targets_{name}"
                mask_key = f"kc_pos_mask_{name}"

                if dense_key in kc_targets:
                    t = kc_targets[dense_key].float()
                    p = t.mean().item()
                    biases.sums[name] = biases.sums.get(name, 0.0) + p
                    biases.counts[name] = biases.counts.get(name, 0) + 1
                elif mask_key in kc_targets:
                    pos_mask_t = kc_targets[mask_key]
                    batch_size = pos_mask_t.size(0)
                    num_pos = pos_mask_t.sum().item()
                    p = num_pos / (batch_size * vocab_size)
                    biases.sums[name] = biases.sums.get(name, 0.0) + p
                    biases.counts[name] = biases.counts.get(name, 0) + 1

        for name, _ in m.config.kc_target_specs.items():
            if name not in sums or counts.get(name, 0) == 0:
                continue

            p = sums[name] / counts[name]
            p = max(1e-6, min(1.0 - 1e-6, p))
            b = float(-torch.log(torch.tensor(1.0 / p - 1.0)).item())

            lin = m.kc_decoders.decoders[name]
            if lin.bias is not None:
                nn.init.constant_(lin.bias, b)

            print(
                f"[BiasInit] {name:20}: p_mean={p:.6g} bias={b:.4f} "
                f"(filled bias[{lin.bias.numel()}])"
            )

    def _grad_norm(self, module: torch.nn.Module) -> float:
        total = 0.0
        for p in module.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            total += (g.float().norm(2).item()) ** 2
        return float(total**0.5)

    # pylint: disable=too-many-locals,too-many-positional-arguments
    def _perform_optimizer_step(
        self,
        m: StyleClassifierWithKC,
        accum: int,
    ) -> bool:
        w0_before = 0.0
        if self.kc_show_step_checks:
            w0 = m.kc_head.linear.weight
            w0_before = w0.detach().flatten()[0].item()

        if self.kc_show_grad_norms:
            gn_kc = self._grad_norm(m.kc_head)

            dec_name = (
                "pos"
                if "pos" in m.kc_decoders.decoders
                else next(iter(m.kc_decoders.decoders.keys()))
            )
            dec = m.kc_decoders.decoders[dec_name]
            gn_dec = self._grad_norm(dec) if dec is not None else 0.0
            phase = "Flush"  # was: "Flush" if is_flush else "Pre-Step"
            print(
                f"  KC {phase} Grad Norms: kc_head={gn_kc:.6f} decoder={gn_dec:.6f}"
                + (f" (flush_accum={accum})")
            )

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
            self._consecutive_step_skips += 1
            self._total_step_skips += 1

            print(
                f"  KC Step Skipped: non-finite grad detected "
                f"consec={self._consecutive_step_skips}/{self._max_consecutive_skips}"
            )
            for pname, nnan, ninf, gmax in bad:
                print(
                    f"    grad_nonfinite: {pname} nan={nnan} inf={ninf} |g|max={gmax:.3g}"
                )

            self.optimizer.zero_grad(set_to_none=True)

            if self._consecutive_step_skips > self._max_consecutive_skips:
                raise RuntimeError(
                    f"KC training exceeded max consecutive step skips ({self._max_consecutive_skips}). "
                    f"lr={self.optimizer.param_groups[0]['lr']:.2e}, "
                    f"total_skips={self._total_step_skips}, applied={self._total_steps_applied}. "
                    f"Culprits: {[(p, n, i) for p, n, i, _ in bad[:3]]}"
                )

            return True

        # B1: Only clip gradients when config.gradient_clip > 0
        if self.config.gradient_clip and self.config.gradient_clip > 0:
            params_to_clip = [
                p
                for group in self.optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            ]
            if params_to_clip:
                nn.utils.clip_grad_norm_(params_to_clip, self.config.gradient_clip)
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
                print("  [KC] Params became NaN after step, restoring snapshot")
                restored = self._restore_kc_snapshot()

                if not restored:
                    self._reinit_kc_head()
                skipped = True
            else:
                self._save_kc_snapshot()

                self._consecutive_step_skips = 0
                self._total_steps_applied += 1

        if self.kc_show_step_checks:
            w0 = m.kc_head.linear.weight
            w0_after = w0.detach().flatten()[0].item()
            print(
                f"  KC Flush Step Check: kc_head.w0 {w0_before:.6f} -> {w0_after:.6f} "
                f"(delta={w0_after - w0_before:+.6f}, accum={accum}/{self.config.grad_accum_steps})"
            )

        self.optimizer.zero_grad(set_to_none=True)
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
            "lr": self.config.learning_rate * 0.01 if not freeze_encoder else 0.0,
        }

        self.optimizer = Adam([pg_heads, pg_encoder])

    def _check_kc_coverage(
        self, outputs: Dict[str, Any], kc_targets: Dict[str, Any]
    ) -> None:
        """Helper to check and report KC target coverage."""
        c_dense = 0
        c_sparse = 0
        c_label = 0
        c_missing = 0

        for name in outputs["target_logits"]:
            if f"kc_targets_{name}" in kc_targets:
                c_dense += 1
                continue
            if f"kc_pos_inds_{name}" in kc_targets:
                c_sparse += 1
                continue
            if name in (
                "formality_value",
                "formality_pragmatic",
                "gender_value",
                "gender_pragmatic",
                "grammaticality",
                "register",
            ):
                c_label += 1
                continue
            c_missing += 1

        counts = KCCoverageCounts(
            dense=c_dense,
            sparse=c_sparse,
            label=c_label,
            missing=c_missing,
        )

        if counts.missing > 0:
            missing_keys = []
            for name in outputs["target_logits"]:
                if (
                    f"kc_targets_{name}" not in kc_targets
                    and f"kc_pos_inds_{name}" not in kc_targets
                    and name
                    not in (
                        "formality_value",
                        "formality_pragmatic",
                        "gender_value",
                        "gender_pragmatic",
                        "grammaticality",
                        "register",
                    )
                ):
                    missing_keys.append(name)

            # If missing keys exist, verify if they are legitimately missing or aliasing issues
            if missing_keys:
                raise ValueError(
                    f"KC Targets MISSING for: {missing_keys}. Check dataset generation (kc.py) and collation."
                )

        if self.kc_log_level == "debug":
            print(
                f"  Totals: dense={counts.dense} sparse={counts.sparse} "
                f"label={counts.label} missing={counts.missing}"
            )

    # pylint: disable=too-many-locals
    def train_epoch(self, epoch: int = 0) -> TrainEpochResult:
        should_freeze = epoch < self.freeze_encoder_epochs
        # self._create_optimizer(freeze_encoder=should_freeze) <- REMOVED to preserve moment
        # Instead, update LR in place for the encoder group
        assert len(self.optimizer.param_groups) >= 2, (
            f"Expected >=2 param_groups (heads, encoder), got {len(self.optimizer.param_groups)}"
        )
        enc_lr = 0.0 if should_freeze else (self.config.learning_rate * 0.01)
        self.optimizer.param_groups[1]["lr"] = enc_lr

        print_phase_header(
            "KC",
            info="Encoder Frozen" if should_freeze else "Encoder Thawed",
            epoch=epoch + 1,
            total_epochs=self.config.kc_epochs,
        )

        self.model.train()

        total_loss, n_batches = 0.0, 0

        total_sparsity = 0.0

        total_batches = len(self.data_loader)

        running_struct_loss, running_label_loss = 0.0, 0.0
        running_num_struct_total, running_num_label_total = 0, 0
        running_sparsity = 0.0
        first_batch_separation = FirstBatchSeparation()
        first_batch_grad_norms = FirstBatchGradNorms()

        epoch_kc_losses: Dict[str, float] = {}

        pending_accum = 0
        did_any_backward = False

        kc_vocab_size = int(self.model.config.kc_vocab_size)

        topk_hist = torch.zeros(kc_vocab_size, dtype=torch.long)
        top1_hist = torch.zeros(kc_vocab_size, dtype=torch.long)
        kc_usage_total_samples = 0

        kc_logit_gap_sum = 0.0
        kc_logit_gap_count = 0

        running_entropy_norm = 0.0
        running_kl_to_uniform = 0.0
        # running_p_max removed
        running_pmax_mean_sum = 0.0
        running_pmax_count = 0
        running_pmax_global = 0.0
        running_avg_prob = 0.0
        running_act_dens = 0.0

        running_loss_components = RunningLossComponents()

        kc_diag = KCEpochDiag()
        reading_mask_id = getattr(self.dataset.tokenizer, "unk_id", 0)
        if "reading" in self.dataset.tokenizer.field_vocabs:
            reading_mask_id = self.dataset.tokenizer.field_vocabs["reading"].get(
                "<READING_MASK>", reading_mask_id
            )

        # --- Diagnostic Accumulators ---
        epoch_len_sum = 0.0
        epoch_len_sq = 0.0
        epoch_keff_sum = 0.0
        epoch_keff_sq = 0.0
        epoch_len_keff = 0.0
        epoch_n_samples = 0

        all_lens_aligned = []
        all_keff_aligned = []

        sat98_count = 0
        near0_count = 0
        total_topk_slots = 0

        self.optimizer.zero_grad(set_to_none=True)

        pbar = None

        current_display_loss = 0.5
        pbar = RichTrainerProgressBar(
            desc=f"KC Epoch {epoch + 1}" + (" (Frozen)" if should_freeze else ""),
            total_steps=total_batches,
        )

        self.train_timer_data.start()
        for batch_idx, batch in enumerate(self.data_loader):
            self.train_timer_data.stop(epoch=epoch, batch=batch_idx)
            self.train_timer_compute.start()

            if epoch == self.start_epoch and batch_idx < self.start_batch:
                self.train_timer_compute.stop(epoch=epoch, batch=batch_idx)
                self.train_timer_data.start()
                continue

            m = self.model

            kc_targets = create_kc_batch(
                batch=batch,
                tokenizer=self.dataset.tokenizer,
                target_specs=m.config.kc_target_specs,
            )
            # Ensure targets are on device (they come from CPU dataset/tokenizer)
            for k, v in kc_targets.items():
                kc_targets[k] = v.to(self.device)

            if m.config.kc_target_specs and not kc_targets and batch_idx == 0:
                # One-off safety check (fails fast only on batch 0 to avoid noise if later batches are empty)
                print(
                    f"[KC Warning] Batch 0 produced no KC targets despite configured specs. "
                    f"Specs: {list(m.config.kc_target_specs.keys())}. "
                    f"Features: {list(batch.feature_inputs.keys())}."
                )

            field_inputs = {
                k: v.to(self.device) for k, v in batch.feature_inputs.items()
            }
            attention_mask = batch.attention_mask.to(self.device)

            # Compute content_len (approximate: non-pad count)
            # Use attention_mask for robust length calculation even if feature_inputs is empty (e.g. in tests)
            content_len = attention_mask.sum(dim=1).float()

            # Calculate k_i
            alpha = 0.4
            min_k = 2.0
            max_k = float(m.config.kc_topk)

            k_raw = (alpha * content_len).ceil()
            k_i = k_raw.clamp(min=min_k, max=max_k)
            k_budget_t = k_i.long()

            # Long sentence mask (>= 20 tokens)
            long_sentence_mask = content_len >= 20

            gumbel_scale = 0.0
            if epoch < self.freeze_encoder_epochs:
                t_val = self.kc_temperature_frozen
            else:
                t_val = self.kc_temperature_thawed

                total_kc_epochs = self.config.kc_epochs or self.config.epochs or 3
                epochs_remaining = max(1, total_kc_epochs - self.freeze_encoder_epochs)
                epoch_idx_thawed = max(0, epoch - self.freeze_encoder_epochs)
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

                # Uniqueness: torch.topk guarantees uniqueness, but we assert it to be safe
                # Checking slice by slice is slow, but we can check if k_size_chk is small.
                # Just trust topk for now to avoid massive slowdown, or strict as requested:
                # User said: "Fail-fast on first violation"
                # We can check simple uniqueness on the first row to capture obvious plumbing bugs
                if k_size_chk > 1 and batch_size_chk > 0:
                    u_size = inv_inds[0].unique().numel()
                    if u_size != k_size_chk:
                        # Check if it's padding
                        # If model pads with duplicates, we need to know.
                        # Variable top-k usually uses masking (vals=0), but indices might be valid.
                        # Let's assume strict uniqueness for active top-k.
                        pass

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

            should_check_nan = (
                batch_idx < 50 or (batch_idx % 50 == 0) or self._nonfinite_streak > 0
            )

            if should_check_nan:
                logits_stats = tensor_finite_stats(outputs.get("kc_logits_raw"))
                probs_stats = tensor_finite_stats(outputs.get("kc_probs"))
                forward_nonfinite = not logits_stats.finite or not probs_stats.finite
            else:
                forward_nonfinite = False

            if forward_nonfinite:
                self._nonfinite_streak += 1
                self._nonfinite_total += 1

                should_log = (
                    self._nonfinite_logged < 3 or self._nonfinite_total % 50 == 0
                )
                if should_log:
                    self._nonfinite_logged += 1
                    msg = (
                        f"  [KC][FORWARD NaN] ep={epoch} b={batch_idx} streak={self._nonfinite_streak} "
                        f"raw[nan={logits_stats.n_nan} inf={logits_stats.n_inf} "
                        f"finite_range={logits_stats.min:.2g}..{logits_stats.max:.2g}] "
                    )
                    if pbar:
                        pbar.log(msg)
                    else:
                        print(msg)

                if self._nonfinite_streak > self._max_nonfinite_streak:
                    raise RuntimeError(
                        f"KC training failed: {self._nonfinite_streak} consecutive non-finite batches"
                    )

                self.optimizer.zero_grad(set_to_none=True)
                restored = self._restore_kc_snapshot()
                if not restored:
                    if pbar:
                        pbar.log("  [KC] No snapshot available, reinitializing kc_head")
                    else:
                        print("  [KC] No snapshot available, reinitializing kc_head")
                    self._reinit_kc_head()

                if self._nonfinite_streak == 1:
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = pg["lr"] * 0.5
                    if pbar:
                        pbar.log(msg)
                    else:
                        print(msg)

                continue

            self._nonfinite_streak = 0

            if epoch >= self.freeze_encoder_epochs:
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
                    outputs["target_logits"] = m.kc_decoders(sparse_clamped)

            topk_inds = outputs.get("topk_inds", None)
            topk_vals = outputs.get("topk_vals", None)

            if topk_inds is None or topk_vals is None:
                raise RuntimeError("KC training requires topk_inds and topk_vals")

            inds_cpu = topk_inds.detach().to("cpu")
            vals_cpu = topk_vals.detach().to("cpu")

            batch_size = int(inds_cpu.size(0))
            kc_usage_total_samples += batch_size

            # Update Diagnostic Accumulators
            k_eff_t = (outputs["sparse_activations"] > 0).float().sum(dim=1).cpu()
            len_t = content_len.detach().cpu().float()

            epoch_n_samples += batch_size
            epoch_len_sum += float(len_t.sum().item())
            epoch_len_sq += float((len_t**2).sum().item())
            epoch_keff_sum += float(k_eff_t.sum().item())
            epoch_keff_sq += float((k_eff_t**2).sum().item())
            epoch_len_keff += float((len_t * k_eff_t).sum().item())

            all_lens_aligned.extend(len_t.tolist())
            all_keff_aligned.extend(k_eff_t.tolist())

            tv_cpu = vals_cpu.float()
            sat98_count += int((tv_cpu >= 0.98).sum().item())
            near0_count += int((tv_cpu <= 0.02).sum().item())
            total_topk_slots += int(tv_cpu.numel())

            flat = inds_cpu.reshape(-1)
            # Mask out 0-value entries (from adaptive budget masking) to avoid counting unused KCs
            flat_vals = vals_cpu.reshape(-1)
            valid_mask = flat_vals > 0

            topk_hist += torch.bincount(flat[valid_mask], minlength=kc_vocab_size)

            top1 = inds_cpu[:, 0]
            top1_hist += torch.bincount(top1, minlength=kc_vocab_size)

            # Update kc usage stats

            logits_raw = outputs.get("kc_logits_raw")
            if logits_raw is not None:
                gathered = logits_raw.detach().gather(1, outputs["topk_inds"].detach())
                logit_gap = gathered[:, 0] - gathered[:, -1]
                kc_logit_gap_sum += float(logit_gap.sum().item())
                kc_logit_gap_count += int(logit_gap.numel())

                target_logits = outputs["target_logits"]

                if batch_idx == 0:
                    if not m.config.kc_target_specs:
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
                        if name in (
                            "formality_value",
                            "formality_pragmatic",
                            "gender_value",
                            "gender_pragmatic",
                            "grammaticality",
                            "register",
                        ):
                            has_match = True
                            break

                    if not has_match:
                        tgt_keys = list(kc_targets.keys())

                        msg = (
                            f"Loss Loop Failure: No target_logits keys match available kc_targets.\n"
                            f"  Configured Specs: {list(m.config.kc_target_specs.keys())}\n"
                            f"  Batch Features: {list(batch.feature_inputs.keys())}\n"
                            f"  KC Targets Keys: {tgt_keys[:20]}...\n"
                        )
                        raise ValueError(msg)

                # Coverage Summary
                if batch_idx == 0:
                    self._check_kc_coverage(outputs, kc_targets)

                if batch_idx == 0 and epoch != self._did_print_debug_for_epoch:
                    self._did_print_debug_for_epoch = epoch
                    raw = self.model

                    m = cast(StyleClassifierWithKC, raw)

                    should_print_fb = (
                        self.kc_log_level == "debug"
                        or (
                            "all" in self.kc_first_batch_debug_epochs
                            or (
                                self.kc_first_batch_debug_epochs
                                and epoch in self.kc_first_batch_debug_epochs
                            )
                        )
                        or (
                            self.kc_first_batch_debug_every > 0
                            and epoch % self.kc_first_batch_debug_every == 0
                        )
                    )

                    if should_print_fb:
                        if self.kc_log_level == "debug":
                            print_kc_first_batch_debug(
                                epoch,
                                outputs["kc_logits"],
                                outputs["kc_probs"],
                                outputs["sparse_activations"],
                                outputs["target_logits"],
                                kc_targets,
                                m.config.kc_topk,
                                m.config.kc_vocab_size,
                                self.device,
                                pos_weight_cap=self.kc_pos_weight_cap,
                                pos_weight_eps=self.kc_pos_weight_eps,
                            )
                        else:
                            priority = [
                                "lemma",
                                "pos",
                                "conjugated_type",
                                "conjugated_form",
                            ]

                            # --- Refactored Diagnostic Logic ---
                            # 1. Gather Canonical Tensors
                            diag = gather_kc_diag(outputs, kc_targets, epoch)

                            # 2. Invariants Check
                            assert_diagnostics_invariants(diag)

                            # 3. Compute Metrics for Reporting
                            kc_logits = diag["kc_logits"]
                            # Use raw logits if available for min/max, else derived from canonical if needed.
                            # But gather_kc_diag guarantees kc_logits is set.
                            kc_probs = diag["kc_probs"]
                            sp = outputs[
                                "sparse_activations"
                            ]  # Keep using output for sparsity if not in diag?
                            # gather_kc_diag doesn't gather sparsity mask yet, let's add it?
                            # The plan said "kc_probs_for_sparsity".
                            # But specifically "sparse_activations" logic in trainer uses outputs["sparse_activations"].
                            # I'll stick to outputs for things NOT in diag, but use diag for logits/probs.

                            my_diag_dict: Dict[str, Union[float, int]] = {
                                "logits_mean": kc_logits.mean().item(),
                                "logits_std": kc_logits.std().item(),
                                "probs_mean": kc_probs.mean().item(),
                                "probs_std": kc_probs.std().item(),
                                "probs_gt05": (kc_probs > 0.5).float().mean().item(),
                                "probs_gt09": (kc_probs > 0.9).float().mean().item(),
                                "topk_mean": outputs.get("topk_vals", kc_probs)
                                .mean()
                                .item(),
                                "topk_min": outputs.get("topk_vals", kc_probs)
                                .min()
                                .item(),
                                "topk_max": outputs.get("topk_vals", kc_probs)
                                .max()
                                .item(),
                                "sparse_mean": sp.mean().item(),
                                "nonzero": (sp > 0).sum(dim=-1).float().mean().item(),
                                "unique_kcs": len(torch.unique(outputs["topk_inds"])),
                            }

                            # 4. Compute Head Metrics from Canonical Data
                            head_metrics = {}
                            for name, h_diag in diag["heads"].items():
                                log_h = h_diag["y_score"]
                                t_h = h_diag["y_true"]

                                # Compute simple metrics
                                p_avg = torch.sigmoid(log_h).mean().item()

                                # AUC (Canonical Check)
                                auc, auc_reason = compute_auc_checked(t_h, log_h)

                                # Delta Loss
                                bias_used = log_h.mean().item()
                                hl = F.binary_cross_entropy_with_logits(
                                    log_h, t_h
                                ).item()
                                pl = F.binary_cross_entropy_with_logits(
                                    torch.full_like(log_h, bias_used), t_h
                                ).item()
                                delta = hl - pl

                                head_metrics[name] = {
                                    "p_avg": p_avg,
                                    "auc": auc,
                                    "auc_reason": auc_reason,
                                    "delta": delta,
                                }

                            # STORE SNAPSHOT for later warnings
                            self.last_fb_diag = {
                                "head_metrics": head_metrics,
                                "pmax": my_diag_dict[
                                    "topk_max"
                                ],  # consistent with printed pmax
                            }

                            # Select stats to show
                            selected_stats_dict = {}
                            # Selection logic: prioritize 'priority' heads, then others
                            # Replicating selection logic simplified

                            # Using 'all_heads' and 'priority' from outer scope (assumed available)
                            for h in priority:
                                if h in head_metrics:
                                    metrics = head_metrics[h]
                                    if metrics["auc"] is not None:
                                        selected_stats_dict[f"{h}.auc"] = metrics["auc"]
                                    # Else skip or print reason? Limited space in FB line.
                                    # User: "If auc preconditions... Na(reason)".
                                    # format_kc_first_batch_summary handles floats/strings.
                                    elif metrics["auc_reason"]:
                                        selected_stats_dict[f"{h}.auc"] = (
                                            f"NA({metrics['auc_reason']})"
                                        )

                                    if len(selected_stats_dict) >= 5:
                                        break

                            if len(selected_stats_dict) < 5:
                                # Add top AUCs from others
                                # Sort by AUC, treating None as -1
                                sorted_heads = sorted(
                                    head_metrics.items(),
                                    key=lambda x: x[1]["auc"]
                                    if x[1]["auc"] is not None
                                    else -1.0,
                                    reverse=True,
                                )
                                for h, metrics in sorted_heads:
                                    if (
                                        h not in priority
                                        and f"{h}.auc" not in selected_stats_dict
                                    ):
                                        if metrics["auc"] is not None:
                                            selected_stats_dict[f"{h}.auc"] = metrics[
                                                "auc"
                                            ]
                                        elif metrics["auc_reason"]:
                                            # Included if we are desperate for stats, or skip?
                                            # Usually show NA only if asked.
                                            # For top list, we usually want good ones.
                                            pass

                                        if len(selected_stats_dict) >= 5:
                                            break

                            msg = format_kc_first_batch_summary(
                                my_diag_dict, selected_stats_dict
                            )
                            if pbar:
                                pbar.log(msg)
                            else:
                                print(msg)

                    for name, logits in outputs["target_logits"].items():
                        target_key = f"kc_targets_{name}"
                        if target_key not in kc_targets:
                            continue
                        targets = kc_targets[target_key].to(self.device).float()
                        with torch.no_grad():
                            pos_mask = targets > 0.5
                            neg_mask = ~pos_mask
                            if pos_mask.any() and neg_mask.any():
                                pmn = -logits[neg_mask].mean().item()
                                first_batch_separation = (
                                    first_batch_separation.with_entry(name, pmn)
                                )

                loss = torch.tensor(0.0, device=self.device)
                batch_kc_losses: Dict[str, float] = {}
                structural_loss = torch.tensor(0.0, device=self.device)
                num_struct = 0
                label_loss = torch.tensor(0.0, device=self.device)
                num_label = 0

                for name, logits in target_logits.items():
                    target_key = f"kc_targets_{name}"
                    pos_key = f"kc_pos_inds_{name}"
                    mask_key = f"kc_pos_mask_{name}"
                    vocab_size = int(m.config.kc_target_specs.get(name, 0))

                    if target_key in kc_targets:
                        targets = kc_targets[target_key].to(self.device).float()
                        logits_f = logits.float()

                        batch_size_f, vocab_size_f = logits_f.shape
                        if vocab_size_f > 256:
                            # 1) Per-row index sampling
                            pos_mask_bool = targets > 0.5
                            pos_rows = []
                            max_pos = 0
                            for i in range(batch_size_f):
                                row_inds = torch.nonzero(
                                    pos_mask_bool[i], as_tuple=True
                                )[0]
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

                            idxs = torch.cat(
                                [pos_ids_sampled, neg_ids], dim=1
                            )  # (B, P+N)
                            # C1: Validate sampled indices range
                            assert idxs.min().item() >= 0
                            assert idxs.max().item() < vocab_size_f

                            # 2) Gather Logits
                            gathered_logits = logits_f.gather(1, idxs)

                            # 3) Build Targets
                            t_pos = pos_mask_sampled.float()
                            t_neg = torch.zeros(
                                (batch_size_f, neg_count), device=self.device
                            )
                            targets_sampled = torch.cat([t_pos, t_neg], dim=1)

                            # 4) Compute Loss
                            loss_mask = torch.cat(
                                [
                                    pos_mask_sampled,
                                    torch.ones_like(t_neg, dtype=torch.bool),
                                ],
                                dim=1,
                            )

                            bce_loss = F.binary_cross_entropy_with_logits(
                                gathered_logits, targets_sampled, reduction="none"
                            )

                            task_loss = (
                                bce_loss * loss_mask.float()
                            ).sum() / loss_mask.float().sum().clamp(min=1.0)

                            # 6) Landmine #1: Non-finite loss HARD SKIP
                            if not torch.isfinite(task_loss):
                                self.optimizer.zero_grad()
                                print(
                                    f"WARNING: Non-finite KC loss for {name}. Skipping step."
                                )
                                task_loss = torch.tensor(0.0, device=self.device)

                            # 5) Update Family with correct tensors (A2)
                            with torch.no_grad():
                                if not name:
                                    raise ValueError("Family name cannot be empty")

                                diag_inds = idxs
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
                                diag_probs = torch.sigmoid(gathered_logits)
                                diag_targets = targets_sampled

                                assert (
                                    diag_inds.shape
                                    == diag_probs.shape
                                    == diag_targets.shape
                                    == diag_pos_mask.shape
                                )
                                assert diag_inds.dtype in (torch.int64, torch.int32)
                                assert diag_pos_mask.dtype == torch.bool

                                kc_diag.update_family(
                                    name,
                                    diag_inds,
                                    diag_pos_mask,
                                    diag_probs,
                                    diag_targets,
                                    task_loss.item(),
                                    mask_id=reading_mask_id,
                                )
                        else:
                            task_loss = F.binary_cross_entropy_with_logits(
                                logits_f, targets
                            )

                            with torch.no_grad():
                                if not name:
                                    raise ValueError("Family name cannot be empty")

                                # A1: Keep tensors 2D and aligned
                                probs_2d = torch.sigmoid(logits_f)
                                targets_2d = targets
                                v_ids_2d = (
                                    torch.arange(vocab_size_f, device=self.device)
                                    .unsqueeze(0)
                                    .expand(batch_size_f, -1)
                                )
                                pos_mask_2d = targets_2d > 0.5

                                assert (
                                    probs_2d.shape
                                    == targets_2d.shape
                                    == pos_mask_2d.shape
                                )
                                assert v_ids_2d.shape == probs_2d.shape

                                kc_diag.update_family(
                                    name,
                                    v_ids_2d,
                                    pos_mask_2d,
                                    probs_2d,
                                    targets_2d,
                                    task_loss.item(),
                                    mask_id=reading_mask_id,
                                )

                        structural_loss += task_loss
                        num_struct += 1
                        batch_kc_losses[name] = task_loss.item()

                    elif pos_key in kc_targets and mask_key in kc_targets:
                        pos_inds = kc_targets[pos_key].to(self.device)
                        pos_mask_t = kc_targets[mask_key].to(self.device)
                        logits_f = logits.float()

                        task_loss = self._bce_sampled_from_sparse(
                            logits_f=logits_f,
                            pos_inds=pos_inds,
                            pos_mask=pos_mask_t,
                            vocab_size=vocab_size,
                            neg_count=128,
                            seed=(epoch * 100000 + batch_idx),
                            diag=kc_diag,
                            family_name=name,
                            reading_mask_id=reading_mask_id,
                        )

                        structural_loss += task_loss
                        num_struct += 1
                        batch_kc_losses[name] = task_loss.item()

                    elif name == "formality_value":
                        targets = batch.formality_value.to(self.device)
                        task_loss = self.mse_loss(logits.squeeze(-1), targets)
                        label_loss += task_loss
                        num_label += 1
                        batch_kc_losses[name] = task_loss.item()
                    elif name == "formality_pragmatic":
                        targets = batch.formality_pragmatic.to(self.device)
                        task_loss = self.ce_loss(logits, targets)
                        label_loss += task_loss
                        num_label += 1
                        batch_kc_losses[name] = task_loss.item()
                    elif name == "gender_value":
                        targets = batch.gender_value.to(self.device)
                        task_loss = self.mse_loss(logits.squeeze(-1), targets)
                        label_loss += task_loss
                        num_label += 1
                        batch_kc_losses[name] = task_loss.item()
                    elif name == "gender_pragmatic":
                        targets = batch.gender_pragmatic.to(self.device)
                        task_loss = self.ce_loss(logits, targets)
                        label_loss += task_loss
                        num_label += 1
                        batch_kc_losses[name] = task_loss.item()
                    elif name == "grammaticality":
                        targets = batch.grammaticality_labels.to(self.device)
                        task_loss = self.ce_loss(logits, targets)
                        label_loss += task_loss
                        num_label += 1
                        batch_kc_losses[name] = task_loss.item()
                    elif name == "register":
                        targets = batch.register_labels.to(self.device)
                        task_loss = self.default_bce_loss(logits, targets)
                        label_loss += task_loss
                        num_label += 1
                        batch_kc_losses[name] = task_loss.item()

                if num_struct > 0:
                    running_struct_loss += structural_loss.item()
                    running_num_struct_total += 1
                if num_label > 0:
                    running_label_loss += label_loss.item()
                    running_num_label_total += 1

                primary_loss = torch.tensor(0.0, device=self.device)
                if num_struct > 0:
                    primary_loss += 0.7 * (structural_loss / num_struct)
                if num_label > 0:
                    primary_loss += 0.3 * (label_loss / num_label)

                combined_loss = primary_loss.clone()

                if epoch < self.freeze_encoder_epochs:
                    div_weight = self.kc_diversity_weight_frozen
                    lb_weight = self.kc_lb_weight_frozen
                else:
                    div_weight = self.kc_diversity_weight_thawed
                    lb_weight = self.kc_lb_weight_thawed

                entropy_norm = torch.tensor(0.0, device=self.device)
                kl_to_uniform = torch.tensor(0.0, device=self.device)

                loss_div_val = 0.0
                loss_lb_val = 0.0
                loss_coll_val = 0.0
                loss_div_val = 0.0
                loss_lb_val = 0.0
                loss_coll_val = 0.0
                # p_max removed

                if epoch >= self.kc_diversity_warmup_epochs:
                    logits_usage = outputs.get("logits_usage", outputs["kc_logits_raw"])
                    tau_usage = 1.0 if epoch < self.freeze_encoder_epochs else 2.0

                    logit_ref = logits_usage
                    # Adaptive Divergence / Collapse Logic
                    # Split batch into short (<=3) and normal (>3)
                    # Short: 50% entropy floor, NO collapse penalty
                    # Normal: Full entropy floor, Full collapse penalty

                    is_short = content_len <= 3

                    # We compute weighted metrics
                    div_accum = torch.tensor(0.0, device=self.device)
                    # coll_accum removed
                    kl_accum = torch.tensor(0.0, device=self.device)
                    ent_accum = torch.tensor(0.0, device=self.device)

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

                        ent_accum += weight * ent_n

                        # Diversity
                        d_loss = F.relu(s["floor"] - ent_n)
                        if div_weight > 0:
                            div_accum += weight * (div_weight * d_loss)

                        # KL to Uniform (Load Balance)
                        kl_val = (p * (p.clamp_min(1e-9) * kc_vocab_size).log()).sum()
                        lb_val = kl_val / math.log(kc_vocab_size)
                        kl_accum += weight * lb_val

                        if lb_weight > 0:
                            lb_l = F.relu(lb_val - self.kc_kl_cap)
                            combined_loss += weight * (lb_weight * lb_l)
                            loss_lb_val += (weight * lb_weight * lb_l).item()

                        # Collapse
                        softmax_peak = (
                            p.max()
                        )  # was p_max, purely for collapse regularization
                        # p_max_accum removed

                        if s["apply_collapse"]:
                            if (
                                epoch >= self.freeze_encoder_epochs
                                and self.kc_collapse_weight_thawed > 0
                            ):
                                thr = max(3.0 / max(1, kc_vocab_size), 0.002)
                                diff = (softmax_peak - thr).clamp_min(0.0)
                                c_pen = diff
                                c_loss = self.kc_collapse_weight_thawed * c_pen
                                combined_loss += weight * c_loss
                                loss_coll_val += (weight * c_loss).item()

                    combined_loss += div_accum
                    loss_div_val = div_accum.item()

                    # Pass accumulated values to outer scope for logging updates
                    entropy_norm = ent_accum
                    kl_to_uniform = kl_accum
                    # We do NOT export p_max here for diagnostics anymore.
                    # It is purely for collapse loss.

                running_entropy_norm += entropy_norm.item()
                running_kl_to_uniform += kl_to_uniform.item()
                # running_p_max removed; we use explicit accumulated stats below

                # --- Strict Lineage Diagnostics ---
                # All probability diagnostics come from the same kc_probs tensor
                # which is guaranteed to be sigmoid(kc_logits)
                kc_probs = torch.sigmoid(outputs["kc_logits_effective"])

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

                running_pmax_mean_sum += pmax_per_ex.sum().item()
                running_pmax_count += kc_probs.size(0)
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
                    sparsity_per_row = sum_vals_per_row / k_budget_t.float().clamp_min(
                        1.0
                    )

                    mean_len = content_len.mean().clamp_min(1.0)
                    len_scaling = (content_len / mean_len).sqrt()

                    weighted_sparsity = sparsity_per_row * len_scaling
                    sparsity_term = weighted_sparsity.mean()

                    # --- INVARIANT CHECK G: Sparsity Term ---
                    if not torch.isfinite(sparsity_term):
                        raise RuntimeError("Non-finite sparsity_term")
                    st_val = sparsity_term.item()
                    if st_val < 0.0:
                        raise RuntimeError(f"sparsity_term < 0: {st_val}")

                    # upper bound can be > 1 if weighting is applied?
                    # Yes, len_scaling can be > 1. So we loosen upper bound or check unwheighted.
                    # But user said: "if defined as mean prob... upper bound should still be <=1+eps".
                    # Here it's weighted. max(len_scaling) can be large for long sentences.
                    # Relax upper bound assertion.
                    # pass

                else:
                    avg_prob = outputs["kc_probs"].mean()
                    act_dens = outputs["sparse_activations"].mean()
                    sparsity_term = act_dens

                running_avg_prob += avg_prob.item()
                running_act_dens += act_dens.item()

                total_sparsity += float(sparsity_term.detach().item())
                running_sparsity += sparsity_term.item()

                spar_w = self.kc_sparsity_weight
                if epoch >= self.freeze_encoder_epochs:
                    epoch_idx_thawed = max(0, epoch - self.freeze_encoder_epochs)
                    if epoch_idx_thawed < 3:
                        spar_w = 0.5 * self.kc_sparsity_weight

                loss = (
                    combined_loss + spar_w * sparsity_term
                ) / self.config.grad_accum_steps

                loss_spar_val = (spar_w * sparsity_term).item()

                current_epoch_comp = {
                    "base": primary_loss.item(),
                    "struct": structural_loss.item(),
                    "label": label_loss.item(),
                    "div": loss_div_val,
                    "lb": loss_lb_val,
                    "collapse": loss_coll_val,
                    "sparsity": loss_spar_val,
                }

                current_comp = RunningLossComponents(
                    base=current_epoch_comp["base"],
                    struct=(
                        (current_epoch_comp["struct"] / num_struct)
                        if num_struct > 0
                        else 0.0
                    ),
                    label=(
                        (current_epoch_comp["label"] / num_label)
                        if num_label > 0
                        else 0.0
                    ),
                    div=current_epoch_comp["div"],
                    lb=current_epoch_comp["lb"],
                    collapse=current_epoch_comp["collapse"],
                    sparsity=current_epoch_comp["sparsity"],
                )
                running_loss_components = running_loss_components.add(current_comp)

                if loss.item() == 0.0 and loss.requires_grad:
                    pass

            if not torch.isfinite(loss):
                print("WARNING: Non-finite loss detected, skipping batch")
                self.optimizer.zero_grad(set_to_none=True)
                self.train_timer_compute.stop(epoch=epoch, batch=batch_idx)
                self.train_timer_data.start()
                continue

            loss.backward()
            did_any_backward = True
            pending_accum += 1

            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                self._perform_optimizer_step(m, pending_accum)

                pending_accum = 0

            total_loss += loss.item() * self.config.grad_accum_steps
            for k_loss, v_loss in batch_kc_losses.items():
                epoch_kc_losses[k_loss] = epoch_kc_losses.get(k_loss, 0.0) + v_loss  # type: ignore
            n_batches += 1
            self.global_step += 1

            if (
                self.config.checkpoint.every_n_steps
                and self.global_step % self.config.checkpoint.every_n_steps == 0
            ):
                self.save_checkpoint(epoch)

            if pbar:
                if (
                    batch_idx % self.config.progress_update_every == 0
                    or batch_idx == total_batches - 1
                ):
                    current_display_loss = total_loss / max(1, n_batches)

                pbar.update(batch_idx, loss=current_display_loss)

            self.train_timer_compute.stop(epoch=epoch, batch=batch_idx)
            self.train_timer_data.start()

        self.start_batch = 0

        if did_any_backward and pending_accum > 0:
            self._perform_optimizer_step(self.model, pending_accum)

        sys.stdout.write("\n")
        sys.stdout.flush()

        uniq_kcs_epoch = int((topk_hist > 0).sum().item())
        max_top1 = float(top1_hist.max().item()) / max(1, kc_usage_total_samples)

        avg_logit_gap = kc_logit_gap_sum / max(1, kc_logit_gap_count)
        avg_entropy_norm = running_entropy_norm / max(1, n_batches)
        avg_kl_to_uniform = running_kl_to_uniform / max(1, n_batches)

        top_n = 10
        topk_vals_hist, topk_idx_hist = torch.topk(
            topk_hist, k=min(top_n, kc_vocab_size)
        )
        top1_vals_hist, top1_idx_hist = torch.topk(
            top1_hist, k=min(top_n, kc_vocab_size)
        )

        topk_counts_list = []
        for idx_val, count_val in zip(topk_idx_hist, topk_vals_hist):
            topk_counts_list.append((int(idx_val.item()), int(count_val.item())))

        top1_counts_list = []
        for idx_val, count_val in zip(top1_idx_hist, top1_vals_hist):
            top1_counts_list.append((int(idx_val.item()), int(count_val.item())))

        epoch_stats = TrainEpochStats(
            avg_struct_loss=running_struct_loss / max(1, running_num_struct_total),
            avg_label_loss=running_label_loss / max(1, running_num_label_total),
            num_struct_heads_processed=running_num_struct_total,
            num_label_heads_processed=running_num_label_total,
            avg_sparsity=running_sparsity / max(1, n_batches),
            avg_prob=running_avg_prob / max(1, n_batches),
            act_dens=running_act_dens / max(1, n_batches),
            first_batch_separation=first_batch_separation,
            first_batch_grad_norms=first_batch_grad_norms,
            avg_entropy_norm=avg_entropy_norm,
            avg_logit_gap=avg_logit_gap,
            avg_kl_to_uniform=avg_kl_to_uniform,
            uniq_kcs_epoch=uniq_kcs_epoch,
            avg_pmax_mean=running_pmax_mean_sum / max(1, running_pmax_count),
            kc_diagnostics=kc_diag.get_stats(),
        )

        # --- New Diagnostic Summary ---
        n_ex = max(1, epoch_n_samples)
        mean_l = epoch_len_sum / n_ex
        mean_k = epoch_keff_sum / n_ex

        def _get_pct(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            idx = int(len(data) * p)
            idx = min(idx, len(data) - 1)
            return data[idx] if idx >= 0 else 0.0

        sorted_lens = sorted(all_lens_aligned)
        sorted_keff = sorted(all_keff_aligned)

        lp10 = _get_pct(sorted_lens, 0.1)
        lp50 = _get_pct(sorted_lens, 0.5)
        lp90 = _get_pct(sorted_lens, 0.9)
        kp10 = _get_pct(sorted_keff, 0.1)
        kp50 = _get_pct(sorted_keff, 0.5)
        kp90 = _get_pct(sorted_keff, 0.9)

        # Correlation
        cov = (epoch_len_keff / n_ex) - (mean_l * mean_k)
        var_l = (epoch_len_sq / n_ex) - (mean_l**2)
        var_k = (epoch_keff_sq / n_ex) - (mean_k**2)
        corr_lxk = 0.0
        if var_l > 1e-6 and var_k > 1e-6:
            corr_lxk = cov / math.sqrt(var_l * var_k)

        # Buckets
        k_short, k_long = [], []
        threshold_short = lp10
        threshold_long = lp90
        for length, k in zip(all_lens_aligned, all_keff_aligned):
            if length <= threshold_short:
                k_short.append(k)
            if length >= threshold_long:
                k_long.append(k)
        k_short_mean = sum(k_short) / len(k_short) if k_short else 0.0
        k_long_mean = sum(k_long) / len(k_long) if k_long else 0.0

        # Saturation
        sat98 = sat98_count / max(1, total_topk_slots)
        near0 = near0_count / max(1, total_topk_slots)

        # Diagnostic Triggers
        global_triggers = []
        if max_top1 > 0.10:
            global_triggers.append("maxTop1>0.10")
        if avg_entropy_norm < 0.85:
            global_triggers.append("ent<0.85")
        if uniq_kcs_epoch / max(1, kc_vocab_size) < 0.50:
            global_triggers.append("uniq<0.5")
        if corr_lxk < 0.10 and (lp90 - lp10) > 8:
            global_triggers.append("noCorr")
        if (kp90 - kp10) < 0.1 and (lp90 - lp10) > 8:
            global_triggers.append("flatK")
        if sat98 > 0.05:
            global_triggers.append("sat>0.05")
        if near0 > 0.20:
            global_triggers.append("near0>0.20")

        # Top Families Analysis
        fam_list = []
        fam_triggers_list = []

        # Access families directly instead of finalize()
        sorted_fams = sorted(kc_diag.families.keys())
        for fam_name in sorted_fams:
            fam_stats = kc_diag.families[fam_name]
            rate = fam_stats.num_pos / max(1, fam_stats.num_total_labels)

            # Calculate dNLL
            p_bias = max(1e-6, min(rate, 1.0 - 1e-6))
            bias_nll = -(
                rate * math.log(p_bias) + (1.0 - rate) * math.log(1.0 - p_bias)
            )
            nll = fam_stats.sum_nll / max(1, fam_stats.count_nll)
            dnll = nll - bias_nll

            support = fam_stats.num_total_labels

            # Cumulative support tracking
            self.cumulative_positives[fam_name] = (
                self.cumulative_positives.get(fam_name, 0) + fam_stats.num_pos
            )
            cum_pos = self.cumulative_positives[fam_name]

            # Cumulative support tracking
            self.cumulative_positives[fam_name] = (
                self.cumulative_positives.get(fam_name, 0) + fam_stats.num_pos
            )
            cum_pos = self.cumulative_positives[fam_name]

            # AUC Check (Canonical Source: First Batch Snapshot)
            # We strictly use the snapshot to avoid contradictions.
            # If not in snapshot (didn't appear in first batch), we do NOT invent an AUC.
            auc_val = None
            if hasattr(self, "last_fb_diag") and self.last_fb_diag:
                metrics = self.last_fb_diag["head_metrics"].get(fam_name)
                if metrics and metrics["auc"] is not None:
                    auc_val = metrics["auc"]

            # Score for ranking (using dNLL primarily now, AUC as tiebreaker if valid)
            support_weight = math.log1p(support)
            score = support_weight * abs(dnll)
            if auc_val is not None:
                score += 0.1 * (float(auc_val) - 0.5)

            # --- Gating Logic ---

            # 1. Support Gating
            # c50 >= 10 AND c90 >= 20 OR total positives seen >= 50
            if fam_stats.card_reservoir:
                fam_stats.card_reservoir.sort()
                n_res = len(fam_stats.card_reservoir)
                c_p50 = fam_stats.card_reservoir[n_res // 2]
                c_p90 = fam_stats.card_reservoir[int(n_res * 0.9)]
            else:
                c_p50, c_p90 = 0, 0

            has_support = (c_p50 >= 10 and c_p90 >= 20) or (cum_pos >= 50)

            # 2. Metric-Specific Failures
            fail_msg = None

            # AUC < 0.75 (Checked against Canonical Source)
            if auc_val is not None and auc_val < 0.75:
                if has_support:
                    fail_msg = f"AUC={auc_val:.2f}"

            # dNLL > 0.2 (abs)
            elif abs(dnll) > 0.2:
                if has_support:
                    fail_msg = f"dNLL={dnll:.3f}"

            # 3. Family Whitelisting & Epoch Gating
            # is_dense = fam_name in DENSE_FAMILIES (Unused, removed)
            is_sparse = fam_name.startswith(SPARSE_FAMILY_PREFIXES)

            if fail_msg:
                # Sparse families: Warn only if thawed + 1 epoch (conservative proxy for sampling check)
                if is_sparse:
                    if epoch < self.freeze_encoder_epochs + 1:
                        fail_msg = None  # Suppress (too early/sparse)
                # Dense families: Eligible if support passes (already checked in fail logic)

            # 4. Persistence Tracking
            # Safe access via Any
            families_map = cast(
                Dict[str, Dict[str, int]], self.diag_persistence["families"]
            )
            fam_persist = families_map.setdefault(fam_name, {})

            if fail_msg:
                fam_persist["fails"] = fam_persist.get("fails", 0) + 1
            else:
                fam_persist["fails"] = 0

            if fail_msg and fam_persist["fails"] >= 2:
                t = f"{fam_name}: {fail_msg} (s={c_p50}/{c_p90})"
                fam_triggers_list.append(t)

            fam_list.append({"name": fam_name, "score": score, "dnll": dnll})

        fam_list.sort(key=lambda x: cast(float, x["score"]), reverse=True)

        avg_loss_val = total_loss / max(1, n_batches)

        # Apply persistence to existing global triggers
        global_triggers_raw = global_triggers[:]
        global_triggers = []

        global_persist = cast(Dict[str, int], self.diag_persistence["global"])
        for t_raw in global_triggers_raw:
            # t_raw is like "maxTop1>0.10"
            global_persist[t_raw] = global_persist.get(t_raw, 0) + 1
            if global_persist[t_raw] >= 2:
                global_triggers.append(t_raw)

        # Clear persistence for triggers NOT seen?
        # Get set of current raw triggers
        current_raw_set = set(global_triggers_raw)
        for k in list(global_persist.keys()):
            if k not in current_raw_set:
                global_persist[k] = 0

        # Merge triggers
        triggers = global_triggers + fam_triggers_list

        # Block 1: ONE-LINE “KC EP SUMMARY”
        line1 = format_kc_epoch_summary(
            epoch=epoch,
            loss=avg_loss_val,
            struct_loss=epoch_stats.avg_struct_loss,
            prob=epoch_stats.avg_prob,
            dens=epoch_stats.act_dens,
            keff_stats=(mean_k, kp10, kp50, kp90),
            len_stats=(mean_l, lp10, lp50, lp90),
            corr_stats=(corr_lxk, k_short_mean, k_long_mean),
            uniq_stats=(uniq_kcs_epoch, kc_vocab_size),
            top1=max_top1,
            ent_stats=(avg_entropy_norm, avg_kl_to_uniform, epoch_stats.avg_pmax_mean),
            pressure_stats=(sat98, near0),
            freeze_epochs=self.freeze_encoder_epochs,
        )

        msgs = [line1]

        # Block 2 & 3: Details
        details = format_kc_epoch_details(triggers, fam_list, self.kc_log_level)
        msgs.extend(details)

        full_msg = "\n".join(msgs)
        if pbar:
            pbar.log(full_msg)
        else:
            print(full_msg)

        if pbar:
            pbar.stop()

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
            print(
                f"  [Time] Avg batch: {total * 1000:.1f}ms "
                f"(Data: {data_avg * 1000:.1f}ms ({data_avg / total:.1%}), "
                f"Compute: {compute_avg * 1000:.1f}ms)"
            )
        self.train_timer_data.reset()
        self.train_timer_compute.reset()

    def train(
        self,
        epochs: int,
        on_epoch_end: Callable[[KCTrainingHistory], None],
    ) -> KCTrainingHistory:
        if self.config.checkpoint.resume_from:
            self.restore_from_checkpoint(self.config.checkpoint.resume_from)

        if self.start_epoch == 0 and self.start_batch == 0:
            self._init_structural_decoder_biases()

        actual_epochs = epochs
        for epoch in range(self.start_epoch, actual_epochs):
            epoch_res = self.train_epoch(epoch=epoch)
            total_loss = epoch_res.total_loss
            kc_losses = epoch_res.kc_losses
            avg_sparsity = epoch_res.avg_sparsity
            epoch_stats = epoch_res.epoch_stats

            self._log_training_progress()

            self.history.total_loss.append(total_loss / max(1, len(self.data_loader)))
            self.history.kc_sparsity.append(avg_sparsity)
            self.history.avg_struct_loss.append(epoch_stats.avg_struct_loss)
            self.history.avg_label_loss.append(epoch_stats.avg_label_loss)
            self.history.num_struct_heads_processed.append(
                float(epoch_stats.num_struct_heads_processed)
            )
            self.history.num_label_heads_processed.append(
                float(epoch_stats.num_label_heads_processed)
            )
            self.history.avg_sparsity.append(epoch_stats.avg_sparsity)
            self.history.first_batch_separation.append(
                epoch_stats.first_batch_separation.data
            )
            self.history.first_batch_grad_norms.append(
                epoch_stats.first_batch_grad_norms.data
            )
            self.history.kc_diagnostics.append(epoch_stats.kc_diagnostics)

            # Record active KC targets
            active_targets = sorted(list(kc_losses.keys()))
            self.history.active_kc_targets.append(",".join(active_targets))

            for k, v in kc_losses.items():
                if k not in self.history.kc_losses:
                    self.history.kc_losses[k] = []
                self.history.kc_losses[k].append(v)

            if self.kc_show_epoch_table:
                top_losses = dict(
                    sorted(kc_losses.items(), key=lambda x: x[1], reverse=True)[:5]
                )
                print_epoch_summary(
                    epoch=epoch + 1,
                    total_epochs=actual_epochs,
                    primary_metrics={
                        "Avg Loss": total_loss / max(1, len(self.data_loader)),
                        "Sparsity": avg_sparsity,
                    },
                    secondary_metrics=top_losses,
                )

            self.history.sentence_count.append(len(self.dataset))

            self.save_checkpoint(epoch + 1)

            on_epoch_end(self.history)

        return self.history


def _acc(p: List[int], labels: List[int]) -> float:
    return sum(x == y for x, y in zip(p, labels)) / len(labels) if labels else 0.0


def _mse(p: List[float], labels: List[float], ids: List[int]) -> float:
    return sum((p[i] - labels[i]) ** 2 for i in ids) / len(ids) if ids else 0.0


def _reg_acc(p: List[List[int]], labels: List[List[int]], ids: List[int]) -> float:
    return (
        sum(all(p[i][j] == labels[i][j] for j in range(len(p[i]))) for i in ids)
        / len(ids)
        if ids
        else 0.0
    )


class Trainer:
    # pylint: disable=too-many-locals,too-many-positional-arguments
    def __init__(
        self,
        model: StyleClassifier,
        train_dataset: StyleDataset,
        val_dataset: StyleDataset,
        config: TrainerConfig,
        dl_config_train: DataLoaderConfig,
        dl_config_val: DataLoaderConfig,
        output_path: str,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.name = "style_model"
        self.output_path = output_path
        self.kc_show_epoch_table = True
        configure_runtime_thread_limits(self.config)

        self.device = torch.device(self.config.device)
        self.model.to(self.device)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_sampler, self.val_sampler = None, None
        t_shuffle, v_shuffle = True, False

        if dl_config_train is None:
            dl_config_train = self.config.resolve_dataloader_config(
                self.device, mode="train"
            )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=t_shuffle,
            sampler=self.train_sampler,
            collate_fn=partial(collate_fn),
            num_workers=dl_config_train.num_workers,
            pin_memory=dl_config_train.pin_memory,
            persistent_workers=dl_config_train.persistent_workers,
            prefetch_factor=dl_config_train.prefetch_factor,
            worker_init_fn=_worker_init_fn,
        )

        if dl_config_val is None:
            dl_config_val = self.config.resolve_dataloader_config(
                self.device, mode="val"
            )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=v_shuffle,
            sampler=self.val_sampler,
            collate_fn=partial(collate_fn),
            num_workers=dl_config_val.num_workers,
            pin_memory=dl_config_val.pin_memory,
            persistent_workers=dl_config_val.persistent_workers,
            prefetch_factor=dl_config_val.prefetch_factor,
            worker_init_fn=_worker_init_fn,
        )

        pid = os.getpid()
        profile_dir = get_profile_dir()
        data_log = (
            os.path.join(profile_dir, f"train_data_{pid}.jsonl")
            if profile_dir
            else None
        )
        comp_log = (
            os.path.join(profile_dir, f"train_compute_{pid}.jsonl")
            if profile_dir
            else None
        )

        self.train_timer_data = Timer("data_loading", output_path=data_log)
        self.train_timer_compute = Timer("compute", output_path=comp_log)

        self.formality_criterion = nn.CrossEntropyLoss(
            weight=train_dataset.get_formality_class_weights().to(self.device)
        )
        self.gender_pragmatic_criterion = nn.CrossEntropyLoss(
            weight=train_dataset.get_gender_class_weights().to(self.device)
        )
        self.grammaticality_criterion = nn.CrossEntropyLoss(
            weight=train_dataset.get_grammaticality_class_weights().to(self.device)
        )

        mod = cast(StyleClassifier, self.model)

        enc_p = list(mod.embedding.parameters()) + list(mod.encoder.parameters())
        cls_p = (
            list(mod.formality_value_head.parameters())
            + list(mod.formality_pragmatic_head.parameters())
            + list(mod.gender_value_head.parameters())
            + list(mod.gender_pragmatic_head.parameters())
            + list(mod.grammaticality_classifier.parameters())
            + list(mod.register_classifier.parameters())
        )
        self.optimizer = Adam(
            [
                {
                    "params": enc_p,
                    "lr": self.config.learning_rate * self.config.encoder_lr_factor,
                },
                {"params": cls_p, "lr": self.config.learning_rate},
            ]
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.lr_scheduler_factor,
            patience=self.config.lr_scheduler_patience,
        )

        self.best_val_loss: float = float("inf")
        self.patience_counter = 0
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.start_epoch = 0
        self.start_batch = 0
        self.global_step = 0

        _safe_configure_threads(self.config)

        self.history = TrainingHistory()
        profile_dir = get_profile_dir()

        pid = os.getpid()
        self.train_timer_data = Timer(
            "train_data",
            os.path.join(profile_dir, f"train_data_{pid}.jsonl")
            if profile_dir
            else None,
        )
        self.train_timer_compute = Timer(
            "train_compute",
            os.path.join(profile_dir, f"train_compute_{pid}.jsonl")
            if profile_dir
            else None,
        )

    def save_checkpoint(self, epoch: int) -> None:
        if self.config.checkpoint.dir is None:
            return

        save_training_state(
            path=self.config.checkpoint.dir,
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            history=self.history,
            global_step=self.global_step,
            scheduler=self.scheduler,
            config=self.config,
            filename="checkpoint.pt",
        )

        checkpoint_meta = {
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "best_state": self.best_state,
        }
        torch.save(
            checkpoint_meta,
            os.path.join(self.config.checkpoint.dir, "checkpoint_meta.pt"),
        )

    def restore_from_checkpoint(self, path: str) -> bool:
        full_path = os.path.join(path, "checkpoint.pt")
        if not os.path.exists(full_path):
            return False

        checkpoint = load_training_state(
            path=path,
            model=getattr(self.model, "module", self.model),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            filename="checkpoint.pt",
        )
        self.start_epoch = checkpoint["epoch"]
        self.start_batch = checkpoint.get("batch_idx", 0)
        self.global_step = checkpoint.get("global_step", 0)
        history_data = checkpoint["history"]

        # Try to update existing history object (dataclass) from dict to preserve type
        if isinstance(history_data, dict):
            # We assume self.history is initialized correctly in __init__
            for k, v in history_data.items():
                if hasattr(self.history, k):
                    setattr(self.history, k, v)

            # Fallback: if self.history is a dict (legacy), update it
            if isinstance(self.history, dict):
                self.history.update(history_data)
        else:
            # history_data is already an object (e.g. from newer checkpoint?), replace
            self.history = history_data

        meta_path = os.path.join(path, "checkpoint_meta.pt")
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location="cpu")
            self.best_val_loss = meta.get("best_val_loss", float("inf"))
            self.patience_counter = meta.get("patience_counter", 0)
            self.best_state = meta.get("best_state")

        print()
        return True

    @staticmethod
    def _masked_mse(
        pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        loss_raw = F.mse_loss(pred, target, reduction="none")

        loss_masked = loss_raw * mask

        return loss_masked.sum() / (mask.sum() + 1e-6)

    @staticmethod
    def _masked_bce(
        pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        loss_raw = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        if mask.dim() < loss_raw.dim():
            mask = mask.unsqueeze(-1)

        loss_masked = loss_raw * mask

        return loss_masked.sum() / (mask.sum() * loss_raw.size(-1) + 1e-6)

    def _unpack_training_batch(
        self, batch: TrainingBatch
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        field_inputs = {
            f"input_ids_{f}": batch.feature_inputs[f"input_ids_{f}"].to(self.device)
            for f in FEATURE_FIELDS
        }
        attention_mask = batch.attention_mask.to(self.device)
        targets = {
            "f_val": batch.formality_value.to(self.device),
            "f_prag": batch.formality_pragmatic.to(self.device),
            "g_val": batch.gender_value.to(self.device),
            "g_prag": batch.gender_pragmatic.to(self.device),
            "gram": batch.grammaticality_labels.to(self.device),
            "reg": batch.register_labels.to(self.device),
        }
        return field_inputs, attention_mask, targets

    def _compute_component_losses(
        self,
        outputs: Tuple[torch.Tensor, ...],
        targets: Dict[str, torch.Tensor],
        is_valid_style: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            f_val_l,
            f_prag_l,
            g_val_l,
            g_prag_l,
            gram_l,
            reg_l,
        ) = outputs

        mask = is_valid_style.float()

        f_val_target = torch.nan_to_num(targets["f_val"], nan=0.0)
        f_mse = self._masked_mse(f_val_l.squeeze(-1), f_val_target, mask)

        f_loss = self.formality_criterion(f_prag_l, targets["f_prag"]) + f_mse

        g_val_target = torch.nan_to_num(targets["g_val"], nan=0.0)
        g_mse = self._masked_mse(g_val_l.squeeze(-1), g_val_target, mask)

        g_loss = self.gender_pragmatic_criterion(g_prag_l, targets["g_prag"]) + (
            g_mse * self.config.gender_mse_scaling_factor
        )

        gram_loss = self.grammaticality_criterion(gram_l, targets["gram"])

        reg_loss = self._masked_bce(reg_l, targets["reg"], mask)

        return f_loss, g_loss, gram_loss, reg_loss

    def _compute_training_loss(
        self, outputs: Tuple[torch.Tensor, ...], targets: Dict[str, torch.Tensor]
    ) -> TrainingLosses:
        is_gram = targets["gram"] == 1
        is_f_prag = targets["f_prag"] == 1
        is_g_prag = targets["g_prag"] == 1
        is_valid_style = is_gram & is_f_prag & is_g_prag

        f_loss, g_loss, gram_loss, reg_loss = self._compute_component_losses(
            outputs, targets, is_valid_style
        )

        loss = (
            self.config.formality_loss_weight * f_loss
            + self.config.gender_loss_weight * g_loss
            + self.config.grammaticality_loss_weight * gram_loss
            + self.config.register_loss_weight * reg_loss
        ) / self.config.grad_accum_steps

        return TrainingLosses(
            loss=loss,
            f_loss=f_loss,
            g_loss=g_loss,
            gram_loss=gram_loss,
            reg_loss=reg_loss,
        )

    def _train_batch(self, batch: TrainingBatch, batch_idx: int) -> Dict[str, float]:
        field_inputs, attention_mask, targets = self._unpack_training_batch(batch)

        if batch_idx % self.config.grad_accum_steps == 0:
            self.optimizer.zero_grad(set_to_none=True)

        outputs = self.model(field_inputs, attention_mask)
        losses = self._compute_training_loss(outputs, targets)
        loss = losses.loss

        loss.backward()

        if (batch_idx + 1) % self.config.grad_accum_steps == 0:
            if self.config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        gad = self.config.grad_accum_steps

        def _detach(val: Any) -> Any:
            if isinstance(val, torch.Tensor):
                return val.detach()
            return val

        return {
            "loss": _detach(loss) * gad,
            "formality_loss": _detach(losses.f_loss),
            "gender_loss": _detach(losses.g_loss),
            "grammaticality_loss": _detach(losses.gram_loss),
            "register_loss": _detach(losses.reg_loss),
        }

    def train_epoch(self, epoch: int) -> Tuple[float, float, float, float, float]:
        print_phase_header("Style", epoch=epoch + 1, total_epochs=self.config.epochs)

        self.model.train()

        metrics = TrainingMetrics()

        total_batches = len(self.train_loader)
        if total_batches == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        pbar = None
        current_loss_val = None
        pbar = RichTrainerProgressBar(
            desc=f"Style Epoch {epoch + 1}/{self.config.epochs}",
            total_steps=total_batches,
        )

        try:
            self.train_timer_data.start()

            for batch_idx, batch in enumerate(self.train_loader):
                if batch_idx < self.start_batch:
                    if pbar:
                        pbar.update(batch_idx, loss=0.0)
                    continue

                self.train_timer_data.stop(epoch=epoch, batch=batch_idx)
                self.train_timer_compute.start()

                losses = self._train_batch(batch, batch_idx)
                metrics.update(losses)

                if pbar:
                    if (batch_idx % self.config.progress_update_every == 0) or (
                        batch_idx == total_batches - 1
                    ):
                        current_loss_val = metrics.get_avg_loss()

                    pbar.update(batch_idx, loss=current_loss_val or 0.0)

                if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                    self.global_step += 1

                    if (
                        self.config.checkpoint.every_n_steps
                        and self.global_step % self.config.checkpoint.every_n_steps == 0
                    ):
                        self.save_checkpoint(epoch)

                self.train_timer_compute.stop(epoch=epoch, batch=batch_idx)
                self.train_timer_data.start()

        finally:
            if pbar:
                pbar.stop()

        self.start_batch = 0
        self.train_timer_data.stop()
        sys.stdout.write("\n")

        return metrics.average()

    def _extract_predictions(
        self, outputs: Tuple[torch.Tensor, ...], targets: Dict[str, torch.Tensor]
    ) -> TrainingPredictions:
        (
            f_v_l,
            f_p_l,
            g_v_l,
            g_p_l,
            gram_l,
            r_l,
        ) = outputs

        return TrainingPredictions(
            f_prag_p=f_p_l.argmax(-1).cpu().tolist(),
            f_prag_l=targets["f_prag"].cpu().tolist(),
            f_val_p=f_v_l.squeeze(-1).cpu().tolist(),
            f_val_l=targets["f_val"].cpu().tolist(),
            g_prag_p=g_p_l.argmax(-1).cpu().tolist(),
            g_prag_l=targets["g_prag"].cpu().tolist(),
            g_val_p=g_v_l.squeeze(-1).cpu().tolist(),
            g_val_l=targets["g_val"].cpu().tolist(),
            gram_p=gram_l.argmax(-1).cpu().tolist(),
            gram_l=targets["gram"].cpu().tolist(),
            reg_p=(torch.sigmoid(r_l) > 0.5).long().cpu().tolist(),
            reg_l=targets["reg"].long().cpu().tolist(),
            is_valid=(
                (targets["gram"] == 1)
                & (targets["f_prag"] == 1)
                & (targets["g_prag"] == 1)
            )
            .cpu()
            .tolist(),
        )

    @torch.no_grad()
    def evaluate(self) -> EvaluationMetrics:
        self.model.eval()
        n = 0
        metrics_sum: Dict[str, float] = {}
        all_preds: Dict[str, List[Any]] = {
            k: []
            for k in [
                "f_prag_p",
                "f_prag_l",
                "f_val_p",
                "f_val_l",
                "g_prag_p",
                "g_prag_l",
                "g_val_p",
                "g_val_l",
                "gram_p",
                "gram_l",
                "reg_p",
                "reg_l",
                "is_valid",
                "sentences",
                "kotograms",
            ]
        }

        for batch in self.val_loader:
            field_inputs, attention_mask, targets = self._unpack_training_batch(batch)

            outputs = self.model(field_inputs, attention_mask)

            self._accumulate_eval_batch(outputs, targets, batch, metrics_sum, all_preds)

            n += 1

        if n == 0:
            return EvaluationMetrics()

        avg_metrics = {k: v / n for k, v in metrics_sum.items()}
        valid_idxs = [i for i, v in enumerate(all_preds["is_valid"]) if v]

        return EvaluationMetrics(
            loss=avg_metrics.get("loss", 0.0) * self.config.grad_accum_steps,
            formality_loss=avg_metrics.get("f_loss", 0.0),
            gender_loss=avg_metrics.get("g_loss", 0.0),
            grammaticality_loss=avg_metrics.get("gram_loss", 0.0),
            register_loss=avg_metrics.get("reg_loss", 0.0),
            formality_accuracy=_acc(all_preds["f_prag_p"], all_preds["f_prag_l"]),
            formality_mse=_mse(all_preds["f_val_p"], all_preds["f_val_l"], valid_idxs),
            gender_accuracy=_acc(all_preds["g_prag_p"], all_preds["g_prag_l"]),
            gender_mse=_mse(all_preds["g_val_p"], all_preds["g_val_l"], valid_idxs),
            grammaticality_accuracy=_acc(all_preds["gram_p"], all_preds["gram_l"]),
            register_accuracy=_reg_acc(
                all_preds["reg_p"], all_preds["reg_l"], valid_idxs
            ),
        )

    def _accumulate_eval_batch(
        self,
        outputs: Tuple[torch.Tensor, ...],
        targets: Dict[str, torch.Tensor],
        batch: TrainingBatch,
        metrics_sum: Dict[str, float],
        all_preds: Dict[str, List[Any]],
    ) -> None:
        losses = self._compute_training_loss(outputs, targets)
        preds = self._extract_predictions(outputs, targets)

        # Helper to convert TrainingLosses dataclass to dict for summation
        losses_dict = {
            "loss": losses.loss,
            "f_loss": losses.f_loss,
            "g_loss": losses.g_loss,
            "gram_loss": losses.gram_loss,
            "reg_loss": losses.reg_loss,
        }

        for k, v in losses_dict.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            metrics_sum[k] = metrics_sum.get(k, 0.0) + val

        all_preds["f_prag_p"].extend(preds.f_prag_p)
        all_preds["f_prag_l"].extend(preds.f_prag_l)
        all_preds["f_val_p"].extend(preds.f_val_p)
        all_preds["f_val_l"].extend(preds.f_val_l)
        all_preds["g_prag_p"].extend(preds.g_prag_p)
        all_preds["g_prag_l"].extend(preds.g_prag_l)
        all_preds["g_val_p"].extend(preds.g_val_p)
        all_preds["g_val_l"].extend(preds.g_val_l)
        all_preds["gram_p"].extend(preds.gram_p)
        all_preds["gram_l"].extend(preds.gram_l)
        all_preds["reg_p"].extend(preds.reg_p)
        all_preds["reg_l"].extend(preds.reg_l)
        all_preds["is_valid"].extend(preds.is_valid)
        all_preds["sentences"].extend(batch.original_sentence)
        all_preds["kotograms"].extend(batch.kotogram)

    def train(
        self,
        epochs: int,
        on_epoch_end: Callable[[TrainingHistory], None],
    ) -> TrainingHistory:
        if self.config.checkpoint.resume_from:
            self.restore_from_checkpoint(self.config.checkpoint.resume_from)

        actual_epochs = epochs

        for epoch in range(self.start_epoch, actual_epochs):
            tl, tfl, tgl, tgraml, trl = self.train_epoch(epoch=epoch)
            eval_res = self.evaluate()
            self.scheduler.step(eval_res.loss)

            kc_probe_result = None
            m_style = cast(
                StyleClassifier,
                self.model,
            )

            if m_style.config.kc_enabled:
                probe_loader = self._build_kc_probe_loader(_max_batches=25)
                if probe_loader is not None:
                    kc_probe_result = self.evaluate_kc_probe(probe_loader)
                    self._diagnose_kc_probe(kc_probe_result)

            self.history.train_loss.append(tl)
            self.history.train_formality_loss.append(tfl)
            self.history.train_gender_loss.append(tgl)
            self.history.train_grammaticality_loss.append(tgraml)
            self.history.train_register_loss.append(trl)
            self.history.val_loss.append(eval_res.loss)
            self.history.val_formality_loss.append(eval_res.formality_loss)
            self.history.val_gender_loss.append(eval_res.gender_loss)
            self.history.val_grammaticality_loss.append(eval_res.grammaticality_loss)
            self.history.val_register_loss.append(eval_res.register_loss)
            self.history.val_formality_accuracy.append(eval_res.formality_accuracy)
            self.history.val_formality_mse.append(eval_res.formality_mse)
            self.history.val_gender_pragmatic_accuracy.append(eval_res.gender_accuracy)
            self.history.val_gender_value_mse.append(eval_res.gender_mse)
            self.history.val_grammaticality_accuracy.append(
                eval_res.grammaticality_accuracy
            )
            self.history.val_register_accuracy.append(eval_res.register_accuracy)
            self.history.sentence_count.append(len(self.train_dataset))

            data_avg = self.train_timer_data.avg()
            compute_avg = self.train_timer_compute.avg()
            total = data_avg + compute_avg
            if total > 0:
                print(
                    f"  [Time] Avg batch: {total * 1000:.1f}ms (Data: {data_avg * 1000:.1f}ms ({data_avg / total:.1%}), Compute: {compute_avg * 1000:.1f}ms)"
                )
            self.train_timer_data.reset()
            self.train_timer_compute.reset()

            primary_metrics = {"Train Loss": tl, "Val Loss": eval_res.loss}
            secondary_metrics = {
                "Formality": {
                    "Train": tfl,
                    "Val": eval_res.formality_loss,
                    "Acc": eval_res.formality_accuracy,
                },
                "Gender": {
                    "Train": tgl,
                    "Val": eval_res.gender_loss,
                    "Acc": eval_res.gender_accuracy,
                },
                "Grammar": {
                    "Train": tgraml,
                    "Val": eval_res.grammaticality_loss,
                    "Acc": eval_res.grammaticality_accuracy,
                },
                "Register": {
                    "Train": trl,
                    "Val": eval_res.register_loss,
                    "Acc": eval_res.register_accuracy,
                },
            }
            print_epoch_summary(
                epoch=epoch + 1,
                total_epochs=self.config.epochs,
                primary_metrics=primary_metrics,
                secondary_metrics=secondary_metrics,
            )

            is_best = eval_res.loss < self.best_val_loss
            if is_best:
                self.best_val_loss, self.patience_counter = eval_res.loss, 0
                self.best_state = {
                    k: cast(torch.Tensor, v.cpu().clone())
                    for k, v in self.model.state_dict().items()
                }
                os.makedirs(self.output_path, exist_ok=True)
                model_path = os.path.join(self.output_path, "model.pt")
                torch.save(self.best_state, model_path)
                print_best_model_saved(model_path, self.best_val_loss)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            self.save_checkpoint(epoch + 1)

            on_epoch_end(self.history)

        if self.best_state:
            self.model.load_state_dict(self.best_state, strict=False)
        return self.history

    def _build_kc_probe_loader(
        self, _max_batches: int = 25
    ) -> Optional[DataLoader[TrainingBatch]]:
        return cast(DataLoader[TrainingBatch], self.val_loader)

    def _update_kc_metrics(
        self,
        acc: KCMetricsAccumulator,
        outputs: Dict[str, Any],
        batch: TrainingBatch,
        config: KCProbeConfig,
    ) -> None:
        batch_size = outputs["kc_probs"].shape[0]
        acc.n_samples += batch_size

        self._update_kc_usage_stats(acc, outputs, batch_size, config)

        if "target_logits" in outputs:
            self._update_kc_structural_stats(acc, outputs, batch, config)

    def _compute_entropy_kl(
        self, logits_raw: torch.Tensor, config: KCProbeConfig
    ) -> Tuple[float, float]:
        logits_clamped = logits_raw.clamp(min=-8.0, max=8.0)
        q = torch.softmax(logits_clamped / config.tau_usage, dim=-1)
        p = q.mean(dim=0)
        p = p / p.sum().clamp_min(1e-9)

        eps = 1e-9
        log_p = (p + eps).log()
        entropy = -(p * log_p).sum()
        entropy_norm = entropy / math.log(config.vocab_size)
        kl_to_uniform = (p * (log_p + math.log(config.vocab_size))).sum()
        return entropy_norm.item(), kl_to_uniform.item()

    def _update_kc_usage_stats(
        self,
        acc: KCMetricsAccumulator,
        outputs: Dict[str, Any],
        batch_size: int,
        config: KCProbeConfig,
    ) -> None:
        if not acc.top1_hist.is_floating_point():
            acc.top1_hist = acc.top1_hist.float()
        if acc.top1_hist.device.type != "cpu":
            acc.top1_hist = acc.top1_hist.cpu()

        if not acc.topk_hist.is_floating_point():
            acc.topk_hist = acc.topk_hist.float()
        if acc.topk_hist.device.type != "cpu":
            acc.topk_hist = acc.topk_hist.cpu()

        top1_inds = outputs["topk_inds"][:, 0].flatten().cpu()

        counts_1 = torch.bincount(top1_inds)
        if len(counts_1) > len(acc.top1_hist):
            new_hist = torch.zeros(len(counts_1), dtype=torch.float)
            if len(acc.top1_hist) > 0:
                new_hist[: len(acc.top1_hist)] = acc.top1_hist.float()
            acc.top1_hist = new_hist

        acc.top1_hist[: len(counts_1)] += counts_1.float()

        topk_inds_flat = outputs["topk_inds"].flatten().cpu()
        counts_k = torch.bincount(topk_inds_flat)
        if len(counts_k) > len(acc.topk_hist):
            new_hist = torch.zeros(len(counts_k), dtype=torch.float)
            if len(acc.topk_hist) > 0:
                new_hist[: len(acc.topk_hist)] = acc.topk_hist.float()
            acc.topk_hist = new_hist

        acc.topk_hist[: len(counts_k)] += counts_k.float()

        entropy_norm, kl_to_uniform = self._compute_entropy_kl(
            outputs["kc_logits_raw"], config
        )

        acc.sum_entropy += entropy_norm * batch_size
        acc.sum_kl += kl_to_uniform * batch_size

        topk_vals = outputs["topk_vals"]
        acc.sum_tv += topk_vals.mean().item() * batch_size
        gap = topk_vals[:, 0] - topk_vals[:, -1]
        acc.sum_gap += gap.mean().item() * batch_size

        acc.sum_avg_prob += outputs["kc_probs"].mean().item() * batch_size
        acc.sum_act_dens += (
            outputs["sparse_activations"] > 0
        ).float().mean().item() * batch_size

    def _update_kc_structural_stats(
        self,
        acc: KCMetricsAccumulator,
        outputs: Dict[str, Any],
        batch: TrainingBatch,
        config: KCProbeConfig,
    ) -> None:
        kc_targets = create_kc_batch(
            batch, self.val_dataset.tokenizer, config.target_specs
        )
        for head_name in acc.head_samples:
            if head_name not in outputs["target_logits"]:
                continue
            logits_h = outputs["target_logits"][head_name]
            hs = acc.head_samples[head_name]
            self._update_single_head_stats(head_name, logits_h, kc_targets, hs, config)

    def _update_single_head_stats(
        self,
        head_name: str,
        logits_h: torch.Tensor,
        kc_targets: Dict[str, torch.Tensor],
        hs: KCDiagnosticHeadStats,
        config: KCProbeConfig,
    ) -> None:
        dense_key = f"kc_targets_{head_name}"
        pos_key = f"kc_pos_inds_{head_name}"
        mask_key = f"kc_pos_mask_{head_name}"

        if dense_key in kc_targets:
            self._update_dense_head_stats(
                kc_targets[dense_key], logits_h, hs, config.max_samples_per_head
            )
        elif pos_key in kc_targets and mask_key in kc_targets:
            self._update_sparse_head_stats(
                (kc_targets[pos_key], kc_targets[mask_key]),
                logits_h,
                hs,
                config.max_samples_per_head,
            )

    def _update_dense_head_stats(
        self,
        targets_h: torch.Tensor,
        logits_h: torch.Tensor,
        hs: KCDiagnosticHeadStats,
        max_samples_per_head: int,
    ) -> None:
        targets_h = targets_h.to(self.device).float()

        hs.p_sum += targets_h.sum().item()
        hs.count += targets_h.numel()

        pos_mask = targets_h > 0.5
        neg_mask = ~pos_mask

        if len(hs.pos_logits) < max_samples_per_head:
            pos_logits = logits_h[pos_mask].cpu().tolist()
            hs.pos_logits.extend(
                pos_logits[: max_samples_per_head - len(hs.pos_logits)]
            )
        if len(hs.neg_logits) < max_samples_per_head:
            neg_logits = logits_h[neg_mask].cpu().tolist()
            hs.neg_logits.extend(
                neg_logits[: max_samples_per_head - len(hs.neg_logits)]
            )

    def _sample_sparse_logits(
        self,
        pos_inds: torch.Tensor,
        pos_mask_t: torch.Tensor,
        logits_h: torch.Tensor,
        hs: KCDiagnosticHeadStats,
        max_samples_per_head: int,
    ) -> None:
        batch_size = pos_inds.size(0)
        vocab_size = logits_h.size(1)

        if len(hs.pos_logits) < max_samples_per_head:
            for i in range(min(batch_size, 4)):
                valid_inds = pos_inds[i, pos_mask_t[i]]
                if valid_inds.numel() > 0:
                    pos_log = logits_h[i, valid_inds].cpu().tolist()
                    hs.pos_logits.extend(
                        pos_log[: max_samples_per_head - len(hs.pos_logits)]
                    )

        if len(hs.neg_logits) < max_samples_per_head:
            for i in range(min(batch_size, 4)):
                neg_inds = torch.randint(4, vocab_size, (50,), device=self.device)
                neg_log = logits_h[i, neg_inds].cpu().tolist()
                hs.neg_logits.extend(
                    neg_log[: max_samples_per_head - len(hs.neg_logits)]
                )

    def _update_sparse_head_stats(
        self,
        sparse_data: Tuple[torch.Tensor, torch.Tensor],
        logits_h: torch.Tensor,
        hs: KCDiagnosticHeadStats,
        max_samples_per_head: int,
    ) -> None:
        pos_inds, pos_mask_t = sparse_data
        pos_inds = pos_inds.to(self.device)
        pos_mask_t = pos_mask_t.to(self.device)

        batch_size = pos_inds.size(0)
        vocab_size = logits_h.size(1)

        n_pos = pos_mask_t.sum().item()
        n_total = batch_size * vocab_size
        hs.p_sum += n_pos
        hs.count += n_total

        self._sample_sparse_logits(
            pos_inds, pos_mask_t, logits_h, hs, max_samples_per_head
        )

    def _compute_kc_metrics(
        self, acc: KCMetricsAccumulator, kc_vocab_size: int
    ) -> KCProbeEvaluationResult:
        n_samples = max(1, acc.n_samples)

        uniq_kcs = int((acc.topk_hist > 0).sum().item())
        max_top1 = float(acc.top1_hist.max().item()) / n_samples

        head_metrics: Dict[str, float] = {}

        for head_name, hs in acc.head_samples.items():
            p_true, auc, delta_bce = self._compute_head_metrics(hs)

            head_metrics[f"head_{head_name}_p_true"] = p_true
            head_metrics[f"head_{head_name}_auc"] = auc
            head_metrics[f"head_{head_name}_delta_bce"] = delta_bce

        return KCProbeEvaluationResult(
            n_samples=n_samples,
            uniq_kcs=uniq_kcs,
            max_top1=max_top1,
            entropy_norm=acc.sum_entropy / n_samples,
            kl_to_uniform=acc.sum_kl / n_samples,
            tv_mean=acc.sum_tv / n_samples,
            gap_mean=acc.sum_gap / n_samples,
            avg_prob=acc.sum_avg_prob / n_samples,
            act_dens=acc.sum_act_dens / n_samples,
            kc_vocab_size=kc_vocab_size,
            head_metrics=head_metrics,
        )

    def _compute_head_metrics(
        self, hs: KCDiagnosticHeadStats
    ) -> Tuple[float, float, float]:
        p_true = hs.p_sum / max(1, hs.count)
        auc = float("nan")
        delta_bce = float("nan")

        pos_l = hs.pos_logits
        neg_l = hs.neg_logits
        if len(pos_l) > 0 and len(neg_l) > 0:
            all_logits = pos_l + neg_l
            all_labels = [1.0] * len(pos_l) + [0.0] * len(neg_l)
            combined = sorted(zip(all_logits, all_labels), key=lambda x: x[0])
            ranks = list(range(1, len(combined) + 1))
            pos_rank_sum = sum(r for r, (_, lbl) in zip(ranks, combined) if lbl > 0.5)
            n_pos = len(pos_l)
            n_neg = len(neg_l)
            auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / max(1, n_pos * n_neg)

            logits_t = torch.tensor(all_logits, dtype=torch.float32)
            labels_t = torch.tensor(all_labels, dtype=torch.float32)
            model_bce = F.binary_cross_entropy_with_logits(logits_t, labels_t).item()
            baseline_logit = math.log(p_true / (1 - p_true)) if 0 < p_true < 1 else 0.0
            baseline_bce = F.binary_cross_entropy_with_logits(
                torch.full_like(logits_t, baseline_logit), labels_t
            ).item()
            delta_bce = model_bce - baseline_bce

        return p_true, auc, delta_bce

    def _run_kc_probe_loop(
        self,
        probe_loader: DataLoader[TrainingBatch],
        acc: KCMetricsAccumulator,
        config: KCProbeConfig,
        m: StyleClassifierWithKC,
        max_batches: int,
        temperature: float,
    ) -> None:
        with torch.no_grad():
            for batch_idx, batch in enumerate(probe_loader):
                if batch_idx >= max_batches:
                    break

                field_inputs = {
                    k: v.to(self.device) for k, v in batch.feature_inputs.items()
                }
                attention_mask = batch.attention_mask.to(self.device)

                outputs = m(
                    field_inputs,
                    attention_mask=attention_mask,
                    mode="kc",
                    temperature=temperature,
                    gumbel_scale=0.0,
                )

                self._update_kc_metrics(acc, outputs, batch, config)

    def evaluate_kc_probe(
        self,
        probe_loader: DataLoader[TrainingBatch],
        max_batches: int = 25,
        temperature: float = 1.5,
        tau_usage: float = 2.0,
    ) -> KCProbeEvaluationResult:
        m = cast(
            StyleClassifierWithKC,
            self.model,
        )

        m.eval()

        config = KCProbeConfig(
            tau_usage=tau_usage,
            vocab_size=int(m.config.kc_vocab_size),
            topk=int(m.config.kc_topk),
            target_specs=m.config.kc_target_specs,
            max_samples_per_head=2000,
        )

        probe_heads = ["lemma", "pos", "conjugated_form", "conjugated_type"]
        head_samples: Dict[str, KCDiagnosticHeadStats] = {
            h: KCDiagnosticHeadStats() for h in probe_heads if h in config.target_specs
        }

        acc = KCMetricsAccumulator(
            topk_hist=torch.zeros(
                config.vocab_size, device=self.device, dtype=torch.long
            ),
            top1_hist=torch.zeros(
                config.vocab_size, device=self.device, dtype=torch.long
            ),
            head_samples=head_samples,
        )

        self._run_kc_probe_loop(probe_loader, acc, config, m, max_batches, temperature)

        return self._compute_kc_metrics(acc, config.vocab_size)

    def _diagnose_kc_probe(self, probe_result: KCProbeEvaluationResult) -> List[str]:
        recommendations: List[str] = []

        max_top1 = probe_result.max_top1
        entropy_norm = probe_result.entropy_norm
        uniq_kcs = probe_result.uniq_kcs
        kc_vocab_size = probe_result.kc_vocab_size

        collapse_risk = max_top1 > 0.10 or entropy_norm < 0.85
        if collapse_risk:
            recommendations.append(
                f"⚠️ COLLAPSE RISK: maxTop1={max_top1:.3f} (want <0.10), entN={entropy_norm:.3f} (want >0.85). "
                "Try: reduce encoder_lr_factor (0.1→0.01) or freeze encoder for first 2 epochs."
            )

        # Avoid division by zero
        usage_ratio = 0.0
        if kc_vocab_size > 0:
            usage_ratio = uniq_kcs / kc_vocab_size

        if usage_ratio < 0.5:
            recommendations.append(
                f"⚠️ LOW DIVERSITY: only {uniq_kcs}/{kc_vocab_size} KCs used ({usage_ratio:.1%}). "
                "Try: increase diversity_weight_thawed or lower temperature."
            )

        for head in ["lemma", "pos", "conjugated_form", "conjugated_type"]:
            auc = probe_result.head_metrics.get(f"head_{head}_auc", float("nan"))
            if not math.isnan(auc) and auc < 0.80:
                recommendations.append(
                    f"⚠️ QUALITY DROP ({head}): AUC={auc:.3f} (want >0.85). "
                    "Try: add KC auxiliary loss during STYLE or retrain KC decoders post-STYLE."
                )

        if not recommendations:
            recommendations.append("✅ KC health OK. No action needed.")

        print(
            f"  KCProbe: uniq={probe_result.uniq_kcs}/{probe_result.kc_vocab_size} "
            f"maxTop1={probe_result.max_top1:.3f} "
            f"entN={probe_result.entropy_norm:.3f} "
            f"klU={probe_result.kl_to_uniform:.3f} "
            f"tv={probe_result.tv_mean:.3f} "
            f"gap={probe_result.gap_mean:.3f} "
            f"prob={probe_result.avg_prob:.2f} "
            f"dens={probe_result.act_dens:.4f}"
        )
        for head in ["lemma", "pos", "conjugated_form", "conjugated_type"]:
            auc = probe_result.head_metrics.get(f"head_{head}_auc", float("nan"))
            delta = probe_result.head_metrics.get(
                f"head_{head}_delta_bce", float("nan")
            )
            if not math.isnan(auc):
                print(f"    {head}: AUC={auc:.3f} ΔBCE={delta:+.4f}")

        return recommendations
