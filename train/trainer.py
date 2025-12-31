# pylint: disable=too-many-lines,not-callable
import math
import os
import sys
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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
    format_kc_first_batch_summary,
    print_best_model_saved,
    print_epoch_summary,
    print_kc_first_batch_debug,
    print_phase_header,
)
from train.distributed import is_main_process
from train.io import (
    load_training_state,
    save_training_state,
)
from train.kc_diagnostics import KCEpochDiag
from train.models import StyleClassifierWithKC
from train.profile import Timer, get_profile_dir
from train.types import (
    EvaluationMetrics,
    KCMetricsAccumulator,
    KCProbeConfig,
    KCTrainingHistory,
    TrainingHistory,
    TrainingMetrics,
)
from train.worker import _worker_init_fn


def tensor_finite_stats(x: Optional[torch.Tensor]) -> Dict[str, Any]:
    if x is None:
        return {
            "finite": True,
            "n_nan": 0,
            "n_inf": 0,
            "min": float("nan"),
            "max": float("nan"),
        }

    if x.isfinite().all():
        return {
            "finite": True,
            "n_nan": 0,
            "n_inf": 0,
            "min": float("nan"),
            "max": float("nan"),
        }

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

    return {
        "finite": False,
        "n_nan": n_nan,
        "n_inf": n_inf,
        "min": min_val,
        "max": max_val,
    }


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

        if dist.is_initialized():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if torch.cuda.is_available():
                self.device = torch.device("cuda", local_rank)
            else:
                self.device = torch.device("cpu")
            self.is_distributed = True
        else:
            self.device = torch.device(self.config.device)
            self.is_distributed = False

        self.model.to(self.device)
        if self.is_distributed:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if self.device.type == "cuda":
                device_ids = [local_rank]
                output_device = local_rank
            else:
                device_ids = None
                output_device = None

            self.model = cast(
                StyleClassifierWithKC,
                DDP(
                    self.model,
                    device_ids=device_ids,
                    output_device=output_device,
                    find_unused_parameters=True,
                ),
            )

        pad_id = dataset.tokenizer.pad_id
        max_seq_len = getattr(self.model, "module", self.model).config.max_seq_len

        self.sampler: Optional[DistributedSampler] = (
            DistributedSampler(
                dataset,
                shuffle=True,
            )
            if self.is_distributed
            else None
        )

        if dl_config is None:
            dl_config = self.config.resolve_dataloader_config(self.device)

        self.data_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(self.sampler is None),
            sampler=self.sampler,
            collate_fn=partial(
                collate_fn,
                pad_id=pad_id,
                max_seq_len=max_seq_len,
                vocab_sizes=dataset.tokenizer.get_vocab_sizes(),
            ),
            num_workers=dl_config.num_workers,
            pin_memory=dl_config.pin_memory,
            persistent_workers=dl_config.persistent_workers,
            prefetch_factor=dl_config.prefetch_factor,
            worker_init_fn=_worker_init_fn,
        )
        self._create_optimizer(freeze_encoder=True)

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

        self.kc_temperature_frozen = float(
            getattr(
                getattr(self.model, "module", self.model).config, "kc_temperature", 1.0
            )
        )
        self.kc_temperature_thawed = self.kc_config.temperature_thawed

        self.kc_log_level = self.kc_config.log_level
        self.kc_first_batch_debug_every = self.kc_config.first_batch_debug_every

        self.kc_first_batch_debug_epochs = list(self.kc_config.first_batch_debug_epochs)

        self.kc_show_epoch_table = self.kc_config.show_epoch_table
        self.kc_show_step_checks = self.kc_config.show_step_checks
        self.kc_show_grad_norms = self.kc_config.show_grad_norms

        self.kc_grad_cap = self.kc_config.kc_grad_cap

        if self.kc_log_level == "debug":
            self.kc_show_epoch_table = True
            self.kc_show_step_checks = True
            self.kc_show_grad_norms = True

        self.history = KCTrainingHistory()
        profile_dir = get_profile_dir()
        pid = os.getpid()
        self.train_timer_data = Timer(
            "kc_data",
            os.path.join(profile_dir, f"kc_data_{pid}.jsonl") if profile_dir else None,
        )
        self.train_timer_compute = Timer(
            "kc_compute",
            os.path.join(profile_dir, f"kc_compute_{pid}.jsonl")
            if profile_dir
            else None,
        )
        self.start_epoch = 0
        self.start_batch = 0
        self.global_step = 0
        self._did_print_debug_for_epoch = -1

        self._kc_last_good_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        self._nonfinite_streak = 0
        self._nonfinite_total = 0
        self._nonfinite_logged = 0
        self._max_nonfinite_streak = 50

        self._consecutive_step_skips = 0
        self._total_step_skips = 0
        self._total_steps_applied = 0
        self._max_consecutive_skips = self.kc_config.max_consecutive_skips

    def save_checkpoint(self, epoch: int, batch_idx: int = 0) -> None:
        if not is_main_process() or self.config.checkpoint.dir is None:
            return

        save_training_state(
            path=self.config.checkpoint.dir,
            model=getattr(self.model, "module", self.model),
            optimizer=self.optimizer,
            epoch=epoch,
            history=self.history,
            global_step=self.global_step,
            batch_idx=batch_idx,
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
            device=str(self.device),
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
        if is_main_process():
            print(
                f"  [Resume] Restored KC checkpoint from {path} "
                f"(epoch {self.start_epoch}, step {self.global_step})"
            )
        return True

    def _save_kc_snapshot(self) -> None:
        raw = self.model.module if self.is_distributed else self.model
        m = cast(StyleClassifierWithKC, raw)
        self._kc_last_good_state = {
            "kc_head": {
                k: v.detach().cpu().clone() for k, v in m.kc_head.state_dict().items()
            },
        }
        if hasattr(m, "kc_decoders"):
            self._kc_last_good_state["kc_decoders"] = {
                k: v.detach().cpu().clone()
                for k, v in m.kc_decoders.state_dict().items()
            }

    def _restore_kc_snapshot(self) -> bool:
        if self._kc_last_good_state is None:
            return False
        raw = self.model.module if self.is_distributed else self.model
        m = cast(StyleClassifierWithKC, raw)

        device = next(m.kc_head.parameters()).device
        restored_head = {
            k: v.to(device) for k, v in self._kc_last_good_state["kc_head"].items()
        }
        m.kc_head.load_state_dict(restored_head, strict=True)

        if "kc_decoders" in self._kc_last_good_state and hasattr(m, "kc_decoders"):
            device_dec = next(m.kc_decoders.parameters()).device
            restored_dec = {
                k: v.to(device_dec)
                for k, v in self._kc_last_good_state["kc_decoders"].items()
            }
            m.kc_decoders.load_state_dict(restored_dec, strict=True)

        return True

    def _reinit_kc_head(self) -> None:
        raw = self.model.module if self.is_distributed else self.model
        m = cast(StyleClassifierWithKC, raw)
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
                probs = torch.sigmoid(gathered)

                valid_bool = valid.bool()
                diag.update_family(
                    family_name,
                    pos_inds,
                    pos_mask,
                    probs[valid_bool],
                    t[valid_bool],
                    loss.item(),
                    mask_id=reading_mask_id,
                )
        return loss

    # pylint: disable=too-many-locals
    def _init_structural_decoder_biases(self, num_batches: int = 10) -> None:
        raw = self.model.module if self.is_distributed else self.model
        m = cast("StyleClassifierWithKC", raw)
        if not hasattr(m, "kc_decoders"):
            return

        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}

        for i, batch in enumerate(self.data_loader):
            if i >= num_batches:
                break

            kc_targets = batch
            if not any(k.startswith("kc_") for k in batch):
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
                    sums[name] = sums.get(name, 0.0) + p
                    counts[name] = counts.get(name, 0) + 1
                elif mask_key in kc_targets:
                    pos_mask_t = kc_targets[mask_key]
                    batch_size = pos_mask_t.size(0)
                    num_pos = pos_mask_t.sum().item()
                    p = num_pos / (batch_size * vocab_size)
                    sums[name] = sums.get(name, 0.0) + p
                    counts[name] = counts.get(name, 0) + 1

        if self.is_distributed:
            names = sorted(sums.keys())

            data = []
            for n in names:
                data.append(sums[n])
                data.append(float(counts[n]))

            t_data = torch.tensor(data, device=self.device)
            dist.all_reduce(t_data, op=dist.ReduceOp.SUM)

            t_list = t_data.tolist()
            for i, n in enumerate(names):
                sums[n] = t_list[2 * i]
                counts[n] = int(t_list[2 * i + 1])

        for name, s in sums.items():
            p = s / max(1, counts[name])
            p = min(max(p, 1e-6), 1 - 1e-6)
            b = math.log(p / (1 - p))

            lin = cast(nn.Linear, m.kc_decoders.decoders[name])
            with torch.no_grad():
                if lin.bias is not None:
                    nn.init.constant_(lin.bias, b)

            if is_main_process():
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
        has_printed_step_check: bool,
        accum: int,
        is_flush: bool = False,
    ) -> bool:
        w0_before = 0.0
        if self.kc_show_step_checks:
            w0 = m.kc_head.linear.weight
            w0_before = w0.detach().flatten()[0].item()

        if is_main_process() and (not has_printed_step_check or is_flush):
            if self.kc_show_grad_norms:
                gn_kc = self._grad_norm(m.kc_head)
                dec_name = (
                    "pos"
                    if "pos" in m.kc_decoders.decoders
                    else next(iter(m.kc_decoders.decoders.keys()))
                )
                dec = m.kc_decoders.decoders[dec_name]
                gn_dec = self._grad_norm(dec) if dec is not None else 0.0
                phase = "Flush" if is_flush else "Pre-Step"
                print(
                    f"  KC {phase} Grad Norms: kc_head={gn_kc:.6f} decoder={gn_dec:.6f}"
                    + (f" (flush_accum={accum})" if is_flush else "")
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

            if is_main_process():
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

        clip_val = self.config.gradient_clip if self.config.gradient_clip > 0 else 1.0
        params_to_clip = [
            p
            for group in self.optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if params_to_clip:
            nn.utils.clip_grad_norm_(params_to_clip, clip_val)

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
                if is_main_process():
                    print("  [KC] Params became NaN after step, restoring snapshot")
                restored = self._restore_kc_snapshot()
                if not restored:
                    self._reinit_kc_head()
                skipped = True
            else:
                self._save_kc_snapshot()

                self._consecutive_step_skips = 0
                self._total_steps_applied += 1

        if is_main_process() and (not has_printed_step_check or is_flush):
            if self.kc_show_step_checks:
                w0 = m.kc_head.linear.weight
                w0_after = w0.detach().flatten()[0].item()
                print(
                    f"  KC {'Flush ' if is_flush else ''}Step Check: kc_head.w0 {w0_before:.6f} -> {w0_after:.6f} "
                    f"(delta={w0_after - w0_before:+.6f}, accum={accum}/{self.config.grad_accum_steps})"
                )

        self.optimizer.zero_grad(set_to_none=True)
        return skipped

    def _create_optimizer(self, freeze_encoder: bool) -> None:
        raw = self.model.module if self.is_distributed else self.model
        m = cast(StyleClassifierWithKC, raw)

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

    # pylint: disable=too-many-locals
    def train_epoch(
        self, epoch: int = 0
    ) -> Tuple[float, Dict[str, float], float, Dict[str, float]]:
        should_freeze = epoch < self.freeze_encoder_epochs
        self._create_optimizer(freeze_encoder=should_freeze)

        if is_main_process():
            print_phase_header(
                "KC",
                info="Encoder Frozen" if should_freeze else "Encoder Thawed",
                epoch=epoch + 1,
                total_epochs=self.config.kc_epochs,
            )

        self.model.train()
        total_loss, n_batches = 0.0, 0
        kc_losses: Dict[str, float] = {}
        total_sparsity = 0.0

        total_batches = len(self.data_loader)

        running_struct_loss, running_label_loss = 0.0, 0.0
        running_num_struct_total, running_num_label_total = 0, 0
        running_sparsity = 0.0
        first_batch_separation: Dict[str, float] = {}
        first_batch_grad_norms: Dict[str, float] = {}

        has_printed_step_check = False

        opt_steps = 0
        flush_steps = 0
        pending_accum = 0
        did_any_backward = False

        raw = self.model.module if self.is_distributed else self.model
        kc_vocab_size = int(cast(StyleClassifierWithKC, raw).config.kc_vocab_size)
        topk_hist = torch.zeros(kc_vocab_size, dtype=torch.long)
        top1_hist = torch.zeros(kc_vocab_size, dtype=torch.long)
        kc_usage_total_samples = 0
        kc_tv_sum = 0.0
        kc_tv_min = float("inf")
        kc_tv_max = float("-inf")
        kc_gap_sum = 0.0
        kc_gap_count = 0
        running_entropy_norm = 0.0
        running_kl_to_uniform = 0.0
        running_p_max = 0.0
        running_avg_prob = 0.0
        running_act_dens = 0.0

        running_loss_components = {
            "base": 0.0,
            "struct": 0.0,
            "label": 0.0,
            "div": 0.0,
            "lb": 0.0,
            "collapse": 0.0,
            "sparsity": 0.0,
        }

        kc_diag = KCEpochDiag()
        reading_mask_id = getattr(self.dataset.tokenizer, "unk_id", 0)
        if "reading" in self.dataset.tokenizer.field_vocabs:
            reading_mask_id = self.dataset.tokenizer.field_vocabs["reading"].get(
                "<READING_MASK>", reading_mask_id
            )

        self.optimizer.zero_grad(set_to_none=True)

        pbar = None

        current_display_loss = 0.5
        if is_main_process():
            pbar = RichTrainerProgressBar(
                f"KC Epoch {epoch + 1}" + (" (Frozen)" if should_freeze else ""),
                total_steps=total_batches,
                transient=False,
            )

        self.train_timer_data.start()
        for batch_idx, batch in enumerate(self.data_loader):
            self.train_timer_data.stop(epoch=epoch, batch=batch_idx)
            self.train_timer_compute.start()

            if epoch == self.start_epoch and batch_idx < self.start_batch:
                self.train_timer_compute.stop(epoch=epoch, batch=batch_idx)
                self.train_timer_data.start()
                continue

            raw = self.model.module if self.is_distributed else self.model
            m = cast(StyleClassifierWithKC, raw)

            if not any(
                k.startswith("kc_targets_") or k.startswith("kc_pos_inds_")
                for k in batch
            ):
                kc_targets = create_kc_batch(
                    batch=batch,
                    tokenizer=self.dataset.tokenizer,
                    target_specs=m.config.kc_target_specs,
                )
                batch.update(kc_targets)

            field_inputs = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k.startswith("input_ids_")
            }
            attention_mask = batch["attention_mask"].to(self.device)

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
            )

            should_check_nan = (
                batch_idx < 50 or (batch_idx % 50 == 0) or self._nonfinite_streak > 0
            )

            if should_check_nan:
                logits_stats = tensor_finite_stats(outputs.get("kc_logits_raw"))
                probs_stats = tensor_finite_stats(outputs.get("kc_probs"))
                forward_nonfinite = (
                    not logits_stats["finite"] or not probs_stats["finite"]
                )
            else:
                forward_nonfinite = False

            if forward_nonfinite:
                self._nonfinite_streak += 1
                self._nonfinite_total += 1

                should_log = (
                    self._nonfinite_logged < 3 or self._nonfinite_total % 50 == 0
                )
                if should_log and is_main_process():
                    self._nonfinite_logged += 1
                    msg = (
                        f"  [KC][FORWARD NaN] ep={epoch} b={batch_idx} streak={self._nonfinite_streak} "
                        f"raw[nan={logits_stats['n_nan']} inf={logits_stats['n_inf']} "
                        f"finite_range={logits_stats['min']:.2g}..{logits_stats['max']:.2g}] "
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
                    if is_main_process():
                        if pbar:
                            pbar.log(
                                "  [KC] No snapshot available, reinitializing kc_head"
                            )
                        else:
                            print(
                                "  [KC] No snapshot available, reinitializing kc_head"
                            )
                    self._reinit_kc_head()

                if self._nonfinite_streak == 1:
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = pg["lr"] * 0.5
                    if is_main_process():
                        msg = f"  [KC] Halved LR to {self.optimizer.param_groups[0]['lr']:.2e}"
                        if pbar:
                            pbar.log(msg)
                        else:
                            print(msg)

                continue

            self._nonfinite_streak = min(self._nonfinite_streak, 0)

            if epoch >= self.freeze_encoder_epochs:
                topk_vals_clamped = outputs["topk_vals"].clamp(max=0.85)

                sparse_clamped = torch.zeros_like(outputs["kc_probs"])
                sparse_clamped.scatter_(1, outputs["topk_inds"], topk_vals_clamped)

                if hasattr(m, "kc_decoders"):
                    outputs["target_logits"] = m.kc_decoders(sparse_clamped)

            topk_inds = outputs.get("topk_inds", None)
            topk_vals = outputs.get("topk_vals", None)

            if topk_inds is not None and topk_vals is not None:
                inds_cpu = topk_inds.detach().to("cpu")
                vals_cpu = topk_vals.detach().to("cpu")

                batch_size = int(inds_cpu.size(0))
                kc_usage_total_samples += batch_size

                flat = inds_cpu.reshape(-1)
                topk_hist += torch.bincount(flat, minlength=kc_vocab_size)

                top1 = inds_cpu[:, 0]
                top1_hist += torch.bincount(top1, minlength=kc_vocab_size)

                kc_tv_sum += float(vals_cpu.sum().item())
                kc_tv_min = min(kc_tv_min, float(vals_cpu.min().item()))
                kc_tv_max = max(kc_tv_max, float(vals_cpu.max().item()))

                gap = vals_cpu[:, 0] - vals_cpu[:, -1]
                kc_gap_sum += float(gap.sum().item())
                kc_gap_count += int(gap.numel())

                target_logits = outputs["target_logits"]

                if (
                    batch_idx == 0
                    and is_main_process()
                    and epoch != self._did_print_debug_for_epoch
                ):
                    self._did_print_debug_for_epoch = epoch
                    raw = self.model.module if self.is_distributed else self.model
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
                        m = cast(StyleClassifierWithKC, raw)

                        if self.kc_log_level == "debug":
                            print_kc_first_batch_debug(
                                epoch,
                                outputs["kc_logits"],
                                outputs["kc_probs"],
                                outputs["sparse_activations"],
                                outputs["target_logits"],
                                batch,
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

                            all_heads = sorted(outputs["target_logits"].keys())

                            # pylint: disable=too-many-locals

                            def get_head_stat(
                                name: str,
                                batch: Dict[str, Any],
                                outputs: Dict[str, Any],
                            ) -> Dict[str, Any]:
                                logits = outputs["target_logits"][name]

                                dense_key = f"kc_targets_{name}"
                                pos_key = f"kc_pos_inds_{name}"
                                mask_key = f"kc_pos_mask_{name}"
                                t: Optional[torch.Tensor] = None
                                pos_inds: Optional[torch.Tensor] = None
                                pos_mask_t: Optional[torch.Tensor] = None

                                if dense_key in batch:
                                    t = batch[dense_key].to(self.device).float()
                                    is_sparse = False
                                elif pos_key in batch and mask_key in batch:
                                    pos_inds = batch[pos_key].to(self.device)
                                    pos_mask_t = batch[mask_key].to(self.device)
                                    is_sparse = True
                                else:
                                    return {}

                                with torch.no_grad():
                                    if is_sparse:
                                        assert pos_inds is not None
                                        assert pos_mask_t is not None
                                        batch_size = pos_mask_t.size(0)
                                        vocab_size = logits.size(1)
                                        n_pos = pos_mask_t.sum().item()
                                        total = batch_size * vocab_size
                                        p = n_pos / (total + self.kc_pos_weight_eps)
                                        p = max(
                                            self.kc_pos_weight_eps,
                                            min(p, 1.0 - self.kc_pos_weight_eps),
                                        )
                                        pos_w = min(
                                            self.kc_pos_weight_cap,
                                            max(1.0, (1.0 - p) / p),
                                        )

                                        probs = torch.sigmoid(logits)
                                        p_avg = probs.mean().item()

                                        auc = 0.0
                                        pos_logits_list = []
                                        neg_logits_list = []
                                        for i in range(min(batch_size, 4)):
                                            valid_inds = pos_inds[i, pos_mask_t[i]]
                                            if valid_inds.numel() > 0:
                                                pos_logits_list.extend(
                                                    probs[i, valid_inds].cpu().tolist()
                                                )
                                            neg_inds = torch.randint(
                                                4, vocab_size, (50,), device=self.device
                                            )
                                            neg_logits_list.extend(
                                                probs[i, neg_inds].cpu().tolist()
                                            )

                                        if pos_logits_list and neg_logits_list:
                                            pos_t = torch.tensor(
                                                pos_logits_list[:500],
                                                device=self.device,
                                            )
                                            neg_t = torch.tensor(
                                                neg_logits_list[:500],
                                                device=self.device,
                                            )
                                            sp = torch.cat([pos_t, neg_t])
                                            sl = torch.cat(
                                                [
                                                    torch.ones(
                                                        len(pos_t), device=self.device
                                                    ),
                                                    torch.zeros(
                                                        len(neg_t), device=self.device
                                                    ),
                                                ]
                                            )
                                            comb = torch.stack([sp, sl], dim=1)
                                            idx = torch.argsort(comb[:, 0])
                                            sl_s = comb[idx, 1]
                                            ranks = torch.arange(
                                                1, sl_s.numel() + 1, device=self.device
                                            ).float()
                                            pos_rank_sum = (ranks * sl_s).sum().item()
                                            n_pos_auc, n_neg_auc = (
                                                len(pos_t),
                                                len(neg_t),
                                            )
                                            if n_pos_auc > 0 and n_neg_auc > 0:
                                                auc = (
                                                    pos_rank_sum
                                                    - n_pos_auc * (n_pos_auc + 1) / 2
                                                ) / (n_pos_auc * n_neg_auc)

                                        delta = 0.0

                                    else:
                                        assert t is not None
                                        pos = t.sum()
                                        total = t.numel()
                                        p = (
                                            pos / (total + self.kc_pos_weight_eps)
                                        ).clamp(
                                            min=self.kc_pos_weight_eps,
                                            max=1.0 - self.kc_pos_weight_eps,
                                        )
                                        pos_w = ((1.0 - p) / p).clamp(
                                            min=1.0, max=self.kc_pos_weight_cap
                                        )

                                        probs = torch.sigmoid(logits)
                                        p_avg = probs.mean().item()

                                        auc = 0.0
                                        pos_mask = t > 0.5
                                        neg_mask = ~pos_mask
                                        if pos_mask.any() and neg_mask.any():
                                            max_s = 1000
                                            idx_p = torch.where(pos_mask.view(-1))[0]
                                            idx_n = torch.where(neg_mask.view(-1))[0]
                                            if idx_p.numel() > max_s:
                                                idx_p = idx_p[:max_s]
                                            if idx_n.numel() > max_s:
                                                idx_n = idx_n[:max_s]

                                            sp = torch.cat(
                                                [
                                                    probs.view(-1)[idx_p],
                                                    probs.view(-1)[idx_n],
                                                ]
                                            )
                                            sl = torch.cat(
                                                [
                                                    torch.ones(
                                                        idx_p.numel(),
                                                        device=self.device,
                                                    ),
                                                    torch.zeros(
                                                        idx_n.numel(),
                                                        device=self.device,
                                                    ),
                                                ]
                                            )

                                            comb = torch.stack([sp, sl], dim=1)
                                            idx = torch.argsort(comb[:, 0])
                                            sl_s = comb[idx, 1]
                                            ranks = torch.arange(
                                                1, sl_s.numel() + 1, device=self.device
                                            ).float()
                                            pos_rank_sum = (ranks * sl_s).sum().item()
                                            n_pos, n_neg = idx_p.numel(), idx_n.numel()
                                            auc = (
                                                pos_rank_sum - n_pos * (n_pos + 1) / 2
                                            ) / (n_pos * n_neg)

                                        bias_used = logits.mean().item()

                                        pw = torch.tensor(
                                            pos_w.item(), device=self.device
                                        )
                                        hl = F.binary_cross_entropy_with_logits(
                                            logits, t, pos_weight=pw
                                        ).item()
                                        pl = F.binary_cross_entropy_with_logits(
                                            torch.full_like(logits, bias_used),
                                            t,
                                            pos_weight=pw,
                                        ).item()
                                        delta = hl - pl

                                        p = p.item()
                                        pos_w = pos_w.item()

                                return {
                                    "name": name,
                                    "p": p if isinstance(p, float) else p,
                                    "pos_w": pos_w
                                    if isinstance(pos_w, float)
                                    else pos_w,
                                    "p_avg": p_avg,
                                    "auc": auc,
                                    "delta": delta,
                                }

                            selected_stats = []
                            seen = set()

                            for h in priority:
                                if h in all_heads:
                                    s = get_head_stat(h, batch, outputs)
                                    if s:
                                        selected_stats.append(s)
                                        seen.add(h)

                            others = [h for h in all_heads if h not in seen]

                            other_stats = []
                            for h in others:
                                s = get_head_stat(h, batch, outputs)
                                if s:
                                    other_stats.append(s)

                            other_stats.sort(key=lambda x: float(x.get("p", 0.0)))
                            selected_stats.extend(other_stats[:2])

                            logits = outputs["kc_logits"]
                            logits_raw = (
                                outputs.get("kc_logits_raw", logits).detach().float()
                            )
                            probs = outputs["kc_probs"]
                            sp = outputs["sparse_activations"]

                            kc_stats = {
                                "logits_mean": logits.mean().item(),
                                "logits_std": logits.std().item(),
                                "raw_logits_mean": logits_raw.mean().item(),
                                "raw_logits_std": logits_raw.std().item(),
                                "raw_logits_min": logits_raw.min().item(),
                                "raw_logits_max": logits_raw.max().item(),
                                "probs_mean": probs.mean().item(),
                                "probs_std": probs.std().item(),
                                "probs_gt05": (probs > 0.5).float().mean().item(),
                                "probs_gt09": (probs > 0.9).float().mean().item(),
                                "topk_mean": outputs.get("topk_vals", probs)
                                .mean()
                                .item(),
                                "topk_min": outputs.get("topk_vals", probs)
                                .min()
                                .item(),
                                "topk_max": outputs.get("topk_vals", probs)
                                .max()
                                .item(),
                                "sparse_mean": sp.mean().item(),
                                "nonzero": (sp > 0).sum(dim=-1).float().mean().item(),
                                "unique_kcs": len(torch.unique(outputs["topk_inds"])),
                            }

                            msg = format_kc_first_batch_summary(
                                kc_stats, selected_stats[:5]
                            )
                            if pbar:
                                pbar.log(msg)
                            else:
                                print(msg)

                    for name, logits in outputs["target_logits"].items():
                        target_key = f"kc_targets_{name}"
                        if target_key not in batch:
                            continue
                        targets = batch[target_key].to(self.device).float()
                        with torch.no_grad():
                            pos_mask = targets > 0.5
                            neg_mask = ~pos_mask
                            if pos_mask.any() and neg_mask.any():
                                pmn = (
                                    logits[pos_mask].mean().item()
                                    - logits[neg_mask].mean().item()
                                )
                                first_batch_separation[name] = pmn

                loss = torch.tensor(0.0, device=self.device)
                batch_kc_losses = {}
                structural_loss = torch.tensor(0.0, device=self.device)
                num_struct = 0
                label_loss = torch.tensor(0.0, device=self.device)
                num_label = 0

                for name, logits in target_logits.items():
                    target_key = f"kc_targets_{name}"
                    pos_key = f"kc_pos_inds_{name}"
                    mask_key = f"kc_pos_mask_{name}"
                    vocab_size = int(m.config.kc_target_specs.get(name, 0))

                    if target_key in batch:
                        targets = batch[target_key].to(self.device).float()
                        logits_f = logits.float()

                        batch_size_f, vocab_size_f = logits_f.shape
                        if vocab_size_f > 256:
                            pos_mask = targets > 0.5
                            neg_count = 128

                            neg_inds = torch.randint(
                                0,
                                vocab_size_f,
                                (batch_size_f, neg_count),
                                device=self.device,
                            )
                            mask = torch.zeros_like(logits_f, dtype=torch.bool)
                            mask.scatter_(1, neg_inds, True)
                            mask = mask | pos_mask

                            if mask.any():
                                subset_logits = logits_f[mask]
                                subset_targets = targets[mask]
                                task_loss = F.binary_cross_entropy_with_logits(
                                    subset_logits, subset_targets
                                )

                                with torch.no_grad():
                                    probs = torch.sigmoid(subset_logits)

                                    dummy_pos = torch.zeros(
                                        (batch_size_f, 0),
                                        device=self.device,
                                        dtype=torch.long,
                                    )
                                    dummy_mask = torch.zeros(
                                        (batch_size_f, 0),
                                        device=self.device,
                                        dtype=torch.bool,
                                    )

                                    kc_diag.update_family(
                                        name,
                                        dummy_pos,
                                        dummy_mask,
                                        probs,
                                        subset_targets,
                                        task_loss.item(),
                                    )

                                    v_ids = (
                                        torch.arange(vocab_size_f, device=self.device)
                                        .unsqueeze(0)
                                        .expand(batch_size_f, -1)
                                    )
                                    kc_diag.update_family(
                                        name,
                                        v_ids,
                                        pos_mask,
                                        probs,
                                        subset_targets,
                                        task_loss.item(),
                                    )
                            else:
                                task_loss = torch.tensor(
                                    0.0, device=self.device, requires_grad=True
                                )
                        else:
                            task_loss = F.binary_cross_entropy_with_logits(
                                logits_f, targets
                            )

                            with torch.no_grad():
                                probs = torch.sigmoid(logits_f)
                                pos_mask = targets > 0.5
                                v_ids = (
                                    torch.arange(vocab_size_f, device=self.device)
                                    .unsqueeze(0)
                                    .expand(batch_size_f, -1)
                                )
                                kc_diag.update_family(
                                    name,
                                    v_ids,
                                    pos_mask,
                                    probs.flatten(),
                                    targets.flatten(),
                                    task_loss.item(),
                                )

                        structural_loss += task_loss
                        num_struct += 1
                        batch_kc_losses[name] = task_loss.item()

                    elif pos_key in batch and mask_key in batch:
                        pos_inds = batch[pos_key].to(self.device)
                        pos_mask_t = batch[mask_key].to(self.device)
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
                        targets = batch["formality_value"].to(self.device)
                        task_loss = self.mse_loss(logits.squeeze(-1), targets)
                        label_loss += task_loss
                        num_label += 1
                        batch_kc_losses[name] = task_loss.item()
                    elif name == "formality_pragmatic":
                        targets = batch["formality_pragmatic"].to(self.device)
                        task_loss = self.ce_loss(logits, targets)
                        label_loss += task_loss
                        num_label += 1
                        batch_kc_losses[name] = task_loss.item()
                    elif name == "gender_value":
                        targets = batch["gender_value"].to(self.device)
                        task_loss = self.mse_loss(logits.squeeze(-1), targets)
                        label_loss += task_loss
                        num_label += 1
                        batch_kc_losses[name] = task_loss.item()
                    elif name == "gender_pragmatic":
                        targets = batch["gender_pragmatic"].to(self.device)
                        task_loss = self.ce_loss(logits, targets)
                        label_loss += task_loss
                        num_label += 1
                        batch_kc_losses[name] = task_loss.item()
                    elif name == "grammaticality":
                        targets = batch["grammaticality_labels"].to(self.device)
                        task_loss = self.ce_loss(logits, targets)
                        label_loss += task_loss
                        num_label += 1
                        batch_kc_losses[name] = task_loss.item()
                    elif name == "register":
                        targets = batch["register_labels"].to(self.device)
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

                combined_loss = torch.tensor(0.0, device=self.device)
                if num_struct > 0:
                    combined_loss += 0.7 * (structural_loss / num_struct)
                if num_label > 0:
                    combined_loss += 0.3 * (label_loss / num_label)

                if epoch < self.freeze_encoder_epochs:
                    div_weight = self.kc_diversity_weight_frozen
                    lb_weight = self.kc_lb_weight_frozen
                else:
                    div_weight = self.kc_diversity_weight_thawed
                    lb_weight = self.kc_lb_weight_thawed

                diversity_loss = torch.tensor(0.0, device=self.device)
                entropy_norm = torch.tensor(0.0, device=self.device)
                kl_to_uniform = torch.tensor(0.0, device=self.device)

                loss_div_val = 0.0
                loss_lb_val = 0.0
                loss_coll_val = 0.0

                if epoch >= self.kc_diversity_warmup_epochs:
                    logits_usage = outputs.get("logits_usage", outputs["kc_logits_raw"])

                    tau_usage = 1.0 if epoch < self.freeze_encoder_epochs else 2.0

                    q = torch.softmax(logits_usage / tau_usage, dim=-1)

                    p = q.mean(dim=0)

                    p_sum = p.sum().clamp_min(self.kc_diversity_eps)
                    p = p / p_sum

                    log_p = (p + self.kc_diversity_eps).log()
                    entropy = -(p * log_p).sum()
                    entropy_norm = entropy / math.log(kc_vocab_size)
                    diversity_loss = 1.0 - entropy_norm

                    kl_val = (p * (p.clamp_min(1e-9) * kc_vocab_size).log()).sum()
                    load_balance_loss = kl_val / math.log(kc_vocab_size)

                    p_max = p.max()

                    if epoch >= self.freeze_encoder_epochs:
                        thr = max(3.0 / max(1, kc_vocab_size), 0.002)

                        diff = (p_max - thr).clamp_min(0.0)

                        if self.kc_collapse_weight_thawed > 0:
                            collapse_penalty = diff
                            combined_loss += (
                                self.kc_collapse_weight_thawed * collapse_penalty
                            )
                            loss_coll_val = (
                                self.kc_collapse_weight_thawed * collapse_penalty
                            ).item()

                    if div_weight > 0:
                        combined_loss += div_weight * diversity_loss
                        loss_div_val = (div_weight * diversity_loss).item()

                    if lb_weight > 0:
                        combined_loss += lb_weight * load_balance_loss
                        loss_lb_val = (lb_weight * load_balance_loss).item()

                    kl_to_uniform = kl_val

                running_entropy_norm += entropy_norm.item()
                running_kl_to_uniform += kl_to_uniform.item()
                running_p_max += p_max.item() if ("p_max" in locals()) else 0.0

                if (
                    self.kc_sparsity_weight > 0
                    and self.kc_sparsity_mode == "target_density"
                ):
                    avg_prob = outputs["kc_probs"].mean()

                    act_dens = (outputs["sparse_activations"] > 0).float().mean()

                    sparsity_term = avg_prob
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
                    "base": combined_loss.item(),
                    "struct": structural_loss.item(),
                    "label": label_loss.item(),
                    "div": loss_div_val,
                    "lb": loss_lb_val,
                    "collapse": loss_coll_val,
                    "sparsity": loss_spar_val,
                }

                running_loss_components["base"] += current_epoch_comp["base"]
                running_loss_components["div"] += current_epoch_comp["div"]
                running_loss_components["lb"] += current_epoch_comp["lb"]
                running_loss_components["collapse"] += current_epoch_comp["collapse"]

                running_loss_components["base"] -= (
                    current_epoch_comp["div"]
                    + current_epoch_comp["lb"]
                    + current_epoch_comp["collapse"]
                )

                running_loss_components["struct"] += (
                    (current_epoch_comp["struct"] / num_struct)
                    if num_struct > 0
                    else 0.0
                )
                running_loss_components["label"] += (
                    (current_epoch_comp["label"] / num_label) if num_label > 0 else 0.0
                )
                running_loss_components["sparsity"] += current_epoch_comp["sparsity"]

                if loss.item() == 0.0 and loss.requires_grad:
                    pass

            nonfinite_reason = None
            if not torch.isfinite(loss):
                nonfinite_reason = "loss"
            elif not torch.isfinite(outputs.get("kc_probs", torch.tensor(0.0))).all():
                nonfinite_reason = "kc_probs"
            elif not torch.isfinite(outputs.get("topk_vals", torch.tensor(0.0))).all():
                nonfinite_reason = "topk_vals"

            if nonfinite_reason:
                if is_main_process():
                    kc_logits_raw = outputs.get("kc_logits_raw", None)
                    kc_probs = outputs.get("kc_probs", None)
                    msg = f"  [KC][NON-FINITE {nonfinite_reason.upper()}] epoch={epoch} batch={batch_idx} loss={loss.item()}"
                    if kc_logits_raw is not None:
                        msg += f" raw[min={kc_logits_raw.min().item():.3g} max={kc_logits_raw.max().item():.3g}]"
                    if kc_probs is not None:
                        msg += f" probs[min={kc_probs.min().item():.3g} max={kc_probs.max().item():.3g}]"

                    print(msg)

                self.optimizer.zero_grad(set_to_none=True)

            loss.backward()
            did_any_backward = True
            pending_accum += 1

            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                self._perform_optimizer_step(
                    m, has_printed_step_check, pending_accum, is_flush=False
                )

                opt_steps += 1
                has_printed_step_check = True
                pending_accum = 0

            total_loss += loss.item() * self.config.grad_accum_steps
            for k, v in batch_kc_losses.items():
                kc_losses[k] = kc_losses.get(k, 0.0) + v
            n_batches += 1
            self.global_step += 1

            if (
                self.config.checkpoint.every_n_steps
                and self.global_step % self.config.checkpoint.every_n_steps == 0
            ):
                self.save_checkpoint(epoch, batch_idx)

            if is_main_process() and pbar:
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
            raw = self.model.module if self.is_distributed else self.model
            m = cast(StyleClassifierWithKC, raw)
            self._perform_optimizer_step(
                m, has_printed_step_check, pending_accum, is_flush=True
            )

            flush_steps += 1
            has_printed_step_check = True

        if is_main_process():
            sys.stdout.write("\n")
            sys.stdout.flush()

        avg_kc_losses = {k: v / n_batches for k, v in kc_losses.items()}
        avg_sparsity = total_sparsity / max(1, n_batches)

        if self.is_distributed and dist.is_initialized():
            hist_dev = topk_hist.to(self.device)
            top1_dev = top1_hist.to(self.device)
            dist.all_reduce(hist_dev, op=dist.ReduceOp.SUM)
            dist.all_reduce(top1_dev, op=dist.ReduceOp.SUM)
            topk_hist = hist_dev.to("cpu")
            top1_hist = top1_dev.to("cpu")

            scal = torch.tensor(
                [kc_usage_total_samples, kc_tv_sum, kc_gap_sum, kc_gap_count],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(scal, op=dist.ReduceOp.SUM)
            kc_usage_total_samples = int(scal[0].item())
            kc_tv_sum = float(scal[1].item())
            kc_gap_sum = float(scal[2].item())
            kc_gap_count = int(scal[3].item())

        uniq_kcs_epoch = int((topk_hist > 0).sum().item())
        max_top1 = float(top1_hist.max().item()) / max(1, kc_usage_total_samples)

        k_val = (
            int(
                getattr(
                    cast(StyleClassifierWithKC, self.model.module).config, "kc_topk", 8
                )
            )
            if self.is_distributed
            else int(getattr(self.model.config, "kc_topk", 8))
        )
        tv_mean = kc_tv_sum / max(1, kc_usage_total_samples * k_val)
        gap_mean = kc_gap_sum / max(1, kc_gap_count)
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

        epoch_stats = {
            "avg_struct_loss": running_struct_loss / max(1, running_num_struct_total),
            "avg_label_loss": running_label_loss / max(1, running_num_label_total),
            "num_struct_heads_processed": running_num_struct_total,
            "num_label_heads_processed": running_num_label_total,
            "avg_sparsity": running_sparsity / max(1, n_batches),
            "avg_prob": running_avg_prob / max(1, n_batches),
            "act_dens": running_act_dens / max(1, n_batches),
            "first_batch_separation": first_batch_separation,
            "first_batch_grad_norms": first_batch_grad_norms,
            "avg_entropy_norm": avg_entropy_norm,
            "avg_kl_to_uniform": avg_kl_to_uniform,
            "uniq_kcs_epoch": uniq_kcs_epoch,
            "max_top1_epoch": max_top1,
            "avg_p_max": running_p_max / max(1, n_batches),
            "kc_diagnostics": kc_diag.get_stats(),
        }

        avg_loss_components = {
            k: v / max(1, n_batches) for k, v in running_loss_components.items()
        }

        if is_main_process() and not self.kc_show_epoch_table:
            if not self.kc_show_epoch_table:
                top_losses = sorted(
                    avg_kc_losses.items(), key=lambda x: x[1], reverse=True
                )[:3]
                amp_stats = {
                    "skips": 0,
                    "start": 1.0,
                    "end": 1.0,
                    "opt_steps": opt_steps,
                    "flush_steps": flush_steps,
                }

                weights_dict = {
                    "div": div_weight if "div_weight" in locals() else 0.0,
                    "lb": lb_weight if "lb_weight" in locals() else 0.0,
                    "collapse": self.kc_collapse_weight_thawed
                    if epoch >= self.freeze_encoder_epochs
                    else 0.0,
                }
                from train.display import (
                    format_kc_epoch_compact_summary,
                    format_kc_loss_breakdown,
                    format_kc_usage_summary,
                )

                lines = []
                lines.append(
                    format_kc_loss_breakdown(avg_loss_components, weights_dict)
                )

                lines.append(
                    format_kc_epoch_compact_summary(
                        epoch + 1,
                        self.config.epochs,
                        total_loss / n_batches,
                        cast(float, epoch_stats["avg_prob"]),
                        cast(float, epoch_stats["act_dens"]),
                        cast(float, epoch_stats["avg_struct_loss"]),
                        top_losses,
                        amp_stats,
                        entropy_norm=avg_entropy_norm,
                        avg_kl_to_uniform=avg_kl_to_uniform,
                        uniq_kcs=uniq_kcs_epoch,
                        avg_p_max=cast(float, epoch_stats.get("avg_p_max")),
                    )
                )

                lines.append(
                    format_kc_usage_summary(
                        uniq=uniq_kcs_epoch,
                        total=kc_usage_total_samples,
                        max_top1=max_top1,
                        tv_mean=tv_mean,
                        gap_mean=gap_mean,
                        topk_counts=topk_counts_list,
                        top1_counts=top1_counts_list,
                        k=k_val,
                    )
                )

                msg = "\n".join(lines)
                if pbar:
                    pbar.log(msg)
                else:
                    print(msg)

        if is_main_process():
            msg = (
                f"  KC Health: maxTop1={max_top1:.3f} uniqKCs={uniq_kcs_epoch}/{kc_vocab_size} "
                f"avgProb={cast(float, epoch_stats['avg_prob']):.3f} actDens={cast(float, epoch_stats['act_dens']):.4f} "
                f"entN={avg_entropy_norm:.3f} klU={avg_kl_to_uniform:.3f}"
            )
            if pbar:
                pbar.log(msg)
            else:
                print(msg)

        if is_main_process():
            diag_lines = kc_diag.finalize(epoch)
            for line in diag_lines:
                if pbar:
                    pbar.log(line)
                else:
                    print(line)

        if pbar:
            pbar.stop()

        return (
            total_loss / n_batches,
            avg_kc_losses,
            avg_sparsity,
            cast(Dict[str, float], epoch_stats),
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
        epochs: Optional[int] = None,
        on_epoch_end: Optional[Callable[[KCTrainingHistory], None]] = None,
    ) -> KCTrainingHistory:
        if self.config.checkpoint.resume_from:
            self.restore_from_checkpoint(self.config.checkpoint.resume_from)

        if self.start_epoch == 0 and self.start_batch == 0:
            self._init_structural_decoder_biases()

        actual_epochs = epochs or self.config.kc_epochs
        for epoch in range(self.start_epoch, actual_epochs):
            if self.is_distributed:
                cast(DistributedSampler, self.sampler).set_epoch(epoch)
            total_loss, kc_losses, avg_sparsity, epoch_stats = self.train_epoch(
                epoch=epoch
            )

            if is_main_process():
                self._log_training_progress()

            self.history.total_loss.append(total_loss)
            self.history.kc_sparsity.append(avg_sparsity)
            self.history.avg_struct_loss.append(epoch_stats["avg_struct_loss"])
            self.history.avg_label_loss.append(epoch_stats["avg_label_loss"])
            self.history.num_struct_heads_processed.append(
                epoch_stats["num_struct_heads_processed"]
            )
            self.history.num_label_heads_processed.append(
                epoch_stats["num_label_heads_processed"]
            )
            self.history.avg_sparsity.append(epoch_stats["avg_sparsity"])
            self.history.first_batch_separation.append(
                epoch_stats["first_batch_separation"]
            )
            self.history.first_batch_grad_norms.append(
                epoch_stats["first_batch_grad_norms"]
            )
            self.history.kc_diagnostics.append(
                cast(Dict[str, Any], epoch_stats.get("kc_diagnostics", {}))
            )

            for k, v in kc_losses.items():
                if k not in self.history.kc_losses:
                    self.history.kc_losses[k] = []
                self.history.kc_losses[k].append(v)

            if is_main_process() and self.kc_show_epoch_table:
                top_losses = dict(
                    sorted(kc_losses.items(), key=lambda x: x[1], reverse=True)[:5]
                )
                print_epoch_summary(
                    epoch + 1,
                    actual_epochs,
                    {"Total Loss": total_loss, "Sparsity": avg_sparsity},
                    top_losses,
                    phase="KC",
                    kc_epoch_stats=epoch_stats,
                )

            self.history.sentence_count.append(len(self.dataset))

            self.save_checkpoint(epoch + 1, 0)

            if on_epoch_end and is_main_process():
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
        output_path: Optional[str] = None,
        kc_show_epoch_table: bool = True,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.name = "style_model"
        self.output_path = output_path or "checkpoints"
        self.kc_show_epoch_table = kc_show_epoch_table
        configure_runtime_thread_limits(self.config)

        if torch.distributed.is_initialized():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if torch.cuda.is_available():
                self.device = torch.device("cuda", local_rank)
            else:
                self.device = torch.device("cpu")
            self.is_distributed = True
        else:
            self.device = torch.device(self.config.device)
            self.is_distributed = False

        self.model.to(self.device)
        if self.is_distributed:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if self.device.type == "cuda":
                device_ids = [local_rank]
                output_device = local_rank
            else:
                device_ids = None
                output_device = None

            self.model = cast(
                StyleClassifierWithKC,
                DDP(
                    self.model,
                    device_ids=device_ids,
                    output_device=output_device,
                    find_unused_parameters=True,
                ),
            )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        pad_id = train_dataset.tokenizer.pad_id
        max_seq_len = getattr(self.model, "module", self.model).config.max_seq_len

        if self.is_distributed:
            self.train_sampler: Optional[DistributedSampler] = DistributedSampler(
                train_dataset,
                shuffle=True,
            )
            self.val_sampler: Optional[DistributedSampler] = DistributedSampler(
                val_dataset,
                shuffle=False,
            )
            t_shuffle, v_shuffle = False, False
        else:
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
            collate_fn=partial(
                collate_fn, pad_id=pad_id, max_seq_len=cast(Optional[int], max_seq_len)
            ),
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
            collate_fn=partial(
                collate_fn, pad_id=pad_id, max_seq_len=cast(Optional[int], max_seq_len)
            ),
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

        mod = cast(
            StyleClassifier, self.model.module if self.is_distributed else self.model
        )
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

    def save_checkpoint(self, epoch: int, batch_idx: int = 0) -> None:
        if not is_main_process() or self.config.checkpoint.dir is None:
            return

        save_training_state(
            path=self.config.checkpoint.dir,
            model=getattr(self.model, "module", self.model),
            optimizer=self.optimizer,
            epoch=epoch,
            history=self.history,
            global_step=self.global_step,
            batch_idx=batch_idx,
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
            device=str(self.device),
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

        if is_main_process():
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
        self, batch: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], torch.Tensor, Dict[str, torch.Tensor]]:
        field_inputs = {
            f"input_ids_{f}": batch[f"input_ids_{f}"].to(self.device)
            for f in FEATURE_FIELDS
        }
        attention_mask = batch["attention_mask"].to(self.device)
        targets = {
            "f_val": batch["formality_value"].to(self.device),
            "f_prag": batch["formality_pragmatic"].to(self.device),
            "g_val": batch["gender_value"].to(self.device),
            "g_prag": batch["gender_pragmatic"].to(self.device),
            "gram": batch["grammaticality_labels"].to(self.device),
            "reg": batch["register_labels"].to(self.device),
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
        g_loss = self.formality_criterion(g_prag_l, targets["g_prag"]) + (
            g_mse * self.config.gender_mse_scaling_factor
        )

        g_loss = self.gender_pragmatic_criterion(g_prag_l, targets["g_prag"]) + (
            g_mse * self.config.gender_mse_scaling_factor
        )

        gram_loss = self.grammaticality_criterion(gram_l, targets["gram"])

        reg_loss = self._masked_bce(reg_l, targets["reg"], mask)

        return f_loss, g_loss, gram_loss, reg_loss

    def _compute_training_loss(
        self, outputs: Tuple[torch.Tensor, ...], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
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

        return {
            "loss": loss,
            "f_loss": f_loss,
            "g_loss": g_loss,
            "gram_loss": gram_loss,
            "reg_loss": reg_loss,
        }

    def _train_batch(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, float]:
        field_inputs, attention_mask, targets = self._unpack_training_batch(batch)

        if batch_idx % self.config.grad_accum_steps == 0:
            self.optimizer.zero_grad(set_to_none=True)

        outputs = self.model(field_inputs, attention_mask)
        losses = self._compute_training_loss(outputs, targets)
        loss = losses["loss"]

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
            "formality_loss": _detach(losses["f_loss"]),
            "gender_loss": _detach(losses["g_loss"]),
            "grammaticality_loss": _detach(losses["gram_loss"]),
            "register_loss": _detach(losses["reg_loss"]),
        }

    def train_epoch(self, epoch: int) -> Tuple[float, float, float, float, float]:
        if is_main_process():
            print_phase_header(
                "Style", epoch=epoch + 1, total_epochs=self.config.epochs
            )

        self.model.train()
        metrics = TrainingMetrics()

        total_batches = len(self.train_loader)
        if total_batches == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        pbar = None
        current_loss_val = None
        if is_main_process():
            pbar = RichTrainerProgressBar(
                f"Style Epoch {epoch + 1}/{self.config.epochs}",
                total_steps=total_batches,
                transient=False,
            )

        try:
            self.train_timer_data.start()

            for batch_idx, batch in enumerate(self.train_loader):
                if batch_idx < self.start_batch:
                    if pbar:
                        pbar.update(batch_idx, desc="Skipping...")
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

                    pbar.update(batch_idx, loss=current_loss_val)

                if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                    self.global_step += 1

                    if (
                        self.config.checkpoint.every_n_steps
                        and self.global_step % self.config.checkpoint.every_n_steps == 0
                    ):
                        self.save_checkpoint(epoch, batch_idx)

                self.train_timer_compute.stop(epoch=epoch, batch=batch_idx)
                self.train_timer_data.start()

        finally:
            if pbar:
                pbar.stop()

        self.start_batch = 0
        self.train_timer_data.stop()
        if is_main_process():
            sys.stdout.write("\n")

        return metrics.average()

    def _extract_predictions(
        self, outputs: Tuple[torch.Tensor, ...], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, List[Any]]:
        (
            f_v_l,
            f_p_l,
            g_v_l,
            g_p_l,
            gram_l,
            r_l,
        ) = outputs

        return {
            "f_prag_p": f_p_l.argmax(-1).cpu().tolist(),
            "f_prag_l": targets["f_prag"].cpu().tolist(),
            "f_val_p": f_v_l.squeeze(-1).cpu().tolist(),
            "f_val_l": targets["f_val"].cpu().tolist(),
            "g_prag_p": g_p_l.argmax(-1).cpu().tolist(),
            "g_prag_l": targets["g_prag"].cpu().tolist(),
            "g_val_p": g_v_l.squeeze(-1).cpu().tolist(),
            "g_val_l": targets["g_val"].cpu().tolist(),
            "gram_p": gram_l.argmax(-1).cpu().tolist(),
            "gram_l": targets["gram"].cpu().tolist(),
            "reg_p": (torch.sigmoid(r_l) > 0.5).long().cpu().tolist(),
            "reg_l": targets["reg"].long().cpu().tolist(),
            "is_valid": (
                (targets["gram"] == 1)
                & (targets["f_prag"] == 1)
                & (targets["g_prag"] == 1)
            )
            .cpu()
            .tolist(),
        }

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
        batch: Dict[str, Any],
        metrics_sum: Dict[str, float],
        all_preds: Dict[str, List[Any]],
    ) -> None:
        losses = self._compute_training_loss(outputs, targets)
        preds = self._extract_predictions(outputs, targets)

        for k, v in losses.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            metrics_sum[k] = metrics_sum.get(k, 0.0) + val

        for k, v in preds.items():
            all_preds[k].extend(v)
        all_preds["sentences"].extend(batch.get("original_sentence", []))
        all_preds["kotograms"].extend(batch.get("kotogram", []))

    def train(
        self,
        epochs: Optional[int] = None,
        on_epoch_end: Optional[Callable[[TrainingHistory], None]] = None,
    ) -> TrainingHistory:
        if self.config.checkpoint.resume_from:
            self.restore_from_checkpoint(self.config.checkpoint.resume_from)

        actual_epochs = epochs or self.config.epochs

        for epoch in range(self.start_epoch, actual_epochs):
            tl, tfl, tgl, tgraml, trl = self.train_epoch(epoch=epoch)
            eval_res = self.evaluate()
            self.scheduler.step(eval_res.loss)

            kc_probe_result = None
            m_style = cast(
                StyleClassifier,
                self.model.module if self.is_distributed else self.model,
            )
            if getattr(m_style.config, "kc_enabled", False):
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

            if is_main_process():
                data_avg = self.train_timer_data.avg()
                compute_avg = self.train_timer_compute.avg()
                total = data_avg + compute_avg
                if total > 0:
                    print(
                        f"  [Time] Avg batch: {total * 1000:.1f}ms (Data: {data_avg * 1000:.1f}ms ({data_avg / total:.1%}), Compute: {compute_avg * 1000:.1f}ms)"
                    )
                self.train_timer_data.reset()
                self.train_timer_compute.reset()

            if is_main_process():
                metrics = {
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
                    epoch + 1,
                    self.config.epochs,
                    {"Train Loss": tl, "Val Loss": eval_res.loss},
                    metrics,
                    phase="Style",
                )

            is_best = eval_res.loss < self.best_val_loss
            if is_best:
                self.best_val_loss, self.patience_counter = eval_res.loss, 0
                self.best_state = {
                    k: cast(torch.Tensor, v.cpu().clone())
                    for k, v in self.model.state_dict().items()
                }
                if is_main_process():
                    os.makedirs(self.output_path, exist_ok=True)
                    model_path = os.path.join(self.output_path, "model.pt")
                    torch.save(self.best_state, model_path)
                    print_best_model_saved(model_path, self.best_val_loss)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    if is_main_process():
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            self.save_checkpoint(epoch + 1, 0)

            if self.is_distributed:
                dist.barrier()

            if on_epoch_end and is_main_process():
                on_epoch_end(self.history)

        if self.best_state:
            self.model.load_state_dict(self.best_state, strict=False)
        return self.history

    def _build_kc_probe_loader(
        self, _max_batches: int = 25
    ) -> Optional[DataLoader[Dict[str, Any]]]:
        return cast(DataLoader[Dict[str, Any]], self.val_loader)

    def _update_kc_metrics(
        self,
        acc: KCMetricsAccumulator,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
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
        batch: Dict[str, Any],
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
        hs: Dict[str, Any],
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
        hs: Dict[str, Any],
        max_samples_per_head: int,
    ) -> None:
        targets_h = targets_h.to(self.device).float()

        hs["p_sum"] += targets_h.sum().item()
        hs["count"] += targets_h.numel()

        pos_mask = targets_h > 0.5
        neg_mask = ~pos_mask

        if len(hs["pos_logits"]) < max_samples_per_head:
            pos_logits = logits_h[pos_mask].cpu().tolist()
            hs["pos_logits"].extend(
                pos_logits[: max_samples_per_head - len(hs["pos_logits"])]
            )
        if len(hs["neg_logits"]) < max_samples_per_head:
            neg_logits = logits_h[neg_mask].cpu().tolist()
            hs["neg_logits"].extend(
                neg_logits[: max_samples_per_head - len(hs["neg_logits"])]
            )

    def _sample_sparse_logits(
        self,
        pos_inds: torch.Tensor,
        pos_mask_t: torch.Tensor,
        logits_h: torch.Tensor,
        hs: Dict[str, Any],
        max_samples_per_head: int,
    ) -> None:
        batch_size = pos_inds.size(0)
        vocab_size = logits_h.size(1)

        if len(hs["pos_logits"]) < max_samples_per_head:
            for i in range(min(batch_size, 4)):
                valid_inds = pos_inds[i, pos_mask_t[i]]
                if valid_inds.numel() > 0:
                    pos_log = logits_h[i, valid_inds].cpu().tolist()
                    hs["pos_logits"].extend(
                        pos_log[: max_samples_per_head - len(hs["pos_logits"])]
                    )

        if len(hs["neg_logits"]) < max_samples_per_head:
            for i in range(min(batch_size, 4)):
                neg_inds = torch.randint(4, vocab_size, (50,), device=self.device)
                neg_log = logits_h[i, neg_inds].cpu().tolist()
                hs["neg_logits"].extend(
                    neg_log[: max_samples_per_head - len(hs["neg_logits"])]
                )

    def _update_sparse_head_stats(
        self,
        sparse_data: Tuple[torch.Tensor, torch.Tensor],
        logits_h: torch.Tensor,
        hs: Dict[str, Any],
        max_samples_per_head: int,
    ) -> None:
        pos_inds, pos_mask_t = sparse_data
        pos_inds = pos_inds.to(self.device)
        pos_mask_t = pos_mask_t.to(self.device)

        batch_size = pos_inds.size(0)
        vocab_size = logits_h.size(1)

        n_pos = pos_mask_t.sum().item()
        n_total = batch_size * vocab_size
        hs["p_sum"] += n_pos
        hs["count"] += n_total

        self._sample_sparse_logits(
            pos_inds, pos_mask_t, logits_h, hs, max_samples_per_head
        )

    def _compute_kc_metrics(
        self, acc: KCMetricsAccumulator, kc_vocab_size: int
    ) -> Dict[str, Any]:
        if self.is_distributed:
            dist.all_reduce(acc.top1_hist, op=dist.ReduceOp.SUM)
            scalars = torch.tensor(
                [
                    acc.n_samples,
                    acc.sum_entropy,
                    acc.sum_kl,
                    acc.sum_tv,
                    acc.sum_gap,
                    acc.sum_avg_prob,
                    acc.sum_act_dens,
                ],
                device=self.device,
            )
            dist.all_reduce(scalars, op=dist.ReduceOp.SUM)
            acc.n_samples = int(scalars[0].item())
            acc.sum_entropy = scalars[1].item()
            acc.sum_kl = scalars[2].item()
            acc.sum_tv = scalars[3].item()
            acc.sum_gap = scalars[4].item()
            acc.sum_avg_prob = scalars[5].item()
            acc.sum_act_dens = scalars[6].item()

        n_samples = max(1, acc.n_samples)

        uniq_kcs = int((acc.topk_hist > 0).sum().item())
        max_top1 = float(acc.top1_hist.max().item()) / n_samples

        result: Dict[str, Any] = {
            "n_samples": n_samples,
            "uniq_kcs": uniq_kcs,
            "max_top1": max_top1,
            "entropy_norm": acc.sum_entropy / n_samples,
            "kl_to_uniform": acc.sum_kl / n_samples,
            "tv_mean": acc.sum_tv / n_samples,
            "gap_mean": acc.sum_gap / n_samples,
            "avg_prob": acc.sum_avg_prob / n_samples,
            "act_dens": acc.sum_act_dens / n_samples,
            "kc_vocab_size": kc_vocab_size,
        }

        if is_main_process():
            for head_name, hs in acc.head_samples.items():
                p_true, auc, delta_bce = self._compute_head_metrics(hs)

                result[f"head_{head_name}_p_true"] = p_true
                result[f"head_{head_name}_auc"] = auc
                result[f"head_{head_name}_delta_bce"] = delta_bce

        return result

    def _compute_head_metrics(self, hs: Dict[str, Any]) -> Tuple[float, float, float]:
        p_true = hs["p_sum"] / max(1, hs["count"])
        auc = float("nan")
        delta_bce = float("nan")

        pos_l = hs["pos_logits"]
        neg_l = hs["neg_logits"]
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
        probe_loader: DataLoader[Dict[str, Any]],
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
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if k.startswith("input_ids_")
                }
                attention_mask = batch["attention_mask"].to(self.device)

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
        probe_loader: DataLoader[Dict[str, Any]],
        max_batches: int = 25,
        temperature: float = 1.5,
        tau_usage: float = 2.0,
    ) -> Dict[str, Any]:
        m = cast(
            StyleClassifierWithKC,
            self.model.module if self.is_distributed else self.model,
        )
        m.eval()

        config = KCProbeConfig(
            tau_usage=tau_usage,
            vocab_size=int(getattr(m.config, "kc_vocab_size", 1024)),
            topk=int(getattr(m.config, "kc_topk", 8)),
            target_specs=getattr(m.config, "kc_target_specs", {}),
            max_samples_per_head=2000,
        )

        probe_heads = ["lemma", "pos", "conjugated_form", "conjugated_type"]
        head_samples: Dict[str, Dict[str, Any]] = {
            h: {"pos_logits": [], "neg_logits": [], "p_sum": 0.0, "count": 0}
            for h in probe_heads
            if h in config.target_specs
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

    def _diagnose_kc_probe(self, probe_result: Dict[str, Any]) -> List[str]:
        recommendations: List[str] = []

        max_top1 = probe_result.get("max_top1", 0.0)
        entropy_norm = probe_result.get("entropy_norm", 1.0)
        uniq_kcs = probe_result.get("uniq_kcs", 0)
        kc_vocab_size = probe_result.get("kc_vocab_size", 1024)

        collapse_risk = max_top1 > 0.10 or entropy_norm < 0.85
        if collapse_risk:
            recommendations.append(
                f"⚠️ COLLAPSE RISK: maxTop1={max_top1:.3f} (want <0.10), entN={entropy_norm:.3f} (want >0.85). "
                "Try: reduce encoder_lr_factor (0.1→0.01) or freeze encoder for first 2 epochs."
            )

        usage_ratio = uniq_kcs / kc_vocab_size
        if usage_ratio < 0.5:
            recommendations.append(
                f"⚠️ LOW DIVERSITY: only {uniq_kcs}/{kc_vocab_size} KCs used ({usage_ratio:.1%}). "
                "Try: increase diversity_weight_thawed or lower temperature."
            )

        for head in ["lemma", "pos", "conjugated_form", "conjugated_type"]:
            auc = probe_result.get(f"head_{head}_auc", float("nan"))
            if not math.isnan(auc) and auc < 0.80:
                recommendations.append(
                    f"⚠️ QUALITY DROP ({head}): AUC={auc:.3f} (want >0.85). "
                    "Try: add KC auxiliary loss during STYLE or retrain KC decoders post-STYLE."
                )

        if not recommendations:
            recommendations.append("✅ KC health OK. No action needed.")

        if is_main_process():
            print(
                f"  KCProbe: uniq={probe_result.get('uniq_kcs', 0)}/{probe_result.get('kc_vocab_size', 0)} "
                f"maxTop1={probe_result.get('max_top1', 0):.3f} "
                f"entN={probe_result.get('entropy_norm', 0):.3f} "
                f"klU={probe_result.get('kl_to_uniform', 0):.3f} "
                f"tv={probe_result.get('tv_mean', 0):.3f} "
                f"gap={probe_result.get('gap_mean', 0):.3f} "
                f"prob={probe_result.get('avg_prob', 0):.2f} "
                f"dens={probe_result.get('act_dens', 0):.4f}"
            )
            for head in ["lemma", "pos", "conjugated_form", "conjugated_type"]:
                auc = probe_result.get(f"head_{head}_auc", float("nan"))
                delta = probe_result.get(f"head_{head}_delta_bce", float("nan"))
                if not math.isnan(auc):
                    print(f"    {head}: AUC={auc:.3f} ΔBCE={delta:+.4f}")

        return recommendations
