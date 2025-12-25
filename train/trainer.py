"""Core training logic and model extensions for style classification."""

import math
import os
import sys
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from kotogram.model import (
    ModelConfig,
    StyleClassifier,
)
from kotogram.tokenizer import FEATURE_FIELDS, Tokenizer
from train.config import TrainerConfig
from train.dataset import StyleDataset, collate_fn
from train.io import save_checkpoint

from .display import (
    print_epoch_summary,
    print_kc_epoch_compact_summary,
    print_kc_first_batch_debug,
    print_kc_first_batch_summary,
    print_kc_usage_summary,
    print_phase_header,
    print_progress_bar,
)

GENDER_LOSS_WEIGHT = 10.0


class MLMHead(nn.Module):
    """Masked language modeling head for feature-based tokens."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.shared_dense = nn.Linear(config.d_model, config.d_model)
        self.shared_norm = nn.LayerNorm(config.d_model)

        self.decoders = nn.ModuleDict()
        for field_name in FEATURE_FIELDS:
            vocab_size = config.vocab_sizes.get(field_name, 100)
            self.decoders[field_name] = nn.Linear(config.d_model, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.shared_dense(hidden_states)
        x = F.gelu(x)
        x = self.shared_norm(x)
        return {field: decoder(x) for field, decoder in self.decoders.items()}


class KCDecoder(nn.Module):
    """Decoder for predicting sentence-level attributes from KC activations."""

    def __init__(self, kc_vocab_size: int, target_specs: Dict[str, int]):
        super().__init__()
        self.decoders = nn.ModuleDict()
        for name, vocab_size in target_specs.items():
            self.decoders[name] = nn.Linear(kc_vocab_size, vocab_size)

    def forward(self, kc_activations: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            name: decoder(kc_activations) for name, decoder in self.decoders.items()
        }


class StyleClassifierWithMLM(StyleClassifier):
    """Multi-task style classifier with MLM and KC pretraining support."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.mlm_head = MLMHead(config)
        if config.kc_enabled:
            self.kc_decoders = KCDecoder(config.kc_vocab_size, config.kc_target_specs)

    def forward(
        self,
        *args: Any,
        mode: str = "classification",
        **kwargs: Any,
    ) -> Any:
        if mode == "mlm":
            return self.forward_mlm(*args, **kwargs)
        if mode == "kc":
            return self.forward_kc(*args, **kwargs)
        return super().forward(*args, **kwargs)

    def forward_kc(
        self,
        field_inputs: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        gumbel_scale: Optional[float] = None,
    ) -> Dict[str, Any]:
        pooled = self._get_pooled_output(field_inputs, attention_mask)

        # Get raw and normalized logits
        if hasattr(self.kc_head, "forward_with_raw"):
            kc_logits_raw, kc_logits = self.kc_head.forward_with_raw(pooled)
        else:
            kc_logits = self.kc_head(pooled)
            kc_logits_raw = kc_logits

        cur_temp = (
            temperature
            if temperature is not None
            else getattr(self.config, "kc_temperature", 1.0)
        )

        # Apply Gumbel Noise for Top-K Selection (Training Only)
        # We use noisy logits for selection, but return clean logits for regularization
        logits_for_selection = kc_logits_raw
        if gumbel_scale is not None and gumbel_scale > 0 and self.training:
            u = torch.rand_like(kc_logits_raw)
            g = -torch.log(-torch.log(u + 1e-9) + 1e-9)
            logits_for_selection = kc_logits_raw + gumbel_scale * g

        # Compute probs from (possibly noisy) logits
        kc_probs = torch.sigmoid(logits_for_selection / cur_temp)

        # Get top-k
        k = getattr(self.config, "kc_topk", 8)
        topk_vals, topk_inds = torch.topk(kc_probs, k, dim=-1)

        # Create sparse activation (everything else zero)
        # We start with zeros and scatter the top-k values back
        sparse_activations = torch.zeros_like(kc_probs)
        sparse_activations.scatter_(1, topk_inds, topk_vals)

        target_logits = self.kc_decoders(sparse_activations)

        return {
            "kc_logits": kc_logits,
            "kc_logits_raw": kc_logits_raw,  # Clean logits for usage reg
            "kc_probs": kc_probs,  # Probabilities used for selection
            "sparse_activations": sparse_activations,
            "topk_vals": topk_vals,
            "topk_inds": topk_inds,
            "target_logits": target_logits,
        }

    def forward_mlm(
        self,
        field_inputs: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoder_output = self.get_encoder_output(field_inputs, attention_mask)
        return cast(Dict[str, torch.Tensor], self.mlm_head(encoder_output))

    def reset_classifier(self) -> None:
        """Reinitialize all classifier head weights."""
        for classifier in [
            self.formality_value_head,
            self.formality_pragmatic_head,
            self.gender_value_head,
            self.gender_pragmatic_head,
            self.grammaticality_classifier,
            self.register_classifier,
        ]:
            if isinstance(classifier, nn.Module):
                for module in classifier.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)


def setup_distributed() -> Tuple[int, int, int]:
    """Initialize distributed training if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                device_id=torch.device(f"cuda:{local_rank}"),
                timeout=timedelta(minutes=60),
            )
            print(f"Distributed init: Rank {rank}/{world_size} (Local {local_rank})")
            return rank, world_size, local_rank

    return 0, 1, 0


def is_main_process() -> bool:
    """Check if we are on the main process (rank 0)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def create_mlm_batch(
    batch: Dict[str, torch.Tensor],
    mask_prob: float = 0.15,
    mask_token_id: int = 3,
    vocab_sizes: Optional[Dict[str, int]] = None,
    special_token_ids: Optional[List[int]] = None,
) -> Dict[str, torch.Tensor]:
    """Create masked language modeling batch for feature-based tokens."""
    special_token_ids = special_token_ids or [0, 1, 2, 3]
    vocab_sizes = vocab_sizes or {}
    HIDDEN_FIELDS = ["surface", "lemma"]
    primary_field = "pos"
    primary_ids = batch[f"input_ids_{primary_field}"].clone()

    maskable = batch["attention_mask"].bool()
    for special_id in special_token_ids:
        maskable &= primary_ids != special_id

    probs = torch.rand_like(primary_ids.float())
    mask = maskable & (probs < mask_prob)
    mask_token_positions = mask & (probs < mask_prob * 0.8)
    random_token_positions = (
        mask & (probs >= mask_prob * 0.8) & (probs < mask_prob * 0.9)
    )

    result = {"attention_mask": batch["attention_mask"]}
    for field in FEATURE_FIELDS:
        field_ids = batch[f"input_ids_{field}"].clone()
        mlm_labels = torch.full_like(field_ids, -100)
        if field in HIDDEN_FIELDS:
            active_tokens = batch["attention_mask"].bool()
            field_ids[active_tokens] = mask_token_id
        else:
            mlm_labels[mask] = field_ids[mask]
            field_ids[mask_token_positions] = mask_token_id
            field_vocab_size = vocab_sizes.get(field)
            if field_vocab_size:
                num_random = int(random_token_positions.sum().item())
                low, high = len(special_token_ids), field_vocab_size
                if num_random > 0 and high > low:
                    field_ids[random_token_positions] = torch.randint(
                        low, high, (num_random,)
                    )
        result[f"mlm_labels_{field}"] = mlm_labels
        result[f"input_ids_{field}"] = field_ids
    return result


def create_kc_batch(
    batch: Dict[str, torch.Tensor],
    tokenizer: Tokenizer,
    target_specs: Dict[str, int],
) -> Dict[str, torch.Tensor]:
    """Create multi-hot target batches for Knowledge Component (KC) training."""
    result = {}
    for name, vocab_size in target_specs.items():
        input_key = f"input_ids_{name}"
        if input_key not in batch:
            continue
        ids = batch[input_key]
        multi_hot = torch.zeros((ids.size(0), vocab_size), device=ids.device)
        mask = batch["attention_mask"].bool()
        for i in range(ids.size(0)):
            unique_ids = torch.unique(ids[i, mask[i]])
            unique_ids = unique_ids[unique_ids >= 4]  # Skip special tokens
            if len(unique_ids) > 0:
                multi_hot[i, unique_ids] = 1.0
        result[f"kc_targets_{name}"] = multi_hot
    return result


class MLMTrainer:
    """Trainer for self-supervised MLM pretraining."""

    def __init__(
        self,
        model: StyleClassifierWithMLM,
        dataset: StyleDataset,
        config: Optional[TrainerConfig] = None,
        mask_prob: float = 0.15,
    ):
        ngrammatic = [s for s in dataset.samples if s.grammaticality_label == 0]
        if ngrammatic:
            dataset = StyleDataset(
                [s for s in dataset.samples if s.grammaticality_label == 1],
                dataset.tokenizer,
            )

        self.model = model
        self.dataset = dataset
        self.config = config or TrainerConfig()
        self.mask_prob = mask_prob

        if self.config.world_size > 1:
            self.device = torch.device("cuda", self.config.local_rank)
            self.is_distributed = True
        else:
            self.device = torch.device(self.config.device)
            self.is_distributed = False

        self.model.to(self.device)
        if self.is_distributed:
            self.model = cast(
                StyleClassifierWithMLM,
                DDP(
                    self.model,
                    device_ids=[self.config.local_rank],
                    output_device=self.config.local_rank,
                    find_unused_parameters=True,
                ),
            )

        device_type = (
            "cuda"
            if "cuda" in str(self.device)
            else ("mps" if "mps" in str(self.device) else "cpu")
        )
        self.scaler = GradScaler(device=device_type, enabled=self.config.use_amp)

        pad_id = dataset.tokenizer.pad_id
        max_seq_len = getattr(self.model, "module", self.model).config.max_seq_len

        self.sampler: Optional[DistributedSampler] = (
            DistributedSampler(
                dataset,
                num_replicas=self.config.world_size,
                rank=dist.get_rank(),
                shuffle=True,
            )
            if self.is_distributed
            else None
        )
        self.data_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(self.sampler is None),
            sampler=self.sampler,
            collate_fn=lambda b: collate_fn(
                b, pad_id, cast(Optional[int], max_seq_len)
            ),
            pin_memory=(self.config.device == "cuda"),
            num_workers=(4 if self.config.device == "cuda" else 0),
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.vocab_sizes = dataset.tokenizer.get_vocab_sizes()
        self.field_weights = {f: 1.0 for f in FEATURE_FIELDS}
        self.history: Dict[str, Any] = {
            "mlm_loss": [],
            "field_losses": {f: [] for f in FEATURE_FIELDS},
            "sentence_count": [],
        }

    def train_epoch(self, verbose: bool = True) -> Tuple[float, Dict[str, float]]:
        if verbose and is_main_process():
            print_phase_header("MLM")

        self.model.train()
        total_loss, n_batches = 0.0, 0
        field_losses = {f: 0.0 for f in FEATURE_FIELDS}
        total_batches = len(self.data_loader)

        for batch_idx, batch in enumerate(self.data_loader):
            mlm_batch = create_mlm_batch(
                batch,
                mask_prob=self.mask_prob,
                mask_token_id=self.dataset.tokenizer.mask_id,
                vocab_sizes=self.vocab_sizes,
            )
            field_inputs = {
                k: v.to(self.device)
                for k, v in mlm_batch.items()
                if k.startswith("input_ids_")
            }
            attention_mask = mlm_batch["attention_mask"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            device_type = (
                "cuda"
                if "cuda" in str(self.device)
                else ("mps" if "mps" in str(self.device) else "cpu")
            )

            with autocast(device_type=device_type, enabled=self.config.use_amp):
                mlm_logits_dict = self.model(
                    field_inputs, attention_mask=attention_mask, mode="mlm"
                )
                batch_loss, valid_fields_count = (
                    torch.tensor(0.0, device=self.device),
                    0,
                )
                for f in FEATURE_FIELDS:
                    logits = mlm_logits_dict[f]
                    labels = mlm_batch[f"mlm_labels_{f}"].to(self.device)
                    if (labels != -100).sum() == 0:
                        continue
                    f_loss = self.criterion(
                        logits.view(-1, logits.size(-1)), labels.view(-1)
                    )
                    if torch.isnan(f_loss):
                        continue
                    batch_loss += self.field_weights[f] * f_loss
                    field_losses[f] += f_loss.item()
                    valid_fields_count += 1
                loss = (
                    (batch_loss / valid_fields_count)
                    if valid_fields_count > 0
                    else torch.tensor(0.0, device=self.device, requires_grad=True)
                )
                loss = loss / self.config.grad_accum_steps

            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * self.config.grad_accum_steps
            n_batches += 1

            if verbose and is_main_process():
                print_progress_bar(batch_idx, total_batches, total_loss / n_batches)

        if verbose and is_main_process():
            sys.stdout.write("\n")
            sys.stdout.flush()
        return total_loss / n_batches, {
            f: loss_val / n_batches for f, loss_val in field_losses.items()
        }

    def train(
        self,
        epochs: Optional[int] = None,
        verbose: bool = True,
        on_epoch_end: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        actual_epochs = epochs or self.config.epochs
        for epoch in range(actual_epochs):
            if self.is_distributed:
                cast(DistributedSampler, self.sampler).set_epoch(epoch)
            mlm_loss, fields = self.train_epoch(verbose=verbose)
            self.history["mlm_loss"].append(mlm_loss)
            for f, v in fields.items():
                self.history["field_losses"][f].append(v)
            if verbose and is_main_process():
                print_epoch_summary(
                    epoch + 1,
                    actual_epochs,
                    {"MLM Loss": mlm_loss},
                    {
                        f: v
                        for f, v in fields.items()
                        if v > 0.0001  # Only show non-trivial losses to declutter
                    },
                    phase="MLM",
                )
            self.history["sentence_count"].append(len(self.dataset.samples))
            if on_epoch_end and is_main_process():
                on_epoch_end(self.history)
        return self.history


class KCTrainer:
    """Trainer for Knowledge Component (KC) learning."""

    def __init__(
        self,
        model: StyleClassifierWithMLM,
        dataset: StyleDataset,
        config: Optional[TrainerConfig] = None,
        kc_config: Optional[Dict[str, Any]] = None,
    ):
        # Filter out agrammatic samples (KC training should only see valid grammar)
        if any(s.grammaticality_label == 0 for s in dataset.samples):
            valid_samples = [s for s in dataset.samples if s.grammaticality_label == 1]
            dataset = StyleDataset(valid_samples, dataset.tokenizer)

        self.model = model
        self.dataset = dataset
        self.config = config or TrainerConfig()

        kc_config = kc_config or {}
        self.kc_sparsity_weight = kc_config.get("sparsity_weight", 0.1)
        self.freeze_encoder_epochs = kc_config.get("freeze_encoder_epochs", 1)

        if self.config.world_size > 1:
            self.device = torch.device("cuda", self.config.local_rank)
            self.is_distributed = True
        else:
            self.device = torch.device(self.config.device)
            self.is_distributed = False

        self.model.to(self.device)
        if self.is_distributed:
            self.model = cast(
                StyleClassifierWithMLM,
                DDP(
                    self.model,
                    device_ids=[self.config.local_rank],
                    output_device=self.config.local_rank,
                    find_unused_parameters=True,
                ),
            )

        device_type = (
            "cuda"
            if "cuda" in str(self.device)
            else ("mps" if "mps" in str(self.device) else "cpu")
        )
        self.scaler = GradScaler(device=device_type, enabled=self.config.use_amp)

        pad_id = dataset.tokenizer.pad_id
        max_seq_len = getattr(self.model, "module", self.model).config.max_seq_len
        # vocab_sizes = dataset.tokenizer.get_vocab_sizes() # Unused

        self.sampler: Optional[DistributedSampler] = (
            DistributedSampler(
                dataset,
                num_replicas=self.config.world_size,
                rank=dist.get_rank(),
                shuffle=True,
            )
            if self.is_distributed
            else None
        )
        self.data_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(self.sampler is None),
            sampler=self.sampler,
            collate_fn=lambda b: collate_fn(
                b, pad_id, cast(Optional[int], max_seq_len)
            ),
            pin_memory=(self.config.device == "cuda"),
            num_workers=(4 if self.config.device == "cuda" else 0),
        )

        self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)

        # Loss functions
        self.default_bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.kc_pos_weight_cap = kc_config.get("pos_weight_cap", 50.0)
        self.kc_pos_weight_eps = kc_config.get("pos_weight_eps", 1e-6)

        # Diversity / Regularization
        self.kc_diversity_weight_frozen = float(kc_config.get("diversity_weight", 1e-3))
        self.kc_diversity_weight_thawed = float(
            kc_config.get("diversity_weight_thawed", 1e-1)
        )
        # self.kc_diversity_mode = "topk" (Implied)
        self.kc_diversity_eps = float(kc_config.get("diversity_eps", 1e-9))
        self.kc_diversity_warmup_epochs = int(
            kc_config.get("diversity_warmup_epochs", 0)
        )
        self.kc_sparsity_mode = "target_density"  # New default

        # Load Balancing
        self.kc_lb_weight_frozen = float(kc_config.get("lb_weight", 0.0))
        self.kc_lb_weight_thawed = float(kc_config.get("lb_weight_thawed", 2e-2))

        # Collapse Penalty
        self.kc_collapse_weight_thawed = float(
            kc_config.get("collapse_weight_thawed", 1.0)
        )

        # Temperature
        self.kc_temperature_frozen = float(
            getattr(
                getattr(self.model, "module", self.model).config, "kc_temperature", 1.0
            )
        )
        self.kc_temperature_thawed = float(kc_config.get("temperature_thawed", 1.8))

        # Logging configuration
        self.kc_log_level = kc_config.get("log_level", "minimal")
        self.kc_first_batch_debug_every = int(
            kc_config.get("first_batch_debug_every", 1)
        )
        # Default minimal behavior: only epoch 0
        self.kc_first_batch_debug_epochs = kc_config.get(
            "first_batch_debug_epochs", [0]
        )

        # Visibility flags
        self.kc_show_epoch_table = bool(kc_config.get("show_epoch_table", False))
        self.kc_show_step_checks = bool(kc_config.get("show_step_checks", False))
        self.kc_show_grad_norms = bool(kc_config.get("show_grad_norms", False))
        self.kc_show_amp_details = bool(kc_config.get("show_amp_details", False))

        # Override if global log_level is debug
        if self.kc_log_level == "debug":
            self.kc_show_epoch_table = True
            self.kc_show_step_checks = True
            self.kc_show_grad_norms = True
            self.kc_show_amp_details = True

        self.history: Dict[str, Any] = {
            "total_loss": [],
            "kc_loss": [],
            "kc_sparsity": [],
            "kc_losses": {},
            "avg_struct_loss": [],
            "avg_label_loss": [],
            "num_struct_heads_processed": [],
            "num_label_heads_processed": [],
            "avg_sparsity": [],
            "sentence_count": [],
            "first_batch_separation": [],
            "first_batch_grad_norms": [],
        }
        self._did_print_debug_for_epoch = -1

    def _init_structural_decoder_biases(self, num_batches: int = 10) -> None:
        raw = self.model.module if self.is_distributed else self.model
        m = cast("StyleClassifierWithMLM", raw)
        if not hasattr(m, "kc_decoders"):
            return

        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}

        # Scan a few batches to estimate base rates
        for i, batch in enumerate(self.data_loader):
            if i >= num_batches:
                break

            # MUST generate targets as they aren't pre-computed in the dataset
            kc_targets = create_kc_batch(
                batch=batch,
                tokenizer=self.dataset.tokenizer,
                target_specs=m.config.kc_target_specs,
            )

            for name in m.config.kc_target_specs.keys():
                key = f"kc_targets_{name}"
                if key not in kc_targets:
                    continue
                # Compute per-head, per-batch density
                t = kc_targets[key].float()
                p = t.mean().item()
                sums[name] = sums.get(name, 0.0) + p
                counts[name] = counts.get(name, 0) + 1

        # Apply logit initialization to biases
        # Round 6: DDP Sync
        if self.is_distributed:
            # Flatten to tensor for all_reduce
            names = sorted(sums.keys())
            # [sum_0, count_0, sum_1, count_1, ...]
            data = []
            for n in names:
                data.append(sums[n])
                data.append(float(counts[n]))

            t_data = torch.tensor(data, device=self.device)
            dist.all_reduce(t_data, op=dist.ReduceOp.SUM)

            # Unpack
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
        """Compute L2 norm of gradients for a given module."""
        total = 0.0
        for p in module.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            total += (g.float().norm(2).item()) ** 2
        return float(total**0.5)

    def _perform_optimizer_step(
        self,
        m: StyleClassifierWithMLM,
        verbose: bool,
        has_printed_step_check: bool,
        accum: int,
        is_flush: bool = False,
    ) -> bool:
        """Perform one optimizer step with unscaling and clipping. Returns True if skipped."""
        # 1. Snapshot w0 before step
        w0_before = 0.0
        if self.kc_show_step_checks:
            w0 = m.kc_head.linear.weight
            w0_before = w0.detach().flatten()[0].item()

        if is_main_process() and (not has_printed_step_check or is_flush):
            if self.kc_show_grad_norms:
                # 2. Compute grad norms BEFORE unscale/clip/step
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
                    f"  KC {phase} Grad Norms: kc_head={gn_kc:.6f} decoder={gn_dec:.6f} scale={self.scaler.get_scale():.1f}"
                    + (f" (flush_accum={accum})" if is_flush else "")
                )

        # --- Round 10: Guard - if any grad is non-finite, skip step and downscale ---
        # Unscale first so we check real magnitudes
        if self.config.use_amp:
            self.scaler.unscale_(self.optimizer)

        found_nonfinite = False
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    found_nonfinite = True
                    break
            if found_nonfinite:
                break

        if found_nonfinite:
            if is_main_process():
                print(
                    f"  KC Step Skipped: non-finite grad detected (scale={self.scaler.get_scale():.1f})"
                )
            self.optimizer.zero_grad(set_to_none=True)
            if self.config.use_amp:
                self.scaler.update(new_scale=max(1.0, self.scaler.get_scale() / 2.0))
            return True

        # Clip grads (already unscaled above when AMP enabled)
        if self.config.gradient_clip > 0:
            if not self.config.use_amp:
                # Only unscale if we haven't already (non-AMP path)
                pass
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

        # 3. Detect if GradScaler skips the step
        scale_before = float(self.scaler.get_scale())
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        scale_after = float(self.scaler.get_scale())
        skipped = scale_after < scale_before

        if is_main_process() and (not has_printed_step_check or is_flush):
            if self.kc_show_amp_details:
                print(
                    f"  KC AMP {'Flush ' if is_flush else ''}Step: scale {scale_before:.1f} -> {scale_after:.1f} skipped={skipped}"
                )

            if self.kc_show_step_checks:
                # 4. Compute parameter delta AFTER step
                w0 = m.kc_head.linear.weight
                w0_after = w0.detach().flatten()[0].item()
                print(
                    f"  KC {'Flush ' if is_flush else ''}Step Check: kc_head.w0 {w0_before:.6f} -> {w0_after:.6f} "
                    f"(delta={w0_after - w0_before:+.6f}, accum={accum}/{self.config.grad_accum_steps})"
                )

        self.optimizer.zero_grad(set_to_none=True)
        return skipped

    def _create_optimizer(self, freeze_encoder: bool) -> None:
        # Re-create optimizer to optionally freeze encoder
        raw = self.model.module if self.is_distributed else self.model
        m = cast(StyleClassifierWithMLM, raw)
        pg = [
            {
                "params": list(m.kc_head.parameters())
                + (
                    list(m.kc_decoders.parameters())
                    if hasattr(m, "kc_decoders")
                    else []
                ),
                "lr": self.config.learning_rate,
            }
        ]
        if not freeze_encoder:
            pg.append(
                {
                    "params": list(m.embedding.parameters())
                    + list(m.encoder.parameters()),
                    "lr": self.config.learning_rate * 0.01,
                }
            )
        self.optimizer = Adam(pg)

    def train_epoch(
        self, epoch: int = 0, verbose: bool = True
    ) -> Tuple[float, Dict[str, float], float, Dict[str, Any]]:
        # Handle freezing
        should_freeze = epoch < self.freeze_encoder_epochs
        self._create_optimizer(freeze_encoder=should_freeze)

        if verbose and is_main_process():
            print_phase_header(
                "KC", info="Encoder Frozen" if should_freeze else "Encoder Thawed"
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
        # first_batch_debug_printed = False  # Unused
        has_printed_step_check = False
        amp_skips = 0
        amp_scale_start = float(self.scaler.get_scale())
        opt_steps = 0
        flush_steps = 0
        pending_accum = 0
        did_any_backward = False

        # Epoch-level KC Usage Accumulators
        raw = self.model.module if self.is_distributed else self.model
        kc_vocab_size = int(cast(StyleClassifierWithMLM, raw).config.kc_vocab_size)
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

        # Zero gradients before starting the epoch loop
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(self.data_loader):
            raw = self.model.module if self.is_distributed else self.model
            m = cast(StyleClassifierWithMLM, raw)

            if batch_idx == 0:
                pass

            # Generate KC targets on-the-fly
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

            device_type = (
                "cuda"
                if "cuda" in str(self.device)
                else ("mps" if "mps" in str(self.device) else "cpu")
            )

            # Determine temperature for this sub-step (approximate)
            gumbel_scale = 0.0
            if epoch < self.freeze_encoder_epochs:
                t_val = self.kc_temperature_frozen
            else:
                t_val = self.kc_temperature_thawed
                # Training-time Gumbel annealing
                # Slower exploration anneal: keep noise higher longer to avoid early KC lock-in.
                # 0.6 -> 0.2 over thawed epochs
                epochs_remaining = max(
                    1, self.config.epochs - self.freeze_encoder_epochs
                )
                epoch_idx_thawed = max(0, epoch - self.freeze_encoder_epochs)
                ratio = min(1.0, epoch_idx_thawed / float(epochs_remaining))
                gumbel_scale = 0.6 * (1.0 - ratio) + 0.2 * ratio

            with autocast(device_type=device_type, enabled=self.config.use_amp):
                # Forward pass
                outputs = self.model(
                    field_inputs,
                    attention_mask=attention_mask,
                    mode="kc",
                    temperature=t_val,
                    gumbel_scale=gumbel_scale,
                )

                # Thawed Clamping (Optional but High Impact)
                # Training-time only tweak to limit gradient explosion from "winning" KCs
                if epoch >= self.freeze_encoder_epochs:
                    # Create local clamped copy
                    topk_vals_clamped = outputs["topk_vals"].clamp(max=0.85)

                    # Rebuild sparse activations for decoder targets ONLY
                    # We can't modify outputs["sparse_activations"] in place easily if it's needed for sparsity loss
                    # But we can re-generate target logits.
                    sparse_clamped = torch.zeros_like(outputs["kc_probs"])
                    sparse_clamped.scatter_(1, outputs["topk_inds"], topk_vals_clamped)

                    # Run decoders directly
                    if hasattr(m, "kc_decoders"):
                        outputs["target_logits"] = m.kc_decoders(sparse_clamped)

            # --- Accumulate KC Usage Stats ---
            topk_inds = outputs.get("topk_inds", None)
            topk_vals = outputs.get("topk_vals", None)

            if topk_inds is not None and topk_vals is not None:
                # Detach to avoid keeping graph
                inds_cpu = topk_inds.detach().to("cpu")  # (B, K)
                vals_cpu = topk_vals.detach().to("cpu")  # (B, K)

                B_sz = int(inds_cpu.size(0))
                kc_usage_total_samples += B_sz

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
                # ---------------------------------
                target_logits = outputs["target_logits"]

                if (
                    batch_idx == 0
                    and is_main_process()
                    and epoch != self._did_print_debug_for_epoch
                ):
                    self._did_print_debug_for_epoch = epoch
                    raw = self.model.module if self.is_distributed else self.model
                    m = cast(StyleClassifierWithMLM, raw)

                    # Condensed or Detailed Debug
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
                        # Prepare data for summary/debug
                        m = cast(StyleClassifierWithMLM, raw)

                        if self.kc_log_level == "debug":
                            # Full output
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
                            # Condensed output
                            # 1. Gather head stats similar to debug tool
                            # head_diagnostics = [] # Unused
                            # Priority heads
                            priority = [
                                "lemma",
                                "pos",
                                "conjugated_type",
                                "conjugated_form",
                            ]
                            # Gather others sorted by density
                            all_heads = sorted(outputs["target_logits"].keys())

                            # Helper to compute single-head stats
                            def get_head_stat(name: str) -> Dict[str, Any]:
                                logits = outputs["target_logits"][name]
                                target_key = f"kc_targets_{name}"
                                if target_key not in batch:
                                    return {}
                                t = batch[target_key].to(self.device).float()
                                with torch.no_grad():
                                    pos = t.sum()
                                    total = t.numel()
                                    p = (pos / (total + self.kc_pos_weight_eps)).clamp(
                                        min=self.kc_pos_weight_eps,
                                        max=1.0 - self.kc_pos_weight_eps,
                                    )
                                    pos_w = ((1.0 - p) / p).clamp(
                                        min=1.0, max=self.kc_pos_weight_cap
                                    )

                                    probs = torch.sigmoid(logits)
                                    p_avg = probs.mean().item()

                                    # AUC (subsampled)
                                    auc = 0.0
                                    pos_mask = t > 0.5
                                    neg_mask = ~pos_mask
                                    if pos_mask.any() and neg_mask.any():
                                        # Simple subsample for display speed
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
                                                    idx_p.numel(), device=self.device
                                                ),
                                                torch.zeros(
                                                    idx_n.numel(), device=self.device
                                                ),
                                            ]
                                        )

                                        # Rank AUC
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

                                    # Delta Loss
                                    bias_used = logits.mean().item()
                                    import torch.nn.functional as F

                                    pw = torch.tensor(pos_w.item(), device=self.device)
                                    hl = F.binary_cross_entropy_with_logits(
                                        logits, t, pos_weight=pw
                                    ).item()
                                    pl = F.binary_cross_entropy_with_logits(
                                        torch.full_like(logits, bias_used),
                                        t,
                                        pos_weight=pw,
                                    ).item()
                                    delta = hl - pl

                                return {
                                    "name": name,
                                    "p": p.item(),
                                    "pos_w": pos_w.item(),
                                    "p_avg": p_avg,
                                    "auc": auc,
                                    "delta": delta,
                                }

                            # Collect selected heads
                            selected_stats = []
                            seen = set()
                            # 1. Priority
                            for h in priority:
                                if h in all_heads:
                                    s = get_head_stat(h)
                                    if s:
                                        selected_stats.append(s)
                                        seen.add(h)

                            # 2. Rarest others
                            others = [h for h in all_heads if h not in seen]
                            # Sort by rough density (we need to compute it or guess, here we compute)
                            other_stats = []
                            for h in others:
                                s = get_head_stat(h)
                                if s:
                                    other_stats.append(s)

                            other_stats.sort(key=lambda x: str(x.get("p", 0)))
                            selected_stats.extend(other_stats[:2])  # Take 2 rarest

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

                            print_kc_first_batch_summary(kc_stats, selected_stats[:5])

                    # Compute separation metrics for storage
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

                # Compute losses for each target present in target_logits AND batch
                # Check for structural targets (bags, hashes)
                for name, logits in target_logits.items():
                    target_key = f"kc_targets_{name}"

                    if target_key in batch:
                        # Structural target (multi-hot)
                        targets = batch[target_key].to(self.device).float()
                        logits_f = logits.float()

                        # Round 6: Optimized Negative Sampling
                        B, V_f = logits_f.shape
                        if V_f > 256:
                            # Large Head: Sample 128 negatives
                            pos_mask = targets > 0.5
                            neg_count = 128

                            # Optim: Use smaller index tensor for scatter, avoid huge boolean mask if possible?
                            # Sticking to valid mask approach but ensuring cleaner code.
                            neg_inds = torch.randint(
                                0, V_f, (B, neg_count), device=self.device
                            )
                            mask = torch.zeros_like(logits_f, dtype=torch.bool)
                            mask.scatter_(1, neg_inds, True)
                            mask = mask | pos_mask

                            if mask.any():
                                task_loss = F.binary_cross_entropy_with_logits(
                                    logits_f[mask], targets[mask]
                                )
                            else:
                                task_loss = torch.tensor(
                                    0.0, device=self.device, requires_grad=True
                                )
                        else:
                            # Small Head: Full BCE
                            task_loss = F.binary_cross_entropy_with_logits(
                                logits_f, targets
                            )

                        structural_loss += task_loss
                        num_struct += 1
                        batch_kc_losses[name] = task_loss.item()

                    # Auxiliary Label Targets
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

                # Track running stats for epoch summary
                if num_struct > 0:
                    running_struct_loss += structural_loss.item()
                    running_num_struct_total += 1
                if num_label > 0:
                    running_label_loss += label_loss.item()
                    running_num_label_total += 1

                # Weighted Loss Combination
                combined_loss = torch.tensor(0.0, device=self.device)
                if num_struct > 0:
                    combined_loss += 0.7 * (structural_loss / num_struct)
                if num_label > 0:
                    combined_loss += 0.3 * (label_loss / num_label)

                # Select Diversity Weight based on epoch
                if epoch < self.freeze_encoder_epochs:
                    div_weight = self.kc_diversity_weight_frozen
                    lb_weight = self.kc_lb_weight_frozen
                else:
                    div_weight = self.kc_diversity_weight_thawed
                    lb_weight = self.kc_lb_weight_thawed

                # Diversity Regularization
                diversity_loss = torch.tensor(0.0, device=self.device)
                entropy_norm = torch.tensor(0.0, device=self.device)
                kl_to_uniform = torch.tensor(0.0, device=self.device)

                # Tracking scalars
                loss_div_val = 0.0
                loss_lb_val = 0.0
                loss_coll_val = 0.0

                if epoch >= self.kc_diversity_warmup_epochs:
                    # ------------------------------------------------------------------
                    # ROUND 5: OPINIONATED STABILITY FIXES
                    # ------------------------------------------------------------------
                    # Step 2: Regularize USAGE (Softmax), not Probabilities (Sigmoid)
                    logits_raw = outputs["kc_logits_raw"]

                    # Round 8: Stability clamp - prevents rare large logits from dominating
                    # the batch usage distribution. This only affects the regularizer path (q/p),
                    # not the forward selection itself.
                    logits_usage = logits_raw.clamp(min=-8.0, max=8.0)

                    tau_usage = 1.0 if epoch < self.freeze_encoder_epochs else 2.0

                    # q: (B, V) soft assignment distribution (sums to 1 per sample)
                    q = torch.softmax(logits_usage / tau_usage, dim=-1)

                    # p: (V,) global batch usage distribution (sums to 1)
                    p = q.mean(dim=0)

                    # Ensure sum to 1 (softmax does this, but floating point drift might occur)
                    p_sum = p.sum().clamp_min(self.kc_diversity_eps)
                    p = p / p_sum

                    # Differentiable Entropy / Diversity
                    # Maximize entropy of p => minimize neg_entropy
                    log_p = (p + self.kc_diversity_eps).log()
                    entropy = -(p * log_p).sum()
                    entropy_norm = entropy / math.log(kc_vocab_size)
                    diversity_loss = 1.0 - entropy_norm

                    # Differentiable Load Balance: KL(p || uniform)
                    kl_val = (p * (p.clamp_min(1e-9) * kc_vocab_size).log()).sum()
                    load_balance_loss = kl_val / math.log(kc_vocab_size)

                    # Step 3: Strict Collapse Penalty (Linear, Thawed Only)
                    p_max = p.max()

                    if epoch >= self.freeze_encoder_epochs:
                        # Threshold scale with V (approx 3x uniform)
                        thr = max(3.0 / max(1, kc_vocab_size), 0.002)

                        # Linear penalty (L1), not squared, to bite immediately
                        diff = (p_max - thr).clamp_min(0.0)

                        # Strong weight (order 1.0)
                        if self.kc_collapse_weight_thawed > 0:
                            # Use weight directly on linear penalty
                            collapse_penalty = diff
                            combined_loss += (
                                self.kc_collapse_weight_thawed * collapse_penalty
                            )
                            loss_coll_val = (
                                self.kc_collapse_weight_thawed * collapse_penalty
                            ).item()

                    # Apply Standard Regularizers
                    if div_weight > 0:
                        combined_loss += div_weight * diversity_loss
                        loss_div_val = (div_weight * diversity_loss).item()

                    if lb_weight > 0:
                        combined_loss += lb_weight * load_balance_loss
                        loss_lb_val = (lb_weight * load_balance_loss).item()

                    # Metrics for logging (Overwrite with differentiable versions)
                    kl_to_uniform = kl_val

                running_entropy_norm += entropy_norm.item()
                running_kl_to_uniform += kl_to_uniform.item()
                running_p_max += p_max.item() if ("p_max" in locals()) else 0.0

                # Sparsity Loss and New Metrics
                if (
                    self.kc_sparsity_weight > 0
                    and self.kc_sparsity_mode == "target_density"
                ):
                    # Avg probability (soft)
                    avg_prob = outputs["kc_probs"].mean()
                    # Activation density (hard, post-topk)
                    act_dens = (outputs["sparse_activations"] > 0).float().mean()

                    # Sparsity loss uses soft probabilities usually for differentiability
                    sparsity_term = avg_prob
                else:
                    avg_prob = outputs["kc_probs"].mean()
                    act_dens = outputs["sparse_activations"].mean()
                    sparsity_term = act_dens

                running_avg_prob += avg_prob.item()
                running_act_dens += act_dens.item()
                # Round 8 Fix 1: Track what we're actually penalizing for avg_sparsity output
                total_sparsity += float(sparsity_term.detach().item())
                running_sparsity += sparsity_term.item()

                # Round 8 Fix 5: Stage sparsity pressure - lighter early in thawed phase
                spar_w = self.kc_sparsity_weight
                if epoch >= self.freeze_encoder_epochs:
                    epoch_idx_thawed = max(0, epoch - self.freeze_encoder_epochs)
                    if epoch_idx_thawed < 3:
                        spar_w = 0.5 * self.kc_sparsity_weight

                loss = (
                    combined_loss + spar_w * sparsity_term
                ) / self.config.grad_accum_steps

                loss_spar_val = (spar_w * sparsity_term).item()

                # Update loss components dict for logging
                # Ensure we add key if missing
                current_epoch_comp = {
                    "base": combined_loss.item(),
                    "struct": structural_loss.item(),
                    "label": label_loss.item(),
                    "div": loss_div_val,
                    "lb": loss_lb_val,
                    "collapse": loss_coll_val,
                    "sparsity": loss_spar_val,
                }

                # Update loss components stats
                running_loss_components["base"] += current_epoch_comp["base"]
                running_loss_components["div"] += current_epoch_comp["div"]
                running_loss_components["lb"] += current_epoch_comp["lb"]
                running_loss_components["collapse"] += current_epoch_comp["collapse"]

                # Subtract regs from base to show pure base loss?
                # combined_loss includes regs. So base = combined - regs.
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

            # --- Round 10: NaN/Inf guard: never backprop a non-finite loss ---
            if not torch.isfinite(loss):
                if is_main_process():
                    kc_logits_raw = outputs.get("kc_logits_raw", None)
                    kc_probs = outputs.get("kc_probs", None)
                    msg = f"  [KC][NON-FINITE LOSS] epoch={epoch} batch={batch_idx} loss={loss.item()}"
                    if kc_logits_raw is not None:
                        msg += f" raw[min={kc_logits_raw.min().item():.3g} max={kc_logits_raw.max().item():.3g}]"
                    if kc_probs is not None:
                        msg += f" probs[min={kc_probs.min().item():.3g} max={kc_probs.max().item():.3g}]"
                    msg += f" scaler={self.scaler.get_scale():.1f}"
                    print(msg)

                # Clear grads and reduce scaler aggressively to recover
                self.optimizer.zero_grad(set_to_none=True)
                if self.config.use_amp:
                    self.scaler.update(
                        new_scale=max(1.0, self.scaler.get_scale() / 2.0)
                    )

                amp_skips += 1
                continue

            self.scaler.scale(loss).backward()
            did_any_backward = True
            pending_accum += 1

            # (Removed old mid-epoch inaccurate grad norm prints)

            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                is_skipped = self._perform_optimizer_step(
                    m, verbose, has_printed_step_check, pending_accum, is_flush=False
                )
                if is_skipped:
                    amp_skips += 1
                opt_steps += 1
                has_printed_step_check = True
                pending_accum = 0

            total_loss += loss.item() * self.config.grad_accum_steps
            for k, v in batch_kc_losses.items():
                kc_losses[k] = kc_losses.get(k, 0.0) + v
            n_batches += 1

            if verbose and is_main_process():
                print_progress_bar(batch_idx, total_batches, total_loss / n_batches)

        if did_any_backward and pending_accum > 0:
            raw = self.model.module if self.is_distributed else self.model
            m = cast(StyleClassifierWithMLM, raw)
            self._perform_optimizer_step(
                m, verbose, has_printed_step_check, pending_accum, is_flush=True
            )
            # Flush doesn't increment amp_skips usually, but we track steps
            flush_steps += 1
            has_printed_step_check = True

        if verbose and is_main_process():
            sys.stdout.write("\n")
            sys.stdout.flush()

        avg_kc_losses = {k: v / n_batches for k, v in kc_losses.items()}
        avg_sparsity = total_sparsity / max(1, n_batches)

        # KC Usage Stats Calculation (DDP Safe)
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
                    cast(StyleClassifierWithMLM, self.model.module).config, "kc_topk", 8
                )
            )
            if self.is_distributed
            else int(getattr(self.model.config, "kc_topk", 8))
        )
        tv_mean = kc_tv_sum / max(1, kc_usage_total_samples * k_val)
        gap_mean = kc_gap_sum / max(1, kc_gap_count)
        avg_entropy_norm = running_entropy_norm / max(1, n_batches)
        avg_kl_to_uniform = running_kl_to_uniform / max(1, n_batches)

        # Prepare histograms for printing
        N = 10
        topk_vals_hist, topk_idx_hist = torch.topk(topk_hist, k=min(N, kc_vocab_size))
        top1_vals_hist, top1_idx_hist = torch.topk(top1_hist, k=min(N, kc_vocab_size))

        topk_counts_list = []
        for i in range(len(topk_idx_hist)):
            topk_counts_list.append(
                (int(topk_idx_hist[i].item()), int(topk_vals_hist[i].item()))
            )

        top1_counts_list = []
        for i in range(len(top1_idx_hist)):
            top1_counts_list.append(
                (int(top1_idx_hist[i].item()), int(top1_vals_hist[i].item()))
            )

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
        }

        avg_loss_components = {
            k: v / max(1, n_batches) for k, v in running_loss_components.items()
        }

        if verbose and is_main_process() and not self.kc_show_epoch_table:
            if not self.kc_show_epoch_table:
                # Minimal mode summary
                top_losses = sorted(
                    avg_kc_losses.items(), key=lambda x: x[1], reverse=True
                )[:3]
                amp_stats = {
                    "skips": amp_skips,
                    "start": amp_scale_start,
                    "end": self.scaler.get_scale(),
                    "opt_steps": opt_steps,
                    "flush_steps": flush_steps,
                }

                # New Loss Breakdown
                weights_dict = {
                    "div": div_weight if "div_weight" in locals() else 0.0,
                    "lb": lb_weight if "lb_weight" in locals() else 0.0,
                    "collapse": self.kc_collapse_weight_thawed
                    if epoch >= self.freeze_encoder_epochs
                    else 0.0,
                }
                from train.display import (
                    print_kc_loss_breakdown,  # Lazy import to avoid circular issues if any (though display is usually fine)
                )

                print_kc_loss_breakdown(avg_loss_components, weights_dict)

                print_kc_epoch_compact_summary(
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
                print_kc_usage_summary(
                    uniq=uniq_kcs_epoch,
                    total=kc_usage_total_samples,
                    max_top1=max_top1,
                    tv_mean=tv_mean,
                    gap_mean=gap_mean,
                    topk_counts=topk_counts_list,
                    top1_counts=top1_counts_list,
                    k=k_val,
                )

        # Round 8 Fix 6: Compact health line (main process only)
        if verbose and is_main_process():
            print(
                f"  KC Health: maxTop1={max_top1:.3f} uniqKCs={uniq_kcs_epoch}/{kc_vocab_size} "
                f"avgProb={cast(float, epoch_stats['avg_prob']):.3f} actDens={cast(float, epoch_stats['act_dens']):.4f} "
                f"entN={avg_entropy_norm:.3f} klU={avg_kl_to_uniform:.3f}"
            )

        return total_loss / n_batches, avg_kc_losses, avg_sparsity, epoch_stats

    def train(
        self,
        epochs: Optional[int] = None,
        verbose: bool = True,
        on_epoch_end: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        actual_epochs = epochs or self.config.epochs

        # Initialize biases to empirical base rates before training
        self._init_structural_decoder_biases()

        for epoch in range(actual_epochs):
            if self.is_distributed:
                cast(DistributedSampler, self.sampler).set_epoch(epoch)
            total_loss, kc_losses, avg_sparsity, epoch_stats = self.train_epoch(
                epoch=epoch, verbose=verbose
            )

            self.history["total_loss"].append(total_loss)
            self.history["kc_sparsity"].append(avg_sparsity)
            self.history["avg_struct_loss"].append(epoch_stats["avg_struct_loss"])
            self.history["avg_label_loss"].append(epoch_stats["avg_label_loss"])
            self.history["num_struct_heads_processed"].append(
                epoch_stats["num_struct_heads_processed"]
            )
            self.history["num_label_heads_processed"].append(
                epoch_stats["num_label_heads_processed"]
            )
            self.history["avg_sparsity"].append(epoch_stats["avg_sparsity"])
            self.history["first_batch_separation"].append(
                epoch_stats["first_batch_separation"]
            )
            self.history["first_batch_grad_norms"].append(
                epoch_stats["first_batch_grad_norms"]
            )

            for k, v in kc_losses.items():
                if k not in self.history["kc_losses"]:
                    self.history["kc_losses"][k] = []
                self.history["kc_losses"][k].append(v)

            if verbose and is_main_process() and self.kc_show_epoch_table:
                # Top 5 contributors
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

            self.history["sentence_count"].append(len(self.dataset.samples))
            if on_epoch_end and is_main_process():
                on_epoch_end(self.history)

        return self.history


class Trainer:
    """Standard trainer for style classification with differential learning rates."""

    def __init__(
        self,
        model: StyleClassifier,
        train_dataset: StyleDataset,
        val_dataset: StyleDataset,
        config: Optional[TrainerConfig] = None,
        encoder_lr_factor: float = 0.1,
        support_dir: Optional[str] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or TrainerConfig()
        self.encoder_lr_factor = encoder_lr_factor
        self.support_dir = support_dir

        if self.config.world_size > 1:
            self.device = torch.device("cuda", self.config.local_rank)
            self.is_distributed = True
        else:
            self.device = torch.device(self.config.device)
            self.is_distributed = False

        self.model.to(self.device)
        if self.is_distributed:
            self.model = cast(
                StyleClassifierWithMLM,
                DDP(
                    self.model,
                    device_ids=[self.config.local_rank],
                    output_device=self.config.local_rank,
                    find_unused_parameters=True,
                ),
            )

        device_type = (
            "cuda"
            if "cuda" in str(self.device)
            else ("mps" if "mps" in str(self.device) else "cpu")
        )
        self.scaler = GradScaler(device=device_type, enabled=self.config.use_amp)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        pad_id = train_dataset.tokenizer.pad_id
        max_seq_len = getattr(self.model, "module", self.model).config.max_seq_len

        if self.is_distributed:
            self.train_sampler: Optional[DistributedSampler] = DistributedSampler(
                train_dataset,
                num_replicas=self.config.world_size,
                rank=dist.get_rank(),
                shuffle=True,
            )
            self.val_sampler: Optional[DistributedSampler] = DistributedSampler(
                val_dataset,
                num_replicas=self.config.world_size,
                rank=dist.get_rank(),
                shuffle=False,
            )
            t_shuffle, v_shuffle = False, False
        else:
            self.train_sampler, self.val_sampler = None, None
            t_shuffle, v_shuffle = True, False

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=t_shuffle,
            sampler=self.train_sampler,
            collate_fn=lambda b: collate_fn(
                b, pad_id, cast(Optional[int], max_seq_len)
            ),
            pin_memory=(self.config.device == "cuda"),
            num_workers=(4 if self.config.device == "cuda" else 0),
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=v_shuffle,
            sampler=self.val_sampler,
            collate_fn=lambda b: collate_fn(
                b, pad_id, cast(Optional[int], max_seq_len)
            ),
            pin_memory=(self.config.device == "cuda"),
            num_workers=(4 if self.config.device == "cuda" else 0),
        )

        if self.config.use_class_weights:
            self.formality_criterion = nn.CrossEntropyLoss(
                weight=train_dataset.get_formality_class_weights().to(self.device)
            )
            self.gender_pragmatic_criterion = nn.CrossEntropyLoss(
                weight=train_dataset.get_gender_class_weights().to(self.device)
            )
            self.grammaticality_criterion = nn.CrossEntropyLoss(
                weight=train_dataset.get_grammaticality_class_weights().to(self.device)
            )
            self.register_criterion = nn.BCEWithLogitsLoss()
        else:
            self.formality_criterion = nn.CrossEntropyLoss()
            self.gender_pragmatic_criterion = nn.CrossEntropyLoss()
            self.grammaticality_criterion = nn.CrossEntropyLoss()
            self.register_criterion = nn.BCEWithLogitsLoss()

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
                {"params": enc_p, "lr": self.config.learning_rate * encoder_lr_factor},
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
        self.history: Dict[str, Any] = {
            k: []
            for k in [
                "train_loss",
                "train_formality_loss",
                "train_gender_loss",
                "train_grammaticality_loss",
                "train_register_loss",
                "val_loss",
                "val_formality_loss",
                "val_gender_loss",
                "val_grammaticality_loss",
                "val_register_loss",
                "val_formality_accuracy",
                "val_formality_mse",
                "val_gender_pragmatic_accuracy",
                "val_gender_value_mse",
                "val_grammaticality_accuracy",
                "val_register_accuracy",
                "sentence_count",
            ]
        }

    def train_epoch(
        self, verbose: bool = True
    ) -> Tuple[float, float, float, float, float]:
        if verbose and is_main_process():
            print_phase_header("Style")

        self.model.train()
        t_loss: float = 0.0
        tf_loss: float = 0.0
        tg_loss: float = 0.0
        tgram_loss: float = 0.0
        tr_loss: float = 0.0
        n = 0
        total_batches = len(self.train_loader)
        if total_batches == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            field_inputs = {
                f"input_ids_{f}": batch[f"input_ids_{f}"].to(self.device)
                for f in FEATURE_FIELDS
            }
            attention_mask = batch["attention_mask"].to(self.device)
            f_val_targets, f_prag_targets = (
                batch["formality_value"].to(self.device),
                batch["formality_pragmatic"].to(self.device),
            )
            g_val_targets, g_prag_targets = (
                batch["gender_value"].to(self.device),
                batch["gender_pragmatic"].to(self.device),
            )
            gram_targets, reg_targets = (
                batch["grammaticality_labels"].to(self.device),
                batch["register_labels"].to(self.device),
            )

            self.optimizer.zero_grad(set_to_none=True)
            device_type = (
                "cuda"
                if "cuda" in str(self.device)
                else ("mps" if "mps" in str(self.device) else "cpu")
            )

            with autocast(device_type=device_type, enabled=self.config.use_amp):
                (
                    f_val_logits,
                    f_prag_logits,
                    g_val_logits,
                    g_prag_logits,
                    gram_logits,
                    reg_logits,
                ) = self.model(field_inputs, attention_mask)
                is_gram, is_f_prag, is_g_prag = (
                    gram_targets == 1,
                    f_prag_targets == 1,
                    g_prag_targets == 1,
                )
                is_valid_style = is_gram & is_f_prag & is_g_prag

                f_loss = self.formality_criterion(f_prag_logits, f_prag_targets) + (
                    F.mse_loss(
                        f_val_logits.squeeze(-1)[is_valid_style],
                        f_val_targets[is_valid_style],
                    )
                    if is_valid_style.any()
                    else 0
                )
                g_loss = self.gender_pragmatic_criterion(
                    g_prag_logits, g_prag_targets
                ) + (
                    F.mse_loss(
                        g_val_logits.squeeze(-1)[is_valid_style],
                        g_val_targets[is_valid_style],
                    )
                    * GENDER_LOSS_WEIGHT
                    if is_valid_style.any()
                    else 0
                )
                gram_loss = self.grammaticality_criterion(gram_logits, gram_targets)
                reg_loss = (
                    self.register_criterion(
                        reg_logits[is_valid_style], reg_targets[is_valid_style]
                    )
                    if is_valid_style.any()
                    else torch.tensor(0.0, device=self.device)
                )

                loss = (
                    self.config.formality_loss_weight * f_loss
                    + self.config.gender_loss_weight * g_loss
                    + self.config.grammaticality_loss_weight * gram_loss
                    + self.config.register_loss_weight * reg_loss
                ) / self.config.grad_accum_steps

            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            t_loss += loss.item() * self.config.grad_accum_steps
            tf_loss += f_loss.item()
            tg_loss += g_loss.item()
            tgram_loss += gram_loss.item()
            tr_loss += reg_loss.item()
            n += 1
            if (
                verbose
                and is_main_process()
                and ((batch_idx + 1) % 100 == 0 or (batch_idx + 1) == total_batches)
            ):
                print_progress_bar(batch_idx, total_batches, t_loss / n)
        if verbose and is_main_process():
            sys.stdout.write("\n")
        return t_loss / n, tf_loss / n, tg_loss / n, tgram_loss / n, tr_loss / n

    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation and return metrics and predictions."""
        self.model.eval()
        tl: float = 0.0
        tfl: float = 0.0
        tgl: float = 0.0
        tgraml: float = 0.0
        trl: float = 0.0
        n = 0
        all_f_prag_p, all_f_prag_l, all_f_val_p, all_f_val_l = [], [], [], []
        all_g_prag_p, all_g_prag_l, all_g_val_p, all_g_val_l = [], [], [], []
        all_gram_p, all_gram_l, all_reg_p, all_reg_l, all_valid = [], [], [], [], []
        all_sentences, all_kotograms = [], []

        for batch in self.val_loader:
            field_inputs = {
                f"input_ids_{f}": batch[f"input_ids_{f}"].to(self.device)
                for f in FEATURE_FIELDS
            }
            attention_mask = batch["attention_mask"].to(self.device)
            f_v_t, f_p_t = (
                batch["formality_value"].to(self.device),
                batch["formality_pragmatic"].to(self.device),
            )
            g_v_t, g_p_t = (
                batch["gender_value"].to(self.device),
                batch["gender_pragmatic"].to(self.device),
            )
            gram_t, reg_t = (
                batch["grammaticality_labels"].to(self.device),
                batch["register_labels"].to(self.device),
            )
            all_sentences.extend(batch.get("original_sentence", []))
            all_kotograms.extend(batch.get("kotogram", []))

            fv_l, fp_l, gv_l, gp_l, gram_l, r_l = self.model(
                field_inputs, attention_mask
            )
            is_valid = (gram_t == 1) & (f_p_t == 1) & (g_p_t == 1)

            f_loss = self.formality_criterion(fp_l, f_p_t) + (
                F.mse_loss(fv_l.squeeze(-1)[is_valid], f_v_t[is_valid])
                if is_valid.any()
                else 0
            )
            g_loss = self.gender_pragmatic_criterion(gp_l, g_p_t) + (
                F.mse_loss(gv_l.squeeze(-1)[is_valid], g_v_t[is_valid])
                * GENDER_LOSS_WEIGHT
                if is_valid.any()
                else 0
            )
            gram_loss = self.grammaticality_criterion(gram_l, gram_t)
            reg_loss = (
                self.register_criterion(r_l[is_valid], reg_t[is_valid])
                if is_valid.any()
                else torch.tensor(0.0, device=self.device)
            )

            tl += (
                self.config.formality_loss_weight * f_loss
                + self.config.gender_loss_weight * g_loss
                + self.config.grammaticality_loss_weight * gram_loss
                + self.config.register_loss_weight * reg_loss
            ).item()
            tfl += f_loss.item()
            tgl += g_loss.item()
            tgraml += gram_loss.item()
            trl += reg_loss.item()
            n += 1

            all_f_prag_p.extend(fp_l.argmax(-1).cpu().tolist())
            all_f_prag_l.extend(f_p_t.cpu().tolist())
            all_f_val_p.extend(fv_l.squeeze(-1).cpu().tolist())
            all_f_val_l.extend(f_v_t.cpu().tolist())
            all_g_prag_p.extend(gp_l.argmax(-1).cpu().tolist())
            all_g_prag_l.extend(g_p_t.cpu().tolist())
            all_g_val_p.extend(gv_l.squeeze(-1).cpu().tolist())
            all_g_val_l.extend(g_v_t.cpu().tolist())
            all_gram_p.extend(gram_l.argmax(-1).cpu().tolist())
            all_gram_l.extend(gram_t.cpu().tolist())
            all_reg_p.extend((torch.sigmoid(r_l) > 0.5).long().cpu().tolist())
            all_reg_l.extend(reg_t.long().cpu().tolist())
            all_valid.extend(is_valid.cpu().tolist())

        def acc(p: List[int], labels: List[int]) -> float:
            return (
                sum(x == y for x, y in zip(p, labels)) / len(labels) if labels else 0.0
            )

        valid_idxs = [i for i, v in enumerate(all_valid) if v]

        def mse(p: List[float], labels: List[float], ids: List[int]) -> float:
            return sum((p[i] - labels[i]) ** 2 for i in ids) / len(ids) if ids else 0.0

        def reg_acc(
            p: List[List[int]], labels: List[List[int]], ids: List[int]
        ) -> float:
            return (
                sum(all(p[i][j] == labels[i][j] for j in range(len(p[i]))) for i in ids)
                / len(ids)
                if ids
                else 0.0
            )

        return {
            "loss": tl / n,
            "formality_loss": tfl / n,
            "gender_loss": tgl / n,
            "grammaticality_loss": tgraml / n,
            "register_loss": trl / n,
            "formality_accuracy": acc(all_f_prag_p, all_f_prag_l),
            "formality_value_mse": mse(all_f_val_p, all_f_val_l, valid_idxs),
            "gender_pragmatic_accuracy": acc(all_g_prag_p, all_g_prag_l),
            "gender_value_mse": mse(all_g_val_p, all_g_val_l, valid_idxs),
            "grammaticality_accuracy": acc(all_gram_p, all_gram_l),
            "register_accuracy": reg_acc(all_reg_p, all_reg_l, valid_idxs),
            # Full results for report generation
            "formality_val_preds": all_f_val_p,
            "formality_val_labels": all_f_val_l,
            "formality_prag_preds": all_f_prag_p,
            "formality_prag_labels": all_f_prag_l,
            "gender_val_preds": all_g_val_p,
            "gender_val_labels": all_g_val_l,
            "gender_prag_preds": all_g_prag_p,
            "gender_prag_labels": all_g_prag_l,
            "grammaticality_preds": all_gram_p,
            "grammaticality_labels": all_gram_l,
            "register_preds": all_reg_p,
            "register_labels": all_reg_l,
            "sentences": all_sentences,
            "kotograms": all_kotograms,
        }

    def train(
        self,
        verbose: bool = True,
        checkpoint_dir: Optional[str] = None,
        checkpoint_args: Optional[Any] = None,
        model_config: Optional[ModelConfig] = None,
        on_epoch_end: Optional[Callable[[Dict[str, List[float]]], None]] = None,
    ) -> Dict[str, List[float]]:
        for epoch in range(self.start_epoch, self.config.epochs):
            tl, tfl, tgl, tgraml, trl = self.train_epoch(verbose=verbose)
            eval_res = self.evaluate()
            self.scheduler.step(eval_res["loss"])

            # KC Probe (Round 9): measure KC health during STYLE training
            kc_probe_result = None
            m = cast(
                StyleClassifier,
                self.model.module if self.is_distributed else self.model,
            )
            if getattr(m.config, "kc_enabled", False):
                probe_loader = self._build_kc_probe_loader(max_batches=25)
                if probe_loader is not None:
                    kc_probe_result = self.evaluate_kc_probe(probe_loader)
                    self._diagnose_kc_probe(kc_probe_result, verbose=verbose)

            for k, v in zip(
                [
                    "train_loss",
                    "train_formality_loss",
                    "train_gender_loss",
                    "train_grammaticality_loss",
                    "train_register_loss",
                ],
                [tl, tfl, tgl, tgraml, trl],
            ):
                self.history[k].append(v)
            for k, v in zip(
                [
                    "val_loss",
                    "val_formality_loss",
                    "val_gender_loss",
                    "val_grammaticality_loss",
                    "val_register_loss",
                ],
                [
                    eval_res["loss"],
                    eval_res["formality_loss"],
                    eval_res["gender_loss"],
                    eval_res["grammaticality_loss"],
                    eval_res["register_loss"],
                ],
            ):
                self.history[k].append(v)
            for k, mk in zip(
                [
                    "val_formality_accuracy",
                    "val_formality_mse",
                    "val_gender_pragmatic_accuracy",
                    "val_gender_value_mse",
                    "val_grammaticality_accuracy",
                    "val_register_accuracy",
                ],
                [
                    "formality_accuracy",
                    "formality_value_mse",
                    "gender_pragmatic_accuracy",
                    "gender_value_mse",
                    "grammaticality_accuracy",
                    "register_accuracy",
                ],
            ):
                self.history[k].append(eval_res[mk])

            self.history["sentence_count"].append(len(self.train_dataset.samples))

            if verbose:
                # Format metrics for nice display
                metrics = {
                    "Formality": {
                        "Train": tfl,
                        "Val": eval_res["formality_loss"],
                        "Acc": eval_res["formality_accuracy"],
                    },
                    "Gender": {
                        "Train": tgl,
                        "Val": eval_res["gender_loss"],
                        "Acc": eval_res["gender_pragmatic_accuracy"],
                    },
                    "Grammar": {
                        "Train": tgraml,
                        "Val": eval_res["grammaticality_loss"],
                        "Acc": eval_res["grammaticality_accuracy"],
                    },
                    "Register": {
                        "Train": trl,
                        "Val": eval_res["register_loss"],
                        "Acc": eval_res["register_accuracy"],
                    },
                }
                print_epoch_summary(
                    epoch + 1,
                    self.config.epochs,
                    {"Train Loss": tl, "Val Loss": eval_res["loss"]},
                    metrics,
                    phase="Style",
                )

            is_best = eval_res["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss, self.patience_counter = eval_res["loss"], 0
                self.best_state = {
                    k: cast(torch.Tensor, v.cpu().clone())
                    for k, v in self.model.state_dict().items()
                }
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            if checkpoint_dir and checkpoint_args and model_config:
                save_checkpoint(
                    checkpoint_dir,
                    self.model,
                    self.train_dataset.tokenizer,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    self.history,
                    self.best_val_loss,
                    self.patience_counter,
                    self.best_state,
                    checkpoint_args,
                    model_config,
                    is_best=is_best,
                )
            if self.is_distributed:
                dist.barrier()

            if on_epoch_end and is_main_process():
                on_epoch_end(self.history)

        if self.best_state:
            self.model.load_state_dict(self.best_state, strict=False)
        return self.history

    # =========================================================================
    # KC Degradation Probe (Round 9)
    # =========================================================================
    # Measures KC health during STYLE training without affecting gradients.
    # Provides actionable diagnostics for KC preservation vs style accuracy tradeoffs.
    # =========================================================================

    def _build_kc_probe_loader(
        self, max_batches: int = 25
    ) -> Optional[DataLoader[Dict[str, Any]]]:
        """Build a DataLoader for KC probe evaluation.

        Returns val_loader if model has KC enabled, else None.
        Filtering to grammatical samples is done during iteration.
        """
        m = cast(
            StyleClassifier, self.model.module if self.is_distributed else self.model
        )
        if not getattr(m.config, "kc_enabled", False):
            return None

        # Just return val_loader - filtering handled in evaluate_kc_probe
        return cast(DataLoader[Dict[str, Any]], self.val_loader)

    def evaluate_kc_probe(
        self,
        probe_loader: DataLoader[Dict[str, Any]],
        max_batches: int = 25,
        temperature: float = 1.5,
        tau_usage: float = 2.0,
    ) -> Dict[str, Any]:
        """Evaluate KC health metrics without affecting gradients.

        Returns dict with:
        - Usage: uniq_kcs, max_top1, entropy_norm, kl_to_uniform, tv_mean, gap_mean, avg_prob, act_dens
        - Structural (per head): p_true, auc, delta_bce
        - (If baseline available): drift metrics

        HOW TO INTERPRET:
        - uniq_kcs: Higher is better (KCs being used). Low = collapse.
        - max_top1: Lower is better (<0.10 good). High = one KC dominates.
        - entropy_norm: Higher is better (>0.85 good). Low = uneven usage.
        - kl_to_uniform: Lower is better (<0.5 good). High = uneven.
        - tv_mean: Mean top-k probability value.
        - gap_mean: Gap between top-1 and top-k (higher = more confident).
        - avg_prob: Mean sigmoid probability (>0.5 = "Gray Goo").
        - act_dens: Should be ~k/V (e.g., 8/1024 = 0.0078).
        - AUC: Higher is better (>0.85 good for structural heads).
        - delta_bce: Negative is better (model beats constant baseline).
        """
        m = cast(
            StyleClassifierWithMLM,
            self.model.module if self.is_distributed else self.model,
        )
        m.eval()

        kc_vocab_size = int(getattr(m.config, "kc_vocab_size", 1024))
        kc_topk = int(getattr(m.config, "kc_topk", 8))
        target_specs = getattr(m.config, "kc_target_specs", {})

        # Initialize accumulators
        topk_hist = torch.zeros(kc_vocab_size, device=self.device, dtype=torch.long)
        top1_hist = torch.zeros(kc_vocab_size, device=self.device, dtype=torch.long)

        n_samples = 0
        sum_entropy = 0.0
        sum_kl = 0.0
        sum_tv = 0.0
        sum_gap = 0.0
        sum_avg_prob = 0.0
        sum_act_dens = 0.0

        # Per-head AUC sampling (reservoir style)
        probe_heads = ["pos", "conjugated_form", "conjugated_type"]
        head_samples: Dict[str, Dict[str, Any]] = {
            h: {"pos_logits": [], "neg_logits": [], "p_sum": 0.0, "count": 0}
            for h in probe_heads
            if h in target_specs
        }
        max_samples_per_head = 2000

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

                # Forward KC (deterministic: gumbel_scale=0.0, eval mode)
                outputs = m(
                    field_inputs,
                    attention_mask=attention_mask,
                    mode="kc",
                    temperature=temperature,
                    gumbel_scale=0.0,
                )

                B = outputs["kc_probs"].shape[0]
                n_samples += B

                # Usage histograms
                topk_inds = outputs["topk_inds"]  # (B, k)
                for i in range(B):
                    for j in range(kc_topk):
                        idx = topk_inds[i, j].item()
                        topk_hist[idx] += 1
                        if j == 0:
                            top1_hist[idx] += 1

                # Entropy and KL from soft usage
                logits_raw = outputs["kc_logits_raw"]
                logits_clamped = logits_raw.clamp(min=-8.0, max=8.0)
                q = torch.softmax(logits_clamped / tau_usage, dim=-1)  # (B, V)
                p = q.mean(dim=0)  # (V,)
                p = p / p.sum().clamp_min(1e-9)

                eps = 1e-9
                log_p = (p + eps).log()
                entropy = -(p * log_p).sum()
                entropy_norm = entropy / math.log(kc_vocab_size)
                kl_to_uniform = (p * (log_p + math.log(kc_vocab_size))).sum()

                sum_entropy += entropy_norm.item() * B
                sum_kl += kl_to_uniform.item() * B

                # Top-k value stats
                topk_vals = outputs["topk_vals"]  # (B, k)
                sum_tv += topk_vals.mean().item() * B
                gap = topk_vals[:, 0] - topk_vals[:, -1]
                sum_gap += gap.mean().item() * B

                # Soft density stats
                sum_avg_prob += outputs["kc_probs"].mean().item() * B
                sum_act_dens += (
                    outputs["sparse_activations"] > 0
                ).float().mean().item() * B

                # Per-head AUC sampling
                if "target_logits" in outputs:
                    kc_targets = create_kc_batch(
                        batch, self.val_dataset.tokenizer, target_specs
                    )
                    for head_name in head_samples:
                        if head_name not in outputs["target_logits"]:
                            continue
                        logits_h = outputs["target_logits"][head_name]
                        target_key = f"kc_targets_{head_name}"
                        if target_key not in kc_targets:
                            continue
                        targets_h = kc_targets[target_key].to(self.device).float()

                        # Update head stats
                        hs = head_samples[head_name]
                        hs["p_sum"] += targets_h.sum().item()
                        hs["count"] += targets_h.numel()

                        # Sample pos and neg logits for AUC
                        pos_mask = targets_h > 0.5
                        neg_mask = ~pos_mask

                        if len(hs["pos_logits"]) < max_samples_per_head:
                            pos_logits = logits_h[pos_mask].cpu().tolist()
                            hs["pos_logits"].extend(
                                pos_logits[
                                    : max_samples_per_head - len(hs["pos_logits"])
                                ]
                            )
                        if len(hs["neg_logits"]) < max_samples_per_head:
                            neg_logits = logits_h[neg_mask].cpu().tolist()
                            hs["neg_logits"].extend(
                                neg_logits[
                                    : max_samples_per_head - len(hs["neg_logits"])
                                ]
                            )

        # DDP sync if needed
        if self.is_distributed:
            dist.all_reduce(topk_hist, op=dist.ReduceOp.SUM)
            dist.all_reduce(top1_hist, op=dist.ReduceOp.SUM)
            scalars = torch.tensor(
                [
                    n_samples,
                    sum_entropy,
                    sum_kl,
                    sum_tv,
                    sum_gap,
                    sum_avg_prob,
                    sum_act_dens,
                ],
                device=self.device,
            )
            dist.all_reduce(scalars, op=dist.ReduceOp.SUM)
            n_samples = int(scalars[0].item())
            sum_entropy = scalars[1].item()
            sum_kl = scalars[2].item()
            sum_tv = scalars[3].item()
            sum_gap = scalars[4].item()
            sum_avg_prob = scalars[5].item()
            sum_act_dens = scalars[6].item()

        # Compute final metrics
        uniq_kcs = int((topk_hist > 0).sum().item())
        max_top1 = float(top1_hist.max().item()) / max(1, n_samples)

        result: Dict[str, Any] = {
            "n_samples": n_samples,
            "uniq_kcs": uniq_kcs,
            "max_top1": max_top1,
            "entropy_norm": sum_entropy / max(1, n_samples),
            "kl_to_uniform": sum_kl / max(1, n_samples),
            "tv_mean": sum_tv / max(1, n_samples),
            "gap_mean": sum_gap / max(1, n_samples),
            "avg_prob": sum_avg_prob / max(1, n_samples),
            "act_dens": sum_act_dens / max(1, n_samples),
            "kc_vocab_size": kc_vocab_size,
        }

        # Per-head AUCs (rank0 only for simplicity)
        if is_main_process():
            for head_name, hs in head_samples.items():
                p_true = hs["p_sum"] / max(1, hs["count"])
                auc = float("nan")
                delta_bce = float("nan")

                pos_l = hs["pos_logits"]
                neg_l = hs["neg_logits"]
                if len(pos_l) > 0 and len(neg_l) > 0:
                    # Rank AUC
                    all_logits = pos_l + neg_l
                    all_labels = [1.0] * len(pos_l) + [0.0] * len(neg_l)
                    combined = sorted(zip(all_logits, all_labels), key=lambda x: x[0])
                    ranks = list(range(1, len(combined) + 1))
                    pos_rank_sum = sum(
                        r for r, (_, lbl) in zip(ranks, combined) if lbl > 0.5
                    )
                    n_pos = len(pos_l)
                    n_neg = len(neg_l)
                    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / max(
                        1, n_pos * n_neg
                    )

                    # Delta BCE vs constant baseline
                    import torch.nn.functional as F

                    logits_t = torch.tensor(all_logits, dtype=torch.float32)
                    labels_t = torch.tensor(all_labels, dtype=torch.float32)
                    model_bce = F.binary_cross_entropy_with_logits(
                        logits_t, labels_t
                    ).item()
                    baseline_logit = (
                        math.log(p_true / (1 - p_true)) if 0 < p_true < 1 else 0.0
                    )
                    baseline_bce = F.binary_cross_entropy_with_logits(
                        torch.full_like(logits_t, baseline_logit), labels_t
                    ).item()
                    delta_bce = model_bce - baseline_bce

                result[f"head_{head_name}_p_true"] = p_true
                result[f"head_{head_name}_auc"] = auc
                result[f"head_{head_name}_delta_bce"] = delta_bce

        return result

    def _diagnose_kc_probe(
        self, probe_result: Dict[str, Any], verbose: bool = True
    ) -> List[str]:
        """Diagnose KC degradation and return actionable recommendations.

        DIAGNOSTIC THRESHOLDS:
        - Collapse: max_top1 > 0.10 OR entropy_norm < 0.85
        - Quality drop: any head AUC < 0.80
        """
        recommendations: List[str] = []

        max_top1 = probe_result.get("max_top1", 0.0)
        entropy_norm = probe_result.get("entropy_norm", 1.0)
        uniq_kcs = probe_result.get("uniq_kcs", 0)
        kc_vocab_size = probe_result.get("kc_vocab_size", 1024)

        # Collapse detection
        collapse_risk = max_top1 > 0.10 or entropy_norm < 0.85
        if collapse_risk:
            recommendations.append(
                f"⚠️ COLLAPSE RISK: maxTop1={max_top1:.3f} (want <0.10), entN={entropy_norm:.3f} (want >0.85). "
                "Try: reduce encoder_lr_factor (0.1→0.01) or freeze encoder for first 2 epochs."
            )

        # Usage check
        usage_ratio = uniq_kcs / kc_vocab_size
        if usage_ratio < 0.5:
            recommendations.append(
                f"⚠️ LOW DIVERSITY: only {uniq_kcs}/{kc_vocab_size} KCs used ({usage_ratio:.1%}). "
                "Try: increase diversity_weight_thawed or lower temperature."
            )

        # Structural quality check
        for head in ["pos", "conjugated_form", "conjugated_type"]:
            auc = probe_result.get(f"head_{head}_auc", float("nan"))
            if not math.isnan(auc) and auc < 0.80:
                recommendations.append(
                    f"⚠️ QUALITY DROP ({head}): AUC={auc:.3f} (want >0.85). "
                    "Try: add KC auxiliary loss during STYLE or retrain KC decoders post-STYLE."
                )

        # Summary
        if not recommendations:
            recommendations.append("✅ KC health OK. No action needed.")

        # Print compact summary
        if verbose and is_main_process():
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
            for head in ["pos", "conjugated_form", "conjugated_type"]:
                auc = probe_result.get(f"head_{head}_auc", float("nan"))
                delta = probe_result.get(f"head_{head}_delta_bce", float("nan"))
                if not math.isnan(auc):
                    print(f"    {head}: AUC={auc:.3f} ΔBCE={delta:+.4f}")

        return recommendations

    def restore_from_checkpoint(
        self, checkpoint: Dict[str, Any], reset_optimizer: bool = False
    ) -> None:
        if not reset_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.history, self.best_val_loss, self.patience_counter, self.best_state = (
            checkpoint["history"],
            checkpoint["best_val_loss"],
            checkpoint["patience_counter"],
            checkpoint["best_state"],
        )
        self.start_epoch = checkpoint["epoch"] + 1
