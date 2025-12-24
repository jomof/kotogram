"""Core training logic and model extensions for style classification."""

import os
import sys
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple, cast

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

GENDER_LOSS_WEIGHT = 10.0


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
    ) -> Dict[str, Any]:
        pooled = self._get_pooled_output(field_inputs, attention_mask)
        kc_logits = self.kc_head(pooled)
        cur_temp = getattr(self.config, "kc_temperature", 1.0)
        kc_probs = torch.sigmoid(kc_logits / cur_temp)

        # Sparsity: Top-K Selection
        k = getattr(self.config, "kc_topk", 8)
        topk_vals, topk_inds = torch.topk(kc_probs, k, dim=-1)

        # Normalize so each sample distributes 1.0 mass across KCs
        topk_sum = topk_vals.sum(dim=-1, keepdim=True) + 1e-9
        topk_vals = topk_vals / topk_sum

        # Create sparse activation (everything else zero)
        # We start with zeros and scatter the top-k values back
        sparse_activations = torch.zeros_like(kc_probs)
        sparse_activations.scatter_(1, topk_inds, topk_vals)

        target_logits = self.kc_decoders(sparse_activations)

        return {
            "kc_logits": kc_logits,
            "kc_probs": kc_probs,
            "sparse_activations": sparse_activations,
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
        }

    def train_epoch(self, verbose: bool = True) -> Tuple[float, Dict[str, float]]:
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
                avg_l = total_loss / n_batches
                progress = (batch_idx + 1) / total_batches
                bar = (
                    "=" * int(30 * progress) + ">" + "." * (30 - int(30 * progress) - 1)
                )
                sys.stdout.write(
                    f"\r  [{bar}] {batch_idx + 1}/{total_batches} loss={avg_l:.4f}"
                )
                sys.stdout.flush()

        if verbose and is_main_process():
            sys.stdout.write("\n")
            sys.stdout.flush()
        return total_loss / n_batches, {
            f: loss_val / n_batches for f, loss_val in field_losses.items()
        }

    def train(
        self, epochs: Optional[int] = None, verbose: bool = True
    ) -> Dict[str, Any]:
        actual_epochs = epochs or self.config.epochs
        for epoch in range(actual_epochs):
            if self.is_distributed:
                cast(DistributedSampler, self.sampler).set_epoch(epoch)
            if verbose and is_main_process():
                print(f"Epoch {epoch + 1}/{actual_epochs}")
            mlm_loss, fields = self.train_epoch(verbose=verbose)
            self.history["mlm_loss"].append(mlm_loss)
            for f, v in fields.items():
                self.history["field_losses"][f].append(v)
            if verbose and is_main_process():
                print(f"  MLM Loss: {mlm_loss:.4f}")
                print(
                    f"  Field losses: {', '.join(f'{f}={loss_val:.3f}' for f, loss_val in fields.items())}"
                )
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
        self.kc_sparsity_weight = kc_config.get("sparsity_weight", 1e-3)
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
        vocab_sizes = dataset.tokenizer.get_vocab_sizes()

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
            # Pass vocab_sizes to collate_fn to correctly size multi-hot targets
            collate_fn=lambda b: collate_fn(
                b, pad_id, cast(Optional[int], max_seq_len), vocab_sizes=vocab_sizes
            ),
            pin_memory=(self.config.device == "cuda"),
            num_workers=(4 if self.config.device == "cuda" else 0),
        )

        self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)

        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.history: Dict[str, Any] = {
            "total_loss": [],
            "kc_loss": [],
            "kc_sparsity": [],
            "kc_losses": {},
        }

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
                    "lr": self.config.learning_rate * 0.1,
                }
            )
        self.optimizer = Adam(pg)

    def train_epoch(
        self, epoch: int = 0, verbose: bool = True
    ) -> Tuple[float, Dict[str, float], float]:
        # Handle freezing
        should_freeze = epoch < self.freeze_encoder_epochs
        self._create_optimizer(freeze_encoder=should_freeze)

        if verbose and is_main_process():
            print(f"KC Epoch (Encoder {'Frozen' if should_freeze else 'Thawed'})")

        self.model.train()
        total_loss, n_batches = 0.0, 0
        kc_losses: Dict[str, float] = {}
        total_sparsity = 0.0

        total_batches = len(self.data_loader)

        for batch_idx, batch in enumerate(self.data_loader):
            field_inputs = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k.startswith("input_ids_")
            }
            attention_mask = batch["attention_mask"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            device_type = (
                "cuda"
                if "cuda" in str(self.device)
                else ("mps" if "mps" in str(self.device) else "cpu")
            )

            with autocast(device_type=device_type, enabled=self.config.use_amp):
                # Forward pass
                outputs = self.model(
                    field_inputs, attention_mask=attention_mask, mode="kc"
                )
                target_logits = outputs["target_logits"]

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
                        targets = batch[target_key].to(self.device)
                        # logits: (B, V), targets: (B, V) float multi-hot
                        task_loss = self.bce_loss(logits, targets)
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
                        task_loss = self.bce_loss(logits, targets)
                        label_loss += task_loss
                        num_label += 1
                        batch_kc_losses[name] = task_loss.item()

                # Weighted Loss Combination
                combined_loss = torch.tensor(0.0, device=self.device)
                if num_struct > 0:
                    combined_loss += 0.7 * (structural_loss / num_struct)
                if num_label > 0:
                    combined_loss += 0.3 * (label_loss / num_label)

                # Sparsity Loss (on ACTUAL sparse activations)
                sparsity = outputs["sparse_activations"].mean()
                total_sparsity += sparsity.item()

                loss = (
                    combined_loss + self.kc_sparsity_weight * sparsity
                ) / self.config.grad_accum_steps

                if loss.item() == 0.0 and loss.requires_grad:
                    pass

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
            for k, v in batch_kc_losses.items():
                kc_losses[k] = kc_losses.get(k, 0.0) + v
            n_batches += 1

            if verbose and is_main_process():
                avg_l = total_loss / n_batches
                progress = (batch_idx + 1) / total_batches
                bar = (
                    "=" * int(30 * progress) + ">" + "." * (30 - int(30 * progress) - 1)
                )
                sys.stdout.write(
                    f"\r  [{bar}] {batch_idx + 1}/{total_batches} loss={avg_l:.4f}"
                )
                sys.stdout.flush()

        if verbose and is_main_process():
            sys.stdout.write("\n")
            sys.stdout.flush()

        avg_kc_losses = {k: v / n_batches for k, v in kc_losses.items()}
        avg_sparsity = total_sparsity / max(1, n_batches)
        return total_loss / n_batches, avg_kc_losses, avg_sparsity

    def train(
        self, epochs: Optional[int] = None, verbose: bool = True
    ) -> Dict[str, Any]:
        actual_epochs = epochs or self.config.epochs
        for epoch in range(actual_epochs):
            if self.is_distributed:
                cast(DistributedSampler, self.sampler).set_epoch(epoch)
            if verbose and is_main_process():
                print(f"Epoch {epoch + 1}/{actual_epochs}")
            total_loss, kc_losses, avg_sparsity = self.train_epoch(
                epoch=epoch, verbose=verbose
            )

            self.history["total_loss"].append(total_loss)
            self.history["kc_sparsity"].append(avg_sparsity)
            for k, v in kc_losses.items():
                if k not in self.history["kc_losses"]:
                    self.history["kc_losses"][k] = []
                self.history["kc_losses"][k].append(v)

            if verbose and is_main_process():
                print(
                    f"  KC Total Loss: {total_loss:.4f}, Sparsity: {avg_sparsity:.4f}"
                )
                # Print top 5 contributors
                top_losses = sorted(
                    kc_losses.items(), key=lambda x: x[1], reverse=True
                )[:5]
                loss_str = ", ".join(f"{k}={v:.3f}" for k, v in top_losses)
                print(f"  Top losses: {loss_str}")

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
        self.history: Dict[str, List[float]] = {
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
            ]
        }

    def train_epoch(
        self, verbose: bool = True
    ) -> Tuple[float, float, float, float, float]:
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
                avg_l = t_loss / n
                progress = (batch_idx + 1) / total_batches
                bar = (
                    "=" * int(30 * progress) + ">" + "." * (30 - int(30 * progress) - 1)
                )
                sys.stdout.write(
                    f"\r  [{bar}] {batch_idx + 1}/{total_batches} loss={avg_l:.4f}"
                )
                sys.stdout.flush()
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
    ) -> Dict[str, List[float]]:
        for epoch in range(self.start_epoch, self.config.epochs):
            if verbose:
                print(f"Epoch {epoch + 1}/{self.config.epochs}")
            tl, tfl, tgl, tgraml, trl = self.train_epoch(verbose=verbose)
            eval_res = self.evaluate()
            self.scheduler.step(eval_res["loss"])

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

            if verbose:
                print(f"  Train Loss: {tl:.4f}  Val Loss: {eval_res['loss']:.4f}")
                print(
                    f"  Formality Acc: {eval_res['formality_accuracy'] * 100:.2f}%  Gram Acc: {eval_res['grammaticality_accuracy'] * 100:.2f}%"
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

        if self.best_state:
            self.model.load_state_dict(self.best_state, strict=False)
        return self.history

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
