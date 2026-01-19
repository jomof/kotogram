# pylint: disable=too-many-lines,not-callable,too-many-nested-blocks,duplicate-code
import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from kotogram.model import (
    InferenceClassifier,
)
from kotogram.tokenizer import ENCODER_FEATURE_FIELDS
from train.config import (
    DataLoaderConfig,
    TrainerConfig,
    _safe_configure_threads,
    configure_runtime_thread_limits,
)
from train.dataset import StyleDataset, collate_fn
from train.display import (
    RichTrainerProgressBar,
)
from train.io import (
    save_checkpoint,
    save_model,
)
from train.profile import Timer, get_profile_dir
from train.pytorch_utils import estimate_optimal_batch_size
from train.trainer_view import (
    BinaryMetric,
    ClassPopulation,
    GradientNorms,
    GrammaticalityMetric,
    StyleEpochStats,
    StyleWorstSample,
    TrainerDiagnosticsView,
    TrainerView,
)
from train.types import (
    EvaluationMetrics,
    TrainingBatch,
    TrainingHistory,
    TrainingLosses,
    TrainingMetrics,
    TrainingPredictions,
)
from train.worker import _worker_init_fn


def _acc(p: List[int], labels: List[int]) -> float:
    return sum(x == y for x, y in zip(p, labels)) / len(labels) if labels else 0.0


def _acc_per_class(p: List[int], labels: List[int], target_class: int) -> float:
    """Compute recall for a specific class (how many of that class were correctly predicted)."""
    class_preds = [pred for pred, lbl in zip(p, labels) if lbl == target_class]
    if not class_preds:
        return 0.0
    return sum(pred == target_class for pred in class_preds) / len(class_preds)


def _mse(p: List[float], labels: List[float], ids: List[int]) -> float:
    return sum((p[i] - labels[i]) ** 2 for i in ids) / len(ids) if ids else 0.0


def _reg_acc(p: List[List[int]], labels: List[List[int]], ids: List[int]) -> float:
    return (
        sum(all(p[i][j] == labels[i][j] for j in range(len(p[i]))) for i in ids)
        / len(ids)
        if ids
        else 0.0
    )


def _compute_head_grad_norms(model: Any) -> Dict[str, float]:
    """Compute L2 gradient norms for each classifier head.

    Returns dict with keys: formality, gender, grammaticality, encoder, pooler
    Note: register is handled by KC decoder, not a separate head
    """
    head_names = {
        "formality": ["formality_pragmatic_head"],
        "gender": ["gender_pragmatic_head"],
        "grammaticality": ["grammaticality_classifier"],
        "encoder": ["encoder"],
        "pooler": ["pooler"],  # Unified pooler (shared by KC and style)
    }

    norms: Dict[str, float] = {}
    for group_name, module_names in head_names.items():
        total_norm = 0.0
        for mod_name in module_names:
            if hasattr(model, mod_name):
                module = getattr(model, mod_name)
                for param in module.parameters():
                    if param.grad is not None:
                        total_norm += param.grad.data.norm(2).item() ** 2
        norms[group_name] = total_norm**0.5  # L2 norm
    return norms


class Trainer:
    # pylint: disable=too-many-locals,too-many-positional-arguments
    def __init__(
        self,
        model: InferenceClassifier,
        train_dataset: StyleDataset,
        val_dataset: StyleDataset,
        config: TrainerConfig,
        dl_config_train: DataLoaderConfig,
        dl_config_val: DataLoaderConfig,
        output_path: str,
        view: Optional[TrainerView] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.name = "style_model"
        self.output_path = output_path
        self.view: TrainerView = view if view is not None else TrainerDiagnosticsView()

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

        batch_size = self.config.batch_size
        if batch_size == -1:
            # Auto-tuning
            optimal_bs = estimate_optimal_batch_size(
                self.device, self.model.config, is_kc=False
            )
            self.view.on_auto_batch_size(optimal_bs, self.device)
            batch_size = optimal_bs

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
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
            batch_size=batch_size,
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

        mod = cast(InferenceClassifier, self.model)

        # Encoder params: unfrozen with lower LR to preserve KC pretraining
        enc_p = list(mod.embedding.parameters()) + list(mod.encoder.parameters())

        # Style-specific params: pooler + classifier heads (full LR)
        # Note: formality/gender value predictions come from KC decoder MSE pathway
        # Note: Register is handled by KC decoder, not included here
        style_p = (
            list(mod.pooler.parameters())
            + list(mod.formality_pragmatic_head.parameters())
            + list(mod.gender_pragmatic_head.parameters())
            + list(mod.grammaticality_classifier.parameters())
        )
        self.optimizer = Adam(
            [
                {
                    "params": enc_p,
                    "lr": self.config.learning_rate * 0.1,  # Fine-tune encoder
                },
                {"params": style_p, "lr": self.config.learning_rate},
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
        self.session_start_epoch: Optional[int] = None
        self.start_batch = 0

        _safe_configure_threads(self.config)

        self.history = TrainingHistory()

        # Gradient norm tracking per head (reset each epoch)
        self.last_epoch_grad_norms: Dict[str, float] = {}
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

    # Note: _masked_bce removed - register loss now handled by KC trainer

    def _unpack_training_batch(
        self, batch: TrainingBatch
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        field_inputs = {
            f"input_ids_{f}": batch.feature_inputs[f"input_ids_{f}"].to(self.device)
            for f in ENCODER_FEATURE_FIELDS
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # forward() now returns 3 outputs: formality, gender, grammaticality pragmatic heads
        # Register loss is handled by KC trainer
        (f_prag_l, g_prag_l, gram_l) = outputs

        f_loss = self.formality_criterion(f_prag_l, targets["f_prag"])

        g_loss = self.gender_pragmatic_criterion(g_prag_l, targets["g_prag"])

        gram_loss = self.grammaticality_criterion(gram_l, targets["gram"])

        # Register loss is zero here - it's handled by KC trainer
        reg_loss = torch.tensor(0.0, device=f_prag_l.device)

        return f_loss, g_loss, gram_loss, reg_loss

    def _compute_training_loss(
        self, outputs: Tuple[torch.Tensor, ...], targets: Dict[str, torch.Tensor]
    ) -> TrainingLosses:
        f_loss, g_loss, gram_loss, reg_loss = self._compute_component_losses(
            outputs, targets
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

    def _train_batch(self, batch: TrainingBatch, batch_idx: int) -> Dict[str, Any]:
        field_inputs, attention_mask, targets = self._unpack_training_batch(batch)

        if batch_idx % self.config.grad_accum_steps == 0:
            self.optimizer.zero_grad(set_to_none=True)

        outputs = self.model(field_inputs, attention_mask)
        losses = self._compute_training_loss(outputs, targets)
        loss = losses.loss

        loss.backward()

        # Compute gradient norms per head (before clipping/stepping)
        grad_norms = _compute_head_grad_norms(self.model)

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
            "grad_norms": grad_norms,
        }

    def train_epoch(self, epoch: int) -> Tuple[float, float, float, float, float]:
        # Use relative epoch from session start for freezing (warm-up)
        # If session_start_epoch is None (e.g. direct call), fall back to absolute
        base_epoch = (
            self.session_start_epoch if self.session_start_epoch is not None else 0
        )
        relative_epoch = max(0, epoch - base_epoch)

        should_freeze = relative_epoch < self.config.freeze_encoder_epochs

        # Update encoder LR (param group 0) based on freezing
        enc_lr = 0.0 if should_freeze else (self.config.learning_rate * 0.1)
        self.optimizer.param_groups[0]["lr"] = enc_lr

        # Set training mode with special handling for frozen epochs:
        # During frozen epochs, put encoder pipeline in eval mode to disable dropout
        # for deterministic outputs, while keeping classifier heads in train mode.
        if should_freeze:
            # Encoder pipeline: eval mode (disable dropout)
            self.model.embedding.eval()
            self.model.position_encoding.eval()
            self.model.encoder.eval()
            self.model.pooler.eval()
            # Classifier heads: train mode (keep dropout active for regularization)
            self.model.formality_pragmatic_head.train()
            self.model.gender_pragmatic_head.train()
            self.model.grammaticality_classifier.train()
            # Note: register_classifier removed, register handled by KC decoder
        else:
            self.model.train()
        self.view.on_epoch_start(epoch, self.config.epochs, should_freeze)

        metrics = TrainingMetrics()

        # Gradient norm accumulators
        grad_norm_sums: Dict[str, float] = {}
        grad_norm_count = 0

        total_batches = len(self.train_loader)
        if total_batches == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        pbar = None
        current_loss_val = None
        pbar_desc = f"Style Epoch {epoch + 1}/{self.config.epochs}"
        if should_freeze:
            pbar_desc += " (Encoder Frozen)"

        pbar = RichTrainerProgressBar(
            desc=pbar_desc,
            total_steps=total_batches,
            batch_size=self.train_loader.batch_size or 1,
        )
        self.view.on_progress_init(pbar_desc, total_batches)

        try:
            self.train_timer_data.start()

            for batch_idx, batch in enumerate(self.train_loader):
                if not hasattr(batch, "feature_inputs"):
                    raise ValueError(f"Batch {batch_idx} missing feature_inputs")
                if batch_idx < self.start_batch:
                    if pbar:
                        pbar.update(batch_idx, loss=0.0)
                    self.view.on_progress_update(batch_idx, 0.0, total_batches)
                    continue

                self.train_timer_data.stop(epoch=epoch, batch=batch_idx)
                self.train_timer_compute.start()

                batch_result = self._train_batch(batch, batch_idx)
                metrics.update(batch_result)

                # Accumulate gradient norms
                grad_norms = batch_result.get("grad_norms", {})
                for key, val in grad_norms.items():
                    grad_norm_sums[key] = grad_norm_sums.get(key, 0.0) + val
                grad_norm_count += 1

                if pbar:
                    current_loss_val = metrics.get_avg_loss()
                    pbar.update(batch_idx, loss=current_loss_val or 0.0)

                self.view.on_progress_update(
                    batch_idx, current_loss_val or 0.0, total_batches
                )

                self.train_timer_compute.stop(epoch=epoch, batch=batch_idx)
                self.train_timer_data.start()

        finally:
            if pbar:
                pbar.stop()
            self.view.on_progress_stop()

        self.start_batch = 0
        self.train_timer_data.stop()
        self.view.on_line_flush()

        # Compute average gradient norms for epoch
        if grad_norm_count > 0:
            self.last_epoch_grad_norms = {
                k: v / grad_norm_count for k, v in grad_norm_sums.items()
            }

        return metrics.average()

    def _extract_predictions(
        self, outputs: Tuple[torch.Tensor, ...], targets: Dict[str, torch.Tensor]
    ) -> TrainingPredictions:
        # forward() now returns 3 outputs: formality, gender, grammaticality pragmatic heads
        # Register predictions are handled by KC decoder during evaluation
        (f_p_l, g_p_l, gram_l) = outputs

        batch_size = f_p_l.size(0)
        # Use zeros for value predictions (KC trainer tracks MSE)
        zeros_list = [0.0] * batch_size
        # Use zeros for register predictions (KC decoder handles this)
        reg_zeros = [[0] * 14 for _ in range(batch_size)]

        return TrainingPredictions(
            f_prag_p=f_p_l.argmax(-1).cpu().tolist(),
            f_prag_l=targets["f_prag"].cpu().tolist(),
            f_val_p=zeros_list,
            f_val_l=targets["f_val"].cpu().tolist(),
            g_prag_p=g_p_l.argmax(-1).cpu().tolist(),
            g_prag_l=targets["g_prag"].cpu().tolist(),
            g_val_p=zeros_list,
            g_val_l=targets["g_val"].cpu().tolist(),
            gram_p=gram_l.argmax(-1).cpu().tolist(),
            gram_l=targets["gram"].cpu().tolist(),
            reg_p=reg_zeros,
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
    def evaluate(self) -> Tuple[EvaluationMetrics, Dict[str, StyleWorstSample]]:
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
                "f_loss_per_sample",
                "g_loss_per_sample",
                "gram_loss_per_sample",
            ]
        }

        for batch in self.val_loader:
            field_inputs, attention_mask, targets = self._unpack_training_batch(batch)

            outputs = self.model(field_inputs, attention_mask)

            self._accumulate_eval_batch(outputs, targets, batch, metrics_sum, all_preds)

            n += 1

        if n == 0:
            return EvaluationMetrics(), {}

        avg_metrics = {k: v / n for k, v in metrics_sum.items()}
        valid_idxs = [i for i, v in enumerate(all_preds["is_valid"]) if v]

        # Compute population counts for all tasks
        f_prag_labels = all_preds["f_prag_l"]
        formality_class0_count = sum(1 for lbl in f_prag_labels if lbl == 0)
        formality_class1_count = sum(1 for lbl in f_prag_labels if lbl == 1)

        g_prag_labels = all_preds["g_prag_l"]
        gender_class0_count = sum(1 for lbl in g_prag_labels if lbl == 0)
        gender_class1_count = sum(1 for lbl in g_prag_labels if lbl == 1)

        gram_labels = all_preds["gram_l"]
        gram_class0_count = sum(1 for lbl in gram_labels if lbl == 0)
        gram_class1_count = sum(1 for lbl in gram_labels if lbl == 1)

        register_count = len(valid_idxs)

        # Find worst samples for each task
        worst_samples: Dict[str, StyleWorstSample] = {}

        # Helper to find worst sample for a task
        def _find_worst(
            task_name: str,
            losses: List[float],
            preds: List[int],
            labels: List[int],
            sentences: List[str],
        ) -> None:
            if not losses:
                return
            worst_idx = max(range(len(losses)), key=lambda i: losses[i])
            worst_samples[task_name] = StyleWorstSample(
                task=task_name,
                loss=losses[worst_idx],
                target=labels[worst_idx],
                prediction=preds[worst_idx],
                sentence=sentences[worst_idx],
                sample_idx=worst_idx,
            )

        _find_worst(
            "formality",
            all_preds["f_loss_per_sample"],
            all_preds["f_prag_p"],
            all_preds["f_prag_l"],
            all_preds["sentences"],
        )
        _find_worst(
            "gender",
            all_preds["g_loss_per_sample"],
            all_preds["g_prag_p"],
            all_preds["g_prag_l"],
            all_preds["sentences"],
        )
        _find_worst(
            "grammaticality",
            all_preds["gram_loss_per_sample"],
            all_preds["gram_p"],
            all_preds["gram_l"],
            all_preds["sentences"],
        )

        metrics = EvaluationMetrics(
            loss=avg_metrics.get("loss", 0.0) * self.config.grad_accum_steps,
            formality_loss=avg_metrics.get("f_loss", 0.0),
            gender_loss=avg_metrics.get("g_loss", 0.0),
            grammaticality_loss=avg_metrics.get("gram_loss", 0.0),
            register_loss=avg_metrics.get("reg_loss", 0.0),
            formality_accuracy=_acc(all_preds["f_prag_p"], all_preds["f_prag_l"]),
            formality_mse=_mse(all_preds["f_val_p"], all_preds["f_val_l"], valid_idxs),
            formality_class0_accuracy=_acc_per_class(
                all_preds["f_prag_p"], all_preds["f_prag_l"], 0
            ),
            formality_class1_accuracy=_acc_per_class(
                all_preds["f_prag_p"], all_preds["f_prag_l"], 1
            ),
            formality_class0_count=formality_class0_count,
            formality_class1_count=formality_class1_count,
            gender_accuracy=_acc(all_preds["g_prag_p"], all_preds["g_prag_l"]),
            gender_mse=_mse(all_preds["g_val_p"], all_preds["g_val_l"], valid_idxs),
            gender_class0_accuracy=_acc_per_class(
                all_preds["g_prag_p"], all_preds["g_prag_l"], 0
            ),
            gender_class1_accuracy=_acc_per_class(
                all_preds["g_prag_p"], all_preds["g_prag_l"], 1
            ),
            gender_class0_count=gender_class0_count,
            gender_class1_count=gender_class1_count,
            grammaticality_accuracy=_acc(all_preds["gram_p"], all_preds["gram_l"]),
            gram_class0_accuracy=_acc_per_class(
                all_preds["gram_p"], all_preds["gram_l"], 0
            ),
            gram_class1_accuracy=_acc_per_class(
                all_preds["gram_p"], all_preds["gram_l"], 1
            ),
            gram_class0_count=gram_class0_count,
            gram_class1_count=gram_class1_count,
            register_accuracy=_reg_acc(
                all_preds["reg_p"], all_preds["reg_l"], valid_idxs
            ),
            register_count=register_count,
        )
        return metrics, worst_samples

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

        # Compute per-sample losses for worst sample tracking
        (f_prag_l, g_prag_l, gram_l) = outputs
        f_per_sample = (
            F.cross_entropy(f_prag_l, targets["f_prag"], reduction="none")
            .cpu()
            .tolist()
        )
        g_per_sample = (
            F.cross_entropy(g_prag_l, targets["g_prag"], reduction="none")
            .cpu()
            .tolist()
        )
        gram_per_sample = (
            F.cross_entropy(gram_l, targets["gram"], reduction="none").cpu().tolist()
        )

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
        sentences = batch.original_sentence
        # If sentences are missing (empty strings) and dataset allows retrieval, fetch them
        if (not sentences or not sentences[0]) and hasattr(
            self.val_dataset, "get_sentence_by_idx"
        ):
            sentences = [
                self.val_dataset.get_sentence_by_idx(int(idx.item()))
                for idx in batch.indices
            ]
        all_preds["sentences"].extend(sentences)
        all_preds["kotograms"].extend(batch.kotogram)
        # Per-sample losses for worst sample tracking
        all_preds["f_loss_per_sample"].extend(f_per_sample)
        all_preds["g_loss_per_sample"].extend(g_per_sample)
        all_preds["gram_loss_per_sample"].extend(gram_per_sample)

    def train(
        self,
        epochs: int,
        on_epoch_end: Callable[[TrainingHistory], None],
        start_epoch: Optional[int] = None,
    ) -> TrainingHistory:
        # Use explicit start_epoch if provided
        effective_start = start_epoch if start_epoch is not None else self.start_epoch

        # Record when this session started to support relative freezing/warmups
        if self.session_start_epoch is None:
            self.session_start_epoch = effective_start
        self.view.on_train_start(epochs, effective_start, self.start_batch)

        # Cache last evaluation result for epochs where we skip evaluation
        last_eval_res: Optional[EvaluationMetrics] = None

        for epoch in range(effective_start, epochs):
            tl, tfl, tgl, tgraml, trl = self.train_epoch(epoch=epoch)

            # Determine if we should run full evaluation this epoch
            # Always eval: first epoch of session, final target epoch, every N epochs
            # Use session_start_epoch (not effective_start) and config.epochs (not epochs arg)
            # to correctly handle incremental train() calls
            session_start = self.session_start_epoch or 0
            total_epochs = self.config.epochs
            is_first = epoch == session_start
            is_last = epoch == total_epochs - 1
            eval_interval = self.config.eval_every_n_epochs
            is_nth = (epoch - session_start) % eval_interval == 0
            should_eval = is_first or is_last or is_nth

            # Track worst samples from last evaluation (for display)
            worst_samples: Dict[str, StyleWorstSample] = {}

            if should_eval:
                eval_res, worst_samples = self.evaluate()
                last_eval_res = eval_res
                self.scheduler.step(eval_res.loss)
            else:
                # Skip evaluation, use training loss for scheduler
                self.scheduler.step(tl)
                # Use cached eval_res for history (or create minimal placeholder)
                if last_eval_res is not None:
                    eval_res = last_eval_res
                else:
                    eval_res = EvaluationMetrics()

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
                self.view.on_timing_summary(
                    total * 1000, data_avg * 1000, compute_avg * 1000, data_avg / total
                )
            self.train_timer_data.reset()
            self.train_timer_compute.reset()

            # Only show detailed stats and update best model when we ran evaluation
            if should_eval:
                avg_acc = (
                    eval_res.formality_accuracy
                    + eval_res.gender_accuracy
                    + eval_res.grammaticality_accuracy
                ) / 3.0

                # Build semantic epoch stats
                f_population = ClassPopulation(
                    class0_count=eval_res.formality_class0_count,
                    class1_count=eval_res.formality_class1_count,
                )
                g_population = ClassPopulation(
                    class0_count=eval_res.gender_class0_count,
                    class1_count=eval_res.gender_class1_count,
                )

                grad_norms = None
                if self.last_epoch_grad_norms:
                    gn = self.last_epoch_grad_norms
                    grad_norms = GradientNorms(
                        formality=gn.get("formality", 0.0),
                        gender=gn.get("gender", 0.0),
                        grammaticality=gn.get("grammaticality", 0.0),
                        encoder=gn.get("encoder", 0.0),
                        pooler=gn.get("pooler", 0.0),
                    )

                epoch_stats = StyleEpochStats(
                    formality=BinaryMetric(
                        loss=eval_res.formality_loss,
                        accuracy=eval_res.formality_accuracy,
                        class0_accuracy=eval_res.formality_class0_accuracy,
                        class1_accuracy=eval_res.formality_class1_accuracy,
                        population=f_population,
                    ),
                    gender=BinaryMetric(
                        loss=eval_res.gender_loss,
                        accuracy=eval_res.gender_accuracy,
                        class0_accuracy=eval_res.gender_class0_accuracy,
                        class1_accuracy=eval_res.gender_class1_accuracy,
                        population=g_population,
                    ),
                    grammaticality=GrammaticalityMetric(
                        loss=eval_res.grammaticality_loss,
                        accuracy=eval_res.grammaticality_accuracy,
                        class0_accuracy=eval_res.gram_class0_accuracy,
                        class1_accuracy=eval_res.gram_class1_accuracy,
                        class0_count=eval_res.gram_class0_count,
                        class1_count=eval_res.gram_class1_count,
                    ),
                    total_loss=eval_res.loss,
                    avg_accuracy=avg_acc,
                    grad_norms=grad_norms,
                )

                self.view.on_style_epoch_eval_stats(epoch + 1, epoch_stats)
                self.view.on_style_worst_samples(worst_samples)

                is_best = eval_res.loss < self.best_val_loss
                if is_best:
                    self.best_val_loss, self.patience_counter = eval_res.loss, 0
                    # Save copy of best state dict (FULL state)
                    if hasattr(self.model, "module"):
                        state = cast(Any, self.model).module.state_dict()
                    else:
                        state = self.model.state_dict()

                    # Deep copy to avoid reference issues if model mutates?
                    # State dict tensors share storage, but we don't mutate weights in place usually.
                    # However, to be safe and independent:
                    self.best_state = {k: v.cpu().clone() for k, v in state.items()}
                    save_model(self.model, self.output_path, self.model.config)
                    save_checkpoint(self.model)
                    model_path = os.path.join(self.output_path, "model.pt")
                    self.view.on_best_model_saved(model_path, self.best_val_loss)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.patience:
                        self.view.on_early_stopping(epoch + 1)
                        break

                self.view.on_epoch_end(
                    epoch + 1,
                    train_metrics=(tl, tfl, tgl, tgraml, trl),
                    eval_metrics=eval_res,
                    avg_acc=avg_acc,
                    is_best=is_best,
                    patience_counter=self.patience_counter,
                )
            else:
                # Skipped evaluation - just notify epoch end without detailed stats
                self.view.on_epoch_end(
                    epoch + 1,
                    train_metrics=(tl, tfl, tgl, tgraml, trl),
                    eval_metrics=None,
                    avg_acc=None,
                    is_best=False,
                    patience_counter=self.patience_counter,
                )

            on_epoch_end(self.history)

        if self.best_state:
            # Restore best state (FULL state including kc_decoders).
            # We used to strip kc_decoders here, but now we keep best_state fully aligned
            # with the training architecture to allow strict loading.
            # Stripping happens ONLY during export to 'models/style/model.pt'.
            self.model.load_state_dict(self.best_state, strict=True)
            # No need for manual missing/unexpected key validation with strict=True.

        self.view.on_train_end(self.history)
        return self.history
