from dataclasses import dataclass, field
from typing import Any, Dict, List, TypedDict

import torch
from torch.utils.data import DataLoader

from kotogram.model import InferenceClassifier
from kotogram.tokenizer import ENCODER_FEATURE_FIELDS


class EvalResultDict(TypedDict):
    """Serialization format for EvalResult."""

    formality_val_preds: List[float]
    formality_val_labels: List[float]
    formality_prag_preds: List[int]
    formality_prag_labels: List[int]
    gender_val_preds: List[float]
    gender_val_labels: List[float]
    gender_prag_preds: List[int]
    gender_prag_labels: List[int]
    grammaticality_preds: List[int]
    grammaticality_labels: List[int]
    register_preds: List[List[int]]
    register_labels: List[List[int]]
    sentences: List[str]
    kotograms: List[str]
    indices: List[int]


@dataclass
class EvalResult:
    """Container for evaluation results."""

    formality_val_preds: List[float] = field(default_factory=list)
    formality_val_labels: List[float] = field(default_factory=list)

    formality_prag_preds: List[int] = field(default_factory=list)
    formality_prag_labels: List[int] = field(default_factory=list)

    gender_val_preds: List[float] = field(default_factory=list)
    gender_val_labels: List[float] = field(default_factory=list)

    gender_prag_preds: List[int] = field(default_factory=list)
    gender_prag_labels: List[int] = field(default_factory=list)

    grammaticality_preds: List[int] = field(default_factory=list)
    grammaticality_labels: List[int] = field(default_factory=list)

    register_preds: List[List[int]] = field(default_factory=list)
    register_labels: List[List[int]] = field(default_factory=list)

    sentences: List[str] = field(default_factory=list)
    kotograms: List[str] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)

    def to_dict(self) -> EvalResultDict:
        """Convert to dictionary for backward compatibility."""
        return {
            "formality_val_preds": self.formality_val_preds,
            "formality_val_labels": self.formality_val_labels,
            "formality_prag_preds": self.formality_prag_preds,
            "formality_prag_labels": self.formality_prag_labels,
            "gender_val_preds": self.gender_val_preds,
            "gender_val_labels": self.gender_val_labels,
            "gender_prag_preds": self.gender_prag_preds,
            "gender_prag_labels": self.gender_prag_labels,
            "grammaticality_preds": self.grammaticality_preds,
            "grammaticality_labels": self.grammaticality_labels,
            "register_preds": self.register_preds,
            "register_labels": self.register_labels,
            "sentences": self.sentences,
            "kotograms": self.kotograms,
            "indices": self.indices,
        }


class Evaluator:
    """Encapsulates model evaluation logic."""

    def __init__(
        self, model: InferenceClassifier, device: torch.device, verbose: bool = True
    ):
        self.model = model
        self.device = device
        self.verbose = verbose

        from rich.console import Console

        self.console = Console()

    def evaluate(self, loader: DataLoader) -> EvalResult:
        # pylint: disable=too-many-locals
        """Run inference on the loader and return results."""
        self.model.eval()
        result = EvalResult()

        # Setup progress bar if verbose
        progress_context = None
        task_id = None

        if self.verbose:
            from rich.progress import (
                BarColumn,
                Progress,
                SpinnerColumn,
                TaskProgressColumn,
                TextColumn,
            )

            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console,
            )
            progress_context = progress
            task_id = progress.add_task("Evaluating...", total=len(loader))
            progress.start()

        elif self.verbose:
            # Fallback not needed as rich is required
            pass

        try:
            with torch.no_grad():
                # Group accumulators to avoid too-many-locals
                # Separate accumulators for better type safety
                preds_val: Dict[str, List[torch.Tensor]] = {
                    "formality_val": [],
                    "gender_val": [],
                }
                preds_class: Dict[str, List[torch.Tensor]] = {
                    "formality_prag": [],
                    "gender_prag": [],
                    "grammaticality": [],
                    "register": [],
                }
                preds_indices: List[torch.Tensor] = []

                # Targets (Async transfer)
                targets_val: Dict[str, List[torch.Tensor]] = {
                    "formality_val": [],
                    "gender_val": [],
                }
                targets_class: Dict[str, List[torch.Tensor]] = {
                    "formality_prag": [],
                    "gender_prag": [],
                    "grammaticality": [],
                    "register": [],
                }

                for batch in loader:
                    field_inputs = {
                        f"input_ids_{f}": batch.feature_inputs[f"input_ids_{f}"].to(
                            self.device
                        )
                        for f in ENCODER_FEATURE_FIELDS
                    }
                    attention_mask = batch.attention_mask.to(self.device)

                    # Targets (Async transfer)
                    # pylint: disable=duplicate-code
                    batch_targets = {
                        "formality_val": batch.formality_value.to(self.device),
                        "formality_prag": batch.formality_pragmatic.to(self.device),
                        "gender_val": batch.gender_value.to(self.device),
                        "gender_prag": batch.gender_pragmatic.to(self.device),
                        "grammaticality": batch.grammaticality_labels.to(self.device),
                        "register": batch.register_labels.to(self.device),
                    }

                    prediction = self.model.predict(field_inputs, attention_mask)

                    # Predictions
                    # Multi-label prediction (Exact match threshold 0.5)
                    register_preds = (prediction.register_probs > 0.5).long()

                    # Accumulate tensors (detach to save memory)
                    preds_val["formality_val"].append(
                        prediction.formality_value.squeeze(-1)
                    )
                    targets_val["formality_val"].append(batch_targets["formality_val"])

                    preds_class["formality_prag"].append(
                        prediction.formality_pragmatic_probs.argmax(dim=-1)
                    )
                    targets_class["formality_prag"].append(
                        batch_targets["formality_prag"]
                    )

                    preds_val["gender_val"].append(prediction.gender_value.squeeze(-1))
                    targets_val["gender_val"].append(batch_targets["gender_val"])

                    preds_class["gender_prag"].append(
                        prediction.gender_pragmatic_probs.argmax(dim=-1)
                    )
                    targets_class["gender_prag"].append(batch_targets["gender_prag"])

                    preds_class["grammaticality"].append(
                        prediction.grammaticality_probs.argmax(dim=-1)
                    )
                    targets_class["grammaticality"].append(
                        batch_targets["grammaticality"]
                    )

                    preds_class["register"].append(register_preds)
                    targets_class["register"].append(batch_targets["register"].long())

                    if batch.indices is not None:
                        preds_indices.append(batch.indices)

                    result.sentences.extend(batch.original_sentence or [])
                    result.kotograms.extend(batch.kotogram or [])

                    if progress_context and task_id is not None:
                        progress_context.update(task_id, advance=1)

                # Consolidate results (Single synchronization point)
                if preds_val["formality_val"]:
                    # Helper to cat list of tensors
                    def cat(ts: List[torch.Tensor]) -> List[Any]:
                        return torch.cat([x.cpu() for x in ts]).tolist()

                    result.formality_val_preds = cat(preds_val["formality_val"])
                    result.formality_val_labels = cat(targets_val["formality_val"])

                    result.formality_prag_preds = cat(preds_class["formality_prag"])
                    result.formality_prag_labels = cat(targets_class["formality_prag"])

                    result.gender_val_preds = cat(preds_val["gender_val"])
                    result.gender_val_labels = cat(targets_val["gender_val"])

                    result.gender_prag_preds = cat(preds_class["gender_prag"])
                    result.gender_prag_labels = cat(targets_class["gender_prag"])

                    result.grammaticality_preds = cat(preds_class["grammaticality"])
                    result.grammaticality_labels = cat(targets_class["grammaticality"])

                    result.register_preds = cat(preds_class["register"])
                    result.register_labels = cat(targets_class["register"])

                    if preds_indices:
                        val = torch.cat(preds_indices)
                        result.indices = val.tolist()

        except KeyboardInterrupt:
            self.console.print("\n[bold red]Evaluation interrupted by user.[/bold red]")
            import sys

            sys.exit(130)
        finally:
            if progress_context:
                progress_context.stop()

        return result
