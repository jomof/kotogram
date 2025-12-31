from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from kotogram.model import StyleClassifier
from kotogram.tokenizer import FEATURE_FIELDS


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

    # Store raw logits if needed later? Maybe too heavy.

    def to_dict(self) -> Dict[str, Any]:
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
        self, model: StyleClassifier, device: torch.device, verbose: bool = True
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
                preds: Dict[str, List[Any]] = {
                    "formality_val": [],
                    "formality_val_targets": [],
                    "formality_prag": [],
                    "formality_prag_targets": [],
                    "gender_val": [],
                    "gender_val_targets": [],
                    "gender_prag": [],
                    "gender_prag_targets": [],
                    "grammaticality": [],
                    "grammaticality_targets": [],
                    "register": [],
                    "register_targets": [],
                    "indices": [],
                }

                for batch in loader:
                    field_inputs = {
                        f"input_ids_{f}": batch.feature_inputs[f"input_ids_{f}"].to(
                            self.device
                        )
                        for f in FEATURE_FIELDS
                    }
                    attention_mask = batch.attention_mask.to(self.device)

                    # Targets (Async transfer)
                    # pylint: disable=duplicate-code
                    targets = {
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
                    preds["formality_val"].append(
                        prediction.formality_value.squeeze(-1)
                    )
                    preds["formality_val_targets"].append(targets["formality_val"])

                    preds["formality_prag"].append(
                        prediction.formality_pragmatic_probs.argmax(dim=-1)
                    )
                    preds["formality_prag_targets"].append(targets["formality_prag"])

                    preds["gender_val"].append(prediction.gender_value.squeeze(-1))
                    preds["gender_val_targets"].append(targets["gender_val"])

                    preds["gender_prag"].append(
                        prediction.gender_pragmatic_probs.argmax(dim=-1)
                    )
                    preds["gender_prag_targets"].append(targets["gender_prag"])

                    preds["grammaticality"].append(
                        prediction.grammaticality_probs.argmax(dim=-1)
                    )
                    preds["grammaticality_targets"].append(targets["grammaticality"])

                    preds["register"].append(register_preds)
                    preds["register_targets"].append(targets["register"].long())

                    if batch.indices is not None:
                        preds["indices"].append(batch.indices)

                    result.sentences.extend(batch.original_sentence or [])
                    result.kotograms.extend(batch.kotogram or [])

                    if progress_context and task_id is not None:
                        progress_context.update(task_id, advance=1)

                # Consolidate results (Single synchronization point)
                if preds["formality_val"]:
                    result.formality_val_preds = torch.cat(
                        [x.cpu() for x in preds["formality_val"]]
                    ).tolist()
                    result.formality_val_labels = torch.cat(
                        [x.cpu() for x in preds["formality_val_targets"]]
                    ).tolist()

                    result.formality_prag_preds = torch.cat(
                        [x.cpu() for x in preds["formality_prag"]]
                    ).tolist()
                    result.formality_prag_labels = torch.cat(
                        [x.cpu() for x in preds["formality_prag_targets"]]
                    ).tolist()

                    result.gender_val_preds = torch.cat(
                        [x.cpu() for x in preds["gender_val"]]
                    ).tolist()
                    result.gender_val_labels = torch.cat(
                        [x.cpu() for x in preds["gender_val_targets"]]
                    ).tolist()

                    result.gender_prag_preds = torch.cat(
                        [x.cpu() for x in preds["gender_prag"]]
                    ).tolist()
                    result.gender_prag_labels = torch.cat(
                        [x.cpu() for x in preds["gender_prag_targets"]]
                    ).tolist()

                    result.grammaticality_preds = torch.cat(
                        [x.cpu() for x in preds["grammaticality"]]
                    ).tolist()
                    result.grammaticality_labels = torch.cat(
                        [x.cpu() for x in preds["grammaticality_targets"]]
                    ).tolist()

                    result.register_preds = torch.cat(
                        [x.cpu() for x in preds["register"]]
                    ).tolist()
                    result.register_labels = torch.cat(
                        [x.cpu() for x in preds["register_targets"]]
                    ).tolist()

                    if preds["indices"]:
                        result.indices = torch.cat(
                            [x.cpu() for x in preds["indices"]]
                        ).tolist()

        except KeyboardInterrupt:
            self.console.print("\n[bold red]Evaluation interrupted by user.[/bold red]")
            import sys

            sys.exit(130)
        finally:
            if progress_context:
                progress_context.stop()

        return result
