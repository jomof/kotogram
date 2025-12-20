from dataclasses import dataclass, field
from typing import List, Dict, Any
import torch
from torch.utils.data import DataLoader
from kotogram.model import StyleClassifier, FEATURE_FIELDS

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
    
    # Store raw logits if needed later? Maybe too heavy.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            'formality_val_preds': self.formality_val_preds,
            'formality_val_labels': self.formality_val_labels,
            'formality_prag_preds': self.formality_prag_preds,
            'formality_prag_labels': self.formality_prag_labels,
            'gender_val_preds': self.gender_val_preds,
            'gender_val_labels': self.gender_val_labels,
            'gender_prag_preds': self.gender_prag_preds,
            'gender_prag_labels': self.gender_prag_labels,
            'grammaticality_preds': self.grammaticality_preds,
            'grammaticality_labels': self.grammaticality_labels,
            'register_preds': self.register_preds,
            'register_labels': self.register_labels,
            'sentences': self.sentences,
            'kotograms': self.kotograms
        }

class Evaluator:
    """Encapsulates model evaluation logic."""
    
    def __init__(self, model: StyleClassifier, device: torch.device, verbose: bool = True):
        self.model = model
        self.device = device
        self.verbose = verbose
        
        from rich.console import Console
        self.console = Console()

    def evaluate(self, loader: DataLoader) -> EvalResult:
        """Run inference on the loader and return results."""
        self.model.eval()
        result = EvalResult()
        
        # Setup progress bar if verbose
        progress_context = None
        task_id = None
        
        if self.verbose:
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console
            )
            progress_context = progress
            task_id = progress.add_task("Evaluating...", total=len(loader))
            progress.start()

        elif self.verbose:
            # Fallback not needed as rich is required
            pass

        try:
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    field_inputs = {
                        f'input_ids_{f}': batch[f'input_ids_{f}'].to(self.device) 
                        for f in FEATURE_FIELDS
                    }
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    formality_value_targets = batch['formality_value'].to(self.device)
                    formality_prag_targets = batch['formality_pragmatic'].to(self.device)
                    gender_value_targets = batch['gender_value'].to(self.device)
                    gender_prag_targets = batch['gender_pragmatic'].to(self.device)
                    grammaticality_targets = batch['grammaticality_labels'].to(self.device)
                    register_targets = batch['register_labels'].to(self.device)

                    prediction = self.model.predict(
                        field_inputs, attention_mask
                    )

                    # Predictions
                    formality_prag_preds = prediction.formality_pragmatic_probs.argmax(dim=-1)
                    gender_prag_preds = prediction.gender_pragmatic_probs.argmax(dim=-1)
                    grammaticality_preds = prediction.grammaticality_probs.argmax(dim=-1)
                    
                    # Multi-label prediction (Exact match threshold 0.5)
                    register_probs = prediction.register_probs
                    register_preds = (register_probs > 0.5).long()

                    # Accumulate
                    result.formality_val_preds.extend(prediction.formality_value.squeeze(-1).cpu().tolist())
                    result.formality_val_labels.extend(formality_value_targets.cpu().tolist())

                    result.formality_prag_preds.extend(formality_prag_preds.cpu().tolist())
                    result.formality_prag_labels.extend(formality_prag_targets.cpu().tolist())
                    
                    result.gender_val_preds.extend(prediction.gender_value.squeeze(-1).cpu().tolist())
                    result.gender_val_labels.extend(gender_value_targets.cpu().tolist())
                    
                    result.gender_prag_preds.extend(gender_prag_preds.cpu().tolist())
                    result.gender_prag_labels.extend(gender_prag_targets.cpu().tolist())

                    result.grammaticality_preds.extend(grammaticality_preds.cpu().tolist())
                    result.grammaticality_labels.extend(grammaticality_targets.cpu().tolist())
                    
                    result.register_preds.extend(register_preds.cpu().tolist())
                    result.register_labels.extend(register_targets.long().cpu().tolist())

                    result.sentences.extend(batch.get('original_sentence', []))
                    result.kotograms.extend(batch.get('kotogram', []))
                    
                    if progress_context and task_id is not None:
                        progress_context.update(task_id, advance=1)
                        
        except KeyboardInterrupt:
            self.console.print("\n[bold red]Evaluation interrupted by user.[/bold red]")
            import sys
            sys.exit(130)
        finally:
            if progress_context:
                progress_context.stop()

        return result
