#!/usr/bin/env python3
"""
Shared utilities for learnability studies and curation.
"""

import json
import os
import sqlite3
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import torch
from rich.console import Console
from rich.table import Table
from torch import nn
from torch.utils.data import DataLoader

from scripts.progress_utils import create_progress

T = TypeVar("T")


# pylint: disable=abstract-method
class BaseStudyDataset(torch.utils.data.Dataset[T], ABC):
    """Base class for study datasets with shared loading logic."""

    def __init__(self, data_dir: str, indices: Optional[torch.Tensor] = None):
        self.data_dir = data_dir

        # Load offsets
        offsets_path = os.path.join(data_dir, "offsets.bin")
        size_bytes = os.path.getsize(offsets_path)
        self.offsets = torch.from_file(
            offsets_path, shared=True, size=size_bytes // 4, dtype=torch.int32
        )

        # Load sentences
        sentences_path = os.path.join(data_dir, "sentences.txt")
        with open(sentences_path, "r", encoding="utf-8") as f:
            self.sentences = [line.strip() for line in f]

        # Use full dataset if indices not provided
        if indices is None:
            self.indices = torch.arange(len(self.offsets) - 1, dtype=torch.long)
        else:
            self.indices = indices

        # Load features (lazily if needed, but here we do it eagerly for simplicity)
        self._load_features()
        self._iter_ptr = 0

    def __iter__(self) -> "BaseStudyDataset":
        self._iter_ptr = 0
        return self

    def __next__(self) -> T:
        if self._iter_ptr >= len(self.indices):
            raise StopIteration
        sample = self[self._iter_ptr]
        self._iter_ptr += 1
        return sample

    @abstractmethod
    def __getitem__(self, index: int) -> T:
        """Get a sample by index."""
        raise NotImplementedError

    def _load_features(self) -> None:
        """Load KC features into memory."""
        self.features: Dict[str, torch.Tensor] = {}
        for filename in os.listdir(self.data_dir):
            if filename.startswith("feat_") and filename.endswith(".bin"):
                field = filename[len("feat_") : -len(".bin")]
                path = os.path.join(self.data_dir, filename)
                size_bytes = os.path.getsize(path)
                # KC features are int16
                self.features[field] = torch.from_file(
                    path, shared=True, size=size_bytes // 2, dtype=torch.int16
                )

    def get_feature_slice(self, start: int, end: int, field: str) -> torch.Tensor:
        """Get a slice of features for a specific field."""
        return self.features[field][start:end].long()

    def __len__(self) -> int:
        return len(self.indices)


class BaseStudyClassifier(nn.Module, ABC):
    """Base class for study classifiers with shared embedding logic."""

    def __init__(self, vocab_sizes: Dict[str, int], embed_dim: int):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.embed_dim = embed_dim
        self.embeddings = nn.ModuleDict()
        for name, vocab_size in vocab_sizes.items():
            self.embeddings[name] = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    @abstractmethod
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass."""


class BaseMLPStudyClassifier(BaseStudyClassifier):
    """Base class for simple MLP-based study classifiers."""

    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__(vocab_sizes, embed_dim)
        total_embed = embed_dim * len(vocab_sizes)
        self.classifier = nn.Sequential(
            nn.Linear(total_embed, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        embeds = []
        for name in sorted(self.vocab_sizes.keys()):
            if name in features:
                feat = features[name]
                emb = self.embeddings[name](feat)
                mask = (feat != 0).unsqueeze(-1).float()
                masked_emb = emb * mask
                seq_lens = mask.sum(dim=1).clamp(min=1)
                pooled = masked_emb.sum(dim=1) / seq_lens
                embeds.append(pooled)

        combined = torch.cat(embeds, dim=-1)
        result: torch.Tensor = self.classifier(combined)
        return result


def get_vocab_sizes(data_dir: str) -> Dict[str, int]:
    """Get vocabulary sizes for all features from vocab.json."""
    vocab_path = os.path.join(data_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        return {}
    with open(vocab_path, "r", encoding="utf-8") as raw_f:
        data = json.load(raw_f)
    field_vocabs = data.get("field_vocabs", {})
    return {field: len(vocab) for field, vocab in field_vocabs.items()}


# pylint: disable=too-many-locals
def compute_confusion_matrix(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    collate_fn: Any,
    num_classes: int,
    batch_size: int = 256,
) -> Tuple[List[List[int]], List[Tuple[str, int, int, float]]]:
    """Compute confusion matrix and identify candidates for review."""
    device = get_device()
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    matrix = [[0] * num_classes for _ in range(num_classes)]
    candidates: List[Tuple[str, int, int, float]] = []

    with torch.no_grad():
        for features, labels, sentences, _ in loader:
            # Move features to device
            features = {k: v.to(device) for k, v in features.items()}
            logits = model(features)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            for i, (label_t, pred_t) in enumerate(zip(labels, preds)):
                true_cls = int(label_t.item())
                pred_cls = int(pred_t.item())
                matrix[true_cls][pred_cls] += 1

                if true_cls != pred_cls:
                    candidates.append(
                        (
                            sentences[i],
                            true_cls,
                            pred_cls,
                            float(probs[i][pred_cls].item()),
                        )
                    )

    # Sort candidates by confidence
    candidates.sort(key=lambda x: x[3], reverse=True)
    return matrix, candidates


def print_confusion_matrix(
    console: Console,
    matrix: List[List[int]],
    class_names: List[str],
    title: str = "Confusion Matrix",
) -> None:
    """Print a formatted confusion matrix table."""
    table = Table(title=title)
    table.add_column("True \\ Pred", style="cyan")
    for name in class_names:
        table.add_column(name, justify="right")

    for i, row_name in enumerate(class_names):
        row_data = [row_name]
        for j, val in enumerate(matrix[i]):
            style = "green" if i == j else "red" if val > 0 else "dim"
            row_data.append(f"[{style}]{val:,}[/{style}]")
        table.add_row(*row_data)

    console.print(table)


# pylint: disable=too-many-positional-arguments
def generate_suggestion_files(
    console: Console,
    candidates: List[Tuple[str, int, int, float]],
    class_names: List[str],
    study_dir: str,
    batch: int = 1,
    batch_size: int = 100,
) -> None:
    """Generate text files with suggested label changes."""
    # Group by predicted class
    by_pred: Dict[int, List[Tuple[str, float]]] = {}
    for sent, _, pred, conf in candidates:
        if pred not in by_pred:
            by_pred[pred] = []
        by_pred[pred].append((sent, conf))

    start_idx = (batch - 1) * batch_size

    for cls_idx, items in by_pred.items():
        cls_name = class_names[cls_idx]
        batch_items = items[start_idx : start_idx + batch_size]

        if not batch_items:
            continue

        filename = f"suggest {cls_name}.txt"
        path = os.path.join(study_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for sent, _ in batch_items:
                f.write(f"{sent}\n")

        console.print(f"  Wrote {len(batch_items)} suggestions to {filename}")


def collate_study_samples(
    batch: List[Any],
    dataset: Any,
    label_attr: str,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[str], List[int]]:
    """Collate samples into batched tensors."""
    labels = torch.tensor([getattr(s, label_attr) for s in batch], dtype=torch.long)
    sentences = [s.sentence for s in batch]
    indices = [s.idx for s in batch]
    max_len = max(s.feature_end - s.feature_start for s in batch)

    features: Dict[str, torch.Tensor] = {}
    for field in dataset.features:
        batch_feat = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, s in enumerate(batch):
            feat_slice = dataset.get_feature_slice(
                s.feature_start, s.feature_end, field
            )
            seq_len = min(len(feat_slice), max_len)
            batch_feat[i, :seq_len] = feat_slice[:seq_len]
        features[field] = batch_feat

    return features, labels, sentences, indices


# pylint: disable=too-many-positional-arguments,unused-argument
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Any,
    progress: Any,
    task_id: Any,
    epoch: int,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for features, labels, _, _ in loader:
        features = {k: v.to(device) for k, v in features.items()}
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device.type, enabled=scaler.is_enabled()):
            logits = model(features)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        progress.update(task_id, advance=1)

    return total_loss / len(loader)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels, _, _ in loader:
            features = {k: v.to(device) for k, v in features.items()}
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


def get_device() -> torch.device:
    """Get the best available device (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_dataset_indices(
    indices: torch.Tensor, train_percent: float = 0.8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split indices into training and validation sets."""
    n_samples = len(indices)
    n_train = int(n_samples * train_percent)

    # Shuffle indices
    perm = torch.randperm(n_samples)
    shuffled = indices[perm]

    return shuffled[:n_train], shuffled[n_train:]


# pylint: disable=too-many-locals,too-many-positional-arguments
# pylint: disable=too-many-locals,too-many-positional-arguments
def run_study_evaluation(
    console: Console,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    collate_fn: Any,
    class_names: List[str],
    model_path: str,
    study_dir: str,
    batch: int = 1,
) -> None:
    """Run model evaluation and generate suggestion files."""
    # Save model
    torch.save(model.state_dict(), model_path)
    console.print(f"[green]Saved model to {model_path}[/green]")

    console.print("[bold blue]Computing confusion matrix...[/bold blue]")
    matrix, candidates = compute_confusion_matrix(
        model, dataset, collate_fn, num_classes=len(class_names)
    )

    print_confusion_matrix(console, matrix, class_names, title="Study Confusion Matrix")

    # Save candidates
    candidates_data = [
        {"sentence": c[0], "true_class": c[1], "pred_class": c[2], "confidence": c[3]}
        for c in candidates
    ]
    candidates_path = os.path.join(study_dir, "candidates.json")
    with open(candidates_path, "w", encoding="utf-8") as f:
        json.dump(candidates_data, f)
    console.print(f"[green]Saved candidates to {candidates_path}[/green]")

    console.print(
        f"[bold blue]Generating suggestion files (batch {batch})...[/bold blue]"
    )
    generate_suggestion_files(console, candidates, class_names, study_dir, batch=batch)


def verify_database_updates(
    console: Console,
    cursor: sqlite3.Cursor,
    updated_sentences: Dict[str, Any],
    column_name: str,
) -> int:
    """Verify that database updates were applied correctly."""
    console.print("[bold blue]Verifying updates...[/bold blue]")
    verified = 0
    mismatches = 0

    for sent, expected_value in updated_sentences.items():
        cursor.execute(f"SELECT {column_name} FROM corpus WHERE sentence = ?", (sent,))
        row = cursor.fetchone()
        if row is None:
            console.print(f"  [red]Missing:[/red] {sent[:40]}...")
            mismatches += 1
        else:
            actual_value = row[0]
            # Handle NULL (None in Python)
            if actual_value is None and expected_value is None:
                verified += 1
            elif actual_value != expected_value:
                console.print(
                    f"  [red]Mismatch:[/red] expected {expected_value}, "
                    f"got {actual_value} for: {sent[:40]}..."
                )
                mismatches += 1
            else:
                verified += 1

    if mismatches == 0:
        console.print(f"  [green]✓ Verified all {verified:,} updates[/green]")
    else:
        console.print(f"  [red]✗ {mismatches:,} mismatches found[/red]")

    return mismatches


# pylint: disable=too-many-locals,too-many-positional-arguments
def setup_study_loaders(
    train_ds: torch.utils.data.Dataset,
    val_ds: torch.utils.data.Dataset,
    collate_fn: Any,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Set up training and validation DataLoader instances."""
    device = get_device()
    use_cuda = device.type == "cuda"

    # num_workers > 0 causes issues with MPS and pickling in some environments
    num_workers = 4 if use_cuda else 0
    pin_memory = use_cuda

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


# pylint: disable=too-many-positional-arguments
def prepare_study_data(
    dataset: Any,
    train_indices: torch.Tensor,
    val_indices: torch.Tensor,
    ds_class: type,
) -> Tuple[Any, Any]:
    """Prepare training and validation datasets by copying shared attributes."""
    train_ds = ds_class(dataset.data_dir, indices=train_indices)
    val_ds = ds_class(dataset.data_dir, indices=val_indices)

    # Common attributes to share
    attrs = [
        "features",
        "offsets",
        "sentences",
        "grammatic",
        "gender_pragmatic",
        "gender_values",
    ]
    for attr in attrs:
        if hasattr(dataset, attr):
            setattr(train_ds, attr, getattr(dataset, attr))
            setattr(val_ds, attr, getattr(dataset, attr))

    return train_ds, val_ds


# pylint: disable=too-many-locals,too-many-positional-arguments
def train_study_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    console: Console,
    max_epochs: int = 100,
    patience: int = 5,
) -> nn.Module:
    """Train a study model with early stopping."""
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state = model.state_dict()
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        with create_progress(console) as progress:
            task = progress.add_task(
                f"Epoch {epoch + 1} training...", total=len(train_loader)
            )
            train_avg_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler,
                progress,
                task,
                epoch,
            )

        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        console.print(
            f"  Epoch {epoch + 1}: "
            f"Loss={train_avg_loss:.4f}, "
            f"Val Loss={val_loss:.4f}, "
            f"Val Acc={val_acc:.2%}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                console.print(
                    f"[yellow]Early stopping at epoch {epoch + 1} "
                    f"(no improvement for {patience} epochs)[/yellow]"
                )
                break

    model.load_state_dict(best_state)
    console.print(
        f"[green]Best validation: Loss={best_val_loss:.4f}, Acc={best_val_acc:.2%}[/green]"
    )
    return model


# pylint: disable=too-many-locals,too-many-positional-arguments
def run_standard_study(
    console: Console,
    study_dir: str,
    model_dir: Optional[str],
    dataset: Any,
    model_class: type,
    collate_fn: Any,
    class_names: List[str],
    train_fn: Any,
    batch: int = 1,
) -> None:
    """Run a standard learnability study with caching and evaluation."""
    os.makedirs(study_dir, exist_ok=True)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir if model_dir else study_dir, "model.pt")
    candidates_path = os.path.join(study_dir, "candidates.json")

    # Fast path: load cached candidates if available and batch > 1
    if batch > 1 and os.path.exists(candidates_path):
        console.print(
            f"[bold blue]Loading cached candidates from {candidates_path}...[/bold blue]"
        )
        with open(candidates_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        candidates = [
            (c["sentence"], c["true_class"], c["pred_class"], c["confidence"])
            for c in data
        ]
        generate_suggestion_files(
            console, candidates, class_names, study_dir, batch=batch
        )
        return

    # Normal path: Train or load model
    vocab_sizes = get_vocab_sizes(dataset.data_dir)
    if batch > 1 and os.path.exists(model_path):
        console.print(f"[bold blue]Loading model from {model_path}[/bold blue]")
        model = model_class(vocab_sizes).to(get_device())
        model.load_state_dict(torch.load(model_path, map_location=get_device()))
    else:
        model = train_fn(dataset, vocab_sizes)

    # Evaluation
    run_study_evaluation(
        console,
        model,
        dataset,
        collate_fn,
        class_names,
        model_path,
        study_dir,
        batch=batch,
    )

    # Compute final accuracy for results
    model.eval()
    matrix, _ = compute_confusion_matrix(
        model, dataset, collate_fn, num_classes=len(class_names)
    )
    correct = sum(matrix[i][i] for i in range(len(class_names)))
    total = sum(sum(row) for row in matrix)

    save_study_results(
        console,
        study_dir,
        {
            "total_samples": len(dataset),
            "confusion_matrix": matrix,
            "accuracy": correct / total if total > 0 else 0,
        },
    )


def check_dataset_cache(console: Console, data_dir: str) -> bool:
    """Check if dataset cache exists and print error if not."""
    if not os.path.exists(os.path.join(data_dir, "offsets.bin")):
        console.print(
            "[bold red]Error:[/bold red] Dataset cache not found. "
            "Please run './train_style --label' first."
        )
        return False
    return True


def save_study_results(
    console: Console,
    study_dir: str,
    results: Dict[str, Any],
) -> None:
    """Save study results to JSON and print summary."""
    results_path = os.path.join(study_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    console.print(f"[green]Saved results to {results_path}[/green]")

    console.print("\n[bold green]Study complete![/bold green]")
    if "accuracy" in results:
        console.print(f"  Accuracy: {results['accuracy']:.2%}")
    console.print(f"  Output directory: {study_dir}")


# pylint: disable=too-many-locals,too-many-positional-arguments
def finalize_database_updates(
    console: Console,
    conn: sqlite3.Connection,
    total_updates: int,
    db_path: str,
) -> None:
    """Finalize database updates: commit, close, and print summary."""
    conn.commit()
    conn.close()
    console.print(
        f"[bold green]Applied {total_updates:,} total updates to {db_path}[/bold green]"
    )
