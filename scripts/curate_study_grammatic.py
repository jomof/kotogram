#!/usr/bin/env python3
"""
Grammatic learnability study for corpus.db.

Trains a focused classifier to learn whether sentences are grammatical
and identifies potential mislabels.
"""

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
from rich.console import Console
from rich.table import Table
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scripts.progress_utils import create_progress

console = Console()

# Class indices
CLASS_UNGRAMMATIC = 0
CLASS_GRAMMATIC = 1
CLASS_NAMES = ["ungrammatic", "grammatic"]

# Output directory
STUDY_DIR = ".cache/curate/study/grammatic"
MODEL_DIR = "models/grammatic"


@dataclass
class GrammaticSample:
    """A sample for grammatic classification."""

    idx: int
    sentence: str
    grammatic: int  # 0 = ungrammatic, 1 = grammatic
    feature_start: int
    feature_end: int


class GrammaticDataset(Dataset[GrammaticSample]):
    """Dataset for grammatic classification study."""

    def __init__(
        self,
        data_dir: str,
        indices: Optional[torch.Tensor] = None,
    ):
        self.data_dir = data_dir

        # Load offsets - clone to regular tensor for worker pickling
        offsets_path = os.path.join(data_dir, "offsets.bin")
        size_bytes = os.path.getsize(offsets_path)
        mmap_offsets = torch.from_file(
            offsets_path, shared=True, size=size_bytes // 4, dtype=torch.int32
        )
        self.offsets = mmap_offsets.clone()

        # Load grammatic flags - clone to regular tensor for worker pickling
        grammatic_path = os.path.join(data_dir, "labels.bin_gram")
        grammatic_size = os.path.getsize(grammatic_path)
        mmap_gram = torch.from_file(
            grammatic_path, shared=True, size=grammatic_size, dtype=torch.uint8
        )
        self.grammatic = mmap_gram.clone()

        # Load sentences
        sentences_path = os.path.join(data_dir, "sentences.txt")
        with open(sentences_path, "r", encoding="utf-8") as f:
            self.sentences = [line.strip() for line in f]

        # Use all samples or provided indices
        total_samples = len(self.offsets) - 1
        if indices is not None:
            self.indices = indices
        else:
            self.indices = torch.arange(total_samples, dtype=torch.long)

        # Load KC features for the model
        self._load_features()

        # Iterator state
        self._iter_idx = 0

    def _load_features(self) -> None:
        """Load KC bag features for input."""
        self.features: Dict[str, torch.Tensor] = {}
        feature_names = [
            "feat_pos.bin",
            "feat_conjugated_type.bin",
            "feat_reading_gram.bin",
            "feat_compound_1.bin",
        ]
        for fname in feature_names:
            path = os.path.join(self.data_dir, fname)
            if os.path.exists(path):
                size = os.path.getsize(path) // 4
                key = fname.replace("feat_", "").replace(".bin", "")
                # Clone to regular tensor (memory-mapped can't be pickled for workers)
                mmap = torch.from_file(path, shared=True, size=size, dtype=torch.int32)
                self.features[key] = mmap.clone()

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> "GrammaticDataset":
        """Iterate over samples."""
        self._iter_idx = 0
        return self

    def __next__(self) -> "GrammaticSample":
        """Get next sample."""
        if self._iter_idx >= len(self):
            raise StopIteration
        sample = self[self._iter_idx]
        self._iter_idx += 1
        return sample

    def __getitem__(self, idx: int) -> GrammaticSample:
        real_idx = int(self.indices[idx].item())
        start = int(self.offsets[real_idx].item())
        end = int(self.offsets[real_idx + 1].item())
        gram = int(self.grammatic[real_idx].item())

        return GrammaticSample(
            idx=real_idx,
            sentence=self.sentences[real_idx],
            grammatic=gram,
            feature_start=start,
            feature_end=end,
        )

    def get_feature_slice(self, start: int, end: int, field: str) -> torch.Tensor:
        """Get feature tensor slice for a sample."""
        if field in self.features:
            return self.features[field][start:end]
        return torch.tensor([], dtype=torch.int32)


class GrammaticClassifier(nn.Module):
    """Transformer-based classifier for grammatic prediction.

    Uses attention to capture word order and syntactic patterns that are
    important for grammaticality detection.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.embed_dim = embed_dim

        # Combined embedding dimension for all features
        total_embed = embed_dim * len(vocab_sizes)

        # Embedding for each feature type
        self.embeddings = nn.ModuleDict()
        for name, vocab_size in vocab_sizes.items():
            self.embeddings[name] = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Positional encoding - use standard initialization
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, total_embed))
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)

        # Input layer norm - critical for transformer stability
        self.input_norm = nn.LayerNorm(total_embed)

        # Transformer encoder - feedforward should be 4x d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=total_embed,
            nhead=num_heads,
            dim_feedforward=total_embed * 4,  # Standard is 4x d_model
            dropout=0.1,
            batch_first=True,
            norm_first=True,  # Pre-LN is more stable
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(total_embed, total_embed),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(total_embed, 2),  # 2 classes
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass. features is dict of (batch, seq_len) tensors."""
        # Get embeddings for each feature type
        embeds = []
        seq_len = 0
        padding_mask = None

        for name in sorted(self.vocab_sizes.keys()):
            if name in features:
                feat = features[name]  # (batch, seq_len)
                seq_len = feat.shape[1]
                emb = self.embeddings[name](feat)  # (batch, seq_len, embed_dim)
                embeds.append(emb)
                if padding_mask is None:
                    padding_mask = feat == 0  # True where padded

        # Concatenate feature embeddings: (batch, seq_len, total_embed)
        combined = torch.cat(embeds, dim=-1)

        # Add positional encoding
        combined = combined + self.pos_encoding[:, :seq_len, :]

        # Apply input normalization for stability
        combined = self.input_norm(combined)

        # Apply transformer (no padding mask for MPS compatibility)
        # The model will learn to ignore padding through the zero embeddings
        transformed = self.transformer(combined)

        # Mean pooling over non-padded positions
        if padding_mask is not None:
            mask = (~padding_mask).unsqueeze(-1).float()
            pooled = (transformed * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = transformed.mean(dim=1)

        result: torch.Tensor = self.classifier(pooled)
        return result


def _collate_fn(
    batch: List[GrammaticSample],
    dataset: GrammaticDataset,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[str], List[int]]:
    """Collate samples into batched tensors."""
    labels = torch.tensor([s.grammatic for s in batch], dtype=torch.long)
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


def _get_vocab_sizes(data_dir: str) -> Dict[str, int]:
    """Get vocabulary sizes from tokenizer."""
    vocab_path = os.path.join(data_dir, "vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    sizes = {}
    field_vocabs = vocab_data.get("field_vocabs", {})
    for field in ["pos", "conjugated_type", "reading_gram", "compound_1"]:
        if field in field_vocabs:
            field_vocab = field_vocabs[field]
            max_id = max(field_vocab.values())
            sizes[field] = max_id + 1
        else:
            sizes[field] = 1000

    return sizes


# pylint: disable=too-many-locals,too-many-positional-arguments
def _train_classifier(
    dataset: GrammaticDataset,
    vocab_sizes: Dict[str, int],
    max_epochs: int = 100,
    patience: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> GrammaticClassifier:
    """Train a grammatic classifier with early stopping on validation loss."""
    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = GrammaticClassifier(vocab_sizes).to(device)

    # Split into train/val
    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    perm = torch.randperm(n_total)
    train_indices = dataset.indices[perm[:n_train]]
    val_indices = dataset.indices[perm[n_train:]]

    train_ds = GrammaticDataset(dataset.data_dir, indices=train_indices)
    val_ds = GrammaticDataset(dataset.data_dir, indices=val_indices)

    # Share loaded features  # pylint: disable=attribute-defined-outside-init
    train_ds.features = dataset.features
    train_ds.offsets = dataset.offsets
    train_ds.grammatic = dataset.grammatic
    train_ds.sentences = dataset.sentences

    val_ds.features = dataset.features  # pylint: disable=attribute-defined-outside-init
    val_ds.offsets = dataset.offsets
    val_ds.grammatic = dataset.grammatic
    val_ds.sentences = dataset.sentences

    # Use partial instead of closure - closures can't be pickled for workers
    collate = partial(_collate_fn, dataset=dataset)

    # Use multiple workers on CUDA for faster data loading
    # Tensors are cloned (not memory-mapped), so they can be pickled for workers
    use_cuda = device.type == "cuda"
    num_workers = 4 if use_cuda else 0
    pin_memory = use_cuda

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    # Compute class weights for imbalanced data
    class_counts = torch.zeros(2, dtype=torch.float32)
    for sample in train_ds:
        class_counts[sample.grammatic] += 1

    total_samples = class_counts.sum()
    class_weights = total_samples / (2 * class_counts.clamp(min=1))
    class_weights = class_weights / class_weights.sum() * 2
    class_weights = class_weights.to(device)

    console.print("[bold blue]Training grammatic classifier[/bold blue]")
    console.print(f"  Train: {len(train_ds):,}, Val: {len(val_ds):,}")
    console.print(f"  Device: {device}")
    console.print(
        f"  Class weights: ungrammatic={class_weights[0]:.2f}, "
        f"grammatic={class_weights[1]:.2f}"
    )
    console.print(f"  Early stopping: patience={patience}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Mixed precision training for CUDA (FP16) - ~2x speedup on modern GPUs
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        console.print("  [cyan]Using FP16 mixed precision[/cyan]")

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state = model.state_dict()
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        start_time = time.perf_counter()
        samples_processed = 0
        with create_progress(console) as progress:
            task = progress.add_task(
                f"Epoch {epoch + 1} training...", total=len(train_loader)
            )
            for features, labels, _, _ in train_loader:
                features = {k: v.to(device) for k, v in features.items()}
                labels = labels.to(device)

                optimizer.zero_grad()

                # Mixed precision forward pass
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(features)
                    loss = criterion(logits, labels)

                # Scaled backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                samples_processed += len(labels)
                elapsed = time.perf_counter() - start_time
                rate = samples_processed / elapsed if elapsed > 0 else 0
                progress.update(
                    task,
                    description=f"Epoch {epoch + 1} [{rate:,.0f} sent/s]",
                    advance=1,
                )

        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for features, labels, _, _ in val_loader:
                features = {k: v.to(device) for k, v in features.items()}
                labels = labels.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(features)
                    loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += len(labels)

        val_loss = val_loss / len(val_loader)

        val_acc = correct / total if total > 0 else 0
        console.print(
            f"  Epoch {epoch + 1}: "
            f"Loss={train_loss / len(train_loader):.4f}, "
            f"Val Loss={val_loss:.4f}, "
            f"Val Acc={val_acc:.2%}"
        )

        # Early stopping based on validation loss
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


def _compute_confusion_matrix(
    model: GrammaticClassifier,
    dataset: GrammaticDataset,
    batch_size: int = 8192,
) -> Tuple[List[List[int]], List[Tuple[str, int, int, float]]]:
    """Compute confusion matrix and collect mislabel candidates."""
    device = next(model.parameters()).device
    model.eval()

    matrix = [[0, 0], [0, 0]]
    candidates: List[Tuple[str, int, int, float]] = []

    def collate(
        batch: List[GrammaticSample],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[str], List[int]]:
        return _collate_fn(batch, dataset)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
    )

    with torch.no_grad():
        for features, labels, sentences, _ in loader:
            features = {k: v.to(device) for k, v in features.items()}
            logits = model(features)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            confidences = probs.max(dim=-1).values

            for sent, true_cls, pred_cls, conf in zip(
                sentences,
                labels.tolist(),
                preds.cpu().tolist(),
                confidences.cpu().tolist(),
            ):
                matrix[true_cls][pred_cls] += 1
                if true_cls != pred_cls:
                    candidates.append((sent, true_cls, pred_cls, conf))

    return matrix, candidates


# pylint: disable=too-many-locals
def _print_confusion_matrix(matrix: List[List[int]]) -> None:
    """Print confusion matrix using Rich."""
    table = Table(title="Grammatic Confusion Matrix")
    table.add_column("True \\ Pred", style="bold")
    for name in CLASS_NAMES:
        table.add_column(name.capitalize(), justify="right")
    table.add_column("Total", justify="right", style="dim")

    for i, row_name in enumerate(CLASS_NAMES):
        row_total = sum(matrix[i])
        table.add_row(
            row_name.capitalize(),
            str(matrix[i][0]),
            str(matrix[i][1]),
            str(row_total),
        )

    col_totals = [sum(matrix[j][i] for j in range(2)) for i in range(2)]
    table.add_row(
        "Total",
        str(col_totals[0]),
        str(col_totals[1]),
        str(sum(col_totals)),
        style="dim",
    )

    console.print(table)


# pylint: disable=too-many-locals
def _generate_suggestion_files(
    candidates: List[Tuple[str, int, int, float]],
    output_dir: str,
    batch: int = 1,
) -> Dict[str, int]:
    """Generate suggestion files for each target class.

    Args:
        candidates: List of (sentence, true_class, pred_class, confidence)
        output_dir: Directory to write suggestion files
        batch: Which batch of 100 to write (1 = items 0-99, 2 = items 100-199, etc.)
    """
    by_pred_class: Dict[int, List[Tuple[str, int, float]]] = {
        CLASS_UNGRAMMATIC: [],
        CLASS_GRAMMATIC: [],
    }

    for sent, true_cls, pred_cls, conf in candidates:
        by_pred_class[pred_cls].append((sent, true_cls, conf))

    for pred_cls, items in by_pred_class.items():
        items.sort(key=lambda x: -x[2])

    counts = {}
    batch_size = 100
    start_idx = (batch - 1) * batch_size
    end_idx = batch * batch_size

    for pred_cls, items in by_pred_class.items():
        class_name = CLASS_NAMES[pred_cls]
        filename = f"suggest {class_name}.txt"
        filepath = os.path.join(output_dir, filename)

        # Get the specified batch
        batch_items = items[start_idx:end_idx]

        with open(filepath, "w", encoding="utf-8") as f:
            for sent, _, _ in batch_items:
                f.write(sent + "\n")

        counts[class_name] = len(batch_items)
        console.print(
            f"  Wrote {len(batch_items):,} candidates to [cyan]{filename}[/cyan] "
            f"(batch {batch}, items {start_idx}-{start_idx + len(batch_items) - 1} of {len(items):,})"
        )

    return counts


# pylint: disable=too-many-locals
def run_grammatic_study(
    db_path: str,  # pylint: disable=unused-argument
    batch: int = 1,
    percent: int = 100,
    batch_size: int = 128,
) -> None:
    """Run the grammatic learnability study.

    Args:
        db_path: Path to corpus.db
        batch: Which batch of 100 candidates to write
        percent: Percentage of data to use (for local testing)
        batch_size: Training batch size (larger for GPU)
    """
    from train import paths

    data_dir = paths.get_style_dataset_cache_dir()
    if not os.path.exists(os.path.join(data_dir, "offsets.bin")):
        console.print(
            "[bold red]Error:[/bold red] Dataset cache not found. "
            "Please run './train_style --label' first."
        )
        return

    os.makedirs(STUDY_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    console.print("[bold blue]Loading dataset...[/bold blue]")
    dataset = GrammaticDataset(data_dir)

    # Apply percent filter if specified
    if percent < 100:
        n_samples = int(len(dataset) * percent / 100)
        dataset.indices = dataset.indices[:n_samples]
        console.print(f"  Using {percent}% of data: {len(dataset):,} samples")
    else:
        console.print(f"  Total samples: {len(dataset):,}")

    candidates_path = os.path.join(STUDY_DIR, "candidates.json")
    model_path = os.path.join(MODEL_DIR, "grammatic.pt")

    # Fast path: if candidates are cached and batch > 1, just load and generate
    if batch > 1 and os.path.exists(candidates_path):
        console.print(
            f"[bold blue]Loading cached candidates from {candidates_path}...[/bold blue]"
        )
        with open(candidates_path, "r", encoding="utf-8") as f:
            candidates_data = json.load(f)
        candidates = [
            (c["sentence"], c["true_class"], c["pred_class"], c["confidence"])
            for c in candidates_data
        ]
        console.print(f"  Loaded {len(candidates):,} candidates")
    else:
        # Count class distribution
        class_counts = [0, 0]
        for sample in dataset:
            class_counts[sample.grammatic] += 1
        console.print(
            f"  Class distribution: ungrammatic={class_counts[0]:,}, "
            f"grammatic={class_counts[1]:,}"
        )

        vocab_sizes = _get_vocab_sizes(data_dir)

        # Train or load model
        if batch > 1 and os.path.exists(model_path):
            console.print(
                f"[bold blue]Loading saved model from {model_path}...[/bold blue]"
            )
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            model = GrammaticClassifier(vocab_sizes).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            console.print(f"  Device: {device}")
        else:
            model = _train_classifier(dataset, vocab_sizes, batch_size=batch_size)
            torch.save(model.state_dict(), model_path)
            console.print(f"[green]Saved model to {model_path}[/green]")

        console.print("[bold blue]Computing confusion matrix...[/bold blue]")
        matrix, candidates = _compute_confusion_matrix(model, dataset)
        _print_confusion_matrix(matrix)

        # Save candidates for future batch requests
        candidates_data = [
            {"sentence": s, "true_class": t, "pred_class": p, "confidence": c}
            for s, t, p, c in candidates
        ]
        with open(candidates_path, "w", encoding="utf-8") as f:
            json.dump(candidates_data, f)
        console.print(
            f"[green]Saved {len(candidates):,} candidates to {candidates_path}[/green]"
        )

        correct = sum(matrix[i][i] for i in range(2))
        total = sum(sum(row) for row in matrix)
        accuracy = correct / total if total > 0 else 0

        results = {
            "total_samples": len(dataset),
            "class_counts": {CLASS_NAMES[i]: class_counts[i] for i in range(2)},
            "confusion_matrix": matrix,
            "accuracy": accuracy,
        }
        results_path = os.path.join(STUDY_DIR, "results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        console.print(f"[green]Saved results to {results_path}[/green]")

    console.print(
        f"[bold blue]Generating suggestion files (batch {batch})...[/bold blue]"
    )
    _generate_suggestion_files(candidates, STUDY_DIR, batch=batch)

    console.print("\n[bold green]Study complete![/bold green]")
    console.print(f"  Output directory: {STUDY_DIR}")


# pylint: disable=too-many-locals
def apply_grammatic_changes(db_path: str) -> None:
    """Apply grammatic changes from suggestion files."""
    console.print("[bold blue]Applying grammatic changes...[/bold blue]")

    total_updates = 0
    updated_sentences: Dict[str, int] = {}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Handle "suggest ungrammatic.txt" - set grammatic to 0
    ungram_file = os.path.join(STUDY_DIR, "suggest ungrammatic.txt")
    if os.path.exists(ungram_file):
        with open(ungram_file, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]

        updated = 0
        for sent in sentences:
            cursor.execute(
                "UPDATE corpus SET grammatic = 0 WHERE sentence = ?",
                (sent,),
            )
            if cursor.rowcount > 0:
                updated += 1
                updated_sentences[sent] = 0

        conn.commit()
        total_updates += updated
        console.print(f"  Updated {updated:,} sentences to ungrammatic (grammatic=0)")
    else:
        console.print("  [dim]Skipping suggest ungrammatic.txt (not found)[/dim]")

    # Handle "suggest grammatic.txt" - set grammatic to 1
    gram_file = os.path.join(STUDY_DIR, "suggest grammatic.txt")
    if os.path.exists(gram_file):
        with open(gram_file, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]

        updated = 0
        for sent in sentences:
            cursor.execute(
                "UPDATE corpus SET grammatic = 1 WHERE sentence = ?",
                (sent,),
            )
            if cursor.rowcount > 0:
                updated += 1
                updated_sentences[sent] = 1

        conn.commit()
        total_updates += updated
        console.print(f"  Updated {updated:,} sentences to grammatic (grammatic=1)")
    else:
        console.print("  [dim]Skipping suggest grammatic.txt (not found)[/dim]")

    # Verification
    console.print("[bold blue]Verifying updates...[/bold blue]")
    verified = 0
    mismatches = 0

    for sent, expected_value in updated_sentences.items():
        cursor.execute("SELECT grammatic FROM corpus WHERE sentence = ?", (sent,))
        row = cursor.fetchone()
        if row is None:
            console.print(f"  [red]Missing:[/red] {sent[:40]}...")
            mismatches += 1
        elif row[0] != expected_value:
            console.print(
                f"  [red]Mismatch:[/red] expected {expected_value}, "
                f"got {row[0]} for: {sent[:40]}..."
            )
            mismatches += 1
        else:
            verified += 1

    conn.close()

    if mismatches == 0:
        console.print(f"  [green]✓ Verified all {verified:,} updates[/green]")
    else:
        console.print(
            f"  [red]✗ {mismatches:,} mismatches, {verified:,} verified[/red]"
        )

    console.print(
        f"[bold green]Applied {total_updates:,} total updates to {db_path}[/bold green]"
    )

    # Clean up stale caches - they need to be regenerated after label changes
    candidates_path = os.path.join(STUDY_DIR, "candidates.json")
    model_path = os.path.join(MODEL_DIR, "grammatic.pt")
    if os.path.exists(candidates_path):
        os.remove(candidates_path)
        console.print(f"  [dim]Removed stale {candidates_path}[/dim]")
    if os.path.exists(model_path):
        os.remove(model_path)
        console.print(f"  [dim]Removed stale {model_path}[/dim]")
    console.print(
        "[yellow]Note: Run './train_style --label' to rebuild dataset cache, "
        "then run 'scripts/curate study grammatic' to retrain.[/yellow]"
    )
