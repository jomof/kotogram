#!/usr/bin/env python3
"""
Gender learnability study for corpus.db.

Trains a focused classifier to learn gender labels and identifies potential mislabels
by finding sentences where the model's predictions disagree with the ground truth.
"""

import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from rich.console import Console
from rich.table import Table
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scripts.progress_utils import create_progress

console = Console()

# Gender class discretization thresholds (matches kotogram.constants)
MASCULINE_THRESHOLD = -0.5
FEMININE_THRESHOLD = 0.5

# Class indices
CLASS_MASCULINE = 0
CLASS_NEUTRAL = 1
CLASS_FEMININE = 2
CLASS_NAMES = ["masculine", "neutral", "feminine"]

# Output directory
STUDY_DIR = ".cache/curate/study/gender"


def _discretize_gender(value: float) -> int:
    """Convert continuous gender value to class index."""
    if value <= MASCULINE_THRESHOLD:
        return CLASS_MASCULINE
    if value >= FEMININE_THRESHOLD:
        return CLASS_FEMININE
    return CLASS_NEUTRAL


@dataclass
class GenderSample:
    """A sample for gender classification."""

    idx: int
    sentence: str
    gender_value: float
    gender_class: int
    feature_start: int
    feature_end: int


class GenderStudyDataset(Dataset[GenderSample]):
    """Dataset for gender classification study using pre-computed features."""

    def __init__(
        self,
        data_dir: str,
        indices: Optional[torch.Tensor] = None,
    ):
        self.data_dir = data_dir

        # Load offsets
        offsets_path = os.path.join(data_dir, "offsets.bin")
        size_bytes = os.path.getsize(offsets_path)
        self.offsets = torch.from_file(
            offsets_path, shared=True, size=size_bytes // 4, dtype=torch.int32
        )

        # Load gender values
        g_val_path = os.path.join(data_dir, "labels.bin_g_val")
        g_size = os.path.getsize(g_val_path) // 4
        self.gender_values = torch.from_file(
            g_val_path, shared=True, size=g_size, dtype=torch.float32
        )

        # Load gender pragmatic flags
        g_prag_path = os.path.join(data_dir, "labels.bin_g_prag")
        g_prag_size = os.path.getsize(g_prag_path)
        self.gender_pragmatic = torch.from_file(
            g_prag_path, shared=True, size=g_prag_size, dtype=torch.uint8
        )

        # Load sentences
        sentences_path = os.path.join(data_dir, "sentences.txt")
        with open(sentences_path, "r", encoding="utf-8") as f:
            self.sentences = [line.strip() for line in f]

        # Filter to pragmatic gender samples only
        total_samples = len(self.offsets) - 1
        if indices is not None:
            self.indices = indices
        else:
            # Find pragmatic samples (g_prag == 1)
            pragmatic_mask = self.gender_pragmatic[:total_samples] == 1
            self.indices = torch.nonzero(pragmatic_mask, as_tuple=True)[0]

        # Load KC features for the model
        self._load_features()

        # Iterator state (initialized in __init__ for pylint compliance)
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
                self.features[key] = torch.from_file(
                    path, shared=True, size=size, dtype=torch.int32
                )

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> "GenderStudyDataset":
        """Iterate over samples."""
        self._iter_idx = 0
        return self

    def __next__(self) -> "GenderSample":
        """Get next sample."""
        if self._iter_idx >= len(self):
            raise StopIteration
        sample = self[self._iter_idx]
        self._iter_idx += 1
        return sample

    def __getitem__(self, idx: int) -> GenderSample:
        real_idx = int(self.indices[idx].item())
        start = int(self.offsets[real_idx].item())
        end = int(self.offsets[real_idx + 1].item())
        gender_val = float(self.gender_values[real_idx].item())
        gender_cls = _discretize_gender(gender_val)

        return GenderSample(
            idx=real_idx,
            sentence=self.sentences[real_idx],
            gender_value=gender_val,
            gender_class=gender_cls,
            feature_start=start,
            feature_end=end,
        )

    def get_feature_slice(self, start: int, end: int, field: str) -> torch.Tensor:
        """Get feature tensor slice for a sample."""
        if field in self.features:
            return self.features[field][start:end]
        return torch.tensor([], dtype=torch.int32)


class GenderClassifier(nn.Module):
    """Simple classifier for gender prediction from KC features."""

    def __init__(
        self, vocab_sizes: Dict[str, int], embed_dim: int = 32, hidden_dim: int = 128
    ):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.embed_dim = embed_dim

        # Embedding for each feature type (using Embedding instead of EmbeddingBag for MPS)
        self.embeddings = nn.ModuleDict()
        for name, vocab_size in vocab_sizes.items():
            self.embeddings[name] = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # MLP classifier
        total_embed = embed_dim * len(vocab_sizes)
        self.classifier = nn.Sequential(
            nn.Linear(total_embed, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 3),  # 3 classes
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass. features is dict of (batch, seq_len) tensors."""
        embeds = []
        for name in sorted(self.vocab_sizes.keys()):
            if name in features:
                feat = features[name]  # (batch, seq_len)
                emb = self.embeddings[name](feat)  # (batch, seq_len, embed_dim)
                # Mean pooling over sequence (ignore padding with mask)
                mask = (feat != 0).unsqueeze(-1).float()  # (batch, seq_len, 1)
                masked_emb = emb * mask
                seq_lens = mask.sum(dim=1).clamp(min=1)  # (batch, 1)
                pooled = masked_emb.sum(dim=1) / seq_lens  # (batch, embed_dim)
                embeds.append(pooled)

        combined = torch.cat(embeds, dim=-1)
        result: torch.Tensor = self.classifier(combined)
        return result


def _collate_fn(
    batch: List[GenderSample],
    dataset: GenderStudyDataset,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[str], List[int]]:
    """Collate samples into batched tensors."""
    labels = torch.tensor([s.gender_class for s in batch], dtype=torch.long)
    sentences = [s.sentence for s in batch]
    indices = [s.idx for s in batch]

    # Find max length
    max_len = max(s.feature_end - s.feature_start for s in batch)

    # Batch features
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
            # vocab is token->id mapping, we need max(values) + 1
            field_vocab = field_vocabs[field]
            max_id = max(field_vocab.values())
            sizes[field] = max_id + 1
        else:
            sizes[field] = 1000  # fallback

    return sizes


# pylint: disable=too-many-locals
def _train_classifier(
    dataset: GenderStudyDataset,
    vocab_sizes: Dict[str, int],
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> GenderClassifier:
    """Train a gender classifier on the dataset."""
    # Device selection: prefer MPS (Metal) on macOS, then CUDA, then CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = GenderClassifier(vocab_sizes).to(device)

    # Split into train/val
    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    perm = torch.randperm(n_total)
    train_indices = dataset.indices[perm[:n_train]]
    val_indices = dataset.indices[perm[n_train:]]

    train_ds = GenderStudyDataset(dataset.data_dir, indices=train_indices)
    val_ds = GenderStudyDataset(dataset.data_dir, indices=val_indices)

    # Share loaded features  # pylint: disable=attribute-defined-outside-init
    train_ds.features = dataset.features
    train_ds.offsets = dataset.offsets
    train_ds.gender_values = dataset.gender_values
    train_ds.sentences = dataset.sentences

    val_ds.features = dataset.features  # pylint: disable=attribute-defined-outside-init
    val_ds.offsets = dataset.offsets
    val_ds.gender_values = dataset.gender_values
    val_ds.sentences = dataset.sentences

    def collate(
        batch: List[GenderSample],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[str], List[int]]:
        return _collate_fn(batch, dataset)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate
    )

    # Compute class weights using inverse frequency (industry best practice for imbalanced data)
    # Count classes in training set
    class_counts = torch.zeros(3, dtype=torch.float32)
    for sample in train_ds:
        class_counts[sample.gender_class] += 1

    # Inverse frequency weighting: weight = total / (n_classes * count)
    # This gives higher weight to minority classes
    total_samples = class_counts.sum()
    class_weights = total_samples / (3 * class_counts.clamp(min=1))
    # Normalize so weights sum to n_classes (optional, helps with learning rate)
    class_weights = class_weights / class_weights.sum() * 3
    class_weights = class_weights.to(device)

    console.print("[bold blue]Training gender classifier[/bold blue]")
    console.print(f"  Train: {len(train_ds):,}, Val: {len(val_ds):,}")
    console.print(f"  Device: {device}")
    console.print(
        f"  Class weights: masculine={class_weights[0]:.2f}, "
        f"neutral={class_weights[1]:.2f}, feminine={class_weights[2]:.2f}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0
    best_state = model.state_dict()

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        import time

        start_time = time.perf_counter()
        samples_processed = 0
        with create_progress(console) as progress:
            task = progress.add_task(
                f"Epoch {epoch + 1}/{epochs} training...", total=len(train_loader)
            )
            for features, labels, _, _ in train_loader:
                features = {k: v.to(device) for k, v in features.items()}
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = model(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                samples_processed += len(labels)
                elapsed = time.perf_counter() - start_time
                rate = samples_processed / elapsed if elapsed > 0 else 0
                progress.update(
                    task,
                    description=f"Epoch {epoch + 1}/{epochs} [{rate:,.0f} sent/s]",
                    advance=1,
                )

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels, _, _ in val_loader:
                features = {k: v.to(device) for k, v in features.items()}
                labels = labels.to(device)
                logits = model(features)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += len(labels)

        val_acc = correct / total if total > 0 else 0
        console.print(
            f"  Epoch {epoch + 1}/{epochs}: "
            f"Loss={train_loss / len(train_loader):.4f}, "
            f"Val Acc={val_acc:.2%}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

    model.load_state_dict(best_state)
    console.print(f"[green]Best validation accuracy: {best_val_acc:.2%}[/green]")
    return model


# pylint: disable=too-many-locals
def _compute_confusion_matrix(
    model: GenderClassifier,
    dataset: GenderStudyDataset,
    batch_size: int = 8192,
) -> Tuple[List[List[int]], List[Tuple[str, int, int, float]]]:
    """Compute confusion matrix and collect mislabel candidates.

    Returns:
        (confusion_matrix, mislabel_candidates)
        where mislabel_candidates is list of (sentence, true_class, pred_class, confidence)
    """
    device = next(model.parameters()).device
    model.eval()

    matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    candidates: List[Tuple[str, int, int, float]] = []

    def collate(
        batch: List[GenderSample],
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
                # Collect mismatches as mislabel candidates
                if true_cls != pred_cls:
                    candidates.append((sent, true_cls, pred_cls, conf))

    return matrix, candidates


def _print_confusion_matrix(matrix: List[List[int]]) -> None:
    """Print confusion matrix using Rich."""
    table = Table(title="Gender Confusion Matrix")
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
            str(matrix[i][2]),
            str(row_total),
        )

    # Add column totals
    col_totals = [sum(matrix[j][i] for j in range(3)) for i in range(3)]
    table.add_row(
        "Total",
        str(col_totals[0]),
        str(col_totals[1]),
        str(col_totals[2]),
        str(sum(col_totals)),
        style="dim",
    )

    console.print(table)


def _generate_suggestion_files(
    candidates: List[Tuple[str, int, int, float]],
    output_dir: str,
) -> Dict[str, int]:
    """Generate suggestion files for each target class.

    Returns dict of class_name -> count of suggestions.
    """
    # Group by predicted class
    by_pred_class: Dict[int, List[Tuple[str, int, float]]] = {
        CLASS_MASCULINE: [],
        CLASS_NEUTRAL: [],
        CLASS_FEMININE: [],
    }

    for sent, true_cls, pred_cls, conf in candidates:
        by_pred_class[pred_cls].append((sent, true_cls, conf))

    # Sort by confidence descending within each group
    for pred_cls, items in by_pred_class.items():
        items.sort(key=lambda x: -x[2])

    counts = {}
    max_suggestions = 100
    for pred_cls, items in by_pred_class.items():
        class_name = CLASS_NAMES[pred_cls]
        filename = f"suggest {class_name}.txt"
        filepath = os.path.join(output_dir, filename)

        # Take only top N most confident
        top_items = items[:max_suggestions]

        with open(filepath, "w", encoding="utf-8") as f:
            for sent, _, _ in top_items:
                f.write(sent + "\n")

        counts[class_name] = len(top_items)
        console.print(
            f"  Wrote {len(top_items):,} candidates to [cyan]{filename}[/cyan] "
            f"(of {len(items):,} total)"
        )

    return counts


# pylint: disable=too-many-locals
def run_gender_study(db_path: str) -> None:  # pylint: disable=unused-argument
    """Run the gender learnability study."""
    from train import paths

    # Ensure cache directory exists
    data_dir = paths.get_style_dataset_cache_dir()
    if not os.path.exists(os.path.join(data_dir, "offsets.bin")):
        console.print(
            "[bold red]Error:[/bold red] Dataset cache not found. "
            "Please run './train_style --label' first."
        )
        return

    os.makedirs(STUDY_DIR, exist_ok=True)

    console.print("[bold blue]Loading dataset...[/bold blue]")
    dataset = GenderStudyDataset(data_dir)
    console.print(f"  Total pragmatic gender samples: {len(dataset):,}")

    # Count class distribution
    class_counts = [0, 0, 0]
    for sample in dataset:
        class_counts[sample.gender_class] += 1
    console.print(
        f"  Class distribution: masculine={class_counts[0]:,}, "
        f"neutral={class_counts[1]:,}, feminine={class_counts[2]:,}"
    )

    # Get vocab sizes
    vocab_sizes = _get_vocab_sizes(data_dir)

    # Train classifier
    model = _train_classifier(dataset, vocab_sizes, epochs=10)

    # Save model
    model_path = os.path.join(STUDY_DIR, "model.pt")
    torch.save(model.state_dict(), model_path)
    console.print(f"[green]Saved model to {model_path}[/green]")

    # Compute confusion matrix
    console.print("[bold blue]Computing confusion matrix...[/bold blue]")
    matrix, candidates = _compute_confusion_matrix(model, dataset)
    _print_confusion_matrix(matrix)

    # Generate suggestion files
    console.print("[bold blue]Generating suggestion files...[/bold blue]")
    suggestion_counts = _generate_suggestion_files(candidates, STUDY_DIR)

    # Compute accuracy
    correct = sum(matrix[i][i] for i in range(3))
    total = sum(sum(row) for row in matrix)
    accuracy = correct / total if total > 0 else 0

    # Save results
    results = {
        "total_samples": len(dataset),
        "class_counts": {CLASS_NAMES[i]: class_counts[i] for i in range(3)},
        "confusion_matrix": matrix,
        "accuracy": accuracy,
        "suggestion_counts": suggestion_counts,
    }
    results_path = os.path.join(STUDY_DIR, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    console.print(f"[green]Saved results to {results_path}[/green]")

    console.print("\n[bold green]Study complete![/bold green]")
    console.print(f"  Accuracy: {accuracy:.2%}")
    console.print(f"  Output directory: {STUDY_DIR}")


# pylint: disable=too-many-locals
def apply_gender_changes(db_path: str) -> None:
    """Apply gender label changes from suggestion files."""
    console.print("[bold blue]Applying gender label changes...[/bold blue]")

    # Map class name to gender value
    class_to_value = {
        "masculine": -1.0,
        "neutral": 0.0,
        "feminine": 1.0,
    }

    total_updates = 0
    # Track sentences we updated for verification
    updated_sentences: Dict[str, float] = {}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for class_name, gender_value in class_to_value.items():
        filename = f"suggest {class_name}.txt"
        filepath = os.path.join(STUDY_DIR, filename)

        if not os.path.exists(filepath):
            console.print(f"  [dim]Skipping {filename} (not found)[/dim]")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]

        if not sentences:
            console.print(f"  [dim]Skipping {filename} (empty)[/dim]")
            continue

        # Update each sentence
        updated = 0
        for sent in sentences:
            cursor.execute(
                "UPDATE corpus SET gender = ? WHERE sentence = ?",
                (gender_value, sent),
            )
            if cursor.rowcount > 0:
                updated += 1
                updated_sentences[sent] = gender_value

        conn.commit()
        total_updates += updated
        console.print(f"  Updated {updated:,} sentences to {class_name}")

    # Verification: read back and confirm values
    console.print("[bold blue]Verifying updates...[/bold blue]")
    verified = 0
    mismatches = 0

    for sent, expected_value in updated_sentences.items():
        cursor.execute("SELECT gender FROM corpus WHERE sentence = ?", (sent,))
        row = cursor.fetchone()
        if row is None:
            console.print(f"  [red]Missing:[/red] {sent[:40]}...")
            mismatches += 1
        elif abs(row[0] - expected_value) > 0.001:
            console.print(
                f"  [red]Mismatch:[/red] expected {expected_value}, got {row[0]} for: {sent[:40]}..."
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
