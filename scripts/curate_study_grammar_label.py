#!/usr/bin/env python3
# pylint: disable=too-many-lines,too-many-locals,too-many-positional-arguments,wrong-import-position,duplicate-code
"""
Hard negative/positive mining for grammar points using PNU (Positive-Negative-Unlabeled) learning.

This script trains a lightweight classifier for a specific grammar point and identifies
hard negative and positive candidates based on loss contribution and prediction uncertainty.

Usage:
    python curate_study_grammar_label.py gp0888
    python curate_study_grammar_label.py gp0888 --apply
"""

import argparse
import copy
import heapq
import json
import math
import os
import random
import shutil
import sqlite3
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

# Add project root to path
if os.path.exists("kotogram"):
    sys.path.insert(0, os.getcwd())

from kotogram.model import ModelConfig, PositionalEncoding
from kotogram.tokenizer import ENCODER_FEATURE_FIELDS, Tokenizer
from scripts.curate_upsert_sentence import curate_upsert_batch
from scripts.progress_utils import create_progress
from train.dataset import StyleDataset
from train.paths import get_style_dataset_cache_dir

console = Console(force_terminal=True)


class GrammarPointDataset(Dataset):
    """Wraps StyleDataset to provide grammar point labels."""

    def __init__(
        self,
        base_dataset: StyleDataset,
        grammar_labels: List[str],
        verbose: bool = True,
    ) -> None:
        self.base_dataset = base_dataset
        self.grammar_labels = grammar_labels
        self.verbose = verbose

        # Load grammar point labels
        self._load_grammar_labels()

    def _load_grammar_labels(self) -> None:
        """Load grammar point positive/negative labels from database for ALL GPs."""
        if self.verbose:
            # console.print("  Loading grammar labels from database...")
            pass

        db_path = os.path.join(
            os.path.dirname(self.base_dataset.data_dir), "..", "data", "corpus.db"
        )

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # sentence -> {gp_index: label}
        # label: 1=pos, 0=neg (if explicitly listed)
        self.sentence_labels: Dict[str, Dict[int, int]] = {}

        self.gp_stats = []

        # Iterate with progress bar if verbose
        progress_ctx = create_progress(console) if self.verbose else None

        try:
            if progress_ctx:
                progress = progress_ctx.__enter__()
                task = progress.add_task(
                    "Preparing grammar point datasets...",
                    total=len(self.grammar_labels),
                )

            for i, gp_label in enumerate(self.grammar_labels):
                # if self.verbose:
                #    console.print(f"  Processing {gp_label}...")

                cursor.execute(
                    """
                    SELECT sentence,
                           CASE WHEN grammar LIKE '%' || ? || '%' THEN 1 ELSE 0 END as positive,
                           CASE WHEN grammar_negative LIKE '%' || ? || '%' THEN 1 ELSE 0 END as negative
                    FROM corpus
                    WHERE grammar LIKE '%' || ? || '%' OR grammar_negative LIKE '%' || ? || '%'
                    """,
                    (gp_label, gp_label, gp_label, gp_label),
                )

                rows = cursor.fetchall()
                pos_count = 0
                neg_count = 0

                for sentence, is_pos, is_neg in rows:
                    if sentence not in self.sentence_labels:
                        self.sentence_labels[sentence] = {}

                    if is_pos:
                        self.sentence_labels[sentence][i] = 1
                        pos_count += 1
                    elif is_neg:
                        self.sentence_labels[sentence][i] = 0
                        neg_count += 1

                # Synthetic negatives if needed
                min_negatives = 20
                if neg_count < min_negatives:
                    num_needed = min_negatives - neg_count
                    # Silent synthetic sampling
                    # if self.verbose:
                    #    console.print(
                    #        f"    [yellow]Synthetically sampling {num_needed} negatives for {gp_label}[/yellow]"
                    #    )
                    added = 0
                    attempts = 0
                    while added < num_needed and attempts < num_needed * 50:
                        idx = random.randint(0, len(self.base_dataset) - 1)
                        sentence = self.base_dataset.get_sentence_by_idx(
                            self.base_dataset[idx].idx
                        )

                        if sentence not in self.sentence_labels:
                            self.sentence_labels[sentence] = {}

                        # Only mark if not already labeled for this GP
                        if i not in self.sentence_labels[sentence]:
                            self.sentence_labels[sentence][i] = 0
                            neg_count += 1
                            added += 1
                        attempts += 1

                # Stats
                total = len(self.base_dataset)
                unlabeled = total - (pos_count + neg_count)
                self.gp_stats.append(
                    {
                        "label": gp_label,
                        "pos": pos_count,
                        "neg": neg_count,
                        "unlabeled": unlabeled,
                        "total": total,
                    }
                )

                # if self.verbose:
                #    console.print(
                #        f"    Pos: {pos_count}, Neg: {neg_count}, Dens: {100.0 * (pos_count + neg_count) / total:.2f}%"
                #    )

                if progress_ctx:
                    progress.advance(task)

        finally:
            if progress_ctx:
                progress_ctx.__exit__(None, None, None)

        conn.close()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample with grammar point labels."""
        sample = self.base_dataset[idx]
        sentence = self.base_dataset.get_sentence_by_idx(sample.idx)

        # Get labels for all GPs
        # Returns vector of shape [Num_GPs].
        # We use -1 for unlabeled.

        labels = []
        if sentence in self.sentence_labels:
            sent_map = self.sentence_labels[sentence]
            for i in range(len(self.grammar_labels)):
                labels.append(sent_map.get(i, -1))
        else:
            labels = [-1] * len(self.grammar_labels)

        # Return list, collate will convert to tensor
        return {
            "sample": sample,
            "labels": labels,  # List[int]
            "sentence": sentence,
            "idx": sample.idx,
        }


class GrammarClassifier(nn.Module):
    """Lightweight binary classifier for grammar point detection."""

    def __init__(self, config: ModelConfig, num_classes: int = 1):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        # Reduce model size for faster training
        self.d_model = 256
        self.hidden_dim = 1024
        self.num_layers = 2
        self.num_heads = 8

        # Embedding layer (same architecture as main model)
        self.embeddings = nn.ModuleDict()
        total_embed_dim = 0

        for field_name in ENCODER_FEATURE_FIELDS:
            vocab_size = config.vocab_sizes.get(field_name, 100)
            embed_dim = config.field_embed_dims.get(field_name, 32)
            self.embeddings[field_name] = nn.Embedding(
                vocab_size, embed_dim, padding_idx=0
            )
            total_embed_dim += embed_dim

        self.projection = nn.Linear(total_embed_dim, self.d_model)
        self.layer_norm_embed = nn.LayerNorm(self.d_model)
        self.dropout_embed = nn.Dropout(0.1)

        # Position encoding
        self.position_encoding = PositionalEncoding(self.d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        # Disable nested tensor optimization for MPS compatibility
        self.encoder = nn.TransformerEncoder(
            encoder_layer, self.num_layers, enable_nested_tensor=False
        )

        # Attention pooling
        self.pooler_query = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.pooler_attention = nn.MultiheadAttention(
            self.d_model, self.num_heads, dropout=0.1, batch_first=True
        )
        self.layer_norm_pool = nn.LayerNorm(self.d_model)

        # Multi-task classification head
        # Output shape: [batch, num_classes, 2]
        # We use a shared hidden layer then split? Or just one big linear?
        # Let's use independent linear heads for each class if num_classes is small,
        # or one big linear layer [hidden, num_classes * 2].
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, num_classes * 2),
        )

    def forward(
        self, field_inputs: Dict[str, torch.Tensor], attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            field_inputs: Dict of token feature tensors
            attention_mask: Attention mask (1=valid, 0=padding)

        Returns:
            logits: [B, num_classes, 2] logits
        """
        # Embed
        field_embeds = []
        for field_name in ENCODER_FEATURE_FIELDS:
            input_ids = field_inputs[f"input_ids_{field_name}"]
            embed = self.embeddings[field_name](input_ids)
            field_embeds.append(embed)

        x = torch.cat(field_embeds, dim=-1)
        x = self.projection(x)
        x = self.layer_norm_embed(x)
        x = self.dropout_embed(x)

        # Position encoding
        x = self.position_encoding(x)

        # Encoder
        src_key_padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Attention pooling
        batch_size = x.size(0)
        query = self.pooler_query.expand(batch_size, -1, -1)
        key_padding_mask = attention_mask == 0
        attn_output, _ = self.pooler_attention(
            query=query, key=x, value=x, key_padding_mask=key_padding_mask
        )
        pooled = attn_output.squeeze(1)
        pooled = self.layer_norm_pool(pooled)

        # Classify
        flat_logits = self.classifier(pooled)  # [B, num_classes * 2]

        # Reshape to [B, num_classes, 2]
        batch_size = flat_logits.size(0)
        logits = flat_logits.view(batch_size, self.num_classes, 2)

        return cast(torch.Tensor, logits)


class EarlyStopper:
    """Early stopping with learning rate decay."""

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0001,
        decay_factor: float = 0.7,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.decay_factor = decay_factor
        self.counter = 0
        self.best_loss = float("inf")
        self.prev_loss = float("inf")

    def check(
        self, current_loss: float, current_batch_size: int, is_best: bool = True
    ) -> Tuple[bool, Optional[int]]:
        """Check if training should stop or batch size should change.

        Returns:
            Tuple[bool, Optional[int]]: (should_stop, new_batch_size)
        """
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            self.prev_loss = current_loss
            return False, None

        self.counter += 1
        if self.counter >= self.patience:
            return True, None

        # Check for local improvement (even if not global best)
        # If we are improving locally, don't switch batch size yet
        if current_loss < self.prev_loss:
            self.prev_loss = current_loss

            msg = f"      Loss plateaued ({self.counter}/{self.patience})."
            if not is_best:
                console.print(f"[dim]{msg}[/dim]")
            else:
                console.print(f"[dim]{msg}[/dim]")  # Always dim this minor warning
            return False, None

        # Loss regressed or stagnated locally AND globally -> Switch batch size
        self.prev_loss = current_loss

        # Rotate between fixed large batch sizes
        # Ensure we actually pick a different batch size
        choices = [1024, 2048]
        if current_batch_size in choices:
            new_batch_size = (
                choices[1] if current_batch_size == choices[0] else choices[0]
            )
        else:
            new_batch_size = random.choice(choices)

        msg = f"      Loss plateaued ({self.counter}/{self.patience}). Changing batch size from {current_batch_size} to {new_batch_size}"
        if not is_best:
            console.print(f"[dim]{msg}[/dim]")
        else:
            console.print(f"[yellow]{msg}[/yellow]")
        return False, new_batch_size


def collate_grammar_batch(batch: List[Dict]) -> Dict:
    """Collate batch for grammar point training."""
    samples = [item["sample"] for item in batch]
    # labels is List[List[int]] -> Tensor [B, Num_GPs]
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    sentences = [item["sentence"] for item in batch]
    indices = [item["idx"] for item in batch]

    # Get max sequence length from feature_ids
    max_len = max(len(s.feature_ids["pos"]) for s in samples)

    # Prepare field inputs
    batch_size = len(samples)
    field_inputs = {}

    for field_name in ENCODER_FEATURE_FIELDS:
        field_tensor = torch.zeros(batch_size, max_len, dtype=torch.long)
        for i, sample in enumerate(samples):
            feature_data = sample.feature_ids[field_name]
            seq_len = len(feature_data)
            # Convert to tensor if it's a list
            if isinstance(feature_data, list):
                field_tensor[i, :seq_len] = torch.tensor(feature_data, dtype=torch.long)
            else:
                field_tensor[i, :seq_len] = feature_data
        field_inputs[f"input_ids_{field_name}"] = field_tensor

    # Attention mask
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, sample in enumerate(samples):
        seq_len = len(sample.feature_ids["pos"])
        attention_mask[i, :seq_len] = 1

    return {
        "field_inputs": field_inputs,
        "attention_mask": attention_mask,
        "labels": labels,
        "sentences": sentences,
        "indices": indices,
    }


def focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> torch.Tensor:
    """Focal loss for handling class imbalance.

    Args:
        logits: [B, 2] logits
        targets: [B] targets (0 or 1)
        alpha: Weighting factor for positive class
        gamma: Focusing parameter (higher = more focus on hard examples)

    Returns:
        loss: Scalar focal loss
    """
    ce_loss = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce_loss)
    focal_weight = (1 - pt) ** gamma

    # Alpha weighting
    alpha_t = torch.where(targets == 1, alpha, 1 - alpha)

    loss = alpha_t * focal_weight * ce_loss
    return cast(torch.Tensor, loss.mean())


def compute_accuracy_stats(
    model: GrammarClassifier,
    loader: DataLoader,
    device: torch.device,
    grammar_labels: List[str],
) -> List[Tuple[int, int, int, int]]:
    """Compute accuracy statistics for each GP.

    Returns:
        List of tuples: (learned_pos, total_pos, learned_neg, total_neg) for each GP
    """
    model.eval()
    num_gps = len(grammar_labels)

    stats = []  # List of [learned_pos, total_pos, learned_neg, total_neg]
    for _ in range(num_gps):
        stats.append([0, 0, 0, 0])

    with torch.no_grad():
        for batch in loader:
            field_inputs = {k: v.to(device) for k, v in batch["field_inputs"].items()}
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)  # [B, Num_GPs]

            # Forward
            logits = model(field_inputs, attention_mask)  # [B, Num_GPs, 2]
            preds = torch.argmax(logits, dim=-1)  # [B, Num_GPs]

            # Loop over GPs
            for i in range(num_gps):
                gp_labels = labels[:, i]
                gp_preds = preds[:, i]

                # Stats
                true_pos_mask = gp_labels == 1
                true_neg_mask = gp_labels == 0

                stats[i][1] += true_pos_mask.sum().item()  # total_pos
                stats[i][3] += true_neg_mask.sum().item()  # total_neg

                stats[i][0] += int(
                    (gp_preds[true_pos_mask] == 1).sum().item()
                )  # learned_pos
                stats[i][2] += int(
                    (gp_preds[true_neg_mask] == 0).sum().item()
                )  # learned_neg

    return [cast(Tuple[int, int, int, int], tuple(s)) for s in stats]


def write_unlearned_samples(
    model: GrammarClassifier,
    loader: DataLoader,
    device: torch.device,
    base_output_dir: str,
    grammar_labels: List[str],
    verbose: bool = True,
) -> List[Tuple[int, int, int, int]]:
    """Identify and write misclassified labeled samples (unlearned) for each GP.

    Returns:
        List of tuples: (hard_pos_learned, hard_pos_total, hard_neg_learned, hard_neg_total) for each GP
    """
    model.eval()
    num_gps = len(grammar_labels)

    # Per GP lists
    unlearned_samples: List[List[Tuple[str, int, int]]] = [[] for _ in range(num_gps)]

    stats = []  # List of [learned_pos, total_pos, learned_neg, total_neg]
    for _ in range(num_gps):
        stats.append([0, 0, 0, 0])

    with torch.no_grad():
        for batch in loader:
            field_inputs = {k: v.to(device) for k, v in batch["field_inputs"].items()}
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)  # [B, Num_GPs]
            sentences = batch["sentences"]

            # Forward
            logits = model(field_inputs, attention_mask)  # [B, Num_GPs, 2]
            preds = torch.argmax(logits, dim=-1)  # [B, Num_GPs]

            # Loop over GPs
            for i in range(num_gps):
                gp_labels = labels[:, i]
                gp_preds = preds[:, i]

                # Mask for valid labels
                valid_mask = gp_labels >= 0
                if not valid_mask.any():
                    continue

                # Mismatches (valid only)
                mismatches = (gp_labels != gp_preds) & valid_mask
                if mismatches.any():
                    mismatch_indices = torch.nonzero(mismatches).squeeze(1)
                    for idx in mismatch_indices:
                        sentence = sentences[idx]
                        true_label = gp_labels[idx].item()
                        pred_label = gp_preds[idx].item()
                        unlearned_samples[i].append(
                            (sentence, int(true_label), int(pred_label))
                        )

                # Stats
                true_pos_mask = gp_labels == 1
                true_neg_mask = gp_labels == 0

                stats[i][1] += true_pos_mask.sum().item()  # total_pos
                stats[i][3] += true_neg_mask.sum().item()  # total_neg

                stats[i][0] += int(
                    (gp_preds[true_pos_mask] == 1).sum().item()
                )  # learned_pos
                stats[i][2] += int(
                    (gp_preds[true_neg_mask] == 0).sum().item()
                )  # learned_neg

    # Write files for each GP
    # Header moved to caller for better log placement

    for i, gp in enumerate(grammar_labels):
        out_dir = os.path.join(base_output_dir, gp)
        os.makedirs(out_dir, exist_ok=True)
        unlearned_file = os.path.join(out_dir, "unlearned.txt")

        with open(unlearned_file, "w", encoding="utf-8") as f:
            f.write(f"# Unlearned Samples for {gp}\n")
            f.write("# Format: True_Label -> Pred_Label | Sentence\n")
            f.write("# " + "-" * 70 + "\n")

            unlearned_samples[i].sort(key=lambda x: (x[1], x[0]), reverse=True)
            for sentence, true_label, pred_label in unlearned_samples[i]:
                true_str = "POS" if true_label == 1 else "NEG"
                pred_str = "POS" if pred_label == 1 else "NEG"
                f.write(f"{true_str} -> {pred_str} | {sentence}\n")

        if verbose and len(unlearned_samples[i]) > 0:
            console.print(
                f"   Wrote {len(unlearned_samples[i])} unlearned samples to {unlearned_file}"
            )

    return [cast(Tuple[int, int, int, int], tuple(s)) for s in stats]


def train_pnu_model(  # pylint: disable=unused-argument
    grammar_labels: List[str],
    dataset: GrammarPointDataset,
    tokenizer: Tokenizer,
    device: torch.device,
    base_output_dir: str,
    num_epochs_warmup: int = 5,
    num_epochs_pnu: int = 15,
    batch_size: int = 32,
    test_mode: bool = False,
    quick_mode: bool = False,
    full_pos_scan_fn: Optional[Callable] = None,
) -> Tuple[GrammarClassifier, List[int]]:
    """Train PNU model and collect loss statistics.

    Args:
        test_mode: If True, stop scanning once we have at least 5 pos/neg/unlabeled samples.

    Returns:
        model: Trained model
        unlabeled_indices: List of indices used as unlabeled pool (important if filtered in test_mode)
    """
    # Create config
    config = ModelConfig(vocab_sizes=tokenizer.get_vocab_sizes())

    # Create model
    num_gps = len(grammar_labels)
    model = GrammarClassifier(config, num_classes=num_gps)
    model = model.to(device)

    # Check for existing model to resume/fine-tune
    model_path = os.path.join(base_output_dir, "model.pt")
    model_reused = False
    model_reuse_reason: Optional[str] = None
    if os.path.exists(model_path):
        checkpoint_state = torch.load(model_path, map_location=device)

        # Check for vocabulary mismatches
        vocab_mismatch = False
        for key, param in model.named_parameters():
            if key in checkpoint_state:
                if checkpoint_state[key].shape != param.shape:
                    console.print(
                        f"[yellow]Shape mismatch for {key}: "
                        f"Checkpoint {checkpoint_state[key].shape} != Model {param.shape}[/yellow]"
                    )
                    vocab_mismatch = True
                    break

        if vocab_mismatch:
            console.print(
                "[yellow]Vocabulary size mismatch detected. Starting with a fresh model.[/yellow]"
            )
            model_reuse_reason = "shape mismatch"
        else:
            model.load_state_dict(checkpoint_state)
            model_reused = True
            console.print(
                f"[green]Resuming/Fine-tuning from existing model: {model_path}[/green]"
            )

    # Enable automatic mixed precision for fp16 (not on MPS due to float64 limitation)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda" if device.type == "cuda" else "cpu", enabled=use_amp
    )

    # Compute class weights per GP
    pos_weights = []
    for gp_stat in dataset.gp_stats:
        # Cast to float to avoid mypy overload issues with max(int, float)
        pw = float(cast(Any, gp_stat["neg"])) / max(int(cast(Any, gp_stat["pos"])), 1)
        pw = min(pw, 50.0)  # Cap at 50x
        pos_weights.append(pw)

    console.print("\n[bold]Training Configuration:[/bold]")
    console.print(f"  Device: {device}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Grammar Points: {len(grammar_labels)}")
    console.print(f"  Use AMP: {use_amp}")

    # Show weights table
    table = Table(box=None, show_header=True, pad_edge=False)
    table.add_column("", style="cyan", header_style="purple")
    table.add_column("pos_weight", justify="left", header_style="purple")

    rows = []
    for i, gp in enumerate(grammar_labels):
        # Store weight as float for sorting
        rows.append((pos_weights[i], gp, f"{pos_weights[i]:.2f}"))

    # Sort descending by weight
    rows.sort(key=lambda x: x[0], reverse=True)

    # Strip sort key
    display_rows = [r[1:] for r in rows]

    n_head = 2
    n_tail = 2

    if len(display_rows) <= (n_head + n_tail):
        for r in display_rows:
            table.add_row(*r)
    else:
        for r in display_rows[:n_head]:
            table.add_row(*r)

        remaining = len(rows) - n_head - n_tail
        table.add_row(f"[dim][{remaining} omitted][/dim]", "")

        for r in display_rows[-n_tail:]:
            table.add_row(*r)

    console.print(Padding(table, (0, 0, 0, 6)))

    # Split into labeled and unlabeled - optimize by reading sentences file once
    console.print("\n  Splitting into labeled/unlabeled sets...")

    split_cache_path = os.path.join(base_output_dir, "split.json")

    # We consider "Labeled" if ANY GP has a label >= 0
    labeled_indices: List[int] = []
    unlabeled_indices: List[int] = []

    use_cached_split = False
    split_skip_reason: Optional[str] = None
    if model_reused:
        if not os.path.exists(split_cache_path):
            split_skip_reason = "split cache missing"
        else:
            with open(split_cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            reasons = []
            if cached.get("grammar_labels") != grammar_labels:
                reasons.append("grammar_labels mismatch")
            if cached.get("dataset_size") != len(dataset):
                reasons.append("dataset_size mismatch")
            if not isinstance(cached.get("labeled_indices"), list) or not isinstance(
                cached.get("unlabeled_indices"), list
            ):
                reasons.append("invalid index lists")
            if not reasons:
                labeled_indices = cached["labeled_indices"]
                unlabeled_indices = cached["unlabeled_indices"]
                use_cached_split = True
                console.print(
                    f"[dim]  Loaded cached split from {split_cache_path}[/dim]"
                )
            else:
                split_skip_reason = ", ".join(reasons)

    if not use_cached_split:
        if model_reused:
            console.print(
                f"[yellow]  Model reuse detected, but split cache was not used "
                f"({split_cache_path}): {split_skip_reason}. "
                f"Rescanning dataset...[/yellow]"
            )
        elif os.path.exists(model_path):
            reason_text = model_reuse_reason or "checkpoint not reused"
            console.print(
                f"[yellow]  Model checkpoint found but not reused ({reason_text}). "
                f"Split cache will be ignored and dataset will be rescanned.[/yellow]"
            )
        # Simpler approach: just iterate through the dataset indices
        # The base_dataset is already filtered to grammatic-only
        with create_progress(console) as progress:
            task = progress.add_task("[cyan]Scanning dataset...", total=len(dataset))

            found_pos_counts = [0] * num_gps
            found_neg_counts = [0] * num_gps
            found_unl_counts = [0] * num_gps

            for idx, sample_data in enumerate(dataset):  # type: ignore[arg-type,var-annotated]
                labels = sample_data["labels"]  # List[int]

                is_labeled_any = False
                for i, label in enumerate(labels):
                    if label == 1:
                        found_pos_counts[i] += 1
                        is_labeled_any = True
                    elif label == 0:
                        found_neg_counts[i] += 1
                        is_labeled_any = True
                    else:
                        found_unl_counts[i] += 1

                if is_labeled_any:
                    labeled_indices.append(idx)
                else:
                    unlabeled_indices.append(idx)

                # Early stopping for test mode
                # Check if ALL GPs have enough samples
                if test_mode:
                    all_ready = True
                    for i in range(num_gps):
                        if found_pos_counts[i] < 5 or found_neg_counts[i] < 5:
                            all_ready = False
                            break

                    if all_ready and len(unlabeled_indices) >= 5:
                        progress.update(task, completed=len(dataset))  # Finish bar
                        console.print(
                            "[yellow]Test mode early stop: Found enough samples for all GPs[/yellow]"
                        )
                        break

                # Update every 10000 samples for better performance
                if idx % 10000 == 0 and idx > 0:
                    progress.update(task, completed=idx)

            progress.update(task, completed=len(dataset))

        os.makedirs(base_output_dir, exist_ok=True)
        with open(split_cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "grammar_labels": grammar_labels,
                    "dataset_size": len(dataset),
                    "labeled_indices": labeled_indices,
                    "unlabeled_indices": unlabeled_indices,
                },
                f,
            )

    # Balance labeled samples in Test Mode (to avoid extreme imbalance like 56 vs 5)
    if test_mode:
        # 1. Re-scan labeled items to get per-GP counts and map
        # (We only have aggregate counts `found_pos_counts` but we need to know WHICH index belongs to WHICH GP logic)
        gp_to_indices: Dict[int, List[int]] = {i: [] for i in range(num_gps)}
        for idx in labeled_indices:
            labels = dataset[idx]["labels"]
            for i, label in enumerate(labels):
                if label == 1:
                    gp_to_indices[i].append(idx)

        # 2. Determine min count and cap
        counts = [len(gp_to_indices[i]) for i in range(num_gps)]
        min_count = min(counts) if counts else 0
        # Cap majority at 4x min or at least 20
        cap = max(min_count * 4, 20)

        console.print(f"  Balancing labeled samples (Min: {min_count}, Cap: {cap})...")

        # 3. Select indices greedily to satisfy caps
        # We want to keep MINORITY samples first.
        # But samples are multi-label.
        # Simple heuristic: Shuffle all labeled indices, keep if adding doesn't violate cap for ALL its active labels?
        # Better: Keep if it helps a GP that hasn't reached cap yet.

        random.shuffle(labeled_indices)
        balanced_indices = []
        current_pos_counts: Dict[int, int] = {i: 0 for i in range(num_gps)}
        current_neg_counts: Dict[int, int] = {i: 0 for i in range(num_gps)}

        for idx in labeled_indices:
            labels = dataset[idx]["labels"]
            pos_gps = [i for i, L in enumerate(labels) if L == 1]
            neg_gps = [i for i, L in enumerate(labels) if L == 0]

            # Keep if ANY active GP is below cap (checking both pos and neg)
            needed = False
            for gp_idx in pos_gps:
                if current_pos_counts[gp_idx] < cap:
                    needed = True
                    break

            if not needed:
                for gp_idx in neg_gps:
                    if current_neg_counts[gp_idx] < cap:
                        needed = True
                        break

            if needed:
                balanced_indices.append(idx)
                for gp_idx in pos_gps:
                    current_pos_counts[gp_idx] += 1
                for gp_idx in neg_gps:
                    current_neg_counts[gp_idx] += 1

        labeled_indices = balanced_indices
        console.print(
            f"  Balanced Labeled samples: {len(labeled_indices):,} (Original: {sum(counts)} raw matches)"
        )

    console.print(f"  Union Labeled samples: {len(labeled_indices):,}")

    # Downsample unlabeled if needed (Test Mode optimization to match user request)
    # User request: downsample to 20x size of labeled samples.
    if test_mode:
        original_count = len(unlabeled_indices)
        target_count = len(labeled_indices) * 20
        if original_count > target_count:
            # Use random.sample to keep it random
            # Note: random is already imported
            unlabeled_indices = random.sample(unlabeled_indices, target_count)
            console.print(
                f"  Total Unlabeled samples: {len(unlabeled_indices):,} (downsampled from {original_count:,})"
            )
        else:
            console.print(f"  Total Unlabeled samples: {len(unlabeled_indices):,}")
    else:
        console.print(f"  Total Unlabeled samples: {len(unlabeled_indices):,}")

    # Create DataLoader for labeled data
    # Create DataLoader for labeled data
    # Split Labeled Data into Train (75%) and Validation (25%)
    random.shuffle(labeled_indices)
    split_idx = int(len(labeled_indices) * 0.75)
    train_labeled_indices = labeled_indices[:split_idx]
    val_labeled_indices = labeled_indices[split_idx:]

    # Ensure at least some validation data if possible
    if len(val_labeled_indices) == 0 and len(labeled_indices) > 1:
        val_labeled_indices = [labeled_indices[-1]]
        train_labeled_indices = labeled_indices[:-1]

    console.print(
        f"  Labeled Split: {len(train_labeled_indices)} Train, {len(val_labeled_indices)} Validation"
    )

    train_labeled_dataset = Subset(dataset, train_labeled_indices)
    train_labeled_loader = DataLoader(
        train_labeled_dataset,
        batch_size=512,  # Use large batch size for Phase 1 warmup
        shuffle=True,
        collate_fn=collate_grammar_batch,
        num_workers=0,  # Avoid multiprocessing issues with MPS
    )

    val_labeled_dataset = Subset(dataset, val_labeled_indices)
    val_labeled_loader = DataLoader(
        val_labeled_dataset,
        batch_size=512,
        shuffle=False,
        collate_fn=collate_grammar_batch,
        num_workers=0,
    )

    # For reporting unlearned samples, we still want to check the FULL labeled set
    # to show overall progress on satisfying constraints.
    full_labeled_dataset = Subset(dataset, labeled_indices)
    full_labeled_loader = DataLoader(
        full_labeled_dataset,
        batch_size=512,
        shuffle=False,
        collate_fn=collate_grammar_batch,
        num_workers=0,
    )

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Learning rate scheduler
    total_steps = num_epochs_warmup * len(train_labeled_loader) + num_epochs_pnu * len(
        train_labeled_loader
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    # Helper to compute validation loss
    def compute_val_loss(loader: DataLoader) -> float:
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in loader:
                field_inputs = {
                    k: v.to(device) for k, v in batch["field_inputs"].items()
                }
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(field_inputs, attention_mask)
                loss = torch.tensor(0.0, device=device)

                for i in range(num_gps):
                    gp_logits = logits[:, i, :]
                    gp_targets = labels[:, i]
                    valid_mask = gp_targets >= 0
                    if valid_mask.any():
                        # Use simple CrossEntropy for validation metric?
                        # Or consistent Focal Loss? Let's use Focal Loss for consistency.
                        valid_logits = gp_logits[valid_mask]
                        valid_targets = gp_targets[valid_mask]
                        gp_loss = focal_loss(
                            valid_logits,
                            valid_targets,
                            alpha=pos_weights[i] / (1 + pos_weights[i]),
                            gamma=2.0,
                        )
                        loss = loss + gp_loss

                val_loss_sum += loss.item()
                val_batches += 1
        model.train()
        return val_loss_sum / max(val_batches, 1)

    # ========== Phase 1: Warmup on Labeled Data ==========
    console.print(
        f"\n[bold cyan]Warmup Training on Labeled Data (in {len(train_labeled_loader)} batches of {train_labeled_loader.batch_size})[/bold cyan]"
    )

    model.train()
    with create_progress(console) as progress:
        task = progress.add_task(
            "[cyan]Training...", total=num_epochs_warmup * len(train_labeled_loader)
        )

        for epoch in range(num_epochs_warmup):
            epoch_loss = 0.0
            for batch in train_labeled_loader:
                # Move to device
                field_inputs = {
                    k: v.to(device) for k, v in batch["field_inputs"].items()
                }
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Forward
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(field_inputs, attention_mask)  # [B, num_gps, 2]

                    # Focal loss summed across GPs where labeled
                    loss = torch.tensor(0.0, device=device)

                    for i in range(num_gps):
                        # Extract logits for this GP: [B, 2]
                        gp_logits = logits[:, i, :]
                        gp_targets = labels[:, i]

                        # Mask for valid labels (>= 0)
                        valid_mask = gp_targets >= 0
                        if valid_mask.any():
                            valid_logits = gp_logits[valid_mask]
                            valid_targets = gp_targets[valid_mask]

                            gp_loss = focal_loss(
                                valid_logits,
                                valid_targets,
                                alpha=pos_weights[i] / (1 + pos_weights[i]),
                                gamma=2.0,
                            )
                            loss = loss + gp_loss

                # Backward
                optimizer.zero_grad()
                if loss.requires_grad:
                    if use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                else:
                    pass

                scheduler.step()

                epoch_loss += loss.item()
                progress.update(
                    task,
                    advance=1,
                    description=f"[cyan]Epoch {epoch + 1}/{num_epochs_warmup} Loss: {loss.item():.6f}",
                )

            avg_loss = epoch_loss / len(train_labeled_loader)
            val_loss = compute_val_loss(val_labeled_loader)
            console.print(
                f"  Epoch {epoch + 1}: Train Loss = {avg_loss:.6f} | Val Loss = {val_loss:.6f}"
            )

    # ========== Phase 2: PNU Training ==========
    console.print(
        f"\n[bold cyan]PNU Training with Unlabeled Data (up to {num_epochs_pnu} epochs)[/bold cyan]"
    )

    # Sample unlabeled data
    unlabeled_sample_size = min(len(unlabeled_indices), len(train_labeled_indices) * 20)
    if unlabeled_sample_size == 0 and len(unlabeled_indices) > 0:
        unlabeled_sample_size = min(
            len(unlabeled_indices), 100
        )  # minimal random sample if labeled is huge? No, if train labeled is tiny.

    sampled_unlabeled = torch.randperm(len(unlabeled_indices))[
        :unlabeled_sample_size
    ].tolist()
    sampled_unlabeled_indices = [unlabeled_indices[i] for i in sampled_unlabeled]

    combined_indices = train_labeled_indices + sampled_unlabeled_indices
    combined_dataset = Subset(dataset, combined_indices)
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=512,
        shuffle=True,
        collate_fn=collate_grammar_batch,
        num_workers=0,
    )

    console.print(
        f"  Training on {len(combined_indices):,} samples ({len(train_labeled_indices):,} labeled + {len(sampled_unlabeled_indices):,} unlabeled)"
    )

    # Initialize early stopper for Phase 2
    stopper = EarlyStopper(patience=24, min_delta=0.000001, decay_factor=0.85)

    # Track best model
    best_phase2_loss = float("inf")  # This will now track VAL loss
    best_phase2_epoch = -1
    best_model_state = None

    model.train()
    with create_progress(console) as progress:
        task = progress.add_task(
            "[cyan]PNU Training...", total=num_epochs_pnu * len(combined_loader)
        )

        for epoch in range(num_epochs_pnu):
            epoch_loss = 0.0

            for batch in combined_loader:
                field_inputs = {
                    k: v.to(device) for k, v in batch["field_inputs"].items()
                }
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                # Forward
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(field_inputs, attention_mask)  # [B, num_gps, 2]

                    # Total loss accumulator
                    total_loss = torch.tensor(0.0, device=device)

                    for i in range(num_gps):
                        gp_logits = logits[:, i, :]
                        gp_targets = labels[:, i]

                        # Separate labeled and unlabeled
                        labeled_mask = gp_targets >= 0
                        unlabeled_mask = gp_targets < 0  # Strictly -1

                        # Loss on labeled samples
                        if labeled_mask.any():
                            labeled_logits = gp_logits[labeled_mask]
                            labeled_targets = gp_targets[labeled_mask]

                            labeled_loss = focal_loss(
                                labeled_logits,
                                labeled_targets,
                                alpha=pos_weights[i] / (1 + pos_weights[i]),
                                gamma=2.0,
                            )
                            total_loss = total_loss + labeled_loss

                        # Pseudo-label unlabeled samples
                        if unlabeled_mask.any():
                            unlabeled_logits = gp_logits[unlabeled_mask]
                            with torch.no_grad():
                                probs = F.softmax(unlabeled_logits, dim=-1)
                                confidence, pseudo_labels = probs.max(dim=-1)

                            # Only use high-confidence pseudo-labels
                            high_conf_mask = confidence > 0.8
                            if high_conf_mask.any():
                                conf_logits = unlabeled_logits[high_conf_mask]
                                conf_labels = pseudo_labels[high_conf_mask]

                                pseudo_loss = F.cross_entropy(
                                    conf_logits, conf_labels, reduction="mean"
                                )
                                total_loss = (
                                    total_loss + 0.3 * pseudo_loss
                                )  # Lower weight for pseudo-labels

                    loss = total_loss

                # Backward
                optimizer.zero_grad()
                if loss.requires_grad:
                    if use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                else:
                    pass

                # Note: We do NOT step the scheduler here for Phase 2
                # We let EarlyStopper manage LR decay based on plateau
                # scheduler.step()

                epoch_loss += loss.item()
                progress.update(
                    task,
                    advance=1,
                    description=f"[cyan]Epoch {epoch + 1}/{num_epochs_pnu} Loss: {loss.item():.6f}",
                )

            avg_loss = epoch_loss / len(combined_loader)
            val_loss = compute_val_loss(val_labeled_loader)

            # Write unlearned samples logic moved to end of loop (conditional on is_best_epoch)

            # Compute stats for reporting
            all_stats_per_gp = compute_accuracy_stats(
                model, full_labeled_loader, device, grammar_labels
            )
            val_stats_per_gp = compute_accuracy_stats(
                model, val_labeled_loader, device, grammar_labels
            )

            # Determine if this is the best epoch so far
            is_best_epoch = val_loss < best_phase2_loss

            # Print Epoch Header (Always Bright)
            header_text = f"  Epoch {epoch + 1} (in {len(combined_loader)} batches of {combined_loader.batch_size}): Train Loss = {avg_loss:.6f} | Val Loss = {val_loss:.6f}"
            console.print(header_text)

            def _d(text: str, is_best: bool = is_best_epoch) -> str:
                return f"[dim]{text}[/dim]" if not is_best else text

            # Create table for epoch stats
            # Use dim style for table headers if not best epoch
            header_style = "purple" if is_best_epoch else "dim"

            table = Table(box=None, show_header=True, pad_edge=False, padding=(0, 0))
            table.add_column("", style="cyan", header_style=header_style)
            table.add_column("", style="dim", header_style=header_style)
            table.add_column("", width=2)
            table.add_column("positive", justify="right", header_style=header_style)
            table.add_column("", justify="left")
            table.add_column("", width=2)
            table.add_column("negative", justify="right", header_style=header_style)
            table.add_column("", justify="left")
            table.add_column("", width=2)
            table.add_column("total", justify="right", header_style=header_style)
            table.add_column("", justify="left")

            def _fmt(num: int, denom: int) -> Tuple[str, str]:
                pct = 100.0 * num / denom if denom > 0 else 0.0
                color = "red"
                if pct >= 99.9:
                    color = "green"
                elif pct >= 90.0:
                    color = "yellow"
                # Add leading space to percentage for separation when padding=0
                return (f"{num} of {denom}", f" ([{color}]{pct:.0f}%[/{color}])")

            # Collect stats for sorting (using ALL stats for sorting order)
            stats_rows = []
            for i in range(num_gps):
                # Process ALL stats
                learned_pos, total_pos, learned_neg, total_neg = all_stats_per_gp[i]
                learned_total = learned_pos + learned_neg
                denom_total = total_pos + total_neg
                accuracy = learned_total / denom_total if denom_total > 0 else 0.0

                p_cnt, p_pct = _fmt(learned_pos, total_pos)
                n_cnt, n_pct = _fmt(learned_neg, total_neg)
                t_cnt, t_pct = _fmt(learned_total, denom_total)

                # Apply dimming to ALL row content
                row_all = [
                    "",  # Indented/Empty GP name for second row
                    " [dim]all[/dim]",
                    "",
                    f"[dim]{p_cnt}[/dim]",
                    f"[dim]{p_pct}[/dim]",
                    "",
                    f"[dim]{n_cnt}[/dim]",
                    f"[dim]{n_pct}[/dim]",
                    "",
                    f"[dim]{t_cnt}[/dim]",
                    f"[dim]{t_pct}[/dim]",
                ]

                # Process VAL stats
                v_learned_pos, v_total_pos, v_learned_neg, v_total_neg = (
                    val_stats_per_gp[i]
                )
                v_learned_total = v_learned_pos + v_learned_neg
                v_denom_total = v_total_pos + v_total_neg

                vp_cnt, vp_pct = _fmt(v_learned_pos, v_total_pos)
                vn_cnt, vn_pct = _fmt(v_learned_neg, v_total_neg)
                vt_cnt, vt_pct = _fmt(v_learned_total, v_denom_total)

                # Apply conditional dimming to VAL row content
                row_val = [
                    _d(grammar_labels[i]),
                    _d(" val"),
                    "",
                    _d(vp_cnt),
                    _d(vp_pct),
                    "",
                    _d(vn_cnt),
                    _d(vn_pct),
                    "",
                    _d(vt_cnt),
                    _d(vt_pct),
                ]

                # Store as tuple: (accuracy_all, row_all, row_val, total_pos_all, total_neg_all)
                stats_rows.append((accuracy, row_all, row_val, total_pos, total_neg))

            # Sort by accuracy (descending)
            stats_rows.sort(key=lambda x: x[0], reverse=True)

            n_head = 2
            n_tail = 2

            # Filter out 100% satisfied (based on ALL stats)
            satisfied_rows = [
                x for x in stats_rows if x[0] >= 0.9999 and x[3] > 0 and x[4] > 0
            ]
            active_rows = [
                x for x in stats_rows if not (x[0] >= 0.9999 and x[3] > 0 and x[4] > 0)
            ]

            # Sort active by accuracy (descending)
            active_rows.sort(key=lambda x: x[0], reverse=True)

            n_head = 2
            n_tail = 2

            # Re-construct display iteration to handle middle insertion
            if len(active_rows) <= (n_head + n_tail):
                for item in active_rows:
                    table.add_row(*item[2])  # Add Val row FIRST
                    table.add_row(*item[1])  # Add All row SECOND
            else:
                # Head
                for i in range(n_head):
                    table.add_row(*active_rows[i][2])
                    table.add_row(*active_rows[i][1])

                # Middle
                remaining = len(active_rows) - n_head - n_tail
                table.add_row(f"[dim][{remaining} omitted][/dim]", *[""] * 10)

                # Tail
                for i in range(len(active_rows) - n_tail, len(active_rows)):
                    table.add_row(*active_rows[i][2])
                    table.add_row(*active_rows[i][1])

            console.print(Padding(table, (0, 0, 0, 6)))

            if len(satisfied_rows) > 0:
                satisfied_gps = [item[2][0].strip() for item in satisfied_rows]
                satisfied_gps.sort()
                console.print(
                    Padding(
                        f"[dim]... and {len(satisfied_rows)} that are 100% satisfied: "
                        f"{', '.join(satisfied_gps)}[/dim]",
                        (0, 0, 0, 6),
                    )
                )

            # Switch back to train mode
            model.train()

            # Save best model based on VAL loss
            if val_loss < best_phase2_loss:  # <--- Changed from avg_loss
                best_phase2_loss = val_loss
                best_phase2_epoch = epoch + 1
                best_model_state = copy.deepcopy(model.state_dict())

                # Checkpoint immediately
                torch.save(model.state_dict(), model_path)
                console.print(f"      Checkpoint saved to {model_path}")

            # Check for perfect learning (using Eval stats on FULL set)
            # If all hard labels are correct and loss is very low, stop early
            all_perfect = True
            for eval_stat in all_stats_per_gp:
                lp, tp, ln, tn = eval_stat
                if lp != tp or ln != tn:
                    all_perfect = False
                    break

            if all_perfect and val_loss < 0.0001:  # <--- Changed from avg_loss
                console.print(
                    "  [green]Perfect learning achieved (Val Loss < 0.0001)! Stopping Phase 2 early.[/green]"
                )
                break

            # Check early stopping based on VAL loss
            should_stop, new_batch_size = stopper.check(
                val_loss, cast(int, combined_loader.batch_size), is_best=is_best_epoch
            )
            if should_stop:
                console.print(
                    f"  [yellow]Early stopping triggered at epoch {epoch + 1}[/yellow]"
                )
                break

            if new_batch_size:
                # Recreate loader with new batch size
                combined_loader = DataLoader(
                    combined_dataset,
                    batch_size=new_batch_size,
                    shuffle=True,
                    collate_fn=collate_grammar_batch,
                    num_workers=0,
                )

            if not all_perfect and is_best_epoch:
                console.print("      Writing Unlearned Samples...")
                write_unlearned_samples(
                    model,
                    full_labeled_loader,
                    device,
                    base_output_dir,
                    grammar_labels,
                    verbose=False,
                )

    # Restore best model
    if best_model_state is not None:
        console.print(
            f"\n  [dim]Using best model (epoch {best_phase2_epoch} with loss {best_phase2_loss:.6f})[/dim]"
        )
        model.load_state_dict(best_model_state)

    # Analyze unlearned samples (Misclassified Labeled Data)
    # Header printed inside function if needed
    write_unlearned_samples(
        model,
        full_labeled_loader,
        device,
        base_output_dir,
        grammar_labels,
        verbose=False,
    )

    # Full Pos Scan (Requested to run after Phase 2)
    if full_pos_scan_fn:
        # Scan both labeled and unlabeled samples collected
        scan_indices = labeled_indices + unlabeled_indices
        scan_indices.sort()  # Keep them sorted for dataset access

        full_pos_scan_fn(
            model=model,
            dataset=dataset,
            device=device,
            _batch_size=1024,
            base_output_dir=base_output_dir,
            grammar_labels=grammar_labels,
            test_mode=test_mode,
            candidate_indices=scan_indices,
            quick_mode=quick_mode,
        )
    # Phase 3 loop sets/resets mode? No, Phase 3 is inference on subset.
    # But actually Phase 3 below computes stats using model(field_inputs) inside no_grad.
    # It's an evaluation loop essentially.

    # Save model
    os.makedirs(
        base_output_dir, exist_ok=True
    )  # Ensure base dir exists, though we might not use it directly for artifacts
    # Save model in the first GP dir? Or a shared one?
    # Let's save in base/model.pt if base is distinct, OR in EACH gp dir?
    # Since we use distinct output dirs for each GP, maybe we should save a copy in each?
    # Or just save once in the first one?
    # Plan said: ".cache/curate/study/gp0888/metrics.json" etc.
    # The user might expect `gp0888/model.pt`.
    # Let's save to EACH output dir.

    # Actually, let's defer saving to caller or save to a shared location?
    # No, let's follow the pattern: save model.pt to the first GP's dir, or all?
    # Saving to all is safest for individual resume.

    # We will let `generate_candidates` handle per-GP output logic?
    # But `train` is supposed to save the model.
    # Let's save to a "shared" folder if multiple GPs, or just save it.

    model_path = os.path.join(
        base_output_dir, "model.pt"
    )  # This might be .cache/curate/study/model.pt if we are not careful.
    # We should probably pass a specific model save path or dir.
    # For now, let's just save to `base_output_dir` assuming it is `.cache/curate/study/RUN_ID` or similar?
    # No, `base_output_dir` comes from `.cache/curate/study` (parent).

    # Let's save to a temporary location or just rely on the return value?
    # The existing code saved to output_dir.

    # Let's return model and let main save it?
    # Or just save to `base_output_dir/model.pt` (shared).
    torch.save(model.state_dict(), model_path)
    console.print(f"\n[green]Model saved to {model_path}[/green]")

    return model, unlabeled_indices


def find_high_certainty_positives(
    model: GrammarClassifier,
    dataset: GrammarPointDataset,
    device: torch.device,
    _batch_size: int,
    base_output_dir: str,
    grammar_labels: List[str],
    test_mode: bool = False,
    candidate_indices: Optional[List[int]] = None,
    quick_mode: bool = False,
) -> None:
    # pylint: disable=unused-argument
    """Find high-certainty positive candidates and report stats for each GP."""
    num_gps = len(grammar_labels)
    # If candidate indices provided (e.g. from filtered scan), use them
    # Otherwise scan everything
    if candidate_indices is not None:
        all_indices = list(candidate_indices)
    else:
        all_indices = list(range(len(dataset)))

    # Shuffle so early stopping samples are representative
    random.shuffle(all_indices)

    # Hardcode large batch size for fast scanning
    scan_batch_size = 2048
    num_batches = math.ceil(len(all_indices) / scan_batch_size)

    console.print(
        f"\n[bold cyan]Scanning full (in {num_batches} batches of {scan_batch_size})[/bold cyan]"
    )

    console.print(f"  Evaluating {len(all_indices):,} samples...")

    # Create loader for everything
    subset = Subset(dataset, all_indices)
    loader = DataLoader(
        subset,
        batch_size=scan_batch_size,
        shuffle=False,
        collate_fn=collate_grammar_batch,
        num_workers=0,
    )

    # Statistics tracking lists per GP
    # We'll store probabilities to compute    # Store results per GP
    probs_all: List[List[float]] = [[] for _ in range(num_gps)]
    probs_unlabeled: List[List[float]] = [[] for _ in range(num_gps)]

    # Loss accumulation
    gp_losses: List[float] = [0.0] * num_gps
    gp_counts: List[int] = [0] * num_gps

    # Store candidates as (score, sentence) tuples
    high_certainty_candidates_unlabeled: List[List[Tuple[float, str]]] = [
        [] for _ in range(num_gps)
    ]
    high_certainty_candidates_negatives_unlabeled: List[List[Tuple[float, str]]] = [
        [] for _ in range(num_gps)
    ]
    # Track most-uncertain samples per GP (closest to 0.5), include labeled + unlabeled
    # Stored as max-heap by distance using negative distance
    most_uncertain: List[List[Tuple[float, float, str]]] = [[] for _ in range(num_gps)]

    # Early stop when prior reaches target precision
    prior_precision_decimals = 4.5
    prior_precision_target = int(math.ceil(10**prior_precision_decimals))
    # Require a minimum number of positive hits to stabilize very low priors
    min_pos_target = 25
    prior_total_counts = [0] * num_gps
    prior_pos_counts = [0] * num_gps

    with torch.no_grad():
        with create_progress(console) as progress:
            task = progress.add_task("[cyan]Evaluating...", total=len(loader))

            for batch in loader:
                field_inputs = {
                    k: v.to(device) for k, v in batch["field_inputs"].items()
                }
                attention_mask = batch["attention_mask"].to(device)
                sentences = batch["sentences"]
                labels = batch["labels"]  # cpu tensor

                logits = model(field_inputs, attention_mask)
                probs = F.softmax(logits, dim=-1)

                # Loop over GPs
                for i in range(num_gps):
                    pos_probs = probs[:, i, 1]
                    pos_probs_cpu = pos_probs.cpu()
                    probs_all[i].extend(pos_probs_cpu.tolist())

                    gp_labels = labels[:, i]
                    is_unlabeled = gp_labels < 0
                    if is_unlabeled.any():
                        unlabeled_probs = pos_probs[is_unlabeled]
                        probs_unlabeled[i].extend(unlabeled_probs.cpu().tolist())

                    # Calculate Loss (Treat Unlabeled as Negative)
                    # Target is 1 if Label=1, else 0 (for Label=0 and Label=-1)
                    targets = torch.zeros_like(pos_probs)
                    targets[gp_labels == 1] = 1.0

                    # Binary Cross Entropy
                    # We can use F.binary_cross_entropy on probs
                    # clamp to avoid log(0)
                    clamped_probs = torch.clamp(pos_probs, 1e-7, 1.0 - 1e-7)
                    batch_loss = F.binary_cross_entropy(
                        clamped_probs, targets.to(device), reduction="sum"
                    )

                    gp_losses[i] += batch_loss.item()
                    gp_counts[i] += len(pos_probs)

                    pred_pos_mask = pos_probs > 0.5
                    candidate_mask = is_unlabeled & pred_pos_mask.cpu()

                    if candidate_mask.any():
                        idxs = torch.nonzero(candidate_mask).squeeze(1)
                        for idx in idxs:
                            sentence = sentences[idx]
                            score = pos_probs[idx].item()
                            high_certainty_candidates_unlabeled[i].append(
                                (score, sentence)
                            )

                    pred_neg_mask = pos_probs < 0.5
                    candidate_neg_mask = is_unlabeled & pred_neg_mask.cpu()

                    if candidate_neg_mask.any():
                        idxs = torch.nonzero(candidate_neg_mask).squeeze(1)
                        for idx in idxs:
                            sentence = sentences[idx]
                            score = pos_probs[idx].item()
                            high_certainty_candidates_negatives_unlabeled[i].append(
                                (score, sentence)
                            )

                    # Track most-uncertain samples (all samples, labeled + unlabeled)
                    heap = most_uncertain[i]
                    for idx, sentence in enumerate(sentences):
                        prob = pos_probs[idx].item()
                        distance = abs(prob - 0.5)
                        item = (-distance, prob, sentence)
                        if len(heap) < 100:
                            heapq.heappush(heap, item)
                        else:
                            if item[0] > heap[0][0]:
                                heapq.heapreplace(heap, item)

                    # Track prior precision counts (based on all samples)
                    prior_total_counts[i] += len(pos_probs)
                    if is_unlabeled.any():
                        unlabeled_pos = pos_probs[is_unlabeled] > 0.5
                        prior_pos_counts[i] += int(unlabeled_pos.sum().item())

                progress.update(task, advance=1)

                if (
                    len(all_indices) >= prior_precision_target
                    and all(
                        count >= prior_precision_target for count in prior_total_counts
                    )
                    and all(count >= min_pos_target for count in prior_pos_counts)
                ):
                    progress.update(task, completed=len(loader))
                    console.print(
                        f"[yellow]Early stop: prior has {prior_precision_decimals} decimals "
                        f"and at least {min_pos_target} positives for all GPs[/yellow]"
                    )
                    break

    # Helper to compute stats
    def compute_stats(name: str, probabilities: List[float]) -> Tuple[List[str], float]:
        if not probabilities:
            return (
                [
                    name,
                    "0",
                    "0.0%",
                    "0.0%",
                    "0.0000 (0)",
                    "0.0000 (0)",
                    "0.0000 (0)",
                    "0.0000 (0)",
                    "0.0000 (0)",
                ],
                0.0,
            )

        t = torch.tensor(probabilities)
        count = len(t)
        # If total_count_override is provided, we use that for Prior calculation logic if needed,
        # but here 'probabilities' list length matches the segment count.

        est_pos = (t > 0.5).sum().item()  # Hard count > 0.5
        # Note: Previous logic for "Include Label" had special mixed logic (Trust Label=1, ignore Label=0 predictions).
        # But if we just look at raw model output statistics requested by user (percentiles), we should probably use raw model probs.
        # However, for "Prior" and "Est. Pos", let's keep the refined logic if possible, OR just use the raw model distribution stats.
        # Given "Include Labeled" usually implies we trust ground truth, but "statistics percentiles" implies model behavior.
        # Let's show Model Behavior Stats for percentiles, and "Prior" based on the distribution.

        # For "Include Labeled", the "Est. Pos" in previous step was (Label==1) + (Unlabeled & Pred>0.5).
        # That logic requires knowing which prob belongs to which label.
        # Since I replaced the loop, I lost the complex accumulation.
        # Let's revert to a simpler "Model View" or re-implement the hybrid logic if critical.
        # User asked for "statistics percentiles".
        # Let's compute pure model stats on the `probabilities` list.

        prior = est_pos / max(count, 1)
        conf = (2 * t - 1).abs().mean().item()

        p50 = torch.quantile(t, 0.5).item()
        p75 = torch.quantile(t, 0.75).item()
        p90 = torch.quantile(t, 0.90).item()
        p95 = torch.quantile(t, 0.95).item()
        p99 = torch.quantile(t, 0.99).item()

        gt_50 = (t > 0.5).sum().item()
        gt_75 = (t > 0.75).sum().item()
        gt_90 = (t > 0.9).sum().item()
        gt_95 = (t > 0.95).sum().item()
        gt_99 = (t > 0.99).sum().item()

        return [
            name,
            f"{count:,}",
            f"{prior * 100:.5f}%",
            f"{conf * 100:.1f}%",
            f"{p50:.4f} ({gt_50:,})",
            f"{p75:.4f} ({gt_75:,})",
            f"{p90:.4f} ({gt_90:,})",
            f"{p95:.4f} ({gt_95:,})",
            f"{p99:.4f} ({gt_99:,})",
        ], conf

    # Re-calculate hybrid stats for "Include Labeled" if we want to be precise about "Prior" vs "Model Stats"?
    # The user request "estimated prior" earlier was "ratio of positive to total".
    # And "Include labeled" = "all samples, even if they have labels".
    # I will stick to pure model predictions for the stats columns to be consistent with "Mean/Median/Percentiles".
    # If the model is good, it predicts 1 for Label=1 and 0 for Label=0.

    # Create single table for all GPs
    table = Table(
        box=None,
        show_header=True,
        pad_edge=False,
    )
    table.add_column("", style="cyan", header_style="purple")
    # New Column: Loss
    table.add_column("loss", justify="right", header_style="purple")
    table.add_column("subset", header_style="purple")
    table.add_column("n", justify="right", header_style="purple")
    table.add_column("prior", justify="right", header_style="purple")
    table.add_column("conf", justify="right", header_style="purple")
    table.add_column("p50", justify="left", header_style="purple")
    table.add_column("p75", justify="left", header_style="purple")
    table.add_column("p90", justify="left", header_style="purple")
    table.add_column("p95", justify="left", header_style="purple")
    table.add_column("p99", justify="left", header_style="purple")

    # Process each GP
    gp_stats_list = []  # (gp, avg_loss, conf, row_all, row_unl)

    for i, gp in enumerate(grammar_labels):
        output_dir = os.path.join(base_output_dir, gp)
        os.makedirs(output_dir, exist_ok=True)

        # Calculate avg loss
        avg_loss = gp_losses[i] / max(gp_counts[i], 1)

        # Add stats rows to table
        row_all, conf_all = compute_stats("all", probs_all[i])
        row_unl, _ = compute_stats("unlabeled", probs_unlabeled[i])

        gp_stats_list.append((gp, avg_loss, conf_all, row_all, row_unl))

        # Write top 100 positives
        candidates = high_certainty_candidates_unlabeled[i]
        candidates.sort(key=lambda x: x[0], reverse=True)

        hc_file = os.path.join(output_dir, "high-certainty-positives.txt")
        with open(hc_file, "w", encoding="utf-8") as f:
            f.write(f"# High Certainty Positives for {gp} (Unlabeled Only)\n")
            f.write("# Format: Prob | Sentence\n")
            f.write("-" * 70 + "\n")
            for score, sentence in candidates[:100]:
                f.write(f"{score:.4f} | {sentence}\n")

        # Write top 100 negatives
        candidates_neg = high_certainty_candidates_negatives_unlabeled[i]
        # Sort ascending by probability (lowest first)
        candidates_neg.sort(key=lambda x: x[0])

        hc_neg_file = os.path.join(output_dir, "high-certainty-negatives.txt")
        with open(hc_neg_file, "w", encoding="utf-8") as f:
            f.write(f"# High Certainty Negatives for {gp} (Unlabeled Only)\n")
            f.write("# Format: Prob | Sentence\n")
            f.write("-" * 70 + "\n")
            for score, sentence in candidates_neg[:100]:
                f.write(f"{score:.4f} | {sentence}\n")

        # Write most uncertain (closest to 0.5) from ALL samples
        uncertain_heap = most_uncertain[i]
        uncertain_sorted = sorted(uncertain_heap, key=lambda x: (-x[0], x[1]))
        uncertain_file = os.path.join(output_dir, "most-uncertain.txt")
        with open(uncertain_file, "w", encoding="utf-8") as f:
            f.write(f"# Most Uncertain Samples for {gp} (All)\n")
            f.write("# Format: Dist | Prob | Sentence\n")
            f.write("-" * 70 + "\n")
            for neg_dist, prob, sentence in uncertain_sorted:
                dist = -neg_dist
                f.write(f"{dist:.4f} | {prob:.4f} | {sentence}\n")

    # Optionally write estimated priors back to corpus.db (grammar table).
    # Back-compat is an anti-goal: if the DB/schema isn't present, fail loudly.
    if not test_mode and not quick_mode:
        db_path = os.path.join(os.getcwd(), "data", "corpus.db")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Missing corpus.db at {db_path}")

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            for i, gp in enumerate(grammar_labels):
                probs_list = probs_all[i]
                if probs_list:
                    # Estimated corpus prior: fraction of sentences with prob > 0.5
                    t = torch.tensor(probs_list)
                    prior_est = float((t > 0.5).float().mean().item())
                    cursor.execute(
                        "UPDATE grammar SET prior = ? WHERE id = ?",
                        (prior_est, gp),
                    )
            conn.commit()
            console.print(
                "[dim]Wrote estimated grammar priors to data/corpus.db (grammar.prior)[/dim]"
            )

    # Sort by Loss (Ascending) as requested
    gp_stats_list.sort(key=lambda x: x[1])

    n_head = 2
    n_tail = 2

    # If list is small enough, just show all
    if len(gp_stats_list) <= (n_head + n_tail):
        display_items = gp_stats_list
        show_ellipsis = False
    else:
        display_items = gp_stats_list[:n_head] + gp_stats_list[-n_tail:]
        show_ellipsis = True

    for i, (gp, avg_loss, _, row_all, row_unl) in enumerate(display_items):
        if show_ellipsis and i == n_head:
            remaining = len(gp_stats_list) - n_head - n_tail
            table.add_row(
                f"[dim][{remaining} omitted][/dim]", *[""] * (len(row_all) + 1 - 1)
            )

        # row_all is [name, count, prior...]
        # We need to inject avg_loss
        # Table columns: GP, Loss, Subset, N, Prior...

        # Row 1: GP Name, Loss, "all", N...
        # row_all[0] is name (e.g. "all"). Wait, compute_stats returns [name, count...]
        # row_all[0] is "all"

        loss_str = f"{avg_loss:.4f}"

        table.add_row(gp, loss_str, *row_all)
        # Row 2: "", "", "unlabeled", N...
        table.add_row("", "", *row_unl, style="dim")

    console.print(Padding(table, (0, 0, 0, 6)))

    console.print()


def apply_curated_labels(grammar_label: str, output_dir: str) -> None:
    """Apply curated labels to corpus.db after manual review."""
    console.print(
        f"\n[bold cyan]Applying Curated Labels for {grammar_label}[/bold cyan]"
    )

    # Read candidate files
    neg_file = os.path.join(output_dir, "best-hard-negative-candidates.txt")
    pos_file = os.path.join(output_dir, "best-hard-positive-candidates.txt")

    if not os.path.exists(neg_file) or not os.path.exists(pos_file):
        console.print(
            "[red]Error: Candidate files not found. Run without --apply first.[/red]"
        )
        return

    # Parse candidate files
    # Support two formats:
    # 1. With metadata: "score | prob | sentence"
    # 2. Sentences only: "sentence"
    new_negatives = []
    with open(neg_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Check if line has metadata (score | prob | sentence format)
            parts = line.split(" | ")
            if len(parts) >= 3:
                # Has metadata - extract sentence from index 2 onwards
                sentence = " | ".join(parts[2:])
            else:
                # No metadata - entire line is the sentence
                sentence = line

            if sentence:
                new_negatives.append(sentence)

    new_positives = []
    with open(pos_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Check if line has metadata (score | prob | sentence format)
            parts = line.split(" | ")
            if len(parts) >= 3:
                # Has metadata - extract sentence from index 2 onwards
                sentence = " | ".join(parts[2:])
            else:
                # No metadata - entire line is the sentence
                sentence = line

            if sentence:
                new_positives.append(sentence)

    console.print(f"  Hard negatives to add: {len(new_negatives):,}")
    console.print(f"  Hard positives to add: {len(new_positives):,}")

    # Connect to database (for checking only)
    db_path = os.path.join(
        os.path.dirname(output_dir), "..", "..", "..", "data", "corpus.db"
    )
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Track statistics
    neg_added = 0
    neg_already_present = 0
    neg_moved = 0  # Moved from grammar to grammar_negative
    pos_added = 0
    pos_already_present = 0
    pos_moved = 0  # Moved from grammar_negative to grammar

    valid_negatives = []
    valid_positives = []

    sentences_moved_to_neg = []
    sentences_moved_to_pos = []

    # Check hard negatives
    for sentence in new_negatives:
        # Get current labels and grammatical status
        cursor.execute(
            "SELECT grammar, grammar_negative, grammatic FROM corpus WHERE sentence = ?",
            (sentence,),
        )
        row = cursor.fetchone()
        if not row:
            console.print(
                f"[yellow]Warning: Sentence not found in corpus: {sentence[:50]}...[/yellow]"
            )
            continue

        grammar, grammar_negative, grammatic = row

        # Check grammaticality
        if grammatic != 1:
            console.print(
                f"[yellow]Warning: Skipping ungrammatic sentence: {sentence[:50]}...[/yellow]"
            )
            continue

        # Check if already present
        if grammar_label in (grammar_negative or ""):
            neg_already_present += 1
            continue  # Skip, already labeled as negative

        # Check moved
        if grammar_label in (grammar or ""):
            neg_moved += 1
            sentences_moved_to_neg.append(sentence)
        else:
            neg_added += 1

        valid_negatives.append(sentence)

    # Check hard positives
    for sentence in new_positives:
        cursor.execute(
            "SELECT grammar, grammar_negative, grammatic FROM corpus WHERE sentence = ?",
            (sentence,),
        )
        row = cursor.fetchone()
        if not row:
            console.print(
                f"[yellow]Warning: Sentence not found in corpus: {sentence[:50]}...[/yellow]"
            )
            continue

        grammar, grammar_negative, grammatic = row

        # Check grammaticality
        if grammatic != 1:
            console.print(
                f"[yellow]Warning: Skipping ungrammatic sentence: {sentence[:50]}...[/yellow]"
            )
            continue

        # Check if already present
        if grammar_label in (grammar or ""):
            pos_already_present += 1
            continue  # Skip, already labeled as positive

        # Check moved
        if grammar_label in (grammar_negative or ""):
            pos_moved += 1
            sentences_moved_to_pos.append(sentence)
        else:
            pos_added += 1

        valid_positives.append(sentence)

    conn.close()

    # Apply updates using batch upsert (safe for normalized schema)
    if valid_negatives:
        console.print(
            f"\n[bold]Applying {len(valid_negatives)} negative labels...[/bold]"
        )
        curate_upsert_batch(
            valid_negatives,
            None,
            None,
            grammar_diff_str=f"-{grammar_label}",
            db_path=db_path,
        )

    if valid_positives:
        console.print(
            f"\n[bold]Applying {len(valid_positives)} positive labels...[/bold]"
        )
        curate_upsert_batch(
            valid_positives,
            None,
            None,
            grammar_diff_str=f"+{grammar_label}",
            db_path=db_path,
        )

    console.print("\n[green]✓ Successfully applied curated labels to database[/green]")

    # Report statistics
    console.print("\n[bold]Summary:[/bold]")
    console.print("  Hard negatives:")
    console.print(f"    • Added: {neg_added}")
    console.print(f"    • Already present: {neg_already_present}")
    console.print(f"    • Moved from positive: {neg_moved}")
    console.print("  Hard positives:")
    console.print(f"    • Added: {pos_added}")
    console.print(f"    • Already present: {pos_already_present}")
    console.print(f"    • Moved from negative: {pos_moved}")

    # List moved sentences
    if sentences_moved_to_neg:
        console.print(
            "\n[bold yellow]Sentences moved from POSITIVE to NEGATIVE:[/bold yellow]"
        )
        for s in sentences_moved_to_neg:
            console.print(f"  • {s}")

    if sentences_moved_to_pos:
        console.print(
            "\n[bold yellow]Sentences moved from NEGATIVE to POSITIVE:[/bold yellow]"
        )
        for s in sentences_moved_to_pos:
            console.print(f"  • {s}")


def export_existing_labels(
    dataset: GrammarPointDataset, base_output_dir: str, grammar_labels: List[str]
) -> None:
    """Export existing hard positive and negative labels to separate files for each GP."""
    console.print("\n[bold cyan]Exporting Existing Labels[/bold cyan]")

    # Collect stats first
    stats_rows = []

    for i, gp in enumerate(grammar_labels):
        output_dir = os.path.join(base_output_dir, gp)
        os.makedirs(output_dir, exist_ok=True)

        pos_file = os.path.join(output_dir, "existing-hard-positive.txt")
        neg_file = os.path.join(output_dir, "existing-hard-negative.txt")

        pos_count = 0
        neg_count = 0

        with (
            open(pos_file, "w", encoding="utf-8") as f_pos,
            open(neg_file, "w", encoding="utf-8") as f_neg,
        ):
            # Scan dataset labels
            for sentence, label_map in dataset.sentence_labels.items():
                label = label_map.get(i, -1)
                if label == 1:
                    f_pos.write(f"{sentence}\n")
                    pos_count += 1
                elif label == 0:
                    f_neg.write(f"{sentence}\n")
                    neg_count += 1

        stats_rows.append((pos_count, gp, f"{pos_count:,}", f"{neg_count:,}"))

    # Sort descending by pos_count
    stats_rows.sort(key=lambda x: x[0], reverse=True)

    # Strip sort key
    display_rows = [r[1:] for r in stats_rows]

    # Display table
    table = Table(box=None, show_header=True, pad_edge=False)
    table.add_column("", style="cyan", header_style="purple")
    table.add_column("pos", justify="left", header_style="purple")
    table.add_column("neg", justify="left", header_style="purple")

    n_head = 2
    n_tail = 2

    if len(display_rows) <= (n_head + n_tail):
        for r in display_rows:
            table.add_row(*r)
    else:
        for r in display_rows[:n_head]:
            table.add_row(*r)

        remaining = len(display_rows) - n_head - n_tail
        table.add_row(f"[dim][{remaining} omitted][/dim]", "", "")

        for r in display_rows[-n_tail:]:
            table.add_row(*r)

    console.print(Padding(table, (0, 0, 0, 6)))


def export_grammar_yaml(grammar_labels: List[str], base_output_dir: str) -> None:
    """Export the grammar point's YAML definition file to the output directory."""
    db_path = os.path.join(
        os.path.dirname(base_output_dir), "..", "..", "data", "corpus.db"
    )
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        for gp in grammar_labels:
            output_dir = os.path.join(base_output_dir, gp)
            os.makedirs(output_dir, exist_ok=True)

            cursor.execute("SELECT name FROM grammar WHERE id = ?", (gp,))
            row = cursor.fetchone()
            if not row:
                console.print(
                    f"[yellow]Warning: Grammar ID {gp} not found in grammar table.[/yellow]"
                )
                continue

            grammar_name = row[0]
            yaml_filename = f"{grammar_name}.yaml"

            # Look for the file in data/grammar
            source_path = os.path.join(
                os.path.dirname(base_output_dir),
                "..",
                "..",
                "data",
                "grammar",
                yaml_filename,
            )

            if not os.path.exists(source_path):
                console.print(
                    f"[yellow]Warning: Grammar YAML file not found at {source_path}[/yellow]"
                )
                continue

            # Copy to output directory
            dest_path = os.path.join(output_dir, yaml_filename)
            shutil.copy2(source_path, dest_path)
            # console.print(f"[green]✓[/green] Exported grammar definition to {dest_path}")

    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PNU-based hard negative/positive mining for grammar points"
    )
    parser.add_argument(
        "grammar_labels", nargs="+", help="Grammar point labels (e.g., gp0888 gp0404)"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply curated labels to database (run after manual review)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to train on",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer epochs and sample subset for testing (much faster)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for dataset split/shuffle and sampling",
    )
    parser.add_argument(
        "--full-pos",
        action="store_true",
        help="Evaluate all unlabeled sentences and find high-certainty positives (>99%%)",
    )

    args = parser.parse_args()

    # Output directory
    test_mode = os.environ.get("CURATE_TEST_MODE") == "1"

    # Base output directory (parent of gp folders)
    # If single GP, use .cache/curate/study/gpXXXX (legacy compat?)
    # OR .cache/curate/study and inside have gpXXXX?
    # User plan says: ".cache/curate/study/gp0888/"
    # If we pass multiple, we want a base.

    if test_mode:
        base_output_dir = os.path.join(".cache", "curate", "study-test")
        console.print(
            "[yellow]Test Mode Active: Using .cache/curate/study-test/[/yellow]"
        )
    else:
        base_output_dir = os.path.join(".cache", "curate", "study")

    # Clean check: if applying, we process individually?
    # CLI structure suggests: curate study gp1 gp2 --apply
    # We should iterate and apply.

    if args.apply:
        # Apply mode - Iterate over labels
        for gp in args.grammar_labels:
            # Construct specific output dir
            gp_dir = os.path.join(base_output_dir, gp)
            apply_curated_labels(gp, gp_dir)
    else:
        # Training mode

        console.print(f"Base Output Directory: {base_output_dir}")
        if len(args.grammar_labels) > 1:
            console.print("[bold green]Shared Model Training enabled.[/bold green]")

        # Seed RNGs for reproducible splits/sampling
        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

        # Export grammar YAMLs
        export_grammar_yaml(args.grammar_labels, base_output_dir)

        # Load dataset
        dataset_dir = get_style_dataset_cache_dir()
        console.print(f"\nLoading dataset from {dataset_dir}...")

        # Load tokenizer
        tokenizer_path = os.path.join(dataset_dir, "vocab.json")
        tokenizer = Tokenizer.load(tokenizer_path)
        console.print(
            f"[green]✓[/green] Loaded tokenizer with {len(tokenizer.field_vocabs)} fields"
        )

        # Load base dataset
        base_dataset = StyleDataset(dataset_dir, tokenizer, sample_ratio=1.0)
        console.print(f"[green]✓[/green] Loaded {len(base_dataset):,} samples")

        # Filter to grammatic sentences only
        console.print("\nFiltering to grammatic sentences...")
        grammatic_dataset = base_dataset.filter_by_grammaticality(label=1)
        console.print(
            f"  Filtered: {len(grammatic_dataset):,} grammatic of {len(base_dataset):,} total"
        )

        # Wrap with grammar point labels
        # console.print("\nPreparing grammar point datasets...")
        dataset = GrammarPointDataset(grammatic_dataset, args.grammar_labels)

        # Check labeled counts
        # Warning if any GP has < 10
        for stat in dataset.gp_stats:
            count = int(cast(Any, stat["pos"])) + int(cast(Any, stat["neg"]))
            if count < 10:
                console.print(
                    f"[red]Error: Not enough labeled data for {stat['label']} ({count}). Need at least 10.[/red]"
                )
                return

        # Export existing labels
        export_existing_labels(dataset, base_output_dir, args.grammar_labels)

        # Train model
        device = (
            torch.device(args.device)
            if args.device
            else torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        )

        if test_mode:
            console.print(
                "[yellow]Test mode enabled: Using minimal epochs and early scan stopping[/yellow]"
            )
            num_warmup = 2
            num_pnu = 2
        elif args.quick:
            console.print(
                "[yellow]Quick mode enabled: Using reduced epochs and sample subset[/yellow]"
            )
            num_warmup = 2
            num_pnu = 5
        else:
            num_warmup = 5
            # Increase epochs if multiple GPs?
            # User wants multiple GPs to transfer/share execution.
            # 50 epochs should be enough for convergence of multiple heads.
            num_pnu = 50

        _model, _ = train_pnu_model(
            args.grammar_labels,
            dataset,
            tokenizer,
            device,
            base_output_dir,
            num_epochs_warmup=num_warmup,
            num_epochs_pnu=num_pnu,
            batch_size=args.batch_size,
            test_mode=test_mode,
            quick_mode=args.quick,
            full_pos_scan_fn=find_high_certainty_positives if args.full_pos else None,
        )

        # Candidate generation removed by request.

        console.print("\n[bold green]✓ Complete![/bold green]")
        console.print("\nNext steps:")
        console.print(f"  1. Review candidates in {base_output_dir}/<gp>/")
        console.print("  2. Manually verify candidate files.")
        console.print(
            f"  3. Run: scripts/curate study {args.grammar_labels[0]} --apply (for example)"
        )


if __name__ == "__main__":
    main()
