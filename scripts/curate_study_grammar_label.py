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
import json
import math
import os
import sqlite3
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from rich.console import Console
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

# Add project root to path
if os.path.exists("kotogram"):
    sys.path.insert(0, os.getcwd())

from kotogram.model import ModelConfig, PositionalEncoding
from kotogram.tokenizer import ENCODER_FEATURE_FIELDS, Tokenizer
from scripts.progress_utils import create_progress
from train.dataset import StyleDataset
from train.paths import get_style_dataset_cache_dir

console = Console()


@dataclass
class SampleLossStats:
    """Statistics for a single sample during training."""

    sentence: str
    total_loss: float = 0.0
    num_updates: int = 0
    avg_pred_positive: float = 0.0
    avg_uncertainty: float = 0.0
    is_labeled_positive: bool = False
    is_labeled_negative: bool = False


class GrammarPointDataset(Dataset):
    """Wraps StyleDataset to provide grammar point labels."""

    def __init__(
        self,
        base_dataset: StyleDataset,
        grammar_label: str,
        verbose: bool = True,
    ):
        self.base_dataset = base_dataset
        self.grammar_label = grammar_label
        self.verbose = verbose

        # Load grammar point labels
        self._load_grammar_labels()

    def _load_grammar_labels(self) -> None:
        """Load grammar point positive/negative labels from database."""
        if self.verbose:
            console.print("  Loading grammar labels from database...")

        db_path = os.path.join(
            os.path.dirname(self.base_dataset.data_dir), "..", "data", "corpus.db"
        )

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all sentences with this grammar point
        # Note: If grammatic_only=True, the base_dataset is already filtered
        cursor.execute(
            """
            SELECT sentence,
                   CASE WHEN grammar LIKE '%' || ? || '%' THEN 1 ELSE 0 END as positive,
                   CASE WHEN grammar_negative LIKE '%' || ? || '%' THEN 1 ELSE 0 END as negative
            FROM corpus
            """,
            (self.grammar_label, self.grammar_label),
        )

        # Build sentence -> label mapping
        self.sentence_to_label: Dict[
            str, int
        ] = {}  # 1=positive, 0=negative, -1=unlabeled
        positive_count = 0
        negative_count = 0

        # Fetch all results (faster than iterating)
        if self.verbose:
            console.print("  Processing labels...")

        results = cursor.fetchall()
        for sentence, positive, negative in results:
            if positive:
                self.sentence_to_label[sentence] = 1
                positive_count += 1
            elif negative:
                self.sentence_to_label[sentence] = 0
                negative_count += 1
            # else: unlabeled, not in dict

        conn.close()

        self.positive_count = positive_count
        self.negative_count = negative_count
        self.labeled_count = positive_count + negative_count
        self.total_count = len(self.base_dataset)
        self.unlabeled_count = self.total_count - self.labeled_count

        if self.verbose:
            console.print(
                f"\n[bold cyan]Grammar Point:[/bold cyan] {self.grammar_label}"
            )
            console.print(f"  Positive: {positive_count:,}")
            console.print(f"  Negative: {negative_count:,}")
            console.print(f"  Unlabeled: {self.unlabeled_count:,}")
            console.print(f"  Total: {self.total_count:,}")
            console.print(
                f"  Label density: {100.0 * self.labeled_count / self.total_count:.4f}%"
            )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample with grammar point label."""
        sample = self.base_dataset[idx]

        # Get sentence text using base_dataset's method
        # sample.idx is the real_idx in the dataset
        sentence = self.base_dataset.get_sentence_by_idx(sample.idx)

        # Get label: 1=positive, 0=negative, -1=unlabeled
        label = self.sentence_to_label.get(sentence, -1)

        # Return sample with grammar label added
        return {
            "sample": sample,
            "label": label,
            "sentence": sentence,
            "idx": sample.idx,  # Use real idx from sample
        }


class GrammarClassifier(nn.Module):
    """Lightweight binary classifier for grammar point detection."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

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
            dropout=0.1,
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

        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 2),  # Binary: [grammar_present, grammar_absent]
        )

    def forward(
        self, field_inputs: Dict[str, torch.Tensor], attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            field_inputs: Dict of token feature tensors
            attention_mask: Attention mask (1=valid, 0=padding)

        Returns:
            logits: [B, 2] logits for [present, absent]
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
        logits = self.classifier(pooled)
        return torch.Tensor(logits)  # type: ignore[return-value]


def collate_grammar_batch(batch: List[Dict]) -> Dict:
    """Collate batch for grammar point training."""
    samples = [item["sample"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
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
    return torch.Tensor(loss.mean())  # type: ignore[return-value]


def train_pnu_model(  # pylint: disable=unused-argument
    grammar_label: str,
    dataset: GrammarPointDataset,
    tokenizer: Tokenizer,
    device: torch.device,
    output_dir: str,
    num_epochs_warmup: int = 5,
    num_epochs_pnu: int = 15,
    num_epochs_eval: int = 5,
    batch_size: int = 32,
    top_k: int = 500,
    eval_seed: Optional[int] = None,
) -> Tuple[GrammarClassifier, Dict[int, SampleLossStats]]:
    """Train PNU model and collect loss statistics.

    Returns:
        model: Trained model
        sample_stats: Dict mapping sample idx to loss statistics
    """
    # Create config
    config = ModelConfig(vocab_sizes=tokenizer.get_vocab_sizes())

    # Create model
    model = GrammarClassifier(config)
    model = model.to(device)

    # Enable automatic mixed precision for fp16 (not on MPS due to float64 limitation)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda" if device.type == "cuda" else "cpu", enabled=use_amp
    )

    # Compute class weights
    pos_weight = dataset.negative_count / max(dataset.positive_count, 1)
    pos_weight = min(pos_weight, 50.0)  # Cap at 50x
    console.print("\n[bold]Training Configuration:[/bold]")
    console.print(f"  Device: {device}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Positive weight: {pos_weight:.2f}")
    console.print(f"  Use AMP: {use_amp}")

    # Split into labeled and unlabeled - optimize by reading sentences file once
    console.print("\n  Splitting into labeled/unlabeled sets...")
    labeled_indices = []
    unlabeled_indices = []

    # Simpler approach: just iterate through the dataset indices
    # The base_dataset is already filtered to grammatic-only
    with create_progress(console) as progress:
        task = progress.add_task("[cyan]Scanning dataset...", total=len(dataset))

        for idx, sample_data in enumerate(dataset):  # type: ignore[arg-type,var-annotated]
            label = sample_data["label"]

            if label >= 0:  # Labeled (0 or 1)
                labeled_indices.append(idx)
            else:  # Unlabeled (-1)
                unlabeled_indices.append(idx)

            # Update every 10000 samples for better performance
            if idx % 10000 == 0 and idx > 0:
                progress.update(task, completed=idx)

        progress.update(task, completed=len(dataset))

    console.print(f"  Labeled samples: {len(labeled_indices):,}")
    console.print(f"  Unlabeled samples: {len(unlabeled_indices):,}")

    # Create DataLoader for labeled data
    labeled_dataset = Subset(dataset, labeled_indices)
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_grammar_batch,
        num_workers=0,  # Avoid multiprocessing issues with MPS
    )

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Learning rate scheduler
    total_steps = num_epochs_warmup * len(labeled_loader) + num_epochs_pnu * len(
        labeled_loader
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    # Statistics tracking
    sample_stats: Dict[int, SampleLossStats] = {}

    # ========== Phase 1: Warmup on Labeled Data ==========
    console.print(
        f"\n[bold cyan]Phase 1: Warmup Training on Labeled Data ({num_epochs_warmup} epochs)[/bold cyan]"
    )

    model.train()
    with create_progress(console) as progress:
        task = progress.add_task(
            "[cyan]Training...", total=num_epochs_warmup * len(labeled_loader)
        )

        for epoch in range(num_epochs_warmup):
            epoch_loss = 0.0
            for batch in labeled_loader:
                # Move to device
                field_inputs = {
                    k: v.to(device) for k, v in batch["field_inputs"].items()
                }
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Forward
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(field_inputs, attention_mask)

                    # Focal loss with class weighting
                    loss = focal_loss(
                        logits, labels, alpha=pos_weight / (1 + pos_weight), gamma=2.0
                    )

                # Backward
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                scheduler.step()

                epoch_loss += loss.item()
                progress.update(
                    task,
                    advance=1,
                    description=f"[cyan]Epoch {epoch + 1}/{num_epochs_warmup} Loss: {loss.item():.4f}",
                )

            avg_loss = epoch_loss / len(labeled_loader)
            console.print(f"  Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f}")

    # ========== Phase 2: PNU Training ==========
    console.print(
        f"\n[bold cyan]Phase 2: PNU Training with Unlabeled Data ({num_epochs_pnu} epochs)[/bold cyan]"
    )

    # Sample unlabeled data
    unlabeled_sample_size = min(len(unlabeled_indices), len(labeled_indices) * 10)
    sampled_unlabeled = torch.randperm(len(unlabeled_indices))[
        :unlabeled_sample_size
    ].tolist()
    sampled_unlabeled_indices = [unlabeled_indices[i] for i in sampled_unlabeled]

    combined_indices = labeled_indices + sampled_unlabeled_indices
    combined_dataset = Subset(dataset, combined_indices)
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_grammar_batch,
        num_workers=0,
    )

    console.print(
        f"  Training on {len(combined_indices):,} samples ({len(labeled_indices):,} labeled + {len(sampled_unlabeled_indices):,} unlabeled)"
    )

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
                indices = batch["indices"]

                # Forward
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(field_inputs, attention_mask)

                    # Separate labeled and unlabeled
                    labeled_mask = labels >= 0
                    unlabeled_mask = labels < 0

                    total_loss = torch.tensor(0.0, device=device)

                    # Loss on labeled samples
                    if labeled_mask.any():
                        labeled_logits = logits[labeled_mask]
                        labeled_targets = labels[labeled_mask]
                        labeled_loss = focal_loss(
                            labeled_logits,
                            labeled_targets,
                            alpha=pos_weight / (1 + pos_weight),
                            gamma=2.0,
                        )
                        total_loss = total_loss + labeled_loss

                    # Pseudo-label unlabeled samples
                    if unlabeled_mask.any():
                        unlabeled_logits = logits[unlabeled_mask]
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
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                scheduler.step()

                epoch_loss += loss.item()
                progress.update(
                    task,
                    advance=1,
                    description=f"[cyan]Epoch {epoch + 1}/{num_epochs_pnu} Loss: {loss.item():.4f}",
                )

            avg_loss = epoch_loss / len(combined_loader)
            console.print(f"  Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f}")

    # ========== Phase 3: Evaluation and Loss Accumulation ==========
    console.print(
        f"\n[bold cyan]Phase 3: Loss Accumulation for Hard Mining ({num_epochs_eval} epochs)[/bold cyan]"
    )

    # For efficiency, sample a subset for evaluation (we don't need ALL samples for mining)
    # Sample labeled + reasonable subset of unlabeled
    # Scale with top_k: need at least top_k * 50 samples to mine from for good diversity
    min_sample_size = max(50000, top_k * 50)  # At least 50K or 50x top_k
    eval_sample_size = min(len(dataset), min_sample_size)
    console.print(
        f"  Sampling {eval_sample_size:,} samples for evaluation (faster than full {len(dataset):,})"
    )

    # Create sampled indices: all labeled + random unlabeled
    import random
    import time

    # Use truly random seed by default so each run evaluates different sentences
    # Or use user-provided seed for reproducibility
    if eval_seed is None:
        # Use current time for randomness - different subset each run
        seed = int(time.time() * 1000000) % (2**32)
        console.print(f"  Using random seed {seed} (different subset each run)")
        console.print(f"  To reproduce this exact subset, use: --seed {seed}")
    else:
        seed = eval_seed
        console.print(f"  Using custom seed {seed} (reproducible subset)")

    random.seed(seed)

    eval_indices = labeled_indices.copy()
    available_unlabeled = [i for i in range(len(dataset)) if i not in labeled_indices]
    unlabeled_sample = random.sample(
        available_unlabeled,
        min(eval_sample_size - len(labeled_indices), len(available_unlabeled)),
    )
    eval_indices.extend(unlabeled_sample)

    eval_subset = Subset(dataset, eval_indices)
    eval_loader = DataLoader(
        eval_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_grammar_batch,
        num_workers=0,
    )

    console.print(
        f"  Processing {len(eval_loader):,} batches per epoch (~{len(eval_subset):,} samples)"
    )
    console.print("  Starting evaluation (this may take a moment)...", end="")
    console.file.flush()  # Force output

    model.eval()
    with create_progress(console) as progress:
        task = progress.add_task(
            "[cyan]Starting evaluation...", total=num_epochs_eval * len(eval_loader)
        )

        for epoch in range(num_epochs_eval):
            console.print(f"\n  Epoch {epoch + 1}/{num_epochs_eval} starting...")
            with torch.no_grad():
                for batch_idx, batch in enumerate(eval_loader):
                    # Update progress more frequently - every 10 batches for responsive feedback
                    if batch_idx % 10 == 0:
                        progress.update(
                            task,
                            advance=10,
                            description=f"[cyan]Epoch {epoch + 1}/{num_epochs_eval} - Batch {batch_idx:,}/{len(eval_loader):,}",
                        )

                    field_inputs = {
                        k: v.to(device) for k, v in batch["field_inputs"].items()
                    }
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    indices = batch["indices"]
                    sentences = batch["sentences"]

                    # Forward
                    logits = model(field_inputs, attention_mask)
                    probs = F.softmax(logits, dim=-1)

                    # Compute per-sample statistics
                    for i, idx in enumerate(indices):
                        label = labels[i].item()
                        prob_pos = probs[i, 1].item()  # Probability of positive class
                        prob_neg = probs[i, 0].item()

                        # Compute loss
                        if label >= 0:  # Labeled
                            sample_loss = F.cross_entropy(
                                logits[i : i + 1], labels[i : i + 1], reduction="mean"
                            ).item()
                        else:  # Unlabeled - use entropy as proxy
                            entropy = -(
                                prob_pos * math.log(prob_pos + 1e-9)
                                + prob_neg * math.log(prob_neg + 1e-9)
                            )
                            sample_loss = entropy

                        # Uncertainty (entropy)
                        uncertainty = -(
                            prob_pos * math.log(prob_pos + 1e-9)
                            + prob_neg * math.log(prob_neg + 1e-9)
                        )

                        # Update or create stats
                        if idx not in sample_stats:
                            sample_stats[idx] = SampleLossStats(
                                sentence=sentences[i],
                                is_labeled_positive=label == 1,
                                is_labeled_negative=label == 0,
                            )

                        stats = sample_stats[idx]
                        stats.total_loss += sample_loss
                        stats.num_updates += 1
                        stats.avg_pred_positive = (
                            stats.avg_pred_positive * (stats.num_updates - 1) + prob_pos
                        ) / stats.num_updates
                        stats.avg_uncertainty = (
                            stats.avg_uncertainty * (stats.num_updates - 1)
                            + uncertainty
                        ) / stats.num_updates

            # Complete remaining progress at end of epoch
            final_batch = len(eval_loader) - 1
            final_progress = (final_batch // 10) * 10
            remaining = len(eval_loader) - final_progress
            if remaining > 0:
                progress.update(task, advance=remaining)

            # Print epoch summary
            console.print(
                f"  Epoch {epoch + 1}: Processed {len(sample_stats):,} unique samples"
            )

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    console.print(f"\n[green]Model saved to {model_path}[/green]")

    return model, sample_stats


def generate_candidates(
    grammar_label: str,
    sample_stats: Dict[int, SampleLossStats],
    output_dir: str,
    top_k: int = 500,
) -> None:
    """Generate hard negative and positive candidates based on loss statistics."""
    console.print("\n[bold cyan]Generating Candidates[/bold cyan]")

    # Score and rank samples
    hard_negative_candidates = []
    hard_positive_candidates = []

    for idx, stats in sample_stats.items():
        # Skip labeled samples (we want to find new labels)
        if stats.is_labeled_positive or stats.is_labeled_negative:
            continue

        # Compute composite score
        avg_loss = stats.total_loss / max(stats.num_updates, 1)
        uncertainty = stats.avg_uncertainty
        pred_positive = stats.avg_pred_positive
        boundary_proximity = (
            1.0 - abs(pred_positive - 0.5) * 2
        )  # 0 at boundaries, 1 at extremes

        # Composite score
        score = (
            (avg_loss * 0.4) + (uncertainty * 0.3) + (boundary_proximity * 0.2) + (0.1)
        )

        # Hard negative: model predicts positive (but likely wrong)
        if pred_positive > 0.5:
            hard_negative_candidates.append((score, stats.sentence, idx, pred_positive))

        # Hard positive: model predicts negative (but likely wrong)
        else:
            hard_positive_candidates.append((score, stats.sentence, idx, pred_positive))

    # Sort by score (descending)
    hard_negative_candidates.sort(reverse=True, key=lambda x: x[0])
    hard_positive_candidates.sort(reverse=True, key=lambda x: x[0])

    # Write hard negatives
    neg_file = os.path.join(output_dir, "best-hard-negative-candidates.txt")
    with open(neg_file, "w", encoding="utf-8") as f:
        f.write(f"# Hard Negative Candidates for {grammar_label}\n")
        f.write(
            "# These sentences were predicted as POSITIVE but are likely FALSE POSITIVES\n"
        )
        f.write("# Review and REMOVE lines that are actual positives\n")
        f.write("# Format: Score | Pred_Prob | Sentence\n")
        f.write("# " + "-" * 70 + "\n")

        for score, sentence, idx, pred_prob in hard_negative_candidates[:top_k]:
            f.write(f"{score:.4f} | {pred_prob:.4f} | {sentence}\n")

    console.print(
        f"[green]✓[/green] Written {min(len(hard_negative_candidates), top_k):,} hard negative candidates to:"
    )
    console.print(f"  {neg_file}")

    # Write hard positives
    pos_file = os.path.join(output_dir, "best-hard-positive-candidates.txt")
    with open(pos_file, "w", encoding="utf-8") as f:
        f.write(f"# Hard Positive Candidates for {grammar_label}\n")
        f.write(
            "# These sentences were predicted as NEGATIVE but are likely FALSE NEGATIVES\n"
        )
        f.write("# Review and REMOVE lines that are actual negatives\n")
        f.write("# Format: Score | Pred_Prob | Sentence\n")
        f.write("# " + "-" * 70 + "\n")

        for score, sentence, idx, pred_prob in hard_positive_candidates[:top_k]:
            f.write(f"{score:.4f} | {pred_prob:.4f} | {sentence}\n")

    console.print(
        f"[green]✓[/green] Written {min(len(hard_positive_candidates), top_k):,} hard positive candidates to:"
    )
    console.print(f"  {pos_file}")

    # Save statistics
    stats_file = os.path.join(output_dir, "metrics.json")
    metrics = {
        "grammar_label": grammar_label,
        "total_samples_evaluated": len(sample_stats),
        "hard_negative_candidates": len(hard_negative_candidates),
        "hard_positive_candidates": len(hard_positive_candidates),
        "top_k": top_k,
    }
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    console.print(f"[green]✓[/green] Saved metrics to {stats_file}")


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

    # Connect to database
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

    # Update hard negatives
    for sentence in new_negatives:
        # Get current labels
        cursor.execute(
            "SELECT grammar, grammar_negative FROM corpus WHERE sentence = ?",
            (sentence,),
        )
        row = cursor.fetchone()
        if not row:
            console.print(
                f"[yellow]Warning: Sentence not found in corpus: {sentence[:50]}...[/yellow]"
            )
            continue

        grammar, grammar_negative = row

        # Check if already present
        if grammar_label in grammar_negative:
            neg_already_present += 1
            continue  # Skip, already labeled as negative

        # Remove from grammar (if was hard positive)
        was_positive = grammar_label in grammar
        if was_positive:
            grammar_list = [g for g in grammar.split(",") if g and g != grammar_label]
            grammar = ",".join(grammar_list)
            neg_moved += 1

        # Add to grammar_negative
        grammar_neg_list = [g for g in grammar_negative.split(",") if g]
        grammar_neg_list.append(grammar_label)
        grammar_negative = ",".join(grammar_neg_list)

        if not was_positive:
            neg_added += 1

        # Update
        cursor.execute(
            "UPDATE corpus SET grammar = ?, grammar_negative = ? WHERE sentence = ?",
            (grammar, grammar_negative, sentence),
        )

    # Update hard positives
    for sentence in new_positives:
        cursor.execute(
            "SELECT grammar, grammar_negative FROM corpus WHERE sentence = ?",
            (sentence,),
        )
        row = cursor.fetchone()
        if not row:
            console.print(
                f"[yellow]Warning: Sentence not found in corpus: {sentence[:50]}...[/yellow]"
            )
            continue

        grammar, grammar_negative = row

        # Check if already present
        if grammar_label in grammar:
            pos_already_present += 1
            continue  # Skip, already labeled as positive

        # Remove from grammar_negative (if was hard negative)
        was_negative = grammar_label in grammar_negative
        if was_negative:
            grammar_neg_list = [
                g for g in grammar_negative.split(",") if g and g != grammar_label
            ]
            grammar_negative = ",".join(grammar_neg_list)
            pos_moved += 1

        # Add to grammar
        grammar_list = [g for g in grammar.split(",") if g]
        grammar_list.append(grammar_label)
        grammar = ",".join(grammar_list)

        if not was_negative:
            pos_added += 1

        # Update
        cursor.execute(
            "UPDATE corpus SET grammar = ?, grammar_negative = ? WHERE sentence = ?",
            (grammar, grammar_negative, sentence),
        )

    # Commit
    conn.commit()
    conn.close()

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PNU-based hard negative/positive mining for grammar points"
    )
    parser.add_argument("grammar_label", help="Grammar point label (e.g., gp0888)")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply curated labels to database (run after manual review)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--top-k", type=int, default=500, help="Number of candidates to generate"
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
        help="Random seed for sampling evaluation set (default: random, different each run)",
    )

    args = parser.parse_args()

    # Output directory
    output_dir = os.path.join(".cache", "curate", "study", args.grammar_label)
    os.makedirs(output_dir, exist_ok=True)

    if args.apply:
        # Apply mode
        apply_curated_labels(args.grammar_label, output_dir)
    else:
        # Training mode
        console.print("[bold]PNU Hard Negative Mining for Grammar Points[/bold]")
        console.print(f"Grammar Label: {args.grammar_label}")
        console.print(f"Output Directory: {output_dir}")

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

        # Filter to grammatic sentences only (grammar points shouldn't appear in ungrammatic text)
        console.print("\nFiltering to grammatic sentences...")
        grammatic_dataset = base_dataset.filter_by_grammaticality(label=1)
        console.print(
            f"  Filtered: {len(grammatic_dataset):,} grammatic of {len(base_dataset):,} total"
        )

        # Wrap with grammar point labels
        console.print("\nPreparing grammar point dataset...")
        dataset = GrammarPointDataset(grammatic_dataset, args.grammar_label)

        # Check if we have enough labeled data
        if dataset.labeled_count < 10:
            console.print(
                f"[red]Error: Not enough labeled data ({dataset.labeled_count}). Need at least 10 labeled samples.[/red]"
            )
            return

        # Train model
        device = torch.device(args.device)

        # Adjust parameters for quick mode
        if args.quick:
            console.print(
                "[yellow]Quick mode enabled: Using reduced epochs and sample subset[/yellow]"
            )
            num_warmup = 2
            num_pnu = 5
            num_eval = 2
        else:
            num_warmup = 5
            num_pnu = 15
            num_eval = 5

        _model, sample_stats = train_pnu_model(
            args.grammar_label,
            dataset,
            tokenizer,
            device,
            output_dir,
            num_epochs_warmup=num_warmup,
            num_epochs_pnu=num_pnu,
            num_epochs_eval=num_eval,
            batch_size=args.batch_size,
            top_k=args.top_k,
            eval_seed=args.seed,
        )

        # Generate candidates
        generate_candidates(
            args.grammar_label, sample_stats, output_dir, top_k=args.top_k
        )

        console.print("\n[bold green]✓ Complete![/bold green]")
        console.print("\nNext steps:")
        console.print(f"  1. Review candidates in {output_dir}/")
        console.print(
            "  2. Remove false positives from best-hard-negative-candidates.txt"
        )
        console.print(
            "  3. Remove false negatives from best-hard-positive-candidates.txt"
        )
        console.print(f"  4. Run: scripts/curate study {args.grammar_label} --apply")


if __name__ == "__main__":
    main()
