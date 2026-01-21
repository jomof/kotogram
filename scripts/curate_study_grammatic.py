#!/usr/bin/env python3
# ruff: noqa: E402
# pylint: disable=wrong-import-position
"""
Grammatic learnability study for corpus.db.

Trains a Transformer-based classifier to identify sentences that are likely
mislabeled as grammatic when they are actually ungrammatic.
"""

import os
import sqlite3
import sys

# Ensure project root is in path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional, cast

import torch
from rich.console import Console
from torch import nn

from scripts import study_utils
from train import paths

console = Console()

# Constants
CLASS_NAMES = ["ungrammatic", "grammatic"]
STUDY_DIR = ".cache/curate/study/grammatic"
MODEL_DIR = "data/models/grammatic"


@dataclass
class GrammaticSample:
    """A sample for grammatic classification."""

    idx: int
    sentence: str
    grammatic: int
    feature_start: int
    feature_end: int


class GrammaticDataset(study_utils.BaseStudyDataset[GrammaticSample]):
    """Dataset for grammatic classification study."""

    def __init__(
        self,
        data_dir: str,
        indices: Optional[torch.Tensor] = None,
    ):
        super().__init__(data_dir, indices)

        # Load grammatic flags
        gram_path = os.path.join(data_dir, "labels.bin_gram")
        gram_size = os.path.getsize(gram_path)
        self.grammatic = torch.from_file(
            gram_path, shared=True, size=gram_size, dtype=torch.uint8
        )

    def __getitem__(self, pos: int) -> GrammaticSample:
        s_idx = int(self.indices[pos].item())
        offs_start = int(self.offsets[s_idx].item())
        offs_end = int(self.offsets[s_idx + 1].item())
        gram_flag = int(self.grammatic[s_idx].item())

        return GrammaticSample(
            idx=s_idx,
            sentence=self.sentences[s_idx],
            grammatic=gram_flag,
            feature_start=offs_start,
            feature_end=offs_end,
        )


class GrammaticClassifier(study_utils.BaseStudyClassifier):
    """Transformer-based classifier for grammatic prediction."""

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 128,
    ):
        super().__init__(vocab_sizes, embed_dim)
        total_embed = embed_dim * len(vocab_sizes)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, total_embed))
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)
        self.input_norm = nn.LayerNorm(total_embed)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=total_embed,
            nhead=num_heads,
            dim_feedforward=total_embed * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(total_embed, total_embed),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(total_embed, 2),
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass. features is dict of (batch, seq_len) tensors."""
        embeds = []
        seq_len = 0
        padding_mask = None

        for name in sorted(self.vocab_sizes.keys()):
            if name in features:
                feat = features[name]
                seq_len = feat.shape[1]
                emb = self.embeddings[name](feat)
                embeds.append(emb)
                if padding_mask is None:
                    padding_mask = feat == 0

        combined = torch.cat(embeds, dim=-1)
        combined = combined + self.pos_encoding[:, :seq_len, :]
        combined = self.input_norm(combined)
        transformed = self.transformer(combined)

        if padding_mask is not None:
            mask = (~padding_mask).unsqueeze(-1).float()
            pooled = (transformed * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = transformed.mean(dim=1)

        result: torch.Tensor = self.classifier(pooled)
        return result


# pylint: disable=too-many-locals
def _train_classifier(
    dataset: GrammaticDataset,
    vocab_sizes: Dict[str, int],
    batch_size: int = 128,
) -> GrammaticClassifier:
    """Train a grammatic classifier."""
    device = study_utils.get_device()
    model = GrammaticClassifier(vocab_sizes).to(device)

    # Prepare data
    train_idx, val_idx = study_utils.split_dataset_indices(dataset.indices)
    train_ds, val_ds = study_utils.prepare_study_data(
        dataset, train_idx, val_idx, GrammaticDataset
    )

    collate = partial(
        study_utils.collate_study_samples, dataset=dataset, label_attr="grammatic"
    )
    train_loader, val_loader = study_utils.setup_study_loaders(
        train_ds, val_ds, collate, batch_size
    )

    # Weights
    counts = torch.zeros(2, dtype=torch.float32)
    for s in train_ds:
        counts[s.grammatic] += 1
    weights = counts.sum() / (2 * counts.clamp(min=1))
    weights = (weights / weights.sum() * 2).to(device)

    console.print(
        f"[bold blue]Training (Train={len(train_ds)}, Val={len(val_ds)})[/bold blue]"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Kick off training
    trained = study_utils.train_study_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        console=console,
    )
    return cast(GrammaticClassifier, trained)


def run_grammatic_study(
    db_path: str,  # pylint: disable=unused-argument
    batch: int = 1,
    percent: int = 100,
    batch_size: int = 128,
) -> None:
    """Execute the grammatic identification study."""
    grammatic_data_path = paths.get_style_dataset_cache_dir()
    if not study_utils.check_dataset_cache(console, grammatic_data_path):
        return

    ds = GrammaticDataset(grammatic_data_path)
    if percent < 100:
        n_samples = int(len(ds) * percent / 100)
        ds.indices = ds.indices[:n_samples]

    study_utils.run_standard_study(
        console,
        STUDY_DIR,
        MODEL_DIR,
        ds,
        GrammaticClassifier,
        partial(study_utils.collate_study_samples, dataset=ds, label_attr="grammatic"),
        CLASS_NAMES,
        partial(_train_classifier, batch_size=batch_size),
        batch=batch,
    )


def apply_grammatic_changes(db_path: str) -> None:
    """Apply suggested grammatic changes."""
    console.print("[bold blue]Applying changes...[/bold blue]")
    updated: Dict[str, int] = {}
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for cls_idx, filename in [
        (0, "suggest ungrammatic.txt"),
        (1, "suggest grammatic.txt"),
    ]:
        path = os.path.join(STUDY_DIR, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    sent = line.strip()
                    if sent:
                        cursor.execute(
                            "UPDATE corpus SET grammatic = ? WHERE sentence = ?",
                            (cls_idx, sent),
                        )
                        if cursor.rowcount > 0:
                            updated[sent] = cls_idx
            conn.commit()

    study_utils.verify_database_updates(console, cursor, updated, "grammatic")
    study_utils.finalize_database_updates(console, conn, len(updated), db_path)


if __name__ == "__main__":
    init_db = os.path.join(paths.get_data_dir(), "corpus.db")
    cmd_args = argparse.ArgumentParser()
    cmd_args.add_argument("--db-path", default=init_db)
    cmd_args.add_argument("--apply", action="store_true")
    cmd_args.add_argument("--batch", type=int, default=1)
    cmd_args.add_argument("--percent", type=int, default=100)
    cmd_args.add_argument("--batch-size", type=int, default=128)
    parsed = cmd_args.parse_args()

    if parsed.apply:
        apply_grammatic_changes(parsed.db_path)
    else:
        run_grammatic_study(
            parsed.db_path,
            batch=parsed.batch,
            percent=parsed.percent,
            batch_size=parsed.batch_size,
        )
