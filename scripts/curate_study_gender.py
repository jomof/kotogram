#!/usr/bin/env python3
# ruff: noqa: E402
# pylint: disable=wrong-import-position
"""
Gender learnability study for corpus.db.

Trains a focused classifier to learn gender labels and identifies potential mislabels
by finding sentences where the model's predictions disagree with the ground truth.
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


class GenderStudyDataset(study_utils.BaseStudyDataset[GenderSample]):
    """Dataset for gender classification study using pre-computed features."""

    def __init__(
        self,
        data_dir: str,
        indices: Optional[torch.Tensor] = None,
    ):
        super().__init__(data_dir, indices)

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

        # Filter to pragmatic gender samples only if indices not provided
        if indices is None:
            total_samples = len(self.offsets) - 1
            pragmatic_mask = self.gender_pragmatic[:total_samples] == 1
            self.indices = torch.nonzero(pragmatic_mask, as_tuple=True)[0]

    def __getitem__(self, idx: int) -> GenderSample:
        sample_idx = int(self.indices[idx].item())
        f_start = int(self.offsets[sample_idx].item())
        f_end = int(self.offsets[sample_idx + 1].item())
        curr_gender_val = float(self.gender_values[sample_idx].item())
        gender_cls = _discretize_gender(curr_gender_val)

        return GenderSample(
            idx=sample_idx,
            sentence=self.sentences[sample_idx],
            gender_value=curr_gender_val,
            gender_class=gender_cls,
            feature_start=f_start,
            feature_end=f_end,
        )


class GenderClassifier(study_utils.BaseMLPStudyClassifier):
    """Simple classifier for gender prediction from KC features."""

    def __init__(
        self, vocab_sizes: Dict[str, int], embed_dim: int = 32, hidden_dim: int = 128
    ):
        super().__init__(vocab_sizes, embed_dim, hidden_dim, num_classes=3)


# pylint: disable=too-many-locals
def _train_classifier(
    dataset: GenderStudyDataset,
    vocab_sizes: Dict[str, int],
    epochs: int = 10,
    batch_size: int = 256,
) -> GenderClassifier:
    """Train a gender classifier."""
    device = study_utils.get_device()
    model = GenderClassifier(vocab_sizes).to(device)

    # Data
    parts = study_utils.split_dataset_indices(dataset.indices)
    train_ds, val_ds = study_utils.prepare_study_data(
        dataset, parts[0], parts[1], GenderStudyDataset
    )

    collate = partial(
        study_utils.collate_study_samples, dataset=dataset, label_attr="gender_class"
    )
    train_loader, val_loader = study_utils.setup_study_loaders(
        train_ds, val_ds, collate, batch_size
    )

    # Distribution of gender classes
    freq = torch.zeros(3, dtype=torch.float32)
    for sample_obj in train_ds:
        freq[sample_obj.gender_class] += 1
    loss_weights = freq.sum() / (3 * freq.clamp(min=1))
    loss_weights = (loss_weights / loss_weights.sum() * 3).to(device)

    console.print(
        f"[bold blue]Training Gender (Train={len(train_ds)}, Val={len(val_ds)})[/bold blue]"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    # Final model training for gender
    tr_lda, vl_lda = train_loader, val_loader
    gender_model = study_utils.train_study_model(
        model,
        tr_lda,
        vl_lda,
        optimizer,
        criterion,
        device,
        console,
        epochs,
    )
    return cast(GenderClassifier, gender_model)


def run_gender_study(db_path: str) -> None:  # pylint: disable=unused-argument
    """Execute gender detection study."""
    gender_cache_dir = paths.get_style_dataset_cache_dir()
    if not study_utils.check_dataset_cache(console, gender_cache_dir):
        return

    gender_ds = GenderStudyDataset(gender_cache_dir)
    study_utils.run_standard_study(
        console,
        STUDY_DIR,
        None,
        gender_ds,
        GenderClassifier,
        partial(
            study_utils.collate_study_samples,
            dataset=gender_ds,
            label_attr="gender_class",
        ),
        CLASS_NAMES,
        _train_classifier,
    )


def apply_gender_changes(db_path: str) -> None:
    """Apply suggested gender changes."""
    console.print("[bold blue]Applying gender changes...[/bold blue]")
    updated: Dict[str, float] = {}
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    val_map = {"masculine": -1.0, "neutral": 0.0, "feminine": 1.0}
    # Iterate through suggestion mappings
    for cls_lbl, g_val in val_map.items():
        s_file = f"suggest {cls_lbl}.txt"
        s_path = os.path.join(STUDY_DIR, s_file)
        if os.path.exists(s_path):
            with open(s_path, "r", encoding="utf-8") as raw_f:
                sents = [line.strip() for line in raw_f if line.strip()]

            for s_text in sents:
                cursor.execute(
                    "UPDATE corpus SET gender = ? WHERE sentence = ?", (g_val, s_text)
                )
                if cursor.rowcount > 0:
                    updated[s_text] = g_val
            conn.commit()
            console.print(f"  Applied {len(sents):,} suggestions from {s_file}")

    study_utils.verify_database_updates(console, cursor, updated, "gender")
    study_utils.finalize_database_updates(console, conn, len(updated), db_path)


if __name__ == "__main__":
    db_file_path = os.path.join(paths.get_data_dir(), "corpus.db")
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument("--db-path", default=db_file_path)
    main_parser.add_argument("--apply", action="store_true")
    run_args = main_parser.parse_args()

    if run_args.apply:
        apply_gender_changes(run_args.db_path)
    else:
        run_gender_study(run_args.db_path)
