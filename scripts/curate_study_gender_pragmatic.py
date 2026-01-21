#!/usr/bin/env python3
# ruff: noqa: E402
# pylint: disable=wrong-import-position
"""
Gender pragmatic learnability study for corpus.db.

Identifies sentences that are likely mislabeled as unpragmatic (neutral gender)
when they actually carry pragmatic gender information.
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
CLASS_NAMES = ["unpragmatic", "pragmatic"]
STUDY_DIR = ".cache/curate/study/gender_pragmatic"


@dataclass
class GenderPragmaticSample:
    """A sample for gender pragmatic classification."""

    idx: int
    sentence: str
    gender_pragmatic: int
    feature_start: int
    feature_end: int


class GenderPragmaticDataset(study_utils.BaseStudyDataset[GenderPragmaticSample]):
    """Dataset for gender pragmatic classification study."""

    def __init__(
        self,
        data_dir: str,
        indices: Optional[torch.Tensor] = None,
    ):
        super().__init__(data_dir, indices)

        # Load gender pragmatic flags
        g_prag_path = os.path.join(data_dir, "labels.bin_g_prag")
        g_prag_size = os.path.getsize(g_prag_path)
        self.gender_pragmatic = torch.from_file(
            g_prag_path, shared=True, size=g_prag_size, dtype=torch.uint8
        )

    def __getitem__(self, index: int) -> GenderPragmaticSample:
        row_id = int(self.indices[index].item())
        offset_val = int(self.offsets[row_id].item())
        next_offset = int(self.offsets[row_id + 1].item())
        is_pragmatic = int(self.gender_pragmatic[row_id].item())

        return GenderPragmaticSample(
            idx=row_id,
            sentence=self.sentences[row_id],
            gender_pragmatic=is_pragmatic,
            feature_start=offset_val,
            feature_end=next_offset,
        )


class GenderPragmaticClassifier(study_utils.BaseMLPStudyClassifier):
    """Simple classifier for gender pragmatic prediction."""

    def __init__(
        self, vocab_sizes: Dict[str, int], embed_dim: int = 32, hidden_dim: int = 128
    ):
        super().__init__(vocab_sizes, embed_dim, hidden_dim, num_classes=2)


# pylint: disable=too-many-locals
def _train_classifier(
    dataset: GenderPragmaticDataset,
    vocab_sizes: Dict[str, int],
    epochs: int = 10,
    batch_size: int = 256,
) -> GenderPragmaticClassifier:
    """Train a gender pragmatic classifier."""
    device = study_utils.get_device()
    model = GenderPragmaticClassifier(vocab_sizes).to(device)

    # Data
    parts = study_utils.split_dataset_indices(dataset.indices)
    train_ds, val_ds = study_utils.prepare_study_data(
        dataset, parts[0], parts[1], GenderPragmaticDataset
    )

    collate = partial(
        study_utils.collate_study_samples,
        dataset=dataset,
        label_attr="gender_pragmatic",
    )
    train_loader, val_loader = study_utils.setup_study_loaders(
        train_ds, val_ds, collate, batch_size
    )

    # Distribution
    cls_counts = torch.zeros(2, dtype=torch.float32)
    for s in train_ds:
        cls_counts[s.gender_pragmatic] += 1
    cls_weights = cls_counts.sum() / (2 * cls_counts.clamp(min=1))
    cls_weights = (cls_weights / cls_weights.sum() * 2).to(device)

    console.print(
        f"[bold blue]Training (Train={len(train_ds)}, Val={len(val_ds)})[/bold blue]"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=cls_weights)

    # Use centralized study training loop for gender pragmatic
    prag_trained = study_utils.train_study_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        console,
        max_epochs=epochs,
    )
    return cast(GenderPragmaticClassifier, prag_trained)


def run_gender_pragmatic_study(db_path: str) -> None:  # pylint: disable=unused-argument
    """Execute gender pragmatic detection study."""
    gender_pragmatic_data = paths.get_style_dataset_cache_dir()
    if not study_utils.check_dataset_cache(console, gender_pragmatic_data):
        return

    gender_pragmatic_ds = GenderPragmaticDataset(gender_pragmatic_data)
    study_utils.run_standard_study(
        console,
        STUDY_DIR,
        None,
        gender_pragmatic_ds,
        GenderPragmaticClassifier,
        partial(
            study_utils.collate_study_samples,
            dataset=gender_pragmatic_ds,
            label_attr="gender_pragmatic",
        ),
        CLASS_NAMES,
        _train_classifier,
    )


def apply_gender_pragmatic_changes(db_path: str) -> None:
    """Apply suggested gender pragmatic changes."""
    console.print("[bold blue]Applying changes...[/bold blue]")
    updated: Dict[str, Optional[float]] = {}
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Process each suggestion file
    for filename, cls_val in [
        ("suggest unpragmatic.txt", None),
        ("suggest pragmatic.txt", 0.0),
    ]:
        fpath = os.path.join(STUDY_DIR, filename)
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as stream:
                for line in stream:
                    sentence_text = line.strip()
                    if sentence_text:
                        if cls_val is None:
                            cursor.execute(
                                "UPDATE corpus SET gender = NULL, grammatic = 0 WHERE sentence = ?",
                                (sentence_text,),
                            )
                        else:
                            cursor.execute(
                                "UPDATE corpus SET gender = ? WHERE sentence = ?",
                                (cls_val, sentence_text),
                            )

                        if cursor.rowcount > 0:
                            updated[sentence_text] = cls_val
            conn.commit()

    study_utils.verify_database_updates(console, cursor, updated, "gender")
    study_utils.finalize_database_updates(console, conn, len(updated), db_path)


if __name__ == "__main__":
    db_loc = os.path.join(paths.get_data_dir(), "corpus.db")
    script_parser = argparse.ArgumentParser()
    script_parser.add_argument("--db-path", default=db_loc)
    script_parser.add_argument("--apply", action="store_true")
    script_args = script_parser.parse_args()

    if script_args.apply:
        apply_gender_pragmatic_changes(script_args.db_path)
    else:
        run_gender_pragmatic_study(script_args.db_path)
