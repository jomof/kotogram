#!/usr/bin/env python3
"""Standalone script to evaluate model confusion and generate mismatch reports.
Extracted from train_style.py.
"""

import csv
import os
from typing import Any, Dict, List, Optional, cast

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch import nn
from torch.utils.data import DataLoader

# pylint: disable=wrong-import-position
from kotogram import locations
from kotogram.evaluator import Evaluator
from kotogram.model import (
    NUM_REGISTER_CLASSES,
    REGISTER_ID_TO_LABEL,
    StyleClassifier,
    load_model,
)
from train.dataset import DatasetConfig, StyleDataset, collate_fn

console = Console()


def calculate_metrics(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, Any]:
    """Run inference using Evaluator."""
    evaluator = Evaluator(cast(StyleClassifier, model), device)
    result = evaluator.evaluate(loader)
    return result.to_dict()


def print_confusion_matrix(
    title: str, labels: List[str], matrix: List[List[int]]
) -> None:
    """Print a confusion matrix using Rich Table."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("True \\ Pred", style="dim")
    for label in labels:
        table.add_column(label[:12])

    # pylint: disable=consider-using-enumerate
    for i in range(len(matrix)):
        row = matrix[i]
        row_str = [str(v) for v in row]
        # Highlight diagonals
        for j in range(len(row)):
            if i == j and row[j] > 0:
                row_str[j] = f"[bold green]{row_str[j]}[/bold green]"
        table.add_row(labels[i], *row_str)

    console.print(table)
    console.print()


def _add_formality_metrics(data: Dict[str, Any], summary: Table) -> None:
    f_prag_acc = sum(
        p == label
        for p, label in zip(data["formality_prag_preds"], data["formality_prag_labels"])
    ) / len(data["formality_prag_preds"])
    summary.add_row("Formality Pragmatic Accuracy", f"{f_prag_acc:.4%}")

    f_prag_mask = [label == 1 for label in data["formality_prag_labels"]]
    f_prag_preds = [p for p, m in zip(data["formality_val_preds"], f_prag_mask) if m]
    f_prag_labels = [
        label for label, m in zip(data["formality_val_labels"], f_prag_mask) if m
    ]
    if f_prag_labels:
        f_mse = sum(
            (p - label) ** 2 for p, label in zip(f_prag_preds, f_prag_labels)
        ) / len(f_prag_labels)
        summary.add_row("Formality Value MSE (Pragmatic samples)", f"{f_mse:.4f}")


def _add_gender_metrics(data: Dict[str, Any], summary: Table) -> None:
    g_prag_acc = sum(
        p == label
        for p, label in zip(data["gender_prag_preds"], data["gender_prag_labels"])
    ) / len(data["gender_prag_preds"])
    summary.add_row("Gender Pragmatic Accuracy", f"{g_prag_acc:.4%}")

    prag_mask = [label == 1 for label in data["gender_prag_labels"]]
    prag_preds = [p for p, m in zip(data["gender_val_preds"], prag_mask) if m]
    prag_labels = [label for label, m in zip(data["gender_val_labels"], prag_mask) if m]
    if prag_labels:
        g_mse = sum(
            (p - label) ** 2 for p, label in zip(prag_preds, prag_labels)
        ) / len(prag_labels)
        summary.add_row("Gender Value MSE (Pragmatic samples)", f"{g_mse:.4f}")


def _add_grammaticality_metrics(data: Dict[str, Any], summary: Table) -> None:
    gram_acc = sum(
        p == label
        for p, label in zip(data["grammaticality_preds"], data["grammaticality_labels"])
    ) / len(data["grammaticality_preds"])
    summary.add_row("Grammaticality Accuracy", f"{gram_acc:.4%}")


def _add_register_metrics(data: Dict[str, Any], summary: Table) -> None:
    reg_acc = sum(
        all(p[i] == label[i] for i in range(len(p)))
        for p, label in zip(data["register_preds"], data["register_labels"])
    ) / len(data["register_preds"])
    summary.add_row("Register Exact Match Accuracy", f"{reg_acc:.4%}")


def _print_confusion_matrices(data: Dict[str, Any]) -> None:
    # Formality Pragmatic
    f_labels = ["Unpragmatic", "Pragmatic"]
    f_confusion = [[0] * 2 for _ in range(2)]
    for p, label in zip(data["formality_prag_preds"], data["formality_prag_labels"]):
        f_confusion[label][p] += 1
    print_confusion_matrix(
        "Formality Pragmatic Confusion Matrix", f_labels, f_confusion
    )

    # Gender Pragmatic
    g_labels = ["Unpragmatic", "Pragmatic"]
    g_confusion = [[0] * 2 for _ in range(2)]
    for p, label in zip(data["gender_prag_preds"], data["gender_prag_labels"]):
        g_confusion[label][p] += 1
    print_confusion_matrix("Gender Pragmatic Confusion Matrix", g_labels, g_confusion)

    # Grammaticality
    gram_labels = ["Agrammatic", "Grammatic"]
    gram_confusion = [[0] * 2 for _ in range(2)]
    for p, label in zip(data["grammaticality_preds"], data["grammaticality_labels"]):
        gram_confusion[label][p] += 1
    print_confusion_matrix(
        "Grammaticality Confusion Matrix", gram_labels, gram_confusion
    )


def _print_register_report(data: Dict[str, Any]) -> None:
    reg_table = Table(
        title="Register Classification Report",
        show_header=True,
        header_style="bold yellow",
    )
    reg_table.add_column("Class", style="bold")
    reg_table.add_column("Precision")
    reg_table.add_column("Recall")
    reg_table.add_column("F1-Score")
    reg_table.add_column("Support")

    for i in range(NUM_REGISTER_CLASSES):
        label = REGISTER_ID_TO_LABEL[i].value
        tp = sum(
            1
            for p, label in zip(data["register_preds"], data["register_labels"])
            if p[i] == 1 and label[i] == 1
        )
        fp = sum(
            1
            for p, label in zip(data["register_preds"], data["register_labels"])
            if p[i] == 1 and label[i] == 0
        )
        fn = sum(
            1
            for p, label in zip(data["register_preds"], data["register_labels"])
            if p[i] == 0 and label[i] == 1
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        support = tp + fn

        reg_table.add_row(
            label, f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", f"{support}"
        )
    console.print(reg_table)
    console.print()


def _save_gender_mse_errors(data: Dict[str, Any], save_dir: str, sub_dir: str) -> None:
    # pylint: disable=too-many-locals
    prag_mask = [label == 1 for label in data["gender_prag_labels"]]
    prag_labels = [label for label, m in zip(data["gender_val_labels"], prag_mask) if m]

    if prag_labels:
        mse_errors = []
        for i, (pred, label, mask) in enumerate(
            zip(data["gender_val_preds"], data["gender_val_labels"], prag_mask)
        ):
            if mask:
                error = (pred - label) ** 2
                mse_errors.append(
                    {
                        "sentence": data["sentences"][i],
                        "predicted": f"{pred:.4f}",
                        "actual": f"{label:.4f}",
                        "error": error,
                        "kotogram": data["kotograms"][i]
                        if i < len(data["kotograms"])
                        else "",
                    }
                )

        if mse_errors:
            mse_errors.sort(key=lambda x: x["error"], reverse=True)
            top_mse = mse_errors[:50]

            out_path_csv = os.path.join(save_dir, "gender_mse_confusion.csv")
            out_path_tsv = os.path.join(sub_dir, "gender_mse_confusion.tsv")

            for out_path in [out_path_csv, out_path_tsv]:
                with open(out_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "sentence",
                            "predicted",
                            "actual",
                            "error",
                            "kotogram",
                        ],
                        delimiter="\t",
                    )
                    writer.writeheader()
                    writer.writerows(top_mse)
            console.print(
                f"[green]Saved top 50 gender MSE errors to {out_path_csv} and {out_path_tsv}[/green]"
            )


def _save_register_mismatches(
    data: Dict[str, Any], save_dir: str, sub_dir: str
) -> None:
    # pylint: disable=too-many-locals
    reg_mismatches = []
    # pylint: disable=consider-using-enumerate
    for i in range(len(data["register_preds"])):
        if any(
            data["register_preds"][i][j] != data["register_labels"][i][j]
            for j in range(NUM_REGISTER_CLASSES)
        ):
            p_names = [
                REGISTER_ID_TO_LABEL[j].value
                for j, val in enumerate(data["register_preds"][i])
                if val == 1
            ]
            l_names = [
                REGISTER_ID_TO_LABEL[j].value
                for j, val in enumerate(data["register_labels"][i])
                if val == 1
            ]
            reg_mismatches.append(
                {
                    "sentence": data["sentences"][i],
                    "predicted": ",".join(p_names),
                    "actual": ",".join(l_names),
                    "kotogram": data["kotograms"][i]
                    if i < len(data["kotograms"])
                    else "",
                }
            )

    if reg_mismatches:
        # Sort by kotogram to group similar grammatical structures, then sentence
        reg_mismatches.sort(key=lambda x: (x["kotogram"], x["sentence"]))

        out_path_csv = os.path.join(save_dir, "register_confusion.csv")
        out_path_tsv = os.path.join(sub_dir, "register_confusion.tsv")

        for out_path in [out_path_csv, out_path_tsv]:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["sentence", "predicted", "actual", "kotogram"],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerows(reg_mismatches)
        console.print(
            f"[green]Saved {len(reg_mismatches)} register mismatches to {out_path_csv} and {out_path_tsv}[/green]"
        )


def _save_mismatches(data: Dict[str, Any], save_dir: str) -> None:
    # pylint: disable=too-many-locals
    os.makedirs(save_dir, exist_ok=True)
    sub_dir = os.path.join(save_dir, "confusion_matrices")
    os.makedirs(sub_dir, exist_ok=True)

    tasks_mismatches = [
        (
            "formality",
            data["formality_prag_preds"],
            data["formality_prag_labels"],
            lambda x: ["Unpragmatic", "Pragmatic"][x],
        ),
        (
            "gender",
            data["gender_prag_preds"],
            data["gender_prag_labels"],
            lambda x: ["Unpragmatic", "Pragmatic"][x],
        ),
        (
            "grammaticality",
            data["grammaticality_preds"],
            data["grammaticality_labels"],
            lambda x: ["Agrammatic", "Grammatic"][x].lower(),
        ),
    ]

    for name, preds, labels, formatter in tasks_mismatches:
        mismatches = []
        for i, (pred, label) in enumerate(zip(preds, labels)):
            if pred != label:
                mismatches.append(
                    {
                        "sentence": data["sentences"][i],
                        "predicted": formatter(pred),
                        "actual": formatter(label),
                        "kotogram": data["kotograms"][i]
                        if i < len(data["kotograms"])
                        else "",
                    }
                )

        if mismatches:
            # Sort by kotogram to group similar grammatical structures, then sentence
            mismatches.sort(key=lambda x: (x["kotogram"], x["sentence"]))

            # Save to root as .csv
            out_path_csv = os.path.join(save_dir, f"{name}_confusion.csv")
            # Save to subdirectory as .tsv
            out_path_tsv = os.path.join(sub_dir, f"{name}_confusion.tsv")

            for out_path in [out_path_csv, out_path_tsv]:
                with open(out_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=["sentence", "predicted", "actual", "kotogram"],
                        delimiter="\t",
                    )
                    writer.writeheader()
                    writer.writerows(mismatches)
            console.print(
                f"[green]Saved {len(mismatches)} {name} mismatches to {out_path_csv} and {out_path_tsv}[/green]"
            )

    _save_gender_mse_errors(data, save_dir, sub_dir)
    _save_register_mismatches(data, save_dir, sub_dir)


def generate_reports(data: Dict[str, Any], save_dir: Optional[str]) -> None:
    """Calculate and display reports."""
    # Summary Table
    summary = Table(
        title="Overall Model Performance", show_header=True, header_style="bold cyan"
    )
    summary.add_column("Task")
    summary.add_column("Accuracy/MSE")

    _add_formality_metrics(data, summary)
    _add_gender_metrics(data, summary)
    _add_grammaticality_metrics(data, summary)
    _add_register_metrics(data, summary)

    console.print(Panel(summary, expand=False))

    _print_confusion_matrices(data)
    _print_register_report(data)

    if save_dir:
        _save_mismatches(data, save_dir)


def main() -> None:
    # pylint: disable=too-many-locals
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate model confusion and generate reports."
    )
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of workers for DataLoader (default: 0 on MPS/CPU, 4 on CUDA)",
    )
    parser.add_argument("--percent", type=float, help="Percentage of data to use")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to unified config.json file"
    )
    # parser.add_argument("--cache-dir", type=str, default=".cache", help="Base directory for dataset cache") # Removed

    args = parser.parse_args()

    # Resolve and inject paths from locations.py into args namespace
    cache_dir = locations.get_cache_dir()
    args.output = locations.get_style_support_dir()
    args.support_dir = args.output
    args.model_dir = locations.get_style_output_dir()
    args.data = os.path.join(cache_dir, "grammatic_combined.tsv")
    args.agrammatic_data = os.path.join(cache_dir, "agrammatic_combined.tsv")

    from train.config import TrainerConfig

    if args.config:
        # Load unified config from file
        _, trainer_config = TrainerConfig.load_config(args.config)
        # Use batch size from config if not explicitly set to something else?
        # Actually, let's just use what's in the config by default, but allow override.
        # But if the user didn't specify --batch-size, args.batch_size will be 512.
        # Let's check if it was changed from default.
        # For now, let's just always prefer the config if it's there.
        args.batch_size = trainer_config.batch_size
        if trainer_config.dataloader.num_workers is not None:
            args.num_workers = trainer_config.dataloader.num_workers

    # Restore percent from checkpoint if not explicitly provided
    checkpoint_path = os.path.join(args.support_dir, "checkpoint.pt")
    if args.percent is None and os.path.exists(checkpoint_path):
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        saved_args = checkpoint_data.get("args", {})
        saved_percent = saved_args.get("percent")
        if saved_percent is not None:
            args.percent = saved_percent
            console.print(
                f"[dim]Restored --percent {args.percent} from checkpoint[/dim]"
            )

    device_name = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    device = torch.device(device_name)
    console.print(
        f"Evaluating [bold cyan]model.pt[/bold cyan] in: [bold cyan]{os.path.abspath(args.model_dir)}[/bold cyan]"
    )
    console.print(
        f"CSV output directory: [bold cyan]{os.path.abspath(args.output)}[/bold cyan]"
    )
    console.print(f"Using device: [bold blue]{device_name}[/bold blue]")

    # Load model and tokenizer
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_dir, device=device_name)

    # Building evaluation files list
    data_files = []
    grammaticality_labels = []

    def add_file(path: Optional[str], force_label: Optional[int] = None) -> None:
        if not path:
            return
        data_files.append(path)
        if force_label is not None:
            grammaticality_labels.append(force_label)
        else:
            # Heuristic based on filename
            is_agrammatic = (
                "agrammatic" in os.path.basename(path).lower()
                or "error" in os.path.basename(path).lower()
            )
            grammaticality_labels.append(0 if is_agrammatic else 1)

    add_file(args.data, force_label=1)  # Main data is always assumed grammatic (or 1)
    # Add agrammatic sentences if provided
    if os.path.exists(args.agrammatic_data):
        add_file(args.agrammatic_data, force_label=0)

    console.print(f"Loading data from: {data_files}")

    # Load dataset
    if len(data_files) > 1:
        dataset = StyleDataset.from_multiple_tsv(
            data_files,
            tokenizer,
            config=DatasetConfig(
                grammaticality_labels=grammaticality_labels,
                sample_ratio=args.percent / 100.0 if args.percent else 1.0,
                verbose=True,
            ),
        )
    else:
        dataset = StyleDataset.from_tsv(
            data_files[0],
            tokenizer,
            config=DatasetConfig(
                sample_ratio=args.percent / 100.0 if args.percent else 1.0,
                verbose=True,
            ),
        )

    # Determine num_workers: 0 is much faster for in-memory datasets on macOS (avoid spawn overhead)
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = 4 if device.type == "cuda" else 0

    from functools import partial

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_fn, pad_id=tokenizer.pad_id, max_seq_len=model.config.max_seq_len
        ),
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Calculate metrics
    results = calculate_metrics(model, loader, device)

    # Generate reports
    generate_reports(results, args.output)

    console.print("[bold green]Confusion analysis complete.[/bold green]")


if __name__ == "__main__":
    main()
