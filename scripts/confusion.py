#!/usr/bin/env python3
"""Standalone script to evaluate model confusion and generate mismatch reports.
Extracted from train_style.py.
"""

import os
import sys
import csv
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Add project root to path to allow imports from kotogram
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kotogram.model import (
    StyleClassifier, Tokenizer, ModelConfig,
    FEATURE_FIELDS, 
    NUM_FORMALITY_CLASSES, NUM_GENDER_PRAGMATIC_CLASSES, NUM_GRAMMATICALITY_CLASSES, NUM_REGISTER_CLASSES,
    FORMALITY_ID_TO_LABEL, GENDER_ID_TO_LABEL, REGISTER_ID_TO_LABEL,
    load_model
)
from scripts.train_style import StyleDataset, collate_fn

console = Console()


from kotogram.evaluator import Evaluator

def calculate_metrics(model, loader, device):
    """Run inference using Evaluator."""
    evaluator = Evaluator(model, device, verbose=True)
    result = evaluator.evaluate(loader)
    return result.to_dict()

def print_confusion_matrix(title, labels, matrix):
    """Print a confusion matrix using Rich Table."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("True \\ Pred", style="dim")
    for label in labels:
        table.add_column(label[:12])
    
    for i, row in enumerate(matrix):
        row_str = [str(v) for v in row]
        # Highlight diagonals
        for j in range(len(row)):
            if i == j and row[j] > 0:
                row_str[j] = f"[bold green]{row_str[j]}[/bold green]"
        table.add_row(labels[i], *row_str)
    
    console.print(table)
    console.print()

def generate_reports(data, save_dir):
    """Calculate and display reports."""
    # Summary Table
    summary = Table(title="Overall Model Performance", show_header=True, header_style="bold cyan")
    summary.add_column("Task")
    summary.add_column("Accuracy/MSE")
    
    # Formality
    f_acc = sum(p == l for p, l in zip(data['formality_preds'], data['formality_labels'])) / len(data['formality_preds'])
    summary.add_row("Formality Accuracy", f"{f_acc:.4%}")
    
    # Gender
    g_prag_acc = sum(p == l for p, l in zip(data['gender_prag_preds'], data['gender_prag_labels'])) / len(data['gender_prag_preds'])
    summary.add_row("Gender Pragmatic Accuracy", f"{g_prag_acc:.4%}")
    
    prag_mask = [l == 1 for l in data['gender_prag_labels']]
    prag_preds = [p for p, m in zip(data['gender_val_preds'], prag_mask) if m]
    prag_labels = [l for l, m in zip(data['gender_val_labels'], prag_mask) if m]
    if prag_labels:
        g_mse = sum((p - l) ** 2 for p, l in zip(prag_preds, prag_labels)) / len(prag_labels)
        summary.add_row("Gender Value MSE (Pragmatic samples)", f"{g_mse:.4f}")
    
    # Grammaticality
    gram_acc = sum(p == l for p, l in zip(data['grammaticality_preds'], data['grammaticality_labels'])) / len(data['grammaticality_preds'])
    summary.add_row("Grammaticality Accuracy", f"{gram_acc:.4%}")
    
    # Register (Exact Match)
    reg_acc = sum(all(p[i] == l[i] for i in range(len(p))) for p, l in zip(data['register_preds'], data['register_labels'])) / len(data['register_preds'])
    summary.add_row("Register Exact Match Accuracy", f"{reg_acc:.4%}")
    
    console.print(Panel(summary, expand=False))

    # Confusion Matrices
    # Formality
    f_labels = [FORMALITY_ID_TO_LABEL[i].value for i in range(NUM_FORMALITY_CLASSES)]
    f_confusion = [[0] * NUM_FORMALITY_CLASSES for _ in range(NUM_FORMALITY_CLASSES)]
    for p, l in zip(data['formality_preds'], data['formality_labels']):
        f_confusion[l][p] += 1
    print_confusion_matrix("Formality Confusion Matrix", f_labels, f_confusion)
    
    # Gender Pragmatic
    g_labels = ["Unpragmatic", "Pragmatic"]
    g_confusion = [[0] * 2 for _ in range(2)]
    for p, l in zip(data['gender_prag_preds'], data['gender_prag_labels']):
        g_confusion[l][p] += 1
    print_confusion_matrix("Gender Pragmatic Confusion Matrix", g_labels, g_confusion)
    
    # Grammaticality
    gram_labels = ["Agrammatic", "Grammatic"]
    gram_confusion = [[0] * 2 for _ in range(2)]
    for p, l in zip(data['grammaticality_preds'], data['grammaticality_labels']):
        gram_confusion[l][p] += 1
    print_confusion_matrix("Grammaticality Confusion Matrix", gram_labels, gram_confusion)

    # Register Report
    reg_table = Table(title="Register Classification Report", show_header=True, header_style="bold yellow")
    reg_table.add_column("Class", style="bold")
    reg_table.add_column("Precision")
    reg_table.add_column("Recall")
    reg_table.add_column("F1-Score")
    reg_table.add_column("Support")
    
    for i in range(NUM_REGISTER_CLASSES):
        label = REGISTER_ID_TO_LABEL[i].value
        tp = sum(1 for p, l in zip(data['register_preds'], data['register_labels']) if p[i] == 1 and l[i] == 1)
        fp = sum(1 for p, l in zip(data['register_preds'], data['register_labels']) if p[i] == 1 and l[i] == 0)
        fn = sum(1 for p, l in zip(data['register_preds'], data['register_labels']) if p[i] == 0 and l[i] == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp + fn
        
        reg_table.add_row(
            label, 
            f"{precision:.4f}", 
            f"{recall:.4f}", 
            f"{f1:.4f}", 
            f"{support}"
        )
    console.print(reg_table)
    console.print()

    # Save Mismatches
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        tasks_mismatches = [
            ('formality', data['formality_preds'], data['formality_labels'], lambda x: FORMALITY_ID_TO_LABEL[x].value),
            ('gender', data['gender_prag_preds'], data['gender_prag_labels'], lambda x: g_labels[x]),
            ('grammaticality', data['grammaticality_preds'], data['grammaticality_labels'], lambda x: gram_labels[x].lower()),
        ]
        
        for name, preds, labels, formatter in tasks_mismatches:
            mismatches = []
            for i in range(len(preds)):
                if preds[i] != labels[i]:
                    mismatches.append({
                        'sentence': data['sentences'][i],
                        'predicted': formatter(preds[i]),
                        'actual': formatter(labels[i]),
                        'kotogram': data['kotograms'][i] if i < len(data['kotograms']) else ''
                    })
            
            if mismatches:
                # Sort by kotogram to group similar grammatical structures, then sentence
                mismatches.sort(key=lambda x: (x['kotogram'], x['sentence']))
                out_path = os.path.join(save_dir, f'{name}_confusion.csv')
                with open(out_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=['sentence', 'predicted', 'actual', 'kotogram'], delimiter='\t')
                    writer.writeheader()
                    writer.writerows(mismatches)
                console.print(f"[green]Saved {len(mismatches)} {name} mismatches to {out_path}[/green]")


        # Gender MSE worst matches
        if prag_labels:
            mse_errors = []
            for i in range(len(data['gender_val_preds'])):
                if prag_mask[i]:
                    error = (data['gender_val_preds'][i] - data['gender_val_labels'][i]) ** 2
                    mse_errors.append({
                        'sentence': data['sentences'][i],
                        'predicted': f"{data['gender_val_preds'][i]:.4f}",
                        'actual': f"{data['gender_val_labels'][i]:.4f}",
                        'error': error,
                        'kotogram': data['kotograms'][i] if i < len(data['kotograms']) else ''
                    })
            
            if mse_errors:
                mse_errors.sort(key=lambda x: x['error'], reverse=True)
                top_mse = mse_errors[:50]
                out_path = os.path.join(save_dir, 'gender_mse_confusion.csv')
                with open(out_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=['sentence', 'predicted', 'actual', 'error', 'kotogram'], delimiter='\t')
                    writer.writeheader()
                    writer.writerows(top_mse)
                console.print(f"[green]Saved top 50 gender MSE errors to {out_path}[/green]")

        # Register mismatches
        reg_mismatches = []
        for i in range(len(data['register_preds'])):
            if any(data['register_preds'][i][j] != data['register_labels'][i][j] for j in range(NUM_REGISTER_CLASSES)):
                p_names = [REGISTER_ID_TO_LABEL[j].value for j, val in enumerate(data['register_preds'][i]) if val == 1]
                l_names = [REGISTER_ID_TO_LABEL[j].value for j, val in enumerate(data['register_labels'][i]) if val == 1]
                reg_mismatches.append({
                    'sentence': data['sentences'][i],
                    'predicted': ",".join(p_names),
                    'actual': ",".join(l_names),
                    'kotogram': data['kotograms'][i] if i < len(data['kotograms']) else ''
                })
        
        if reg_mismatches:
            # Sort by kotogram to group similar grammatical structures, then sentence
            reg_mismatches.sort(key=lambda x: (x['kotogram'], x['sentence']))
            out_path = os.path.join(save_dir, 'register_confusion.csv')
            with open(out_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['sentence', 'predicted', 'actual', 'kotogram'], delimiter='\t')
                writer.writeheader()
                writer.writerows(reg_mismatches)
            console.print(f"[green]Saved {len(reg_mismatches)} register mismatches to {out_path}[/green]")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model confusion and generate reports.")
    parser.add_argument("--output", type=str, required=True, help="Model directory containing checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to evaluation data TSV")
    parser.add_argument("--agrammatic-sentences", type=str, help="Path to agrammatic sentences TSV (label 0)")
    parser.add_argument("--agrammatic-data", type=str, help="Path to agrammatic evaluation data TSV")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, help="Number of workers for DataLoader (default: 0 on MPS/CPU, 4 on CUDA)")
    parser.add_argument("--max-samples", type=int, default=None, help="Stop after N samples")
    parser.add_argument("--percent", type=float, help="Percentage of data to use")
    
    args = parser.parse_args()
    
    device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    console.print(f"Evaluating [bold cyan]model.pt[/bold cyan] in: [bold cyan]{os.path.abspath(args.output)}[/bold cyan]")
    console.print(f"Using device: [bold blue]{device_name}[/bold blue]")
    
    # Load model and tokenizer
    try:
        model, tokenizer = load_model(args.output, device=device)
    except Exception as e:
        console.print(f"[bold red]Error loading model from {args.output}: {e}[/bold red]")
        sys.exit(1)
    
    # Building evaluation files list
    data_files = []
    grammaticality_labels = []

    def add_file(path, force_label=None):
        if not path: return
        data_files.append(path)
        if force_label is not None:
             grammaticality_labels.append(force_label)
        else:
             # Heuristic based on filename
             is_agrammatic = "agrammatic" in os.path.basename(path).lower() or "error" in os.path.basename(path).lower()
             grammaticality_labels.append(0 if is_agrammatic else 1)

    add_file(args.data, force_label=1) # Main data is always assumed grammatic (or 1)
    # Add agrammatic sentences if provided
    if args.agrammatic_sentences:
        add_file(args.agrammatic_sentences, force_label=0) 
    if args.agrammatic_data:
        add_file(args.agrammatic_data, force_label=0)

        
    console.print(f"Loading data from: {data_files}")
    
    # Load dataset
    if len(data_files) > 1:
        dataset = StyleDataset.from_multiple_tsv(
            data_files,
            tokenizer,
            labeled=True,
            grammaticality_labels=grammaticality_labels,
            max_samples=args.max_samples,
            sample_ratio=args.percent / 100.0 if args.percent else 1.0,
            verbose=True,
            cache_dir=".cache/dataset_cache"
        )
    else:
        dataset = StyleDataset.from_tsv(
            data_files[0],
            tokenizer,
            labeled=True,
            max_samples=args.max_samples,
            sample_ratio=args.percent / 100.0 if args.percent else 1.0,
            verbose=True,
            cache_dir=".cache/dataset_cache"
        )
    
    # Determine num_workers: 0 is much faster for in-memory datasets on macOS (avoid spawn overhead)
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = 4 if device.type == 'cuda' else 0

    from functools import partial
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_id=tokenizer.pad_id, max_seq_len=model.config.max_seq_len),
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    # Calculate metrics
    results = calculate_metrics(model, loader, device)
    
    # Generate reports
    generate_reports(results, args.output)
    
    console.print("[bold green]Confusion analysis complete.[/bold green]")

if __name__ == "__main__":
    main()
