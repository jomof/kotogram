#!/usr/bin/env python3
"""Standalone script to label and cache Japanese sentences for style classification.
Factored out from train_style.py.

NOTE FOR SELF: Never use conditional imports surrounded by try/except.
It makes the code harder to reason about and can hide installation issues.
"""

import os
import sys
import csv
import glob
import hashlib
import json
import time
import multiprocessing as mp
from collections import Counter
from typing import Dict, List, Optional, Tuple, Set, NamedTuple, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn

from scripts.cache import get_kotogram_cache
from kotogram.model import FORMALITY_LABEL_TO_ID, FORMALITY_ID_TO_LABEL, REGISTER_LABEL_TO_ID, REGISTER_ID_TO_LABEL

console = Console()

class ProcessedSample(NamedTuple):
    sentence: str
    sentence_id: str
    kotogram: str
    formality_id: int
    gender_value: float
    gender_pragmatic: int
    register_ids: List[int]
    gram_label: int
    success: int

def _process_sentence_batch(batch: List[Tuple[str, str, int]]) -> List[ProcessedSample]:
    """Process a batch of sentences in a worker process."""
    from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
    from kotogram.analysis import FormalityLevel, GenderLevel, RegisterLevel
    
    from scripts.rule_based_analysis import analyze_formality, analyze_gender, analyze_register


    parser = SudachiJapaneseParser()
    results = []

    for sentence, sentence_id, gram_label in batch:
        try:
            kotogram = parser.japanese_to_kotogram(sentence)
            formality_enum = analyze_formality(kotogram)
            gender_enum = analyze_gender(kotogram)
            register_enums = analyze_register(kotogram)
            
            formality_id = FORMALITY_LABEL_TO_ID.get(formality_enum, FORMALITY_LABEL_TO_ID[FormalityLevel.NEUTRAL])
            
            if gender_enum == GenderLevel.MASCULINE:
                gender_val, gender_prag = -1.0, 1
            elif gender_enum == GenderLevel.FEMININE:
                gender_val, gender_prag = 1.0, 1
            elif gender_enum == GenderLevel.NEUTRAL:
                gender_val, gender_prag = 0.0, 1
            else: # UNPRAGMATIC_GENDER
                gender_val, gender_prag = 0.0, 0
            
            register_ids = [REGISTER_LABEL_TO_ID[r] for r in register_enums if r in REGISTER_LABEL_TO_ID]
            if not register_ids:
                register_ids = [REGISTER_LABEL_TO_ID[RegisterLevel.NEUTRAL]]

            results.append(ProcessedSample(
                sentence=sentence,
                sentence_id=sentence_id,
                kotogram=kotogram,
                formality_id=formality_id,
                gender_value=gender_val,
                gender_pragmatic=gender_prag,
                register_ids=register_ids,
                gram_label=gram_label,
                success=1
            ))
        except Exception:
            results.append(ProcessedSample(sentence, sentence_id, "", 0, 0.0, 0, [], gram_label, 0))
    return results

def _compute_labels_batch(batch: List[Tuple[str, str, int]]) -> List[ProcessedSample]:
    """Compute labels for a batch of sentences (where kotogram is already cached)."""
    from kotogram.analysis import FormalityLevel, GenderLevel, RegisterLevel
    
    from scripts.rule_based_analysis import analyze_formality, analyze_gender, analyze_register


    results = []
    
    for sentence, kotogram, gram_label in batch:
        try:
            formality_enum = analyze_formality(kotogram)
            gender_enum = analyze_gender(kotogram)
            register_enums = analyze_register(kotogram)
            
            formality_id = FORMALITY_LABEL_TO_ID.get(formality_enum, FORMALITY_LABEL_TO_ID[FormalityLevel.NEUTRAL])
            
            if gender_enum == GenderLevel.MASCULINE:
                gender_val, gender_prag = -1.0, 1
            elif gender_enum == GenderLevel.FEMININE:
                gender_val, gender_prag = 1.0, 1
            elif gender_enum == GenderLevel.NEUTRAL:
                gender_val, gender_prag = 0.0, 1
            else: # UNPRAGMATIC_GENDER
                gender_val, gender_prag = 0.0, 0
            
            register_ids = [REGISTER_LABEL_TO_ID[r] for r in register_enums if r in REGISTER_LABEL_TO_ID]
            if not register_ids:
                register_ids = [REGISTER_LABEL_TO_ID[RegisterLevel.NEUTRAL]]

            results.append(ProcessedSample(
                sentence=sentence,
                sentence_id="",
                kotogram=kotogram,
                formality_id=formality_id,
                gender_value=gender_val,
                gender_pragmatic=gender_prag,
                register_ids=register_ids,
                gram_label=gram_label,
                success=1
            ))
        except Exception:
            results.append(ProcessedSample(sentence, "", kotogram, 0, 0.0, 0, [], gram_label, 0))
            
    return results

def print_stats(results: List[ProcessedSample]):
    """Print attractive statistics about the labeling results."""
    if not results:
        return

    # Formality Stats
    formality_counts = Counter(r.formality_id for r in results if r.success)
    f_table = Table(title="Formality Distribution", show_header=True, header_style="bold magenta")
    f_table.add_column("Level", style="dim")
    f_table.add_column("Count", justify="right")
    f_table.add_column("Percentage", justify="right")
    
    total = sum(formality_counts.values())
    for fid in sorted(formality_counts.keys()):
        label = FORMALITY_ID_TO_LABEL[fid].value
        count = formality_counts[fid]
        f_table.add_row(label, f"{count:,}", f"{100*count/total:.1f}%")
    
    # Gender Stats
    gender_counts = Counter(r.gender_pragmatic for r in results if r.success)
    g_table = Table(title="Gender Pragmatic Distribution", show_header=True, header_style="bold cyan")
    g_table.add_column("Type", style="dim")
    g_table.add_column("Count", justify="right")
    g_table.add_column("Percentage", justify="right")
    
    g_map = {1: "Pragmatic", 0: "Unpragmatic"}
    for gid in sorted(gender_counts.keys()):
        count = gender_counts[gid]
        g_table.add_row(g_map[gid], f"{count:,}", f"{100*count/total:.1f}%")

    # Register Stats
    register_counts = Counter()
    for r in results:
        if r.success:
            for rid in r.register_ids:
                register_counts[rid] += 1
    
    r_table = Table(title="Register Distribution", show_header=True, header_style="bold yellow")
    r_table.add_column("Register", style="dim")
    r_table.add_column("Count", justify="right")
    r_table.add_column("Percentage", justify="right")
    
    for rid in sorted(register_counts.keys()):
        label = REGISTER_ID_TO_LABEL[rid].value
        count = register_counts[rid]
        r_table.add_row(label, f"{count:,}", f"{100*count/total:.1f}%")

    # Grammaticality Stats
    gram_counts = Counter(r.gram_label for r in results if r.success)
    gram_table = Table(title="Grammaticality Distribution", show_header=True, header_style="bold green")
    gram_table.add_column("Type", style="dim")
    gram_table.add_column("Count", justify="right")
    gram_table.add_column("Percentage", justify="right")
    
    gram_map = {1: "Grammatic", 0: "Agrammatic"}
    for gid in sorted(gram_counts.keys()):
        count = gram_counts[gid]
        gram_table.add_row(gram_map[gid], f"{count:,}", f"{100*count/total:.1f}%")

    console.print(Panel.fit(f_table, border_style="magenta"))
    console.print(Panel.fit(g_table, border_style="cyan"))
    console.print(Panel.fit(r_table, border_style="yellow"))
    console.print(Panel.fit(gram_table, border_style="green"))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Label and cache Japanese sentences.")
    parser.add_argument("--data", type=str, required=True, help="Primary TSV data file(s) (glob pattern)")
    parser.add_argument("--agrammatic-sentences", type=str, help="Path to agrammatic TSV file (label 0)")
    parser.add_argument("--agrammatic-pattern", type=str, help="Agrammatic TSV pattern")
    parser.add_argument("--output-grammatic", type=str, help="Path to save combined/deduplicated grammatic data")
    parser.add_argument("--output-agrammatic", type=str, help="Path to save combined/deduplicated agrammatic data")
    parser.add_argument("--num-workers", type=int, help="Number of workers")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to process")
    parser.add_argument("--percent", type=float, help="Percentage of data to use")
    parser.add_argument("--output-dir", type=str, default=".cache", help="Output directory for dataset cache")
    
    args = parser.parse_args()
    
    num_workers = args.num_workers or max(1, mp.cpu_count() - 1)
    
    def process_file_group(patterns, gram_label, output_path=None):
        if not patterns: return [], 0
        
        file_list = []
        if isinstance(patterns, str):
            file_list = glob.glob(patterns)
        else:
            for p in patterns:
                file_list.extend(glob.glob(p))
                
        if not file_list:
            return [], 0

        unique_rows = [] # (sentence, id, gram_label)
        seen = set()
        raw_rows = [] # (id, lang, sentence) to be written to output
        
        for f_path in sorted(file_list):
            with open(f_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if len(row) < 3: continue
                    sentence = row[2]
                    if sentence not in seen:
                        seen.add(sentence)
                        unique_rows.append((sentence, row[0], gram_label))
                        raw_rows.append(row)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            import io
            out = io.StringIO()
            writer = csv.writer(out, delimiter='\t', lineterminator='\n')
            writer.writerows(raw_rows)
            new_content = out.getvalue()
            
            should_write = True
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    if f.read() == new_content:
                        console.print(f"  [dim]{os.path.basename(output_path)} unchanged, skipping write.[/dim]")
                        should_write = False
            
            if should_write:
                console.print(f"  Writing {len(raw_rows):,} unique rows to [bold]{output_path}[/bold]...")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                    
        return unique_rows, len(file_list)

    all_rows = []
    
    # Process grammatic (only primary data)
    gram_patterns = [args.data]
    
    console.print(f"Processing [bold]grammatic[/bold] data ({len(gram_patterns)} patterns) with {num_workers} workers...")
    rows, count = process_file_group(gram_patterns, 1, args.output_grammatic)
    all_rows.extend(rows)
    if count > 0:
        console.print(f"  Matched {count} grammatic files.")
    
    # Process agrammatic (agrammatic-sentences + agrammatic-pattern)
    agram_patterns = []
    if args.agrammatic_sentences:
        agram_patterns.append(args.agrammatic_sentences)
    if args.agrammatic_pattern:
        agram_patterns.append(args.agrammatic_pattern)
        
    if agram_patterns:
        console.print(f"Processing [bold]agrammatic[/bold] data ({len(agram_patterns)} patterns)...")
        rows, count = process_file_group(agram_patterns, 0, args.output_agrammatic)
        all_rows.extend(rows)
        if count > 0:
            console.print(f"  Matched {count} agrammatic files.")

    if not all_rows:
        console.print("[red]No data sentences found. Check your patterns.[/red]")
        sys.exit(1)
        
    console.print(f"Total unique sentences to check: [bold]{len(all_rows):,}[/bold]")
    
    cache = get_kotogram_cache()
    cached_batch = cache.get_batch([r[0] for r in all_rows])
    
    uncached_rows = []
    unlabeled_rows = []
    final_results = []
    
    for sentence, sentence_id, gram_label in all_rows:
        entry = cached_batch.get(sentence)
        if entry:
            k, f, g_val, g_prag, r_lbls = entry
            if f is not None and g_val is not None and g_prag is not None and r_lbls is not None:
                final_results.append(ProcessedSample(sentence, sentence_id, k, f, g_val, g_prag, r_lbls, gram_label, 1))
            else:
                unlabeled_rows.append((sentence, k, gram_label))
        else:
            uncached_rows.append((sentence, sentence_id, gram_label))
            
    console.print(f"Cache status: {len(final_results):,} hits, {len(unlabeled_rows):,} partial, {len(uncached_rows):,} misses")
    
    ctx = mp.get_context('spawn')
    
    if uncached_rows or unlabeled_rows:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            console=console
        ) as progress:
            
            if uncached_rows:
                task = progress.add_task("[green]Parsing & Labeling...", total=len(uncached_rows))
                batches = [uncached_rows[i:i + args.batch_size] for i in range(0, len(uncached_rows), args.batch_size)]
                
                new_entries = []
                with ctx.Pool(num_workers) as pool:
                    for batch_results in pool.imap(_process_sentence_batch, batches):
                        for res in batch_results:
                            if res.success:
                                final_results.append(res)
                                new_entries.append((res.sentence, res.kotogram, res.formality_id, res.gender_value, res.gender_pragmatic, res.register_ids))
                        progress.update(task, advance=len(batch_results))
                
                if new_entries:
                    cache.put_batch(new_entries)
                    
            if unlabeled_rows:
                task = progress.add_task("[cyan]Re-labeling...", total=len(unlabeled_rows))
                batches = [unlabeled_rows[i:i + args.batch_size] for i in range(0, len(unlabeled_rows), args.batch_size)]
                
                new_entries = []
                with ctx.Pool(num_workers) as pool:
                    for batch_results in pool.imap(_compute_labels_batch, batches):
                        for res in batch_results:
                            if res.success:
                                final_results.append(res)
                                new_entries.append((res.sentence, res.kotogram, res.formality_id, res.gender_value, res.gender_pragmatic, res.register_ids))
                        progress.update(task, advance=len(batch_results))
                
                if new_entries:
                    cache.put_batch(new_entries)

    console.print(f"\n[bold green]Processing complete![/bold green] Total processed: {len(final_results):,}")
    print_stats(final_results)

    # Phase 3: Build vocabulary and encode samples (Warming the StyleDataset cache)
    # Only run if we have output paths (meaning we're running in preprocessing mode)
    if args.output_grammatic:
        from scripts.train_style import StyleDataset, Tokenizer
        
        console.print("\n[bold blue]Phase 3: Building vocabulary and encoding samples...[/bold blue]")
        # We pass the combined files generated earlier
        eval_files = [args.output_grammatic]
        eval_labels = [1]
        if args.output_agrammatic and os.path.exists(args.output_agrammatic):
            eval_files.append(args.output_agrammatic)
            eval_labels.append(0)
        
        # Initialize tokenizer (StyleDataset will handle freezing)
        tokenizer = Tokenizer()
        
        # This call will build the vocabulary and save the binary cache
        # Note: We always use max_samples=None and sample_ratio=1.0 here to create the full cache
        # This allows training and evaluation to load from the full cache and subsample as needed
        dataset = StyleDataset.from_multiple_tsv(
            eval_files,
            tokenizer,
            max_samples=None,
            sample_ratio=1.0,
            grammaticality_labels=eval_labels,
            verbose=False,  # Suppress redundant distribution stats
            cache_dir=os.path.join(args.output_dir, "dataset_cache")
        )
        
        # Print Phase 3-specific statistics
        vocab_sizes = tokenizer.get_vocab_sizes()
        console.print(f"\n[bold cyan]Phase 3 Statistics:[/bold cyan]")
        console.print(f"  Encoded samples: [bold]{len(dataset)}[/bold]")
        console.print(f"  Vocabulary sizes:")
        console.print(f"    Surface forms: {vocab_sizes['surface']:,}")
        console.print(f"    Lemmas: {vocab_sizes['lemma']:,}")
        console.print(f"    POS tags: {vocab_sizes['pos']}")
        console.print(f"    Conjugation types: {vocab_sizes['conjugated_type']}")
        console.print(f"    Conjugation forms: {vocab_sizes['conjugated_form']}")
        console.print(f"  Binary cache: [cyan]{os.path.join(args.output_dir, 'dataset_cache')}[/cyan]")
        console.print(f"\n[bold green]Preprocessing Phase 3 complete.[/bold green]")

if __name__ == "__main__":
    main()
