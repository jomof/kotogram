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
import random
import multiprocessing as mp
from collections import Counter
from typing import Dict, List, Optional, Tuple, Set, NamedTuple, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Lazy load rich components to avoid top-level dependency
# from rich.console import Console
# from rich.table import Table
# from rich.panel import Panel
# from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn

from scripts.cache import get_kotogram_cache
from kotogram.constants import FORMALITY_LABEL_TO_ID, FORMALITY_ID_TO_LABEL, REGISTER_LABEL_TO_ID, REGISTER_ID_TO_LABEL, NUM_REGISTER_CLASSES

# Global override map: sentence -> List[RegisterLevel]
_REGISTER_OVERRIDES: Dict[str, List[Any]] = {}

def load_register_overrides():
    """Load manual register overrides from data/jpn_sentences_<register>.tsv."""
    global _REGISTER_OVERRIDES
    from kotogram.analysis import RegisterLevel
    
    # Map register string to RegisterLevel
    reg_map = {r.value: r for r in RegisterLevel}
    
    overrides = {}
    
    # Pattern to match individual register files
    pattern = "data/jpn_sentences_*.tsv"
    for file_path in glob.glob(pattern):
        basename = os.path.basename(file_path)
            
        reg_str = basename.replace("jpn_sentences_", "").replace(".tsv", "")
        if reg_str not in reg_map:
            continue
            
        reg_level = reg_map[reg_str]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 3:
                        continue
                    from kotogram.preprocess import preprocess
                    sentence, _types = preprocess(parts[2])
                    if sentence not in overrides:
                        overrides[sentence] = set()
                    overrides[sentence].add(reg_level)
        except Exception as e:
            get_console().print(f"[yellow]Warning: Could not load override file {file_path}: {e}[/yellow]")

    # Convert sets to sorted lists
    _REGISTER_OVERRIDES = {k: sorted(list(v), key=lambda x: str(x)) for k, v in overrides.items()}

def init_worker():
    """Initialize worker process with register overrides."""
    load_register_overrides()

_CONSOLE = None
def get_console():
    """Lazy init console."""
    global _CONSOLE
    if _CONSOLE is None:
        from rich.console import Console
        _CONSOLE = Console()
    return _CONSOLE

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

def compute_labels(kotogram: str, sentence: str = "", register_overrides: Optional[List[Any]] = None) -> ProcessedSample:
    """Compute all labels for a given kotogram."""
    from kotogram.analysis import FormalityLevel, GenderLevel, RegisterLevel
    from scripts.rule_based_analysis import analyze_formality, analyze_gender, analyze_register
    
    try:
        formality_enum = analyze_formality(kotogram)
        gender_enum = analyze_gender(kotogram)

        # Check for overrides
        if register_overrides is not None:
             register_enums = register_overrides
        elif sentence in _REGISTER_OVERRIDES:
            register_enums = _REGISTER_OVERRIDES[sentence]
        else:
            register_enums = analyze_register(kotogram)
        
        formality_id = FORMALITY_LABEL_TO_ID.get(formality_enum, FORMALITY_LABEL_TO_ID[FormalityLevel.NEUTRAL])
        
        if gender_enum == GenderLevel.MASCULINE:
            gender_val, gender_prag = -1.0, 1
        elif gender_enum == GenderLevel.FEMININE:
            gender_val, gender_prag = 1.0, 1
        elif gender_enum == GenderLevel.NEUTRAL:
            gender_val, gender_prag = 0.0, 1
            
            # Infer gender from register if neutral
            masculine_registers = {RegisterLevel.DANSEIGO, RegisterLevel.GUNTAI, RegisterLevel.BUSHI, RegisterLevel.KYOSHIGO}
            feminine_registers = {RegisterLevel.JOSEIGO, RegisterLevel.OJOUSAMA, RegisterLevel.BURIKKO}
            
            is_masc = any(r in masculine_registers for r in register_enums)
            is_fem = any(r in feminine_registers for r in register_enums)
            
            if is_masc and is_fem:
                # Mixed gender markers = Unpragmatic
                gender_val, gender_prag = 0.0, 0
            elif is_masc:
                gender_val = -1.0
            elif is_fem:
                gender_val = 1.0
        else: # UNPRAGMATIC_GENDER
            gender_val, gender_prag = 0.0, 0
        

            
        register_ids = [REGISTER_LABEL_TO_ID[r] for r in register_enums if r in REGISTER_LABEL_TO_ID]
        if not register_ids:
            register_ids = [REGISTER_LABEL_TO_ID[RegisterLevel.NEUTRAL]]

        return ProcessedSample(
            sentence=sentence,
            sentence_id="",
            kotogram=kotogram,
            formality_id=formality_id,
            gender_value=gender_val,
            gender_pragmatic=gender_prag,
            register_ids=register_ids,
            gram_label=1, # Default, caller handles this
            success=1
        )
    except Exception:
         return ProcessedSample(sentence, "", kotogram, 0, 0.0, 0, [], 0, 0)

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
            sample = compute_labels(kotogram, sentence)
            
            # Update specific fields
            results.append(sample._replace(
                sentence_id=sentence_id,
                gram_label=gram_label
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
        # Pre-computed kotogram, but we can re-compute labels
        sample = compute_labels(kotogram, sentence)
         # Update specific fields
        results.append(sample._replace(
            gram_label=gram_label
        ))
            
    return results

def print_stats(results: List[ProcessedSample]):
    """Print attractive statistics about the labeling results."""
    if not results:
        return

    # Formality Stats
    from rich.table import Table
    from rich.panel import Panel
    
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

    get_console().print(Panel.fit(f_table, border_style="magenta"))
    get_console().print(Panel.fit(g_table, border_style="cyan"))
    get_console().print(Panel.fit(r_table, border_style="yellow"))
    get_console().print(Panel.fit(gram_table, border_style="green"))

def save_register_samples(results: List[ProcessedSample], output_grammatic_path: Optional[str]):
    """Save 3 examples of each register from grammatic sentences to CSV."""
    if not output_grammatic_path:
        return
    
    # Get output directory from the grammatic output path
    output_dir = os.path.dirname(output_grammatic_path)
    if not output_dir:
        output_dir = "models/style"
    else:
        # Replace .cache with models/style
        if ".cache" in output_dir:
            output_dir = "models/style"
    
    output_file = os.path.join(output_dir, "register_samples.csv")
    
    # Collect ALL samples by register (only grammatic sentences)
    all_by_register = {}
    for result in results:
        if not result.success or result.gram_label != 1:  # Only grammatic
            continue
        
        for reg_id in result.register_ids:
            if reg_id not in all_by_register:
                all_by_register[reg_id] = []
            all_by_register[reg_id].append(result)
            
    # Randomly sample 3 from each
    random.seed(int(time.time() * 1000)) # Precision seed
    register_samples = {}
    for reg_id, samples in all_by_register.items():
        if len(samples) <= 3:
            register_samples[reg_id] = samples
        else:
            register_samples[reg_id] = random.sample(samples, 3)
    
    # Write to CSV
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['register', 'register_id', 'sentence', 'formality', 'gender_value'])
        
        for reg_id in sorted(register_samples.keys()):
            register_name = REGISTER_ID_TO_LABEL[reg_id].value
            formality_map = {v: k.value for k, v in FORMALITY_LABEL_TO_ID.items()}
            
            for sample in register_samples[reg_id]:
                formality_name = formality_map.get(sample.formality_id, "unknown")
                writer.writerow([
                    register_name,
                    reg_id,
                    sample.sentence,
                    formality_name,
                    f"{sample.gender_value:.2f}"
                ])
    
    get_console().print(f"\n[bold cyan]Saved register samples to:[/bold cyan] {output_file}")
    get_console().print(f"  Registers with samples: {len(register_samples)}")


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
                    from kotogram.preprocess import preprocess
                    sentence, _types = preprocess(row[2])
                    if sentence not in seen:
                        seen.add(sentence)
                        unique_rows.append((sentence, row[0], gram_label))
                        raw_rows.append(row)
                    
                    if args.max_samples and len(unique_rows) >= args.max_samples:
                        break
                if args.max_samples and len(unique_rows) >= args.max_samples:
                    break
        
        # Apply percentage if requested
        if args.percent and args.percent < 100.0:
            count = int(len(unique_rows) * (args.percent / 100.0))
            if count < len(unique_rows):
                # Use fixed seed for consistency
                random.seed(42)
                indices = random.sample(range(len(unique_rows)), count)
                unique_rows = [unique_rows[i] for i in sorted(indices)]
                raw_rows = [raw_rows[i] for i in sorted(indices)]

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
                        get_console().print(f"  [dim]{os.path.basename(output_path)} unchanged, skipping write.[/dim]")
                        should_write = False
            
            if should_write:
                get_console().print(f"  Writing {len(raw_rows):,} unique rows to [bold]{output_path}[/bold]...")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                    
        return unique_rows, len(file_list)

    all_rows = []
    
    # Process grammatic (only primary data)
    gram_patterns = [args.data]
    
    get_console().print(f"Processing [bold]grammatic[/bold] data ({len(gram_patterns)} patterns) with {num_workers} workers...")
    rows, count = process_file_group(gram_patterns, 1, args.output_grammatic)
    all_rows.extend(rows)
    if count > 0:
        get_console().print(f"  Matched {count} grammatic files.")
    
    # Process agrammatic (agrammatic-sentences + agrammatic-pattern)
    agram_patterns = []
    if args.agrammatic_sentences:
        agram_patterns.append(args.agrammatic_sentences)
    if args.agrammatic_pattern:
        agram_patterns.append(args.agrammatic_pattern)
        
    if agram_patterns:
        get_console().print(f"Processing [bold]agrammatic[/bold] data ({len(agram_patterns)} patterns)...")
        rows, count = process_file_group(agram_patterns, 0, args.output_agrammatic)
        all_rows.extend(rows)
        if count > 0:
            get_console().print(f"  Matched {count} agrammatic files.")

    if not all_rows:
        get_console().print("[red]No data sentences found. Check your patterns.[/red]")
        sys.exit(1)
        
    get_console().print(f"Total unique sentences to check: [bold]{len(all_rows):,}[/bold]")
    
    cache = get_kotogram_cache()
    cached_batch = cache.get_batch([r[0] for r in all_rows])
    
    uncached_rows = []
    unlabeled_rows = []
    final_results = []
    
    # Pre-load overrides in main process
    load_register_overrides()
    if _REGISTER_OVERRIDES:
        get_console().print(f"Loaded [bold cyan]{len(_REGISTER_OVERRIDES):,}[/bold cyan] register overrides.")
    
    for sentence, sentence_id, gram_label in all_rows:
        entry = cached_batch.get(sentence)
        k = entry[0] if entry else None
        
        # If sentence is in overrides, we MUST force re-labeling to ensure correct register labels
        if sentence in _REGISTER_OVERRIDES:
            if k:
                unlabeled_rows.append((sentence, k, gram_label))
            else:
                uncached_rows.append((sentence, sentence_id, gram_label))
            continue

        if entry:
            k, f, g_val, g_prag, r_lbls = entry
            if f is not None and g_val is not None and g_prag is not None and r_lbls is not None:
                final_results.append(ProcessedSample(sentence, sentence_id, k, f, g_val, g_prag, r_lbls, gram_label, 1))
            else:
                unlabeled_rows.append((sentence, k, gram_label))
        else:
            uncached_rows.append((sentence, sentence_id, gram_label))
            
    get_console().print(f"Cache status: {len(final_results):,} hits, {len(unlabeled_rows):,} partial, {len(uncached_rows):,} misses")
    
    ctx = mp.get_context('spawn')
    
    if uncached_rows or unlabeled_rows:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            console=get_console()
        ) as progress:
            
            if uncached_rows:
                task = progress.add_task("[green]Parsing & Labeling...", total=len(uncached_rows))
                batches = [uncached_rows[i:i + args.batch_size] for i in range(0, len(uncached_rows), args.batch_size)]
                
                new_entries = []
                with ctx.Pool(num_workers, initializer=init_worker) as pool:
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
                with ctx.Pool(num_workers, initializer=init_worker) as pool:
                    for batch_results in pool.imap(_compute_labels_batch, batches):
                        for res in batch_results:
                            if res.success:
                                final_results.append(res)
                                new_entries.append((res.sentence, res.kotogram, res.formality_id, res.gender_value, res.gender_pragmatic, res.register_ids))
                        progress.update(task, advance=len(batch_results))
                
                if new_entries:
                    cache.put_batch(new_entries)

    get_console().print(f"\n[bold green]Processing complete![/bold green] Total processed: {len(final_results):,}")
    print_stats(final_results)
    
    # Save register samples to CSV
    save_register_samples(final_results, args.output_grammatic)

    # Phase 3: Build vocabulary and encode samples (Warming the StyleDataset cache)
    # Only run if we have output paths (meaning we're running in preprocessing mode)
    if args.output_grammatic:
        from scripts.train_style import StyleDataset, Tokenizer
        
        get_console().print("\n[bold blue]Phase 3: Building vocabulary and encoding samples...[/bold blue]")
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
        get_console().print(f"\n[bold cyan]Phase 3 Statistics:[/bold cyan]")
        get_console().print(f"  Encoded samples: [bold]{len(dataset)}[/bold]")
        get_console().print(f"  Vocabulary sizes:")
        get_console().print(f"    Surface forms: {vocab_sizes['surface']:,}")
        get_console().print(f"    Lemmas: {vocab_sizes['lemma']:,}")
        get_console().print(f"    POS tags: {vocab_sizes['pos']}")
        get_console().print(f"    Conjugation types: {vocab_sizes['conjugated_type']}")
        get_console().print(f"    Conjugation forms: {vocab_sizes['conjugated_form']}")
        get_console().print(f"  Binary cache: [cyan]{os.path.join(args.output_dir, 'dataset_cache')}[/cyan]")
        get_console().print(f"\n[bold green]Preprocessing Phase 3 complete.[/bold green]")

if __name__ == "__main__":
    main()
