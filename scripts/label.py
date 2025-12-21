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
import time
import random
import multiprocessing as mp
from collections import Counter
import json
from typing import Dict, List, Optional, Tuple, Any, cast




from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn

from scripts.cache import get_kotogram_cache
from scripts.style_data import ProcessedSample
from scripts.train_style import StyleDataset, CACHE_VERSION

from kotogram.model import FORMALITY_LABEL_TO_ID, FORMALITY_ID_TO_LABEL, REGISTER_LABEL_TO_ID, REGISTER_ID_TO_LABEL, FEATURE_FIELDS, Tokenizer
from kotogram.kotogram import split_kotogram, extract_token_features

# Global variable for worker processes only
_worker_overrides: Optional[Dict[str, List[Any]]] = None

DEFAULT_BATCH_SIZE = 1000

def _build_and_save_vocab(tokenizer: Tokenizer, merged_counters: Dict[str, Counter], cache_dir: str, cache_name: str) -> None:
    """Build vocabulary from counters and save to disk."""
    for field in FEATURE_FIELDS:
        counter = merged_counters.get(field, Counter())
        # Add values sorted by frequency (descending)
        for value, _ in counter.most_common():
             tokenizer._add_value(field, value)
    
    vocab_path = os.path.join(cache_dir, cache_name)
    tokenizer.save(vocab_path)

def load_register_overrides() -> Dict[str, List[Any]]:
    """Load manual register overrides from data/jpn_sentences_<register>.tsv."""
    from kotogram.analysis import RegisterLevel
    
    # Map register string to RegisterLevel
    reg_map = {r.value: r for r in RegisterLevel}
    
    overrides: Dict[str, Any] = {}
    
    # Pattern to match individual register files
    pattern = "data/jpn_sentences_*.tsv"
    for file_path in glob.glob(pattern):
        basename = os.path.basename(file_path)
            
        reg_str = basename.replace("jpn_sentences_", "").replace(".tsv", "")
        if reg_str not in reg_map:
            continue
            
        reg_level = reg_map[reg_str]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                sentence = parts[2]
                if sentence not in overrides:
                    overrides[sentence] = set()
                overrides[sentence].add(reg_level)

    # Convert sets to sorted lists
    return {k: sorted(list(v), key=lambda x: str(x)) for k, v in overrides.items()}

def init_worker(overrides: Dict[str, List[Any]]) -> None:
    """Initialize worker process with register overrides."""
    global _worker_overrides
    _worker_overrides = overrides

def get_file_fingerprint(path: str) -> Optional[Dict[str, Any]]:
    """Return mtime and size of a file for change detection."""
    if not path or not os.path.exists(path):
        return None
    stat = os.stat(path)
    return {'mtime': stat.st_mtime, 'size': stat.st_size}

def get_dependencies_fingerprint(args: Any) -> Dict[str, Any]:
    """Collect fingerprints of all input dependencies."""
    fingerprints = {}
    
    # Primary patterns
    for name, pattern in [('grammatic', args.grammatic_pattern), 
                         ('agrammatic', args.agrammatic_pattern)]:
        if not pattern:
            continue
        files = sorted(glob.glob(pattern))
        fingerprints[name] = {f: get_file_fingerprint(f) for f in files}
        
    # Register overrides
    override_files = sorted(glob.glob("data/jpn_sentences_*.tsv"))
    fingerprints['overrides'] = {f: get_file_fingerprint(f) for f in override_files}
    
    return fingerprints

console = Console()



def infer_gender_from_register(gender_enum: Any, register_enums: List[Any]) -> Tuple[float, int]:
    """Infer gender value and pragmatic flag from gender enum and registers.
    
    Refined logic:
    1. If gender is explicitly MASCULINE/FEMININE, use that.
    2. If gender is NEUTRAL, infer from registers:
       - Masculine registers: DANSEIGO, GUNTAI, BUSHI (Excluded KYOSHIGO)
       - Feminine registers: JOSEIGO, OJOUSAMA, BURIKKO
    3. If registers have both masculine and feminine markers, return UNPRAGMATIC (0.0, 0).
    4. Otherwise return NEUTRAL (0.0, 1) or the inferred gender.
    """
    from kotogram.analysis import GenderLevel, RegisterLevel
    
    if gender_enum == GenderLevel.MASCULINE:
        return -1.0, 1
    elif gender_enum == GenderLevel.FEMININE:
        return 1.0, 1
    elif gender_enum == GenderLevel.NEUTRAL:
        # Infer gender from register if neutral
        masculine_registers = {RegisterLevel.DANSEIGO, RegisterLevel.GUNTAI, RegisterLevel.BUSHI}
        feminine_registers = {RegisterLevel.JOSEIGO, RegisterLevel.OJOUSAMA, RegisterLevel.BURIKKO}
        
        is_masc = any(r in masculine_registers for r in register_enums)
        is_fem = any(r in feminine_registers for r in register_enums)
        
        if is_masc and is_fem:
            # Conflicting registers -> Unpragmatic
            return 0.0, 0
        elif is_masc:
            return -1.0, 1
        elif is_fem:
            return 1.0, 1
        else:
            return 0.0, 1
    else: # UNPRAGMATIC_GENDER
        return 0.0, 0

def _process_sentence_batch(batch: List[Tuple[str, str, int]]) -> Tuple[List[ProcessedSample], Dict[str, Counter]]:
    """Process a batch of sentences in a worker process."""
    from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
    from kotogram.analysis import FormalityLevel, RegisterLevel
    from kotogram.kotogram import split_kotogram, extract_token_features
    from kotogram.model import FEATURE_FIELDS
    
    from scripts.rule_based_analysis import analyze_formality, analyze_gender, analyze_register


    parser = SudachiJapaneseParser()
    results = []
    counters: Dict[str, Counter[Any]] = {f: Counter() for f in FEATURE_FIELDS}

    for sentence, sentence_id, gram_label in batch:
        kotogram = parser.japanese_to_kotogram(sentence)
        formality_enum = analyze_formality(kotogram)
        gender_enum = analyze_gender(kotogram)

        # Token collection for vocabulary
        tokens = split_kotogram(kotogram)
        for token in tokens:
            token_feat = extract_token_features(token)
            for field in FEATURE_FIELDS:
                value = getattr(token_feat, field)
                counters[field][value] += 1

        # Check for overrides
        overrides = _worker_overrides or {}
        if sentence in overrides:
            register_enums = overrides[sentence]
        else:
            register_enums = list(analyze_register(kotogram))
        
        formality_id = FORMALITY_LABEL_TO_ID.get(formality_enum, FORMALITY_LABEL_TO_ID[FormalityLevel.NEUTRAL])
        
        gender_val, gender_prag = infer_gender_from_register(gender_enum, register_enums)
        
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
    return results, counters

def _compute_labels_batch(batch: List[Tuple[str, str, int]]) -> Tuple[List[ProcessedSample], Dict[str, Counter]]:
    """Compute labels for a batch of sentences (where kotogram is already cached)."""
    from kotogram.analysis import FormalityLevel, RegisterLevel
    from kotogram.kotogram import split_kotogram, extract_token_features
    from kotogram.model import FEATURE_FIELDS
    
    from scripts.rule_based_analysis import analyze_formality, analyze_gender, analyze_register


    results = []
    counters: Dict[str, Counter[Any]] = {f: Counter() for f in FEATURE_FIELDS}
    
    for sentence, kotogram, gram_label in batch:
        formality_enum = analyze_formality(kotogram)
        gender_enum = analyze_gender(kotogram)
        
        # Token collection for vocabulary
        tokens = split_kotogram(kotogram)
        for token in tokens:
            token_feat = extract_token_features(token)
            for field in FEATURE_FIELDS:
                value = getattr(token_feat, field)
                counters[field][value] += 1

        register_enums = list(analyze_register(kotogram))
        
        formality_id = FORMALITY_LABEL_TO_ID.get(formality_enum, FORMALITY_LABEL_TO_ID[FormalityLevel.NEUTRAL])
        
        gender_val, gender_prag = infer_gender_from_register(gender_enum, register_enums)
        
        # Check for overrides
        overrides = _worker_overrides or {}
        if sentence in overrides:
            register_enums = overrides[sentence]
        else:
            register_enums = list(analyze_register(kotogram))

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
            
    return results, counters

def print_stats(results: List[ProcessedSample]) -> None:
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
    register_counts: Counter[int] = Counter()
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

def save_register_samples(results: List[ProcessedSample], model_dir: Optional[str]) -> None:
    """Save 3 examples of each register from grammatic sentences to CSV."""
    if not model_dir:
        return
    
    output_file = os.path.join(model_dir, "register_samples.csv")
    
    # Collect ALL samples by register (only grammatic sentences)
    all_by_register: Dict[int, List[ProcessedSample]] = {}
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
    os.makedirs(model_dir, exist_ok=True)
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
    
    console.print(f"\n[bold cyan]Saved register samples to:[/bold cyan] {output_file}")
    console.print(f"  Registers with samples: {len(register_samples)}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Label and cache Japanese sentences.")
    parser.add_argument("--grammatic-pattern", type=str, required=True, help="Primary TSV data file(s) (glob pattern)")
    parser.add_argument("--agrammatic-pattern", type=str, help="Agrammatic TSV pattern")
    parser.add_argument("--output-grammatic", type=str, required=True, help="Path to save combined/deduplicated grammatic data")
    parser.add_argument("--output-agrammatic", type=str, required=True, help="Path to save combined/deduplicated agrammatic data")
    parser.add_argument("--model-dir", type=str, help="Output directory for results (e.g. register samples)")
    parser.add_argument("--cache-dir", type=str, default=".cache", help="Output directory for dataset cache")
    parser.add_argument("--force-relabel", action="store_true", help="Force re-computation of labels even if cached")
    
    args = parser.parse_args()
    
    # Fast-skip check
    metadata_path = os.path.join(args.cache_dir, "label_metadata.json")
    current_fingerprints = get_dependencies_fingerprint(args)
    
    if os.path.exists(metadata_path) and not args.force_relabel:
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                
            vocab_path = os.path.join(args.cache_dir, saved_data.get('vocab_file', 'vocab.json'))
            if (saved_data.get('fingerprints') == current_fingerprints and 
                saved_data.get('cache_version') == CACHE_VERSION and
                os.path.exists(vocab_path)):
                console.print("[green]Using cached labels[/green]")
                return
        except Exception:
            pass # Fall back to processing
            
    num_workers = max(1, mp.cpu_count() - 1)
    
    def process_file_group(patterns: Any, gram_label: int, output_path: Optional[str] = None) -> Tuple[List[Any], int]:
        if not patterns:
            return [], 0
        
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
                    if len(row) < 3:
                        continue
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
    gram_patterns = [args.grammatic_pattern]
    
    console.print(f"Processing [bold]grammatic[/bold] data ({len(gram_patterns)} patterns) with {num_workers} workers...")
    rows, count = process_file_group(gram_patterns, 1, args.output_grammatic)
    all_rows.extend(rows)
    if count > 0:
        console.print(f"  Matched {count} grammatic files.")
    
    # Process agrammatic (agrammatic-pattern)
    agram_patterns = []
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
    
    merged_counters: Dict[str, Counter[Any]] = {f: Counter() for f in FEATURE_FIELDS}
    
    # Pre-load overrides in main process
    # Pre-load overrides in main process
    register_overrides = load_register_overrides()
    if register_overrides:
        console.print(f"Loaded [bold cyan]{len(register_overrides):,}[/bold cyan] register overrides.")
    
    for sentence, sentence_id, gram_label in all_rows:
        entry = cached_batch.get(sentence)
        k = entry[0] if entry else None
        
        # If sentence is in overrides, we MUST force re-labeling to ensure correct register labels
        if sentence in register_overrides:
            if k:
                unlabeled_rows.append((sentence, cast(str, k), gram_label))
            else:
                uncached_rows.append((sentence, sentence_id, gram_label))
            continue

        if entry:
            k, f_id, g_val, g_prag, r_lbls, _ = entry
            if not args.force_relabel and f_id is not None and g_val is not None and g_prag is not None and r_lbls is not None:
                final_results.append(ProcessedSample(
                    sentence=sentence,
                    sentence_id=sentence_id,
                    kotogram=cast(str, k),
                    formality_id=f_id,
                    gender_value=g_val,
                    gender_pragmatic=g_prag,
                    register_ids=r_lbls,
                    gram_label=gram_label,
                    success=1
                ))
                # Add to counters for vocabulary
                tokens = split_kotogram(cast(str, k))
                for token in tokens:
                    token_feat = extract_token_features(token)
                    for field in FEATURE_FIELDS:
                        value = getattr(token_feat, field)
                        merged_counters[field][value] += 1
            else:
                unlabeled_rows.append((sentence, cast(str, k), gram_label))
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
                batches = [uncached_rows[i:i + DEFAULT_BATCH_SIZE] for i in range(0, len(uncached_rows), DEFAULT_BATCH_SIZE)]
                
                new_entries = []
                with ctx.Pool(num_workers, initializer=init_worker, initargs=(register_overrides,)) as pool:
                    for batch_results, batch_counters in pool.imap(_process_sentence_batch, batches):
                        # Merge counters
                        for field, b_counter in batch_counters.items():
                            merged_counters[field].update(b_counter)
                            
                        for res in batch_results:
                            if res.success:
                                final_results.append(res)
                                new_entries.append((
                                    cast(str, res.sentence), 
                                    cast(str, res.kotogram), 
                                    cast(Optional[int], res.formality_id), 
                                    cast(Optional[float], res.gender_value), 
                                    cast(Optional[int], res.gender_pragmatic), 
                                    cast(Optional[List[int]], res.register_ids),
                                    cast(Optional[int], res.gram_label)
                                ))
                        progress.update(task, advance=len(batch_results))
                
                if new_entries:
                    cache.put_batch(new_entries)
                    
            if unlabeled_rows:
                task = progress.add_task("[cyan]Re-labeling...", total=len(unlabeled_rows))
                batches = [unlabeled_rows[i:i + DEFAULT_BATCH_SIZE] for i in range(0, len(unlabeled_rows), DEFAULT_BATCH_SIZE)]
                
                new_entries = []
                with ctx.Pool(num_workers, initializer=init_worker, initargs=(register_overrides,)) as pool:
                    for batch_results, batch_counters in pool.imap(_compute_labels_batch, batches):
                        # Merge counters
                        for field, b_counter in batch_counters.items():
                            merged_counters[field].update(b_counter)
                            
                        for res in batch_results:
                            if res.success:
                                final_results.append(res)
                                new_entries.append((
                                    cast(str, res.sentence), 
                                    cast(str, res.kotogram), 
                                    cast(Optional[int], res.formality_id), 
                                    cast(Optional[float], res.gender_value), 
                                    cast(Optional[int], res.gender_pragmatic), 
                                    cast(Optional[List[int]], res.register_ids),
                                    cast(Optional[int], res.gram_label)
                                ))
                        progress.update(task, advance=len(batch_results))
                
                if new_entries:
                    cache.put_batch(new_entries)

    console.print(f"\n[bold green]Processing complete![/bold green] Total processed: {len(final_results):,}")
    print_stats(final_results)
    
    # Save register samples to CSV
    save_register_samples(final_results, args.model_dir)

    vocab_file = "vocab.json"
    if args.output_grammatic:
        console.print("\n[bold blue]Finalizing dataset and building vocabulary...[/bold blue]")
        
        from kotogram.model import Tokenizer
        tokenizer = Tokenizer()
        
        # Build and save vocabulary explicitly
        _build_and_save_vocab(tokenizer, merged_counters, args.cache_dir, vocab_file)
        console.print(f"  Saved vocabulary to {os.path.join(args.cache_dir, vocab_file)}")

        dataset = StyleDataset.from_processed_samples(
            final_results,
            tokenizer,
            verbose=False,  # Suppress redundant distribution stats
            cache_dir=args.cache_dir,
            cache_name=vocab_file,
            sample_ratio=1.0,
        )
        
        # Print statistics
        vocab_sizes = tokenizer.get_vocab_sizes()
        console.print("\n[bold cyan]Dataset Statistics:[/bold cyan]")
        console.print(f"  Encoded samples: [bold]{len(dataset)}[/bold]")
        console.print("  Vocabulary sizes:")
        console.print(f"    Surface forms: {vocab_sizes['surface']:,}")
        console.print(f"    Lemmas: {vocab_sizes['lemma']:,}")
        console.print(f"    POS tags: {vocab_sizes['pos']}")
        console.print(f"    Conjugation types: {vocab_sizes['conjugated_type']}")
        console.print(f"    Conjugation forms: {vocab_sizes['conjugated_form']}")
        console.print(f"  Vocabulary cache: [cyan]{os.path.join(args.cache_dir, vocab_file)}[/cyan]")
        console.print("\n[bold green]Dataset finalization complete.[/bold green]")

    # Final: Save metadata for fast-skip
    output_fingerprints = {}
    if args.output_grammatic and os.path.exists(args.output_grammatic):
         output_fingerprints['grammatic'] = get_file_fingerprint(args.output_grammatic)
    if args.output_agrammatic and os.path.exists(args.output_agrammatic):
         output_fingerprints['agrammatic'] = get_file_fingerprint(args.output_agrammatic)

    metadata = {
        'timestamp': time.time(),
        'fingerprints': current_fingerprints, # Source fingerprints
        'output_fingerprints': output_fingerprints, # Output fingerprints (combined files)
        'cache_version': CACHE_VERSION,
        'vocab_file': vocab_file
    }
    os.makedirs(args.cache_dir, exist_ok=True)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()
