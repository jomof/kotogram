#!/usr/bin/env python3
"""Standalone script to label and cache Japanese sentences for style classification."""

import array
import csv
import glob
import json
import multiprocessing as mp
import os

# pylint: disable=wrong-import-position, too-many-lines
import random
import sys
import time
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional, Tuple, cast

import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

# pylint: disable=wrong-import-position
from kotogram import locations
from kotogram.kotogram import extract_token_features, split_kotogram
from kotogram.model import (
    FORMALITY_ID_TO_LABEL,
    FORMALITY_LABEL_TO_ID,
    REGISTER_ID_TO_LABEL,
    REGISTER_LABEL_TO_ID,
)
from kotogram.tokenizer import FEATURE_FIELDS, Tokenizer
from train.cache import get_kotogram_cache
from train.dataset import CACHE_VERSION, parse_tsv
from train.profile import PhaseTimer, get_profile_dir
from train.types import ProcessedSample
from train.worker import encode_batch_fast
from train.worker import init_worker as encode_init_worker

# Global variable for worker processes only
_WORKER_OVERRIDES: Optional[Dict[str, List[Any]]] = None


DEFAULT_BATCH_SIZE = 1000


def _build_and_save_vocab(
    tokenizer: Tokenizer,
    merged_counters: Dict[str, Counter],
    cache_dir: str,
    cache_name: str,
) -> None:
    """Build vocabulary from counters and save to disk."""
    for field in FEATURE_FIELDS:
        counter = merged_counters.get(field, Counter())
        # Add values sorted by frequency (descending)
        for value, _ in counter.most_common():
            # pylint: disable=protected-access
            tokenizer._add_value(field, value)

    os.makedirs(cache_dir, exist_ok=True)
    vocab_path = os.path.join(cache_dir, cache_name)
    tokenizer.save(vocab_path, version=CACHE_VERSION)


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

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                sentence = parse_tsv(line)
                if sentence not in overrides:
                    overrides[sentence] = set()
                overrides[sentence].add(reg_level)

    # Convert sets to sorted lists
    return {k: sorted(list(v), key=str) for k, v in overrides.items()}


def init_worker(overrides: Dict[str, List[Any]]) -> None:
    """Initialize worker process with register overrides."""
    # pylint: disable=global-statement
    global _WORKER_OVERRIDES
    _WORKER_OVERRIDES = overrides


def get_file_fingerprint(path: str) -> Optional[Dict[str, Any]]:
    """Return mtime and size of a file for change detection."""
    if not path or not os.path.exists(path):
        return None
    stat = os.stat(path)
    return {"mtime": stat.st_mtime, "size": stat.st_size}


def get_dependencies_fingerprint(args: Any) -> Dict[str, Any]:
    """Collect fingerprints of all input dependencies."""
    fingerprints = {}

    def normalize_path(p: str) -> str:
        """Remove leading './' for consistent cache keys."""
        return p[2:] if p.startswith("./") else p

    # Primary patterns
    for name, pattern in [
        ("grammatic", args.grammatic_pattern),
        ("agrammatic", args.agrammatic_pattern),
    ]:
        if not pattern:
            continue
        files = sorted(glob.glob(pattern))
        fingerprints[name] = {normalize_path(f): get_file_fingerprint(f) for f in files}

    # Register overrides
    override_files = sorted(glob.glob("data/jpn_sentences_*.tsv"))
    fingerprints["overrides"] = {
        normalize_path(f): get_file_fingerprint(f) for f in override_files
    }

    return fingerprints


console = Console()


def infer_gender_from_register(
    gender_enum: Any, register_enums: List[Any]
) -> Tuple[float, int]:
    # pylint: disable=too-many-return-statements
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
    if gender_enum == GenderLevel.FEMININE:
        return 1.0, 1
    if gender_enum == GenderLevel.NEUTRAL:
        # Infer gender from register if neutral
        masculine_registers = {
            RegisterLevel.DANSEIGO,
            RegisterLevel.GUNTAI,
            RegisterLevel.BUSHI,
        }
        feminine_registers = {
            RegisterLevel.JOSEIGO,
            RegisterLevel.OJOUSAMA,
            RegisterLevel.BURIKKO,
        }

        is_masc = any(r in masculine_registers for r in register_enums)
        is_fem = any(r in feminine_registers for r in register_enums)

        if is_masc and is_fem:
            # Conflicting registers -> Unpragmatic
            return 0.0, 0
        if is_masc:
            return -1.0, 1
        if is_fem:
            return 1.0, 1

        return 0.0, 1

    return 0.0, 0


def _process_sentence_batch(
    batch: List[Tuple[str, int]],
) -> Tuple[Dict[str, Any], Dict[str, Counter]]:
    # pylint: disable=too-many-locals
    """Process a batch of sentences in a worker process."""
    # pylint: disable=redefined-outer-name, reimported
    from kotogram.analysis import FormalityLevel, RegisterLevel
    from kotogram.kotogram import extract_token_features, split_kotogram
    from kotogram.model import FEATURE_FIELDS
    from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
    from scripts.rule_based_analysis import (
        analyze_formality,
        analyze_gender,
        analyze_register,
    )

    parser = SudachiJapaneseParser()
    results: Dict[str, Any] = {
        "sentences": [],
        "kotograms": [],
        "f_ids": [],
        "g_vals": [],
        "g_prags": [],
        "gram_labels": [],
        "reg_ids_flat": [],
        "reg_ids_lens": [],
    }
    counters: Dict[str, Counter[Any]] = {f: Counter() for f in FEATURE_FIELDS}

    for sentence, gram_label in batch:
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
        overrides = _WORKER_OVERRIDES or {}
        if sentence in overrides:
            register_enums = overrides[sentence]
        else:
            register_enums = list(analyze_register(kotogram))

        formality_id = FORMALITY_LABEL_TO_ID.get(
            formality_enum, FORMALITY_LABEL_TO_ID[FormalityLevel.NEUTRAL]
        )

        gender_val, gender_prag = infer_gender_from_register(
            gender_enum, register_enums
        )

        register_ids = [
            REGISTER_LABEL_TO_ID[r] for r in register_enums if r in REGISTER_LABEL_TO_ID
        ]
        if not register_ids:
            register_ids = [REGISTER_LABEL_TO_ID[RegisterLevel.NEUTRAL]]

        # Flattened accumulation
        results["sentences"].append(sentence)
        results["kotograms"].append(kotogram)
        results["f_ids"].append(formality_id)
        results["g_vals"].append(gender_val)
        results["g_prags"].append(gender_prag)
        results["gram_labels"].append(gram_label)

        # Flattened lists
        results["reg_ids_flat"].extend(register_ids)
        results["reg_ids_lens"].append(len(register_ids))

    return results, counters


def _compute_labels_batch(
    batch: List[Tuple[str, str, int]],
) -> Tuple[Dict[str, Any], Dict[str, Counter]]:
    # pylint: disable=too-many-locals
    """Compute labels for a batch of sentences (where kotogram is already cached)."""
    from kotogram.analysis import FormalityLevel, RegisterLevel
    from scripts.rule_based_analysis import (
        analyze_formality,
        analyze_gender,
        analyze_register,
    )

    results: Dict[str, Any] = {
        "sentences": [],
        "kotograms": [],
        "f_ids": [],
        "g_vals": [],
        "g_prags": [],
        "gram_labels": [],
        "reg_ids_flat": [],
        "reg_ids_lens": [],
    }
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

        formality_id = FORMALITY_LABEL_TO_ID.get(
            formality_enum, FORMALITY_LABEL_TO_ID[FormalityLevel.NEUTRAL]
        )

        gender_val, gender_prag = infer_gender_from_register(
            gender_enum, register_enums
        )

        # Check for overrides
        overrides = _WORKER_OVERRIDES or {}
        if sentence in overrides:
            register_enums = overrides[sentence]
        else:
            register_enums = list(analyze_register(kotogram))

        register_ids = [
            REGISTER_LABEL_TO_ID[r] for r in register_enums if r in REGISTER_LABEL_TO_ID
        ]
        if not register_ids:
            register_ids = [REGISTER_LABEL_TO_ID[RegisterLevel.NEUTRAL]]

        # Flattened accumulation
        results["sentences"].append(sentence)
        results["kotograms"].append(kotogram)
        results["f_ids"].append(formality_id)
        results["g_vals"].append(gender_val)
        results["g_prags"].append(gender_prag)
        results["gram_labels"].append(gram_label)

        # Flattened lists
        results["reg_ids_flat"].extend(register_ids)
        results["reg_ids_lens"].append(len(register_ids))

    return results, counters


def print_stats(results: List[ProcessedSample]) -> None:
    # pylint: disable=too-many-locals
    """Print attractive statistics about the labeling results."""
    if not results:
        return

    def _print_dist(
        title: str,
        style: str,
        counts: Counter,
        map_func: Optional[Any] = None,
    ) -> None:
        table = Table(title=title, show_header=True, header_style=style)
        table.add_column("Type", style="dim")
        table.add_column("Count", justify="right")
        table.add_column("Percentage", justify="right")
        total = sum(counts.values()) or 1
        for kid in sorted(counts.keys()):
            label = map_func(kid) if map_func else str(kid)
            count = counts[kid]
            table.add_row(label, f"{count:,}", f"{100 * count / total:.1f}%")
        console.print(table)

    # Formality
    _print_dist(
        "Formality Distribution",
        "bold magenta",
        Counter(r.formality_id for r in results if r.success),
        lambda x: FORMALITY_ID_TO_LABEL[x].value,
    )

    # Gender
    _print_dist(
        "Gender Pragmatic Distribution",
        "bold cyan",
        Counter(r.gender_pragmatic for r in results if r.success),
        lambda x: {1: "Pragmatic", 0: "Unpragmatic"}[x],
    )

    # Register
    reg_counts: Counter[int] = Counter()
    for r in results:
        if r.success:
            reg_counts.update(r.register_ids)
    _print_dist(
        "Register Distribution",
        "bold yellow",
        reg_counts,
        lambda x: REGISTER_ID_TO_LABEL[x].value,
    )

    # Grammaticality
    _print_dist(
        "Grammaticality Distribution",
        "bold green",
        Counter(r.gram_label for r in results if r.success),
        lambda x: {1: "Grammatic", 0: "Agrammatic"}[x],
    )


def save_register_samples(results: List[ProcessedSample]) -> None:
    """Save 3 examples of each register from grammatic sentences to CSV."""
    output_dir = locations.get_cache_dir()
    output_file = os.path.join(output_dir, "register_samples.csv")

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
    random.seed(int(time.time() * 1000))  # Precision seed
    register_samples = {}
    for reg_id, samples in all_by_register.items():
        if len(samples) <= 3:
            register_samples[reg_id] = samples
        else:
            register_samples[reg_id] = random.sample(samples, 3)

    # Write to CSV
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["register", "register_id", "sentence", "formality", "gender_value"]
        )

        for reg_id in sorted(register_samples.keys()):
            register_name = REGISTER_ID_TO_LABEL[reg_id].value
            formality_map = {v: k.value for k, v in FORMALITY_LABEL_TO_ID.items()}

            for sample in register_samples[reg_id]:
                formality_name = formality_map.get(sample.formality_id, "unknown")
                writer.writerow(
                    [
                        register_name,
                        reg_id,
                        sample.sentence,
                        formality_name,
                        f"{sample.gender_value:.2f}",
                    ]
                )

    console.print(f"\n[bold cyan]Saved register samples to:[/bold cyan] {output_file}")
    console.print(f"  Registers with samples: {len(register_samples)}")


def main() -> None:
    # pylint: disable=too-many-locals
    import argparse

    parser = argparse.ArgumentParser(description="Label and cache Japanese sentences.")
    parser.add_argument(
        "--grammatic-pattern",
        type=str,
        required=True,
        help="Primary TSV data file(s) (glob pattern)",
    )
    parser.add_argument("--agrammatic-pattern", type=str, help="Agrammatic TSV pattern")
    # parser.add_argument("--cache-dir", type=str, default=".cache", help="Base directory for cache") # Removed
    parser.add_argument(
        "--force-relabel",
        action="store_true",
        help="Force re-computation of labels even if cached",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    args = parser.parse_args()

    args = parser.parse_args()

    # Resolve and inject paths from locations.py into args namespace
    cache_dir = locations.get_cache_dir()

    profile_dir = get_profile_dir()
    if profile_dir:
        os.makedirs(profile_dir, exist_ok=True)
    timer = PhaseTimer(console, profile_dir)

    args.output_grammatic = os.path.join(cache_dir, "grammatic_combined.tsv")
    args.output_agrammatic = os.path.join(cache_dir, "agrammatic_combined.tsv")
    args.model_dir = locations.get_style_output_dir()
    args.support_dir = locations.get_style_support_dir()

    # Fast-skip check
    dataset_cache_dir = locations.get_style_dataset_cache_dir()
    metadata_path = os.path.join(dataset_cache_dir, "label_metadata.json")
    current_fingerprints = get_dependencies_fingerprint(args)

    if os.path.exists(metadata_path) and not args.force_relabel:
        with open(metadata_path, "r", encoding="utf-8") as meta_file:
            saved_data = json.load(meta_file)

        vocab_path = os.path.join(
            dataset_cache_dir, saved_data.get("vocab_file", "vocab.json")
        )
        if (
            saved_data.get("fingerprints") == current_fingerprints
            and saved_data.get("cache_version") == CACHE_VERSION
            and os.path.exists(vocab_path)
        ):
            console.print("[green]Using cached labels[/green]")
            return

    num_workers = max(1, mp.cpu_count() - 1)

    def process_file_group(patterns: Any, gram_label: int) -> Tuple[List[Any], int]:
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

        unique_rows = []  # (sentence, gram_label)
        seen = set()

        for f_path in sorted(file_list):
            with open(f_path, "r", encoding="utf-8") as file_handle:
                for line in file_handle:
                    sentence = parse_tsv(line)

                    if sentence not in seen:
                        seen.add(sentence)
                        unique_rows.append((sentence, gram_label))

        # Writing is now handled in main() after filtering
        #
        # if output_path:
        #     ... (removed)

        return unique_rows, len(file_list)

    all_rows = []

    # Process grammatic (only primary data)
    gram_patterns = [args.grammatic_pattern]

    console.print(
        f"Processing [bold]grammatic[/bold] data ({len(gram_patterns)} patterns) with {num_workers} workers..."
    )
    rows, count = process_file_group(gram_patterns, 1)
    all_rows.extend(rows)
    if count > 0:
        console.print(f"  Matched {count} grammatic files.")

    # Process agrammatic (agrammatic-pattern)
    agram_patterns = []
    if args.agrammatic_pattern:
        agram_patterns.append(args.agrammatic_pattern)

    if agram_patterns:
        console.print(
            f"Processing [bold]agrammatic[/bold] data ({len(agram_patterns)} patterns)..."
        )
        rows, count = process_file_group(agram_patterns, 0)
        all_rows.extend(rows)
        if count > 0:
            console.print(f"  Matched {count} agrammatic files.")

    if not all_rows:
        console.print("[red]No data sentences found. Check your patterns.[/red]")
        sys.exit(1)

    # Global Deduplication of all_rows (preserving order)
    # This handles potential overlap between grammatic/agrammatic files or duplicates across files.
    # We keep the FIRST occurrence (prioritizing Grammatic if loading order is Gram -> Agram).
    seen_global = set()
    unique_all_rows = []
    for r in all_rows:
        if r[0] not in seen_global:
            seen_global.add(r[0])
            unique_all_rows.append(r)
    all_rows = unique_all_rows

    console.print(f"Total unique sentences to check: [bold]{len(all_rows):,}[/bold]")
    timer.mark("Scanning Input")

    cache = get_kotogram_cache()
    cached_batch = cache.get_batch([r[0] for r in all_rows])

    uncached_rows = []
    unlabeled_rows = []
    final_results: List[ProcessedSample] = []

    merged_counters: Dict[str, Counter[Any]] = {f: Counter() for f in FEATURE_FIELDS}

    # Pre-load overrides in main process
    # Pre-load overrides in main process
    register_overrides = load_register_overrides()
    if register_overrides:
        console.print(
            f"Loaded [bold cyan]{len(register_overrides):,}[/bold cyan] register overrides."
        )

    for sentence, gram_label in all_rows:
        entry = cached_batch.get(sentence)
        k = entry[0] if entry else None

        # If sentence is in overrides, we MUST force re-labeling to ensure correct register labels
        if sentence in register_overrides:
            if k:
                unlabeled_rows.append((sentence, cast(str, k), gram_label))
            else:
                uncached_rows.append((sentence, gram_label))
            continue

        if entry:
            k, f_id, g_val, g_prag, r_lbls, _, f_ids = entry
            if (
                not args.force_relabel
                and f_id is not None
                and g_val is not None
                and g_prag is not None
                and r_lbls is not None
            ):
                final_results.append(
                    ProcessedSample(
                        sentence=sentence,
                        kotogram=cast(str, k),
                        formality_id=f_id,
                        gender_value=g_val,
                        gender_pragmatic=g_prag,
                        register_ids=r_lbls,
                        gram_label=gram_label,
                        success=1,
                        feature_ids=f_ids,
                    )
                )
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
            uncached_rows.append((sentence, gram_label))

    console.print(
        f"Cache status: {len(final_results):,} hits, {len(unlabeled_rows):,} partial, {len(uncached_rows):,} misses"
    )

    total_tasks = len(uncached_rows) + len(unlabeled_rows)
    # Optimization: For small datasets, run sequentially in main process to avoid
    # multiprocessing spawn overhead (which can be seconds on macOS).
    # Threshold determined empirically (profiling small tests vs large runs).
    small_dataset_threshold = 500

    if 0 < total_tasks < small_dataset_threshold:
        console.print(
            f"[yellow]Small dataset ({total_tasks} < {small_dataset_threshold}), running sequentially...[/yellow]"
        )
        # Initialize worker global state in main process
        init_worker(register_overrides)

        if uncached_rows:
            new_entries = []
            batches = [
                uncached_rows[i : i + DEFAULT_BATCH_SIZE]
                for i in range(0, len(uncached_rows), DEFAULT_BATCH_SIZE)
            ]
            for batch in batches:
                batch_results, batch_counters = _process_sentence_batch(batch)
                # Merge counters
                for field, b_counter in batch_counters.items():
                    merged_counters[field].update(b_counter)

                # Reconstruct ProcessedSample from columns
                cnt = len(batch_results["sentences"])
                reg_offset = 0
                for idx in range(cnt):
                    r_len = batch_results["reg_ids_lens"][idx]
                    r_ids = batch_results["reg_ids_flat"][
                        reg_offset : reg_offset + r_len
                    ]
                    reg_offset += r_len

                    sample = ProcessedSample(
                        sentence=batch_results["sentences"][idx],
                        kotogram=batch_results["kotograms"][idx],
                        formality_id=batch_results["f_ids"][idx],
                        gender_value=batch_results["g_vals"][idx],
                        gender_pragmatic=batch_results["g_prags"][idx],
                        register_ids=r_ids,
                        gram_label=batch_results["gram_labels"][idx],
                        success=1,
                        feature_ids=None,
                    )
                    final_results.append(sample)
                    new_entries.append(
                        (
                            sample.sentence,
                            sample.kotogram,
                            sample.formality_id,
                            sample.gender_value,
                            sample.gender_pragmatic,
                            sample.register_ids,
                            sample.gram_label,
                            None,
                        )
                    )
            if new_entries:
                from train.cache import CacheEntryType

                cache.put_batch(cast(List[CacheEntryType], new_entries))

        if unlabeled_rows:
            batches_unlabeled = [
                unlabeled_rows[i : i + DEFAULT_BATCH_SIZE]
                for i in range(0, len(unlabeled_rows), DEFAULT_BATCH_SIZE)
            ]
            for batch_rows in batches_unlabeled:
                batch_results, batch_counters = _compute_labels_batch(batch_rows)
                # Merge counters
                for field, b_counter in batch_counters.items():
                    merged_counters[field].update(b_counter)

                # Reconstruct ProcessedSample from columns
                cnt = len(batch_results["sentences"])
                reg_offset = 0
                for idx in range(cnt):
                    r_len = batch_results["reg_ids_lens"][idx]
                    r_ids = batch_results["reg_ids_flat"][
                        reg_offset : reg_offset + r_len
                    ]
                    reg_offset += r_len

                    sample = ProcessedSample(
                        sentence=batch_results["sentences"][idx],
                        kotogram=batch_results["kotograms"][idx],
                        formality_id=batch_results["f_ids"][idx],
                        gender_value=batch_results["g_vals"][idx],
                        gender_pragmatic=batch_results["g_prags"][idx],
                        register_ids=r_ids,
                        gram_label=batch_results["gram_labels"][idx],
                        success=1,
                        feature_ids=None,
                    )
                    final_results.append(sample)

    else:
        ctx = mp.get_context("spawn")

        if uncached_rows or unlabeled_rows:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                console=console,
            ) as progress:
                if uncached_rows:
                    task = progress.add_task(
                        "[green]Parsing & Labeling...", total=len(uncached_rows)
                    )
                    batches = [
                        uncached_rows[i : i + DEFAULT_BATCH_SIZE]
                        for i in range(0, len(uncached_rows), DEFAULT_BATCH_SIZE)
                    ]

                    new_entries = []
                    with ctx.Pool(
                        num_workers,
                        initializer=init_worker,
                        initargs=(register_overrides,),
                    ) as pool:
                        for batch_results, batch_counters in pool.imap(
                            _process_sentence_batch, batches
                        ):
                            # Merge counters
                            for field, b_counter in batch_counters.items():
                                merged_counters[field].update(b_counter)

                            # Reconstruct ProcessedSample from columns
                            cnt = len(batch_results["sentences"])
                            reg_offset = 0
                            for idx in range(cnt):
                                r_len = batch_results["reg_ids_lens"][idx]
                                r_ids = batch_results["reg_ids_flat"][
                                    reg_offset : reg_offset + r_len
                                ]
                                reg_offset += r_len

                                sample = ProcessedSample(
                                    sentence=batch_results["sentences"][idx],
                                    kotogram=batch_results["kotograms"][idx],
                                    formality_id=batch_results["f_ids"][idx],
                                    gender_value=batch_results["g_vals"][idx],
                                    gender_pragmatic=batch_results["g_prags"][idx],
                                    register_ids=r_ids,
                                    gram_label=batch_results["gram_labels"][idx],
                                    success=1,
                                    feature_ids=None,
                                )
                                final_results.append(sample)
                                new_entries.append(
                                    (
                                        sample.sentence,
                                        sample.kotogram,
                                        sample.formality_id,
                                        sample.gender_value,
                                        sample.gender_pragmatic,
                                        sample.register_ids,
                                        sample.gram_label,
                                        None,
                                    )
                                )
                            progress.update(task, advance=cnt)

                    if new_entries:
                        from train.cache import CacheEntryType

                        cache.put_batch(cast(List[CacheEntryType], new_entries))

                if unlabeled_rows:
                    task = progress.add_task(
                        "[cyan]Re-labeling...", total=len(unlabeled_rows)
                    )
                    batches_unlabeled = [
                        unlabeled_rows[i : i + DEFAULT_BATCH_SIZE]
                        for i in range(0, len(unlabeled_rows), DEFAULT_BATCH_SIZE)
                    ]

                    new_entries = []
                    with ctx.Pool(
                        num_workers,
                        initializer=init_worker,
                        initargs=(register_overrides,),
                    ) as pool:
                        for batch_results, batch_counters in pool.imap(
                            _compute_labels_batch, batches_unlabeled
                        ):
                            # Merge counters
                            for field, b_counter in batch_counters.items():
                                merged_counters[field].update(b_counter)

                            # Reconstruct ProcessedSample from columns
                            cnt = len(batch_results["sentences"])
                            reg_offset = 0
                            for idx in range(cnt):
                                r_len = batch_results["reg_ids_lens"][idx]
                                r_ids = batch_results["reg_ids_flat"][
                                    reg_offset : reg_offset + r_len
                                ]
                                reg_offset += r_len

                                # Optimization: Create ProcessedSample only if needed?
                                # But final_results stores them.
                                sample = ProcessedSample(
                                    sentence=batch_results["sentences"][idx],
                                    kotogram=batch_results["kotograms"][idx],
                                    formality_id=batch_results["f_ids"][idx],
                                    gender_value=batch_results["g_vals"][idx],
                                    gender_pragmatic=batch_results["g_prags"][idx],
                                    register_ids=r_ids,
                                    gram_label=batch_results["gram_labels"][idx],
                                    success=1,
                                    feature_ids=None,
                                )
                                final_results.append(sample)
                            progress.update(task, advance=cnt)

    # Restore deterministic order based on all_rows (Input Globs)
    # This prevents cache-hit permutation from breaking StyleDataset split logic.
    if final_results:
        result_map = {r.sentence: r for r in final_results}
        final_results = []
        for sentence, _ in all_rows:
            if sentence in result_map:
                final_results.append(result_map[sentence])
            # Failures are silently dropped here (already logged or skipped)

    console.print(
        f"\n[bold green]Processing complete![/bold green] Total processed: {len(final_results):,}"
    )
    timer.mark("Parsing & Labeling (Phase 1)")
    display_results = [r for r in final_results if r.success]
    print_stats(display_results)

    # Write filtered, valid sentences to output files (single column)
    # This ensures consistency: The file on disk ONLY contains sentences that are
    # guaranteed to be in the cache.

    # 1. Grammatic
    if args.output_grammatic:
        gram_sent = [r.sentence for r in display_results if r.gram_label == 1]
        if gram_sent:
            console.print(
                f"Writing {len(gram_sent):,} grammatic sentences to [bold]{args.output_grammatic}[/bold]..."
            )
            os.makedirs(os.path.dirname(args.output_grammatic), exist_ok=True)
            with open(args.output_grammatic, "w", encoding="utf-8") as f:
                for s in gram_sent:
                    f.write(s + "\n")

    # 2. Agrammatic
    if args.output_agrammatic:
        agram_sent = [r.sentence for r in display_results if r.gram_label == 0]
        if agram_sent:
            console.print(
                f"Writing {len(agram_sent):,} agrammatic sentences to [bold]{args.output_agrammatic}[/bold]..."
            )
            os.makedirs(os.path.dirname(args.output_agrammatic), exist_ok=True)
            with open(args.output_agrammatic, "w", encoding="utf-8") as f:
                for s in agram_sent:
                    f.write(s + "\n")

    # Save register samples to CSV
    save_register_samples(final_results)

    vocab_file = "vocab.json"
    if args.output_grammatic:
        console.print(
            "\n[bold blue]Finalizing dataset and building vocabulary...[/bold blue]"
        )

        # Tokenizer is already imported at top level
        tokenizer = Tokenizer()

        # Build and save vocabulary explicitly
        _build_and_save_vocab(tokenizer, merged_counters, dataset_cache_dir, vocab_file)
        console.print(
            f"  Saved vocabulary to {os.path.join(dataset_cache_dir, vocab_file)}"
        )

        # Prepare tokenizer state for workers

        # Accumulators for Streaming Construction
        all_encodings: Dict[str, array.array] = {
            f: array.array("I") for f in FEATURE_FIELDS
        }
        all_offsets = array.array("I", [0])
        current_offset = 0
        f_vals = array.array("f")
        f_prags = array.array("B")
        g_vals = array.array("f")
        g_prags = array.array("B")
        gram_labels = array.array("B")
        all_reg_ids = array.array("B")
        reg_offsets = array.array("I", [0])
        cur_reg_offset = 0
        kc_collections: Dict[str, Dict[str, Any]] = {}

        # Sort results to ensure determinism before shuffling
        # This is critical for consistent vocabulary generation across runs
        final_results.sort(key=lambda x: x.sentence)

        # Shuffle results to ensure random sampling for down-stream tasks that use contiguous slicing
        random.seed(42)
        random.shuffle(final_results)

        tokenizer_state = {"field_vocabs": tokenizer.field_vocabs}

        timer.mark("Dataset Construction & Assembly (Streaming)")

        console.print(
            "[cyan]Streaming encoding and tensor assembly (skipping cache write)...[/cyan]"
        )

        # Parallel Encoding Stream
        chunk_size = 1000

        # Generator for batches
        def batch_gen() -> Iterator[List[ProcessedSample]]:
            for i in range(0, len(final_results), chunk_size):
                yield final_results[i : i + chunk_size]

        # import multiprocessing as mp  <-- Removed to avoid shadowing
        # Use spawn context to avoid fork-safety issues on macOS/Py3.12+
        ctx_enc: Any = mp.get_context("spawn")

        with ctx_enc.Pool(
            num_workers, initializer=encode_init_worker, initargs=(tokenizer_state,)
        ) as pool:
            # imap_unordered is fine because we just need to collect all data,
            # and shuffled beforehand anyway (except we must keep offsets consistent within sample?)
            # Wait, order matters for 'all_offsets' if we align with something?
            # But we are building the dataset from scratch.
            # We just need internal consistency (ids[i] matches label[i]).
            # Since we process a BATCH atomically, the batch is consistent.
            # And we append batches. So order of batches doesn't matter as long as all lists correspond.
            # Yes.

            for batch_res in pool.imap_unordered(encode_batch_fast, batch_gen()):
                # Unpack column batch (Array optimized)
                # Features
                batch_lens = None

                for field in FEATURE_FIELDS:
                    if field in batch_res["features_flat"]:
                        flat_vals = batch_res["features_flat"][field]
                        # Capture lengths from the first valid field (usually all same)
                        if batch_lens is None:
                            batch_lens = batch_res["features_lens"][field]

                        # Extend flattened array - FAST
                        all_encodings[field].extend(flat_vals)

                if batch_lens is None:
                    continue  # Should not happen

                # Offsets (Iterate array - fast enough)
                for length in batch_lens:
                    current_offset += length
                    all_offsets.append(current_offset)

                # KC Targets
                for k_key, k_val in batch_res["kc_ids"].items():
                    if k_key not in kc_collections:
                        kc_collections[k_key] = {
                            "ids": array.array("I"),
                            "counts": array.array("I"),
                        }
                    kc_collections[k_key]["ids"].extend(k_val)
                    # counts
                    kc_collections[k_key]["counts"].extend(
                        batch_res["kc_counts"][k_key]
                    )

                # Metadata arrays
                f_vals.extend(batch_res["f_val"])
                f_prags.extend(batch_res["f_prag"])
                g_vals.extend(batch_res["g_val"])
                g_prags.extend(batch_res["g_prag"])
                gram_labels.extend(batch_res["gram"])

                # Registers (Flattened array)
                all_reg_ids.extend(batch_res["reg_flat"])
                for length in batch_res["reg_lens"]:
                    cur_reg_offset += length
                    reg_offsets.append(cur_reg_offset)
        # End of parallel loop

        # Convert to tensors
        # Convert to tensors (Zero-copy from array.array)
        tensor_data: Dict[str, Any] = {
            "offsets": torch.frombuffer(all_offsets, dtype=torch.int32),
            "labels": {
                "f_val": torch.frombuffer(f_vals, dtype=torch.float32),
                "f_prag": torch.frombuffer(f_prags, dtype=torch.uint8).long(),
                "g_val": torch.frombuffer(g_vals, dtype=torch.float32),
                "g_prag": torch.frombuffer(g_prags, dtype=torch.uint8).long(),
                "gram": torch.frombuffer(gram_labels, dtype=torch.uint8).long(),
                "reg_ids": torch.frombuffer(
                    all_reg_ids, dtype=torch.uint8
                ).long(),  # Flattened
                "reg_offsets": torch.frombuffer(reg_offsets, dtype=torch.int32),
            },
            "version": 2,
        }

        # Add KC target tensors
        if kc_collections:
            tensor_data["kc_targets"] = {}
            for k_key, accum in kc_collections.items():
                counts = torch.frombuffer(accum["counts"], dtype=torch.int32)
                # Generate offsets from counts: [0, c1, c1+c2, ...]
                offsets = torch.zeros(len(counts) + 1, dtype=torch.int32)
                torch.cumsum(counts, dim=0, out=offsets[1:])

                tensor_data["kc_targets"][k_key] = {
                    "ids": torch.frombuffer(accum["ids"], dtype=torch.int32).long(),
                    "offsets": offsets,
                }

        # Add feature tensors
        for field, values in all_encodings.items():
            if len(values) > 0:
                tensor_data[field] = torch.frombuffer(values, dtype=torch.int32)

        torch.save(tensor_data, os.path.join(dataset_cache_dir, "dataset_tensors.pt"))
        console.print(
            f"  Saved binary tensors to [cyan]{os.path.join(dataset_cache_dir, 'dataset_tensors.pt')}[/cyan]"
        )

        # Print statistics
        vocab_sizes = tokenizer.get_vocab_sizes()
        console.print("\n[bold cyan]Dataset Statistics:[/bold cyan]")
        console.print(f"  Encoded samples: [bold]{len(final_results)}[/bold]")
        console.print("  Vocabulary sizes:")
        console.print(f"    Surface forms: {vocab_sizes['surface']:,}")
        console.print(f"    Lemmas: {vocab_sizes['lemma']:,}")
        console.print(f"    POS tags: {vocab_sizes['pos']}")
        console.print(f"    Conjugation types: {vocab_sizes['conjugated_type']}")
        console.print(f"    Conjugation forms: {vocab_sizes['conjugated_form']}")
        console.print(
            f"  Vocabulary cache: [cyan]{os.path.join(dataset_cache_dir, vocab_file)}[/cyan]"
        )
        console.print("\n[bold green]Dataset finalization complete.[/bold green]")
        timer.stop("Tensor Assembly & Saving (Phase 3)")

    # Final: Save metadata for fast-skip
    output_fingerprints = {}
    if args.output_grammatic and os.path.exists(args.output_grammatic):
        output_fingerprints["grammatic"] = get_file_fingerprint(args.output_grammatic)
    if args.output_agrammatic and os.path.exists(args.output_agrammatic):
        output_fingerprints["agrammatic"] = get_file_fingerprint(args.output_agrammatic)

    metadata = {
        "timestamp": time.time(),
        "fingerprints": current_fingerprints,  # Source fingerprints
        "output_fingerprints": output_fingerprints,  # Output fingerprints (combined files)
        "cache_version": CACHE_VERSION,
        "vocab_file": vocab_file,
    }
    os.makedirs(dataset_cache_dir, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
