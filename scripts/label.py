#!/usr/bin/env python3
"""Standalone script to label and cache Japanese sentences for style classification (V2 Binary Format)."""

import glob
import multiprocessing as mp
import os
import queue
import shutil
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, cast

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.table import Table

from kotogram import locations
from kotogram.kotogram import extract_token_features, split_kotogram
from kotogram.model import (
    FORMALITY_ID_TO_LABEL,
    FORMALITY_LABEL_TO_ID,
    REGISTER_ID_TO_LABEL,
    REGISTER_LABEL_TO_ID,
)
from kotogram.tokenizer import FEATURE_FIELDS, Tokenizer
from train.binary_io import (
    EXT_FEAT_PREFIX,
    EXT_KC_PREFIX,
    EXT_KOTOGRAMS,
    EXT_LABELS,
    EXT_OFFSETS,
    EXT_SENTENCES,
    merge_offset_shards,
    merge_shards,
    write_float_array,
    write_int_array,
)
from train.kc import compute_kc_targets
from train.profile import PhaseTimer, get_profile_dir
from train.tsv import parse_tsv

# Cache version for tokenizer compatibility
CACHE_VERSION = 12

# Global variable for worker processes only
_WORKER_OVERRIDES: Optional[Dict[str, List[Any]]] = None
_WORKER_ID: int = -1
_SHARD_DIR: str = ""
_TOKENIZER: Optional[Tokenizer] = None


console = Console()


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


def init_worker(
    worker_id: int,
    shard_dir: str,
    overrides: Dict[str, List[Any]],
    tokenizer_state: Dict[str, Any],
) -> None:
    """Initialize worker process with shard config, overrides, and tokenizer state."""
    # pylint: disable=global-statement
    global _WORKER_OVERRIDES, _WORKER_ID, _SHARD_DIR, _TOKENIZER
    _WORKER_OVERRIDES = overrides
    _WORKER_ID = worker_id
    _SHARD_DIR = shard_dir
    _TOKENIZER = Tokenizer()
    _TOKENIZER.load_state(tokenizer_state)


def analyze_batch(
    batch: List[Tuple[str, int]],
) -> Dict[str, Any]:
    # pylint: disable=too-many-locals, redefined-outer-name, reimported
    """Phase 1: Analyze labels, count vocab, return buffers."""
    from kotogram.analysis import FormalityLevel, RegisterLevel
    from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
    from scripts.rule_based_analysis import (
        analyze_formality,
        analyze_gender,
        analyze_register,
        formality_to_weight,
        infer_gender_from_register,
    )

    parser = SudachiJapaneseParser()

    # Buffers
    sentences_buf: List[str] = []
    kotograms_buf: List[str] = []

    f_val_buf: List[float] = []
    f_prag_buf: List[int] = []
    g_val_buf: List[float] = []
    g_prag_buf: List[int] = []
    gram_buf: List[int] = []
    reg_ids_buf: List[int] = []
    # Local offsets for this batch, starting at 0
    reg_offsets_buf: List[int] = [0]
    current_reg_offset = 0

    vocab_counters: Dict[str, Counter[Any]] = {f: Counter() for f in FEATURE_FIELDS}
    label_stats: Dict[str, Counter] = {
        "formality": Counter(),
        "gender_prag": Counter(),
        "register": Counter(),
        "grammatic": Counter(),
    }
    reg_samples: Dict[str, List[Any]] = {}

    for sentence, gram_label in batch:
        kotogram_obj = parser.japanese_to_kotogram(sentence)
        formality_enum = analyze_formality(kotogram_obj)
        gender_enum = analyze_gender(kotogram_obj)

        # Token features and vocabulary counting
        tokens = split_kotogram(kotogram_obj)

        for token in tokens:
            token_feat = extract_token_features(token)
            for field in FEATURE_FIELDS:
                val = getattr(token_feat, field)
                vocab_counters[field][val] += 1

        # Check for overrides
        overrides = _WORKER_OVERRIDES or {}
        if sentence in overrides:
            register_enums = overrides[sentence]
        else:
            register_enums = list(analyze_register(kotogram_obj))

        # ID for stats only
        formality_id = FORMALITY_LABEL_TO_ID.get(
            formality_enum, FORMALITY_LABEL_TO_ID[FormalityLevel.NEUTRAL]
        )

        # Calculate formality weight for training
        f_val, f_prag = formality_to_weight(formality_enum)
        if f_prag == 0:
            f_val = float("nan")

        # Assuming infer_gender_from_register is available in scope (it is)
        gender_val, gender_prag = infer_gender_from_register(
            gender_enum, register_enums
        )
        if gender_prag == 0:
            gender_val = float("nan")

        register_ids = [
            REGISTER_LABEL_TO_ID[r] for r in register_enums if r in REGISTER_LABEL_TO_ID
        ]
        if not register_ids:
            register_ids = [REGISTER_LABEL_TO_ID[RegisterLevel.NEUTRAL]]

        # The gender_val assignment to NaN if gender_prag is 0 is already handled above.
        # The diff snippet seems to imply a re-ordering or re-assignment, but the logic
        # for g_val (gender_val) being NaN based on g_prag (gender_prag) is already in place.
        # We will keep the existing placement of `if gender_prag == 0: gender_val = float("nan")`
        # and just add the new `final_gram` logic.

        # STRICT GRAMMATICALITY:
        # gram=1 iff source=1 AND f_prag=1 AND g_prag=1
        final_gram = gram_label and f_prag and gender_prag

        # Accumulate
        sentences_buf.append(sentence)
        kotograms_buf.append(kotogram_obj)
        f_val_buf.append(f_val)
        f_prag_buf.append(f_prag)
        g_val_buf.append(gender_val)
        g_prag_buf.append(gender_prag)
        gram_buf.append(final_gram)

        reg_ids_buf.extend(register_ids)
        current_reg_offset += len(register_ids)
        reg_offsets_buf.append(current_reg_offset)

        # Stats
        label_stats["formality"][formality_id] += 1
        label_stats["gender_prag"][gender_prag] += 1
        label_stats["grammatic"][gram_label] += 1
        for reg_id in register_ids:
            label_stats["register"][reg_id] += 1
            if len(reg_samples.get(str(reg_id), [])) < 3:
                if str(reg_id) not in reg_samples:
                    reg_samples[str(reg_id)] = []
                reg_samples[str(reg_id)].append(
                    {
                        "sentence": sentence,
                        "formality_id": formality_id,
                        "gender_value": gender_val,
                    }
                )

    return {
        "sentences": sentences_buf,
        "kotograms": kotograms_buf,
        "f_val": f_val_buf,
        "f_prag": f_prag_buf,
        "g_val": g_val_buf,
        "g_prag": g_prag_buf,
        "gram": gram_buf,
        "reg_ids": reg_ids_buf,
        "reg_offsets": reg_offsets_buf,
        "vocab": vocab_counters,
        "stats": label_stats,
        "reg_samples": reg_samples,
    }


def _encode_shard_phase2(worker_id: int) -> None:
    # pylint: disable=too-many-locals
    """Phase 2: Read kotograms, encode features/KC, write binaries."""
    # This runs ONCE per worker, processing the entire shard file created in Phase 1.
    shard_prefix = os.path.join(_SHARD_DIR, f"shard_{worker_id}")
    koto_path = f"{shard_prefix}.{EXT_KOTOGRAMS}"

    if not os.path.exists(koto_path):
        return

    # Buffers
    feat_buffers: Dict[str, List[int]] = {f: [] for f in FEATURE_FIELDS}
    token_lengths_buf: List[int] = []
    offsets_buf: List[int] = [0]
    current_offset = 0

    kc_buffers: Dict[str, Dict[str, Any]] = {}

    with open(koto_path, "r", encoding="utf-8") as f:
        for line in f:
            kotogram_obj = line.strip()
            if not kotogram_obj:
                continue

            tokens = split_kotogram(kotogram_obj)

            token_lengths_buf.append(len(tokens))
            current_offset += len(tokens)
            offsets_buf.append(current_offset)

            # Features
            feat_ids_map: Dict[str, List[int]] = {f: [] for f in FEATURE_FIELDS}
            for token in tokens:
                token_feat = extract_token_features(token)
                for field in FEATURE_FIELDS:
                    val = getattr(token_feat, field)
                    if _TOKENIZER:
                        fid = _TOKENIZER.get_id(field, val)
                        feat_buffers[field].append(fid)
                        feat_ids_map[field].append(fid)

            # KC Targets
            kc_targets = compute_kc_targets(cast(Any, feat_ids_map))
            for k_key, vals in kc_targets.items():
                if k_key not in kc_buffers:
                    kc_buffers[k_key] = {"ids": [], "offsets": [0], "cur_off": 0}

                # compute_kc_targets returns Dict[str, List[int]] (ids)
                # target is just list of IDs
                ids = vals
                if isinstance(ids, list):
                    # Mypy check passed via compute_kc_targets return type normally, but being safe
                    cast(List[int], kc_buffers[k_key]["ids"]).extend(ids)
                    kc_buffers[k_key]["cur_off"] = cast(
                        int, kc_buffers[k_key]["cur_off"]
                    ) + len(ids)
                    cast(List[int], kc_buffers[k_key]["offsets"]).append(
                        cast(int, kc_buffers[k_key]["cur_off"])
                    )

    # Write Features
    write_int_array(f"{shard_prefix}.{EXT_OFFSETS}", offsets_buf, "i")

    for field in FEATURE_FIELDS:
        # Fixed: EXT_FEAT_PREFIX already includes '_'
        f_path = f"{shard_prefix}.{EXT_FEAT_PREFIX}{field}.bin"
        write_int_array(f_path, feat_buffers[field], "i")

    # Write KC
    for k_key, accum in kc_buffers.items():
        # Fixed: EXT_KC_PREFIX already includes '_'
        write_int_array(
            f"{shard_prefix}.{EXT_KC_PREFIX}{k_key}_ids.bin", accum["ids"], "i"
        )
        write_int_array(
            f"{shard_prefix}.{EXT_KC_PREFIX}{k_key}_{EXT_OFFSETS}",
            accum["offsets"],
            "i",
        )


def print_stats(label_stats: Dict[str, Counter]) -> None:
    # pylint: disable=too-many-locals
    """Print attractive statistics about the labeling results."""
    if not label_stats:
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
        label_stats["formality"],
        lambda x: FORMALITY_ID_TO_LABEL[x].value,
    )

    # Gender
    _print_dist(
        "Gender Pragmatic Distribution",
        "bold cyan",
        label_stats["gender_prag"],
        lambda x: {1: "Pragmatic", 0: "Unpragmatic"}[x],
    )

    # Register
    _print_dist(
        "Register Distribution",
        "bold yellow",
        label_stats["register"],
        lambda x: REGISTER_ID_TO_LABEL[x].value,
    )

    # Grammaticality
    _print_dist(
        "Grammaticality Distribution",
        "bold green",
        label_stats["grammatic"],
        lambda x: {1: "Grammatic", 0: "Agrammatic"}[x],
    )


def worker_p2_wrapper(wid: int, s_dir: str, tok_state: Dict[str, Any]) -> None:
    init_worker(wid, s_dir, {}, tok_state)
    _encode_shard_phase2(wid)


def worker_p1_wrapper(
    wid: int,
    chunk: List[Tuple[str, int]],
    s_dir: str,
    overrides: Dict[str, List[Any]],
    result_queue_arg: Any,
) -> None:
    # pylint: disable=too-many-locals
    if overrides is None:
        overrides = {}
    init_worker(wid, s_dir, overrides, {})
    b_size = 2000

    # Group buffers to reduce locals
    buffers = {
        "sentences": [],
        "kotograms": [],
        "f_val": [],
        "f_prag": [],
        "g_val": [],
        "g_prag": [],
        "gram": [],
        "reg_ids": [],
        "reg_offsets": [0],
        "vocab": {f: Counter() for f in FEATURE_FIELDS},
        "stats": {
            "formality": Counter(),
            "gender_prag": Counter(),
            "register": Counter(),
            "grammatic": Counter(),
        },
        "reg_samples": {},
    }

    total_reg_ids_so_far = 0

    for i in range(0, len(chunk), b_size):
        batch = chunk[i : i + b_size]
        res = analyze_batch(batch)

        # Extend data
        cast(List[str], buffers["sentences"]).extend(res["sentences"])
        cast(List[str], buffers["kotograms"]).extend(res["kotograms"])
        cast(List[float], buffers["f_val"]).extend(res["f_val"])
        cast(List[int], buffers["f_prag"]).extend(res["f_prag"])
        cast(List[float], buffers["g_val"]).extend(res["g_val"])
        cast(List[int], buffers["g_prag"]).extend(res["g_prag"])
        cast(List[int], buffers["gram"]).extend(res["gram"])
        cast(List[int], buffers["reg_ids"]).extend(res["reg_ids"])

        # Handle offsets
        shifted = [o + total_reg_ids_so_far for o in res["reg_offsets"][1:]]
        cast(List[int], buffers["reg_offsets"]).extend(shifted)
        total_reg_ids_so_far += len(res["reg_ids"])

        # Merge stats
        for f in FEATURE_FIELDS:
            cast(Dict[str, Counter], buffers["vocab"])[f].update(res["vocab"][f])
        for k in cast(Dict[str, Counter], buffers["stats"]):
            cast(Dict[str, Counter], buffers["stats"])[k].update(res["stats"][k])

        reg_samples_buf = cast(Dict[str, List[Any]], buffers["reg_samples"])
        for rid, samps in res["reg_samples"].items():
            if rid not in reg_samples_buf:
                reg_samples_buf[rid] = []
            reg_samples_buf[rid].extend(samps)

    # Write ALL data once
    shard_prefix = os.path.join(s_dir, f"shard_{wid}")
    os.makedirs(s_dir, exist_ok=True)

    with open(f"{shard_prefix}.{EXT_SENTENCES}", "w", encoding="utf-8") as outfile:
        for s in cast(List[str], buffers["sentences"]):
            outfile.write(s + "\n")
    with open(f"{shard_prefix}.{EXT_KOTOGRAMS}", "w", encoding="utf-8") as outfile:
        for k in cast(List[str], buffers["kotograms"]):
            outfile.write(k + "\n")

    write_float_array(
        f"{shard_prefix}.{EXT_LABELS}_f_val",
        cast(List[float], buffers["f_val"]),
        "f",
    )
    write_int_array(
        f"{shard_prefix}.{EXT_LABELS}_f_prag",
        cast(List[int], buffers["f_prag"]),
        "B",
    )
    write_float_array(
        f"{shard_prefix}.{EXT_LABELS}_g_val",
        cast(List[float], buffers["g_val"]),
        "f",
    )
    write_int_array(
        f"{shard_prefix}.{EXT_LABELS}_g_prag",
        cast(List[int], buffers["g_prag"]),
        "B",
    )
    write_int_array(
        f"{shard_prefix}.{EXT_LABELS}_gram", cast(List[int], buffers["gram"]), "B"
    )
    write_int_array(
        f"{shard_prefix}.{EXT_LABELS}_reg_ids.bin",
        cast(List[int], buffers["reg_ids"]),
        "i",
    )
    write_int_array(
        f"{shard_prefix}.{EXT_LABELS}_reg_ids_{EXT_OFFSETS}",
        cast(List[int], buffers["reg_offsets"]),
        "i",
    )

    result_queue_arg.put((buffers["vocab"], buffers["stats"], buffers["reg_samples"]))


def main() -> None:
    # pylint: disable=too-many-locals, too-many-statements
    import argparse

    parser = argparse.ArgumentParser(description="Label and cache Japanese sentences.")
    parser.add_argument(
        "--grammatic-pattern",
        type=str,
        required=True,
        help="Primary TSV data file(s) (glob pattern)",
    )
    parser.add_argument("--agrammatic-pattern", type=str, help="Agrammatic TSV pattern")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument(
        "--num-workers", type=int, default=0, help="Number of workers (default: CPU-1)"
    )

    args = parser.parse_args()

    # Resolve and inject paths from locations.py into args namespace
    dataset_cache_dir = locations.get_style_dataset_cache_dir()
    shard_dir = os.path.join(dataset_cache_dir, "shards")

    profile_dir = get_profile_dir()
    if profile_dir:
        os.makedirs(profile_dir, exist_ok=True)
    timer = PhaseTimer(console, profile_dir)

    # Clean up previous shards
    if os.path.exists(shard_dir):
        shutil.rmtree(shard_dir)
    os.makedirs(shard_dir, exist_ok=True)

    if args.num_workers > 0:
        num_workers = args.num_workers
    else:
        num_workers = max(1, mp.cpu_count() - 1)

    # ... (Glob parsing same as before)
    def process_file_group(patterns: Any, gram_label: int) -> List[Tuple[str, int]]:
        if not patterns:
            return []
        file_list = []
        if isinstance(patterns, str):
            file_list = glob.glob(patterns)
        else:
            for p in patterns:
                file_list.extend(glob.glob(p))
        if not file_list:
            return []
        unique_rows = []
        seen = set()
        for f_path in sorted(file_list):
            with open(f_path, "r", encoding="utf-8") as file_handle:
                for line in file_handle:
                    sentence = parse_tsv(line)
                    if sentence not in seen:
                        seen.add(sentence)
                        unique_rows.append((sentence, gram_label))
        return unique_rows

    all_rows = []
    gram_patterns = [args.grammatic_pattern]
    console.print(f"Scanning data with {num_workers} workers...")

    # ... (Scanning logic)
    rows = process_file_group(gram_patterns, 1)
    all_rows.extend(rows)
    if args.agrammatic_pattern:
        rows = process_file_group([args.agrammatic_pattern], 0)
        all_rows.extend(rows)

    seen_global = set()
    unique_all_rows = []
    for r in all_rows:
        if r[0] not in seen_global:
            seen_global.add(r[0])
            unique_all_rows.append(r)
    all_rows = unique_all_rows

    console.print(f"Total unique sentences: [bold]{len(all_rows):,}[/bold]")
    timer.mark("Scanning Input")

    # Init main stats
    merged_counters: Dict[str, Counter] = {f: Counter() for f in FEATURE_FIELDS}
    merged_label_stats: Dict[str, Counter] = {
        "formality": Counter(),
        "gender_prag": Counter(),
        "register": Counter(),
        "grammatic": Counter(),
    }
    merged_reg_samples: Dict[str, List[Any]] = {}

    from scripts.rule_based_analysis import load_register_overrides

    register_overrides = load_register_overrides()

    # PHASE 1: Analyze & Shard Text
    ctx = mp.get_context("spawn")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task1 = progress.add_task("[green]Phase 1: Analyzing...", total=len(all_rows))

        # Split into N chunks
        chunk_size = (len(all_rows) + num_workers - 1) // num_workers
        chunks = [
            all_rows[i : i + chunk_size] for i in range(0, len(all_rows), chunk_size)
        ]

        result_queue = ctx.Queue()
        procs = []

        for i, chunk in enumerate(chunks):
            p = ctx.Process(
                target=worker_p1_wrapper,
                args=(i, chunk, shard_dir, register_overrides, result_queue),
            )
            procs.append(p)
            p.start()

        # Collect results
        finished_count = 0
        while finished_count < len(procs):
            try:
                # 1. Try to get result
                res = result_queue.get(timeout=0.1)

                # 2. Process result
                vc, ls, rs = res
                # Merge global
                for f in FEATURE_FIELDS:
                    merged_counters[f].update(vc[f])
                for k, counter in merged_label_stats.items():
                    counter.update(ls[k])
                for rid, samps in rs.items():
                    if rid not in merged_reg_samples:
                        merged_reg_samples[rid] = []
                    merged_reg_samples[rid].extend(samps)

                # label_stats["grammatic"] has count of sentences.
                count = sum(ls["grammatic"].values())
                progress.update(task1, advance=count)

                finished_count += 1

            except queue.Empty:
                # 3. Check for dead workers
                failed = False
                for p in procs:
                    if not p.is_alive() and p.exitcode != 0:
                        # Worker died without yielding result (since queue is empty and we are still looping)
                        console.print(
                            f"[red]Worker {p.pid} failed/died unexpectedly (exit code: {p.exitcode})[/red]"
                        )
                        failed = True

                if failed:
                    for p in procs:
                        p.terminate()
                    sys.exit(1)

        for p in procs:
            p.join()

    timer.mark("Phase 1: Analysis Complete")
    print_stats(merged_label_stats)

    # Build Vocab
    vocab_file = "vocab.json"
    dataset_cache_dir = locations.get_style_dataset_cache_dir()
    tokenizer = Tokenizer()
    _build_and_save_vocab(tokenizer, merged_counters, dataset_cache_dir, vocab_file)
    console.print(f"Saved vocab to {vocab_file}")

    # PHASE 2: Encoding
    # Run _encode_shard_phase2 via Pool/Process.
    # Use Same N workers.

    console.print("Phase 2: Encoding Shards...")
    procs_p2 = []

    # Need to reload tokenizer state to pass to workers?
    # Yes, we modified it.
    # tokenizer state is picklable.

    for i in range(len(chunks)):  # Same N workers
        p = ctx.Process(
            target=worker_p2_wrapper,
            args=(i, shard_dir, {"field_vocabs": tokenizer.field_vocabs}),
        )
        procs_p2.append(p)
        p.start()

    for p in procs_p2:
        p.join()

    timer.mark("Phase 2: Encoding Complete")

    # PHASE 3: Merge Shards (Main Process)
    # We iterate N workers and cat their files.
    # Also adjust offsets globally?
    # Yes.
    # Global 'dataset_offsets.bin' = [0, len(s1), len(s1)+len(s2)...]
    # We need to read shard_offsets.bin (which are [0, l1, l2...])
    # And offset them by the global cumulative total.

    console.print("Phase 3: Merging Shards...")

    # Merge Offsets (Sentences)
    # shard_{i}.offsets.bin
    console.print("  Merging sentence offsets...")
    merge_offset_shards(
        shard_dir,
        os.path.join(dataset_cache_dir, EXT_OFFSETS),
        len(chunks),
        "shard_{}." + EXT_OFFSETS,
    )

    # Merge Features
    for field in FEATURE_FIELDS:
        console.print(f"  Merging feature: {field}...")
        merge_shards(
            shard_dir,
            os.path.join(dataset_cache_dir, f"{EXT_FEAT_PREFIX}{field}.bin"),
            len(chunks),
            "shard_{}." + f"{EXT_FEAT_PREFIX}{field}.bin",
        )

    # Merge Text (Sentences, Kotograms) - Simple Concatenation
    for ext in [EXT_SENTENCES, EXT_KOTOGRAMS]:
        out_f = os.path.join(dataset_cache_dir, ext)
        console.print(f"  Merging text: {ext}...")
        with open(out_f, "w", encoding="utf-8") as outfile:
            for i in range(len(chunks)):
                fname = f"shard_{i}.{ext}"
                path = os.path.join(shard_dir, fname)
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as infile:
                        shutil.copyfileobj(infile, outfile)
                else:
                    # Should not happen if worker ran successfully, but safe behavior?
                    pass

    # Merge Labels
    # Files are: "shard_{}_labels_f_val", etc. based on Phase 1 logic.
    # Phase 1:
    # write_float_array(f"{shard_prefix}.{EXT_LABELS}_f_val", f_val_buf, "f")
    # shard_prefix = .../shard_{worker_id}
    # So file is .../shard_{worker_id}.labels.bin_f_val
    # Wait, EXT_LABELS is "labels.bin".
    # So it's "shard_{}.labels.bin_f_val".

    # Note: merge_shards uses a format string.
    # If shard_template is "shard_{}." + f"{EXT_LABELS}_f_val"
    # It formats to "shard_0.labels.bin_f_val". Correct.

    label_suffixes = [
        f"{EXT_LABELS}_f_val",
        f"{EXT_LABELS}_f_prag",
        f"{EXT_LABELS}_g_val",
        f"{EXT_LABELS}_g_prag",
        f"{EXT_LABELS}_gram",
        f"{EXT_LABELS}_reg_ids.bin",
    ]

    for lf in label_suffixes:
        console.print(f"  Merging label: {lf}...")
        merge_shards(
            shard_dir,
            os.path.join(dataset_cache_dir, lf),
            len(chunks),
            "shard_{}." + lf,
            dtype_size=4 if "f_val" in lf or "g_val" in lf or "ids" in lf else 1,
        )

    # Merge Register Offsets
    console.print("  Merging register offsets...")
    merge_offset_shards(
        shard_dir,
        os.path.join(dataset_cache_dir, f"{EXT_LABELS}_reg_ids_{EXT_OFFSETS}"),
        len(chunks),
        "shard_{}." + f"{EXT_LABELS}_reg_ids_{EXT_OFFSETS}",
    )

    # Merge KC Targets
    # Check for KC files in first shard
    kc_files = glob.glob(os.path.join(shard_dir, f"shard_0.{EXT_KC_PREFIX}*"))
    kc_keys = set()
    for kf in kc_files:
        # Expected: shard_0.kc_TargetName_ids.bin
        # or shard_0.kc_TargetName_offsets.bin
        base = os.path.basename(kf)
        if "_ids.bin" in base:
            # remove shard_0.
            # remove _ids.bin
            # remove kc_ prefix
            # base: shard_0.kc_trigram_ids.bin

            # split by .
            parts = base.split(".")
            # parts[0] = shard_0
            # parts[1] = kc_trigram_ids
            # parts[2] = bin

            # actually we used f"{shard_prefix}.{EXT_KC_PREFIX}_{k_key}_ids.bin"
            # EXT_KC_PREFIX = "kc_"
            # So "shard_0.kc__trigram_ids.bin" ?
            # Check Phase 2:
            # write_int_array(f"{shard_prefix}.{EXT_KC_PREFIX}_{k_key}_ids.bin", ...)
            # if k_key is "trigram", and EXT_KC_PREFIX is "kc_"
            # "shard_0.kc__trigram_ids.bin" (double underscore?)
            # No, line 396: f"{...}.{EXT_KC_PREFIX}_{k_key}_ids.bin"
            # logic in label.py line 49: EXT_KC_PREFIX = "kc" (without underscore?)
            # binary_io.py line 12: EXT_KC_PREFIX = "kc_"
            # label.py imports it.
            # So f"{...}.kc__{k_key}_ids.bin"
            # That seems like a bug in Phase 2 writing if double underscore unintended.
            # But consistent reading deals with it.

            # Let's check imports in label.py content I viewed.
            # Line 49: EXT_KC_PREFIX,
            # It comes from train.binary_io
            # In binary_io.py: EXT_KC_PREFIX = "kc_"
            # So Phase 2 writes: f".../shard_i.kc__key_ids.bin"
            # It has double underscore.
            # I should support that here.

            # parts[1] is e.g. "kc_trigram_ids"
            mid = parts[1]
            # EXT_KC_PREFIX is "kc_"
            if mid.startswith(EXT_KC_PREFIX):
                key = mid[len(EXT_KC_PREFIX) :]  # "trigram_ids"
            else:
                key = mid

            key = key.replace("_ids", "")
            kc_keys.add(key)

    for key in kc_keys:
        console.print(f"  Merging KC target: {key}...")
        # IDs
        # Filename: ... .kc_{key}_ids.bin
        suffix_ids = f"{EXT_KC_PREFIX}{key}_ids.bin"
        merge_shards(
            shard_dir,
            os.path.join(dataset_cache_dir, suffix_ids),
            len(chunks),
            "shard_{}." + suffix_ids,
        )
        # Offsets
        suffix_off = f"{EXT_KC_PREFIX}{key}_{EXT_OFFSETS}"
        merge_offset_shards(
            shard_dir,
            os.path.join(dataset_cache_dir, suffix_off),
            len(chunks),
            "shard_{}." + suffix_off,
        )

    timer.mark("Phase 3: Merging Complete")

    # Cleanup Shards
    console.print("Cleaning up shards...")
    if os.path.exists(shard_dir):
        shutil.rmtree(shard_dir)

    console.print(
        f"[bold green]Labeling Complete![/bold green] Data saved to {dataset_cache_dir}"
    )


if __name__ == "__main__":
    main()
