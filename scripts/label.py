#!/usr/bin/env python3

"""
Standalone script to label and cache Japanese sentences for style classification (V2 Binary Format).

This script implements a high-performance, multi-phase data processing pipeline that converts raw
Japanese text (or pre-parsed DB rows) into a binary dataset optimized for training deep learning models.

The pipeline consists of three distinct phases:

    Phase 1: Analysis & Vocabulary Building
        - Parallel processing of sentences using `kotogram` analysis.
        - Computes formality, register, and gender pragmatic scores.
        - Builds global vocabulary counters for all feature fields.
        - Buffers intermediate results (sentence text, kotograms, raw scores) to sharded temporary files.
        - This phase is CPU-bound and scales linearly with `num_workers`.

    Phase 2: Encoding & KC Target Computation
        - Reads the intermediate kotograms generated in Phase 1.
        - Encodes token features into integer IDs using the globally finalized tokenizer from Phase 1.
        - Computes Knowledge Component (KC) targets (e.g., n-grams, structural hashes) for pretraining.
        - Writes these encoded integer arrays to binary shard files.

    Phase 3: Merging & Finalization
        - Aggregates all sharded binary files into a single, contiguous memory-mappable dataset.
        - Merges vocabulary and offset indices.
        - Produces the final `models/style/dataset` directory structure.

Usage:
    python3 scripts/label.py --grammatic-pattern "data/*.tsv" --num-workers 16
    python3 scripts/label.py --source-db data/corpus.db --num-workers 16

Output:
    The script populates the directory returned by `locations.get_style_dataset_cache_dir()`
    (typically `models/style/dataset`) with:
    - `sentences`: All unique sentences.
    - `kotograms`: Analyzed kotogram representations.
    - `labels_*`: Binary arrays for various style labels.
    - `kc_*`: Binary arrays for Knowledge Component targets.
    - `tokenizer.json`: The fitted tokenizer vocabulary.
"""

import argparse
import glob
import math
import multiprocessing as mp
import os
import queue
import shutil
import sqlite3
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple, cast

from rich.console import Console
from rich.table import Table

from kotogram.analysis import FormalityLevel, RegisterLevel
from kotogram.constants import (
    FORMALITY_ID_TO_LABEL,
    FORMALITY_LABEL_TO_ID,
    REGISTER_ID_TO_LABEL,
    REGISTER_LABEL_TO_ID,
)
from kotogram.japanese_parser import KotogramFormat
from kotogram.kotogram import extract_token_features, split_kotogram
from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
from kotogram.tokenizer import FEATURE_FIELDS, Tokenizer, get_vocab_strings
from scripts.progress_utils import create_progress
from scripts.rule_based_analysis import (
    analyze_formality,
    analyze_gender,
    analyze_register,
    formality_to_weight,
    infer_gender_from_register,
    parse_gp_ids,
)
from train import io as train_io
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
from train.kc import KcFamilyId, compute_kc_targets, initialize_disallow_filter
from train.profile import PhaseTimer, get_profile_dir
from train.tsv import parse_tsv


def _debug_constant_check() -> int:
    """Dummy function to verify parameter recorder."""
    # This function is intended to verify parameter recording,
    # but currently contains hardcoded logic.
    # The value 12345 is part of this hardcoded logic.
    val = 12345
    return val


_debug_constant_check()


# Global state for worker processes.
# These variables are initialized via `init_worker` in each child process.
# pylint: disable=too-many-lines
# pylint: disable=too-many-locals
# pylint: disable=global-statement
# pylint: disable=protected-access

_WORKER_OVERRIDES: Optional[Dict[str, List[Any]]] = None
_WORKER_ID: int = -1
_SHARD_DIR: str = ""
_TOKENIZER: Optional[Tokenizer] = None


console = Console()


def _load_gp_priors_from_db(db_path: str) -> List[float]:
    """Load per-grammar-point priors vector from corpus.db.

    Expected schema (user-managed source of truth):
      grammar(id TEXT PRIMARY KEY, name TEXT NOT NULL, prior REAL NULL)

    Returns:
        A dense float vector where index == gp numeric id (e.g. gp0123 -> 123),
        value is the prior in [0,1], and missing/unset priors are NaN.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        # Determine max numeric gp id from the grammar dictionary itself.
        c.execute("SELECT id FROM grammar")
        ids = [row[0] for row in c.fetchall()]
        max_id = 0
        for gid_str in ids:
            if isinstance(gid_str, str) and gid_str.startswith("gp"):
                num = gid_str[2:]
                if num.isdigit():
                    max_id = max(max_id, int(num))

        # Always return a vector if column exists: NaN means "unset, use defaults".
        priors: List[float] = [float("nan")] * (max_id + 1)

        # Fill entries for any explicitly-set priors.
        # NOTE: We intentionally do NOT check whether grammar.prior exists.
        # If the DB schema is missing it, SQLite will raise and we want that to fail loudly.
        c.execute("SELECT id, prior FROM grammar WHERE prior IS NOT NULL")
        for gid_str, prior_val in c.fetchall():
            if not isinstance(gid_str, str) or not gid_str.startswith("gp"):
                continue
            num = gid_str[2:]
            if not num.isdigit():
                continue
            idx = int(num)
            priors[idx] = float(prior_val)

        return priors
    finally:
        conn.close()


def _validate_register_mapping_against_db(db_path: str) -> None:
    """Validate that corpus.db register table matches kotogram.constants mapping.

    This ensures the source of truth (kotogram/constants.py) is in sync with the DB.
    The code mapping is what's used at inference time, so DB must match it.

    Args:
        db_path: Path to corpus.db

    Raises:
        ValueError: If mappings don't match
    """
    if not os.path.exists(db_path):
        raise ValueError(
            f"Database not found at {db_path}. "
            "The register mapping validation requires a valid corpus.db."
        )

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if register table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='register'"
    )
    if not cursor.fetchone():
        conn.close()
        raise ValueError(
            f"Register table not found in {db_path}. "
            "The database must have a 'register' table with the mapping from kotogram.constants. "
            "Rebuild the database using: python scripts/curate build <source>"
        )

    cursor.execute("SELECT id, label FROM register ORDER BY id")
    db_rows = cursor.fetchall()
    conn.close()

    db_mapping = {row[0]: row[1].upper() for row in db_rows}

    # Validate all DB entries are in code
    mismatches = []
    for db_id, db_label_upper in db_mapping.items():
        code_label = REGISTER_ID_TO_LABEL.get(db_id)
        if code_label is None:
            mismatches.append(
                f"  ID {db_id}: exists in DB ('{db_label_upper}') but not in kotogram.constants"
            )
        elif code_label.name != db_label_upper:
            mismatches.append(
                f"  ID {db_id}: DB has '{db_label_upper}' but kotogram.constants has '{code_label.name}'"
            )

    # Validate all code entries are in DB
    for code_id, code_label in REGISTER_ID_TO_LABEL.items():
        if code_id not in db_mapping:
            mismatches.append(
                f"  ID {code_id}: exists in kotogram.constants ('{code_label.name}') but not in DB"
            )

    if mismatches:
        error_msg = (
            "\n[bold red]ERROR: Register mapping mismatch between corpus.db and kotogram.constants![/bold red]\n\n"
            "[yellow]The source of truth is kotogram/constants.py (used at inference time).[/yellow]\n"
            "[yellow]The corpus.db register table must match it exactly.[/yellow]\n\n"
            "Mismatches found:\n" + "\n".join(mismatches) + "\n\n"
            "[bold]Fix options:[/bold]\n"
            "  1. Rebuild corpus.db using: python scripts/curate build <source_file>\n"
            "  2. If kotogram.constants is wrong, fix it and rebuild the DB\n"
        )
        console.print(error_msg)
        raise ValueError("Register mapping validation failed")

    console.print(
        f"[green]✓ Register mapping validated: {len(db_mapping)} labels match between DB and code[/green]"
    )


def _build_and_save_vocab(
    tokenizer: Tokenizer,
    merged_counters: Dict[str, Counter],
    cache_dir: str,
) -> None:
    """
    Build vocabulary from merged counters and save to disk.

    This function takes the aggregated counters from all workers (Phase 1) and populates
    the tokenizer's internal vocabulary structures. It then persists the tokenizer state
    to a JSON file for use in Phase 2 and subsequent training.

    Args:
        tokenizer: The tokenizer instance to populate.
        merged_counters: Dictionary mapping feature field names to Counter objects.
        cache_dir: Directory to save the vocab file.
        cache_name: Filename for the vocab file (e.g., 'tokenizer.json').
    """
    from kotogram.tokenizer import UNK_TOKEN

    for field in FEATURE_FIELDS:
        counter = merged_counters.get(field, Counter())
        vocab = tokenizer.field_vocabs[field]
        # Add tokens to tokenizer in order of frequency (most common first).
        for value, _ in counter.most_common():
            if not value:
                value = UNK_TOKEN
            if value not in vocab:
                vocab[value] = len(vocab)

    # Ensure the <READING_MASK> sentinel is in the vocabulary so it gets a stable ID.
    vocab_reading = tokenizer.field_vocabs["reading_gram"]
    if "<READING_MASK>" not in vocab_reading:
        vocab_reading["<READING_MASK>"] = len(vocab_reading)

    os.makedirs(cache_dir, exist_ok=True)
    cache_name = "vocab.json"
    train_io.save_tokenizer(tokenizer, os.path.join(cache_dir, cache_name))


def init_worker(
    worker_id: int,
    shard_dir: str,
    overrides: Dict[str, List[Any]],
    tokenizer_state: Dict[str, Any],
) -> None:
    """
    Initialize worker process with shard config, overrides, and tokenizer state.

    This function is called at the start of each worker process to set up global state.
    It handles:
    1. Setting the worker ID for unique shard naming.
    2. Loading manual register overrides (if any).
    3. Rehydrating the tokenizer from state (Phase 2 only).

    Args:
        worker_id: Unique 0-indexed ID for this worker.
        shard_dir: Directory where shards should be written.
        overrides: Dictionary of manual register level overrides.
        tokenizer_state: State dict to restore the Tokenizer from (used in Phase 2).
    """
    # Use global keywords to set module-level variables in the worker process.
    global _WORKER_OVERRIDES, _WORKER_ID, _SHARD_DIR, _TOKENIZER
    _WORKER_OVERRIDES = overrides
    _WORKER_ID = worker_id
    _SHARD_DIR = shard_dir
    _TOKENIZER = Tokenizer()
    _TOKENIZER.load_state(tokenizer_state)
    # Initialize disallow filter once per worker
    compound_1_vocab = _TOKENIZER.field_vocabs.get("compound_1", {})
    initialize_disallow_filter(compound_1_vocab)


def analyze_batch(
    batch: List[Tuple[str, int]],
) -> Dict[str, Any]:
    """
    Phase 1: Analyze a batch of raw sentences.

    This function performs the heavy lifting of linguistic analysis. It:
    1. Parses sentences into `kotograms`.
    2. Analyzes formality, gender, and register using rule-based heuristics.
    3. Aggregates vocabulary counts for the global tokenizer.
    4. Buffers all results for efficient batch writing to disk.

    Args:
        batch: List of tuples (sentence, grammatic_label).

    Returns:
        Dict containing buffered lists of features, stats counters, and sample registers.
    """
    # Import locally to avoid top-level dependency issues or circular imports if any.

    parser = SudachiJapaneseParser()

    # Initialize buffers for columnar data storage.
    # These map directly to the binary output format.
    sentences_buf: List[str] = []
    kotograms_buf: List[str] = []

    f_val_buf: List[float] = []
    f_prag_buf: List[int] = []
    g_val_buf: List[float] = []
    g_prag_buf: List[int] = []
    gram_buf: List[int] = []
    reg_ids_buf: List[int] = []
    # Offsets for jagged array of register IDs (one-to-many relationship).
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
        kotogram_obj = parser.japanese_to_kotogram(
            sentence, fmt=KotogramFormat.TRAINING_MASK
        )
        formality_enum = analyze_formality(kotogram_obj)
        gender_enum = analyze_gender(kotogram_obj)

        # Tokenize and collect feature statistics for vocabulary building.
        tokens = split_kotogram(kotogram_obj)

        for token in tokens:
            token_feat = extract_token_features(token)
            # Get vocab-ready strings using centralized function
            vocab_strings = get_vocab_strings(token_feat)
            for field in FEATURE_FIELDS:
                vocab_counters[field][vocab_strings[field]] += 1

        # Check for manual overrides for register analysis (used for corrections).
        overrides = _WORKER_OVERRIDES or {}
        if sentence in overrides:
            register_enums = overrides[sentence]
        else:
            register_enums = list(analyze_register(kotogram_obj))

        # Convert Enums to integer IDs for storage.
        formality_id = FORMALITY_LABEL_TO_ID.get(
            formality_enum, FORMALITY_LABEL_TO_ID[FormalityLevel.NEUTRAL]
        )

        # Convert formality level to a continuous weight (-1.0 to 1.0).
        f_val, f_prag = formality_to_weight(formality_enum)
        if f_prag == 0:
            f_val = float("nan")

        # Infer gender probability based on register if explicit gender markers are missing.
        # This helps propagate gender signals from register context.
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

        # Final grammaticality check.
        # Logic: A sentence is "grammatic" for training ONLY if:
        # 1. It was labeled as grammatic in source (gram_label=1).
        # 2. It has pragmatic formality (f_prag=1).
        # 3. It has pragmatic gender (gender_prag=1).
        # This ensures we don't train on ambiguous or neutral-only data that adds noise.
        final_gram = gram_label and f_prag and gender_prag

        # Append to columnar buffers.
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

        # Update stats.
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


def analyze_batch_from_db(
    batch: List[Tuple[Any, ...]],
) -> Dict[str, Any]:
    """
    Phase 1 (DB): Process DB rows directly without re-analysis.

    This variant of `analyze_batch` optimizes for speed when the input source is a SQLite database
    that already contains valid analysis results (golden labels). It trusts the DB columns for
    key metrics but still re-computes `kotograms` to ensure the correct features are extracted
    for the current tokenizer version.

    Args:
        batch: List of tuples (sentence, formality, gender, grammatic, r_ids_str).

    Returns:
        Dict containing buffered lists of features, stats counters, and sample registers.
    """

    parser = SudachiJapaneseParser()

    # Initialize buffers for columnar data storage.
    sentences_buf: List[str] = []
    kotograms_buf: List[str] = []

    f_val_buf: List[float] = []
    f_prag_buf: List[int] = []
    g_val_buf: List[float] = []
    g_prag_buf: List[int] = []
    gram_buf: List[int] = []
    reg_ids_buf: List[int] = []
    reg_offsets_buf: List[int] = [0]
    current_reg_offset = 0

    # Grammar point buffers (jagged arrays with offsets)
    gp_pos_ids_buf: List[int] = []
    gp_pos_offsets_buf: List[int] = [0]
    gp_neg_ids_buf: List[int] = []
    gp_neg_offsets_buf: List[int] = [0]
    current_gp_pos_offset = 0
    current_gp_neg_offset = 0

    vocab_counters: Dict[str, Counter[Any]] = {f: Counter() for f in FEATURE_FIELDS}
    label_stats: Dict[str, Counter] = {
        "formality": Counter(),
        "gender_prag": Counter(),
        "register": Counter(),
        "grammatic": Counter(),
    }
    reg_samples: Dict[str, List[Any]] = {}

    for row in batch:
        # Unpack row from `corpus` table (including grammar columns).
        sentence, formality, gender, grammatic, r_ids_str, grammar, grammar_negative = (
            row
        )

        # Always re-parse to get the latest feature extraction logic.
        kotogram_obj = parser.japanese_to_kotogram(
            sentence, fmt=KotogramFormat.TRAINING_MASK
        )

        # Tokenize and collect feature statistics.
        tokens = split_kotogram(kotogram_obj)
        for token in tokens:
            token_feat = extract_token_features(token)
            # Get vocab-ready strings using centralized function
            vocab_strings = get_vocab_strings(token_feat)
            for field in FEATURE_FIELDS:
                vocab_counters[field][vocab_strings[field]] += 1

        # Use DB values for labels if present, otherwise set to NaN/Unpragmatic.
        # This trusts that the DB content was generated by a valid analysis process.
        if formality is not None:
            f_val = float(formality)
            f_prag = 1
            # Reconstruct FormalityLevel for stats tracking.
            # Ideally we would store the Enum ID in the DB, but legacy format is float.
            # Using cutoffs that match `formality_to_weight` logic.
            if f_val >= 0.75:
                f_enum = FormalityLevel.VERY_FORMAL
            elif f_val >= 0.25:
                f_enum = FormalityLevel.FORMAL
            elif f_val >= -0.25:
                f_enum = FormalityLevel.NEUTRAL
            elif f_val >= -0.75:
                f_enum = FormalityLevel.CASUAL
            else:
                f_enum = FormalityLevel.VERY_CASUAL
            f_id = FORMALITY_LABEL_TO_ID[f_enum]
        else:
            f_val = float("nan")
            f_prag = 0
            f_id = FORMALITY_LABEL_TO_ID[FormalityLevel.UNPRAGMATIC_FORMALITY]

        # Gender handling from DB columns.
        if gender is not None:
            g_val = float(gender)
            g_prag = 1
        else:
            g_val = float("nan")
            g_prag = 0

        # Grammaticality flag from DB.
        final_gram = int(grammatic)  # Typically 0 or 1.

        # Validation: Grammatical sentences must have valid gender AND formality values
        if final_gram == 1:
            if math.isnan(f_val) or math.isnan(g_val):
                raise ValueError(
                    f"Sentence marked as grammatic=1 but has null gender or formality:\n"
                    f"  Sentence: {sentence}\n"
                    f"  Gender: {gender} (g_val={g_val})\n"
                    f"  Formality: {formality} (f_val={f_val})\n"
                    f"Grammatical sentences must have valid gender AND formality values."
                )

        # Register IDs handling (stored as comma-separated string in DB).
        register_ids = []
        if r_ids_str:
            for s in r_ids_str.split(","):
                if s:
                    register_ids.append(int(s))

        if not register_ids:
            register_ids = [REGISTER_LABEL_TO_ID[RegisterLevel.NEUTRAL]]

        # Append to buffers.
        sentences_buf.append(sentence)
        kotograms_buf.append(kotogram_obj)
        f_val_buf.append(f_val)
        f_prag_buf.append(f_prag)
        g_val_buf.append(g_val)
        g_prag_buf.append(g_prag)
        gram_buf.append(final_gram)

        reg_ids_buf.extend(register_ids)
        current_reg_offset += len(register_ids)
        reg_offsets_buf.append(current_reg_offset)

        # Grammar point IDs (format: "gp0597,gp0123" → [597, 123])
        gp_pos_ids = parse_gp_ids(grammar if grammar else "")
        gp_neg_ids = parse_gp_ids(grammar_negative if grammar_negative else "")

        gp_pos_ids_buf.extend(gp_pos_ids)
        current_gp_pos_offset += len(gp_pos_ids)
        gp_pos_offsets_buf.append(current_gp_pos_offset)

        gp_neg_ids_buf.extend(gp_neg_ids)
        current_gp_neg_offset += len(gp_neg_ids)
        gp_neg_offsets_buf.append(current_gp_neg_offset)

        # Update stats.
        if f_prag:
            label_stats["formality"][f_id] += (
                1  # Only count pragmatic samples for distribution.
            )
        label_stats["gender_prag"][g_prag] += 1
        label_stats["grammatic"][final_gram] += 1
        for reg_id in register_ids:
            label_stats["register"][reg_id] += 1

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
        "gp_pos_ids": gp_pos_ids_buf,
        "gp_pos_offsets": gp_pos_offsets_buf,
        "gp_neg_ids": gp_neg_ids_buf,
        "gp_neg_offsets": gp_neg_offsets_buf,
        "vocab": vocab_counters,
        "stats": label_stats,
        "reg_samples": reg_samples,
    }


def _encode_shard_phase2(worker_id: int) -> None:
    """
    Phase 2: Read kotograms, encode features/KC, and write binaries.

    This function runs in Phase 2 workers. It:
    1. Reads the intermediate `kotograms` text file written in Phase 1.
    2. Re-tokenizes the kotograms to extract features.
    3. Encodes features using the *final* tokenizer (which has global vocabulary).
    4. Computes Knowledge Component (KC) targets (e.g. n-grams) based on feature IDs.
    5. Writes the final binary arrays for training features and KC targets.

    Args:
        worker_id: The worker ID, used to locate the shard files.
    """
    shard_prefix = os.path.join(_SHARD_DIR, f"shard_{worker_id}")
    koto_path = f"{shard_prefix}.{EXT_KOTOGRAMS}"

    if not os.path.exists(koto_path):
        return

    # Buffers for encoded feature IDs (one array per feature field).
    feat_buffers: Dict[str, List[int]] = {f: [] for f in FEATURE_FIELDS}
    token_lengths_buf: List[int] = []
    offsets_buf: List[int] = [0]
    current_offset = 0

    # Buffer for KC targets (keyed by ID)
    kc_buffers: Dict[KcFamilyId, Dict[str, Any]] = {}

    # Track unique KC IDs per family for collision detection (amortized to label phase)
    kc_unique_ids: Dict[KcFamilyId, Set[int]] = {}

    with open(koto_path, "r", encoding="utf-8") as f:
        for line in f:
            kotogram_obj = line.strip()
            if not kotogram_obj:
                continue

            tokens = split_kotogram(kotogram_obj)

            token_lengths_buf.append(len(tokens))
            current_offset += len(tokens)
            offsets_buf.append(current_offset)

            # Extract features and encode with tokenizer.
            # Note: _TOKENIZER must be loaded with the full vocabulary at this point.
            feat_ids_map: Dict[str, List[int]] = {f: [] for f in FEATURE_FIELDS}
            for token in tokens:
                token_feat = extract_token_features(token)
                # Get vocab-ready strings using centralized function (same as Phase 1)
                vocab_strings = get_vocab_strings(token_feat)
                for field in FEATURE_FIELDS:
                    val = vocab_strings[field]

                    if _TOKENIZER:
                        fid = _TOKENIZER.get_id(field, val)
                        feat_buffers[field].append(fid)
                        feat_ids_map[field].append(fid)

            # Compute KC targets based on feature IDs.
            # Compute KC targets based on feature IDs.
            kc_targets = compute_kc_targets(cast(Any, feat_ids_map))
            for k_key, vals in kc_targets.items():
                if k_key not in kc_buffers:
                    kc_buffers[k_key] = {"ids": [], "offsets": [0], "cur_off": 0}

                # KC targets can be variable length per sentence (e.g. n-grams).
                # We store them as a jagged array (flat values + offsets).
                ids = vals
                if isinstance(ids, list):
                    # For list-type targets (like n-grams), extend the flat list.
                    # Otherwise (scalar targets), we would append single values (not handled here).
                    cast(List[int], kc_buffers[k_key]["ids"]).extend(ids)
                    kc_buffers[k_key]["cur_off"] = cast(
                        int, kc_buffers[k_key]["cur_off"]
                    ) + len(ids)
                    cast(List[int], kc_buffers[k_key]["offsets"]).append(
                        cast(int, kc_buffers[k_key]["cur_off"])
                    )

                    # Track unique IDs for collision detection (amortized)
                    if k_key not in kc_unique_ids:
                        kc_unique_ids[k_key] = set()
                    kc_unique_ids[k_key].update(ids)

    # Write the main sentence offsets (token boundaries).
    write_int_array(f"{shard_prefix}.{EXT_OFFSETS}", offsets_buf, "i")

    for field in FEATURE_FIELDS:
        # Write each feature field as a flat binary array used for embedding lookup.
        f_path = f"{shard_prefix}.{EXT_FEAT_PREFIX}{field}.bin"
        write_int_array(f_path, feat_buffers[field], "i")

    # Write KC target binaries.
    # Write KC target binaries.

    # Actually: I need to update the logic above to ensure kc_buffers uses string keys OR handle conversion here.
    # The simplest is to convert to string (value) for file writing.

    for k_key_obj, accum in kc_buffers.items():
        # k_key_obj is KcFamilyId
        k_key_str = (
            k_key_obj.value if isinstance(k_key_obj, KcFamilyId) else str(k_key_obj)
        )

        # Write the IDs and the Offsets for jagged access during training.
        write_int_array(
            f"{shard_prefix}.{EXT_KC_PREFIX}{k_key_str}_ids.bin", accum["ids"], "i"
        )
        write_int_array(
            f"{shard_prefix}.{EXT_KC_PREFIX}{k_key_str}_{EXT_OFFSETS}",
            accum["offsets"],
            "i",
        )

    # Write unique ID counts per KC family (for collision detection at training time)
    import json

    unique_counts = {
        (k.value if isinstance(k, KcFamilyId) else str(k)): len(v)
        for k, v in kc_unique_ids.items()
    }
    with open(f"{shard_prefix}.kc_unique_counts.json", "w", encoding="utf-8") as jf:
        json.dump(unique_counts, jf)


def print_stats(label_stats: Dict[str, Counter]) -> None:
    """Print attractive statistics about the labeling results."""
    if not label_stats:
        return

    def _print_dist(
        title: str,
        style: str,
        counts: Counter,
        map_func: Any,
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

    _print_dist(
        "Formality Distribution",
        "bold magenta",
        label_stats["formality"],
        lambda x: FORMALITY_ID_TO_LABEL[x].value,
    )

    _print_dist(
        "Gender Pragmatic Distribution",
        "bold cyan",
        label_stats["gender_prag"],
        lambda x: {1: "Pragmatic", 0: "Unpragmatic"}[x],
    )

    _print_dist(
        "Register Distribution",
        "bold yellow",
        label_stats["register"],
        lambda x: REGISTER_ID_TO_LABEL[x].value,
    )

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
    if overrides is None:
        overrides = {}
    init_worker(wid, s_dir, overrides, {})
    b_size = 2000

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

        cast(List[str], buffers["sentences"]).extend(res["sentences"])
        cast(List[str], buffers["kotograms"]).extend(res["kotograms"])
        cast(List[float], buffers["f_val"]).extend(res["f_val"])
        cast(List[int], buffers["f_prag"]).extend(res["f_prag"])
        cast(List[float], buffers["g_val"]).extend(res["g_val"])
        cast(List[int], buffers["g_prag"]).extend(res["g_prag"])
        cast(List[int], buffers["gram"]).extend(res["gram"])
        cast(List[int], buffers["reg_ids"]).extend(res["reg_ids"])

        shifted = [o + total_reg_ids_so_far for o in res["reg_offsets"][1:]]
        cast(List[int], buffers["reg_offsets"]).extend(shifted)
        total_reg_ids_so_far += len(res["reg_ids"])

        for f in FEATURE_FIELDS:
            cast(Dict[str, Counter], buffers["vocab"])[f].update(res["vocab"][f])
        for k in cast(Dict[str, Counter], buffers["stats"]):
            cast(Dict[str, Counter], buffers["stats"])[k].update(res["stats"][k])

        reg_samples_buf = cast(Dict[str, List[Any]], buffers["reg_samples"])
        for rid, samps in res["reg_samples"].items():
            if rid not in reg_samples_buf:
                reg_samples_buf[rid] = []
            reg_samples_buf[rid].extend(samps)

    _write_shard_data(wid, s_dir, buffers)

    result_queue_arg.put((buffers["vocab"], buffers["stats"], buffers["reg_samples"]))


def worker_p1_db_wrapper(
    wid: int,
    chunk: List[Tuple[Any, ...]],
    s_dir: str,
    result_queue_arg: Any,
) -> None:
    """Worker wrapper for DB source (skips overrides as DB has golden labels)."""

    init_worker(wid, s_dir, {}, {})
    b_size = 2000

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
        "gp_pos_ids": [],
        "gp_pos_offsets": [0],
        "gp_neg_ids": [],
        "gp_neg_offsets": [0],
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
    total_gp_pos_ids_so_far = 0
    total_gp_neg_ids_so_far = 0

    for i in range(0, len(chunk), b_size):
        batch = chunk[i : i + b_size]

        res = analyze_batch_from_db(batch)

        cast(List[str], buffers["sentences"]).extend(res["sentences"])
        cast(List[str], buffers["kotograms"]).extend(res["kotograms"])
        cast(List[float], buffers["f_val"]).extend(res["f_val"])
        cast(List[int], buffers["f_prag"]).extend(res["f_prag"])
        cast(List[float], buffers["g_val"]).extend(res["g_val"])
        cast(List[int], buffers["g_prag"]).extend(res["g_prag"])
        cast(List[int], buffers["gram"]).extend(res["gram"])
        cast(List[int], buffers["reg_ids"]).extend(res["reg_ids"])

        shifted = [o + total_reg_ids_so_far for o in res["reg_offsets"][1:]]
        cast(List[int], buffers["reg_offsets"]).extend(shifted)
        total_reg_ids_so_far += len(res["reg_ids"])

        # Grammar point aggregation
        cast(List[int], buffers["gp_pos_ids"]).extend(res["gp_pos_ids"])
        shifted_gp_pos = [
            o + total_gp_pos_ids_so_far for o in res["gp_pos_offsets"][1:]
        ]
        cast(List[int], buffers["gp_pos_offsets"]).extend(shifted_gp_pos)
        total_gp_pos_ids_so_far += len(res["gp_pos_ids"])

        cast(List[int], buffers["gp_neg_ids"]).extend(res["gp_neg_ids"])
        shifted_gp_neg = [
            o + total_gp_neg_ids_so_far for o in res["gp_neg_offsets"][1:]
        ]
        cast(List[int], buffers["gp_neg_offsets"]).extend(shifted_gp_neg)
        total_gp_neg_ids_so_far += len(res["gp_neg_ids"])

        for f in FEATURE_FIELDS:
            cast(Dict[str, Counter], buffers["vocab"])[f].update(res["vocab"][f])
        for k in cast(Dict[str, Counter], buffers["stats"]):
            cast(Dict[str, Counter], buffers["stats"])[k].update(res["stats"][k])

        reg_samples_buf = cast(Dict[str, List[Any]], buffers["reg_samples"])
        for rid, samps in res["reg_samples"].items():
            if rid not in reg_samples_buf:
                reg_samples_buf[rid] = []
            reg_samples_buf[rid].extend(samps)

    _write_shard_data(wid, s_dir, buffers)

    result_queue_arg.put((buffers["vocab"], buffers["stats"], buffers["reg_samples"]))


def _write_shard_data(wid: int, s_dir: str, buffers: Dict[str, Any]) -> None:
    """Helper to write shard data from buffers."""
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
    )
    write_int_array(
        f"{shard_prefix}.{EXT_LABELS}_f_prag",
        cast(List[int], buffers["f_prag"]),
        "B",
    )
    write_float_array(
        f"{shard_prefix}.{EXT_LABELS}_g_val",
        cast(List[float], buffers["g_val"]),
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

    # Grammar point shard files (jagged arrays)
    # Only write if grammar data exists (DB source)
    if "gp_pos_ids" in buffers:
        write_int_array(
            f"{shard_prefix}.gp_pos_ids.bin",
            cast(List[int], buffers["gp_pos_ids"]),
            "i",
        )
        write_int_array(
            f"{shard_prefix}.gp_pos_{EXT_OFFSETS}",
            cast(List[int], buffers["gp_pos_offsets"]),
            "i",
        )
        write_int_array(
            f"{shard_prefix}.gp_neg_ids.bin",
            cast(List[int], buffers["gp_neg_ids"]),
            "i",
        )
        write_int_array(
            f"{shard_prefix}.gp_neg_{EXT_OFFSETS}",
            cast(List[int], buffers["gp_neg_offsets"]),
            "i",
        )


def main() -> None:
    """
    Main entry point for the labeling script.

    Orchestrates the 3-phase pipeline:
    1. Setup: Parses args, initializes timers and directories.
    2. Phase 1 (Scanning & Analysis): Dispatches workers to process input data (DB or TSV)
       and build vocabulary.
    3. Phase 2 (Encoding): Re-spawns workers to encode feature IDs and KC targets using
       the finalized vocabulary from Phase 1.
    4. Phase 3 (Merging): Aggregates all sharded outputs into the final dataset directory.
    """

    parser = argparse.ArgumentParser(description="Label and cache Japanese sentences.")
    parser.add_argument(
        "--grammatic-pattern",
        type=str,
        help="Primary TSV data file(s) (glob pattern)",
    )
    parser.add_argument("--agrammatic-pattern", type=str, help="Agrammatic TSV pattern")
    parser.add_argument(
        "--source-db", type=str, help="Path to corpus.db (replaces file patterns)"
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument(
        "--num-workers", type=int, default=0, help="Number of workers (default: CPU-1)"
    )
    parser.add_argument(
        "--force-relabel",
        action="store_true",
        help="Wipe existing cache results before starting",
    )

    args = parser.parse_args()

    # validate arguments.
    if not args.source_db and not args.grammatic_pattern:
        parser.error("Must provide either --source-db or --grammatic-pattern")

    # standard cache directory setup.
    # pylint: disable=import-outside-toplevel
    from train import paths as train_paths

    cache_dir = train_paths.get_style_dataset_cache_dir()
    shard_dir = os.path.join(cache_dir, "shards")

    if args.force_relabel and os.path.exists(cache_dir):
        console.print(f"Cleaning existing cache: {cache_dir}")
        shutil.rmtree(cache_dir)

    profile_dir = get_profile_dir()
    if profile_dir:
        os.makedirs(profile_dir, exist_ok=True)
    timer = PhaseTimer(console, profile_dir)

    # ensure shard directory is clean before starting.
    if os.path.exists(shard_dir):
        shutil.rmtree(shard_dir)
    os.makedirs(shard_dir, exist_ok=True)

    if args.num_workers > 0:
        num_workers = args.num_workers
    else:
        num_workers = max(1, mp.cpu_count() - 1)

    all_rows: List[Any] = []
    gp_priors_vec: Optional[List[float]] = None

    if args.source_db:
        # DB PATH: fast loading of golden labels.
        console.print(f"Loading data from DB: {args.source_db}")

        # Validate that DB register table matches our code constants (source of truth)
        _validate_register_mapping_against_db(args.source_db)

        # Grammar-point priors (user-managed in corpus.db grammar table)
        gp_priors_vec = _load_gp_priors_from_db(args.source_db)
        console.print(
            f"[green]✓[/green] Loaded grammar point priors for indices 0..{len(gp_priors_vec) - 1}"
        )

        conn = sqlite3.connect(args.source_db)
        c = conn.cursor()
        # Fetch all sentences and their metadata.
        # This is memory-intensive but simple. For massive DBs, this might need chunking.
        # But currently optimized for <10M rows which fits in RAM.
        c.execute(
            "SELECT sentence, formality, gender, grammatic, register_ids, grammar, grammar_negative FROM corpus"
        )
        all_rows = c.fetchall()
        conn.close()
        console.print(f"Loaded {len(all_rows):,} rows from DB.")

    else:
        # FILE PATH: parsing raw TSV files.
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

        gram_patterns = [args.grammatic_pattern]
        console.print(f"Scanning data with {num_workers} workers...")

        rows = process_file_group(gram_patterns, 1)
        all_rows.extend(rows)
        if args.agrammatic_pattern:
            # Agrammatic sentences are explicitly labeled 0 for grammar task.
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

    # -------------------------------------------------------------------------
    # Phase 1: Analysis
    # -------------------------------------------------------------------------
    merged_counters: Dict[str, Counter] = {f: Counter() for f in FEATURE_FIELDS}
    merged_label_stats: Dict[str, Counter] = {
        "formality": Counter(),
        "gender_prag": Counter(),
        "register": Counter(),
        "grammatic": Counter(),
    }
    merged_reg_samples: Dict[str, List[Any]] = {}

    from scripts.rule_based_analysis import load_register_overrides

    # Load overrides mainly for file-based processing where heuristics are primary.
    register_overrides = load_register_overrides() if not args.source_db else {}

    # Use 'spawn' for safety with CUDA/torch (though not used here yet) and macOS.
    ctx = mp.get_context("spawn")

    with create_progress(console) as progress:
        task1 = progress.add_task("[green]Phase 1: Analyzing...", total=len(all_rows))

        # Split work into chunks for parallel processing.
        chunk_size = (len(all_rows) + num_workers - 1) // num_workers
        chunks = [
            all_rows[i : i + chunk_size] for i in range(0, len(all_rows), chunk_size)
        ]

        result_queue = ctx.Queue()
        procs = []

        for i, chunk in enumerate(chunks):
            if args.source_db:
                p = ctx.Process(
                    target=worker_p1_db_wrapper,
                    args=(i, chunk, shard_dir, result_queue),
                )
            else:
                p = ctx.Process(
                    target=worker_p1_wrapper,
                    args=(i, chunk, shard_dir, register_overrides, result_queue),
                )
            procs.append(p)
            p.start()

        # Collect results from workers as they finish.
        finished_count = 0
        while finished_count < len(procs):
            try:
                # Poll queue with timeout to allow checking process aliveness.
                res = result_queue.get(timeout=0.1)

                # Aggregate stats from this chunk.
                vc, ls, rs = res
                # Merge counters.
                for f in FEATURE_FIELDS:
                    merged_counters[f].update(vc[f])
                for k, counter in merged_label_stats.items():
                    counter.update(ls[k])
                for rid, samps in rs.items():
                    if rid not in merged_reg_samples:
                        merged_reg_samples[rid] = []
                    merged_reg_samples[rid].extend(samps)

                # Advance progress bar by number of items processed in this batch stats.
                count = sum(ls["grammatic"].values())
                progress.update(task1, advance=count)

                finished_count += 1

            except queue.Empty:
                # Check for dead workers if queue is empty.
                failed = False
                for p in procs:
                    if not p.is_alive() and p.exitcode != 0:
                        # Panic if any worker died unexpectedly (e.g. segfault, OOM).
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

    # Build final tokenizer from aggregated counters.
    vocab_file = "vocab.json"
    # pylint: disable=import-outside-toplevel
    dataset_cache_dir = train_paths.get_style_dataset_cache_dir()
    tokenizer = Tokenizer()
    _build_and_save_vocab(tokenizer, merged_counters, dataset_cache_dir)
    console.print(f"Saved vocab to {vocab_file}")

    # -------------------------------------------------------------------------
    # Phase 2: Encoding
    # -------------------------------------------------------------------------
    # Workers are respawned here because they need the *complete* tokenizer
    # which was only finalized after Phase 1 completed across all workers.

    console.print("Phase 2: Encoding Shards...")
    procs_p2 = []

    # Workers need the tokenizer state to encode features correctly.
    # Passing state dict avoids pickling large objects or race conditions.
    # Note: Using `tokenizer.field_vocabs` directly.

    for i in range(len(chunks)):  # One P2 worker per P1 shard.
        p = ctx.Process(
            target=worker_p2_wrapper,
            args=(i, shard_dir, {"field_vocabs": tokenizer.field_vocabs}),
        )
        procs_p2.append(p)
        p.start()

    for p in procs_p2:
        p.join()

    timer.mark("Phase 2: Encoding Complete")

    # -------------------------------------------------------------------------
    # Phase 3: Merging
    # -------------------------------------------------------------------------
    # Combine all sharded files into single large binary files.
    # This allows memory-mapping the entire dataset during training.

    console.print("Phase 3: Merging Shards...")

    # 1. Merge sentence offsets (indexing into the text files).
    # Essential for random access to sentences.
    console.print("  Merging sentence offsets...")
    merge_offset_shards(
        shard_dir,
        os.path.join(dataset_cache_dir, EXT_OFFSETS),
        len(chunks),
        "shard_{}." + EXT_OFFSETS,
    )

    # 2. Merge feature arrays.
    for field in FEATURE_FIELDS:
        console.print(f"  Merging feature: {field}...")
        merge_shards(
            shard_dir,
            os.path.join(dataset_cache_dir, f"{EXT_FEAT_PREFIX}{field}.bin"),
            len(chunks),
            "shard_{}." + f"{EXT_FEAT_PREFIX}{field}.bin",
        )

    # 3. Merge raw text files (sentences and kotograms) for debugging/inspection.
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
                    # Should not happen unless P2 failed silently.
                    pass

    # 4. Merge Label arrays.
    # Include both scalar labels (f_val, gram) and variable-length (register IDs).
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

    # Merge register offsets (because register IDs are a jagged array).
    console.print("  Merging register offsets...")
    merge_offset_shards(
        shard_dir,
        os.path.join(dataset_cache_dir, f"{EXT_LABELS}_reg_ids_{EXT_OFFSETS}"),
        len(chunks),
        "shard_{}." + f"{EXT_LABELS}_reg_ids_{EXT_OFFSETS}",
    )

    # 4b. Merge Grammar Point arrays (only if DB source was used)
    gp_pos_path = os.path.join(shard_dir, "shard_0.gp_pos_ids.bin")
    if os.path.exists(gp_pos_path):
        console.print("  Merging grammar point labels...")
        # Positive GP IDs
        merge_shards(
            shard_dir,
            os.path.join(dataset_cache_dir, "gp_pos_ids.bin"),
            len(chunks),
            "shard_{}.gp_pos_ids.bin",
        )
        merge_offset_shards(
            shard_dir,
            os.path.join(dataset_cache_dir, f"gp_pos_{EXT_OFFSETS}"),
            len(chunks),
            "shard_{}.gp_pos_" + EXT_OFFSETS,
        )
        # Negative GP IDs
        merge_shards(
            shard_dir,
            os.path.join(dataset_cache_dir, "gp_neg_ids.bin"),
            len(chunks),
            "shard_{}.gp_neg_ids.bin",
        )
        merge_offset_shards(
            shard_dir,
            os.path.join(dataset_cache_dir, f"gp_neg_{EXT_OFFSETS}"),
            len(chunks),
            "shard_{}.gp_neg_" + EXT_OFFSETS,
        )

        # Optional: write gp_priors.bin alongside gp_pos/gp_neg
        console.print("  Writing grammar point priors...")
        if gp_priors_vec is None:
            raise RuntimeError(
                "gp_priors_vec is missing unexpectedly. When using --source-db, priors must be loaded."
            )
        write_float_array(
            os.path.join(dataset_cache_dir, "gp_priors.bin"),
            gp_priors_vec,
        )

    # 5. Merge KC Targets.
    # Discover which KC targets were generated (based on config/computation in P2).
    kc_files = glob.glob(os.path.join(shard_dir, f"shard_0.{EXT_KC_PREFIX}*"))
    kc_keys = set()
    for kf in kc_files:
        # P2 generates keys dynamically, so we scan shard 0 to find them.
        # Filename format: shard_0.kc_TARGETNAME_ids.bin
        base = os.path.basename(kf)
        if "_ids.bin" in base:
            # Found an IDs file, extract the key.
            # examples: kc_ngram_ids.bin -> key="ngram"
            #           kc_dep_ids.bin   -> key="dep"

            # Remove prefix/suffix to get raw key.
            parts = base.split(".")
            # parts expected: ['shard_0', 'kc_KEY_ids', 'bin']
            # or simply rely on string splitting.

            # Robust parsing:
            mid = parts[1]
            # mid is "kc_KEY_ids"
            if mid.startswith(EXT_KC_PREFIX):
                key = mid[len(EXT_KC_PREFIX) :]  # remove "kc_"
            else:
                key = mid

            key = key.replace("_ids", "")
            kc_keys.add(key)

    for key in kc_keys:
        console.print(f"  Merging KC target: {key}...")
        # Merge ID array.
        suffix_ids = f"{EXT_KC_PREFIX}{key}_ids.bin"
        merge_shards(
            shard_dir,
            os.path.join(dataset_cache_dir, suffix_ids),
            len(chunks),
            "shard_{}." + suffix_ids,
        )
        # Merge Offset array.
        suffix_off = f"{EXT_KC_PREFIX}{key}_{EXT_OFFSETS}"
        merge_offset_shards(
            shard_dir,
            os.path.join(dataset_cache_dir, suffix_off),
            len(chunks),
            "shard_{}." + suffix_off,
        )

    timer.mark("Phase 3: Merging Complete")

    # 6. Merge unique KC ID counts from all shards
    # These are used at training time for collision detection (amortized from training)
    console.print("  Merging KC unique ID counts...")
    import json

    merged_unique_counts: Dict[str, int] = {}
    for shard_idx in range(len(chunks)):
        shard_json = os.path.join(shard_dir, f"shard_{shard_idx}.kc_unique_counts.json")
        if os.path.exists(shard_json):
            with open(shard_json, "r", encoding="utf-8") as jf:
                shard_counts = json.load(jf)
                for key, count in shard_counts.items():
                    # Note: Unique counts across shards need to be summed as an approximation
                    # (true unique count would require set union, but sum is a useful lower bound)
                    merged_unique_counts[key] = merged_unique_counts.get(key, 0) + count

    # Write merged counts to dataset directory
    with open(
        os.path.join(dataset_cache_dir, "kc_unique_counts.json"), "w", encoding="utf-8"
    ) as jf:
        json.dump(merged_unique_counts, jf, indent=2)

    # Cleanup temporary shards to save space.
    console.print("Cleaning up shards...")
    if os.path.exists(shard_dir):
        shutil.rmtree(shard_dir)

    console.print(
        f"[bold green]Labeling Complete![/bold green] Data saved to {dataset_cache_dir}"
    )


if __name__ == "__main__":
    main()
