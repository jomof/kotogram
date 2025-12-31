#!/usr/bin/env python3
"""Script to validate all Tatoeba sentences and find unmapped features.

Usage:
    python scripts/validate_tatoeba.py [count]

    count: Number of sentences to validate (default: 100, use 'all' for all sentences)

Examples:
    python scripts/validate_tatoeba.py 100          # Validate first 100 sentences
    python scripts/validate_tatoeba.py all          # Validate all sentences
"""

import json
import os
import subprocess
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from kotogram import SudachiJapaneseParser, extract_token_features
from kotogram.exceptions import MissingMappingError
from kotogram.kotogram import split_kotogram
from scripts import (
    _setup_path,  # type: ignore # noqa: F401 # pylint: disable=import-private-name
)
from train.tsv import parse_tsv

_vulture_marker = _setup_path  # Vulture: Used for side effects


def validate_sentences(
    parser: SudachiJapaneseParser,
    parser_name: str,
    tsv_file: str,
    max_sentences: Optional[int] = None,
) -> Tuple[Dict[str, Set[str]], List[Dict[str, str]], List[str]]:
    # pylint: disable=too-many-locals
    """Validate sentences and collect unmapped features.

    Args:
        parser: Parser instance (SudachiJapaneseParser)
        parser_name: Name of the parser for display purposes
        tsv_file: Path to the Tatoeba TSV file
        max_sentences: Maximum number of sentences to process (None for all)

    Returns:
        Tuple of (unmapped_features dict, failed_sentences list, kotograms list)
    """
    unmapped_features: Dict[str, Set[str]] = defaultdict(set)
    failed_sentences: List[Dict[str, str]] = []
    successful_count = 0
    kotograms: List[str] = []

    with open(tsv_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_sentences and i >= max_sentences:
                break

            text = parse_tsv(line)

            try:
                kotogram = parser.japanese_to_kotogram(text)
                kotograms.append(kotogram)
                successful_count += 1
            except MissingMappingError as e:
                unmapped_features[e.map_name].add(e.key)
                failed_sentences.append(
                    {
                        "text": text,
                        "map": e.map_name,
                        "key": e.key,
                        "error": str(e),
                    }
                )

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"{parser_name.upper()} VALIDATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {len(failed_sentences)}")
    print()

    if unmapped_features:
        print("UNMAPPED FEATURES BY MAP:")
        print(f"{'-' * 80}")
        for map_name, keys in sorted(unmapped_features.items()):
            print(f"\n{map_name}: {len(keys)} unmapped keys")
            for key in sorted(keys):
                print(f"  '{key}'")

        print(f"\n{'=' * 80}")
        print("FIRST 10 FAILED SENTENCES:")
        print(f"{'=' * 80}")
        for failure in failed_sentences[:10]:
            print(f"\nText: {failure['text']}")
            print(f"Map: {failure['map']}, Key: '{failure['key']}'")
    else:
        print(f"✅ All sentences validated successfully with {parser_name}!")

    return unmapped_features, failed_sentences, kotograms


def compare_token_features(
    kotograms: List[str], project_root: str
) -> List[Dict[str, Any]]:
    # pylint: disable=too-many-locals
    """Compare Python and TypeScript extract_token_features results.

    Args:
        kotograms: List of kotogram strings to validate
        project_root: Path to project root for calling Node.js script

    Returns:
        List of mismatch dictionaries with details
    """
    mismatches: List[Dict[str, Any]] = []

    # Collect all tokens from all kotograms
    all_tokens: List[str] = []
    token_to_kotogram: List[int] = []  # Track which kotogram each token came from

    for idx, kotogram in enumerate(kotograms):
        tokens = split_kotogram(kotogram)
        for token in tokens:
            all_tokens.append(token)
            token_to_kotogram.append(idx)

    if not all_tokens:
        return mismatches

    # Call TypeScript in batches to avoid command line limits
    batch_size = 1000
    ts_results: List[Dict[str, str]] = []

    for batch_start in range(0, len(all_tokens), batch_size):
        batch_end = min(batch_start + batch_size, len(all_tokens))
        batch_tokens = all_tokens[batch_start:batch_end]

        # Call Node.js script with tokens as JSON via stdin
        script_path = os.path.join(project_root, "scripts", "extract_features_ts.mjs")
        result = subprocess.run(
            ["node", script_path],
            input=json.dumps(batch_tokens),
            capture_output=True,
            text=True,
            cwd=project_root,
            check=False,
        )

        if result.returncode != 0:
            print(f"TypeScript script failed: {result.stderr}")
            return mismatches

        batch_results = json.loads(result.stdout)
        ts_results.extend(batch_results)

    # Compare Python vs TypeScript for each token
    # Map Python field names to TypeScript field names
    field_mapping = {
        "surface": "surface",
        "pos": "pos",
        "pos_detail1": "posDetail1",
        "pos_detail2": "posDetail2",
        "pos_detail3": "posDetail3",
        "conjugated_type": "conjugatedType",
        "conjugated_form": "conjugatedForm",
        "base_orth": "baseOrth",
        "lemma": "lemma",
        "reading": "reading",
    }

    for i, token in enumerate(all_tokens):
        py_features = extract_token_features(token)
        ts_features = ts_results[i]

        # Compare all fields
        for py_field, ts_field in field_mapping.items():
            py_val = getattr(py_features, py_field)
            ts_val = ts_features.get(ts_field, "")

            if py_val != ts_val:
                mismatches.append(
                    {
                        "token": token,
                        "field": py_field,
                        "python_value": py_val,
                        "typescript_value": ts_val,
                        "kotogram_idx": token_to_kotogram[i],
                    }
                )

    return mismatches


def main() -> None:
    """Main validation function."""

    # Use path relative to script location (scripts/) -> data/ is in sibling or parent?
    # data is in project_root/data
    # script_dir is project/scripts
    # project_root is project
    # Assumes run from project root or scripts dir

    # Try to find data dir relative to current script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    tsv_file = os.path.join(project_root, "data", "jpn_sentences.tsv")

    # Parse command line arguments
    max_sentences: Optional[int] = 100  # Default

    if len(sys.argv) > 1:
        if sys.argv[1] == "all":
            max_sentences = None
        else:
            if not sys.argv[1].isdigit():
                print("Usage: python scripts/validate_tatoeba.py [count]")
                sys.exit(1)
            max_sentences = int(sys.argv[1])

    print(
        f"Validating {'all' if max_sentences is None else max_sentences} sentences from {tsv_file}"
    )
    print("This may take a while...\n")

    parser = SudachiJapaneseParser(dict_type="full", validate=True)
    print(f"\n{'#' * 80}")
    print("# VALIDATING WITH SUDACHI")
    print(f"{'#' * 80}")
    _, _, kotograms = validate_sentences(parser, "Sudachi", tsv_file, max_sentences)

    # Cross-language validation
    print(f"\n{'#' * 80}")
    print("# CROSS-LANGUAGE VALIDATION (Python vs TypeScript)")
    print(f"{'#' * 80}")

    mismatches = compare_token_features(kotograms, project_root)

    if mismatches:
        print(f"\n❌ Found {len(mismatches)} mismatches between Python and TypeScript!")
        print(f"\n{'=' * 80}")
        print("FIRST 20 MISMATCHES:")
        print(f"{'=' * 80}")
        for m in mismatches[:20]:
            print(f"\nToken: {m['token'][:60]}...")
            print(f"Field: {m['field']}")
            print(f"  Python:     '{m['python_value']}'")
            print(f"  TypeScript: '{m['typescript_value']}'")
    else:
        total_tokens = sum(len(split_kotogram(k)) for k in kotograms)
        print(f"\n✅ All {total_tokens} tokens match between Python and TypeScript!")


if __name__ == "__main__":
    main()
