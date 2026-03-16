#!/usr/bin/env python3
"""Round-trip test to verify kotogram format is lossless and orthogonal.

This script compares RAW sudachipy output against reconstructed raw output:
1. Parses sentences with sudachipy → captures raw token data (pos_tuple, etc.)
2. Converts to kotogram string via our parser
3. Parses kotogram string back → extracts TokenFeatures
4. Reconstructs raw sudachipy-like data from extracted features (reverse mapping)
5. Compares: original raw data == reconstructed raw data

This exposes any information loss in the kotogram format.
"""

import argparse
import os
import sqlite3
import sys
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Tuple

from sudachipy import SplitMode, dictionary

from kotogram.japanese_parser import (
    CONJUGATED_FORM_MAP,
    CONJUGATED_TYPE_MAP,
    POS1_MAP,
    POS2_MAP,
    POS3_MAP,
    POS_MAP,
)
from kotogram.kotogram import TokenFeatures, extract_token_features, split_kotogram
from kotogram.sudachi_japanese_parser import SudachiJapaneseParser


# Reverse mappings: our abbreviated format -> raw Sudachi format
def build_reverse_map(fwd_map: Dict[str, str]) -> Dict[str, str]:
    """Build reverse mapping from our abbreviation to raw Sudachi value."""
    return {v: k for k, v in fwd_map.items() if v}


REV_POS_MAP = build_reverse_map(POS_MAP)
REV_POS1_MAP = build_reverse_map(POS1_MAP)
REV_POS2_MAP = build_reverse_map(POS2_MAP)
REV_POS3_MAP = build_reverse_map(POS3_MAP)
REV_CONJUGATED_TYPE_MAP = build_reverse_map(CONJUGATED_TYPE_MAP)
REV_CONJUGATED_FORM_MAP = build_reverse_map(CONJUGATED_FORM_MAP)

# Process-local globals for worker caching (initialized in workers)
_worker_dict: Any = None  # pylint: disable=invalid-name
_worker_parser: Any = None  # pylint: disable=invalid-name
_worker_tokenizer: Any = None  # pylint: disable=invalid-name


def get_raw_sudachi_data(
    sudachi_token: Any,  # sudachipy.morpheme.Morpheme
) -> Dict[str, str]:
    """Extract raw data from a sudachipy token."""
    pos_tuple = sudachi_token.part_of_speech()
    return {
        "surface": sudachi_token.surface(),
        "pos_0": pos_tuple[0] if len(pos_tuple) > 0 else "*",
        "pos_1": pos_tuple[1] if len(pos_tuple) > 1 else "*",
        "pos_2": pos_tuple[2] if len(pos_tuple) > 2 else "*",
        "pos_3": pos_tuple[3] if len(pos_tuple) > 3 else "*",
        "pos_4": pos_tuple[4] if len(pos_tuple) > 4 else "*",  # conjugated_type
        "pos_5": pos_tuple[5] if len(pos_tuple) > 5 else "*",  # conjugated_form
        "lemma": sudachi_token.dictionary_form(),
        "reading": sudachi_token.reading_form(),
    }


def reconstruct_raw_from_features(features: "TokenFeatures") -> Dict[str, str]:
    """Reconstruct raw sudachi-like data from extracted TokenFeatures."""

    def rev_lookup(rev_map: dict, value: str) -> str:
        if not value:
            return "*"
        return rev_map.get(value, value)

    return {
        "surface": features.surface,
        "pos_0": rev_lookup(REV_POS_MAP, features.pos),
        "pos_1": rev_lookup(REV_POS1_MAP, features.pos_detail_1),
        "pos_2": rev_lookup(REV_POS2_MAP, features.pos_detail_2),
        "pos_3": rev_lookup(REV_POS3_MAP, features.pos_detail_3),
        "pos_4": rev_lookup(REV_CONJUGATED_TYPE_MAP, features.conjugated_type),
        "pos_5": rev_lookup(REV_CONJUGATED_FORM_MAP, features.conjugated_form),
        "lemma": features.lemma,
        "reading": features.reading,
    }


def compare_raw_data(
    original: Dict[str, str], reconstructed: Dict[str, str]
) -> List[str]:
    """Compare raw sudachi data and return list of differences."""
    differences = []
    for field in original:
        orig_val = original[field]
        recon_val = reconstructed.get(field, "")
        if orig_val != recon_val:
            differences.append(
                f"  {field}: original={orig_val!r} != reconstructed={recon_val!r}"
            )
    return differences


def _check_sentence_worker(sentence: str) -> Optional[str]:
    """Worker function for parallel testing. Returns None on success, error message on failure.

    Uses process-local global caching to avoid re-creating Sudachi Dictionary for each sentence.
    """
    # Process-local globals for performance (initialized once per worker process)
    global _worker_dict, _worker_parser, _worker_tokenizer  # pylint: disable=global-statement
    if "_worker_dict" not in globals() or _worker_dict is None:
        _worker_dict = dictionary.Dictionary(dict="core")
        _worker_parser = SudachiJapaneseParser()
        _worker_tokenizer = _worker_dict.create(mode=SplitMode.C)

    sudachi_tokens = list(_worker_tokenizer.tokenize(sentence))
    kotogram_string = _worker_parser.japanese_to_kotogram(sentence)
    kotogram_tokens = split_kotogram(kotogram_string)

    if len(sudachi_tokens) != len(kotogram_tokens):
        return f"Token count mismatch for '{sentence}': {len(sudachi_tokens)} vs {len(kotogram_tokens)}"

    for i, (sudachi_tok, kotogram_token) in enumerate(
        zip(sudachi_tokens, kotogram_tokens)
    ):
        original_raw = get_raw_sudachi_data(sudachi_tok)
        extracted_features = extract_token_features(kotogram_token)
        reconstructed_raw = reconstruct_raw_from_features(extracted_features)
        differences = compare_raw_data(original_raw, reconstructed_raw)

        if differences:
            return (
                f"FAILURE for sentence: {sentence}\n"
                f"Kotogram: {kotogram_string}\n"
                f"Token [{i}]: {kotogram_token}\n"
                f"Original raw: {original_raw}\n"
                f"Reconstructed: {reconstructed_raw}\n"
                f"Differences:\n" + "\n".join(differences)
            )
    return None  # Success


def load_corpus_sentences(db_path: str) -> List[str]:
    """Load all sentences from corpus.db."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sentence FROM corpus")
    sentences = [row[0] for row in cursor.fetchall()]
    conn.close()
    return sentences


def run_parallel_tests(
    sentences: List[str], num_workers: int
) -> Tuple[int, int, Optional[str]]:
    """Run tests in parallel. Returns (passed, failed, first_failure_sentence)."""
    passed = 0
    failed = 0
    first_failure_sentence: Optional[str] = None

    with Pool(num_workers) as pool:
        # Use imap for ordered results to find first failure
        for idx, result in enumerate(pool.imap(_check_sentence_worker, sentences)):
            if result is None:
                passed += 1
                if idx % 1000 == 0:
                    print(f"Progress: {idx}/{len(sentences)} sentences tested...")
            else:
                failed += 1
                if first_failure_sentence is None:
                    first_failure_sentence = sentences[idx]
                    print(f"\n=== FIRST FAILURE ===\n{result}")
                    pool.terminate()
                    break

    return passed, failed, first_failure_sentence


def run_tests(sentences: List[str], fail_fast: bool = True) -> Tuple[int, int]:
    """Run round-trip tests (single-threaded for default sentences)."""
    passed = 0
    failed = 0

    for sentence in sentences:
        result = _check_sentence_worker(sentence)
        if result is None:
            passed += 1
        else:
            failed += 1
            print(f"\n{result}")
            if fail_fast:
                print("\nStopped after first failure (--fail-fast)")
                return passed, failed

    return passed, failed


def main() -> int:
    arg_parser = argparse.ArgumentParser(
        description="Test kotogram round-trip is lossless"
    )
    arg_parser.add_argument(
        "--sentence", "-s", type=str, help="Single sentence to test"
    )
    arg_parser.add_argument(
        "--all",
        action="store_true",
        help="Test all sentences in corpus.db (parallelized)",
    )
    arg_parser.add_argument(
        "--no-fail-fast", action="store_true", help="Continue after failures"
    )
    arg_parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=cpu_count(),
        help="Number of parallel workers",
    )
    args = arg_parser.parse_args()

    # Default test sentences covering various token types
    default_sentences = [
        "食べる",  # Simple verb
        "食べます",  # Verb + aux-masu
        "学生です",  # Noun + aux-desu
        "学生だ",  # Noun + aux-da
        "猫を食べる",  # Noun + particle + verb
        "私は学生です",  # Pronoun + particle + noun + aux
        "高い",  # Adjective
        "走れ",  # Imperative verb
        "君が走れ",  # With conjugated form
        "ブラジル",  # Proper noun with reading='*' (reading == surface)
        "あそこでは多くのタイプの食品や雑貨を売ている。",  # Okurigana abbreviation (attributive)
        "1+2-3*4/6=1は「1足す2引く3かける4割る6は1」とと読む。",  # Literal '*' (not compression)
        "しかし水とは濡れているものだと教わても水についてほとんどわからないのと同様に、そんなことを言っても何も語っていることにはならないのである。",  # Okurigana abbr (continuative)
        "そのあと君はどしたたの？",  # aux-dosu (Kyoto dialect)
        "マユコはつきあて愉快な子だ。",  # classical-lower-nidan-ta
        "彼は長い事わずらている。",  # invariable conjugation type
    ]

    if args.sentence:
        sentences = [args.sentence]
        fail_fast = not args.no_fail_fast
        print("Testing single sentence...")
        passed, failed = run_tests(sentences, fail_fast=fail_fast)
    elif args.all:
        # Load sentences from corpus.db
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        db_path = os.path.join(project_root, "data", "corpus.db")
        if not os.path.exists(db_path):
            print(f"Error: corpus.db not found at {db_path}")
            return 1

        sentences = load_corpus_sentences(db_path)
        print(f"Loaded {len(sentences)} sentences from corpus.db")
        print(f"Running with {args.workers} workers...")

        passed, failed, first_failure = run_parallel_tests(sentences, args.workers)

        if first_failure:
            print("\n=== ADD THIS SENTENCE TO DEFAULT TESTS ===")
            print(f'        "{first_failure}",  # Corpus failure')
    else:
        sentences = default_sentences
        fail_fast = not args.no_fail_fast
        print(f"Testing {len(sentences)} sentence(s)...")
        passed, failed = run_tests(sentences, fail_fast=fail_fast)

    print("\n=== Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
