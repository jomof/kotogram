#!/usr/bin/env python3
"""Automatically suggest high-value patterns from missed agrammatic sentences.

This script analyzes missed sentences and suggests patterns that:
1. Would detect many agrammatic sentences
2. Have zero false positives on grammatical sentences

Usage:
    python scripts/suggest_patterns.py
    python scripts/suggest_patterns.py --min-detections 50
"""
import argparse
import hashlib
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
from kotogram.kotogram import split_kotogram, extract_token_features
from kotogram.analysis import grammaticality

# Cache directory
CACHE_DIR = Path(".cache/kotogram")


def get_cache_path(data_file: str, cache_type: str) -> Path:
    """Get cache file path for a data file."""
    stat = os.stat(data_file)
    cache_key = f"{data_file}_{stat.st_mtime}_{stat.st_size}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{cache_type}_{cache_hash}.pkl"


def load_cached_features(data_file: str, cache_type: str) -> Optional[List[Tuple[str, List[Dict]]]]:
    """Load cached features if available."""
    cache_path = get_cache_path(data_file, cache_type)
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    return None


def save_cached_features(data_file: str, cache_type: str, features: List[Tuple[str, List[Dict]]]):
    """Save features to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = get_cache_path(data_file, cache_type)
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(features, f)
    except Exception:
        pass


def parse_sentences_with_cache(
    sentences: List[str],
    parser: SudachiJapaneseParser,
    data_file: str,
    cache_type: str,
    verbose: bool = True
) -> List[Tuple[str, List[Dict]]]:
    """Parse sentences and cache the results."""
    cached = load_cached_features(data_file, cache_type)
    if cached is not None:
        if verbose:
            print(f"  (loaded {len(cached)} cached parses)")
        return cached

    if verbose:
        print(f"  (parsing {len(sentences)} sentences...)")

    results = []
    for sentence in sentences:
        kotogram = parser.japanese_to_kotogram(sentence)
        tokens = split_kotogram(kotogram)
        feats = [extract_token_features(t) for t in tokens]
        results.append((sentence, feats))

    save_cached_features(data_file, cache_type, results)
    if verbose:
        print(f"  (cached parses for next run)")

    return results


def get_pattern_key(curr: Dict, nxt: Dict, nxt2: Optional[Dict] = None) -> Tuple[str, ...]:
    """Generate a pattern key from token features.

    Only creates patterns for likely error-indicating combinations:
    - Verbs/adjectives with specific conjugation forms
    - Auxiliary verbs
    - Particles with specific surfaces
    """
    # Skip generic noun/prt combinations (too common)
    curr_pos = curr.get('pos', '')
    nxt_pos = nxt.get('pos', '')

    # Only consider patterns involving verbs, adjectives, or auxiliaries
    interesting_pos = {'v', 'adj', 'auxv', 'shp'}
    if curr_pos not in interesting_pos and nxt_pos not in interesting_pos:
        return ()  # Skip generic patterns

    key = (
        curr_pos,
        curr.get('conjugated_type', ''),
        curr.get('conjugated_form', ''),
        nxt_pos,
        nxt.get('conjugated_type', ''),
        nxt.get('conjugated_form', ''),
        nxt.get('surface', '') if nxt_pos in ('prt', 'auxs') else '',
    )
    return key


def pattern_to_code(key: Tuple[str, ...]) -> str:
    """Convert pattern key to executable Python code."""
    conditions = []

    # curr conditions
    if key[0]:
        conditions.append(f"curr.get('pos')=={key[0]!r}")
    if key[1]:
        conditions.append(f"curr.get('conjugated_type')=={key[1]!r}")
    if key[2]:
        conditions.append(f"curr.get('conjugated_form')=={key[2]!r}")

    # nxt conditions
    if key[3]:
        conditions.append(f"nxt.get('pos')=={key[3]!r}")
    if key[4]:
        conditions.append(f"nxt.get('conjugated_type')=={key[4]!r}")
    if key[5]:
        conditions.append(f"nxt.get('conjugated_form')=={key[5]!r}")
    if key[6]:
        conditions.append(f"nxt.get('surface')=={key[6]!r}")

    return " and ".join(conditions)


def pattern_to_description(key: Tuple[str, ...]) -> str:
    """Generate human-readable description of pattern."""
    parts = []
    if key[0]:
        parts.append(f"{key[0]}")
    if key[1]:
        parts.append(f"({key[1]})")
    if key[2]:
        parts.append(f"[{key[2]}]")
    parts.append(" + ")
    if key[3]:
        parts.append(f"{key[3]}")
    if key[4]:
        parts.append(f"({key[4]})")
    if key[5]:
        parts.append(f"[{key[5]}]")
    if key[6]:
        parts.append(f"'{key[6]}'")
    return "".join(parts)


def check_false_positives_on_features(pattern_code: str, parsed_sentences: List[Tuple[str, List[Dict]]]) -> List[str]:
    """Check for false positives on pre-parsed grammatical sentences."""
    matches = []
    for sentence, feats in parsed_sentences:
        for i in range(len(feats) - 1):
            curr = feats[i]
            nxt = feats[i + 1]
            nxt2 = feats[i + 2] if i + 2 < len(feats) else {}

            try:
                if eval(pattern_code):
                    matches.append(sentence)
                    break
            except Exception:
                pass

    return matches


def main():
    argparser = argparse.ArgumentParser(description="Suggest high-value patterns")
    argparser.add_argument("--min-detections", type=int, default=30,
                          help="Minimum detections to consider (default: 30)")
    argparser.add_argument("--max-suggestions", type=int, default=10,
                          help="Maximum patterns to suggest (default: 10)")
    argparser.add_argument("--verify-fp", type=int, default=20000,
                          help="Number of grammatical sentences to check for FP (default: 20000)")
    argparser.add_argument("--no-cache", action="store_true",
                          help="Disable caching of parsed sentences")
    args = argparser.parse_args()

    parser = SudachiJapaneseParser()

    # Load data
    agram_file = 'data/agrammatic_sentences.tsv'
    gram_file = 'data/jpn_sentences.tsv'

    print("Loading agrammatic sentences...")
    with open(agram_file, 'r', encoding='utf-8') as f:
        agram_lines = [line.strip().split('\t')[2] for line in f
                      if line.strip() and len(line.strip().split('\t')) >= 3]

    print("Loading grammatical sentences...")
    known_errors = set()
    with open('data/jpn_sentences_known_errors.tsv', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split('\t')
                if parts:
                    known_errors.add(parts[0])

    with open(gram_file, 'r', encoding='utf-8') as f:
        gram_lines = []
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 3 and parts[0] not in known_errors:
                    gram_lines.append(parts[2])

    # Parse sentences with caching
    print("\nLoading/parsing agrammatic sentences...")
    if args.no_cache:
        agram_parsed = [(s, [extract_token_features(t) for t in split_kotogram(parser.japanese_to_kotogram(s))]) for s in agram_lines]
    else:
        agram_parsed = parse_sentences_with_cache(agram_lines, parser, agram_file, "agram")

    gram_cache_type = f"gram_{args.verify_fp}"
    print(f"\nLoading/parsing grammatical sentences ({args.verify_fp})...")
    if args.no_cache:
        gram_parsed = [(s, [extract_token_features(t) for t in split_kotogram(parser.japanese_to_kotogram(s))]) for s in gram_lines[:args.verify_fp]]
    else:
        gram_parsed = parse_sentences_with_cache(gram_lines[:args.verify_fp], parser, gram_file, gram_cache_type)

    # Find missed sentences and count patterns
    print("\nAnalyzing missed sentences...")
    pattern_counts: Counter = Counter()
    pattern_examples: Dict[Tuple, List[str]] = {}

    missed_count = 0
    for sentence, feats in agram_parsed:
        kotogram = parser.japanese_to_kotogram(sentence)
        result = grammaticality(kotogram, use_model=False)

        if result:  # Missed
            missed_count += 1

            for i in range(len(feats) - 1):
                curr, nxt = feats[i], feats[i + 1]
                nxt2 = feats[i + 2] if i + 2 < len(feats) else None

                key = get_pattern_key(curr, nxt, nxt2)
                pattern_counts[key] += 1

                if key not in pattern_examples:
                    pattern_examples[key] = []
                if len(pattern_examples[key]) < 3:
                    pattern_examples[key].append(sentence)

    print(f"Total missed: {missed_count} / {len(agram_lines)} ({100*missed_count/len(agram_lines):.1f}%)")

    # Filter patterns with minimum detections
    candidates = [(k, c) for k, c in pattern_counts.most_common(100)
                  if c >= args.min_detections]

    print(f"\nFound {len(candidates)} patterns with >= {args.min_detections} detections")
    print(f"Checking for false positives (using {args.verify_fp} grammatical sentences)...\n")

    # Check each candidate for false positives
    safe_patterns = []
    for key, count in candidates:
        if not key or len(key) < 7:  # Skip empty or malformed keys
            continue
        code = pattern_to_code(key)
        if not code:
            continue

        fp = check_false_positives_on_features(code, gram_parsed)

        if len(fp) == 0:
            safe_patterns.append((key, count, code))
            print(f"✓ {count:4d} detections | {pattern_to_description(key)}")
            if len(safe_patterns) >= args.max_suggestions:
                break
        else:
            print(f"✗ {count:4d} detections, {len(fp)} FP | {pattern_to_description(key)}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"SUGGESTED PATTERNS ({len(safe_patterns)} safe patterns found):")
    print(f"{'='*70}\n")

    for i, (key, count, code) in enumerate(safe_patterns, 1):
        print(f"{i}. Would detect: {count} sentences")
        print(f"   Pattern: {pattern_to_description(key)}")
        print(f"   Code: {code}")
        print(f"   Examples:")
        for ex in pattern_examples.get(key, [])[:2]:
            print(f"     - {ex}")
        print()


if __name__ == "__main__":
    main()
