#!/usr/bin/env python3
"""Check a pattern for detection count and false positives.

Usage:
    python scripts/check_pattern.py "PATTERN_CODE"
    python scripts/check_pattern.py "PATTERN_CODE" --analyze

The PATTERN_CODE is a Python expression that checks curr and nxt token features.
Example:
    python scripts/check_pattern.py "curr.get('pos')=='adj' and curr.get('conjugated_form')=='terminal' and nxt.get('surface')=='ない'"
"""
import argparse
import hashlib
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
from kotogram.kotogram import split_kotogram, extract_token_features
from kotogram.analysis import grammaticality

# Cache directory
CACHE_DIR = Path(".cache/kotogram")


def get_cache_path(data_file: str, cache_type: str) -> Path:
    """Get cache file path for a data file."""
    # Use file modification time and size for cache invalidation
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
    # Try to load from cache
    cached = load_cached_features(data_file, cache_type)
    if cached is not None:
        if verbose:
            print(f"  (loaded {len(cached)} cached parses)")
        return cached

    # Parse all sentences
    if verbose:
        print(f"  (parsing {len(sentences)} sentences...)")

    results = []
    for sentence in sentences:
        kotogram = parser.japanese_to_kotogram(sentence)
        tokens = split_kotogram(kotogram)
        feats = [extract_token_features(t) for t in tokens]
        results.append((sentence, feats))

    # Save to cache
    save_cached_features(data_file, cache_type, results)
    if verbose:
        print(f"  (cached parses for next run)")

    return results


def check_pattern_on_features(
    pattern_code: str,
    parsed_sentences: List[Tuple[str, List[Dict]]],
    check_grammaticality: bool = False,
    parser: Optional[SudachiJapaneseParser] = None
) -> List[str]:
    """Check pattern against pre-parsed sentences."""
    matches = []

    for sentence, feats in parsed_sentences:
        # For agrammatic sentences, only check missed ones
        if check_grammaticality and parser:
            kotogram = parser.japanese_to_kotogram(sentence)
            result = grammaticality(kotogram, use_model=False)
            if not result:  # Already detected
                continue

        for i in range(len(feats) - 1):
            curr = feats[i]
            nxt = feats[i + 1]
            nxt2 = feats[i + 2] if i + 2 < len(feats) else {}
            prev = feats[i - 1] if i > 0 else {}

            try:
                if eval(pattern_code):
                    matches.append(sentence)
                    break
            except Exception:
                pass

    return matches


def analyze_matches(pattern_code: str, parser, sentences, limit=5):
    """Analyze and display token details for matching sentences."""
    count = 0
    for sentence in sentences:
        kotogram = parser.japanese_to_kotogram(sentence)
        tokens = split_kotogram(kotogram)
        feats = [extract_token_features(t) for t in tokens]

        for i in range(len(feats) - 1):
            curr = feats[i]
            nxt = feats[i + 1]
            nxt2 = feats[i + 2] if i + 2 < len(feats) else {}
            prev = feats[i - 1] if i > 0 else {}

            try:
                if eval(pattern_code):
                    print(f"\nSentence: {sentence}")
                    # Print context around match
                    start = max(0, i - 1)
                    end = min(len(feats), i + 4)
                    for j in range(start, end):
                        marker = ">>>" if j == i else "   "
                        f = feats[j]
                        print(f"  {marker} [{j}] {f.get('surface')!r:12s} "
                              f"pos={f.get('pos')!r:6s} "
                              f"type={f.get('conjugated_type')!r:20s} "
                              f"form={f.get('conjugated_form')!r}")
                    count += 1
                    if count >= limit:
                        return
                    break
            except Exception:
                pass


def main():
    argparser = argparse.ArgumentParser(description="Check pattern detection and false positives")
    argparser.add_argument("pattern", nargs='+', help="Python expression(s) using curr, nxt, nxt2, prev dicts")
    argparser.add_argument("--show", type=int, default=5, help="Number of examples to show")
    argparser.add_argument("--analyze", action="store_true", help="Show detailed token analysis")
    argparser.add_argument("--fp-limit", type=int, default=50000,
                          help="Number of grammatical sentences to check (default: 50000)")
    argparser.add_argument("--no-cache", action="store_true",
                          help="Disable caching of parsed sentences")
    args = argparser.parse_args()

    patterns = args.pattern

    parser = SudachiJapaneseParser()

    # Load agrammatic sentences
    print("Loading agrammatic sentences...")
    agram_file = 'data/agrammatic_sentences.tsv'
    with open(agram_file, 'r', encoding='utf-8') as f:
        agram_lines = [line.strip().split('\t')[2] for line in f if line.strip() and len(line.strip().split('\t')) >= 3]

    # Load grammatical sentences
    print("Loading grammatical sentences...")
    gram_file = 'data/jpn_sentences.tsv'
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

    # Parse sentences (with caching)
    if args.no_cache:
        print("\nParsing agrammatic sentences...")
        agram_parsed = [(s, [extract_token_features(t) for t in split_kotogram(parser.japanese_to_kotogram(s))]) for s in agram_lines]
        print(f"\nParsing grammatical sentences ({args.fp_limit})...")
        gram_parsed = [(s, [extract_token_features(t) for t in split_kotogram(parser.japanese_to_kotogram(s))]) for s in gram_lines[:args.fp_limit]]
    else:
        print("\nLoading/parsing agrammatic sentences...")
        agram_parsed = parse_sentences_with_cache(agram_lines, parser, agram_file, "agram")

        # For grammatical, create a virtual cache key based on limit
        gram_cache_type = f"gram_{args.fp_limit}"
        print(f"\nLoading/parsing grammatical sentences ({args.fp_limit})...")
        gram_parsed = parse_sentences_with_cache(gram_lines[:args.fp_limit], parser, gram_file, gram_cache_type)

    # Process each pattern
    results = []
    for pattern in patterns:
        print(f"\n{'='*60}")
        print(f"PATTERN: {pattern}")
        print(f"{'='*60}")

        # Check agrammatic (missed only)
        print(f"\nChecking pattern on missed agrammatic sentences...")
        agram_matches = check_pattern_on_features(pattern, agram_parsed, check_grammaticality=True, parser=parser)

        print(f"\nAgrammatic matches (would detect): {len(agram_matches)}")
        if args.analyze and agram_matches:
            print("\nDetailed analysis of agrammatic matches:")
            analyze_matches(pattern, parser, agram_matches, args.show)
        elif agram_matches[:args.show]:
            print("Examples:")
            for ex in agram_matches[:args.show]:
                print(f"  {ex}")

        # Check grammatical (false positives)
        print(f"\nChecking pattern on grammatical sentences...")
        gram_matches = check_pattern_on_features(pattern, gram_parsed, check_grammaticality=False)

        print(f"\nFalse positives: {len(gram_matches)}")
        if args.analyze and gram_matches:
            print("\nDetailed analysis of false positives:")
            analyze_matches(pattern, parser, gram_matches, args.show)
        elif gram_matches[:args.show]:
            print("Examples:")
            for ex in gram_matches[:args.show]:
                print(f"  {ex}")

        # Store results
        is_safe = len(gram_matches) == 0 and len(agram_matches) > 0
        results.append((pattern, len(agram_matches), len(gram_matches), is_safe))

        # Per-pattern status
        if is_safe:
            print(f"\n  ✓ Safe to implement!")
        elif len(gram_matches) > 0:
            print(f"\n  ✗ Has false positives - needs refinement")
        else:
            print(f"\n  ✗ No matches found - check pattern syntax")

    # Final summary for multiple patterns
    if len(patterns) > 1:
        print(f"\n{'='*60}")
        print(f"SUMMARY OF ALL PATTERNS:")
        print(f"{'='*60}")
        safe_count = 0
        for pattern, agram, gram, is_safe in results:
            status = "✓" if is_safe else "✗"
            print(f"  {status} {agram:4d} detections, {gram:3d} FP | {pattern[:60]}...")
            if is_safe:
                safe_count += 1
        print(f"\n  {safe_count}/{len(patterns)} patterns are safe to implement")


if __name__ == "__main__":
    main()
