#!/usr/bin/env python3
"""Analyze missed agrammatic sentences to find detectable patterns.

Usage:
    python scripts/analyze_missed.py [--limit N] [--surface PATTERN]

Examples:
    python scripts/analyze_missed.py --limit 50
    python scripts/analyze_missed.py --surface "でいる"
    python scripts/analyze_missed.py --surface "ない"
"""
import argparse
from collections import Counter
from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
from kotogram.kotogram import split_kotogram, extract_token_features
from kotogram.analysis import grammaticality


def analyze_sentence(parser, sentence):
    """Return token features for a sentence."""
    kotogram = parser.japanese_to_kotogram(sentence)
    tokens = split_kotogram(kotogram)
    return [extract_token_features(t) for t in tokens]


def print_tokens(feats, sentence):
    """Print token analysis for a sentence."""
    print(f"\nSentence: {sentence}")
    for i, f in enumerate(feats):
        print(f"  [{i}] {f.get('surface')!r:12s} pos={f.get('pos')!r:6s} "
              f"type={f.get('conjugated_type')!r:20s} form={f.get('conjugated_form')!r}")


def main():
    argparser = argparse.ArgumentParser(description="Analyze missed agrammatic patterns")
    argparser.add_argument("--limit", type=int, default=30, help="Number of examples to show")
    argparser.add_argument("--surface", type=str, help="Filter by surface pattern (e.g., 'でいる')")
    argparser.add_argument("--analyze", type=str, help="Analyze a specific sentence")
    argparser.add_argument("--count-bigrams", action="store_true", help="Count POS bigram patterns")
    args = argparser.parse_args()

    parser = SudachiJapaneseParser()

    if args.analyze:
        feats = analyze_sentence(parser, args.analyze)
        print_tokens(feats, args.analyze)
        return

    # Load agrammatic sentences
    with open('data/agrammatic_sentences.tsv', 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Find missed sentences
    missed = []
    for line in lines:
        parts = line.split('\t')
        if len(parts) >= 3:
            sentence = parts[2]
            kotogram = parser.japanese_to_kotogram(sentence)
            result = grammaticality(kotogram, use_model=False)
            if result:  # Missed
                missed.append(sentence)

    print(f"Total missed: {len(missed)} / {len(lines)} ({100*len(missed)/len(lines):.1f}%)")

    if args.count_bigrams:
        # Count bigram patterns
        bigrams = Counter()
        for sentence in missed[:10000]:
            feats = analyze_sentence(parser, sentence)
            for i in range(len(feats) - 1):
                curr, nxt = feats[i], feats[i + 1]
                pattern = (
                    f"{curr.get('pos','')}:{curr.get('conjugated_form','')[:6]} -> "
                    f"{nxt.get('pos','')}:{nxt.get('conjugated_form','')[:6]}"
                )
                bigrams[pattern] += 1

        print("\nTop 30 POS bigrams in missed sentences:")
        for pattern, count in bigrams.most_common(30):
            print(f"  {count:5d}: {pattern}")
        return

    # Filter by surface pattern if specified
    if args.surface:
        filtered = [s for s in missed if args.surface in s]
        print(f"\nFiltered by '{args.surface}': {len(filtered)} sentences")
        missed = filtered

    # Show examples with token analysis
    print(f"\nShowing {min(args.limit, len(missed))} examples:\n")
    for sentence in missed[:args.limit]:
        feats = analyze_sentence(parser, sentence)
        print_tokens(feats, sentence)


if __name__ == "__main__":
    main()
