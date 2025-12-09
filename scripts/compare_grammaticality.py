#!/usr/bin/env python3
"""Analyze grammaticality model predictions.

Since there's no rule-based grammaticality checker, this script examines
model predictions to identify patterns in misclassifications.
"""

import argparse
import csv
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kotogram import SudachiJapaneseParser, grammaticality


def main():
    parser = argparse.ArgumentParser(
        description="Analyze grammaticality model predictions"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/jpn_sentences.tsv",
        help="Path to grammatic sentences TSV (default: data/jpn_sentences.tsv)",
    )
    parser.add_argument(
        "--agrammatic-data",
        type=str,
        default="data/jpn_agrammatic.tsv",
        help="Path to agrammatic sentences TSV (default: data/jpn_agrammatic.tsv)",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=20,
        help="Maximum errors to display per category (default: 20)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to check per file (default: all)",
    )
    args = parser.parse_args()

    jp_parser = SudachiJapaneseParser(dict_type='full')

    # Track results
    grammatic_correct = 0
    grammatic_wrong = []  # Grammatic sentences predicted as agrammatic (false negatives)
    agrammatic_correct = 0
    agrammatic_wrong = []  # Agrammatic sentences predicted as grammatic (false positives)

    # Check grammatic sentences (should predict True)
    data_path = Path(args.data)
    if data_path.exists():
        print(f"Checking grammatic sentences from: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            count = 0

            for row in reader:
                if len(row) < 3:
                    continue

                sentence_id, lang, sentence = row[0], row[1], row[2]

                if lang != 'jpn':
                    continue

                try:
                    kotogram = jp_parser.japanese_to_kotogram(sentence)
                    is_grammatic = grammaticality(kotogram, use_model=True)

                    if is_grammatic:
                        grammatic_correct += 1
                    else:
                        grammatic_wrong.append({
                            'id': sentence_id,
                            'sentence': sentence,
                            'kotogram': kotogram,
                        })

                    count += 1
                    if count % 10000 == 0:
                        print(f"  Checked {count} grammatic sentences...")

                    if args.max_samples and count >= args.max_samples:
                        break

                except Exception as e:
                    print(f"Error processing {sentence_id}: {e}")
                    continue

        print(f"  Total grammatic: {grammatic_correct + len(grammatic_wrong)}")
        print(f"  Correctly identified: {grammatic_correct}")
        print(f"  Misclassified as agrammatic: {len(grammatic_wrong)}")
        print()

    # Check agrammatic sentences (should predict False)
    agram_path = Path(args.agrammatic_data)
    if agram_path.exists():
        print(f"Checking agrammatic sentences from: {agram_path}")
        with open(agram_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            count = 0

            for row in reader:
                if len(row) < 3:
                    continue

                sentence_id, lang, sentence = row[0], row[1], row[2]

                if lang != 'jpn':
                    continue

                try:
                    kotogram = jp_parser.japanese_to_kotogram(sentence)
                    is_grammatic = grammaticality(kotogram, use_model=True)

                    if not is_grammatic:
                        agrammatic_correct += 1
                    else:
                        agrammatic_wrong.append({
                            'id': sentence_id,
                            'sentence': sentence,
                            'kotogram': kotogram,
                        })

                    count += 1
                    if count % 10000 == 0:
                        print(f"  Checked {count} agrammatic sentences...")

                    if args.max_samples and count >= args.max_samples:
                        break

                except Exception as e:
                    print(f"Error processing {sentence_id}: {e}")
                    continue

        print(f"  Total agrammatic: {agrammatic_correct + len(agrammatic_wrong)}")
        print(f"  Correctly identified: {agrammatic_correct}")
        print(f"  Misclassified as grammatic: {len(agrammatic_wrong)}")
        print()

    # Summary
    total = grammatic_correct + len(grammatic_wrong) + agrammatic_correct + len(agrammatic_wrong)
    correct = grammatic_correct + agrammatic_correct
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total sentences: {total}")
    print(f"Overall accuracy: {correct}/{total} ({100*correct/total:.2f}%)")
    print()
    print(f"Grammatic accuracy: {grammatic_correct}/{grammatic_correct + len(grammatic_wrong)} "
          f"({100*grammatic_correct/(grammatic_correct + len(grammatic_wrong)):.2f}%)")
    print(f"Agrammatic accuracy: {agrammatic_correct}/{agrammatic_correct + len(agrammatic_wrong)} "
          f"({100*agrammatic_correct/(agrammatic_correct + len(agrammatic_wrong)):.2f}%)")
    print()

    # Show false negatives (grammatic predicted as agrammatic)
    if grammatic_wrong:
        print("=" * 60)
        print(f"FALSE NEGATIVES (grammatic -> agrammatic): {len(grammatic_wrong)} total")
        print("These are valid sentences the model thinks are ungrammatical:")
        print("=" * 60)
        for i, item in enumerate(grammatic_wrong[:args.max_errors], 1):
            print(f"\n{i}. ID: {item['id']}")
            print(f"   Sentence: {item['sentence']}")
            print(f"   Kotogram: {item['kotogram'][:80]}...")

        if len(grammatic_wrong) > args.max_errors:
            print(f"\n... and {len(grammatic_wrong) - args.max_errors} more")
        print()

    # Show false positives (agrammatic predicted as grammatic)
    if agrammatic_wrong:
        print("=" * 60)
        print(f"FALSE POSITIVES (agrammatic -> grammatic): {len(agrammatic_wrong)} total")
        print("These are ungrammatical sentences the model thinks are valid:")
        print("=" * 60)
        for i, item in enumerate(agrammatic_wrong[:args.max_errors], 1):
            print(f"\n{i}. ID: {item['id']}")
            print(f"   Sentence: {item['sentence']}")
            print(f"   Kotogram: {item['kotogram'][:80]}...")

        if len(agrammatic_wrong) > args.max_errors:
            print(f"\n... and {len(agrammatic_wrong) - args.max_errors} more")
        print()


if __name__ == "__main__":
    main()
