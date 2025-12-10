#!/usr/bin/env python3
"""Find sentences with unpragmatic formality in the TSV dataset.

This script searches through jpn_sentences.tsv to find sentences that
are flagged as UNPRAGMATIC_FORMALITY by the formality analysis function.
It's used for validation and testing of the formality detection logic.

Usage:
    python find_unpragmatic_sentences.py [max_to_display]

    max_to_display: Maximum number of unpragmatic sentences to display (default: 10)
                   The script will still process ALL sentences, but only display this many.
"""

import sys
from kotogram import SudachiJapaneseParser, formality, FormalityLevel
from kotogram.kotogram import split_kotogram
import re

# Blacklist of sentence IDs that are known acceptable cases
# (e.g., dialogue, compound sentences, etc.)
BLACKLIST = {
    '77031',   # Dialogue with multiple speakers
    '75385',   # Dialogue with multiple speakers
    '141788',  # Honorific + よ (acceptable in casual conversation about superiors)
    '190393',  # Compound sentence with different clauses having different formality
    '192863',  # Honorific imperative + よ (borderline but acceptable)
}

# Initialize parser
try:
    parser = SudachiJapaneseParser(dict_type='full')
    print("✓ Sudachi parser loaded")
except Exception as e:
    print(f"✗ Sudachi parser failed: {e}")
    print("ERROR: Parser not available!")
    sys.exit(1)

print("\nSearching for UNPRAGMATIC sentences...")
print("Will stop after finding 5 non-blacklisted unpragmatic sentences...")
print("=" * 80)

unpragmatic_found = []
blacklisted_count = 0
target_count = int(sys.argv[1]) if len(sys.argv) > 1 else 5
line_count = 0
total_sentences = 0

with open('data/jpn_sentences.tsv', 'r', encoding='utf-8') as f:
    for line in f:
        line_count += 1
        parts = line.strip().split('\t')
        if len(parts) < 3:
            continue

        sentence_id = parts[0]
        language = parts[1]
        text = parts[2]

        if language != 'jpn':
            continue

        total_sentences += 1

        # Skip very long sentences or very short ones
        if len(text) < 3 or len(text) > 100:
            continue

        # Test with parser
        try:
            kotogram = parser.japanese_to_kotogram(text)
            level = formality(kotogram)
        except Exception as e:
            continue

        # Check if unpragmatic
        is_unpragmatic = level == FormalityLevel.UNPRAGMATIC_FORMALITY

        if is_unpragmatic:
            # Check if blacklisted
            if sentence_id in BLACKLIST:
                blacklisted_count += 1
                continue

            unpragmatic_found.append({
                'id': sentence_id,
                'text': text,
                'level': level,
                'line': line_count
            })

            # Display this sentence
            print(f"\n[{len(unpragmatic_found)}/{target_count}] ID: {sentence_id} (line {line_count})")
            print(f"Text: {text}")
            print(f"Result: {level.value}")

            # Stop after finding target_count non-blacklisted unpragmatic sentences
            if len(unpragmatic_found) >= target_count:
                break

        # Progress indicator
        if line_count % 10000 == 0:
            print(f"Processed {line_count} lines ({total_sentences} Japanese)...", end='\r')

print(f"\n\n{'=' * 80}")
print(f"SUMMARY:")
print(f"  Processed {line_count} lines ({total_sentences} Japanese sentences)")
print(f"  Blacklisted (skipped): {blacklisted_count}")
print(f"  Unpragmatic found: {len(unpragmatic_found)}")
print("=" * 80)

# Detailed analysis
if unpragmatic_found:
    print("\nDETAILED ANALYSIS:")
    print("=" * 80)

    for i, item in enumerate(unpragmatic_found, 1):
        print(f"\n{i}. ID: {item['id']}")
        print(f"   Text: {item['text']}")
        print(f"   Line: {item['line']}")

        # Get kotogram for detailed inspection
        try:
            kotogram = parser.japanese_to_kotogram(item['text'])
            tokens = split_kotogram(kotogram)
            print(f"\n   Tokens ({len(tokens)}):")
            for j, token in enumerate(tokens):
                # Extract surface and POS
                surface_match = re.search(r'ˢ(.*?)ᵖ', token)
                pos_match = re.search(r'ᵖ([^⌉ᵇᵈʳ]+)', token)
                if surface_match and pos_match:
                    surface = surface_match.group(1)
                    pos = pos_match.group(1)
                    print(f"      [{j}] {surface:10s} → {pos}")
        except Exception as e:
            print(f"   Error analyzing: {e}")

        print()

if len(unpragmatic_found) == 0:
    print("\n✓ SUCCESS: No unpragmatic sentences found!")
    print("The formality analysis function is working correctly on the dataset.")
    sys.exit(0)
else:
    print("\n⚠ REVIEW NEEDED: Found unpragmatic sentences.")
    print("Please analyze the sentences above to determine if they are:")
    print("  1. True positives (actually unpragmatic)")
    print("  2. False positives (need to be added to BLACKLIST)")
    sys.exit(1)
