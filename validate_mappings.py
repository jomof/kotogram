#!/usr/bin/env python3
"""Script to validate all Tatoeba sentences and find unmapped features."""

import sys
from collections import defaultdict
from kotogram import MecabJapaneseParser

def validate_sentences(tsv_file: str, max_sentences: int = None):
    """Validate sentences and collect unmapped features.

    Args:
        tsv_file: Path to the Tatoeba TSV file
        max_sentences: Maximum number of sentences to process (None for all)
    """
    # Try to create parser with validation
    try:
        parser = MecabJapaneseParser(validate=True)
    except Exception as e:
        print(f"Warning: Could not initialize MeCab parser: {e}")
        print("MeCab and unidic may not be installed. Skipping validation.")
        return {}, []

    unmapped_features = defaultdict(set)
    failed_sentences = []
    successful_count = 0

    with open(tsv_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_sentences and i >= max_sentences:
                break

            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue

            sentence_id = parts[0]
            language = parts[1]
            text = parts[2]

            if language != 'jpn':
                continue

            try:
                kotogram = parser.japanese_to_kotogram(text)
                successful_count += 1
            except KeyError as e:
                # Get the actual error message - args[0] is the message passed to KeyError()
                error_msg = e.args[0] if e.args else str(e)

                # Parse the error message to extract map name and key
                if "Missing mapping in" in error_msg:
                    # Extract map name
                    map_start = error_msg.find("Missing mapping in ") + len("Missing mapping in ")
                    map_end = error_msg.find(":", map_start)
                    map_name = error_msg[map_start:map_end]

                    # Extract key - find the text between key=' and ' not found
                    key_start = error_msg.find("key='") + len("key='")
                    key_end = error_msg.find("' not found", key_start)
                    if key_end == -1:
                        key_end = error_msg.find("'", key_start)
                    key = error_msg[key_start:key_end]

                    unmapped_features[map_name].add(key)
                    failed_sentences.append({
                        'id': sentence_id,
                        'text': text,
                        'map': map_name,
                        'key': key,
                        'error': error_msg
                    })
                else:
                    print(f"Unexpected error format: {error_msg}")
            except Exception as e:
                print(f"Unexpected error processing sentence {sentence_id}: {e}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {len(failed_sentences)}")
    print()

    if unmapped_features:
        print(f"UNMAPPED FEATURES BY MAP:")
        print(f"{'-'*80}")
        for map_name, keys in sorted(unmapped_features.items()):
            print(f"\n{map_name}: {len(keys)} unmapped keys")
            for key in sorted(keys):
                print(f"  '{key}'")

        print(f"\n{'='*80}")
        print(f"FIRST 10 FAILED SENTENCES:")
        print(f"{'='*80}")
        for failure in failed_sentences[:10]:
            print(f"\nID: {failure['id']}")
            print(f"Text: {failure['text']}")
            print(f"Map: {failure['map']}, Key: '{failure['key']}'")
    else:
        print("✅ All sentences validated successfully!")

    return unmapped_features, failed_sentences

if __name__ == "__main__":
    import os

    # Use path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tsv_file = os.path.join(script_dir, "data", "jpn_sentences.tsv")

    # Start with first 100 sentences to see what we find
    max_sentences = 100
    if len(sys.argv) > 1:
        max_sentences = int(sys.argv[1]) if sys.argv[1] != "all" else None

    print(f"Validating {'all' if max_sentences is None else max_sentences} sentences from {tsv_file}")
    print("This may take a while...\n")

    unmapped, failed = validate_sentences(tsv_file, max_sentences)
