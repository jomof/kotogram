#!/usr/bin/env python3
"""Script to validate all Tatoeba sentences and find unmapped features.

Usage:
    python scripts/validate_tatoeba.py [count]

    count: Number of sentences to validate (default: 100, use 'all' for all sentences)

Examples:
    python scripts/validate_tatoeba.py 100          # Validate first 100 sentences
    python scripts/validate_tatoeba.py all          # Validate all sentences
"""

import sys
import os
from collections import defaultdict
from typing import Tuple, Dict, List, Set, Optional



from kotogram import SudachiJapaneseParser  # noqa: E402
from kotogram.exceptions import MissingMappingError  # noqa: E402


def validate_sentences(
    parser: SudachiJapaneseParser,
    parser_name: str,
    tsv_file: str,
    max_sentences: Optional[int] = None
) -> Tuple[Dict[str, Set[str]], List[Dict[str, str]]]:
    """Validate sentences and collect unmapped features.

    Args:
        parser: Parser instance (SudachiJapaneseParser)
        parser_name: Name of the parser for display purposes
        tsv_file: Path to the Tatoeba TSV file
        max_sentences: Maximum number of sentences to process (None for all)

    Returns:
        Tuple of (unmapped_features dict, failed_sentences list)
    """
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
                parser.japanese_to_kotogram(text)
                successful_count += 1
            except MissingMappingError as e:
                unmapped_features[e.map_name].add(e.key)
                failed_sentences.append({
                    'id': sentence_id,
                    'text': text,
                    'map': e.map_name,
                    'key': e.key,
                    'error': str(e)
                })



    # Print summary
    print(f"\n{'='*80}")
    print(f"{parser_name.upper()} VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {len(failed_sentences)}")
    print()

    if unmapped_features:
        print("UNMAPPED FEATURES BY MAP:")
        print(f"{'-'*80}")
        for map_name, keys in sorted(unmapped_features.items()):
            print(f"\n{map_name}: {len(keys)} unmapped keys")
            for key in sorted(keys):
                print(f"  '{key}'")

        print(f"\n{'='*80}")
        print("FIRST 10 FAILED SENTENCES:")
        print(f"{'='*80}")
        for failure in failed_sentences[:10]:
            print(f"\nID: {failure['id']}")
            print(f"Text: {failure['text']}")
            print(f"Map: {failure['map']}, Key: '{failure['key']}'")
    else:
        print(f"✅ All sentences validated successfully with {parser_name}!")

    return unmapped_features, failed_sentences


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
            try:
                max_sentences = int(sys.argv[1])
            except ValueError:
                print("Usage: python scripts/validate_tatoeba.py [count]")
                sys.exit(1)

    print(f"Validating {'all' if max_sentences is None else max_sentences} sentences from {tsv_file}")
    print("This may take a while...\n")

    parser = SudachiJapaneseParser(dict_type='full', validate=True)
    print(f"\n{'#'*80}")
    print("# VALIDATING WITH SUDACHI")
    print(f"{'#'*80}")
    validate_sentences(parser, "Sudachi", tsv_file, max_sentences)


if __name__ == "__main__":
    main()
