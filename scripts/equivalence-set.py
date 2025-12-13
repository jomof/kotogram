#!/usr/bin/env python3
"""
Build an equivalence set YAML file that groups all equivalent Japanese sentences
under their common English key.

Reads from: .tmp-inspiration/cloze-data/resources/processed/ai-cleaned-merge-grammars/*.yaml
Outputs to: data/equivalence-set.yml

Uses kotogram.augment to augment equivalence sets and filter ungrammatical sentences.
"""

import re
import yaml
from pathlib import Path
from collections import defaultdict
from kotogram.augment import augment


def main():
    resources_dir = Path(".tmp-inspiration/cloze-data/resources/processed/ai-cleaned-merge-grammars")
    output_file = Path("data/equivalence-set.yml")

    # Load all yaml files
    yaml_files = sorted(resources_dir.glob("*.yaml"))
    print(f"Found {len(yaml_files)} yaml files")

    # Group by English sentence
    equivalences = defaultdict(set)
    
    for yf in yaml_files:
        with open(yf, 'r') as f:
            data = yaml.safe_load(f)
            
        if not data or "examples" not in data:
            continue
            
        for example in data["examples"]:
            english = example.get("english", "").strip()
            # Replace '' with " for cleaner English text
            english = english.replace("''", '"')
            # Handle list or string for japanese
            jps_raw = example.get("japanese", [])
            if isinstance(jps_raw, str):
                jps_raw = [jps_raw]
            
            if not english or not jps_raw:
                continue
                
            for jp in jps_raw:
                # Clean up the Japanese sentence (remove curly braces used for highlighting)
                jp_clean = jp.replace("{", "").replace("}", "").strip()

                # Handle parenthetical annotations
                # For optional characters like (や), create both versions
                # Note: operate on the spaced string here, checking regex
                optional_match = re.search(r'\(([ぁ-んァ-ン])\)', jp_clean)
                if optional_match:
                    # Add version with the optional character
                    jp_with = re.sub(r'\(([ぁ-んァ-ン])\)', r'\1', jp_clean)
                    equivalences[english].add(jp_with)
                    # Add version without the optional character
                    jp_without = re.sub(r'\(([ぁ-んァ-ン])\)', '', jp_clean)
                    equivalences[english].add(jp_without)
                else:
                    # Strip other parenthetical annotations (readings, explanations)
                    jp_clean = re.sub(r'\([^)]+\)', '', jp_clean)
                    equivalences[english].add(jp_clean)

    print(f"Found {len(equivalences)} unique English sentences")
    
    # Initialize logic handled by augment module (lazy load)
    print("Processing sentences...")

    result = {}
    total_after_filter = 0

    import argparse
    parser_args = argparse.ArgumentParser()
    parser_args.add_argument("--limit", type=int, help="Limit number of English sentences to process")
    args = parser_args.parse_args()

    for i, (eng, jps) in enumerate(sorted(equivalences.items())):
        if args.limit and i >= args.limit:
            break

        if (i + 1) % 1000 == 0:
            print(f"  Processing {i + 1}/{len(equivalences)}...")

        # Augment and filter using the module
        # Note: jps is a set, convert to list
        filtered = augment(list(jps))
        
        if filtered:
            result[eng] = filtered
            total_after_filter += len(filtered)

    print(f"Total Japanese equivalents (after filtering): {total_after_filter}")

    # Write result to yaml
    with open(output_file, 'w') as f:
        # Custom dump to ensure utf-8 characters are readable
        yaml.dump(result, f, allow_unicode=True, sort_keys=True)
        
    print(f"Output written to {output_file}")


if __name__ == "__main__":
    main()
