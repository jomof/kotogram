#!/usr/bin/env python3
"""Scrape Japanese sentences from grammar point YAML files.

This script reads YAML files from the specified directory, extracts Japanese sentences
from specific fields, cleans them (removing spaces and braces), assigns unique IDs,
and saves them to a TSV file.
"""

import glob
import os
import yaml
import csv
import argparse
from typing import List, Dict, Set

def clean_sentence(sentence: str) -> str:
    """Remove spaces, braces from the sentence."""
    if not sentence:
        return ""
    # Remove half-width spaces
    sentence = sentence.replace(" ", "")
    # Remove braces {}
    sentence = sentence.replace("{", "").replace("}", "")
    return sentence

def main():
    parser = argparse.ArgumentParser(description="Scrape Japanese sentences from grammar point YAML files.")
    parser.add_argument(
        "--input-dir",
        default=".tmp-inspiration/cloze-data/resources/processed/ai-cleaned-merge-grammars",
        help="Directory containing grammar point YAML files."
    )
    parser.add_argument(
        "--output-file",
        default="data/jpn_sentences_gp.tsv",
        help="Output TSV file."
    )
    args = parser.parse_args()

    input_pattern = os.path.join(args.input_dir, "*.yaml")
    yaml_files = sorted(glob.glob(input_pattern))

    if not yaml_files:
        print(f"No YAML files found in {args.input_dir}")
        return

    print(f"Found {len(yaml_files)} YAML files.")

    unique_sentences: Set[str] = set()
    output_rows: List[Dict[str, str]] = []

    for file_path in yaml_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        if not data:
            continue

        gp_id = data.get('id')
        if not gp_id:
            print(f"Skipping {file_path}: No 'id' field found.")
            continue

        sentences_to_add = []

        # Extract from examples -> japanese
        examples = data.get('examples', [])
        if examples:
            for example in examples:
                japanese_list = example.get('japanese', [])
                if isinstance(japanese_list, list):
                    sentences_to_add.extend(japanese_list)
                elif isinstance(japanese_list, str):
                    sentences_to_add.append(japanese_list)
                
                # Extract from examples -> competing_grammar -> competing_japanese
                competing_grammar = example.get('competing_grammar', [])
                if competing_grammar:
                    for cg in competing_grammar:
                        competing_japanese_list = cg.get('competing_japanese', [])
                        if isinstance(competing_japanese_list, list):
                            sentences_to_add.extend(competing_japanese_list)
                        elif isinstance(competing_japanese_list, str):
                            sentences_to_add.append(competing_japanese_list)

        # Process gathered sentences
        sentence_counter = 0
        for raw_sentence in sentences_to_add:
            cleaned_sentence = clean_sentence(raw_sentence)
            
            if not cleaned_sentence:
                continue

            if cleaned_sentence in unique_sentences:
                continue

            unique_sentences.add(cleaned_sentence)
            
            # Generate ID: gpXXXX_N
            sentence_id = f"{gp_id}_{sentence_counter}"
            sentence_counter += 1

            output_rows.append({
                'id': sentence_id,
                'lang': 'jpn',
                'sentence': cleaned_sentence
            })

    print(f"Extracted {len(output_rows)} unique sentences.")

    # Write to TSV
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    try:
        with open(args.output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'lang', 'sentence'], delimiter='\t')
            # writer.writeheader() # TSV usually doesn't have header in this dataset format based on previous file view
            for row in output_rows:
                 # Manually writing to ensure order and no header if not needed, 
                 # but based on data/jpn_sentences.tsv view, it seemed to have data straight away.
                 # Let's check jpn_sentences.tsv content again to be sure about header.
                 # Wait, Step 8 showed `10307	jpn	これが一番得意分野です。` - NO HEADER.
                 # DictWriter writes header only if writeheader() is called.
                 # But we need to write rows in order of fieldnames.
                 f.write(f"{row['id']}\t{row['lang']}\t{row['sentence']}\n")
        print(f"Successfully wrote to {args.output_file}")
    except Exception as e:
        print(f"Error writing to {args.output_file}: {e}")

if __name__ == "__main__":
    main()
