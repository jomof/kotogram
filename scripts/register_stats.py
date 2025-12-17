#!/usr/bin/env python3
"""Analyze register label distribution across training data.

This script reads all jpn_sentences*.tsv files and runs the rule-based
register labeler on each sentence to collect statistics about register
distribution in the training data.
"""

import sys
import glob
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Set

# Add project root to path to enable imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kotogram.kotogram import split_kotogram, extract_token_features
from scripts.rule_based_analysis import rule_based_register, analyze_register
from kotogram.analysis import RegisterLevel


def load_sentences_from_tsv(tsv_path: str) -> List[str]:
    """Load sentences from a TSV file.
    
    TSV format: sentence_id, lang, sentence
    We need the raw Japanese sentence (column 3).
    """
    sentences = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                sentence = parts[2]  # Raw Japanese text
                sentences.append(sentence)
    return sentences


def analyze_sentence(sentence: str) -> Set[RegisterLevel]:
    """Analyze a raw Japanese sentence and return detected registers.
    
    This function tokenizes the sentence using Sudachi and then runs
    register detection on the extracted features.
    """
    try:
        import sudachipy
        from sudachipy import tokenizer, dictionary
        
        # Initialize Sudachi tokenizer (cached globally)
        if not hasattr(analyze_sentence, 'tokenizer'):
            tok_obj = dictionary.Dictionary().create()
            analyze_sentence.tokenizer = tok_obj
        
        # Tokenize
        mode = tokenizer.Tokenizer.SplitMode.C
        tokens = analyze_sentence.tokenizer.tokenize(sentence, mode)
        
        # Extract features
        features = []
        for token in tokens:
            pos_parts = token.part_of_speech()
            features.append({
                'surface': token.surface(),
                'pos': pos_parts[0] if len(pos_parts) > 0 else '',
                'pos_detail1': pos_parts[1] if len(pos_parts) > 1 else '',
                'lemma': token.dictionary_form(),
                'conjugated_type': pos_parts[4] if len(pos_parts) > 4 else '',
                'conjugated_form': pos_parts[5] if len(pos_parts) > 5 else '',
            })
        
        # Run register detection
        return rule_based_register(features)
        
    except Exception as e:
        print(f"Warning: Failed to analyze sentence: {sentence[:50]}... Error: {e}", file=sys.stderr)
        return {RegisterLevel.NEUTRAL}


def main():
    # Find all jpn_sentences*.tsv files
    data_dir = Path(__file__).parent.parent / "data"
    tsv_files = sorted(glob.glob(str(data_dir / "jpn_sentences*.tsv")))
    
    if not tsv_files:
        print(f"No jpn_sentences*.tsv files found in {data_dir}")
        return
    
    print(f"Found {len(tsv_files)} TSV file(s):")
    for tsv_file in tsv_files:
        print(f"  - {Path(tsv_file).name}")
    print()
    
    # Overall statistics
    total_sentences = 0
    overall_register_counts = Counter()
    register_combinations = Counter()
    
    # Per-file statistics
    per_file_stats = {}
    
    # Process each file
    for tsv_file in tsv_files:
        file_name = Path(tsv_file).name
        print(f"Processing {file_name}...", end=" ", flush=True)
        
        sentences = load_sentences_from_tsv(tsv_file)
        file_register_counts = Counter()
        file_combinations = Counter()
        
        for sentence in sentences:
            registers = analyze_sentence(sentence)
            
            # Count individual registers
            for register in registers:
                overall_register_counts[register.value] += 1
                file_register_counts[register.value] += 1
            
            # Count combinations (sorted tuple for consistency)
            combo = tuple(sorted([r.value for r in registers]))
            register_combinations[combo] += 1
            file_combinations[combo] += 1
        
        total_sentences += len(sentences)
        per_file_stats[file_name] = {
            'count': len(sentences),
            'registers': file_register_counts,
            'combinations': file_combinations
        }
        
        print(f"{len(sentences)} sentences")
    
    print()
    print("=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total sentences: {total_sentences:,}")
    print()
    
    print("Register Distribution:")
    print("-" * 80)
    for register, count in sorted(overall_register_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_sentences) * 100
        print(f"  {register:20} {count:8,} ({percentage:5.2f}%)")
    print()
    
    print("Top 20 Register Combinations:")
    print("-" * 80)
    for combo, count in register_combinations.most_common(20):
        percentage = (count / total_sentences) * 100
        combo_str = ", ".join(combo) if combo else "none"
        print(f"  {combo_str:40} {count:8,} ({percentage:5.2f}%)")
    print()
    
    # Per-file breakdown
    print("=" * 80)
    print("PER-FILE BREAKDOWN")
    print("=" * 80)
    for file_name, stats in per_file_stats.items():
        print(f"\n{file_name} ({stats['count']:,} sentences):")
        print("-" * 80)
        
        # Top registers
        print("  Top 5 registers:")
        for register, count in sorted(stats['registers'].items(), key=lambda x: x[1], reverse=True)[:5]:
            percentage = (count / stats['count']) * 100
            print(f"    {register:20} {count:8,} ({percentage:5.2f}%)")
        
        # Top combinations
        print("  Top 5 combinations:")
        for combo, count in stats['combinations'].most_common(5):
            percentage = (count / stats['count']) * 100
            combo_str = ", ".join(combo) if combo else "none"
            print(f"    {combo_str:40} {count:8,} ({percentage:5.2f}%)")
    
    print()
    print("=" * 80)
    print("KYOSHIGO Analysis")
    print("=" * 80)
    
    # Count sentences with KYOSHIGO
    kyoshigo_count = overall_register_counts.get('kyoshigo', 0)
    kyoshigo_percentage = (kyoshigo_count / total_sentences) * 100 if total_sentences > 0 else 0
    
    print(f"Sentences with KYOSHIGO label: {kyoshigo_count:,} ({kyoshigo_percentage:.2f}%)")
    print()
    print("KYOSHIGO combinations:")
    for combo, count in sorted(register_combinations.items(), key=lambda x: x[1], reverse=True):
        if 'kyoshigo' in combo:
            percentage = (count / total_sentences) * 100
            combo_str = ", ".join(combo)
            print(f"  {combo_str:40} {count:8,} ({percentage:5.2f}%)")


if __name__ == "__main__":
    main()
