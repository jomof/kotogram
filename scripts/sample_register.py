#!/usr/bin/env python3
"""Sample sentences with specific register labels for quality checking."""

import sys
import random
import glob
from pathlib import Path
from typing import List, Set, Tuple

# Add project root to path to enable imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kotogram.analysis import RegisterLevel
from scripts.register_stats import load_sentences_from_tsv, analyze_sentence


def find_register_sentences(tsv_files: List[str], target_register: str, sample_size: int = 10) -> List[Tuple[str, Set[RegisterLevel]]]:
    """Find sentences matching a specific register.
    
    Returns:
        List of (sentence, registers) tuples
    """
    matching = []
    
    for tsv_file in tsv_files:
        sentences = load_sentences_from_tsv(tsv_file)
        
        for sentence in sentences:
            registers = analyze_sentence(sentence)
            
            # Check if target register is in the detected set
            if any(target_register.lower() in str(r).lower() for r in registers):
                matching.append((sentence, registers))
    
    # Random sample
    if len(matching) > sample_size:
        matching = random.sample(matching, sample_size)
    
    return matching


def main():
    register = sys.argv[1] if len(sys.argv) > 1 else "kyoshigo"
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    # Find all jpn_sentences*.tsv files
    data_dir = project_root / "data"
    tsv_files = sorted(glob.glob(str(data_dir / "jpn_sentences*.tsv")))
    
    print(f"Sampling {sample_size} sentences with register: {register.upper()}")
    print(f"Searching in {len(tsv_files)} TSV files...")
    print()
    
    samples = find_register_sentences(tsv_files, register, sample_size)
    
    if not samples:
        print(f"No sentences found with register: {register}")
        return
    
    print(f"Found {len(samples)} sample(s):")
    print("=" * 80)
    
    for i, (sentence, registers) in enumerate(samples, 1):
        register_str = ", ".join(sorted([r.value for r in registers]))
        print(f"\n{i}. {sentence}")
        print(f"   Detected: {register_str}")
    
    print()


if __name__ == "__main__":
    main()
