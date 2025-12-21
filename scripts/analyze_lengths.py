
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from kotogram.model import Tokenizer
from scripts.train_style import StyleDataset

def analyze():
    print("Loading dataset for analysis (this may take a moment)...")
    tokenizer = Tokenizer()
    # Load a representative sample (e.g. 20% or 100% depending on speed)
    # Using 100% for accuracy, assuming cache exists or it's fast enough.
    # If it's too slow, the user can interrupt or we can reduce ratio.
    dataset = StyleDataset.from_tsv(
        "data/jpn_sentences.tsv", 
        tokenizer, 
        labeled=True,
        verbose=True
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    lengths = [s.seq_len for s in dataset.samples]
    lengths = np.array(lengths)
    
    print("\nSentence Length Statistics (Tokens):")
    print(f"  Min: {np.min(lengths)}")
    print(f"  Max: {np.max(lengths)}")
    print(f"  Mean: {np.mean(lengths):.2f}")
    print(f"  Median: {np.median(lengths):.2f}")
    
    percentiles = [50, 75, 90, 95, 99, 99.5, 99.9]
    print("\nPercentiles:")
    for p in percentiles:
        val = np.percentile(lengths, p)
        print(f"  {p}th: {val:.1f}")
        
    # Count samples above certain thresholds
    thresholds = [32, 48, 64, 80, 96, 128, 256, 512]
    print("\nImpact of Cutoffs (Excluded Samples):")
    for t in thresholds:
        count = np.sum(lengths > t)
        percent = (count / len(lengths)) * 100
        print(f"  > {t:3d}: {count:5d} samples ({percent:5.2f}%)")

if __name__ == "__main__":
    analyze()
