
import sys
import os
import csv
from collections import Counter
from kotogram.sudachi_japanese_parser import SudachiJapaneseParser

# Add scripts to path
sys.path.append(os.path.abspath('scripts'))
from rule_based_analysis import analyze_register

def run_stats(tsv_path):
    parser = SudachiJapaneseParser()
    counts = Counter()
    total = 0
    
    print(f"Reading {tsv_path}...")
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if not row: continue
            # Assuming format based on typical TSV: often col 0 is sentence or ID.
            # Let's peek at the file content printed by previous run if it worked, but it failed.
            # safetfy: check if col 0 looks like a Japanese sentence (contains hiragana/kanji)
            # or if it is an ID (ascii/numeric).
            
            # Simple heuristic: if col 1 exists and col 0 is short ascii (e.g. 8 chars), maybe col 1 is sentence?
            
            if len(row) >= 3:
                sent = row[2]
            else:
                 continue
            
            total += 1
            kotogram = parser.japanese_to_kotogram(sent)
            registers = analyze_register(kotogram)
            for reg in registers:
                counts[reg.value] += 1
            
            if total % 1000 == 0:
                print(f"Processed {total}...", end='\r')
                
    print(f"\nTotal sentences: {total}")
    print("Register Stats:")
    for reg, count in counts.most_common():
        print(f"  {reg}: {count} ({count/total*100:.2f}%)")

if __name__ == "__main__":
    run_stats("data/jpn_sentences.tsv")

