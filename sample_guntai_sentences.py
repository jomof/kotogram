import sys
import os
import csv
from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
from kotogram.analysis import RegisterLevel

# Add scripts to path
sys.path.append(os.path.abspath('scripts'))
from rule_based_analysis import analyze_register

def sample_guntai(tsv_path, limit=20):
    parser = SudachiJapaneseParser()
    found = 0
    
    print(f"Sampling Guntai sentences from {tsv_path}...")
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if not row or len(row) < 3: continue
            sent = row[2]
            
            kotogram = parser.japanese_to_kotogram(sent)
            registers = analyze_register(kotogram)
            
            if RegisterLevel.GUNTAI in registers:
                print(f"- {sent}")
                found += 1
                if found >= limit:
                    break

if __name__ == "__main__":
    sample_guntai("data/jpn_sentences.tsv")
