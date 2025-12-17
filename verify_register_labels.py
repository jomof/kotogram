
import sys
import os
import csv
from typing import Set

# Add paths
sys.path.append(os.path.abspath('scripts'))
from rule_based_analysis import analyze_register
from kotogram.analysis import RegisterLevel
from kotogram.sudachi_japanese_parser import SudachiJapaneseParser

def normalize_register_name(name: str) -> str:
    # Extract register name from ID (e.g. sonkeigo_001 -> sonkeigo)
    parts = name.split('_')
    return parts[0]

def verify_registers(tsv_path):
    parser = SudachiJapaneseParser()
    
    print(f"Verifying {tsv_path}...\n")
    print(f"{'ID':<15} {'Sentence':<40} {'Expected':<12} {'Detected':<30} {'Result'}")
    print("-" * 110)
    
    label_map = {
        'sonkeigo': RegisterLevel.SONKEIGO,
        'kenjogo': RegisterLevel.KENJOGO,
        'kansaiben': RegisterLevel.KANSAIBEN,
        'hakataben': RegisterLevel.HAKATABEN,
        'kyoshigo': RegisterLevel.KYOSHIGO,
        'netslang': RegisterLevel.NETSLANG,
        'neutral': RegisterLevel.NEUTRAL,
    }

    correct_count = 0
    total_count = 0

    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 3: continue
            
            row_id = row[0]
            sentence = row[2]
            
            expected_str = normalize_register_name(row_id)
            expected_enum = label_map.get(expected_str)
            
            if not expected_enum:
                print(f"Skipping unknown register ID prefix: {row_id}")
                continue

            kotogram = parser.japanese_to_kotogram(sentence)
            detected_set: Set[RegisterLevel] = analyze_register(kotogram)
            
            # Check if expected is in detected set
            # For Neutral, we might expect ONLY neutral, or allow neutral + others? 
            # Usually if specific register is present, Neutral is implicit/base, but our set returns specific ones.
            # If nothing specific found -> Neutral.
            
            match = expected_enum in detected_set
            
            # Special case: If we expect Neutral, we want {NEUTRAL} exactly (or maybe subset?)
            # Currently analyze_register returns {NEUTRAL} if no other features found.
            # But if we have mixed sentence, it might have features.
            # Let's stick to "Is expected label in detected set?" logic for now.
            # Actually for 'neutral' examples, if we detect 'sonkeigo', that's a false positive.
            if expected_enum == RegisterLevel.NEUTRAL:
                match = (detected_set == {RegisterLevel.NEUTRAL})
            
            result_str = "PASS" if match else "FAIL"
            if match: correct_count += 1
            total_count += 1
            
            detected_str = ", ".join([r.value for r in detected_set])
            
            # Highlight FAILs
            if not match:
                result_str = f"FAIL << Expected {expected_str}"
                print(f"{row_id:<15} {sentence[:35]:<40} {expected_str:<12} {detected_str:<30} {result_str}")
                print(f"   Koto: {kotogram}")
            else:
                print(f"{row_id:<15} {sentence[:35]:<40} {expected_str:<12} {detected_str:<30} {result_str}")
            
    print("-" * 110)
    print(f"Accuracy: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")

if __name__ == "__main__":
    verify_registers("data/jpn_sentences_register.tsv")
