#!/ reentry/env python3
"""
Cleanup script to remove confirmed agrammatic sentences from grammatic TSV files.
"""

import os
import sys
import csv
import glob
import torch
from typing import List, Set, Dict, Tuple

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
from kotogram.analysis import _load_style_model
from kotogram.model import FEATURE_FIELDS

def load_agrammatic_sentences(patterns: List[str]) -> Set[str]:
    """Load all unique Japanese sentences from agrammatic TSV files."""
    agrammatic_sentences = set()
    for pattern in patterns:
        for file_path in glob.glob(pattern):
            print(f"Loading agrammatic sentences from {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if len(row) >= 3:
                        agrammatic_sentences.add(row[2])
    print(f"Total unique agrammatic sentences: {len(agrammatic_sentences):,}")
    return agrammatic_sentences

def check_grammaticality_batch(sentences: List[str], parser: SudachiJapaneseParser, model, tokenizer, device: str) -> List[bool]:
    """Check grammaticality of a batch of sentences using the model."""
    if not sentences:
        return []
    
    # 1. Convert to kotograms
    kotograms = []
    for s in sentences:
        try:
            kotograms.append(parser.japanese_to_kotogram(s))
        except Exception:
            kotograms.append("")

    # 2. Encode
    all_feature_ids = []
    max_len = 0
    for k in kotograms:
        if not k:
            all_feature_ids.append(None)
            continue
        ids = tokenizer.encode(k, add_cls=True)
        all_feature_ids.append(ids)
        max_len = max(max_len, len(ids[FEATURE_FIELDS[0]]))

    # 3. Batch tensors
    batch_size = len(sentences)
    field_inputs = {}
    for field in FEATURE_FIELDS:
        tensor_data = []
        for ids in all_feature_ids:
            if ids is None:
                tensor_data.append([tokenizer.pad_id] * max_len)
            else:
                seq = ids[field]
                tensor_data.append(seq + [tokenizer.pad_id] * (max_len - len(seq)))
        field_inputs[f'input_ids_{field}'] = torch.tensor(tensor_data, dtype=torch.long).to(device)

    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long).to(device)
    for i, ids in enumerate(all_feature_ids):
        if ids:
            attention_mask[i, :len(ids[FEATURE_FIELDS[0]])] = 1

    # 4. Predict
    model.eval()
    with torch.no_grad():
        _, _, _, grammaticality_probs, _ = model.predict(field_inputs, attention_mask)
        # index 1 = grammatic, 0 = agrammatic
        is_grammatic = grammaticality_probs.argmax(dim=-1) == 1
        
    results = is_grammatic.cpu().tolist()
    # If parsing failed, we treat as agrammatic (since we can't confirm it's grammatic)
    # But wait, for THIS script we only want to delete if the model confirms it's AGRAMMATIC.
    # So if parsing fails, we should probably NOT delete (be conservative).
    for i, ids in enumerate(all_feature_ids):
        if ids is None:
            results[i] = True # Treat as grammatic so it's not removed
            
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cleanup confirmed agrammatic sentences from grammatic files.")
    parser.add_argument("--grammatic-data", type=str, default="data/jpn_sentences*.tsv", help="Pattern for grammatic files")
    parser.add_argument("--agrammatic-data", type=str, default="data/jpn_agrammatic*.tsv", help="Pattern for agrammatic files")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--dry-run", action="store_true", help="Don't delete, just report")
    args = parser.parse_args()

    # Load agrammatic set
    agrammatic_set = load_agrammatic_sentences([args.agrammatic_data])

    # Find grammatic files
    grammatic_files = glob.glob(args.grammatic_data)
    if not grammatic_files:
        print("No grammatic files found.")
        return

    # Initialize model
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, tokenizer = _load_style_model()
    model.to(device)
    parser = SudachiJapaneseParser()

    for file_path in grammatic_files:
        print(f"\nProcessing {file_path}...")
        sentences_to_check = []
        indices_to_check = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            all_rows = list(reader)

        print(f"Total rows: {len(all_rows):,}")
        
        for i, row in enumerate(all_rows):
            if len(row) < 3:
                continue
            
            sentence = row[2]
            if sentence in agrammatic_set:
                sentences_to_check.append(sentence)
                indices_to_check.append(i)

        if not sentences_to_check:
            print("No overlaps found.")
            continue

        print(f"Found {len(sentences_to_check):,} overlaps to check.")
        
        # Batch analysis
        confirmed_agrammatic_mask = [False] * len(all_rows)
        for i in range(0, len(sentences_to_check), args.batch_size):
            batch = sentences_to_check[i : i + args.batch_size]
            batch_results = check_grammaticality_batch(batch, parser, model, tokenizer, device)
            for j, is_gram in enumerate(batch_results):
                if not is_gram: # model says agrammatic
                    original_idx = indices_to_check[i + j]
                    confirmed_agrammatic_mask[original_idx] = True
            
            if (i // args.batch_size) % 10 == 0:
                print(f"  Processed {min(i + args.batch_size, len(sentences_to_check)):,}/{len(sentences_to_check):,}...", end="\r")
        print(f"  Processed {len(sentences_to_check):,}/{len(sentences_to_check):,}... Done.")

        # Final filtering
        final_rows = []
        removed_count = 0
        for i, row in enumerate(all_rows):
            if i < len(confirmed_agrammatic_mask) and confirmed_agrammatic_mask[i]:
                removed_count += 1
                continue
            final_rows.append(row)

        print(f"Sentences removed: {removed_count:,}")
        
        if not args.dry_run:
            temp_path = file_path + ".tmp"
            print(f"Writing to {temp_path}...")
            with open(temp_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, delimiter='\t', lineterminator='\n')
                writer.writerows(final_rows)
            
            print(f"Renaming {temp_path} to {file_path}...")
            os.replace(temp_path, file_path)
        else:
            print("[Dry Run] Skipping file overwrite.")

    print("\nCleanup complete.")

if __name__ == "__main__":
    main()
