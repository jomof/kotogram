#!/usr/bin/env python3
import re
import sys
import os
import glob

def split_japanese_sentences(text):
    if not text or len(text) <= 25:
        return [text] if text else []
    
    # 1. Split at terminal punctuation (one or more) followed by optional closing characters.
    TERMINALS = r'[。！？.?!]'
    CLOSING = r'[」』)\]]'
    
    pattern = f'({TERMINALS}+{CLOSING}*)'
    parts = re.split(pattern, text)
    
    sentences = []
    current = ""
    quote_depth = 0
    
    # re.split with capturing group returns: [text, punct, text, punct, ...]
    for i in range(0, len(parts) - 1, 2):
        text_part = parts[i]
        punct_part = parts[i+1]
        
        quote_depth += text_part.count('「') + text_part.count('『')
        quote_depth -= text_part.count('」') + text_part.count('』')
        quote_depth += punct_part.count('「') + punct_part.count('『')
        quote_depth -= punct_part.count('」') + punct_part.count('』')
        
        current += text_part + punct_part
        
        should_split = False
        if i + 2 < len(parts):
            next_text = parts[i+2]
            
            if next_text.startswith(('と', 'だ', 'が', 'に', 'の', 'て', 'で', 'は', 'を')):
                should_split = False
            elif punct_part.endswith(('」', '』')) and next_text.strip().startswith(('「', '『')):
                should_split = True
            elif '。' in punct_part and quote_depth <= 0:
                should_split = True
            elif not next_text.strip():
                should_split = True
            elif quote_depth <= 0:
                if next_text and not next_text.startswith(('と', 'だ')):
                    should_split = True
        else:
            should_split = True

        if should_split:
            sentences.append(current.strip())
            current = ""
        
    if len(parts) % 2 == 1:
        current += parts[-1]
    
    if current.strip():
        # Clean up: if current is just punctuation, merge with last sentence
        if sentences and re.fullmatch(r'[。！？.?! \t\.]+', current.strip()):
            sentences[-1] += current.strip()
        else:
            sentences.append(current.strip())
        
    result = [s for s in sentences if s]
    
    # NEW: Also not split if any of the split-out sentences <= 5 length
    if any(len(s) <= 5 for s in result):
        return [text]
        
    return result

def process_file(filepath):
    print(f"Processing {filepath}...")
    temp_path = filepath + ".tmp"
    
    lines_written = 0
    original_lines = 0
    
    with open(filepath, 'r', encoding='utf-8') as f_in, \
         open(temp_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip():
                f_out.write(line)
                continue
                
            original_lines += 1
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3:
                f_out.write(line)
                continue
                
            id_val = parts[0]
            lang = parts[1]
            sentence = parts[2]
            extra = parts[3:]
            
            split_sentences = split_japanese_sentences(sentence)
            
            if len(split_sentences) <= 1:
                f_out.write(line)
                lines_written += 1
            else:
                for idx, s in enumerate(split_sentences, 1):
                    new_id = f"{id_val}_{idx}_of_{len(split_sentences)}"
                    new_line_parts = [new_id, lang, s] + extra
                    f_out.write('\t'.join(new_line_parts) + '\n')
                    lines_written += 1
                    
    os.replace(temp_path, filepath)
    print(f"  Done. Original lines: {original_lines}, New lines: {lines_written}")

def main():
    files = glob.glob('data/jpn_*.tsv')
    if not files:
        print("No jpn_*.tsv files found in data/")
        return
        
    for filepath in sorted(files):
        process_file(filepath)

if __name__ == "__main__":
    main()
