#!/usr/bin/env python3
"""Generate counter variations (both agrammatic and grammatic).

This script takes sentences from jpn_sentences.tsv and programmatically generates
variations by swapping counters (助数詞).

Modes:
- agrammatic: Swaps counters with incorrect ones (avoiding synonyms).
- grammatic: Swaps counters with valid ones (synonyms or generic generalizations).

Usage:
    python scripts/generate_agrammatic_counters.py --input data/jpn_sentences.tsv --output data/out.tsv --mode agrammatic --target-count 60000
"""

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
import argparse
import csv
import random
import re
from typing import List, Tuple, Optional, Dict, Callable, Set, Any

from kotogram import split_kotogram, extract_token_features, kotogram_to_japanese


# ============================================================================
# CONSTANTS & UTILITIES
# ============================================================================

# Common counters to use for swapping (Harvested from dataset)
COMMON_COUNTERS = [
    'GB', 'L', 'MB', 'V', 'cc', 'g', 'k', 'kg', 'km', 'm', 'ml', 'mm', 
    'えん', 'かい', 'か国', 'か所', 'か月', 'きれ', 'こ', 'さい', 'さつ', 'じ', 
    'せき', 'そう', 'たび', 'つ', 'てき', 'ど', 'ひき', 'ぴき', 'ぺん', 'まい', 
    'まん', 'カ国', 'カ所', 'カ月', 'キロ', 'コ', 'トン', 'ドル', 'ペソ', 'ミリ', 
    'モル', 'ヵ国', 'ヵ月', 'ヶ国', 'ヶ月', '丁', '丁目', '世', '両', '中', '乗', 
    '事', '人', '人組', '代', '件', '位', '体', '作', '個', '倍', '元', '光年', 
    '兎', '児', '党', '円', '冊', '刀', '分', '分け', '分間', '切れ', '割', '匹', 
    '升', '厘', '台', '号', '号室', '号機', '号線', '合', '名', '周', '周年', '品', 
    '問', '回', '回分', '回戦', '国', '坪', '型', '声', '子', '室', '寸', '対', 
    '局', '巻', '年', '年次', '年生', '年間', '床', '度', '度目', '性', '戸', '才', 
    '打', '把', '択', '振り', '敗', '斤', '日', '日間', '時', '時半', '時間', '月', 
    '服', '期', '本', '束', '条', '杯', '枚', '校', '株', '桁', '機', '次', '歩', 
    '歳', '歳児', '段', '段式', '泊', '流', '滴', '点', '片', '版', '球', '生', '田', 
    '画', '番', '番目', '畳', '発', '目', '着', '石', '社', '票', '秒', '秒間', '種', 
    '等', '箇所', '箇月', '節', '篇', '級', '組', '線', '編', '缶', '羽', '者', '脚', 
    '膳', '艘', '行', '袋', '角形', '課', '足', '軒', '輪', '通', '通り', '週', '週間', 
    '進', '過ぎ', '部', '里', '重', '針', '銭', '錠', '間', '階', '隻', '面', '頁', 
    '頭', '題', '食', '飯', '首', '馬身', 'Ｋｍ', 'Ｍ', 'Ｎ', 'ｃｍ', 'ｇ', 'ｋｍ', 'ｍ',
    # Newly discovered from runtime checks
    'インチ', 'フィート', 'メートル', 'ダース', 'ポンド', 'ヤード', '％', 'ヤール', 
    'マイル', 'ペニー', '年契約', 'パーセント', 'ページ', 'センチ', 'セント', 'エーカー', 
    'ポイント', 'バレル', '＄', 'リッター', 'ペンス', '℃', 'キログラム', '平方メートル', 
    'ボルト', 'クオート', '%', '年問題', 'ガロン', 'メーター', 'キロメートル', 'リットル', 
    'パイント', 'グラム', 'ギガバイト', 'ワット', 'つがい', 'ユーロ', 'センチメートル', 
    'ヘクタール', 'ビット', 'ちゃん', 'ドラクマ',
    'つ折り', 'オンス'
]

# Counters that are "generic" and often overused/misused by learners
GENERIC_COUNTERS = ['つ', '個']

# Set to track printed unknown counters in worker process
_unknown_counters_seen = set()

def check_and_register_counter(surface: str):
    """Check if counter is known, if not print and add to list."""
    if surface not in COMMON_COUNTERS:
        if surface not in _unknown_counters_seen:
            # print(f"Unrecognized counter encountered: {surface}")
            _unknown_counters_seen.add(surface)
        COMMON_COUNTERS.append(surface)


# Sets of counters that are synonyms or near-synonyms.
# Swapping between these might result in a grammatically correct sentence.
# We should AVOID swapping items within the same set.
CONFLICTING_COUNTERS = [
    {'人', '名'},                 # People
    {'歳', '才'},                 # Age
    {'回', '度'},                 # Times/Frequency
    {'本', '束'},                 # Bunches/Long things (sometimes similar)
    {'匹', '頭', '羽'},           # Animals (sometimes interchangeable depending on size/context)
    {'ヶ月', 'ヵ月', 'カ月', 'か月'}, # Months duration variants
    {'か国', 'カ国', 'ヵ国', 'ヶ国'}, # Countries variants
    {'分', '分間'},               # Minutes vs Duration
    {'秒', '秒間'},               # Seconds vs Duration
    {'週', '週間'},               # Weeks vs Duration
    {'年', '年間'},               # Years vs Duration
    {'日', '日間'},               # Days vs Duration
    {'時', '時間'},               # Hours (Time point vs Duration - sometimes swappable)
]

def get_conflicting_counters(surface: str) -> set:
    """Return a set of counters that conflict with the given surface form."""
    conflicts = {surface}
    for group in CONFLICTING_COUNTERS:
        if surface in group:
            conflicts.update(group)
    return conflicts

# Synonyms that are safely swappable in most contexts (for grammatic generation)
TRUE_SYNONYMS = [
    {'人', '名'},                 # People
    {'歳', '才'},                 # Age
    {'回', '度'},                 # Times/Frequency
    {'ヶ月', 'ヵ月', 'カ月', 'か月'}, # Months duration variants
    {'か国', 'カ国', 'ヵ国', 'ヶ国'}, # Countries variants
]

def get_true_synonyms(surface: str) -> set:
    """Return a set of true synonyms for the given surface."""
    syns = set()
    for group in TRUE_SYNONYMS:
        if surface in group:
            syns.update(group)
    return syns

# Counters that should NOT be swapped to generic 'つ' or '個' because it's invalid/unnatural
# These are used to determine if a generic swap is Valid (Grammatic) or Invalid (Agrammatic)
NO_GENERIC_SWAP_COUNTERS = {
    '年', '月', '日', '時', '分', '秒', 
    '年間', '日間', '時間', '分間', '秒間', '週間',
    'ヶ月', 'ヵ月', 'カ月', 'か月',
    '歳', '才', 
    '円', 'ドル', 'ペソ', 'ユーロ', 
    'ページ', '％', 'パーセント', '度', '回', '点', '番', '級',
    '代',
    '倍', '歩', '周年', '期', '石',
    # Metric / Physical Units
    'm', 'mm', 'cm', 'km', 'kg', 'g', 'mg', 'l', 'ml', 'cc', 
    'MB', 'GB', 'TB', 'V', 'W', 'A', 'Hz', 'dB',
    'Ｋｍ', 'Ｍ', 'Ｎ', 'ｃｍ', 'ｇ', 'ｋｍ', 'ｍ',
    'メートル', 'キロ', 'センチ', 'ミリ', 'グラム', 'キログラム', 'トン',
    'リットル', 'ガロン', 'エーカー', 'ヘクタール', 'マイル', 'ヤード',
    'フィート', 'インチ', 'ボルト', 'ワット', 'アンペア',
    # People (Dehumanizing to use generic counters)
    # People (Dehumanizing to use generic counters)
    '人', '名',
    # Ani
    # Animals (Usually wrong to use Generic '個' for animate objects)
    '匹', '頭', '羽', 
}

# ...

def is_age_context(tokens: List[str], index: int) -> bool:
    """Check if the context around the counter implies Age or Abstract difference (where '個' is invalid)."""
    # Check next token (most common: 2つ年下, 2つ違い)
    if index + 1 < len(tokens):
        next_surface = extract_surface(tokens[index+1])
        if '年' in next_surface or '違' in next_surface or '離' in next_surface:
            return True
    return False

def generate_grammatic_candidates(kotogram: str, num_variations: int = 1) -> List[Tuple[str, str]]:
    """Generate GRAMMATIC (valid) variations using synonyms or generics."""
    tokens = split_kotogram(kotogram)
    if not tokens:
        return []

    candidates = []
    indices = []
    for i, token in enumerate(tokens):
        if is_counter(token):
            if i > 0 and is_numeral(tokens[i-1]):
                indices.append(i)

    if not indices:
        return []

    attempts = 0
    max_attempts = num_variations * 10

    while len(candidates) < num_variations and attempts < max_attempts:
        attempts += 1
        idx = random.choice(indices)
        token = tokens[idx]
        surface = extract_surface(token)
        
        check_and_register_counter(surface)

        opts = []
        
        # Option 1: Valid Synonym Swap (Strict)
        # We use TRUE_SYNONYMS instead of CONFLICTING_COUNTERS to avoid Date/Duration mixups
        syns = get_true_synonyms(surface)
        synonyms = [c for c in syns if c != surface]
        if synonyms:
             target = random.choice(synonyms)
             opts.append((target, f'synonym_{surface}_to_{target}'))
             
        # Option 2: Generic Swaps (Specific->Generic OR Generic->Generic)
        # Valid if counter is NOT in blacklist (NO_GENERIC_SWAP_COUNTERS)
        if surface not in NO_GENERIC_SWAP_COUNTERS:
             if surface in GENERIC_COUNTERS:
                 # Generic -> Generic
                 targets = [g for g in GENERIC_COUNTERS if g != surface]
             else:
                 # Specific -> Generic
                 targets = GENERIC_COUNTERS
             
             if targets:
                 # Special Linguistic Check:
                 # If we are swapping 'つ' -> '個', we must ensure it's NOT an age/abstract context.
                 # "2つ年下" (Valid) -> "2個年下" (Invalid/Agrammatic).
                 # So if context is Age, we BLOCK '個' generation here.
                 if surface == 'つ' and '個' in targets:
                     if is_age_context(tokens, idx):
                         # This is an age context. 'つ' is valid (like Age), but '個' is NOT.
                         # So we cannot generate '個' here as a Grammatic variant.
                         targets = [t for t in targets if t != '個']
                         
                 if targets:
                     target = random.choice(targets)
                     opts.append((target, f'valid_generic_{target}_instead_of_{surface}'))
             
        if not opts:
            continue
            
        target_counter, type_str = random.choice(opts)
        new_token = replace_surface_in_token(token, target_counter)
        new_tokens = tokens[:idx] + [new_token] + tokens[idx+1:]
        res = (''.join(new_tokens), type_str)
        if res not in candidates:
            candidates.append(res)
            
    return candidates


def extract_surface(token: str) -> str:
    """Extract surface form from a kotogram token."""
    return extract_token_features(token)['surface']


def extract_pos_details(token: str) -> Tuple[str, str, str]:
    """Extract POS details from a kotogram token."""
    features = extract_token_features(token)
    return features.get('pos', ''), features.get('pos_detail1', ''), features.get('pos_detail2', '')


def is_counter(token: str) -> bool:
    """Check if a token is a counter (助数詞)."""
    pos, detail1, detail2 = extract_pos_details(token)
    
    # Direct check for Sudachi POS structure
    if detail1 == 'counter' or detail2 == 'counter' or detail2 == 'counter-possible':
        return True
    
    # Some counters might be classified as noun suffixes
    if pos == 'suff' and (detail1 == 'noun_like' or detail1 == 'counter'):
        return True

    return False


def replace_surface_in_token(token: str, new_surface: str) -> str:
    """Replace the surface form in a kotogram token while keeping other tags."""
    # Kotogram format: ⌈ˢsurfaceᵖpos:detail...⌉
    # We want to replace 'surface'
    return re.sub(r'ˢ[^ᵖ]+ᵖ', f'ˢ{new_surface}ᵖ', token)


# ============================================================================
# ERROR GENERATORS
# ============================================================================

def is_numeral(token: str) -> bool:
    """Check if a token is a numeral."""
    features = extract_token_features(token)
    return features.get('pos_detail1') == 'numeral'


def generate_agrammatic_candidates(kotogram: str, num_variations: int = 1) -> List[Tuple[str, str]]:
    """Generate agrammatic variations for a sentence."""
    tokens = split_kotogram(kotogram)
    if not tokens:
        return []

    candidates = []
    
    # Identify counters available for swapping
    indices = []
    for i, token in enumerate(tokens):
        if is_counter(token):
            if i > 0 and is_numeral(tokens[i-1]):
                indices.append(i)
                
    if not indices:
        return []
        
    attempts = 0
    max_attempts = num_variations * 10
    
    while len(candidates) < num_variations and attempts < max_attempts:
        attempts += 1
        idx = random.choice(indices)
        token = tokens[idx]
        surface = extract_surface(token)
        
        check_and_register_counter(surface)
        
        # Don't swap if it's not a recognizable counter surface
        if len(surface) > 2 or not any(c.isalnum() for c in surface):
             continue
             
        # Decide type of error: Specific Swap (80%) or Generic (20%)
        error_type_val = random.random()
        
        if error_type_val < 0.2:
            # Generic Error: Replace specific with generic
            # BUT: If it's a VALID generic swap (i.e. not in blacklist), SKIP IT.
            # Because valid swaps belong in the Grammatic dataset.
            if surface in GENERIC_COUNTERS:
                continue
            
            if surface not in NO_GENERIC_SWAP_COUNTERS:
                # This is a valid swap (e.g. 本 -> 個), so it's Grammatic.
                # Do NOT generate it here.
                continue

            target_counter = random.choice(GENERIC_COUNTERS)
            
            # Special Linguistic Check for Agrammatic Mode:
            # If we are swapping '歳'/'才' -> 'つ', and it IS an age context (e.g. 2つ年下),
            # then the result is VALID (Grammatic).
            # We must SKIP this, because we want to generate Agrammatic sentences here.
            if target_counter == 'つ' and (surface == '歳' or surface == '才'):
                if is_age_context(tokens, idx):
                    continue
            
            new_token = replace_surface_in_token(token, target_counter)
            new_tokens = tokens[:idx] + [new_token] + tokens[idx+1:]
            res = (''.join(new_tokens), f'generic_{target_counter}_instead_of_{surface}')
            if res not in candidates:
                candidates.append(res)
        else:
            # Specific Swap Error: Replace with random OTHER counter
            if surface in GENERIC_COUNTERS:
                 # Don't swap FROM generic for agrammatic examples as discussed
                 continue
                 
            conflicts = get_conflicting_counters(surface)
            # Pick a random target counter that is DIFFERENT from current AND not a synonym
            # IMPORTANT: Exclude GENERIC_COUNTERS from random specific targets to avoid collision with Grammatic set
            potential_replacements = [c for c in COMMON_COUNTERS if c not in conflicts and c not in GENERIC_COUNTERS]
            if not potential_replacements:
                continue
                
            target_counter = random.choice(potential_replacements)
            new_token = replace_surface_in_token(token, target_counter)
            new_tokens = tokens[:idx] + [new_token] + tokens[idx+1:]
            res = (''.join(new_tokens), f'swap_{surface}_to_{target_counter}')
            if res not in candidates:
                candidates.append(res)

    return candidates




# ============================================================================
# MULTIPROCESSING WORKERS
# ============================================================================

_worker_parser = None
_worker_mode = None
_worker_variations = 1

def init_worker(mode: str, variations: int):
    """Initialize worker process."""
    global _worker_parser, _worker_mode, _worker_variations
    from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
    _worker_parser = SudachiJapaneseParser()
    _worker_mode = mode
    _worker_variations = variations


def process_sentence_worker(row: List[str]) -> List[Tuple[str, str, str, str, str, str]]:
    """Process a single row in a worker process."""
    if len(row) < 3:
        return []
    
    sentence_id, lang, sentence = row[0], row[1], row[2]
    if lang != 'jpn':
        return []

    results = []
    try:
        kotogram = _worker_parser.japanese_to_kotogram(sentence)
        
        if _worker_mode == 'agrammatic':
            candidates = generate_agrammatic_candidates(kotogram, num_variations=_worker_variations)
        elif _worker_mode == 'grammatic':
            candidates = generate_grammatic_candidates(kotogram, num_variations=_worker_variations)
        else:
            return []
            
        for new_kotogram, error_type in candidates:
             new_surface = kotogram_to_japanese(new_kotogram, spaces=False)
             # Even if surface is same (unlikely given checks), exclude
             if new_surface != sentence:
                 results.append((None, 'jpn', new_surface, sentence_id, new_kotogram, error_type))
                 
    except Exception:
        pass
        
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate counter variations")
    parser.add_argument("--input", "-i", type=str, default="data/jpn_sentences.tsv", help="Input TSV")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output TSV")
    parser.add_argument("--mode", type=str, choices=['agrammatic', 'grammatic'], required=True, help="Generation mode")
    parser.add_argument("--target-count", type=int, default=1000, help="Target number of output sentences")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    print(f"Reading from {args.input}...")
    all_rows = []
    with open(args.input, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            all_rows.append(row)
            
    total_sentences = len(all_rows)
    print(f"Loaded {total_sentences} source sentences.")
    
    # Calculate how many variations per sentence we need on average
    # Add a buffer since not all sentences have counters
    # Heuristic: ~2% of sentences might have counters suitable for swapping (conservative estimate)
    estimated_hit_rate = 0.02
    needed_per_hit = (args.target_count / (total_sentences * estimated_hit_rate))
    variations_per_sentence = max(1, int(needed_per_hit * 2.0)) # Higher safety buffer
    
    print(f"Target: {args.target_count}. Strategy: {variations_per_sentence} vars/sentence.")
    
    ctx = mp.get_context('spawn')
    num_workers = max(1, mp.cpu_count() - 1)
    
    generated_count = 0
    
    with open(args.output, 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        
        # Shuffle inputs so we get random distribution if we stop early? 
        # Actually random sampling is better if target < total potential.
        # But we process all in parallel.
        
        with ctx.Pool(num_workers, initializer=init_worker, initargs=(args.mode, variations_per_sentence)) as pool:
            for chunk_results in pool.imap_unordered(process_sentence_worker, all_rows, chunksize=100):
                 if generated_count >= args.target_count:
                     break
                     
                 for res in chunk_results:
                     if generated_count >= args.target_count:
                         break
                         
                     new_id = f"{'err' if args.mode == 'agrammatic' else 'aug'}_cnt_{generated_count}"
                     row_to_write = [new_id] + list(res[1:])
                     writer.writerow(row_to_write)
                     generated_count += 1
                     
                     if generated_count % 1000 == 0:
                         print(f"Generated {generated_count} samples...", end='\r')

    print(f"\nDone! Generated {generated_count} {args.mode} counter examples.")
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
