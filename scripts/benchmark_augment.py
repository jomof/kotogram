import sys
import os

# Optimize for parallel execution of neural models
os.environ["TORCH_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time
import csv
import csv
import random
from statistics import mean
from concurrent.futures import ProcessPoolExecutor, as_completed
from kotogram.augment import Augmenter, split_kotogram, extract_token_features, get_surface, Token
from dataclasses import asdict

# Initialize a global augmenter for worker processes
_worker_augmenter = None

def get_augmenter() -> Augmenter:
    global _worker_augmenter
    if _worker_augmenter is None:
        _worker_augmenter = Augmenter()
    return _worker_augmenter

def process_single_sentence(
    sentence: str,
    augmenter: Augmenter,
    limit_per_sentence: int, # This argument is added but not used in the current logic
    timeout: float = 1.0
) -> Dict[str, Any]:
    try:
        parser = augmenter.get_parser()
        start_s = time.time()

        # 0. Check initial grammaticality
        k_orig = parser.japanese_to_kotogram(sentence)
        analysis_orig = augmenter.grammar(k_orig)
        is_orig_gramm = analysis_orig.is_grammatic
        is_orig_pragm = (analysis_orig.formality.value != 'unpragmatic_formality' and
                        analysis_orig.gender.value != 'unpragmatic_gender')

        is_orig_valid = is_orig_gramm and is_orig_pragm

        deadline = start_s + timeout

        # 1. Parsing for augmentation
        kotogram = parser.japanese_to_kotogram(sentence)
        tokens_kotogram = split_kotogram(kotogram)
        token_features = []
        for t in tokens_kotogram:
            f = extract_token_features(t)
            token_features.append(Token(f.surface, asdict(f)))

        token_tuple = tuple(token_features)

        # 2. Augment (respecting timeout)
        start_aug = time.time()
        aug_result = augmenter.augment_tokens(token_tuple, deadline=deadline)
        candidate_surfaces = set()
        surface_to_rules = {}

        for aug_tuple in aug_result.candidates:
            surface = "".join(get_surface(token) for token in aug_tuple)
            candidate_surfaces.add(surface)
            surface_to_rules[surface] = aug_result.provenance.get(aug_tuple, set())

        num_candidates = len(candidate_surfaces)
        aug_duration = time.time() - start_aug

        # 3. Filter (if time remains)
        timed_out = False
        start_filter = time.time()
        rule_total = {} # rule -> count of candidates
        rule_valid = {} # rule -> count of valid

        # Pre-aggregate rule totals for this sentence
        for rules in surface_to_rules.values():
            for r in rules:
                rule_total[r] = rule_total.get(r, 0) + 1

        if time.time() < deadline:
            valid_sentences = augmenter.filter_grammatical(candidate_surfaces, deadline=deadline)
            num_valid = len(valid_sentences)
            if time.time() > deadline:
                timed_out = True

            # Aggregate valid rules
            for s_valid in valid_sentences:
                rules = surface_to_rules.get(s_valid, set())
                for r in rules:
                    rule_valid[r] = rule_valid.get(r, 0) + 1
        else:
            timed_out = True
            num_valid = 0

        filter_duration = time.time() - start_filter
        total_duration = time.time() - start_s

        return {
            'sentence': sentence,
            'is_orig_valid': is_orig_valid,
            'is_orig_gramm': is_orig_gramm,
            'is_orig_pragm': is_orig_pragm,
            'candidates': num_candidates,
            'valid': num_valid,
            'invalid_removed': num_candidates - num_valid if not timed_out else 0,
            'rule_total': rule_total,
            'rule_valid': rule_valid,
            'aug_duration': aug_duration,
            'filter_duration': filter_duration,
            'total_duration': total_duration,
            'timed_out': timed_out,
            'error': None
        }
    except Exception as e:
        import traceback
        return {
            'sentence': sentence,
            'error': f"{str(e)}\n{traceback.format_exc()}"
        }

def print_stats(name: str, data: List[float]) -> None:
    data = sorted(data)
    n = len(data)
    if n == 0:
        return
    print(f"\n{name}:")
    print(f"  Min:    {min(data)}")
    print(f"  P25:    {data[int(n*0.25)]}")
    print(f"  Median: {data[int(n*0.50)]}")
    print(f"  P75:    {data[int(n*0.75)]}")
    print(f"  P90:    {data[int(n*0.90)]}")
    print(f"  P99:    {data[int(n*0.99)] if n > 100 else data[-1]}")
    print(f"  Max:    {max(data)}")
    print(f"  Mean:   {mean(data):.2f}")

def load_sentences(filepath: str, limit: int = 200000) -> List[str]:
    all_sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 3:
                all_sentences.append(row[2])
            if len(all_sentences) >= limit:
                break
    return all_sentences

def main(limit: int = 1000) -> None:
    tsv_path = "data/jpn_sentences.tsv"
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    if not os.path.exists(tsv_path):
        print(f"Error: {tsv_path} not found")
        return

    # Read ALL sentences (column 3)
    all_sentences = load_sentences(tsv_path)

    if limit < len(all_sentences):
        print(f"Sampling {limit} random sentences from {len(all_sentences)} (Fixed Seed 42)...")
        random.seed(42)
        sentences = random.sample(all_sentences, limit)
    else:
        sentences = all_sentences

    stats = []
    crashes = []
    zero_valid = []
    recovered_samples = [] # Sentence that was invalid but got valid variations
    orig_ungrammatic_count = 0
    orig_unpragmatic_count = 0
    
    start_all = time.time()
    num_cpus = os.cpu_count()
    print(f"Using {num_cpus} parallel processes...")

    timeout_count = 0
    global_rule_total = {}
    global_rule_valid = {}

    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = {executor.submit(process_single_sentence, s): s for s in sentences}
        
        try:
            for i, future in enumerate(as_completed(futures, timeout=300)):
                if i > 0 and i % 100 == 0:
                    print(f"Completed {i}/{len(sentences)}...")
                
                res = future.result()
                if res['error']:
                    crashes.append(res)
                else:
                    stats.append(res)
                    if res['timed_out']:
                        timeout_count += 1
                    
                    # Aggregate rule stats
                    for r, count in res.get('rule_total', {}).items():
                        global_rule_total[r] = global_rule_total.get(r, 0) + count
                    for r, count in res.get('rule_valid', {}).items():
                        global_rule_valid[r] = global_rule_valid.get(r, 0) + count
                    
                    if res['total_duration'] > 1.1:
                        print(f"SLOW ({res['total_duration']:.2f}s): Aug={res['aug_duration']:.2f}s, Filter={res['filter_duration']:.2f}s | {res['sentence']}")
                    if not res['is_orig_gramm']:
                        orig_ungrammatic_count += 1
                    if not res['is_orig_pragm']:
                        orig_unpragmatic_count += 1
                    
                    if not res['is_orig_valid'] and res['valid'] > 0:
                        recovered_samples.append(res['sentence'])
                    
                    if res['valid'] == 0:
                        zero_valid.append(res['sentence'])
                    
                    if res['total_duration'] > 15:
                        print(f"SLOW ({res['total_duration']:.1f}s): Aug={res['aug_duration']:.1f}s, Filter={res['filter_duration']:.1f}s | {res['sentence']}")
        except Exception as e:
            print(f"Benchmark loop encountered an error or timeout: {e}")

    total_duration = time.time() - start_all
    
    if not stats:
        print("No stats collected.")
        return

    print("\n" + "="*40)
    print("AUGMENTATION BENCHMARK REPORT")
    print("="*40)
    print(f"Total Sentences:     {len(sentences)}")
    print(f"Initial Ungrammatic: {orig_ungrammatic_count} ({orig_ungrammatic_count/len(sentences)*100:.2f}%)")
    print(f"Initial Unpragmatic: {orig_unpragmatic_count} ({orig_unpragmatic_count/len(sentences)*100:.2f}%)")
    print(f"Recovered via Augment: {len(recovered_samples)} (Invalid input -> Valid variations)")
    print(f"Total Time (Wall):   {total_duration:.2f}s")
    
    total_durations = [s['total_duration'] for s in stats]
    aug_durations = [s['aug_duration'] for s in stats]
    filter_durations = [s['filter_duration'] for s in stats]
    
    print(f"Avg Time (CPU):      {mean(total_durations):.4f}s")
    print(f"  - Augmentation:    {mean(aug_durations):.4f}s")
    print(f"  - Filtration:      {mean(filter_durations):.4f}s")
    print(f"Crashes:             {len(crashes)}")
    
    print_percentiles("Candidate Generation", [s['candidates'] for s in stats])
    print_percentiles("Valid Variations", [s['valid'] for s in stats])
    print_percentiles("Total Duration (s)", total_durations)
    print_percentiles("Augmentation Duration (s)", aug_durations)
    print_percentiles("Filtration Duration (s)", filter_durations)

    if zero_valid:
        print("\nShortest sentences with 0 valid variations (post-filter):")
        sorted_zero = sorted(zero_valid, key=len)
        for s in sorted_zero[:5]:
            print(f"  - {s}")

    if recovered_samples:
        print("\nSamples of recovery (Invalid Input -> Valid Variations):")
        for s in recovered_samples[:10]:
            print(f"  - {s}")

    if stats:
        print("\n" + "="*40)
        print("RULE EFFECTIVENESS (Candidates -> Valid)")
        print("="*40)
        # Sort by success rate or volume? Let's sort by volume of valid.
        sorted_rules = sorted(global_rule_total.keys(), key=lambda x: global_rule_valid.get(x, 0), reverse=True)
        for r in sorted_rules:
            total = global_rule_total[r]
            valid = global_rule_valid.get(r, 0)
            rate = (valid / total * 100) if total > 0 else 0
            print(f"{r:25} : {total:6} -> {valid:6} ({rate:5.1f}%)")

        slowest = max(stats, key=lambda x: x['total_duration'])
        print("\n" + "="*40)
        print("SLOWEST SENTENCE ANALYSIS")
        print("="*40)
        print(f"Sentence: {slowest['sentence']}")
        print(f"Total Time:  {slowest['total_duration']:.4f}s")
        print(f"Aug Time:    {slowest['aug_duration']:.4f}s")
        print(f"Filter Time: {slowest['filter_duration']:.4f}s")
        print(f"Candidates:  {slowest['candidates']}")
        print(f"Valid:       {slowest['valid']}")
        print(f"Timed Out:   {slowest['timed_out']}")

    if crashes:
        print("\nCrash Details:")
        for c in crashes[:5]:
            print(f"  - {c['sentence']}: {c['error']}")

if __name__ == "__main__":
    main()
