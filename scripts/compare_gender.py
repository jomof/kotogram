#!/usr/bin/env python3
"""Compare rule-based vs model-based gender predictions.

Finds sentences where the two approaches disagree, helping identify
cases where either the rules or the model need improvement.
"""

import csv
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kotogram import SudachiJapaneseParser, gender

# Sentences where the MODEL is known to be incorrect
# (rule-based is correct for these)
MODEL_INCORRECT_SENTENCES: set[str] = {
    # Model predicts NEUTRAL but has 僕 (masculine pronoun)
    "君は僕にとてもいらいらしている。",  # ID 4741 - Model: neutral, Rule: masculine (correct)
    "僕が最後に自分の考えを伝えた人は、僕を気違いだと思ったようだ。",  # ID 4745
    "大抵の人は僕を気違いだと思っている。",  # ID 4747
    "僕はすごく太ってる。",  # ID 4764
    "僕は不幸かも知れないけれど自殺はしない。",  # ID 4782
}

# Sentences where NEITHER is definitively wrong (design differences)
# Rule-based focuses on explicit markers; model considers overall tone
DESIGN_DIFFERENCE_SENTENCES: set[str] = {
    # Model predicts feminine for sentences with よ particle
    # よ can be used by anyone but model may associate it with feminine speech
    "きみにちょっとしたものをもってきたよ。",  # ID 1297 - Rule: neutral, Model: feminine
    "行くよ。",  # ID 4739 - Rule: neutral, Model: feminine

    # No explicit gender markers, model predicts feminine based on tone
    "何と言ったらいいか・・・。",  # ID 4713 - Rule: neutral, Model: feminine

    # のかな pattern - can have feminine associations
    "みんなもそうなのかな、と思うことくらいしかできない。",  # ID 4732 - Rule: neutral, Model: feminine
}

def main():
    parser = SudachiJapaneseParser(dict_type='full')
    data_path = Path(__file__).parent.parent / "data" / "jpn_sentences.tsv"

    disagreements = []
    total_checked = 0
    max_disagreements = 5

    print(f"Comparing rule-based vs model-based gender predictions...")
    print(f"Data: {data_path}")
    print(f"Stopping after {max_disagreements} disagreements\n")

    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')

        for row in reader:
            if len(row) < 3:
                continue

            sentence_id, lang, sentence = row[0], row[1], row[2]

            if lang != 'jpn':
                continue

            # Skip known model-incorrect sentences and design difference cases
            if sentence in MODEL_INCORRECT_SENTENCES or sentence in DESIGN_DIFFERENCE_SENTENCES:
                continue

            try:
                kotogram = parser.japanese_to_kotogram(sentence)

                rule_result = gender(kotogram, use_model=False)
                model_result = gender(kotogram, use_model=True)

                total_checked += 1

                if rule_result != model_result:
                    disagreements.append({
                        'id': sentence_id,
                        'sentence': sentence,
                        'kotogram': kotogram,
                        'rule': rule_result,
                        'model': model_result,
                    })

                    print(f"Disagreement #{len(disagreements)}:")
                    print(f"  ID: {sentence_id}")
                    print(f"  Sentence: {sentence}")
                    print(f"  Rule-based: {rule_result.value}")
                    print(f"  Model:      {model_result.value}")
                    print(f"  Kotogram:   {kotogram[:100]}...")
                    print()

                    if len(disagreements) >= max_disagreements:
                        break

                if total_checked % 1000 == 0:
                    print(f"  Checked {total_checked} sentences, {len(disagreements)} disagreements so far...")

            except Exception as e:
                print(f"Error processing {sentence_id}: {e}")
                continue

    print(f"\nSummary:")
    print(f"  Total checked: {total_checked}")
    print(f"  Disagreements found: {len(disagreements)}")

    if disagreements:
        print(f"\nDisagreement details for analysis:")
        print("=" * 60)
        for i, d in enumerate(disagreements, 1):
            print(f"\n{i}. {d['sentence']}")
            print(f"   Rule: {d['rule'].value}, Model: {d['model'].value}")

if __name__ == "__main__":
    main()
