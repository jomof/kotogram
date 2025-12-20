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

    # Model predicts NEUTRAL but has くだらねえ (masculine contraction)
    "「くだらねえ都市伝説だろう」「でも、火のないところに煙は立たないというけどね」",  # ID 74728

    # Model predicts NEUTRAL but has オレ (masculine pronoun)
    "ああ、オレも実際、こうして目の当たりにするまでは半信半疑だったが・・・。",  # ID 75203
    "雪がしんしんと降り積もる・・・オレの体に。",  # ID 75936

    # Model predicts NEUTRAL but has ぞ particle (masculine)
    "買い物の割に遅かったな。どこぞでよろしくやっていたのか？",  # ID 75272

    # Model predicts NEUTRAL but has のよ (feminine pattern)
    "いいのよ。それより、早く行かないとタイムセール終わっちゃう。",  # ID 75407
    "こーゆーのは、買うのが楽しいのよ。使うか使わないかなんてのは、二の次なんだって。",  # ID 76768

    # Model predicts NEUTRAL but has ねえ (masculine rough negation)
    "なんだってずらからねえんだ！",  # ID 76498

    # Model predicts NEUTRAL but has オレ (masculine pronoun)
    "助けてください！オレ、毎晩同じ悪夢を見るんです。",  # ID 147383
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

    # Model predicts masculine based on aggressive tone (やっとる is regional/masculine)
    # Rule sees no explicit markers
    "こ、こら！ナニやっとるかぁぁ！！",  # ID 74701 - Rule: neutral, Model: masculine

    # Model predicts masculine for 見して (casual imperative) - can have masculine associations
    "「マナカの絵、見して」「えーー、恥ずかしいですよー」",  # ID 75055 - Rule: neutral, Model: masculine

    # Model predicts feminine for sentences with sentence-final よ
    # よ can be used by anyone but model may associate it with feminine speech
    "料理をするのを手伝ってよ。",  # ID 77922 - Rule: neutral, Model: feminine
    "彼が君の申し出を引き受けるのは請け合うよ。",  # ID 120446 - Rule: neutral, Model: feminine

    # Model predicts masculine for おれ but it's the imperative verb form of 構える, not pronoun オレ
    "詳しいことが全部わかるまでは、あわててその場にふみこむな。見当がつくまでは、慎重にかまえておれ。",  # ID 146327

    # Model predicts feminine for のでね pattern - reasonable interpretation
    "出来ればそうしたいのだが、行くところがあるのでね。",  # ID 147668
    "十分間に合うように出かけよう。危険は犯したくないのでね。",  # ID 148031

    # Model predicts masculine for unclear reasons
    "私はその少年がほんを読むのに反対しない。",  # ID 160025

    # Model predicts feminine, seeing 煮よ as sentence-final よ (it's actually part of 煮 cooking style)
    "私の十八番、チキンのレモン煮よ。",  # ID 163353
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
