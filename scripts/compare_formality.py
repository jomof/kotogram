#!/usr/bin/env python3
"""Compare rule-based vs model-based formality predictions.

Finds sentences where the two approaches disagree, helping identify
cases where either the rules or the model need improvement.
"""

import csv
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kotogram import SudachiJapaneseParser, formality

# Sentences where the MODEL is known to be incorrect
# (rule-based is correct for these)
MODEL_INCORRECT_SENTENCES = {
    # Model predicts NEUTRAL for だった sentences, but rule correctly says CASUAL
    # (だった is plain past copula, contrasting with formal でした)
    "私はいつも不思議な性格の方が好きだった。",  # ID 4737 - Model: neutral, Rule: casual (correct)

    # Model predicts NEUTRAL but has のよ (feminine casual marker)
    "みんなあなたに会いたがってる。あなたは有名なのよ！",  # ID 4885 - Model: neutral, Rule: casual (correct)

    # Model predicts NEUTRAL but has かしら (feminine wondering particle)
    "その答えを知るのにあなたは本当にその質問をする必要があるのかしら。",  # ID 4927 - Model: neutral, Rule: casual (correct)

    # Model predicts NEUTRAL but has ください (polite imperative)
    "もし間違いを見つけたら訂正してください。",  # ID 4973 - Model: neutral, Rule: formal (correct)

    # Model predicts NEUTRAL but has ください (polite imperative)
    "最初着る前に洗濯してください。",  # ID 4978 - Model: neutral, Rule: formal (correct)

    # Model predicts NEUTRAL but has のね (casual particle)
    "驚かない所をみると知ってたのね。",  # ID 4988 - Model: neutral, Rule: casual (correct)

    # Model predicts NEUTRAL but has かしら (feminine wondering particle)
    "どれくらいかかるのかしら。",  # ID 5010 - Model: neutral, Rule: casual (correct)

    # Model predicts NEUTRAL but has よ (casual particle)
    "彼がそれを聞いたら喜ぶよ。",  # ID 5136 - Model: neutral, Rule: casual (correct)

    # Model predicts NEUTRAL but has のね (casual particle combination)
    "あなた私が誰か知らないのね。",  # ID 5159 - Model: neutral, Rule: casual (correct)

    # Model predicts NEUTRAL but has だった (casual past copula)
    "あの日が私の人生で最高の日だった。",  # ID 5173 - Model: neutral, Rule: casual (correct)

    # Model predicts NEUTRAL but has のね (casual particle combination)
    "あなたは私に夢を見させてくれるのね。",  # ID 5180 - Model: neutral, Rule: casual (correct)

    # Model predicts NEUTRAL but has のね (casual particle combination)
    "わざとやったのね！",  # ID 5189 - Model: neutral, Rule: casual (correct)

    # Model predicts NEUTRAL but has のね (casual particle combination)
    "あれ？あなたまだここにいたのね！",  # ID 5219 - Model: neutral, Rule: casual (correct)

    # Model predicts NEUTRAL but has だった (casual past copula)
    "ジェイソンは無口な人だったので彼が何かを言うたび驚いていた。",  # ID 5235 - Model: neutral, Rule: casual (correct)

    # Model predicts NEUTRAL but has だった (casual past copula)
    "彼の評論は問題の表面的な分析結果を取り上げていただけだったのでクラスで最上位の成績を得たことにとても驚いた。",  # ID 5239 - Model: neutral, Rule: casual (correct)

    # Model predicts UNPRAGMATIC but has できますか (formal question)
    "もし電気がないと、私たちの暮らしがどのようなものになるか想像できますか。",  # ID 5314 - Model: unpragmatic, Rule: formal (correct)

    # Model predicts NEUTRAL but has ください (polite imperative)
    "ここで写真を撮らないでください。",  # ID 6004 - Model: neutral, Rule: formal (correct)

    # Model predicts FORMAL but has 致す (humble keigo)
    "納税者の目線で努力を致したいと思います。",  # ID 74054 - Model: formal, Rule: very_formal (correct)

    # Model predicts NEUTRAL but has のさ (casual particle)
    "俺を哀れに思って助けてくれたのさ。",  # ID 74069 - Model: neutral, Rule: casual (correct)

    # Model predicts CASUAL but has じゃ (very casual contraction)
    "「じゃ留年しなかったら付き合ってくれんの？」「タラレバ話って好きじゃないの」",  # ID 74076 - Model: casual, Rule: very_casual (correct)

    # Model predicts UNPRAGMATIC but has ますか (formal question)
    "迷彩のショートパンツを履く場合、上は何色のTシャツが合いますか？",  # ID 74085 - Model: unpragmatic, Rule: formal (correct)

    # Model predicts UNPRAGMATIC but has ですか (formal question)
    "あなたはどこの国の出身ですか。",  # ID 74108 - Model: unpragmatic, Rule: formal (correct)

    # Model predicts UNPRAGMATIC but has でしょうか (formal tentative)
    "赤ちゃんは泣かせたままにしてはいけないのでしょうか？",  # ID 74110 - Model: unpragmatic, Rule: formal (correct)

    # Model predicts NEUTRAL but has 下さい (polite imperative)
    "窓を開ける時は、カーテンは閉めないで下さい。",  # ID 74163 - Model: neutral, Rule: formal (correct)

    # Model predicts UNPRAGMATIC but has んですね (formal)
    "惰性に身を任せているがために今のような現在があるんですね。",  # ID 74180 - Model: unpragmatic, Rule: formal (correct)

    # Model predicts NEUTRAL but has だった (casual past copula)
    "結局、法案は提出断念に追い込まれたのだった。",  # ID 74182 - Model: neutral, Rule: casual (correct)

    # Model predicts NEUTRAL but has 頂きたい (humble keigo)
    "傍目八目という言葉があるように一度協会から離れて、日本サッカーをみて頂きたい。",  # ID 74200 - Model: neutral, Rule: very_formal (correct)

    # Model predicts NEUTRAL but has ください (polite imperative)
    "細かい意訳誤訳は気にしないでください。",  # ID 74226 - Model: neutral, Rule: formal (correct)

    # Model predicts UNPRAGMATIC but has ですね (formal with acceptable particle)
    "パプアニューギニアに住むメラネシア人の多くは、かなり強い天然パーマですね。",  # ID 74261 - Model: unpragmatic, Rule: formal (correct)
}

# Sentences where NEITHER is definitively wrong (design differences)
# Rule-based focuses on explicit markers; model considers overall tone
DESIGN_DIFFERENCE_SENTENCES = {
    # No explicit casual markers, but informal overall tone
    "なぜみんなが私のことを気違いだと思うのか、遂に説明してくれてありがとう。",  # ID 4820

    # Dialogue with two speakers (formal question, casual answer)
    # Model sees as unpragmatic mixing, rule sees as formal (from ですか)
    "「どなたですか」「お母さんよ」",  # ID 4914

    # だから何？ - design choice: rule treats as neutral (conjunction), model as casual
    "だから何？",  # ID 4765

    # Plain conditional + ellipsis - gray area between neutral and casual
    "何と言ったらいいか・・・。",  # ID 4713

    # Quoted speech vs narrative context - rule looks at だね in quotes, model sees 言います (formal)
    "「これはとてもおもしろそうだね」とひろしが言います。",  # ID 4944

    # んですよ pattern - rule sees よ as casual, model sees です as formal
    "申し訳ないけど長居できないんですよ。",  # ID 4976

    # Embedded だと clause - rule sees it as neutral (embedded quote), model sees as casual
    "私はいつも思っていた、心筋梗塞を患うことは死期を知らせる前兆だと。",  # ID 5033

    # てる contraction with です ending - rule sees formal (ending), model sees mixing
    "私の命が危ないと言ってるわけですか？",  # ID 5289 - Model: unpragmatic, Rule: formal

    # Mixed formality within multi-sentence text
    "同性愛者ですが何か？それが犯罪だとでも？",  # ID 5293 - Model: unpragmatic, Rule: formal

    # Rule sees わよ as casual, model sees でてた contraction as very casual
    "早川くん、中間テストの結果でてたわよ。またトップ！",  # ID 74078 - Rule: casual, Model: very_casual

    # なあ parsed as interjection, not particle - model sees casual tone
    "「なあ、そこの姉さん」「え？」「ちょっと相談に乗ってくれないか？」",  # ID 74123 - Rule: neutral, Model: casual

    # んですけどね - rule sees formal (です), model sees casual mixing
    "僕的にはもっと地味ってか渋い服が欲しいんですけどね。",  # ID 74175 - Rule: formal, Model: unpragmatic

    # ってのは with ですよね - rule sees formal, model sees mixing
    "男でも女でも腹が据わっているってのはカッコいいですよね。",  # ID 74199 - Rule: formal, Model: unpragmatic

    # するか？ - rule sees neutral (plain question), model sees casual tone
    "町中見物でもするか？",  # ID 74247 - Rule: neutral, Model: casual
}

def main():
    parser = SudachiJapaneseParser(dict_type='full')
    data_path = Path(__file__).parent.parent / "data" / "jpn_sentences.tsv"

    disagreements = []
    total_checked = 0
    max_disagreements = 5

    print(f"Comparing rule-based vs model-based formality predictions...")
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

                rule_result = formality(kotogram, use_model=False)
                model_result = formality(kotogram, use_model=True)

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
