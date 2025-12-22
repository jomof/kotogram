from kotogram.analysis import grammar, grammars
from kotogram.sudachi_japanese_parser import SudachiJapaneseParser


def test_batch_grammar() -> None:
    parser = SudachiJapaneseParser()
    sentences = [
        "こんにちは。",
        "元気ですか？",
        "今日はいい天気ですね。",
        "僕は学生です。",
        "あたしは幸せよ。",
    ]

    kotograms = [parser.japanese_to_kotogram(s) for s in sentences]

    print("Testing single-sentence grammar()...")
    res1 = grammar(kotograms[0])
    print(f"Sentence: {sentences[0]}")
    print(f"Grammatical: {res1.is_grammatic}")
    print(f"Formality: {res1.formality.value}")
    print(f"Gender: {res1.gender.value}")

    print("\nTesting multi-sentence grammars()...")
    results = grammars(kotograms)

    for s, res in zip(sentences, results):
        print(f"--- {s} ---")
        print(f"Grammatical: {res.is_grammatic}")
        print(f"Formality: {res.formality.value}")
        print(f"Gender: {res.gender.value}")
        print(f"Score: {res.grammaticality_score:.4f}")


if __name__ == "__main__":
    test_batch_grammar()
