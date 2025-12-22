from kotogram.analysis import grammar
from kotogram.augment import Augmenter
from kotogram.sudachi_japanese_parser import SudachiJapaneseParser

sentences = [
    "彼は酒にふけている。",
    "折り返しを電話する。",
    "彼は音楽にふけている。",
    "彼はもの思いにふけていた。",
    "温かいスープが飲みたいです。",
]

parser = SudachiJapaneseParser()
augmenter = Augmenter()

for s in sentences:
    print(f"\nSentence: {s}")
    k = parser.japanese_to_kotogram(s)
    analysis = grammar(k)
    print(f"  Grammatic: {analysis.is_grammatic}")
    print(f"  Formality: {analysis.formality.value}")
    print(f"  Gender:    {analysis.gender.value}")

    # Try augmenting
    results = augmenter.process_sentence(s)
    print(f"  Total Augmented (Pre-filter): {len(results)}")

    # Check if original is in results (if grammatic)
    if s in results:
        print("  Original preserved.")
    else:
        print("  Original filtered out or not in candidates.")
