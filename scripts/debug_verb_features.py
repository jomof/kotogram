from kotogram.augment import Augmenter, extract_token_features, split_kotogram


def debug_verb_features() -> None:
    augmenter = Augmenter()
    parser = augmenter.get_parser()

    sentences = ["猫を食べる。", "私は行きます。"]
    for s in sentences:
        print(f"\nSentence: {s}")
        k = parser.japanese_to_kotogram(s)
        print(f"Kotogram: {k}")
        tokens = split_kotogram(k)
        for t in tokens:
            f = extract_token_features(t)
            print(f"Token: {t}")
            print(f"  Surface: {f.surface}")
            print(f"  POS: {f.pos}")
            print(f"  Type: {f.conjugated_type}")
            print(f"  Form: {f.conjugated_form}")
            print(f"  Lemma: {f.lemma}")


if __name__ == "__main__":
    debug_verb_features()
