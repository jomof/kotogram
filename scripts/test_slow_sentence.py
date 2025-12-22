import time

from kotogram.augment import Augmenter


def test_slow_sentence() -> None:
    augmenter = Augmenter()
    # The 26s outlier from the 10k run
    sentence = "私は世界でもっとも人口が多い島、インドネシア・ジャワ島出身の風来坊です。私は猫をこよなく愛し、コーヒーは私の主食です。"

    print(f"Testing SLOW sentence: {sentence}")
    print("Budget: 1.0s")

    start = time.time()
    # Manual steps to profile
    parser = augmenter.get_parser()
    k = parser.japanese_to_kotogram(sentence)
    from dataclasses import asdict

    from kotogram.augment import Token, get_surface
    from kotogram.kotogram import extract_token_features, split_kotogram

    tokens = [
        Token(extract_token_features(t).surface or t, asdict(extract_token_features(t)))
        for t in split_kotogram(k)
    ]

    aug_start = time.time()
    candidates = augmenter.augment_tokens(tuple(tokens), deadline=time.time() + 1.0)
    aug_duration = time.time() - aug_start

    surfaces = {"".join(get_surface(t) for t in c) for c in candidates}
    print(f"Candidates generated: {len(surfaces)} (Time: {aug_duration:.4f}s)")

    filter_start = time.time()
    results = augmenter.filter_grammatical(
        surfaces, deadline=time.time() + (1.0 - (time.time() - start))
    )
    filter_duration = time.time() - filter_start

    duration = time.time() - start
    print(f"Variations valid: {len(results)} (Filter Time: {filter_duration:.4f}s)")
    print(f"Total Duration: {duration:.4f}s")

    if duration > 1.1:
        print("WARNING: Exceeded budget significantly!")
    else:
        print("SUCCESS: Respected budget.")


if __name__ == "__main__":
    test_slow_sentence()
