import time
from dataclasses import asdict

from kotogram.augment import Augmenter, Token, extract_token_features, split_kotogram


def test_extreme_outlier() -> None:
    augmenter = Augmenter()
    sentence = "それから幾千年かを隔てた後、この魂は無数の流転を閲して、また生を人間に託さなければならなくなった。それがこう云う私に宿っている魂なのである。だから私は現代に生れはしたが、何一つ意味のある仕事が出来ない。昼も夜も漫然と夢みがちな生活を送りながら、ただ、何か来るべき不可思議なものばかりを待っている。ちょうどあの尾生が薄暮の橋の下で、永久に来ない恋人をいつまでも待ち暮したように。"

    print(f"Testing EXTREME outlier sentence (Length: {len(sentence)})")
    print("Budget: 1.0s")

    start = time.time()
    deadline = start + 1.0

    parser = augmenter.get_parser()
    k = parser.japanese_to_kotogram(sentence)
    tokens = [
        Token(extract_token_features(t).surface or t, asdict(extract_token_features(t)))
        for t in split_kotogram(k)
    ]

    aug_start = time.time()
    candidates = augmenter.augment_tokens(tuple(tokens), deadline=deadline)
    aug_duration = time.time() - aug_start

    print(f"Augmentation Duration: {aug_duration:.4f}s")
    print(f"Candidates: {len(candidates)}")

    if aug_duration > 1.2:
        print("FAILURE: Augmentation phase ignored the deadline!")
    else:
        print("SUCCESS: Augmentation phase (mostly) respected the deadline.")


if __name__ == "__main__":
    test_extreme_outlier()
