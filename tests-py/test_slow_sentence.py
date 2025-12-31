import time

# pylint: disable=too-many-locals
import unittest
from unittest.mock import patch  # Added for inlined code

# from training_test_utils import setup_mock_style_model # Removed as it's being inlined
from kotogram.augment import Augmenter


class TestSlowSentence(unittest.TestCase):
    def setUp(self):
        # pylint: disable=duplicate-code
        # Manually setup mock model/tokenizer
        # from sudachipy import Dictionary
        # from sudachipy.tokenizer import Tokenizer as SudachiTokenizer

        from kotogram.model import ModelConfig, StyleClassifier
        from kotogram.tokenizer import Tokenizer

        self.tokenizer = Tokenizer()
        # pylint: disable=invalid-name
        from kotogram.sudachi_japanese_parser import SudachiJapaneseParser

        # pylint: disable=invalid-name
        self.parser = SudachiJapaneseParser(dict_type="full")
        # pylint: disable=protected-access
        self.tokenizer._frozen = True

        config = ModelConfig(vocab_sizes=self.tokenizer.get_vocab_sizes())
        self.model = StyleClassifier(config)
        self.model.eval()

        patcher = patch(
            "kotogram.analysis.StyleAnalyzer.load",
            return_value=(self.model, self.tokenizer),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    # pylint: disable=too-many-locals
    def test_slow_sentence(self) -> None:
        augmenter = Augmenter()
        # The 26s outlier from the 10k run
        sentence = "私は世界でもっとも人口が多い島、インドネシア・ジャワ島出身の風来坊です。私は猫をこよなく愛し、コーヒーは私の主食です。"

        print(f"Testing SLOW sentence: {sentence}")
        print("Budget: 1.0s")

        start = time.time()
        # Manual steps to profile
        parser = augmenter.get_parser()
        from kotogram_test_utils import KotogramTestUtils

        from kotogram.augment import get_surface

        tokens = KotogramTestUtils.tokenize_sentence(sentence, parser)

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
    unittest.main()
