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

        from kotogram.model import InferenceClassifier, ModelConfig
        from kotogram.tokenizer import Tokenizer

        self.tokenizer = Tokenizer()
        # pylint: disable=invalid-name
        from kotogram.sudachi_japanese_parser import SudachiJapaneseParser

        # pylint: disable=invalid-name
        self.parser = SudachiJapaneseParser()
        # pylint: disable=protected-access
        self.tokenizer._frozen = True

        config = ModelConfig(vocab_sizes=self.tokenizer.get_vocab_sizes())
        self.model = InferenceClassifier(config)
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
        # Give ample time for the control run to ensure we get results
        results = augmenter.filter_grammatical(surfaces, deadline=time.time() + 30.0)
        filter_duration = time.time() - filter_start

        duration = time.time() - start
        print(f"Variations valid: {len(results)} (Filter Time: {filter_duration:.4f}s)")
        print(f"Total Duration: {duration:.4f}s")

        if duration > 1.1:
            print("WARNING: Exceeded budget significantly!")
        else:
            print("Budget respected.")

    def test_deadline_parameter_effect(self) -> None:
        """Verify that the deadline parameter effectively truncates processing."""
        augmenter = Augmenter()
        sentence = "猫が好き"

        # 1. Generate candidates
        parser = augmenter.get_parser()
        from kotogram_test_utils import KotogramTestUtils

        from kotogram.augment import get_surface

        tokens = KotogramTestUtils.tokenize_sentence(sentence, parser)
        candidates = augmenter.augment_tokens(
            tuple(tokens), deadline=time.time() + 30.0
        )
        surfaces = {"".join(get_surface(t) for t in c) for c in candidates}

        # 2. Filter with generous deadline (Value A)
        start_t = time.time()
        _ = augmenter.filter_grammatical(surfaces, deadline=start_t + 10.0)
        dur_full = time.time() - start_t

        # 3. Filter with immediate deadline (Value B - distinct from A)
        start_t = time.time()
        _ = augmenter.filter_grammatical(surfaces, deadline=start_t + 0.000001)
        dur_timeout = time.time() - start_t

        # 4. Assert effect on execution time
        # The full run processes ~300 items (slow inference), timeout run should exit early.
        # We can't rely on result count because dummy model rejects everything.
        self.assertLess(dur_timeout, dur_full)
        if dur_full > 0.1:
            self.assertLess(dur_timeout, dur_full * 0.5)


if __name__ == "__main__":
    unittest.main()
