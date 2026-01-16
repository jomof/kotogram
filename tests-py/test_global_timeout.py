import time
import unittest
from unittest.mock import patch

from kotogram.augment import augment
from kotogram.model import InferenceClassifier, ModelConfig
from kotogram.tokenizer import Tokenizer


class TestGlobalTimeout(unittest.TestCase):
    # pylint: disable=invalid-name
    def setUp(self):
        # pylint: disable=duplicate-code
        # Manually setup mock model/tokenizer
        self.tokenizer = Tokenizer()
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

    def test_global_augment_timeout(self) -> None:
        sentences = [
            "私は彼にその本を上げました。",
            "今日はいい天気ですね。",
            "彼は学生です。",
        ]

        print("Testing global augment with 2.0s timeout...")
        start = time.time()
        results_normal = augment(sentences, timeout=2.0)
        duration_normal = time.time() - start
        print(f"Results: {len(results_normal)}, Duration: {duration_normal:.4f}s")

        print("\nTesting global augment with 0.0001s timeout...")
        start = time.time()
        results_fast = augment(sentences, timeout=0.0001)
        duration_fast = time.time() - start
        # Should return at least the original sentences if it timed out immediately
        print(f"Results: {len(results_fast)}, Duration: {duration_fast:.4f}s")
        print(
            f"Recovered originals (if matched): {set(sentences).issubset(set(results_fast))}"
        )


if __name__ == "__main__":
    unittest.main()
