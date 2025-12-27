import time
import unittest

from training_test_utils import setup_mock_style_model

from kotogram.augment import augment


class TestGlobalTimeout(unittest.TestCase):
    def setUp(self):
        setup_mock_style_model(self)

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
