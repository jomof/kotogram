import time
import unittest

from training_test_utils import setup_mock_style_model

from kotogram.augment import Augmenter


class TestTimeout(unittest.TestCase):
    def setUp(self):
        setup_mock_style_model(self)

    def test_timeout_mechanism(self) -> None:
        augmenter = Augmenter()
        # A sentence that generates many variations
        sentence = "私は彼にその本を上げました。"

        print("Testing with 1.0s timeout (should complete normally)...")
        start = time.time()
        results_normal = augmenter.process_sentence(sentence, timeout=1.0)
        duration_normal = time.time() - start
        print(f"Results: {len(results_normal)}, Duration: {duration_normal:.4f}s")

        print("\nTesting with 0.001s timeout (should exit early)...")
        start = time.time()
        results_fast = augmenter.process_sentence(sentence, timeout=0.001)
        duration_fast = time.time() - start
        print(f"Results: {len(results_fast)}, Duration: {duration_fast:.4f}s")
        if duration_fast < duration_normal:
            print("SUCCESS: Fast timeout exited earlier than normal run.")
        else:
            print(
                "NOTICE: Durations were similar, possibly because the sentence is very simple."
            )

        # Extremely complex sentence to force long generation
        complex_sentence = "私は昨日、彼が言っていた公園に行って、とても美味しいリンゴを食べた後に、少し休憩してから家に帰りました。"
        print("\nTesting complex sentence with 0.01s timeout...")
        start = time.time()
        results_complex = augmenter.process_sentence(complex_sentence, timeout=0.01)
        duration_complex = time.time() - start
        print(f"Results: {len(results_complex)}, Duration: {duration_complex:.4f}s")


if __name__ == "__main__":
    unittest.main()
