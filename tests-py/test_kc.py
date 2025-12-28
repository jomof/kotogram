import unittest

from train.kc import (
    KC_HASH_BUCKETS,
    KC_NGRAM_ORDER,
    compute_kc_targets,
)


class TestKC(unittest.TestCase):
    def test_compute_kc_targets_basic(self):
        # Setup inputs
        feature_ids = {
            "lemma": [1, 2, 3, 1, 2],
            "pos": [10, 20, 30, 10, 20],
            "conjugated_form": [100, 200, 300, 100, 200],
            "surface": [1, 2, 3, 1, 2],  # Needed for tail_surface
        }

        targets = compute_kc_targets(feature_ids)

        # 1. Bag tests (Sets match)
        self.assertEqual(set(targets["bag_lemma"]), {1, 2, 3})
        self.assertEqual(set(targets["bag_pos"]), {10, 20, 30})
        self.assertEqual(set(targets["bag_conjugated_form"]), {100, 200, 300})

        # 2. Tail tests (Last 5)
        # Input length is 5, so tail should be all unique elements
        self.assertEqual(set(targets["tail_lemma"]), {1, 2, 3})

        # Test truncation for longer sequence
        long_ids = list(range(10))  # 0..9
        targets_long = compute_kc_targets({"lemma": long_ids})
        self.assertEqual(len(targets_long["tail_lemma"]), 5)
        self.assertEqual(set(targets_long["tail_lemma"]), {5, 6, 7, 8, 9})

        # 3. N-gram tests
        # pos: [10, 20, 30, 10, 20]
        # Bigrams: (10, 20), (20, 30), (30, 10), (10, 20) -> Unique: {(10, 20), (20, 30), (30, 10)}
        # Trigrams: (10, 20, 30), (20, 30, 10), (30, 10, 20)

        ngrams = targets["ngram_pos"]
        # Just check validity of hashes
        self.assertTrue(all(0 <= x < KC_HASH_BUCKETS for x in ngrams))

        # 4. Pair tests
        pairs = targets["pair_pos_conj"]
        self.assertTrue(all(0 <= x < KC_HASH_BUCKETS for x in pairs))
        # Logic check: (10, 100), (20, 200), (30, 300), (10, 100), (20, 200) -> 3 unique pairs
        self.assertEqual(len(pairs), 3)

    def test_compute_kc_targets_empty(self):
        feature_ids = {"lemma": [], "pos": []}
        targets = compute_kc_targets(feature_ids)

        self.assertEqual(targets["bag_lemma"], [])
        self.assertEqual(targets["tail_lemma"], [])
        self.assertEqual(targets["ngram_pos"], [])

    def test_compute_kc_targets_short(self):
        # Sequence shorter than n-gram order
        feature_ids = {
            "pos": [10, 20]  # Length 2. Bigrams generated, Trigrams not.
        }
        if KC_NGRAM_ORDER >= 3:
            # Should produce bigram hash but not trigram
            targets = compute_kc_targets(feature_ids)
            # Only 1 bigram (10, 20)
            self.assertEqual(len(targets["ngram_pos"]), 1)

    def test_compute_kc_targets_missing_fields(self):
        # Missing fields should not crash and just omit targets
        feature_ids = {"lemma": [1]}
        targets = compute_kc_targets(feature_ids)

        self.assertIn("bag_lemma", targets)
        self.assertNotIn("bag_pos", targets)
        self.assertNotIn("pair_pos_conj", targets)


if __name__ == "__main__":
    unittest.main()
