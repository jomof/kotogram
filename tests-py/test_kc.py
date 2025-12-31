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
            "reading": [1, 2, 3, 1, 2],
            "pos": [10, 20, 30, 10, 20],
            "conjugated_form": [100, 200, 300, 100, 200],
            "surface": [1, 2, 3, 1, 2],  # Needed for tail_surface
        }

        from unittest.mock import MagicMock

        mock_tokenizer = MagicMock()
        mock_tokenizer.get_id.return_value = 999
        mock_tokenizer._rev_pos_cache = None  # pylint: disable=protected-access
        # Whitelist IDs: 10 (verb), 30 (particle). 20 (noun) is not.
        mock_tokenizer.field_vocabs = {
            "pos": {"verb": 10, "noun": 20, "particle": 30},
            "reading": {"<READING_MASK>": 999},
        }

        targets = compute_kc_targets(feature_ids, tokenizer=mock_tokenizer)

        # 1. Bag tests (Sets match)
        # 1. Bag tests (Sets match)
        # reading: [1 (10:verb), 2 (20:noun), 3 (30:particle), 1 (10:verb), 2 (20:noun)]
        # rg_ids: [999, 999, 3, 999, 999] -> unique: {3, 999}
        self.assertEqual(set(targets["bag_reading_gram"]), {3, 999})
        self.assertEqual(set(targets["bag_pos"]), {10, 20, 30})
        self.assertEqual(set(targets["bag_conjugated_form"]), {100, 200, 300})

        # 2. Tail tests (Last 5)
        # Input length is 5, so tail should be all unique elements
        self.assertEqual(set(targets["tail_reading_gram"]), {3, 999})

        # Test truncation for longer sequence
        long_ids = list(range(10))  # 0..9
        # Mock for long sequence: use unique whitelisted POS strings for IDs 5-9.
        # Note: "verb" is now masked, so we swap ID 5 to something whitelisted ("aux-verb")
        # to ensure we keep 5 distinct values if we want to test truncation capacity,
        # OR we accept that 5 becomes 999.
        # Let's use "aux-verb" for 5 to keep it clean and distinct.
        mock_tokenizer._rev_pos_cache = None  # pylint: disable=protected-access
        mock_tokenizer.field_vocabs["pos"] = {
            "aux-verb": 5,
            "particle": 6,
            "conj": 7,
            "prefix": 8,
            "suffix": 9,
        }
        targets_long = compute_kc_targets(
            {"reading": long_ids, "pos": long_ids}, tokenizer=mock_tokenizer
        )
        self.assertEqual(len(targets_long["tail_reading_gram"]), 5)
        self.assertEqual(set(targets_long["tail_reading_gram"]), {5, 6, 7, 8, 9})

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
        feature_ids = {"reading": [], "pos": []}
        targets = compute_kc_targets(feature_ids)

        self.assertEqual(targets.get("bag_reading_gram", []), [])
        self.assertEqual(targets.get("tail_reading_gram", []), [])
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
        feature_ids = {"reading": [1]}
        targets = compute_kc_targets(feature_ids)

        self.assertNotIn("bag_reading_gram", targets)
        self.assertNotIn("bag_pos", targets)
        self.assertNotIn("pair_pos_conj", targets)


if __name__ == "__main__":
    unittest.main()
