import unittest

import torch

from train.kc import SALT, compute_kc_targets, stable_hash_ints


class TestKCTargets(unittest.TestCase):
    def test_compute_kc_targets_with_pos_detail(self):
        # Input with pos_detail_1 and conjugated_type
        feature_ids = {
            "pos": [1, 2, 3],
            "pos_detail_1": [10, 11, 10],  # Includes duplicate
            "conjugated_type": [20, 21, 22],
            "conjugated_form": [30, 31, 32],
            "reading": [100, 101, 102],
            "surface": [200, 201, 202],  # Should be ignored
        }

        from unittest.mock import MagicMock

        mock_tokenizer = MagicMock()
        mock_tokenizer.get_id.return_value = 999  # mask_id
        mock_tokenizer._rev_pos_cache = None  # pylint: disable=protected-access
        # Mock field_vocabs for reverse lookup in kc.py
        mock_tokenizer.field_vocabs = {
            "pos": {
                "verb": 1,
                "noun": 2,
                "particle": 3,
            },
            "reading": {"<READING_MASK>": 999},
        }

        targets = compute_kc_targets(feature_ids, tokenizer=mock_tokenizer)

        # Verify bag targets (sorted)
        self.assertEqual(targets["bag_pos"], [1, 2, 3])
        self.assertEqual(targets["bag_pos_detail_1"], [10, 11])  # Sorted, unique
        # reading_gram derivation:
        # 100 (verb) -> 999 (mask), 101 (noun) -> 999 (mask), 102 (particle) -> 102
        self.assertEqual(targets["bag_reading_gram"], [102, 999])
        self.assertNotIn("bag_reading", targets)
        self.assertEqual(targets["bag_conjugated_form"], [30, 31, 32])

        # Verify tail targets
        self.assertIn("tail_pos_detail_1", targets)

        # Verify ngram targets
        self.assertIn("ngram_pos", targets)
        self.assertIn("ngram_pos_detail_1", targets)
        self.assertIn("ngram_conjugated_form", targets)
        self.assertIn("ngram_conjugated_type", targets)

        # Verify tail ngram targets
        self.assertIn("tail_ngram_pos", targets)
        self.assertIn("tail_ngram_pos_detail_1", targets)
        self.assertIn("tail_ngram_conjugated_form", targets)
        self.assertIn("tail_ngram_conjugated_type", targets)

        # Verify pair targets
        self.assertIn("pair_pos_conj", targets)
        self.assertIn("pair_pos1_conjform", targets)
        self.assertIn("pair_pos1_conjtype", targets)

    def test_compute_kc_targets_missing_fields(self):
        # Input missing pos_detail_1
        feature_ids = {
            "pos": [1, 2, 3],
            "conjugated_form": [30, 31, 32],
            "reading": [100, 101, 102],
        }

        targets = compute_kc_targets(feature_ids)

        self.assertNotIn("bag_pos_detail_1", targets)
        self.assertNotIn("ngram_pos_detail_1", targets)
        self.assertNotIn("tail_ngram_pos_detail_1", targets)
        # reading_gram will be empty because no tokenizer was passed
        self.assertNotIn("bag_reading_gram", targets)

    def test_empty_sequences(self):
        feature_ids = {
            "pos": [],
            "pos_detail_1": [],
            "conjugated_form": [],
            "reading": [],
        }
        targets = compute_kc_targets(feature_ids)

        self.assertEqual(targets["bag_pos"], [])
        self.assertEqual(targets["ngram_pos"], [])
        self.assertEqual(targets.get("tail_ngram_pos", []), [])
        self.assertEqual(targets.get("bag_reading_gram", []), [])

    def test_deterministic_hashing(self):
        # stable_hash_ints should return same value for same input
        input_data = [1, 2, 3, 4, 1000, 50000]
        h1 = stable_hash_ints(input_data)
        h2 = stable_hash_ints(input_data)
        self.assertEqual(h1, h2)

        # Different input should have different hash (ideally)
        h3 = stable_hash_ints([1, 2, 3, 4, 1000, 50001])
        self.assertNotEqual(h1, h3)

        # Check a few known values (regression test for implementation changes)
        # Note: These values depend on the exact implementation details
        # but they must remain stable.
        self.assertEqual(stable_hash_ints([1]), 0xBEEB8DA1658EEC67)
        self.assertEqual(stable_hash_ints([1, 2]), 0x53A9C5FDCEF668D7)

    def test_domain_separation(self):
        # Different SALT should lead to different hashes for same sequence
        ngram = [1, 2, 3]
        h1 = stable_hash_ints([SALT["ngram_pos"], *ngram])
        h2 = stable_hash_ints([SALT["ngram_pos_detail_1"], *ngram])
        self.assertNotEqual(h1, h2)

        # Verify SALT constants are unique
        self.assertEqual(len(set(SALT.values())), len(SALT))

    def test_tensor_input(self):
        feature_ids = {
            "pos": torch.tensor([3, 1, 2]),
            "pos_detail_1": torch.tensor([11, 10, 12]),
        }
        targets = compute_kc_targets(feature_ids)
        self.assertEqual(targets["bag_pos"], [1, 2, 3])  # Sorted
        self.assertEqual(targets["bag_pos_detail_1"], [10, 11, 12])
        # Verify tail_reading_gram exists if reading and pos are present (even without tokenizer, rg_ids will be empty)
        # Actually with rg_ids being empty, bag_reading_gram won't exist in targets.
        self.assertNotIn("bag_reading_gram", targets)

    def test_reading_gram_derivation_self_check(self):
        """Moved from train/kc.py self-check."""
        from unittest.mock import MagicMock

        from train.kc import derive_reading_gram_ids

        mock_tokenizer = MagicMock()
        mock_tokenizer.get_id.return_value = 999
        # Initialize cache to None so logic detects it missing
        mock_tokenizer._rev_pos_cache = None  # pylint: disable=protected-access

        # Explicit usage to satisfy Vulture if necessary (though MagicMock handles lazy access)
        assert mock_tokenizer.get_id.return_value == 999

        # Test Case 1: Japanese labels (must be normalized via POS_MAP)
        mock_tokenizer.field_vocabs = {
            "pos": {
                "助詞": 10,  # particle (whitelisted)
                "名詞": 20,  # noun (masked)
            },
            "reading": {"<READING_MASK>": 999},
        }
        test_ids_jp = {"reading": [1, 2], "pos": [10, 20]}
        derived_jp = derive_reading_gram_ids(test_ids_jp, mock_tokenizer)
        self.assertEqual(
            derived_jp, [1, 999], f"Japanese label check failed: {derived_jp}"
        )

        # Test Case 2: English labels (already normalized)
        # Clear cache from previous run on the same mock object
        if hasattr(mock_tokenizer, "_rev_pos_cache"):
            delattr(mock_tokenizer, "_rev_pos_cache")

        mock_tokenizer.field_vocabs = {
            "pos": {
                "particle": 30,  # whitelisted
                "verb": 40,  # masked (removed from whitelist)
            },
            "reading": {"<READING_MASK>": 999},
        }
        test_ids_en = {"reading": [3, 4], "pos": [30, 40]}
        derived_en = derive_reading_gram_ids(test_ids_en, mock_tokenizer)
        self.assertEqual(
            derived_en, [3, 999], f"English label check failed: {derived_en}"
        )

        # Verify caching: _rev_pos_cache and fingerprint should be set on tokenizer
        self.assertTrue(hasattr(mock_tokenizer, "_rev_pos_cache"))
        self.assertTrue(hasattr(mock_tokenizer, "_rev_pos_cache_fingerprint"))

        # Test Case 3: Fingerprint-based cache invalidation
        # Change pos vocab without clearing _rev_pos_cache manually
        mock_tokenizer.field_vocabs["pos"] = {
            "particle": 30,
            "verb": 40,
            "noun": 50,  # New entry
        }
        # Re-run: old fingerprint was (2, 40, 70), new is (3, 50, 120)
        test_ids_new = {"reading": [3, 4, 5], "pos": [30, 40, 50]}
        derived_new = derive_reading_gram_ids(test_ids_new, mock_tokenizer)
        self.assertEqual(derived_new, [3, 999, 999])
        self.assertEqual(mock_tokenizer._rev_pos_cache_fingerprint, (3, 50, 120))  # pylint: disable=protected-access
        self.assertEqual(mock_tokenizer._rev_pos_cache[50], "noun")  # pylint: disable=protected-access,unsubscriptable-object

        # Verify missing sentinel robustness (now returns unk_id instead of empty list)
        mock_tokenizer_broken = MagicMock()
        mock_tokenizer_broken.unk_id = 111
        mock_tokenizer_broken.field_vocabs = {}
        # Should return [111, 111] (the unk_ids)
        derived_broken = derive_reading_gram_ids(test_ids_jp, mock_tokenizer_broken)
        self.assertEqual(derived_broken, [111, 111])

        # Verify missing unk_id fallback to 0
        mock_tokenizer_no_unk = MagicMock(spec=["field_vocabs"])  # No unk_id
        mock_tokenizer_no_unk.field_vocabs = {}
        # Should return [0, 0]
        derived_no_unk = derive_reading_gram_ids(test_ids_jp, mock_tokenizer_no_unk)
        self.assertEqual(derived_no_unk, [0, 0])


if __name__ == "__main__":
    unittest.main()
