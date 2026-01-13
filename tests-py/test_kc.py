import unittest

from train.kc import (
    ALL_KC_FAMILIES,
    KcFamilyId,
    compute_kc_targets,
    is_family_db_sourced,
)


class TestKC(unittest.TestCase):
    def test_compute_kc_targets_basic(self):
        # Setup inputs
        feature_ids = {
            "reading": [1, 2, 3, 1, 2],
            "pos": [10, 20, 30, 10, 20],
            "conjugated_form": [100, 200, 300, 100, 200],
            "surface": [1, 2, 3, 1, 2],  # Needed for tail_surface
            "reading_gram": [999, 999, 3, 999, 999],
        }

        from unittest.mock import MagicMock

        mock_tokenizer = MagicMock()
        mock_tokenizer.get_id.return_value = 999

        # Whitelist IDs: 10 (verb), 30 (particle). 20 (noun) is not.
        mock_tokenizer.field_vocabs = {
            "pos": {"verb": 10, "noun": 20, "particle": 30},
            "reading": {"<READING_MASK>": 999},
        }

        targets = compute_kc_targets(feature_ids)

        # 1. Bag tests (Sets match)
        # 1. Bag tests (Sets match)
        # reading: [1 (10:verb), 2 (20:noun), 3 (30:particle), 1 (10:verb), 2 (20:noun)]
        # rg_ids: [999, 999, 3, 999, 999] -> unique: {3, 999}
        self.assertEqual(set(targets[KcFamilyId.BAG_READING_GRAM]), {3, 999})
        self.assertEqual(set(targets[KcFamilyId.BAG_POS]), {10, 20, 30})
        # self.assertEqual(set(targets[KcFamilyId.BAG_CONJUGATED_FORM]), {100, 200, 300})

        # 2. Tail tests (Last 5)
        # Input length is 5, so tail should be all unique elements
        self.assertEqual(set(targets[KcFamilyId.TAIL_READING_GRAM]), {3, 999})

        # Test truncation for longer sequence
        long_ids = list(range(10))  # 0..9
        # Mock for long sequence: use unique whitelisted POS strings for IDs 5-9.
        # Note: "verb" is now masked, so we swap ID 5 to something whitelisted ("aux-verb")
        # to ensure we keep 5 distinct values if we want to test truncation capacity,
        # OR we accept that 5 becomes 999.
        # Let's use "aux-verb" for 5 to keep it clean and distinct.

        mock_tokenizer.field_vocabs["pos"] = {
            "aux-verb": 5,
            "particle": 6,
            "conj": 7,
            "suffix": 9,
        }
        # reading_gram input should reflect what we expect (derivation is done upstream now)
        # 0-4 are masked (999), 5-9 are preserved (5-9)
        long_rg_ids = [999] * 5 + list(range(5, 10)) + [999] * (len(long_ids) - 10)
        targets = compute_kc_targets(
            {"reading": long_ids, "pos": long_ids, "reading_gram": long_rg_ids}
        )
        # Basic check
        # bag_reading_gram -> 1, 999 (from "reading", filtered/masked)
        # 5, 6, 7, 8, 9 are whitelisted in the mock. 0, 1, 2, 3, 4 map to 999 (default)
        self.assertEqual(
            set(targets[KcFamilyId.BAG_READING_GRAM]), {5, 6, 7, 8, 9, 999}
        )
        # bag_pos -> 0..9 except 2 (CLS excluded by SPECIAL_TOKEN_IDS)
        self.assertEqual(set(targets[KcFamilyId.BAG_POS]), {0, 1, 3, 4, 5, 6, 7, 8, 9})
        # ngram_pos -> hashed values
        self.assertTrue(len(targets[KcFamilyId.NGRAM_POS]) > 0)
        self.assertTrue(all(isinstance(x, int) for x in targets[KcFamilyId.NGRAM_POS]))

    def test_compute_kc_targets_long(self):
        """Test with long input (checking truncation/windowing)."""
        long_ids = list(range(100))
        # Use reading_gram IDs that don't include UNK (ID 1) - start from 3
        long_rg_ids = list(range(3, 103))  # 3..102
        from unittest.mock import MagicMock

        mock_tokenizer = MagicMock()
        mock_tokenizer.get_id.return_value = 999

        mock_tokenizer.field_vocabs = {
            "pos": {"verb": 10, "noun": 20, "particle": 30},
            "reading": {"<READING_MASK>": 999},
        }

        # We need "reading" and "pos" in input
        targets_long = compute_kc_targets(
            {"reading": long_ids, "pos": long_ids, "reading_gram": long_rg_ids}
        )

        # tail_pos should only have last KC_POS_BIASED_WINDOW items
        self.assertEqual(
            targets_long[KcFamilyId.TAIL_POS],
            long_ids[-5:],  # KC_POS_BIASED_WINDOW=5
        )

    def test_compute_kc_targets_short(self):
        """Test with short input."""
        short_ids = [5]  # Avoid UNK (1) and CLS (2)
        from unittest.mock import MagicMock

        mock_tokenizer = MagicMock()
        mock_tokenizer.get_id.return_value = 999

        mock_tokenizer.field_vocabs = {
            "pos": {"verb": 10, "noun": 20, "particle": 30},
            "reading": {"<READING_MASK>": 999},
        }
        targets = compute_kc_targets(
            {"reading": short_ids, "pos": short_ids, "reading_gram": short_ids}
        )
        # Should be present
        self.assertEqual(targets[KcFamilyId.BAG_POS], [5])
        # Tail should calculate even if short
        self.assertEqual(targets[KcFamilyId.TAIL_POS], [5])
        # Ngram order 3 requires 3 items?
        # compute_kc_targets logic: range(2, ORDER+1) -> 2, 3.
        # If len < 2, no ngrams.
        self.assertEqual(targets[KcFamilyId.NGRAM_POS], [])

    def test_compute_kc_targets_empty(self):
        """Test with empty input."""
        from unittest.mock import MagicMock

        mock_tokenizer = MagicMock()
        mock_tokenizer.get_id.return_value = 999

        mock_tokenizer.field_vocabs = {
            "pos": {"verb": 10, "noun": 20, "particle": 30},
            "reading": {"<READING_MASK>": 999},
        }
        targets = compute_kc_targets({})
        # All computed families (excluding DB-sourced) are present
        expected_count = sum(
            1 for fid in ALL_KC_FAMILIES if not is_family_db_sourced(fid)
        )
        self.assertEqual(len(targets), expected_count)
        # All should be empty lists
        for family_id, vals in targets.items():
            self.assertEqual(vals, [], f"Family {family_id} should be empty")

    def test_compute_kc_targets_missing_fields(self):
        """Test with partial fields."""
        # Only pos provided
        feature_ids = {"pos": [1, 2, 3]}
        from unittest.mock import MagicMock

        mock_tokenizer = MagicMock()
        mock_tokenizer.get_id.return_value = 999

        mock_tokenizer.field_vocabs = {
            "pos": {"verb": 10, "noun": 20, "particle": 30},
            "reading": {"<READING_MASK>": 999},
        }
        targets = compute_kc_targets(feature_ids)

        # All computed families are present (excluding DB-sourced)
        expected_count = sum(
            1 for fid in ALL_KC_FAMILIES if not is_family_db_sourced(fid)
        )
        self.assertEqual(len(targets), expected_count)
        # BAG_POS should have values
        self.assertIn(KcFamilyId.BAG_POS, targets)
        self.assertTrue(len(targets[KcFamilyId.BAG_POS]) > 0)
        # BAG_READING_GRAM should be present but empty (no reading_gram in input)
        self.assertIn(KcFamilyId.BAG_READING_GRAM, targets)
        self.assertEqual(targets[KcFamilyId.BAG_READING_GRAM], [])
        # NGRAM_POS should have values
        self.assertIn(KcFamilyId.NGRAM_POS, targets)
        self.assertTrue(len(targets[KcFamilyId.NGRAM_POS]) > 0)


if __name__ == "__main__":
    unittest.main()
