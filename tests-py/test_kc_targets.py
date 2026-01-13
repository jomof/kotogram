import unittest

import torch

from train.kc import (
    ALL_KC_FAMILIES,
    KcFamilyId,
    compute_kc_targets,
    get_family,
    is_family_db_sourced,
    stable_hash_ints,
)


class TestKCTargets(unittest.TestCase):
    def test_compute_kc_targets_with_pos_detail(self):
        # Input with compound_1 and conjugated_type
        # Note: Avoid IDs 0 (PAD), 1 (UNK), 2 (CLS) as they have special handling
        feature_ids = {
            "pos": [3, 4, 5],
            "compound_1": [10, 11, 10],  # Includes duplicate
            "conjugated_type": [20, 21, 22],
            "conjugated_form": [30, 31, 32],
            "reading": [100, 101, 102],
            "surface": [200, 201, 202],  # Should be ignored
            "reading_gram": [999, 999, 102],
        }

        from unittest.mock import MagicMock

        mock_tokenizer = MagicMock()
        mock_tokenizer.get_id.return_value = 999  # mask_id

        # Mock field_vocabs for reverse lookup in kc.py
        mock_tokenizer.field_vocabs = {
            "pos": {
                "verb": 3,
                "noun": 4,
                "particle": 5,
            },
            "reading": {"<READING_MASK>": 999},
        }

        targets = compute_kc_targets(feature_ids)

        # Verify bag targets (sorted)
        self.assertEqual(targets[KcFamilyId.BAG_POS], [3, 4, 5])
        self.assertEqual(targets[KcFamilyId.BAG_COMPOUND_1], [10, 11])  # Sorted, unique
        # reading_gram derivation:
        # 100 (verb) -> 999 (mask), 101 (noun) -> 999 (mask), 102 (particle) -> 102
        self.assertEqual(targets[KcFamilyId.BAG_READING_GRAM], [102, 999])
        self.assertNotIn("bag_reading", targets)
        # self.assertEqual(targets[KcFamilyId.BAG_CONJUGATED_FORM], [30, 31, 32])

        # Verify tail targets
        self.assertIn(KcFamilyId.TAIL_COMPOUND_1, targets)

        # Verify ngram targets
        self.assertIn(KcFamilyId.NGRAM_POS, targets)
        self.assertIn(KcFamilyId.NGRAM_COMPOUND_1, targets)
        # self.assertIn(KcFamilyId.NGRAM_CONJUGATED_FORM, targets)
        self.assertIn(KcFamilyId.NGRAM_CONJUGATED_TYPE, targets)

        # Verify tail ngram targets
        self.assertIn(KcFamilyId.TAIL_NGRAM_POS, targets)
        self.assertIn(KcFamilyId.TAIL_NGRAM_COMPOUND_1, targets)
        # self.assertIn(KcFamilyId.TAIL_NGRAM_CONJUGATED_FORM, targets)
        self.assertIn(KcFamilyId.TAIL_NGRAM_CONJUGATED_TYPE, targets)

        # Verify pair targets
        # self.assertIn(KcFamilyId.PAIR_POS_CONJFORM, targets)
        # self.assertIn(KcFamilyId.PAIR_POS1_CONJFORM, targets)
        # self.assertIn(KcFamilyId.PAIR_POS1_CONJTYPE, targets)

    def test_compute_kc_targets_missing_fields(self):
        # Input missing compound_1 - use IDs > 2 to avoid special tokens
        feature_ids = {
            "pos": [3, 4, 5],
            "conjugated_form": [30, 31, 32],
            "reading": [100, 101, 102],
        }

        targets = compute_kc_targets(feature_ids)

        # All computed families (excluding DB-sourced) are present
        expected_count = sum(
            1 for fid in ALL_KC_FAMILIES if not is_family_db_sourced(fid)
        )
        self.assertEqual(len(targets), expected_count)
        # compound_1 should be empty (not in input)
        self.assertEqual(targets[KcFamilyId.BAG_COMPOUND_1], [])
        self.assertEqual(targets[KcFamilyId.NGRAM_COMPOUND_1], [])
        self.assertEqual(targets[KcFamilyId.TAIL_NGRAM_COMPOUND_1], [])
        # reading_gram should be empty (no tokenizer was passed)
        self.assertEqual(targets[KcFamilyId.BAG_READING_GRAM], [])

    def test_empty_sequences(self):
        feature_ids = {
            "pos": [],
            "compound_1": [],
            "conjugated_form": [],
            "reading": [],
        }
        targets = compute_kc_targets(feature_ids)

        self.assertEqual(targets[KcFamilyId.BAG_POS], [])
        self.assertEqual(targets[KcFamilyId.NGRAM_POS], [])
        self.assertEqual(targets.get(KcFamilyId.TAIL_NGRAM_POS, []), [])
        self.assertEqual(targets.get(KcFamilyId.BAG_READING_GRAM, []), [])

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
        # Different families have different salt values for domain separation
        ngram = [1, 2, 3]
        salt_pos = get_family(KcFamilyId.NGRAM_POS).salt
        salt_compound = get_family(KcFamilyId.NGRAM_COMPOUND_1).salt
        assert salt_pos is not None and salt_compound is not None
        h1 = stable_hash_ints([salt_pos, *ngram])
        h2 = stable_hash_ints([salt_compound, *ngram])
        self.assertNotEqual(h1, h2)

        # Verify salt values are unique across ngram families
        ngram_families = [fid for fid in KcFamilyId if get_family(fid).salt is not None]
        salt_values = [get_family(fid).salt for fid in ngram_families]
        self.assertEqual(len(set(salt_values)), len(salt_values))

    def test_tensor_input(self):
        # Use IDs > 2 to avoid CLS exclusion
        feature_ids = {
            "pos": torch.tensor([5, 3, 4]),
            "compound_1": torch.tensor([11, 10, 12]),
        }
        targets = compute_kc_targets(feature_ids)
        self.assertEqual(targets[KcFamilyId.BAG_POS], [3, 4, 5])  # Sorted
        self.assertEqual(targets[KcFamilyId.BAG_COMPOUND_1], [10, 11, 12])
        # Verify tail_reading_gram exists if reading and pos are present (even without tokenizer, rg_ids will be empty)
        # All families are present, reading_gram should be empty
        self.assertEqual(targets[KcFamilyId.BAG_READING_GRAM], [])

    def test_get_kc_pos_indices_variations(self):
        """Vary field and vocab_size in _get_kc_pos_indices."""
        # pylint: disable=import-private-name
        from train.dataset import _get_kc_pos_indices

        kc_targets = [{"reading": [1, 2]}, {"pos": [3, 4]}]
        device = torch.device("cpu")
        special_ids = {0}

        # 1. Vary field 'pos'
        inds, mask = _get_kc_pos_indices(
            kc_targets,
            field="pos",
            vocab_size=100,
            device=device,
            special_ids=special_ids,
        )
        # Should match index 1
        self.assertEqual(inds[1, 0], 3)
        self.assertEqual(mask[1, 0], True)
        self.assertEqual(inds[0, 0], 0)  # field not in dict 0
        self.assertEqual(mask[0, 0], False)  # empty

        # 2. Vary vocab_size
        inds_v, _ = _get_kc_pos_indices(
            kc_targets, "reading", vocab_size=50, device=device, special_ids=special_ids
        )
        self.assertEqual(inds_v[0, 0], 1)

    def test_cls_token_excluded_from_targets(self):
        """CLS token (ID 2) should be excluded from all KC family targets."""
        from train.kc import SPECIAL_TOKEN_IDS

        # Verify CLS is in special tokens
        self.assertIn(2, SPECIAL_TOKEN_IDS)

        # Input with CLS token (ID 2) mixed with real tokens
        feature_ids = {
            "pos": [2, 10, 11, 12],  # CLS at start, then real tokens
            "compound_1": [2, 20, 21, 22],
            "conjugated_type": [2, 30, 31, 32],
            "reading_gram": [2, 40, 41, 42],
        }

        targets = compute_kc_targets(feature_ids)

        # CLS (ID 2) should NOT appear in any bag targets
        self.assertNotIn(2, targets[KcFamilyId.BAG_POS])
        self.assertNotIn(2, targets[KcFamilyId.BAG_COMPOUND_1])
        self.assertNotIn(2, targets[KcFamilyId.BAG_CONJUGATED_TYPE])
        self.assertNotIn(2, targets[KcFamilyId.BAG_READING_GRAM])

        # Verify real tokens are still present
        self.assertEqual(targets[KcFamilyId.BAG_POS], [10, 11, 12])
        self.assertEqual(targets[KcFamilyId.BAG_COMPOUND_1], [20, 21, 22])

        # CLS should NOT appear in tail targets either
        self.assertNotIn(2, targets[KcFamilyId.TAIL_POS])
        self.assertNotIn(2, targets[KcFamilyId.TAIL_COMPOUND_1])

    def test_pad_and_unk_not_excluded(self):
        """PAD (ID 0) and UNK (ID 1) should NOT be excluded - kept for analysis."""
        from train.kc import SPECIAL_TOKEN_IDS

        # Verify PAD and UNK are NOT in special tokens
        self.assertNotIn(0, SPECIAL_TOKEN_IDS)
        self.assertNotIn(1, SPECIAL_TOKEN_IDS)

        # Input with PAD and UNK tokens
        feature_ids = {
            "pos": [0, 1, 10, 11],  # PAD, UNK, then real tokens
        }

        targets = compute_kc_targets(feature_ids)

        # PAD and UNK should still appear
        self.assertIn(0, targets[KcFamilyId.BAG_POS])
        self.assertIn(1, targets[KcFamilyId.BAG_POS])


if __name__ == "__main__":
    unittest.main()
