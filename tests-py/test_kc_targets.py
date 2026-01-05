import unittest

import torch

from train.kc import SALT, KcFamilyId, compute_kc_targets, stable_hash_ints


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
            "reading_gram": [999, 999, 102],
        }

        from unittest.mock import MagicMock

        mock_tokenizer = MagicMock()
        mock_tokenizer.get_id.return_value = 999  # mask_id

        # Mock field_vocabs for reverse lookup in kc.py
        mock_tokenizer.field_vocabs = {
            "pos": {
                "verb": 1,
                "noun": 2,
                "particle": 3,
            },
            "reading": {"<READING_MASK>": 999},
        }

        targets = compute_kc_targets(feature_ids)

        # Verify bag targets (sorted)
        # Verify bag targets (sorted)
        self.assertEqual(targets[KcFamilyId.BAG_POS], [1, 2, 3])
        self.assertEqual(
            targets[KcFamilyId.BAG_POS_DETAIL_1], [10, 11]
        )  # Sorted, unique
        # reading_gram derivation:
        # 100 (verb) -> 999 (mask), 101 (noun) -> 999 (mask), 102 (particle) -> 102
        self.assertEqual(targets[KcFamilyId.BAG_READING_GRAM], [102, 999])
        self.assertNotIn("bag_reading", targets)
        # self.assertEqual(targets[KcFamilyId.BAG_CONJUGATED_FORM], [30, 31, 32])

        # Verify tail targets
        self.assertIn(KcFamilyId.TAIL_POS_DETAIL_1, targets)

        # Verify ngram targets
        self.assertIn(KcFamilyId.NGRAM_POS, targets)
        self.assertIn(KcFamilyId.NGRAM_POS_DETAIL_1, targets)
        # self.assertIn(KcFamilyId.NGRAM_CONJUGATED_FORM, targets)
        self.assertIn(KcFamilyId.NGRAM_CONJUGATED_TYPE, targets)

        # Verify tail ngram targets
        self.assertIn(KcFamilyId.TAIL_NGRAM_POS, targets)
        self.assertIn(KcFamilyId.TAIL_NGRAM_POS_DETAIL_1, targets)
        # self.assertIn(KcFamilyId.TAIL_NGRAM_CONJUGATED_FORM, targets)
        self.assertIn(KcFamilyId.TAIL_NGRAM_CONJUGATED_TYPE, targets)

        # Verify pair targets
        # self.assertIn(KcFamilyId.PAIR_POS_CONJFORM, targets)
        # self.assertIn(KcFamilyId.PAIR_POS1_CONJFORM, targets)
        # self.assertIn(KcFamilyId.PAIR_POS1_CONJTYPE, targets)

    def test_compute_kc_targets_missing_fields(self):
        # Input missing pos_detail_1
        feature_ids = {
            "pos": [1, 2, 3],
            "conjugated_form": [30, 31, 32],
            "reading": [100, 101, 102],
        }

        targets = compute_kc_targets(feature_ids)

        self.assertNotIn(KcFamilyId.BAG_POS_DETAIL_1, targets)
        self.assertNotIn(KcFamilyId.NGRAM_POS_DETAIL_1, targets)
        self.assertNotIn(KcFamilyId.TAIL_NGRAM_POS_DETAIL_1, targets)
        # reading_gram will be empty because no tokenizer was passed
        self.assertNotIn(KcFamilyId.BAG_READING_GRAM, targets)

    def test_empty_sequences(self):
        feature_ids = {
            "pos": [],
            "pos_detail_1": [],
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
        # Different SALT should lead to different hashes for same sequence
        # Different SALT should lead to different hashes for same sequence
        ngram = [1, 2, 3]
        h1 = stable_hash_ints([SALT[KcFamilyId.NGRAM_POS], *ngram])
        h2 = stable_hash_ints([SALT[KcFamilyId.NGRAM_POS_DETAIL_1], *ngram])
        self.assertNotEqual(h1, h2)

        # Verify SALT constants are unique
        self.assertEqual(len(set(SALT.values())), len(SALT))

    def test_tensor_input(self):
        feature_ids = {
            "pos": torch.tensor([3, 1, 2]),
            "pos_detail_1": torch.tensor([11, 10, 12]),
        }
        targets = compute_kc_targets(feature_ids)
        self.assertEqual(targets[KcFamilyId.BAG_POS], [1, 2, 3])  # Sorted
        self.assertEqual(targets[KcFamilyId.BAG_POS_DETAIL_1], [10, 11, 12])
        # Verify tail_reading_gram exists if reading and pos are present (even without tokenizer, rg_ids will be empty)
        # Actually with rg_ids being empty, bag_reading_gram won't exist in targets.
        self.assertNotIn(KcFamilyId.BAG_READING_GRAM, targets)

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


if __name__ == "__main__":
    unittest.main()
