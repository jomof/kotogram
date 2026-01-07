import unittest
from unittest.mock import MagicMock

import torch

from train.dataset import create_kc_batch
from train.kc import KcFamilyId
from train.types import TrainingBatch


class TestKCDatasetMask(unittest.TestCase):
    def test_global_mask_creation(self):
        """Verifies that create_kc_batch produces a correct global effective mask."""
        # Setup
        # Family A: V=10
        # Family B: V=10
        # Batch size 3
        # Ex 0: A has pos, B has pos -> Global True
        # Ex 1: A has pos, B no pos -> Global True
        # Ex 2: A no pos, B no pos -> Global False

        fid_a = KcFamilyId.BAG_POS
        fid_b = KcFamilyId.NGRAM_POS
        target_specs = {fid_a: 10, fid_b: 10}

        tokenizer = MagicMock()
        tokenizer.unk_id = 998
        tokenizer.cls_id = 999

        # Determine keys used in dataset (dataset.py doesn't strictly define input keys to create_kc_batch,
        # it expects batch.kc_targets to be List[Dict]).
        # create_kc_batch(batch, tokenizer, specs)

        # Construct TrainingBatch info
        # kc_targets is List[Dict[KcFamilyId, List[int]]]

        targets_0 = {fid_a: [1], fid_b: [2]}
        targets_1 = {fid_a: [1], fid_b: []}
        targets_2 = {fid_a: [], fid_b: []}

        batch = MagicMock(spec=TrainingBatch)
        batch.kc_targets = [targets_0, targets_1, targets_2]
        batch.attention_mask = torch.ones(3, 10)  # determining device

        # Execute
        result = create_kc_batch(batch, tokenizer, target_specs)

        # Verify
        self.assertIn("kc_has_pos_effective", result)
        mask = result["kc_has_pos_effective"]

        self.assertEqual(mask.shape, (3,))
        self.assertTrue(mask[0].item())
        self.assertTrue(mask[1].item())
        self.assertFalse(mask[2].item())

        # Check individual masks too
        self.assertTrue(result["kc_pos_mask_bag_pos"][0].any())
        self.assertTrue(result["kc_pos_mask_bag_pos"][1].any())
        self.assertFalse(result["kc_pos_mask_bag_pos"][2].any())


if __name__ == "__main__":
    unittest.main()
