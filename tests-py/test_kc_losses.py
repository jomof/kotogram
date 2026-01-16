# pylint: disable=protected-access
import unittest
from unittest.mock import MagicMock, patch

import torch
from kc_test_utils import KCTrainerTestBase

from train.kc import KcFamilyId


class TestKCLosses(KCTrainerTestBase):
    def setUp(self) -> None:
        super().setUp()
        # Mock kc_decoders with decoders attribute for bias delta tracking
        self.model.kc_decoders = MagicMock()
        self.model.kc_decoders.decoders = {}
        self.model.kc_decoders.return_value = {
            KcFamilyId.BAG_POS.name.lower(): torch.randn(2, 10, requires_grad=True)
        }
        object.__setattr__(
            self.trainer.config, "kc_target_specs", {KcFamilyId.BAG_POS: 10}
        )

    @patch("train.kc_trainer.create_kc_batch")
    def test_sparsity_term_uses_topk(self, mock_create_batch):
        """Test that sparsity term uses topk_vals instead of avg_prob."""
        self.trainer.kc_sparsity_weight = 1.0
        self.trainer.kc_sparsity_mode = "target_density"
        self.trainer.freeze_encoder_epochs = 0

        batch = MagicMock()
        batch.feature_inputs = {}
        batch.attention_mask = torch.ones(2, 5)
        batch.formality_value = torch.zeros(2)  # Neutral formality
        batch.gender_value = torch.zeros(2)  # Neutral gender
        batch.register_labels = torch.zeros(2, 14)  # All neutral registers
        self.mock_loader.__iter__.return_value = iter([batch])
        self.mock_loader.__len__.return_value = 1

        mock_create_batch.return_value = {
            f"kc_targets_{KcFamilyId.BAG_POS.name.lower()}": torch.zeros(2, 10)
        }

        # Be sure to enable gradient on tensors involved in loss
        probs = torch.full((2, 100), 0.5, requires_grad=True)
        topk_vals = torch.full((2, 10), 0.8, requires_grad=True)
        topk_inds = torch.zeros((2, 10), dtype=torch.long)
        sparse_act = torch.zeros((2, 100), requires_grad=True)
        logits_raw = torch.zeros((2, 100), requires_grad=True)

        target_logits_val = torch.randn((2, 10), requires_grad=True)

        outputs = {
            "kc_logits_raw": logits_raw,
            "kc_logits_effective": logits_raw,
            "kc_probs": probs,
            "topk_vals": topk_vals,
            "topk_inds": topk_inds,
            "sparse_activations": sparse_act,
            "target_logits": {KcFamilyId.BAG_POS.name.lower(): target_logits_val},
        }
        self.model.return_value = outputs
        self.model.config.kc_topk = 8

        res = self.trainer.train_epoch(epoch=0)
        # k_i = ceil(0.4 * 5) + 6 (k_bonus for short) = 8. spar = 8.0 / 8 = 1.0.
        self.assertAlmostEqual(res.avg_sparsity, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
