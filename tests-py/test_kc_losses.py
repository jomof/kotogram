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
    def test_kl_sparse_loss_at_boundary(self, mock_create_batch):
        """Test that KL-sparse loss is positive when all probs are 0.5 (far from target ρ)."""
        self.trainer.kl_sparse_weight = 1.0
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

        # All probs at 0.5 → sharpening loss p*(1-p) = 0.25 (maximum)
        probs = torch.full((2, 100), 0.5, requires_grad=True)
        logits_raw = torch.zeros((2, 100), requires_grad=True)

        target_logits_val = torch.randn((2, 10), requires_grad=True)

        outputs = {
            "kc_logits_raw": logits_raw,
            "kc_logits_effective": logits_raw,
            "kc_probs": probs,
            "kc_probs_clean": probs.detach(),
            "target_logits": {KcFamilyId.BAG_POS.name.lower(): target_logits_val},
            "logits_usage": logits_raw,
        }
        self.model.return_value = outputs

        res = self.trainer.train_epoch(epoch=0)
        # All probs at 0.5, ρ̂ = 0.5 >> ρ = 0.05 → large KL divergence
        self.assertGreater(res.avg_kl_sparse, 0.0)


if __name__ == "__main__":
    unittest.main()
