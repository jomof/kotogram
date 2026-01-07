import unittest
from unittest.mock import MagicMock, patch

import torch
from kc_test_utils import KCTrainerTestBase

from train.kc import KcFamilyId
from train.kc_diagnostics import KCEpochDiag

# pylint: disable=protected-access,too-many-locals,unused-variable


class TestKCDiagAlignment(KCTrainerTestBase):
    def test_update_family_rejects_shape_mismatch(self):
        # T1
        diag = KCEpochDiag()

        inds = torch.zeros((2, 5), dtype=torch.long)
        pos_mask = torch.zeros((2, 5), dtype=torch.bool)
        probs = torch.zeros((10,), dtype=torch.float)  # Wrong
        targets = torch.zeros((2, 5), dtype=torch.float)

        with self.assertRaisesRegex(ValueError, "pos_ids must be 2D"):
            diag.update_family(
                KcFamilyId.BAG_POS.name.lower(),
                inds.flatten(),
                pos_mask.flatten(),
                probs,
                targets.flatten(),
                0.0,
            )

        with self.assertRaisesRegex(ValueError, "Shape mismatch"):
            # Ensure all passed are 2D but mismatched
            diag.update_family(
                KcFamilyId.BAG_POS.name.lower(),
                inds,
                pos_mask,
                torch.zeros((2, 6)),
                targets,
                0.0,
            )

    @patch("train.trainer.create_kc_batch")
    @patch("train.trainer.KCEpochDiag")
    def test_dense_small_vocab_calls_update_family_with_2d_tensors(
        self, mock_kc_diag_cls, mock_create_batch
    ):
        # T2
        mock_diag = MagicMock()
        mock_kc_diag_cls.return_value = mock_diag

        vocab_size = 12

        batch_size = 3

        object.__setattr__(
            self.trainer.config, "kc_target_specs", {KcFamilyId.BAG_POS: vocab_size}
        )
        self.model.config.kc_vocab_size = vocab_size
        self.trainer.data_loader.__len__.return_value = 1

        batch_iter = MagicMock()
        batch_iter.feature_inputs = {}
        batch_iter.attention_mask = torch.ones(batch_size, 10)
        self.trainer.data_loader.__iter__.return_value = iter([batch_iter])

        # Targets
        targets = torch.zeros((batch_size, vocab_size))
        targets[0, 1] = 1.0
        mock_create_batch.return_value = {
            f"kc_targets_{KcFamilyId.BAG_POS.name.lower()}": targets
        }

        # Outputs
        outputs = self._create_mock_outputs(
            batch_size, vocab_size, KcFamilyId.BAG_POS.name.lower()
        )
        self.model.return_value = outputs
        # Fix for unconditional decoding overwriting target_logits
        self.model.kc_decoders.return_value = outputs["target_logits"]

        self.trainer.train_epoch(0)

        # Expect update_family call
        mock_diag.update_family.assert_called()
        args = mock_diag.update_family.call_args.args
        name, v_ids, pos_mask, probs, tgs, _ = args[:6]

        self.assertEqual(name, KcFamilyId.BAG_POS.name.lower())
        self.assertEqual(v_ids.dim(), 2)
        self.assertEqual(pos_mask.dim(), 2)
        self.assertEqual(probs.dim(), 2)
        self.assertEqual(tgs.dim(), 2)
        self.assertEqual(v_ids.shape, (batch_size, vocab_size))

    @patch("train.trainer.create_kc_batch")
    @patch("train.trainer.KCEpochDiag")
    def test_dense_large_vocab_sampled_diag_alignment(
        self, mock_kc_diag_cls, mock_create_batch
    ):
        # T3
        mock_diag = MagicMock()
        mock_kc_diag_cls.return_value = mock_diag

        vocab_size = 1000
        batch_size = 2

        object.__setattr__(
            self.trainer.config, "kc_target_specs", {KcFamilyId.BAG_POS: vocab_size}
        )
        self.model.config.kc_vocab_size = vocab_size
        self.trainer.data_loader.__len__.return_value = 1

        batch_mock = MagicMock()
        batch_mock.feature_inputs = {}
        batch_mock.attention_mask = torch.ones(batch_size, 10)
        self.trainer.data_loader.__iter__.return_value = iter([batch_mock])

        targets = torch.zeros((batch_size, vocab_size))
        targets[0, 100] = 1.0
        mock_create_batch.return_value = {
            f"kc_targets_{KcFamilyId.BAG_POS.name.lower()}": targets
        }

        outputs = self._create_mock_outputs(
            batch_size, vocab_size, KcFamilyId.BAG_POS.name.lower()
        )
        self.model.return_value = outputs
        # Fix for unconditional decoding overwriting target_logits
        self.model.kc_decoders.return_value = outputs["target_logits"]

        self.trainer.train_epoch(0)

        mock_diag.update_family.assert_called()
        args = mock_diag.update_family.call_args.args
        name, inds, pos_mask, probs, tgs, _ = args[:6]

        self.assertEqual(name, KcFamilyId.BAG_POS.name.lower())
        self.assertEqual(inds.dim(), 2)
        # Should be (B, P + N). P depends on max positives in rows.
        # Row 0 has 1 pos. Row 1 has 0 pos. Max pos = 1 (or min 1).
        # N = 128. Total width approx 129.
        self.assertEqual(inds.size(1), 1 + 128)
        self.assertEqual(pos_mask.shape, inds.shape)
        self.assertEqual(probs.shape, inds.shape)
        self.assertEqual(tgs.shape, inds.shape)

        # Check pos_mask populated correctly (should be first column True for row 0)
        self.assertTrue(pos_mask[0, 0].item())
        self.assertFalse(pos_mask[0, 1].item())  # Next is negative sample

    def test_sparse_bce_sampled_diag_alignment(self):
        # T4
        mock_diag = MagicMock(spec=KCEpochDiag)

        batch_size = 2
        vocab_size = 500
        logits_f = torch.randn(batch_size, vocab_size)
        pos_inds = torch.tensor([[10, 20], [30, 40]], dtype=torch.long)
        pos_mask = torch.tensor([[True, True], [True, False]], dtype=torch.bool)

        self.trainer._bce_sampled_from_sparse(
            logits_f,
            pos_inds,
            pos_mask,
            vocab_size,
            neg_count=50,
            diag=mock_diag,
            family_name=KcFamilyId.BAG_POS.name.lower(),
        )

        mock_diag.update_family.assert_called()
        args = mock_diag.update_family.call_args.args
        name, inds, pm, probs, tgs, _ = args[:6]

        self.assertEqual(name, KcFamilyId.BAG_POS.name.lower())
        self.assertEqual(inds.dim(), 2)
        # Width: P (2) + N (50) = 52
        self.assertEqual(inds.size(1), 52)
        self.assertEqual(pm.shape, inds.shape)
        self.assertEqual(probs.shape, inds.shape)
        self.assertEqual(tgs.shape, inds.shape)

    @patch("torch.nn.utils.clip_grad_norm_")
    def test_grad_clip_disabled_means_no_clip_called(self, mock_clip):
        # T5
        # T5
        object.__setattr__(self.trainer.config, "gradient_clip", 0.0)

        # Setup fake params with grad
        p = torch.nn.Parameter(torch.tensor([1.0]))
        p.grad = torch.tensor([10.0])  # large grad

        self.trainer.optimizer.param_groups = [{"params": [p]}]

        self.trainer._perform_optimizer_step(self.model)

        mock_clip.assert_not_called()

        # Enable clip
        object.__setattr__(self.trainer.config, "gradient_clip", 1.0)
        p.grad = torch.tensor([10.0])  # Reset (zero_grad clears it?)
        # _perform_optimizer_step calls zero_grad at end.

        self.trainer._perform_optimizer_step(self.model)
        mock_clip.assert_called()


if __name__ == "__main__":
    unittest.main()
