import unittest
from unittest.mock import MagicMock, patch

import torch

from train.config import KCConfig, TrainerConfig
from train.kc import KcFamilyId
from train.trainer import KCTrainer


class TestKCDenseTraining(unittest.TestCase):
    def setUp(self):
        # pylint: disable=too-many-locals, duplicate-code
        self.config = TrainerConfig(
            device="cpu",
            batch_size=2,
            kc_epochs=1,
            grad_accum_steps=1,
            learning_rate=0.001,
            kc_target_specs={KcFamilyId.BAG_POS: 1000},
        )
        self.kc_config = KCConfig(
            # defaults
        )

        # Mock model
        self.model = MagicMock()
        self.model.config.kc_target_specs = {
            KcFamilyId.BAG_POS: 1000
        }  # > 256 to trigger dense path
        self.model.config.kc_vocab_size = 100
        del self.model.kc_decoders

        # Mock dataset
        self.dataset = MagicMock()
        self.dataset.tokenizer.field_vocabs = {}

        # Mock DataLoader
        self.dl_config = MagicMock()
        with patch("train.trainer.DataLoader"):
            self.trainer = KCTrainer(
                model=self.model,
                dataset=self.dataset,
                config=self.config,
                dl_config=self.dl_config,
                kc_config=self.kc_config,
            )
            self.mock_loader = MagicMock()
            self.trainer.data_loader = self.mock_loader
            self.trainer.optimizer = MagicMock()
            self.trainer.optimizer.param_groups = [
                {"params": [], "lr": 0.001},
                {"params": [], "lr": 0.001},
            ]

    @patch("train.trainer.create_kc_batch")
    @patch("train.trainer.KCEpochDiag")
    def test_dense_sampling_shapes(self, mock_kc_diag_cls, mock_create_batch):
        # pylint: disable=too-many-locals
        # Setup batch with targets
        batch_size = 2
        vocab_size = 1000

        # Mock instance
        mock_diag_instance = MagicMock()
        mock_kc_diag_cls.return_value = mock_diag_instance

        # Targets: row 0 has 2 positives, row 1 has 3 positives
        targets = torch.zeros(batch_size, vocab_size)
        targets[0, [10, 20]] = 1.0
        targets[1, [30, 40, 50]] = 1.0

        logits = torch.randn(batch_size, vocab_size)

        batch = MagicMock()
        batch.feature_inputs = {}
        batch.attention_mask = torch.ones(batch_size, 5)
        self.mock_loader.__iter__.return_value = iter([batch])
        self.mock_loader.__len__.return_value = 1

        mock_create_batch.return_value = {
            f"kc_targets_{KcFamilyId.BAG_POS.name.lower()}": targets
        }

        outputs = {
            "kc_logits": torch.zeros(
                (batch_size, 10), requires_grad=True
            ),  # irrelevant
            "kc_logits_raw": torch.zeros((batch_size, 10), requires_grad=True),
            "kc_logits_effective": torch.zeros((batch_size, 10), requires_grad=True),
            "kc_probs": torch.sigmoid(
                torch.zeros((batch_size, 10), requires_grad=True)
            ),
            "topk_vals": torch.zeros((batch_size, 5), requires_grad=True),
            "topk_inds": torch.zeros((batch_size, 5), dtype=torch.long),
            "sparse_activations": torch.zeros((batch_size, 10), requires_grad=True),
            "target_logits": {KcFamilyId.BAG_POS.name.lower(): logits},
        }
        self.model.return_value = outputs

        self.trainer.train_epoch(epoch=0)

        # Verify update_family arguments
        # We expect one call for "bag_pos" from KcFamilyId.BAG_POS
        call_args = mock_diag_instance.update_family.call_args
        self.assertIsNotNone(call_args)

        kwargs = call_args.kwargs
        if not kwargs and call_args.args:
            # Bind args manually if necessary, or just check args by index
            # Signature: update_family(family_name, pos_ids, pos_mask, probs, targets, nll, ...)
            family_name = call_args.args[0]
            pos_ids = call_args.args[1]
            pos_mask = call_args.args[2]
            probs = call_args.args[3]
            targets_arg = call_args.args[4]
        else:
            family_name = kwargs.get("family_name") or call_args.args[0]
            pos_ids = kwargs.get("pos_ids") or call_args.args[1]
            pos_mask = kwargs.get("pos_mask") or call_args.args[2]
            probs = kwargs.get("probs") or call_args.args[3]
            targets_arg = kwargs.get("targets") or call_args.args[4]

        self.assertEqual(family_name, KcFamilyId.BAG_POS.name.lower())

        # Check shapes - The User Request Standard
        # pos_ids: (B, P_pos)
        self.assertEqual(pos_ids.dim(), 2, f"pos_ids should be 2D, got {pos_ids.shape}")
        self.assertEqual(pos_ids.size(0), batch_size, "pos_ids batch size mismatch")

        # pos_mask: (B, P_pos)
        self.assertEqual(
            pos_mask.dim(), 2, f"pos_mask should be 2D, got {pos_mask.shape}"
        )
        self.assertEqual(pos_mask.shape, pos_ids.shape, "pos_mask should match pos_ids")

        # probs: (B, P_sample)
        self.assertEqual(probs.dim(), 2, f"probs should be 2D, got {probs.shape}")
        self.assertEqual(probs.size(0), batch_size, "probs batch size mismatch")

        # targets: (B, P_sample)
        self.assertEqual(
            targets_arg.dim(), 2, f"targets should be 2D, got {targets_arg.shape}"
        )
        self.assertEqual(targets_arg.shape, probs.shape, "targets should match probs")

    @patch("train.trainer.create_kc_batch")
    @patch("train.trainer.KCEpochDiag")
    def test_missing_topk_raises_error(self, _mock_kc_diag_cls, mock_create_batch):
        # Setup batch
        mock_create_batch.return_value = {}
        batch = MagicMock()
        batch.attention_mask = torch.ones(2, 5)
        batch.feature_inputs = {}
        self.mock_loader.__len__.return_value = 1
        self.mock_loader.__iter__.return_value = iter([batch])

        # Outputs missing topk_inds
        outputs = {
            "kc_logits_raw": torch.zeros((2, 10)),
            "kc_logits_effective": torch.zeros((2, 10)),
            "kc_probs": torch.sigmoid(torch.zeros((2, 10))),
            # "topk_inds" is purposefully missing
            "topk_vals": torch.zeros((2, 5)),
            "target_logits": {},
        }
        self.model.return_value = outputs

        with self.assertRaisesRegex(RuntimeError, "KC training requires topk_inds"):
            self.trainer.train_epoch(0)

    @patch("train.trainer.create_kc_batch")
    @patch("train.trainer.KCEpochDiag")
    @patch("torch.randint")
    def test_dense_negative_sampling_range(
        self, mock_randint, mock_kc_diag_cls, mock_create_batch
    ):
        # Mock instance
        mock_diag_instance = MagicMock()
        mock_kc_diag_cls.return_value = mock_diag_instance

        batch = MagicMock()
        batch.attention_mask = torch.ones(2, 5)
        batch.feature_inputs = {}
        self.mock_loader.__len__.return_value = 1
        self.mock_loader.__iter__.return_value = iter([batch])

        # Force dense path
        self.model.config.kc_target_specs = {KcFamilyId.BAG_POS: 500}
        mock_create_batch.return_value = {
            f"kc_targets_{KcFamilyId.BAG_POS.name.lower()}": torch.zeros((2, 500))
        }

        # Fake logits > 256
        logits = torch.randn(2, 500, requires_grad=True)
        outputs = {
            "kc_logits": torch.zeros((2, 10)),
            "kc_logits_raw": torch.zeros((2, 10)),
            "kc_logits_effective": torch.zeros((2, 10)),
            "kc_probs": torch.sigmoid(torch.zeros((2, 10))),
            "topk_inds": torch.zeros((2, 5), dtype=torch.long),
            "topk_vals": torch.zeros((2, 5)),
            "sparse_activations": torch.zeros((2, 10)),
            "target_logits": {KcFamilyId.BAG_POS.name.lower(): logits},
        }
        self.model.return_value = outputs

        # Return zeros from randint
        mock_randint.return_value = torch.zeros((2, 128), dtype=torch.long)

        self.trainer.train_epoch(0)

        call_args = mock_randint.call_args
        self.assertIsNotNone(call_args, "randint not called")
        # args: (low, high, size, ...)
        # low should be 4
        self.assertEqual(
            call_args.args[0], 4, f"randint low should be 4, got {call_args.args[0]}"
        )


if __name__ == "__main__":
    unittest.main()
