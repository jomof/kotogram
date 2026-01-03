# pylint: disable=protected-access
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from train.config import KCConfig, TrainerConfig
from train.trainer import KCTrainer


class TestKCLosses(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.config = TrainerConfig(
            device=self.device,
            batch_size=2,
            kc_epochs=1,
            grad_accum_steps=1,
            learning_rate=0.001,
        )
        self.kc_config = KCConfig(
            sparsity_weight=1.0,
            diversity_weight=0.1,
            lb_weight=0.1,
            temperature_thawed=1.0,
            log_level="info",
            show_epoch_table=False,
            show_step_checks=False,
            show_grad_norms=False,
            first_batch_debug_every=0,
            first_batch_debug_epochs=[],
        )

        # Mock model
        self.model = MagicMock()
        del self.model.kc_decoders

        self.model.config.kc_temperature = 1.0
        self.model.config.kc_vocab_size = 100
        self.model.config.kc_target_specs = {"test_target": 10}
        self.model.kc_head.linear.weight = nn.Parameter(torch.randn(10, 10))
        self.model.kc_head.parameters.return_value = [self.model.kc_head.linear.weight]
        self.model.to.return_value = self.model
        self.model.named_parameters.return_value = [
            ("kc_head.linear.weight", self.model.kc_head.linear.weight)
        ]

        # Mock dataset
        self.dataset = MagicMock()
        self.dataset.filter_by_grammaticality.return_value = self.dataset
        self.dataset.tokenizer.field_vocabs = {}

        # Mock DataLoader
        self.dl_config = MagicMock()
        self.dl_config.num_workers = 0
        self.dl_config.pin_memory = False
        self.dl_config.persistent_workers = False
        self.dl_config.prefetch_factor = None

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
    def test_sparsity_term_uses_topk(self, mock_create_batch):
        """Test that sparsity term uses topk_vals instead of avg_prob."""
        self.trainer.kc_sparsity_weight = 1.0
        self.trainer.kc_sparsity_mode = "target_density"
        self.trainer.freeze_encoder_epochs = 0

        batch = MagicMock()
        batch.feature_inputs = {}
        batch.attention_mask = torch.ones(2, 5)
        self.mock_loader.__iter__.return_value = iter([batch])
        self.mock_loader.__len__.return_value = 1

        mock_create_batch.return_value = {"kc_targets_test_target": torch.zeros(2, 10)}

        # Be sure to enable gradient on tensors involved in loss
        probs = torch.full((2, 100), 0.5, requires_grad=True)
        topk_vals = torch.full((2, 10), 0.8, requires_grad=True)
        topk_inds = torch.zeros((2, 10), dtype=torch.long)
        sparse_act = torch.zeros((2, 100), requires_grad=True)
        logits_raw = torch.zeros((2, 100), requires_grad=True)

        target_logits_val = torch.randn((2, 10), requires_grad=True)

        outputs = {
            "kc_logits_raw": logits_raw,
            "kc_probs": probs,
            "topk_vals": topk_vals,
            "topk_inds": topk_inds,
            "sparse_activations": sparse_act,
            "target_logits": {"test_target": target_logits_val},
        }
        self.model.return_value = outputs

        res = self.trainer.train_epoch(epoch=0)
        self.assertAlmostEqual(res.avg_sparsity, 0.5, places=4)

    @patch("train.trainer.create_kc_batch")
    def test_logit_gap_accumulation(self, mock_create_batch):
        """Test that logit gap is correctly calculated and reported."""
        self.trainer.kc_sparsity_weight = 0.0
        self.trainer.freeze_encoder_epochs = 0

        batch = MagicMock()
        batch.feature_inputs = {}
        batch.attention_mask = torch.ones(2, 5)
        self.mock_loader.__iter__.return_value = iter([batch])
        self.mock_loader.__len__.return_value = 1

        mock_create_batch.return_value = {"kc_targets_test_target": torch.zeros(2, 10)}

        logits_raw = torch.randn((2, 10), requires_grad=True)
        # We need to manually modifying values in a way that preserves gradient,
        # or just create new tensor.
        with torch.no_grad():
            logits_raw[:, 0] = 10.0
            logits_raw[:, 9] = 5.0
        # Re-enable grad? modifying leaf variable inplace is tricky if used?
        # Creating fresh leaf
        d = torch.randn((2, 10))
        d[:, 0] = 10.0
        d[:, 9] = 5.0
        logits_raw = d.clone().detach().requires_grad_(True)

        topk_inds = torch.arange(10).repeat(2, 1)

        outputs = {
            "kc_logits_raw": logits_raw,
            "kc_probs": torch.zeros_like(logits_raw, requires_grad=True),
            "topk_vals": torch.zeros((2, 10), requires_grad=True),
            "topk_inds": topk_inds,
            "sparse_activations": torch.zeros_like(logits_raw, requires_grad=True),
            "target_logits": {"test_target": torch.randn((2, 10), requires_grad=True)},
        }
        self.model.return_value = outputs

        res = self.trainer.train_epoch(epoch=0)
        self.assertAlmostEqual(res.epoch_stats.avg_logit_gap, 5.0, places=4)

    @patch("train.trainer.create_kc_batch")
    def test_kl_metric_matches_load_balance(self, mock_create_batch):
        """Test that kl_to_uniform is based on load_balance_loss logic."""
        self.trainer.kc_diversity_warmup_epochs = 0
        self.trainer.freeze_encoder_epochs = 0
        self.trainer.kc_lb_weight_thawed = 1.0

        batch = MagicMock()
        batch.feature_inputs = {}
        batch.attention_mask = torch.ones(2, 5)
        self.mock_loader.__iter__.side_effect = lambda: iter([batch])
        self.mock_loader.__len__.return_value = 1

        mock_create_batch.return_value = {"kc_targets_test_target": torch.zeros(2, 10)}

        logits_raw = torch.zeros((2, 10), requires_grad=True)
        outputs = {
            "kc_logits_raw": logits_raw,
            "kc_probs": torch.zeros_like(logits_raw, requires_grad=True),
            "topk_vals": torch.zeros((2, 5), requires_grad=True),
            "topk_inds": torch.zeros(2, 5, dtype=torch.long),
            "sparse_activations": torch.zeros_like(logits_raw, requires_grad=True),
            "target_logits": {"test_target": torch.randn((2, 10), requires_grad=True)},
        }
        self.model.return_value = outputs
        self.model.config.kc_vocab_size = 10

        res = self.trainer.train_epoch(epoch=0)
        self.assertAlmostEqual(res.epoch_stats.avg_kl_to_uniform, 0.0, places=4)

        # Case 2: Peaky
        self.model.config.kc_vocab_size = 100
        logits_raw_peaky = torch.zeros((2, 100))
        logits_raw_peaky[:, 0] = 1000.0
        logits_raw_peaky = logits_raw_peaky.clone().detach().requires_grad_(True)

        outputs["kc_logits_raw"] = logits_raw_peaky
        outputs["kc_probs"] = torch.zeros((2, 100), requires_grad=True)  # just dummy
        # also need to update other shapes if vocab size changed?
        # vocab size 100 in mock model config
        outputs["sparse_activations"] = torch.zeros((2, 100), requires_grad=True)

        res = self.trainer.train_epoch(epoch=0)
        self.assertAlmostEqual(res.epoch_stats.avg_kl_to_uniform, 1.0, places=2)

    @patch("train.trainer.create_kc_batch")
    def test_nonfinite_streak_reset(self, mock_create_batch):
        """Test that nonfinite streak resets to 0."""
        self.trainer._nonfinite_streak = -10
        self.trainer.kc_sparsity_weight = 0.0

        batch = MagicMock()
        batch.feature_inputs = {}
        batch.attention_mask = torch.ones(2, 5)
        self.mock_loader.__iter__.return_value = iter([batch])
        self.mock_loader.__len__.return_value = 1

        mock_create_batch.return_value = {"kc_targets_test_target": torch.zeros(2, 10)}

        outputs = {
            "kc_logits_raw": torch.zeros((2, 10), requires_grad=True),
            "kc_probs": torch.zeros((2, 10), requires_grad=True),
            "topk_vals": torch.zeros((2, 5), requires_grad=True),
            "topk_inds": torch.zeros(2, 5, dtype=torch.long),
            "sparse_activations": torch.zeros((2, 10), requires_grad=True),
            "target_logits": {"test_target": torch.randn((2, 10), requires_grad=True)},
        }
        self.model.return_value = outputs

        self.trainer.train_epoch(epoch=0)
        self.assertEqual(self.trainer._nonfinite_streak, 0)


if __name__ == "__main__":
    unittest.main()
