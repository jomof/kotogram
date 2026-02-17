import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from train.config import KCConfig, TrainerConfig
from train.trainer import KCTrainer


class KCTrainerTestBase(unittest.TestCase):
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
            kl_sparse_weight=1.0,
            diversity_weight=0.1,
            entropy_weight=0.0,
            temperature_thawed=1.0,
        )

        # Mock model
        self.model = MagicMock()
        # Some tests deliberately delete this, but we can provide it by default or not?
        # test_kc_losses deletes it: del self.model.kc_decoders
        # test_kc_diag_alignment doesn't mention it (so MagicMock has it by default).
        # We'll leave it as MagicMock property. Tests can del it.

        self.model.config.kc_temperature = 1.0
        self.model.config.kc_vocab_size = 100
        self.model.config.kc_target_specs = {"test_target": 10}

        # Mock head parameters
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

        # Mock DataLoader config
        self.dl_config = MagicMock()
        self.dl_config.num_workers = 0
        self.dl_config.pin_memory = False
        self.dl_config.persistent_workers = False
        self.dl_config.prefetch_factor = None

        # Create Trainer with patched DataLoader
        with patch("train.kc_trainer.DataLoader"):
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

    def _create_mock_outputs(self, batch_size, vocab_size, target_key):
        return {
            "target_logits": {
                target_key: torch.randn(batch_size, vocab_size, requires_grad=True)
            },
            "kc_logits": torch.randn(batch_size, vocab_size, requires_grad=True),
            "kc_logits_raw": torch.randn(batch_size, vocab_size, requires_grad=True),
            "kc_logits_effective": torch.randn(
                batch_size, vocab_size, requires_grad=True
            ),
            "kc_probs": torch.rand(batch_size, vocab_size, requires_grad=True),
            "kc_probs_clean": torch.rand(batch_size, vocab_size),
            "logits_usage": torch.randn(batch_size, vocab_size, requires_grad=True),
        }
