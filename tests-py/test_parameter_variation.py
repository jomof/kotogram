import os
import tempfile
import unittest

# pylint: disable=no-name-in-module
from unittest.mock import MagicMock

import torch
from torch import nn

from train.config import DataLoaderConfig, HardwareConfig, KCConfig, TrainerConfig
from train.dataset import StyleDataset
from train.profile import Timer
from train.trainer import KCTrainer, Trainer


class DummyKCModel(nn.Module):
    """A dummy model for KCTrainer testing."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(1, 1)
        self.encoder = nn.Linear(1, 1)
        self.kc_head = nn.Linear(1, 1)
        self.config = MagicMock()
        # Mock config attributes needed by KCTrainer checks
        self.config.kc_topk = 1
        self.config.kc_vocab_size = 10
        self.kc_decoders = nn.Linear(1, 1)

        # Pragmatic/classification heads required by Trainer
        # Note: value heads removed - MSE predictions via KC decoder
        # Note: Register is now handled by KC decoder, not a separate head
        self.formality_pragmatic_head = nn.Linear(1, 1)
        self.gender_pragmatic_head = nn.Linear(1, 1)
        self.grammaticality_head = nn.Linear(1, 1)

        # Unified pooler required by Trainer
        self.pooler = nn.Linear(1, 1)

        # Legacy/Alias support
        self.grammaticality_classifier = self.grammaticality_head

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Not used in these tests")


class TestParameterVariation(unittest.TestCase):
    def setUp(self):
        # pylint: disable=consider-using-with
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_mock_dataset(self):
        mock = MagicMock(spec=StyleDataset)
        mock.__len__.return_value = 100
        # Trainer.__init__ calls these to initialize loss functions
        mock.get_formality_class_weights.return_value = torch.ones(5)
        mock.get_gender_class_weights.return_value = torch.ones(3)
        mock.get_grammaticality_class_weights.return_value = torch.ones(2)
        # KCTrainer usage
        mock.filter_by_grammaticality.return_value = mock
        return mock

    def test_trainer_output_path_variation(self):
        """Test Trainer initialization with varying output paths."""
        mock_model = DummyKCModel()
        # Explicit usage to satisfy dead code analysis
        self.assertIsNotNone(mock_model.formality_pragmatic_head)

        mock_dataset = self._create_mock_dataset()
        config = TrainerConfig(
            hardware=HardwareConfig(cpu_reserve_cores=1), device="cpu"
        )
        dl_config = DataLoaderConfig(
            num_workers=0, pin_memory=False, persistent_workers=False
        )

        # Variation 1: Default-ish path
        path1 = os.path.join(self.output_path, "default")
        trainer1 = Trainer(
            mock_model,
            mock_dataset,
            mock_dataset,
            config,
            dl_config,
            dl_config,
            output_path=path1,
        )
        self.assertEqual(trainer1.output_path, path1)

        # Variation 2: Different path
        path2 = os.path.join(self.output_path, "custom_variation")
        trainer2 = Trainer(
            mock_model,
            mock_dataset,
            mock_dataset,
            config,
            dl_config,
            dl_config,
            output_path=path2,
        )
        self.assertEqual(trainer2.output_path, path2)

    def test_kc_trainer_variation(self):
        """Test KCTrainer parameters including freeze_encoder and accum."""
        model = DummyKCModel()
        # Explicit usage to satisfy dead code analysis
        self.assertIsNotNone(model.grammaticality_head)
        self.assertIsNotNone(model.formality_pragmatic_head)
        self.assertIsNotNone(model.gender_pragmatic_head)

        mock_dataset = self._create_mock_dataset()

        dl_config = DataLoaderConfig(
            num_workers=0, pin_memory=False, persistent_workers=False
        )

        # Case 1: freeze_encoder_epochs > 0 (default=1), accum=1
        config1 = TrainerConfig(device="cpu", grad_accum_steps=1)
        kc_config1 = KCConfig(freeze_encoder_epochs=1)
        trainer1 = KCTrainer(model, mock_dataset, config1, dl_config, kc_config1)
        trainer1.optimizer = MagicMock()
        # pylint: disable=protected-access
        trainer1._perform_optimizer_step(model)

        # Case 2: freeze_encoder_epochs = 0, accum=2
        config2 = TrainerConfig(device="cpu", grad_accum_steps=2)
        kc_config2 = KCConfig(freeze_encoder_epochs=0)
        trainer2 = KCTrainer(model, mock_dataset, config2, dl_config, kc_config2)
        trainer2.optimizer = MagicMock()
        # pylint: disable=protected-access
        trainer2._perform_optimizer_step(model)

    def test_timer_variation(self):
        """Test Timer with varying initialization parameters."""
        # Variation 1: Minimal args
        t1 = Timer("test_minimal")
        self.assertIsNone(t1.output_path)
        self.assertIsNone(t1.profile_dir)
        self.assertIsNone(t1.console)

        # Variation 2: All args
        out_path = os.path.join(self.output_path, "timer.jsonl")
        prof_dir = os.path.join(self.output_path, "profile")
        mock_console = MagicMock()
        t2 = Timer(
            "test_full",
            output_path=out_path,
            profile_dir=prof_dir,
            console=mock_console,
        )
        self.assertEqual(t2.output_path, out_path)
        self.assertEqual(t2.profile_dir, prof_dir)
        self.assertEqual(t2.console, mock_console)


if __name__ == "__main__":
    unittest.main()
