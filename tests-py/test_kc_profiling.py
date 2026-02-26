import unittest
from unittest.mock import MagicMock, patch

import torch

from train.config import DataLoaderConfig, TrainerConfig
from train.dataset import StyleDataset
from train.models import TrainingClassifier
from train.trainer import KCTrainer


class DummyDataset(StyleDataset):
    # pylint: disable=super-init-not-called
    def __init__(self):
        self.tokenizer = MagicMock()
        self.tokenizer.pad_id = 0
        # Add required attributes for style oversampling
        self.indices = torch.arange(10)
        self.offsets = torch.arange(0, 110, 10, dtype=torch.long)
        self.labels = {
            "f_val": torch.zeros(10, dtype=torch.float32),
            "g_val": torch.zeros(10, dtype=torch.float32),
        }

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return {}

    def filter_by_grammaticality(self, label: int = 1):
        return self


class TestKCProfiling(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock(spec=TrainingClassifier)
        # Mocking config within model
        # We need to explicitly set the config mock to allow setting attributes on it
        self.mock_model.config = MagicMock()
        self.mock_model.config.max_seq_len = 128
        self.mock_model.config.kc_vocab_size = 100

        # Mock submodules
        self.mock_model.kc_head = MagicMock()
        self.mock_model.kc_head.parameters.return_value = [
            torch.nn.Parameter(torch.empty(1))
        ]
        self.mock_model.embedding = MagicMock()
        self.mock_model.embedding.parameters.return_value = [
            torch.nn.Parameter(torch.empty(1))
        ]
        self.mock_model.encoder = MagicMock()
        self.mock_model.encoder.parameters.return_value = [
            torch.nn.Parameter(torch.empty(1))
        ]

        self.mock_dataset = DummyDataset()

        self.mock_config = TrainerConfig(
            device="cpu",
            batch_size=2,
        )
        self.mock_dl_config = DataLoaderConfig(
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None,
        )

    @patch("train.kc_trainer.get_profile_dir")
    @patch("train.kc_trainer.os.getpid")
    def test_kc_trainer_timer_initialization(self, mock_getpid, mock_get_profile_dir):
        # Setup mocks
        mock_get_profile_dir.return_value = "/tmp/fake_profile_dir"
        mock_getpid.return_value = 12345

        from train.config import KCConfig

        # Initialize KCTrainer
        trainer = KCTrainer(
            model=self.mock_model,
            dataset=self.mock_dataset,
            config=self.mock_config,
            dl_config=self.mock_dl_config,
            kc_config=KCConfig(),
        )

        # Assertion: Check if timers are initialized with correct paths
        # Note: This test expects the FIX to be implemented.
        # Without the fix, these output_paths would be None.

        self.assertIsNotNone(trainer.train_timer_data.output_path)
        expected_data_path = "/tmp/fake_profile_dir/kc_data_12345.jsonl"
        self.assertEqual(trainer.train_timer_data.output_path, expected_data_path)

        self.assertIsNotNone(trainer.train_timer_compute.output_path)
        expected_compute_path = "/tmp/fake_profile_dir/kc_compute_12345.jsonl"
        self.assertEqual(trainer.train_timer_compute.output_path, expected_compute_path)

    @patch("train.kc_trainer.get_profile_dir")
    def test_kc_trainer_timer_no_profile_env(self, mock_get_profile_dir):
        # Simulate TRAIN_PROFILE=0 case where get_profile_dir returns None
        mock_get_profile_dir.return_value = None

        from train.config import KCConfig

        trainer = KCTrainer(
            model=self.mock_model,
            dataset=self.mock_dataset,
            config=self.mock_config,
            dl_config=self.mock_dl_config,
            kc_config=KCConfig(),
        )

        # Assertion: Timers should have None as output_path
        self.assertIsNone(trainer.train_timer_data.output_path)
        self.assertIsNone(trainer.train_timer_compute.output_path)


if __name__ == "__main__":
    unittest.main()
