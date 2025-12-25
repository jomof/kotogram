import os
import unittest
from unittest.mock import patch

import torch

from train.config import TrainerConfig
from train.trainer import configure_runtime_thread_limits, get_safe_dataloader_config


class TestResourceSafety(unittest.TestCase):
    def test_get_safe_dataloader_config_cuda_normal(self):
        config = TrainerConfig()
        device = torch.device("cuda")

        with patch("os.cpu_count", return_value=32):
            with patch("os.getloadavg", return_value=(1.0, 1.0, 1.0)):
                # Normal conditions on CUDA
                # min(4, 32//8) = 4
                settings = get_safe_dataloader_config(config, device)
                self.assertEqual(settings["num_workers"], 4)
                self.assertTrue(settings["pin_memory"])
                self.assertTrue(settings["persistent_workers"])

    def test_get_safe_dataloader_config_cpu(self):
        config = TrainerConfig()
        device = torch.device("cpu")

        settings = get_safe_dataloader_config(config, device)
        self.assertEqual(settings["num_workers"], 0)
        self.assertFalse(settings["pin_memory"])

    def test_get_safe_dataloader_config_stressed(self):
        config = TrainerConfig()
        device = torch.device("cuda")

        # High load: loadavg=100 on 8-core CPU
        with patch("os.cpu_count", return_value=8):
            with patch("os.getloadavg", return_value=(100.0, 100.0, 100.0)):
                settings = get_safe_dataloader_config(config, device)
                # min(4, 8//8=1) -> 1. Downgrade doesn't reduce 1 further in my impl if workers=1
                # But if it was 4, it would be 2.

        # High load with 32 cores: base=4, stressed=2
        with patch("os.cpu_count", return_value=32):
            with patch("os.getloadavg", return_value=(100.0, 100.0, 100.0)):
                settings = get_safe_dataloader_config(config, device)
                self.assertEqual(settings["num_workers"], 2)
                self.assertFalse(settings["pin_memory"])

    def test_configure_runtime_thread_limits(self):
        config = TrainerConfig(cpu_threads=4, interop_threads=2)

        with patch("torch.set_num_threads") as mock_set_threads:
            with patch("torch.set_num_interop_threads") as mock_set_interop:
                with patch.dict(os.environ, {}, clear=True):
                    configure_runtime_thread_limits(config)
                    mock_set_threads.assert_called_once_with(4)
                    mock_set_interop.assert_called_once_with(2)
                    self.assertEqual(os.environ["OMP_NUM_THREADS"], "4")
                    self.assertEqual(os.environ["TOKENIZERS_PARALLELISM"], "false")


if __name__ == "__main__":
    unittest.main()
