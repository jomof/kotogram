import os
import unittest
from unittest.mock import patch

import torch

from train.config import (
    HardwareConfig,
    TrainerConfig,
    configure_runtime_thread_limits,
)


class TestResourceSafety(unittest.TestCase):
    def test_get_safe_dataloader_config_cuda_normal(self):
        config = TrainerConfig()
        device = torch.device("cuda")

        with patch("os.cpu_count", return_value=32):
            with patch("os.getloadavg", return_value=(1.0, 1.0, 1.0)):
                # Normal conditions on CUDA
                # min(4, 32//8) = 4
                dl_config = config.resolve_dataloader_config(device, is_main=True)
                self.assertEqual(dl_config.num_workers, 4)
                self.assertTrue(dl_config.pin_memory)
                self.assertTrue(dl_config.persistent_workers)

    def test_get_safe_dataloader_config_cpu(self):
        config = TrainerConfig()
        device = torch.device("cpu")

        dl_config = config.resolve_dataloader_config(device, is_main=True)
        self.assertEqual(dl_config.num_workers, 0)
        self.assertFalse(dl_config.pin_memory)

    def test_configure_runtime_thread_limits(self):
        config = TrainerConfig(
            hardware=HardwareConfig(torch_num_threads=4, torch_num_interop_threads=2)
        )

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
