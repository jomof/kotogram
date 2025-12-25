import unittest
from unittest.mock import patch

import torch

from train.config import DataLoaderSettings, HardwareConfig, TrainerConfig


class TestInteractiveMode(unittest.TestCase):
    def test_choose_workers_interactive(self):
        """Test worker selection in interactive mode."""
        config = TrainerConfig(
            interactive_mode=True,
            hardware=HardwareConfig(cpu_reserve_cores=2),
            dataloader=DataLoaderSettings(interactive_dataloader=False),
        )

        # Mock cpu_count to return 8
        with patch("os.cpu_count", return_value=8):
            # 8 cores, reserve 2 = 6 usable.
            # Interactive: min(4, max(1, 6 // 4)) = min(4, 1) = 1?
            # Wait, 6 // 4 = 1.
            # Let's try 16 cores. 16 - 2 = 14. 14 // 4 = 3.

            # Case 1: 8 Cores
            dl_config = config.resolve_dataloader_config(
                torch.device("cpu"), is_main=True
            )
            # usable = 6.
            # In current config.py logic, CPU always yields 0 workers.
            self.assertEqual(dl_config.num_workers, 0)

        # Case 2: 16 Cores
        with patch("os.cpu_count", return_value=16):
            # 16 cores. resolve_dataloader_config on CPU always 0.
            dl_config = config.resolve_dataloader_config(
                torch.device("cpu"), is_main=True
            )
            self.assertEqual(dl_config.num_workers, 0)

    def test_choose_workers_non_interactive(self):
        """Test worker selection in non-interactive mode."""
        config = TrainerConfig(
            interactive_mode=False,
            dataloader=DataLoaderSettings(interactive_dataloader=False),
        )

        # Mock cpu_count to return 8
        with patch("os.cpu_count", return_value=8):
            dl_config = config.resolve_dataloader_config(
                torch.device("cuda"), is_main=True
            )
            # 8 cores. 8 // 8 = 1. max(2, 1) = 2.
            self.assertEqual(dl_config.num_workers, 2)

    def test_choose_torch_threads_interactive(self):
        """Test torch thread selection in interactive mode."""
        # Mock cpu_count to return 8
        with patch("os.cpu_count", return_value=8):
            # 8-2 = 6 usable.
            # Intra-op: max(1, 6 // 2) = 3.
            # Inter-op: max(1, min(4, 6 // 4)) = 1.
            config = TrainerConfig(
                interactive_mode=True,
                hardware=HardwareConfig(cpu_reserve_cores=2),
                dataloader=DataLoaderSettings(interactive_dataloader=False),
            )
            self.assertEqual(config.hardware.cpu_threads, 3)
            self.assertEqual(config.hardware.interop_threads, 1)

    def test_choose_torch_threads_non_interactive(self):
        """Test torch thread selection in non-interactive mode."""
        # Mock cpu_count to return 8
        with patch("os.cpu_count", return_value=8):
            # 8 usable.
            # Intra-op: max(1, 8 // 1) = 8.
            # Inter-op: max(1, min(4, 8 // 4)) = 2.
            config = TrainerConfig(
                interactive_mode=False,
                dataloader=DataLoaderSettings(interactive_dataloader=False),
            )
            self.assertEqual(config.hardware.cpu_threads, 8)
            self.assertEqual(config.hardware.interop_threads, 2)

    def test_explicit_overrides(self):
        """Test that explicit config overrides auto-tuning."""
        config = TrainerConfig(
            interactive_mode=True,
            dataloader=DataLoaderSettings(num_workers=10, interactive_dataloader=False),
            hardware=HardwareConfig(torch_num_threads=5),
        )
        with patch("os.cpu_count", return_value=4):
            dl_config = config.resolve_dataloader_config(
                torch.device("cuda"), is_main=True
            )
            self.assertEqual(dl_config.num_workers, 10)

            self.assertEqual(config.hardware.cpu_threads, 5)


if __name__ == "__main__":
    unittest.main()
