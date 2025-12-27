import os
import sys
import unittest
from unittest.mock import MagicMock

# Ensure projects root is in path
sys.path.append(os.getcwd())

# pylint: disable=wrong-import-position
from train.config import TrainerConfig


class TestDataLoaderConfig(unittest.TestCase):
    def test_mps_acceleration_enabled(self):
        """Verify explicit MPS acceleration logic in config."""
        config = TrainerConfig()

        # Determine strict device logic
        # Mocking torch.device behavior
        mps_device = MagicMock()
        mps_device.type = "mps"

        # Case 1: MPS Device
        dl_config = config.resolve_dataloader_config(mps_device)
        self.assertGreater(dl_config.num_workers, 0, "MPS should have > 0 workers")
        self.assertFalse(
            dl_config.pin_memory, "MPS should disable pinned memory (not supported)"
        )

        # Case 2: CPU Device (should still be 0 unless forced)
        cpu_device = MagicMock()
        cpu_device.type = "cpu"
        dl_config_cpu = config.resolve_dataloader_config(cpu_device)
        self.assertEqual(
            dl_config_cpu.num_workers, 0, "CPU should have 0 workers by default"
        )

    def test_cuda_defaults_preserved(self):
        """Verify CUDA logic is maintained."""
        config = TrainerConfig()
        cuda_device = MagicMock()
        cuda_device.type = "cuda"
        str(cuda_device)  # Ensure it doesn't crash

        # Since logic checks "cuda" in str(device)
        # We need to mock that behavior or use a real string
        # But resolve_dataloader_config takes a device object usually.
        # Let's rely on the implementation detail: if is_cuda is False, but device.type is cuda?
        # Actually in config.py: `is_cuda = "cuda" in str(device)`

        # Using a MagicMock that returns "cuda:0" for str()
        cuda_device.__str__.return_value = "cuda:0"

        dl_config = config.resolve_dataloader_config(cuda_device)
        self.assertGreater(dl_config.num_workers, 0)
        self.assertTrue(dl_config.pin_memory)


if __name__ == "__main__":
    unittest.main()
