import unittest
from unittest.mock import patch

from train.config import TrainerConfig
from train.trainer import choose_torch_threads, choose_workers


class TestInteractiveMode(unittest.TestCase):
    def test_choose_workers_interactive(self):
        """Test worker selection in interactive mode."""
        config = TrainerConfig(interactive_mode=True, cpu_reserve_cores=2)

        # Mock cpu_count to return 8
        with patch("os.cpu_count", return_value=8):
            # 8 cores, reserve 2 = 6 usable.
            # Interactive: min(4, max(1, 6 // 4)) = min(4, 1) = 1?
            # Wait, 6 // 4 = 1.
            # Let's try 16 cores. 16 - 2 = 14. 14 // 4 = 3.

            # Case 1: 8 Cores
            settings = choose_workers(config)
            # usable = 6. 6//4 = 1.
            self.assertEqual(settings["num_workers"], 1)

        # Case 2: 16 Cores
        with patch("os.cpu_count", return_value=16):
            # 16-2 = 14. 14//4 = 3.
            settings = choose_workers(config)
            self.assertEqual(settings["num_workers"], 3)

    def test_choose_workers_non_interactive(self):
        """Test worker selection in non-interactive mode."""
        config = TrainerConfig(interactive_mode=False)

        # Mock cpu_count to return 8
        with patch("os.cpu_count", return_value=8):
            # 8 cores. usable = 8.
            # Non-interactive: min(8, max(2, 8 // 2)) = 4.
            settings = choose_workers(config)
            self.assertEqual(settings["num_workers"], 4)

    def test_choose_torch_threads_interactive(self):
        """Test torch thread selection in interactive mode."""
        config = TrainerConfig(interactive_mode=True, cpu_reserve_cores=2)

        # Mock cpu_count to return 8
        with patch("os.cpu_count", return_value=8):
            # 8-2 = 6 usable.
            # Intra-op: max(1, 6 // 2) = 3.
            # Inter-op: max(1, min(4, 6 // 4)) = 1.
            intra, inter = choose_torch_threads(config)
            self.assertEqual(intra, 3)
            self.assertEqual(inter, 1)

    def test_choose_torch_threads_non_interactive(self):
        """Test torch thread selection in non-interactive mode."""
        config = TrainerConfig(interactive_mode=False)

        # Mock cpu_count to return 8
        with patch("os.cpu_count", return_value=8):
            # 8 usable.
            # Intra-op: max(1, 8 // 1) = 8.
            # Inter-op: max(1, min(4, 8 // 4)) = 2.
            intra, inter = choose_torch_threads(config)
            self.assertEqual(intra, 8)
            self.assertEqual(inter, 2)

    def test_explicit_overrides(self):
        """Test that explicit config overrides auto-tuning."""
        config = TrainerConfig(
            interactive_mode=True, dataloader_num_workers=10, torch_num_threads=5
        )
        with patch("os.cpu_count", return_value=4):
            settings = choose_workers(config)
            self.assertEqual(settings["num_workers"], 10)

            intra, inter = choose_torch_threads(config)
            self.assertEqual(intra, 5)


if __name__ == "__main__":
    unittest.main()
