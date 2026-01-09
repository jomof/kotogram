import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

from kotogram.model import ModelConfig

# pylint: disable=import-private-name
from train.pytorch_utils import (
    calculate_element_size_bytes,
    calculate_model_static_memory,
    estimate_optimal_batch_size,
)


class TestAutoBatchSize(unittest.TestCase):
    def setUp(self):
        # Default small config
        self.config = ModelConfig(
            vocab_sizes={"pos": 100},
            d_model=256,
            hidden_dim=512,
            num_layers=2,
            max_seq_len=128,
            num_heads=4,
            kc_vocab_size=1000,
        )

    def test_static_memory_calculation(self):
        mem = calculate_model_static_memory(self.config)
        self.assertGreater(mem, 0)
        # 1M params * 16 bytes ~ 16MB roughly.
        # Check rough order of magnitude [1MB, 100MB]
        self.assertTrue(
            10**6 <= mem <= 10**8, f"Static memory {mem} out of expected range"
        )

    def test_element_size_calculation(self):
        # Base
        base_size = calculate_element_size_bytes(self.config, is_kc=False)

        # KC
        kc_size = calculate_element_size_bytes(self.config, is_kc=True)

        # KC should be larger due to head overhead
        self.assertGreater(kc_size, base_size)

        # Check magnitude: 128 seq * 256 dim * 2 layers * 16 ~ 1MB
        self.assertTrue(
            10**5 <= base_size <= 10**7, f"Element size {base_size} out of range"
        )

    def test_heuristic_cuda_generic(self):
        # 8GB VRAM
        with patch("torch.cuda.get_device_properties") as mock_props:
            mock_props.return_value.total_memory = 8 * (1024**3)
            device = torch.device("cuda:0")

            bs = estimate_optimal_batch_size(device, self.config, is_kc=False)

            # Check reasonable bounds
            self.assertGreater(bs, 32)
            # Check power of 2
            self.assertEqual(bs & (bs - 1), 0, f"Batch size {bs} is not power of 2")

    def test_heuristic_mps_generic(self):
        with patch("importlib.util.find_spec", return_value=True):
            with patch.dict(sys.modules, {"psutil": MagicMock()}):
                mock_psutil = sys.modules["psutil"]
                # 16GB System RAM -> 15% available -> ~2.4GB
                mock_psutil.virtual_memory.return_value.total = 16 * (1024**3)
                device = torch.device("mps")

                bs = estimate_optimal_batch_size(device, self.config, is_kc=False)

                # Check reasonable bounds
                self.assertGreater(bs, 32)
                # Check power of 2
                self.assertEqual(bs & (bs - 1), 0, f"Batch size {bs} is not power of 2")

    def test_kc_vs_trainer_difference(self):
        """Ensure KCTrainer (is_kc=True) gets smaller batch size if KC head is huge."""
        huge_kc_config = ModelConfig(
            vocab_sizes={"pos": 100},
            d_model=256,
            kc_vocab_size=100000,  # Large KC vocab
            num_layers=2,
        )

        with patch("torch.cuda.get_device_properties") as mock_props:
            mock_props.return_value.total_memory = 4 * (1024**3)  # 4GB
            device = torch.device("cuda:0")

            bs_trainer = estimate_optimal_batch_size(
                device, huge_kc_config, is_kc=False
            )
            bs_kc = estimate_optimal_batch_size(device, huge_kc_config, is_kc=True)

            # KC should have larger element size -> smaller batch size
            self.assertGreater(bs_trainer, 32)
            self.assertGreater(bs_kc, 32)

            # Since we round down to power of 2, strict inequality might not hold if they fall in same bin
            # But with huge vocab difference, they should differ.
            self.assertLessEqual(bs_kc, bs_trainer)

    def test_heuristic_cpu(self):
        device = torch.device("cpu")
        bs = estimate_optimal_batch_size(device, self.config, is_kc=False)
        self.assertEqual(bs, 32)

    def test_heuristic_unknown_device(self):
        device = torch.device("meta")
        with self.assertRaises(RuntimeError):
            estimate_optimal_batch_size(device, self.config, is_kc=False)
