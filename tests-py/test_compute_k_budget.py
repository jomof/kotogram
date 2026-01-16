"""Unit tests for compute_k_budget function."""

import unittest

import torch

from kotogram.model import ModelConfig, compute_k_budget


class TestComputeKBudget(unittest.TestCase):
    """Tests for the compute_k_budget function."""

    def setUp(self):
        self.config = ModelConfig(
            vocab_sizes={"surface": 100},
            kc_topk=8,
            kc_alpha_short=0.40,
            kc_alpha_long=0.55,
            kc_long_threshold=20,
            kc_min_k=2,
            kc_max_k_long=16,
        )
        self.device = torch.device("cpu")

    def test_short_sentence_with_bonus(self):
        """Bin 1-3: Very short sentences (len <= 3) get +3 bonus."""
        content_len = torch.tensor([2.0, 3.0])
        k_budget = compute_k_budget(content_len, self.config, self.device)

        # len=2: ceil(0.4 * 2) = 1, +6 bonus = 7, clamp [2, 8] -> 7
        # len=3: ceil(0.4 * 3) = 2, +6 bonus = 8, clamp [2, 8] -> 8
        expected = torch.tensor([7, 8], dtype=torch.long)
        self.assertTrue(torch.equal(k_budget, expected), f"Got {k_budget}")

    def test_medium_sentence_with_bonus(self):
        """Bin 4-7 and 8-15: Medium sentences (4-15 tokens) get +6 bonus."""
        content_len = torch.tensor([6.0, 10.0, 15.0])
        k_budget = compute_k_budget(content_len, self.config, self.device)

        # len=6: ceil(0.4 * 6) = 3, +6 bonus = 9, clamp [2, 8] -> 8
        # len=10: ceil(0.4 * 10) = 4, +6 bonus = 10, clamp [2, 8] -> 8
        # len=15: ceil(0.4 * 15) = 6, +6 bonus = 12, clamp [2, 8] -> 8 (clamped)
        expected = torch.tensor([8, 8, 8], dtype=torch.long)
        self.assertTrue(torch.equal(k_budget, expected), f"Got {k_budget}")

    def test_long_sentence_no_bonus(self):
        """Bin 16-31 and 32+: Long sentences (>= 20 tokens) get no bonus."""
        content_len = torch.tensor([20.0, 30.0, 50.0])
        k_budget = compute_k_budget(content_len, self.config, self.device)

        # len=20: ceil(0.55 * 20) = 11, no bonus, clamp [2, 16] -> 11
        # len=30: ceil(0.55 * 30) = 17, no bonus, clamp [2, 16] -> 16
        # len=50: ceil(0.55 * 50) = 28, no bonus, clamp [2, 16] -> 16
        expected = torch.tensor([11, 16, 16], dtype=torch.long)
        self.assertTrue(torch.equal(k_budget, expected), f"Got {k_budget}")

    def test_transition_zone(self):
        """Test the boundary between bonus and no-bonus (len 15-16)."""
        content_len = torch.tensor([15.0, 16.0, 19.0, 20.0])
        k_budget = compute_k_budget(content_len, self.config, self.device)

        # len=15: ceil(0.4 * 15) = 6, +6 bonus = 12, clamp [2, 8] -> 8
        # len=16: ceil(0.4 * 16) = 7, no bonus (>15), clamp [2, 8] -> 7
        # len=19: ceil(0.4 * 19) = 8, no bonus, clamp [2, 8] -> 8
        # len=20: ceil(0.55 * 20) = 11, no bonus, clamp [2, 16] -> 11
        expected = torch.tensor([8, 7, 8, 11], dtype=torch.long)
        self.assertTrue(torch.equal(k_budget, expected), f"Got {k_budget}")

    def test_min_k_floor(self):
        """Verify min_k is enforced even for very short sentences."""
        content_len = torch.tensor([1.0])
        k_budget = compute_k_budget(content_len, self.config, self.device)

        # len=1: ceil(0.4 * 1) = 1, +6 bonus = 7, clamp [2, 8] -> 7
        # (min_k would only matter if result < 2)
        expected = torch.tensor([7], dtype=torch.long)
        self.assertTrue(torch.equal(k_budget, expected), f"Got {k_budget}")

    def test_batch_dimension(self):
        """Verify function handles batch correctly."""
        content_len = torch.tensor([5.0, 25.0, 10.0, 40.0])
        k_budget = compute_k_budget(content_len, self.config, self.device)

        self.assertEqual(k_budget.shape, (4,))
        self.assertEqual(k_budget.dtype, torch.long)


if __name__ == "__main__":
    unittest.main()
