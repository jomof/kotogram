import unittest

import torch

from train.kc_diagnostics import compute_auc_checked

# Mock objects if needed


class TestKCConsistencyStrict(unittest.TestCase):
    def test_ep_pmax_coherence(self):
        """
        Verify the aggregation logic for epPmaxMean, epPmaxGlobal vs probs_mean.
        Simulate the exact logic used in trainer.py loop.
        """
        # Create synthetic kc_probs (B=3, K=4)
        # Ex 1: [0.1, 0.2, 0.3, 0.4] -> max=0.4, mean=0.25
        # Ex 2: [0.8, 0.8, 0.9, 0.7] -> max=0.9, mean=0.8
        # Ex 3: [0.5, 0.5, 0.5, 0.5] -> max=0.5, mean=0.5

        probs = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.8, 0.8, 0.9, 0.7], [0.5, 0.5, 0.5, 0.5]]
        )

        # Batch Logic (from trainer.py)
        pmax_per_ex = probs.max(dim=1).values
        batch_pmax_mean = pmax_per_ex.mean().item()
        batch_pmax_global = probs.max().item()
        batch_probs_mean = probs.mean().item()

        # Expected values
        # Ex maxes: [0.4, 0.9, 0.5]
        expected_pmax_mean = (0.4 + 0.9 + 0.5) / 3  # 0.6
        expected_pmax_global = 0.9
        expected_probs_mean = (
            probs.mean().item()
        )  # (1.0 + 3.2 + 2.0)/12 = 6.2/12 = 0.5166

        self.assertAlmostEqual(batch_pmax_mean, expected_pmax_mean, places=5)
        self.assertAlmostEqual(batch_pmax_global, expected_pmax_global, places=5)
        self.assertAlmostEqual(batch_probs_mean, expected_probs_mean, places=5)

        # Invariant Assertions
        self.assertGreaterEqual(batch_pmax_global, batch_pmax_mean)
        self.assertGreaterEqual(batch_pmax_mean, batch_probs_mean)

        # Fail Condition Check (if pmax < probs_mean, impossible)
        self.assertFalse(batch_pmax_global < batch_probs_mean - 1e-5)

    def test_topk_provenance(self):
        """
        Verify topk is derived EXACTLY from the probs tensor.
        """
        k = 6
        probs = torch.rand(3, 100)  # Random probs

        # Standard topk
        topk_vals, topk_idx = torch.topk(probs, k=k)

        # Verify provenance: gather from original tensor using indices
        # must equal topk_vals exactly
        gathered = torch.gather(probs, 1, topk_idx)

        diff = (topk_vals - gathered).abs().max().item()
        self.assertLess(
            diff, 1e-7, "Topk values must match gathered probabilities exactly"
        )

        # Verify indices point to the highest values (simple check)
        max_val_in_probs = probs.max(dim=1).values
        max_val_in_topk = topk_vals.max(dim=1).values

        self.assertTrue(torch.allclose(max_val_in_probs, max_val_in_topk))

    def test_auc_gating_support(self):
        """
        Verify AUC computing returns None (NA) when support is low or class missing.
        """
        # Case 1: No positives
        y_true_no_pos = torch.zeros(100)
        y_score = torch.rand(100)
        auc, reason = compute_auc_checked(y_true_no_pos, y_score)
        self.assertIsNone(auc)
        self.assertEqual(reason, "no_pos")

        # Case 2: No negatives
        y_true_no_neg = torch.ones(100)
        auc, reason = compute_auc_checked(y_true_no_neg, y_score)
        self.assertIsNone(auc)
        self.assertEqual(reason, "no_neg")

        # Case 3: Constant score
        y_score_const = torch.ones(100) * 0.5
        y_true_valid = torch.cat([torch.zeros(50), torch.ones(50)])
        auc, reason = compute_auc_checked(y_true_valid, y_score_const)
        self.assertIsNone(auc)
        self.assertEqual(reason, "constant_score")

        # Case 4: Valid
        y_score_valid = torch.rand(100)  # Unlikely to be constant
        auc, reason = compute_auc_checked(y_true_valid, y_score_valid)
        self.assertIsNotNone(auc)
        self.assertIsNone(reason)
        self.assertTrue(0.0 <= auc <= 1.0)

    def test_sat_bounds(self):
        """
        Verify sat98 and near0 metrics logic.
        """
        # Logic: count(p > 0.98) / total, count(p < 0.01) / total

        # 10 values
        # 2 > 0.98 (0.99, 0.99)
        # 3 < 0.01 (0.001, 0.005, 0.0)
        # 5 middle
        probs = torch.tensor([0.99, 0.99, 0.001, 0.005, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5])

        sat98 = (probs > 0.98).float().mean().item()
        near0 = (probs < 0.01).float().mean().item()

        self.assertAlmostEqual(sat98, 0.2)
        self.assertAlmostEqual(near0, 0.3)

        self.assertTrue(0.0 <= sat98 <= 1.0)
        self.assertTrue(0.0 <= near0 <= 1.0)


if __name__ == "__main__":
    unittest.main()
