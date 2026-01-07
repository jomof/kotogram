import unittest

import torch

# We need to mock KCTrainer or isolate the logic.
# Since logic is embedded in train_epoch, it's hard to unit test directly without a full harness.
# However, we can create a small test that constructs the tensors and verifies the math logic
# mirroring what's in the trainer.


class TestSaturationGating(unittest.TestCase):
    def test_gating_logic(self):
        """Verifies that penalty is only computed on examples with positive targets."""
        # 1. Setup Data
        # Batch size 4
        # Ex 0: Has Pos, Low Logits -> No Pen
        # Ex 1: Has Pos, High Logits -> High Pen
        # Ex 2: No Pos, High Logits -> No Pen (Gated)
        # Ex 3: No Pos, Low Logits -> No Pen

        has_pos_mask = torch.tensor([True, True, False, False])

        # Logit Threshold is 3.0
        # Ex 0: Max=2.0 (Excess=0)
        # Ex 1: Max=4.0 (Excess=1.0) -> Pen = 1.0
        # Ex 2: Max=5.0 (Excess=2.0) -> Pen = 4.0 if ungated, 0 if gated
        # Ex 3: Max=1.0 (Excess=0)

        pmax_logit_per_ex = torch.tensor([2.0, 4.0, 5.0, 1.0], requires_grad=True)
        logit_thr = 3.0

        # 2. Replicate Logic
        sat_excess = (pmax_logit_per_ex - logit_thr).clamp_min(0.0)

        # Global (for comparison)
        sat_pen_global = (sat_excess * sat_excess).mean()
        # Ex 1 (1.0) + Ex 2 (4.0) = 5.0 / 4 = 1.25
        self.assertAlmostEqual(sat_pen_global.item(), 1.25)

        # Gated
        # Only use sat_excess where has_pos_mask is True
        # Filtered: [0.0, 1.0] (Ex 0 and 1)
        # Mean: (0+1)/2 = 0.5
        # Or does .mean() apply to determining the penalty over the full batch?
        # The implementation uses:
        # sat_excess_pos = sat_excess[has_pos_mask]
        # sat_pen = (sat_excess_pos * sat_excess_pos).mean()
        # So it is the mean over POSITIVE examples only.

        if has_pos_mask.any():
            sat_excess_pos = sat_excess[has_pos_mask]
            sat_pen = (sat_excess_pos * sat_excess_pos).mean()
        else:
            sat_pen = pmax_logit_per_ex.sum() * 0.0

        # Expected: Ex 0 (0.0) and Ex 1 (1.0^2=1.0). Mean = 0.5.
        self.assertAlmostEqual(sat_pen.item(), 0.5)

        # 3. Verify Gradients
        sat_pen.backward()
        grad = pmax_logit_per_ex.grad
        # Ex 0: < Thr, grad 0
        # Ex 1: > Thr, used. d/dx (x-3)^2 = 2(x-3) = 2(1) = 2. Div by N_pos=2 -> 1.0.
        # Ex 2: > Thr, unused. grad 0.
        # Ex 3: < Thr, grad 0.

        self.assertAlmostEqual(grad[0].item(), 0.0)
        self.assertAlmostEqual(grad[1].item(), 1.0)  # 2(4-3)/2 = 1.0
        self.assertEqual(grad[2].item(), 0.0)  # Gated out!
        self.assertEqual(grad[3].item(), 0.0)

    def test_no_positives(self):
        """Verifies behavior when no examples have positives."""
        has_pos_mask = torch.tensor([False, False])
        pmax_logit_per_ex = torch.tensor([4.0, 5.0], requires_grad=True)
        # Both high, but no positives.

        if has_pos_mask.any():
            sat_excess_pos = (pmax_logit_per_ex - 3.0).clamp_min(0.0)[has_pos_mask]
            sat_pen = (sat_excess_pos**2).mean()
        else:
            sat_pen = pmax_logit_per_ex.sum() * 0.0

        self.assertEqual(sat_pen.item(), 0.0)

        sat_pen.backward()
        self.assertTrue(pmax_logit_per_ex.grad is not None)
        self.assertEqual(pmax_logit_per_ex.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
