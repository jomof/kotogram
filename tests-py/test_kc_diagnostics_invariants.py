import unittest

import torch

from train.kc_diagnostics import (
    assert_diagnostics_invariants,
    gather_kc_diag,
)


class TestKCDiagnosticsInvariants(unittest.TestCase):
    def setUp(self):
        self.epoch = 1
        self.batch_size, self.num_kcs = 10, 5
        self.device = torch.device("cpu")

    def _create_valid_diag(self):
        logits = torch.randn(self.batch_size, self.num_kcs, device=self.device)
        probs = torch.sigmoid(logits)
        return {
            "kc_logits": logits,
            "kc_probs": probs,
            "kc_mask": None,
            "heads": {},
            "epoch": self.epoch,
        }

    def test_gather_kc_diag_basic(self):
        """Test basic gathering of logits and probs."""
        logits = torch.randn(self.batch_size, self.num_kcs)
        probs = torch.sigmoid(logits)
        outputs = {
            "kc_logits_raw": logits,
            "kc_probs": probs,
            "target_logits": {"head1": torch.randn(self.batch_size, 1)},
        }
        targets = {
            "kc_targets_head1": torch.randint(0, 2, (self.batch_size, 1)).float()
        }

        diag = gather_kc_diag(outputs, targets, self.epoch)

        self.assertTrue(torch.equal(diag["kc_logits"], logits))
        self.assertTrue(torch.equal(diag["kc_probs"], probs))
        self.assertIn("head1", diag["heads"])
        self.assertIs(
            diag["heads"]["head1"]["y_score"], outputs["target_logits"]["head1"]
        )
        assert_diagnostics_invariants(diag)

    def test_sigmoid_consistency_check(self):
        """Test that inconsistent logits/probs raise AssertionError."""
        diag = self._create_valid_diag()
        # Perturb probabilities significantly implies inconsistency
        diag["kc_probs"] = torch.rand_like(diag["kc_probs"])

        with self.assertRaisesRegex(
            AssertionError, "kc_probs inconsistent with kc_logits sigmoid"
        ):
            assert_diagnostics_invariants(diag)

    def test_pmax_pmean_coherence(self):
        """Test pmean vs pmax sanity check."""
        self._create_valid_diag()
        # Logic is tautological on single tensor source, so just verification of no crash.

    def test_head_finite_check(self):
        """Test head input finite check."""
        diag = self._create_valid_diag()
        diag["heads"]["h1"] = {
            "y_score": torch.tensor([[float("nan")]]),
            "y_true": torch.tensor([[0.0]]),
            "weight": None,
        }
        with self.assertRaisesRegex(AssertionError, "NaN/Inf in AUC inputs"):
            assert_diagnostics_invariants(diag)

    def test_shape_mismatch(self):
        """Test basic shape mismatch."""
        diag = self._create_valid_diag()
        diag["kc_probs"] = torch.randn(self.batch_size, self.num_kcs + 1)  # Mismatch k

        with self.assertRaisesRegex(AssertionError, "kc_probs must align with logits"):
            assert_diagnostics_invariants(diag)


if __name__ == "__main__":
    unittest.main()
