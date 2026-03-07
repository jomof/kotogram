import os
import unittest

from training_test_utils import Bottle


@unittest.skipIf(os.environ.get("GITHUB_ACTIONS") == "true", "Skipping on GitHub CI")
class TestTrainingResumption(unittest.TestCase):
    def test_combined_resumption_scenarios(self):
        """
        Combines resumption scenarios to save test execution time:
        1. Style Auto-Resume (default when checkpoint exists)
        2. Config Transition (Style -> KC)
        3. KC Resumption (implicit - no flag needed)
        4. Retrain behavior (--retrain to start fresh)
        """
        common_args = "--embed-dim 64 --hidden-dim 128 --num-layers 1 --num-heads 2"

        with Bottle(self) as bottle:
            bottle.populate_test_data()
            bottle.train_style("--label")

            # Paths
            model_path = bottle.resolve_path("[models]/style/model.pt")
            continuation_path = bottle.resolve_path("[models]/style/continuation.json")

            # =========================================================================
            # PART 1: Style Auto-Resume (From test_auto_resume.py)
            # =========================================================================
            print("\n[UnifiedTest] Part 1: Style Training - Epoch 1")

            # 1.A: Train Epoch 1 (Fresh)
            res = bottle.train_style(f"--epochs 1 --kc-epochs 0 {common_args}")
            bottle.assert_style_epochs_trained([1])
            self.assertNotIn("Resuming: Loaded model weights from", res.stdout)
            self.assertTrue(os.path.exists(model_path), "model.pt should exist")
            self.assertTrue(
                os.path.exists(continuation_path), "continuation.json should exist"
            )

            # Verify KC config - KC is always enabled now
            # Config no longer has kc_enabled field

            print("\n[UnifiedTest] Part 1: Style Training - Epoch 2 (Auto-Resume)")
            # 1.B: Train Epoch 2 (Auto-Resume) - resume is default when continuation exists
            res = bottle.train_style(f"--epochs 2 --kc-epochs 0 {common_args}")
            bottle.assert_style_epochs_trained([1, 2])
            self.assertIn("Resuming: Loaded model weights from", res.stdout)

            print("\n[UnifiedTest] Part 1: Style Training - Epoch 3 (Auto-Resume)")
            # 1.C: Train Epoch 3 (Auto-Resume continues) - no explicit flag needed
            res = bottle.train_style(f"--epochs 3 --kc-epochs 0 {common_args}")
            bottle.assert_style_epochs_trained([1, 2, 3])
            self.assertIn("Resuming: Loaded model weights from", res.stdout)

            # =========================================================================
            # PART 2: Config Transition & KC Pretrain (From test_kc_pretrain.py)
            # =========================================================================
            print("\n[UnifiedTest] Part 2: KC Pretrain Injection")

            # KC training - KC is always enabled now
            # Just run KC epochs to verify resumption scenario
            res = bottle.train_style(
                f"--kc-epochs 1 --kc-vocab-size 256 --epochs 3 {common_args}"
            )
            bottle.assert_kc_epochs_trained([1])
            # Style shouldn't re-run if it's already at epoch 3 and we asked for 3,
            # BUT the KC pretrain might invalidate style checkpoint compatibility if embeddings changed?
            # In this simple case, vocab/embeddings are stable.
            # The test `test_kc_pretrain` used `epochs 1`.

            self.assertTrue(os.path.exists(model_path), "model.pt should exist")

            # =========================================================================
            # PART 3: KC Resumption (From test_resume_pretrain.py)
            # =========================================================================
            # 3.A: IMPLICIT KC Resume (NO --resume flag)
            # This is the critical regression test: running with higher --kc-epochs
            # WITHOUT explicit --resume should still resume from KC checkpoint, not restart.
            print("\n[UnifiedTest] Part 3A: KC Implicit Resume - Epoch 2 (NO --resume)")

            res = bottle.train_style(f"--kc-epochs 2 --epochs 3 {common_args}")
            # Must train epoch 2, not restart at epoch 1
            bottle.assert_kc_epochs_trained([1, 2])

            # 3.B: KC Resume Epoch 3 (implicit, no flag needed)
            print("\n[UnifiedTest] Part 3B: KC Resume - Epoch 3")
            res = bottle.train_style(f"--kc-epochs 3 --epochs 3 {common_args}")
            bottle.assert_kc_epochs_trained([1, 2, 3])

            # =========================================================================
            # PART 4: Retrain Behavior (From test_auto_resume.py)
            # =========================================================================
            print("\n[UnifiedTest] Part 4: Retrain Style")

            # Force retrain (ignore existing model/continuation).
            # We'll ask for epochs 1 for speed.
            # Vary percent to 50% and verify it's logged.
            res = bottle.train_style(
                f"--epochs 1 --retrain --percent 50 --kc-epochs 0 {common_args}"
            )
            bottle.assert_style_epochs_trained([1])
            self.assertNotIn("Resuming: Loaded model weights from", res.stdout)
            self.assertIn("Retrain:        from scratch", res.stdout)
            self.assertIn("Data usage:", res.stdout)
            self.assertIn("50.0%", res.stdout)

            # KC should NOT have been touched/retrained since we didn't ask for it
            # KC resumption respects kc-epochs


if __name__ == "__main__":
    unittest.main()
