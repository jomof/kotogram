import os
import unittest

from training_test_utils import Bottle


@unittest.skipIf(os.environ.get("GITHUB_ACTIONS") == "true", "Skipping on GitHub CI")
class TestTrainingResumption(unittest.TestCase):
    def test_combined_resumption_scenarios(self):
        """
        Combines resumption scenarios to save test execution time:
        1. Style Auto-Resume & Manual Resume
        2. Config Transition (Style -> KC)
        3. KC Resumption
        4. Retrain behavior
        """
        common_args = "--embed-dim 64 --hidden-dim 128 --num-layers 1 --num-heads 2"

        with Bottle(self) as bottle:
            bottle.populate_test_data()
            bottle.train_style("--label")

            # Paths
            checkpoint_path = bottle.resolve_path(
                "[models]/style-support/checkpoint.pt"
            )
            kc_ckpt = bottle.resolve_path("[models]/style-support/checkpoint_kc.pt")

            # =========================================================================
            # PART 1: Style Auto-Resume (From test_auto_resume.py)
            # =========================================================================
            print("\n[UnifiedTest] Part 1: Style Training - Epoch 1")

            # 1.A: Train Epoch 1 (Fresh)
            res = bottle.train_style(f"--epochs 1 --kc-epochs 0 {common_args}")
            bottle.assert_style_epochs_trained([1])
            self.assertNotIn("Auto-resume enabled", res.stdout)
            self.assertTrue(os.path.exists(checkpoint_path), "Checkpoint should exist")

            # Verify KC config - KC is always enabled now
            # Config no longer has kc_enabled field

            print("\n[UnifiedTest] Part 1: Style Training - Epoch 2 (Auto-Resume)")
            # 1.B: Train Epoch 2 (Auto-Resume)
            res = bottle.train_style(f"--epochs 2 --kc-epochs 0 {common_args}")
            bottle.assert_style_epochs_trained([1, 2])
            self.assertIn("Auto-resume enabled", res.stdout)

            print("\n[UnifiedTest] Part 1: Style Training - Epoch 3 (Explicit Resume)")
            # 1.C: Train Epoch 3 (Explicit Resume)
            res = bottle.train_style(f"--epochs 3 --resume --kc-epochs 0 {common_args}")
            bottle.assert_style_epochs_trained([1, 2, 3])
            self.assertIn("Resume:         from checkpoint", res.stdout)

            # =========================================================================
            # PART 2: Config Transition & KC Pretrain (From test_kc_pretrain.py)
            # =========================================================================
            print("\n[UnifiedTest] Part 2: KC Pretrain Injection")

            # KC training - KC is always enabled now
            # Just run KC epochs to verify resumption scenario
            res = bottle.train_style(
                f"--kc-epochs 1 --kc-k 256 --epochs 3 {common_args}"
            )
            bottle.assert_kc_epochs_trained([1])
            # Style shouldn't re-run if it's already at epoch 3 and we asked for 3,
            # BUT the KC pretrain might invalidate style checkpoint compatibility if embeddings changed?
            # In this simple case, vocab/embeddings are stable.
            # The test `test_kc_pretrain` used `epochs 1`.

            self.assertTrue(os.path.exists(kc_ckpt), "KC checkpoint should exist")

            # =========================================================================
            # PART 3: KC Resumption (From test_resume_pretrain.py)
            # =========================================================================
            print("\n[UnifiedTest] Part 3: KC Resumption - Epoch 2")

            res = bottle.train_style(f"--resume --kc-epochs 2 --epochs 3 {common_args}")
            bottle.assert_kc_epochs_trained([1, 2])

            # =========================================================================
            # PART 4: Retrain Behavior (From test_auto_resume.py)
            # =========================================================================
            print("\n[UnifiedTest] Part 4: Retrain Style")

            # Force retrain of style (ignore checkpoint).
            # This should wipe the style checkpoint and start over.
            # We'll ask for epochs 1 for speed.
            # Vary percent to 50% and verify it's logged.
            res = bottle.train_style(
                f"--epochs 1 --retrain --percent 50 --kc-epochs 0 {common_args}"
            )
            bottle.assert_style_epochs_trained([1])
            self.assertIn("Retrain:        from scratch", res.stdout)
            self.assertIn("Sampling 50.0% of dataset...", res.stdout)

            # KC should NOT have been touched/retrained since we didn't ask for it
            # KC resumption respects kc-epochs


if __name__ == "__main__":
    unittest.main()
