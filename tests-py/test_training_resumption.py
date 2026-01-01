import json
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
        common_args = "--embed-dim 64 --hidden-dim 128 --num-layers 1 --num-heads 2 --no-confusion"

        with Bottle(self) as bottle:
            bottle.populate_test_data()
            bottle.train_style("--label")

            # Paths
            checkpoint_path = bottle.resolve_path(
                "[models]/style-support/checkpoint.pt"
            )
            config_path = bottle.get_file("[models]/style-support/config.json")
            kc_ckpt = bottle.resolve_path("[models]/style-support/checkpoint_kc.pt")

            # =========================================================================
            # PART 1: Style Auto-Resume (From test_auto_resume.py)
            # =========================================================================
            print("\n[UnifiedTest] Part 1: Style Training - Epoch 1")

            # 1.A: Train Epoch 1 (Fresh)
            res = bottle.train_style(f"--epochs 1 {common_args}")
            bottle.assert_style_epochs_trained([1])
            self.assertNotIn("Auto-resume enabled", res.stdout)
            self.assertTrue(os.path.exists(checkpoint_path), "Checkpoint should exist")

            # Verify KC config is False (default) - From test_kc_pretrain.py logic
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.assertFalse(
                config.get("kc_enabled", True),
                "KC should be disabled after style-only train",
            )

            print("\n[UnifiedTest] Part 1: Style Training - Epoch 2 (Auto-Resume)")
            # 1.B: Train Epoch 2 (Auto-Resume)
            res = bottle.train_style(f"--epochs 2 {common_args}")
            bottle.assert_style_epochs_trained([1, 2])
            self.assertIn("Auto-resume enabled", res.stdout)

            print("\n[UnifiedTest] Part 1: Style Training - Epoch 3 (Explicit Resume)")
            # 1.C: Train Epoch 3 (Explicit Resume)
            res = bottle.train_style(f"--epochs 3 --resume {common_args}")
            bottle.assert_style_epochs_trained([1, 2, 3])
            self.assertIn("Resume:         from checkpoint", res.stdout)

            # =========================================================================
            # PART 2: Config Transition & KC Pretrain (From test_kc_pretrain.py)
            # =========================================================================
            print("\n[UnifiedTest] Part 2: KC Pretrain Injection")

            # Run with --pretrain-kc. Should override the valid config.json which has kc_enabled=False
            # We assume it starts KC from scratch (epoch 1)
            # Note: We keep --epochs 3 for style to ensure it doesn't try to retrain style unless needed
            # Actually, standard behavior is: pretrain KC *then* train style.
            # If we say --pretrain-kc --kc-epochs 1, it should run KC epoch 1.
            # If --epochs is met (3), it might skip style training or run if checkpoint is valid.
            # Let's target KC training specifically.

            res = bottle.train_style(
                f"--pretrain-kc --kc-epochs 1 --kc-k 256 --epochs 3 {common_args}"
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

            # Resume KC to 2 epochs
            res = bottle.train_style(
                f"--resume --pretrain-kc --kc-epochs 2 --epochs 3 {common_args}"
            )
            bottle.assert_kc_epochs_trained([1, 2])

            # =========================================================================
            # PART 4: Retrain Behavior (From test_auto_resume.py)
            # =========================================================================
            print("\n[UnifiedTest] Part 4: Retrain Style")

            # Force retrain of style (ignore checkpoint).
            # This should wipe the style checkpoint and start over.
            # We'll ask for epochs 1 for speed.
            # Vary percent to 50% and verify it's logged.
            res = bottle.train_style(f"--epochs 1 --retrain --percent 50 {common_args}")
            bottle.assert_style_epochs_trained([1])
            self.assertIn("Retrain:        from scratch", res.stdout)
            self.assertIn("Sampling 50.0% of dataset...", res.stdout)

            # KC should NOT have been touched/retrained unless we asked for it?
            # --retrain normally applies to the main loop (style).
            # If we didn't pass --pretrain-kc, KC is skipped.


if __name__ == "__main__":
    unittest.main()
