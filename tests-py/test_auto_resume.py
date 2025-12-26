import os
import unittest

from training_test_utils import Bottle


@unittest.skipIf(os.environ.get("GITHUB_ACTIONS") == "true", "Skipping on GitHub CI")
class TestAutoResume(unittest.TestCase):
    def test_auto_resume(self):
        """Verifies auto-resume affects *training*, not just printing."""
        common_args = "--embed-dim 64 --hidden-dim 128 --num-layers 1 --num-heads 2"

        with Bottle(self) as bottle:
            bottle.populate_test_data()

            # 0) Prepare cache/vocab once
            bottle.train_style("--label", timeout=10)

            checkpoint_path = bottle.resolve_path(
                "[models]/style-support/checkpoint.pt"
            )

            # Case A: No checkpoint, no flags => train 1 epoch only
            result = bottle.train_style(
                f"--epochs 1 --no-confusion {common_args}", timeout=10
            )
            bottle.assertEpochsTrained(result, [1])
            self.assertNotIn("Auto-resume enabled", result.stdout)

            # Case B: Checkpoint exists (epoch 1), no flags => SHOULD auto-resume to epoch 2
            self.assertTrue(
                os.path.exists(checkpoint_path), "Expected checkpoint after training"
            )
            result = bottle.train_style(
                f"--epochs 2 --no-confusion {common_args}", timeout=10
            )
            # If auto-resume works, it sees epoch 1 done, trains epoch 2.
            bottle.assertEpochsTrained(result, [2])
            self.assertIn("Auto-resume enabled", result.stdout)

            # Case C: Checkpoint exists, --retrain => should NOT auto-resume; trains [1,2] from scratch
            result = bottle.train_style(
                f"--epochs 2 --no-confusion {common_args} --retrain", timeout=10
            )
            bottle.assertEpochsTrained(result, [1, 2])
            self.assertNotIn("Auto-resume enabled", result.stdout)
            self.assertIn("Retrain:        from scratch", result.stdout)

            # Case D: Checkpoint exists (epoch 2 now), explicit --resume => trains [3] if we ask for 3
            # We must increase epochs to verify resume works from the new state
            result = bottle.train_style(
                f"--epochs 3 --no-confusion {common_args} --resume", timeout=10
            )
            bottle.assertEpochsTrained(result, [3])
            self.assertNotIn("Auto-resume enabled", result.stdout)
            self.assertIn("Resume:         from checkpoint", result.stdout)


if __name__ == "__main__":
    unittest.main()
