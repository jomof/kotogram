"""Minimal test to verify train_style execution and performance logs."""

import unittest

from training_test_utils import Bottle


class TestTrainMinimal(unittest.TestCase):
    def test_train_one_epoch(self):
        """Run train_style for 1 epoch only."""
        common_args = "--embed-dim 64 --hidden-dim 128 --num-layers 1 --num-heads 2 --batch-size 4"

        env = {"TARGET_GLOBAL_BATCH_SIZE": "4"}
        with Bottle(self, env=env) as bottle:
            bottle.populate_test_data()

            # Single training run (1 epoch)
            print("\n>>> Running minimal 1-epoch training...")
            bottle.train_style(f"--epochs 1 --no-confusion {common_args}")
            print(">>> Minimal training complete.")


if __name__ == "__main__":
    unittest.main()
