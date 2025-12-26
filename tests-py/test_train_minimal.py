"""Minimal test to verify train_style execution and performance logs."""

import os
import sys
import unittest

# Add tests-py directory to path to allow importing utility module
sys.path.append(os.path.dirname(__file__))
from training_test_utils import Bottle


class TestTrainMinimal(unittest.TestCase):
    def test_train_one_epoch(self):
        """Run train_style for 1 epoch only."""
        COMMON_ARGS = "--embed-dim 64 --hidden-dim 128 --num-layers 1 --num-heads 2 --batch-size 4"

        with Bottle(self) as bottle:
            bottle.populate_test_data()

            # Single training run (1 epoch)
            print("\n>>> Running minimal 1-epoch training...")
            env = {"TARGET_GLOBAL_BATCH_SIZE": "4"}
            bottle.train_style(
                f"--epochs 1 --no-confusion {COMMON_ARGS}", env_overrides=env
            )
            print(">>> Minimal training complete.")


if __name__ == "__main__":
    unittest.main()
