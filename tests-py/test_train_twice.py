"""Test that running train_style twice produces expected differences."""

import os
import sys
import unittest

# Add tests-py directory to path to allow importing utility module
sys.path.append(os.path.dirname(__file__))
from training_test_utils import Bottle


@unittest.skipIf(os.environ.get("GITHUB_ACTIONS") == "true", "Skipping on GitHub CI")
class TestTrainTwice(unittest.TestCase):
    def test_train_twice(self):
        """Run train_style, snapshot, run again, assert diff."""
        COMMON_ARGS = "--embed-dim 64 --hidden-dim 128 --num-layers 1 --num-heads 2"

        with Bottle(self) as bottle:
            bottle.populate_test_data()

            # First training run (1 epoch)
            bottle.train_style(f"--epochs 1 --no-confusion {COMMON_ARGS}")

            # Snapshot after first training
            bottle.snapshot("after_first_train")

            # Second training run (1 more epoch)
            bottle.train_style(f"--epochs 1 --no-confusion {COMMON_ARGS}")

            # Assert differences between first and second run
            # With hash-based comparison, most files should be identical if no new work done.
            # config.json should be identical too because we fixed resume_from.
            EXPECTED_DIFFERENCES = [
                # Support files: logs always grow, even if training skips
                "[models]/style-support/training.log MODIFIED",
            ]
            bottle.assert_dir_diff("after_first_train", EXPECTED_DIFFERENCES)


if __name__ == "__main__":
    unittest.main()
