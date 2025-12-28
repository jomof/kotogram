"""Test that running train_style twice produces expected differences."""

import os
import unittest

from training_test_utils import Bottle


@unittest.skipIf(os.environ.get("GITHUB_ACTIONS") == "true", "Skipping on GitHub CI")
class TestTrainTwice(unittest.TestCase):
    def test_train_twice(self):
        """Run train_style, snapshot, run again, assert diff."""
        common_args = "--embed-dim 64 --hidden-dim 128 --num-layers 1 --num-heads 2"

        with Bottle(self) as bottle:
            bottle.populate_test_data()

            # First training run (1 epoch)
            bottle.train_style(f"--epochs 1 --no-confusion {common_args}")

            # Snapshot after first training
            bottle.snapshot("after_first_train")

            # Second training run (1 more epoch)
            bottle.train_style(f"--epochs 1 --no-confusion {common_args}")

            # Assert differences between first and second run
            # With hash-based comparison, most files should be identical if no new work done.
            # config.json should be identical too because we fixed resume_from.
            expected_differences = [
                # Support files: logs always grow, even if training skips
                "[models]/style-support/training.log MODIFIED",
            ]
            bottle.assert_dir_diff("after_first_train", expected_differences)


if __name__ == "__main__":
    unittest.main()
