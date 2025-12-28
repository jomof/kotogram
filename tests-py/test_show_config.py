import json
import os
import unittest

from training_test_utils import Bottle


@unittest.skipIf(os.environ.get("GITHUB_ACTIONS") == "true", "Skipping on GitHub CI")
class TestShowConfig(unittest.TestCase):
    def test_show_config(self):
        """Verify --show-config output matches default TrainerConfig."""
        from train.config import TrainerConfig

        with Bottle(self) as bottle:
            result = bottle.train_style("--show-config")
            # Should only contain the JSON, no other output
            output_config = json.loads(result.stdout)

        # Instantiate a default TrainerConfig
        default_config = TrainerConfig()

        # Compare the two
        self.assertEqual(output_config, default_config.to_dict())


if __name__ == "__main__":
    unittest.main()
