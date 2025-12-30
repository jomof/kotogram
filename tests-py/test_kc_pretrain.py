import json
import unittest

from training_test_utils import Bottle


class TestKCPretrainRepro(unittest.TestCase):
    def test_repro_crash(self):
        """
        Reproduce the KC pretrain crash by:
        1. Running standard training (generates config.json with kc_enabled=False)
        2. Running with --pretrain-kc (should Override config and succeed)
        """
        with Bottle(self) as bottle:
            # 1. Setup data using standard helper
            bottle.populate_test_data()

            # 2. Standard training (1 epoch)
            # This implicitly runs labeling and generates config.json with kc_enabled=False (default)
            bottle.train_style(
                "--epochs 1 --no-confusion --embed-dim 64 --hidden-dim 128 --num-layers 1 --num-heads 2"
            )

            # Verify config is indeed False
            config_path = bottle.get_file("[models]/style-support/config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.assertFalse(
                config.get("kc_enabled", True),
                "Initial config should have kc_enabled=False",
            )

            # 3. Run WITH --pretrain-kc
            # This was crashing because it loaded the config (kc_enabled=False) and ignored the CLI flag
            # We expect this to SUCCEED now with the fix
            bottle.train_style(
                "--pretrain-kc --kc-epochs 1 --kc-k 256 --epochs 1 --no-confusion --embed-dim 64 --hidden-dim 128 --num-layers 1 --num-heads 2"
            )

            # Verify KC training actually happened
            bottle.assert_kc_epochs_trained([1])


if __name__ == "__main__":
    unittest.main()
