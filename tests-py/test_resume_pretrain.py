import os
import unittest

from training_test_utils import Bottle


@unittest.skipIf(os.environ.get("GITHUB_ACTIONS") == "true", "Skipping on GitHub CI")
class TestResumePretrain(unittest.TestCase):
    def test_resume_kc(self):
        """Verify KC pretraining can be resumed."""
        common_args = "--embed-dim 64 --hidden-dim 128 --num-layers 1 --num-heads 2"

        with Bottle(self) as bottle:
            bottle.populate_test_data()
            bottle.train_style("--label")

            # Step 1: Run KC for 1 epoch
            result1 = bottle.train_style(
                f"--pretrain-kc --kc-epochs 1 --epochs 0 --no-confusion {common_args}",
            )
            bottle.assertEpochsTrained(result1, [1])  # 1 KC, 0 Style

            # Verify KC checkpoint exists
            kc_ckpt = bottle.resolve_path("[models]/style-support/checkpoint_kc.pt")
            self.assertTrue(
                os.path.exists(kc_ckpt), f"KC checkpoint not found at {kc_ckpt}"
            )

            # Step 2: Resume KC to 2 epochs
            result2 = bottle.train_style(
                f"--resume --pretrain-kc --kc-epochs 2 --epochs 0 --no-confusion {common_args}",
            )
            # Should only train the 2nd KC epoch
            bottle.assertEpochsTrained(result2, [2])

            # Verify history
            history = bottle.get_epoch_history()
            kc_entries = [e for e in history if e["type"] == "pretrain-kc"]
            self.assertEqual(len(kc_entries), 2)
            self.assertEqual(kc_entries[0]["epoch"], 1)
            self.assertEqual(kc_entries[1]["epoch"], 2)


if __name__ == "__main__":
    unittest.main()
