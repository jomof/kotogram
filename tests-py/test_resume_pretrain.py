import os
import sys
import unittest

# Add tests-py directory to path to allow importing utility module
sys.path.append(os.path.dirname(__file__))
from training_test_utils import Bottle


@unittest.skipIf(os.environ.get("GITHUB_ACTIONS") == "true", "Skipping on GitHub CI")
class TestResumePretrain(unittest.TestCase):
    def test_resume_mlm(self):
        """Verify MLM pretraining can be resumed."""
        COMMON_ARGS = "--embed-dim 64 --hidden-dim 128 --num-layers 1 --num-heads 2"

        with Bottle(self) as bottle:
            bottle.populate_test_data()
            bottle.train_style("--label")

            # Step 1: Run MLM for 1 epoch
            result1 = bottle.train_style(
                f"--pretrain-mlm --pretrain-epochs 1 --epochs 0 --no-confusion {COMMON_ARGS}",
            )
            bottle.assertEpochsTrained(result1, [1])  # 1 MLM, 0 Style

            # Verify MLM checkpoint exists
            mlm_ckpt = bottle.resolve_path("[models]/style-support/checkpoint_mlm.pt")
            self.assertTrue(
                os.path.exists(mlm_ckpt), f"MLM checkpoint not found at {mlm_ckpt}"
            )

            # Step 2: Resume MLM to 2 epochs
            result2 = bottle.train_style(
                f"--resume --pretrain-mlm --pretrain-epochs 2 --epochs 0 --no-confusion {COMMON_ARGS}",
            )
            # Should only train the 2nd MLM epoch
            bottle.assertEpochsTrained(result2, [2])

            # Verify history
            history = bottle.get_epoch_history()
            mlm_entries = [e for e in history if e["type"] == "pretrain-mlm"]
            self.assertEqual(
                len(mlm_entries),
                2,
                f"Expected 2 MLM history entries, got {len(mlm_entries)}",
            )
            self.assertEqual(mlm_entries[0]["epoch"], 1)
            self.assertEqual(mlm_entries[1]["epoch"], 2)

    def test_resume_kc(self):
        """Verify KC pretraining can be resumed."""
        COMMON_ARGS = "--embed-dim 64 --hidden-dim 128 --num-layers 1 --num-heads 2"

        with Bottle(self) as bottle:
            bottle.populate_test_data()
            bottle.train_style("--label")

            # Step 1: Run KC for 1 epoch
            result1 = bottle.train_style(
                f"--pretrain-kc --kc-epochs 1 --epochs 0 --no-confusion {COMMON_ARGS}",
            )
            bottle.assertEpochsTrained(result1, [1])  # 1 KC, 0 Style

            # Verify KC checkpoint exists
            kc_ckpt = bottle.resolve_path("[models]/style-support/checkpoint_kc.pt")
            self.assertTrue(
                os.path.exists(kc_ckpt), f"KC checkpoint not found at {kc_ckpt}"
            )

            # Step 2: Resume KC to 2 epochs
            result2 = bottle.train_style(
                f"--resume --pretrain-kc --kc-epochs 2 --epochs 0 --no-confusion {COMMON_ARGS}",
            )
            # Should only train the 2nd KC epoch
            bottle.assertEpochsTrained(result2, [2])

            # Verify history
            history = bottle.get_epoch_history()
            kc_entries = [e for e in history if e["type"] == "pretrain-kc"]
            self.assertEqual(len(kc_entries), 2)
            self.assertEqual(kc_entries[0]["epoch"], 1)
            self.assertEqual(kc_entries[1]["epoch"], 2)

    def test_resume_combined(self):
        """Verify resumption when BOTH MLM and KC are active and epochs are increased."""
        COMMON_ARGS = "--embed-dim 64 --hidden-dim 128 --num-layers 1 --num-heads 2"

        with Bottle(self) as bottle:
            bottle.populate_test_data()
            bottle.train_style("--label")

            # Step 1: Run Combined Pretraining for 1 epoch each
            # --epochs 0 to skip style fine-tuning for speed
            result1 = bottle.train_style(
                f"--pretrain-mlm --pretrain-epochs 1 --pretrain-kc --kc-epochs 1 --kc-k 256 --epochs 0 --no-confusion {COMMON_ARGS}"
            )
            bottle.assertEpochsTrained(result1, [1, 1])  # MLM(1), KC(1)

            # Step 2: Resume BOTH to 2 epochs
            result2 = bottle.train_style(
                f"--resume --pretrain-mlm --pretrain-epochs 2 --pretrain-kc --kc-epochs 2 --kc-k 256 --epochs 0 --no-confusion {COMMON_ARGS}"
            )
            # Should train MLM epoch 2, then KC epoch 2
            bottle.assertEpochsTrained(result2, [2, 2])

            # Verify history
            history = bottle.get_epoch_history()

            mlm_entries = [e for e in history if e["type"] == "pretrain-mlm"]
            self.assertEqual(len(mlm_entries), 2, "Should have 2 MLM entries")
            self.assertEqual(mlm_entries[1]["epoch"], 2)

            kc_entries = [e for e in history if e["type"] == "pretrain-kc"]
            self.assertEqual(len(kc_entries), 2, "Should have 2 KC entries")
            self.assertEqual(kc_entries[1]["epoch"], 2)



if __name__ == "__main__":
    unittest.main()
