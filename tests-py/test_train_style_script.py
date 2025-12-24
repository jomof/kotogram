import json
import os
import sys
import unittest

# Add tests-py directory to path to allow importing utility module
sys.path.append(os.path.dirname(__file__))
from training_test_utils import Bottle


@unittest.skipIf(os.environ.get("GITHUB_ACTIONS") == "true", "Skipping on GitHub CI")
class TestTrainStyleScript(unittest.TestCase):
    def test_train(self):
        # Common arguments to reduce model size for faster testing
        COMMON_ARGS = "--embed-dim 64 --hidden-dim 128 --num-layers 1 --num-heads 2"

        # Test both regular training and training with MLM pretraining
        test_configs = [
            {"name": "regular", "extra_args": f"{COMMON_ARGS}"},
            {
                "name": "pretrain-mlm",
                "extra_args": f"{COMMON_ARGS} --pretrain-mlm --pretrain-epochs 1",
            },
            {
                "name": "pretrain-kc",
                "extra_args": f"{COMMON_ARGS} --pretrain-kc --kc-epochs 1 --kc-k 256",
            },
        ]

        for config in test_configs:
            with self.subTest(config=config["name"]):
                with Bottle(self) as bottle:
                    bottle.populate_test_data()

                    # Take initial snapshot (also resets profile counters)
                    overrides = {"TRAIN_ROOT": bottle.root_dir, "SKIP_DEPS": "1"}
                    bottle.snapshot("initial")

                    # Step 1: Pre-run labeling to setup cache/data
                    bottle.train_style("--label")

                    # Verify label phase output files using glob patterns
                    EXPECTED_LABEL_MANIFEST = [
                        # --- Source Data ---
                        "[data]/jpn_agrammatic_*.tsv",
                        "[data]/jpn_sentences*.tsv",
                        # --- Generated Cache (Metadata & Vocab) ---
                        "[.cache]/register_samples.csv",
                        "[.cache]/style_dataset/label_metadata.json",
                        "[.cache]/style_dataset/vocab.json",
                        # --- Generated Cache (Databases) ---
                        "[.cache]/agrammatic_combined.tsv",
                        "[.cache]/grammatic_combined.tsv",
                        "[.cache]/kotogram_shards/*.db",
                    ]
                    bottle.assert_dir_layout(EXPECTED_LABEL_MANIFEST)

                    bottle.snapshot("after_label")

                    # Step 2: Run train_style.sh for 1 epoch (with optional pretrain-mlm)
                    # Use --no-label to skip re-running the label phase (metadata already setup in Step 1)
                    # Use --no-confusion to skip generating confusion matrices (saves time)
                    train_args = f"--epochs 1 --no-label --no-confusion {config['extra_args']}".strip()
                    result1 = bottle.train_style(train_args, env_overrides=overrides)

                    # Verify epochs trained
                    # For pretrain-mlm/kc: expect [1, 1] (1 pretrain + 1 fine-tune)
                    # For regular: expect [1] (just 1 fine-tune)
                    expected_epochs = (
                        [1, 1]
                        if "pretrain-mlm" in config["extra_args"]
                        or "pretrain-kc" in config["extra_args"]
                        else [1]
                    )
                    bottle.assertEpochsTrained(result1, expected_epochs)

                    # Verify changes since snapshot (should only be training artifacts)
                    EXPECTED_TRAIN_DIFFERENCES = [
                        # Model Output
                        "[models]/style-support/training.log ADDED",
                        "[models]/style-support/epochs.json ADDED",
                        "[models]/style-support/checkpoint.pt ADDED",
                        "[models]/style-support/checkpoint_optim.pt ADDED",
                        "[models]/style-support/config.json ADDED",
                        "[models]/style-support/tokenizer.json ADDED",
                        "[models]/style/model.pt ADDED",
                        "[models]/style/config.json ADDED",
                        "[models]/style/labels.json ADDED",
                        "[models]/style/model_type.txt ADDED",
                        "[models]/style/tokenizer.json ADDED",
                    ]

                    bottle.assert_dir_diff("after_label", EXPECTED_TRAIN_DIFFERENCES)

                    # Step 3: Take snapshot after first training pass
                    bottle.snapshot("after_epoch_1")

                    # Step 4: Resume training with --epochs 2 (should train only epoch 2)
                    # Use --no-label here as well
                    result2 = bottle.train_style(
                        "--resume --epochs 2 --no-label --no-confusion",
                        env_overrides=overrides,
                    )

                    # Verify only epoch 2 was trained (resume from epoch 1)
                    bottle.assertEpochsTrained(result2, [2])

                    # Verify changes since first training pass
                    EXPECTED_RESUME_DIFFERENCES = [
                        # Training artifacts should be modified
                        "[models]/style-support/training.log MODIFIED",
                        "[models]/style-support/epochs.json MODIFIED",
                        "[models]/style-support/checkpoint.pt MODIFIED",
                        "[models]/style-support/checkpoint_optim.pt MODIFIED",
                        "[models]/style-support/config.json MODIFIED",
                        "[models]/style-support/tokenizer.json MODIFIED",
                        # Model output should be updated
                        "[models]/style/model.pt MODIFIED",
                        "[models]/style/config.json MODIFIED",
                        "[models]/style/labels.json MODIFIED",
                        "[models]/style/model_type.txt MODIFIED",
                        "[models]/style/tokenizer.json MODIFIED",
                    ]

                    bottle.assert_dir_diff("after_epoch_1", EXPECTED_RESUME_DIFFERENCES)

                    # Verify CLI tool works with the newly trained model
                    result = bottle.kotogram_cli("grammar", "こんにちは")

                    # 1. Verify stderr is empty (no reload/path error messages)
                    self.assertEqual(
                        result.stderr.strip(),
                        "",
                        msg=f"CLI stderr should be empty, but got:\n{result.stderr}",
                    )

                    # 2. Verify result is valid JSON
                    data = json.loads(result.stdout)
                    self.assertIn("kotogram", data)
                    self.assertIn("formality", data)

                    # Check for kcs if this is the KC model
                    if "pretrain-kc" in config["extra_args"]:
                        self.assertIn(
                            "kc_top",
                            data,
                            "KC-trained model should output 'kc_top' field",
                        )
                        # Verify KC structure (should be a dict of {str(id): prob})
                        # IMPORTANT: json.loads converts keys to strings!
                        kc_top = data["kc_top"]
                        self.assertIsInstance(kc_top, dict)

                        if len(kc_top) > 0:
                            # Get first key/val
                            k_id_str, prob = next(iter(kc_top.items()))
                            # Key should be convertible to int
                            self.assertTrue(
                                k_id_str.isdigit(), f"Key {k_id_str} should be digits"
                            )
                            self.assertIsInstance(prob, float)
                    else:
                        self.assertNotIn(
                            "kc_top",
                            data,
                            "Non-KC model should usually not output 'kc_top' (unless enabled)",
                        )

                    model_path = bottle.get_file("[models]/style/model.pt")
                    bottle.assertModelIsFp8(model_path)

                    # 4. Verify epochs.json matches training history
                    epochs_path = bottle.resolve_path(
                        "[models]/style-support/epochs.json"
                    )
                    self.assertTrue(os.path.exists(epochs_path), "epochs.json missing")
                    with open(epochs_path, "r") as f:
                        history = json.load(f)

                    # Check epochs count
                    if config["name"] == "regular":
                        # 1 epoch initially, then trained 2nd epoch on resume
                        # Wait, snapshot was taken after epoch 1.
                        # Then we resumed and trained epoch 2.
                        # So history should have 2 entries: epoch 1 and epoch 2
                        self.assertEqual(len(history), 2)
                        self.assertEqual(history[0]["epoch"], 1)
                        self.assertEqual(history[1]["epoch"], 2)
                        self.assertEqual(history[0]["type"], "style")
                        self.assertEqual(history[1]["type"], "style")
                    elif config["name"] == "pretrain-mlm":
                        # 1 MLM epoch + 1 style epoch (initial) + 1 style epoch (resume)
                        # Total 3 entries
                        self.assertEqual(len(history), 3)
                        self.assertEqual(history[0]["type"], "pretrain-mlm")
                        self.assertEqual(history[1]["type"], "style")
                        self.assertEqual(history[2]["type"], "style")
                    elif config["name"] == "pretrain-kc":
                        # 1 KC epoch + 1 style epoch (initial) + 1 style epoch (resume)
                        self.assertEqual(len(history), 3)
                        self.assertEqual(history[0]["type"], "pretrain-kc")
                        self.assertEqual(history[1]["type"], "style")
                        self.assertEqual(history[2]["type"], "style")

    def test_auto_resume(self):
        """Verifies auto-resume affects *training*, not just printing."""
        COMMON_ARGS = "--embed-dim 64 --hidden-dim 128 --num-layers 1 --num-heads 2"

        with Bottle(self) as bottle:
            bottle.populate_test_data()

            # 0) Prepare cache/vocab once
            bottle.train_style("--label")

            checkpoint_path = bottle.resolve_path(
                "[models]/style-support/checkpoint.pt"
            )

            # Case A: No checkpoint, no flags => train 1 epoch only
            result = bottle.train_style(
                f"--epochs 1 --no-label --no-confusion {COMMON_ARGS}"
            )
            bottle.assertEpochsTrained(result, [1])
            self.assertNotIn("Auto-resume enabled", result.stdout)

            # Case B: Checkpoint exists (epoch 1), no flags => SHOULD auto-resume to epoch 2
            self.assertTrue(
                os.path.exists(checkpoint_path), "Expected checkpoint after training"
            )
            result = bottle.train_style(
                f"--epochs 2 --no-label --no-confusion {COMMON_ARGS}"
            )
            # If auto-resume works, it sees epoch 1 done, trains epoch 2.
            bottle.assertEpochsTrained(result, [2])
            self.assertIn("Auto-resume enabled", result.stdout)

            # Case C: Checkpoint exists, --retrain => should NOT auto-resume; trains [1,2] from scratch
            result = bottle.train_style(
                f"--epochs 2 --no-label --no-confusion {COMMON_ARGS} --retrain"
            )
            bottle.assertEpochsTrained(result, [1, 2])
            self.assertNotIn("Auto-resume enabled", result.stdout)
            self.assertIn("Retrain:        from scratch", result.stdout)

            # Case D: Checkpoint exists (epoch 2 now), explicit --resume => trains [3] if we ask for 3
            # We must increase epochs to verify resume works from the new state
            result = bottle.train_style(
                f"--epochs 3 --no-label --no-confusion {COMMON_ARGS} --resume"
            )
            bottle.assertEpochsTrained(result, [3])
            self.assertNotIn("Auto-resume enabled", result.stdout)
            self.assertIn("Resume:         from checkpoint", result.stdout)


if __name__ == "__main__":
    unittest.main()
