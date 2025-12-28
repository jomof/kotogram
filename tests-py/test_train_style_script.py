import json
import os
import shutil
import unittest

from training_test_utils import Bottle


@unittest.skipIf(os.environ.get("GITHUB_ACTIONS") == "true", "Skipping on GitHub CI")
class TestTrainStyleScript(unittest.TestCase):
    # pylint: disable=too-many-locals

    def test_train(self):
        # Common arguments to reduce model size for faster testing
        common_args = "--embed-dim 64 --hidden-dim 128 --num-layers 1 --num-heads 2"

        # Test both regular training and pretraining (KC)
        test_configs = [
            {"name": "regular", "extra_args": f"{common_args}"},
            {
                "name": "pretrain-kc",
                "extra_args": f"{common_args} --pretrain-kc --kc-epochs 1 --kc-k 256",
            },
        ]

        for config in test_configs:
            with self.subTest(config=config["name"]):
                # Force profiling on for integration test logic so we can verify profile artifacts
                env = {"TRAIN_PROFILE": "1"}
                with Bottle(self, env=env) as bottle:
                    bottle.populate_test_data()

                    # Take initial snapshot
                    bottle.snapshot("initial")

                    # Step 1: Pre-run labeling to setup cache/data
                    bottle.train_style("--label")
                    shutil.rmtree(bottle.resolve_path("[data]"))

                    # Verify label phase output files using glob patterns
                    expected_label_manifest = [
                        "[.cache]/style_dataset/vocab.json",
                        "[.cache]/style_dataset/sentences.txt",
                        "[.cache]/style_dataset/kotograms.txt",
                        "[.cache]/style_dataset/kc_*.bin",
                        "[.cache]/style_dataset/feat_*.bin",
                        "[.cache]/style_dataset/labels.bin_*",
                        "[.cache]/style_dataset/offsets.bin",
                        "[models]/style-support/config.json",
                        "[models]/style/tokenizer.json",
                    ]
                    bottle.assert_dir_layout(expected_label_manifest)

                    bottle.snapshot("after_label")

                    # Step 2: Run train_style.sh for 1 epoch (with optional pretraining)
                    # Use --no-confusion to skip generating confusion matrices (saves time)
                    train_args = (
                        f"--epochs 1 --no-confusion {config['extra_args']}".strip()
                    )
                    bottle.train_style(train_args)

                    # Verify epochs trained
                    # For pretrain-kc: expect [1, 1] (1 pretrain + 1 fine-tune)
                    # For regular: expect [1] (just 1 fine-tune)
                    if "pretrain-kc" in config["name"]:
                        # Expect 1 KC event and 1 Style event
                        bottle.assert_kc_epochs_trained([1])
                        bottle.assert_style_epochs_trained([1])
                    else:
                        bottle.assert_style_epochs_trained([1])

                    # Verify changes since snapshot (should only be training artifacts)
                    # Use assert_files_exist to ensure key artifacts are present
                    bottle.assert_files_exist(["[models]/style-support/training.log"])

                    expected_train_differences = [
                        "[models]/style-support/training.log ADDED",
                        "[models]/style-support/training-history.tsv ADDED",
                        "[models]/style-support/checkpoint.pt ADDED",
                        "[models]/style-support/checkpoint_meta.pt ADDED",
                        "[.profile]/* ADDED",
                        "[models]/style-support/config.json MODIFIED",
                        "[models]/style/__init__.py ADDED",
                        "[models]/style/model.pt ADDED",
                        "[models]/style/model.json ADDED",
                        "[models]/style/labels.json ADDED",
                        "[models]/style/model_type.txt ADDED",
                    ]

                    if "pretrain-kc" in config["extra_args"]:
                        expected_train_differences.append(
                            "[models]/style-support/checkpoint_kc.pt ADDED"
                        )

                    bottle.assert_dir_diff("after_label", expected_train_differences)

                    # Step 3: Take snapshot after first training pass
                    bottle.snapshot("after_epoch_1")

                    # Step 4: Resume training with --epochs 2 (should train only epoch 2)
                    bottle.train_style("--resume --epochs 2 --no-confusion")

                    # Verify only epoch 2 was trained (resume from epoch 1)
                    # Verify only epoch 2 was trained (resume from epoch 1)
                    if "pretrain-kc" in config["name"]:
                        # Resume should just add Style epoch 2. KC history remains [1].
                        bottle.assert_kc_epochs_trained([1])
                        bottle.assert_style_epochs_trained([1, 2])
                    else:
                        bottle.assert_style_epochs_trained([1, 2])

                    # Verify changes since first training pass
                    expected_resume_differences = [
                        # Training artifacts should be modified
                        "[models]/style-support/training.log MODIFIED",
                        "[models]/style-support/training-history.tsv MODIFIED",
                        "[models]/style-support/checkpoint.pt MODIFIED",
                        "[models]/style-support/checkpoint_meta.pt MODIFIED",
                        "[.profile]/* ADDED",
                        "[.profile]/training-profile.txt MODIFIED",
                        "[models]/style-support/config.json MODIFIED",
                        "[models]/style/model.pt MAYBE-MODIFIED",
                    ]

                    # NOTE: checkpoint_kc.pt should NOT be modified
                    # because we are resuming and pretraining is already complete for this test case.

                    bottle.assert_dir_diff("after_epoch_1", expected_resume_differences)

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
                    bottle.assert_model_is_fp8(model_path)

                    # 4. Verify history in epochs.json
                    history = bottle.get_epoch_history()
                    self.assertTrue(len(history) > 0, "epochs.json should not be empty")

                    # Calculate expected counts based on actual data in bottle
                    expected_counts = bottle.calculate_expected_counts()

                    if config["name"] == "regular":
                        self.assertEqual(len(history), 2)
                        self.assertEqual(history[0].get_type_name(), "STYLE_EPOCH")
                        # Style training uses full train split (labeled_train)
                        self.assertEqual(
                            history[0].metrics["sentence_count"],
                            expected_counts["total_train_split_size"],
                        )
                    elif config["name"] == "pretrain-kc":
                        self.assertGreaterEqual(len(history), 3)
                        self.assertEqual(history[0].get_type_name(), "KC_EPOCH")

                        # Verify counts
                        # KC pretraining uses ONLY grammatical sentences from the training split,
                        # so it must be strictly less than the total style training set (which includes agrammatic).
                        self.assertLess(
                            history[0].metrics["sentence_count"],
                            expected_counts["total_train_split_size"],
                        )

                        # KC metrics check
                        self.assertIn("avg_struct_loss", history[0].metrics)

                    # All remaining entries are always standard style fine-tuning
                    # Remaining entries are style fine-tuning
                    start_idx = 1
                    if config["name"] == "pretrain-kc":
                        start_idx = 1

                    for i in range(start_idx, len(history)):
                        self.assertEqual(history[i].get_type_name(), "STYLE_EPOCH")
                        # Style fine-tuning uses the full training split (gram + agram)
                        self.assertEqual(
                            history[i].metrics["sentence_count"],
                            expected_counts["total_train_split_size"],
                        )

                    # Verify performance profile coherence (clean jsonl, present txt)
                    bottle.assert_coherent_performance_profile()


if __name__ == "__main__":
    unittest.main()
