import json
import os
import shutil
import unittest

import torch
from training_test_utils import Bottle


@unittest.skipIf(os.environ.get("GITHUB_ACTIONS") == "true", "Skipping on GitHub CI")
class TestTrainStyleScript(unittest.TestCase):
    # pylint: disable=too-many-locals

    def test_train(self):
        # Common arguments to reduce model size for faster testing
        common_args = "--embed-dim 64 --hidden-dim 128 --num-layers 1 --num-heads 2 --batch-size 128"
        # KC is always enabled now; test with kc-epochs 1 for speed
        test_configs = [
            {
                "name": "kc-training",
                "extra_args": f"{common_args} --kc-epochs 1",
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
                    # Calculate expected counts before DB is deleted
                    expected_counts = bottle.calculate_expected_counts()

                    label_args = "--label --force-relabel"
                    bottle.train_style(label_args)

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
                    train_args = f"--epochs 1 {config['extra_args']}".strip()
                    result = bottle.train_style(train_args)

                    if "kc-training" in config["name"]:
                        # Verify diagnostics event in history
                        history = bottle.get_epoch_history()
                        diag_events = [
                            e for e in history if e.get_type_name() == "KC_DIAG"
                        ]
                        self.assertTrue(
                            len(diag_events) > 0,
                            "KC diagnostics event missing from history",
                        )
                        # Basic validation of content
                        stats = diag_events[0].stats
                        self.assertIn("families", stats)

                    # Verify epochs trained
                    # For pretrain-kc: expect [1, 1] (1 pretrain + 1 fine-tune)
                    # For regular: expect [1] (just 1 fine-tune)
                    if "kc-training" in config["name"]:
                        # Expect 1 KC event and 1 Style event
                        bottle.assert_kc_epochs_trained([1])
                        bottle.assert_style_epochs_trained([1])
                    else:
                        bottle.assert_style_epochs_trained([1])

                    # Verify no NaNs in history
                    bottle.assert_no_nans_in_history()

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
                        "[models]/style-support/*confusion.csv ADDED",
                        "[models]/style-support/confusion_matrices/*.tsv ADDED",
                    ]

                    if "kc-training" in config["name"]:
                        expected_train_differences.append(
                            "[models]/style-support/checkpoint_kc.pt ADDED"
                        )

                    bottle.assert_dir_diff("after_label", expected_train_differences)

                    # Step 3: Take snapshot after first training pass
                    bottle.snapshot("after_epoch_1")

                    # Step 4: Resume training with --epochs 2 (should train only epoch 2)
                    bottle.train_style("--resume --epochs 2")

                    # Verify only epoch 2 was trained (resume from epoch 1)
                    if "kc-training" in config["name"]:
                        # Resume should just add Style epoch 2. KC history remains [1].
                        bottle.assert_kc_epochs_trained([1])
                        bottle.assert_style_epochs_trained([1, 2])
                    else:
                        bottle.assert_style_epochs_trained([1, 2])

                    # Verify no NaNs in history after resume
                    bottle.assert_no_nans_in_history()

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
                        "[models]/style-support/*confusion.csv MAYBE-MODIFIED",
                        "[models]/style-support/confusion_matrices/*.tsv MAYBE-MODIFIED",
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
                    # 1b. Verify no warnings in stdout (e.g. from model loading)
                    self.assertNotIn(
                        "WARNING:",
                        result.stdout,
                        msg=f"CLI stdout contains warnings:\n{result.stdout}",
                    )

                    # 2. Verify result is valid JSON
                    data = json.loads(result.stdout)
                    self.assertIn("kotogram", data)
                    self.assertIn("formality", data)

                    self.assertIn(
                        "kc_top",
                        data,
                        "KC-enabled model (ubiquitous) should always output 'kc_top' field",
                    )
                    # Verify KC structure (should be a dict of {str(id): prob})
                    # IMPORTANT: json.loads converts keys to strings!
                    kc_top = data["kc_top"]
                    self.assertIsInstance(kc_top, dict)

                    self.assertGreater(
                        len(kc_top),
                        0,
                        "KC predictions should be present for ubiquitous KC model",
                    )
                    # Get first key/val
                    k_id_str, prob = next(iter(kc_top.items()))
                    # Key should be convertible to int
                    self.assertTrue(
                        k_id_str.isdigit(), f"Key {k_id_str} should be digits"
                    )
                    self.assertIsInstance(prob, float)

                    # --- New Model Verification Logic ---
                    style_model = bottle.get_file("[models]/style/model.pt")
                    support_checkpoint = bottle.get_file(
                        "[models]/style-support/checkpoint.pt"
                    )
                    style_config_path = bottle.get_file("[models]/style/model.json")

                    # 1. Assert FP8 for export
                    bottle.assert_model_is_fp8(style_model)

                    # 2. Assert Slim vs Full Content using torch.load
                    # Map location CPU to avoid CUDA errors if on CPU only machine
                    slim_state = torch.load(
                        style_model, map_location="cpu", weights_only=True
                    )
                    full_state = torch.load(
                        support_checkpoint, map_location="cpu", weights_only=False
                    )

                    # Full state should contain kc_decoders
                    # full_state is dict with "model_state_dict"
                    self.assertTrue(
                        any(
                            k.startswith("kc_decoders.")
                            for k in full_state["model_state_dict"]
                        ),
                        "Support checkpoint MUST contain kc_decoders (Full State)",
                    )

                    # Slim state should NOT contain kc_decoders
                    # slim_state is the state dict directly
                    self.assertFalse(
                        any(k.startswith("kc_decoders.") for k in slim_state),
                        "Exported model MUST NOT contain kc_decoders (Slim State)",
                    )

                    # 3. Assert Config Slimming
                    with open(style_config_path, "r", encoding="utf-8") as f:
                        style_config = json.load(f)

                    self.assertNotIn(
                        "kc_target_specs",
                        style_config,
                        "Exported model.json should NOT have kc_target_specs (Slim Config)",
                    )

                    # 4. Assert Physical Size
                    # Calculate expected size based on the ACTUAL slim config

                    from train.pytorch_utils import (
                        calculate_detailed_size,
                        verify_model_size_policy,
                    )

                    expected_breakdown = calculate_detailed_size(style_config)
                    expected_size = sum(expected_breakdown.values())
                    actual_size = os.path.getsize(style_model)

                    def lazy_load_state_dict(path=style_model):
                        return torch.load(path, map_location="cpu", weights_only=True)

                    verify_model_size_policy(
                        actual_size,
                        expected_size,
                        expected_breakdown,
                        lazy_load_state_dict,
                    )

                    # 4. Verify history in epochs.json
                    history = bottle.get_epoch_history()
                    self.assertTrue(len(history) > 0, "epochs.json should not be empty")

                    # Calculate expected counts based on actual data in bottle
                    # (Taken before data/corpus.db is deleted)
                    # expected_counts = bottle.calculate_expected_counts()

                    # Filter to just epoch events for sequence verification
                    epoch_events = [
                        e
                        for e in history
                        if e.get_type_name() in ["KC_EPOCH", "STYLE_EPOCH"]
                    ]

                    if config["name"] == "regular":
                        self.assertEqual(len(epoch_events), 2)
                        self.assertEqual(epoch_events[0].get_type_name(), "STYLE_EPOCH")
                        # Style training uses full train split (labeled_train)
                        self.assertEqual(
                            epoch_events[0].metrics["sentence_count"],
                            expected_counts["total_train_split_size"],
                        )
                    elif config["name"] == "kc-training":
                        self.assertGreaterEqual(len(epoch_events), 2)
                        self.assertEqual(epoch_events[0].get_type_name(), "KC_EPOCH")

                        # Verify counts
                        # KC pretraining now uses ALL grammatical sentences (full dataset, not split),
                        # so it should equal the total grammatical count.
                        # Note: epoch_events[0] is the KC_EPOCH
                        self.assertEqual(
                            epoch_events[0].metrics["sentence_count"],
                            expected_counts["total_grammatic_sentences"],
                        )

                        # KC metrics check
                        self.assertIn("avg_struct_loss", epoch_events[0].metrics)

                    # Remaining entries are style fine-tuning
                    # We start checking from the first style epoch
                    start_idx = 1 if config["name"] == "kc-training" else 0

                    # Just verify remaining are style epochs
                    for i in range(start_idx, len(epoch_events)):
                        self.assertEqual(epoch_events[i].get_type_name(), "STYLE_EPOCH")

                        # Style fine-tuning uses the full training split (gram + agram)
                        self.assertEqual(
                            epoch_events[i].metrics["sentence_count"],
                            expected_counts["total_train_split_size"],
                        )

                    # Verify performance profile coherence (clean jsonl, present txt)
                    bottle.assert_coherent_performance_profile()


if __name__ == "__main__":
    unittest.main()
