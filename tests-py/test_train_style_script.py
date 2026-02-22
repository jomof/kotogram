import json
import os
import shutil
import unittest

import torch
from training_test_utils import Bottle


@unittest.skipIf(os.environ.get("GITHUB_ACTIONS") == "true", "Skipping on GitHub CI")
class TestTrainStyleScript(unittest.TestCase):
    # pylint: disable=too-many-locals,too-many-statements

    def test_train(self):
        # Common arguments to reduce model size for faster testing
        # KC is always enabled; use 2 KC epochs with skip_first_metrics=1
        # so epoch 0 diagnostics are skipped but epoch 1 generates them
        train_style_args = (
            "--embed-dim 64 --hidden-dim 128 --num-layers 1 --num-heads 2 "
            "--batch-size 128 --kc-epochs 2 --kc-skip-first-metrics 1"
        )

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
                "[.cache]/style_dataset/kc_unique_counts.json",
                "[.cache]/style_dataset/feat_*.bin",
                "[.cache]/style_dataset/labels.bin_*",
                "[.cache]/style_dataset/offsets.bin",
                "[.cache]/style_dataset/gp_*.bin",  # Grammar point ids/offsets
                "[history]/config.json",
                "[models]/style/tokenizer.json",
            ]
            bottle.assert_dir_layout(expected_label_manifest)

            bottle.snapshot("after_label")

            # Step 2: Run train_style.sh for 1 style epoch (with 2 KC pretrain epochs)
            result = bottle.train_style(f"--epochs 1 {train_style_args}")

            # With --kc-skip-first-metrics 1 and --kc-epochs 2,
            # epoch 0 skips diagnostics but epoch 1 generates them
            history = bottle.get_epoch_history()
            diag_events = [e for e in history if e.get_type_name() == "KC_DIAG"]
            self.assertTrue(
                len(diag_events) > 0,
                "KC diagnostics event missing from history",
            )
            # Basic validation of content
            stats = diag_events[0].stats
            self.assertIn("families", stats)

            # Verify epochs trained: expect [1, 2] KC epochs and [1] style epoch
            bottle.assert_kc_epochs_trained([1, 2])
            bottle.assert_style_epochs_trained([1])
            bottle.assert_continuation_epochs(kc=2, style=1)

            # Verify no NaNs in history
            bottle.assert_no_nans_in_history()

            # Verify changes since snapshot (should only be training artifacts)
            bottle.assert_files_exist(["[history]/training.log"])

            expected_train_differences = [
                "[history]/training.log ADDED",
                "[history]/training-history.tsv ADDED",
                "[.profile]/* ADDED",
                "[history]/config.json MODIFIED",
                "[models]/style/__init__.py ADDED",
                "[models]/style/model.pt ADDED",
                "[models]/style/model.json ADDED",
                "[models]/style/labels.json ADDED",
                "[models]/style/model_type.txt ADDED",
                "[models]/style/continuation.json ADDED",
                "[.cache]/checkpoint.pt ADDED",
            ]

            bottle.assert_dir_diff("after_label", expected_train_differences)

            # Step 3: Take snapshot after first training pass
            bottle.snapshot("after_epoch_1")

            # Step 4: Resume training with --epochs 2 (should train only style epoch 2)
            # Resume is now the default behavior when checkpoints exist
            bottle.train_style("--epochs 2")

            # Verify only style epoch 2 was trained. KC history remains [1, 2].
            bottle.assert_kc_epochs_trained([1, 2])
            bottle.assert_style_epochs_trained([1, 2])
            bottle.assert_continuation_epochs(kc=2, style=2)

            # Verify no NaNs in history after resume
            bottle.assert_no_nans_in_history()

            # Verify changes since first training pass
            expected_resume_differences = [
                # Training artifacts should be modified
                "[history]/training.log MODIFIED",
                "[history]/training-history.tsv MODIFIED",
                "[.profile]/* ADDED",
                "[.profile]/training-profile.txt MODIFIED",
                "[history]/config.json MODIFIED",
                "[models]/style/model.pt MAYBE-MODIFIED",
                "[models]/style/model.json MAYBE-MODIFIED",
                "[models]/style/continuation.json MODIFIED",
            ]

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
            self.assertTrue(k_id_str.isdigit(), f"Key {k_id_str} should be digits")
            self.assertIsInstance(prob, float)

            # --- Model Verification Logic ---
            style_model = bottle.get_file("[models]/style/model.pt")
            style_config_path = bottle.get_file("[models]/style/model.json")

            # 1. Assert FP8 for export
            bottle.assert_model_is_fp8(style_model)

            # 2. Assert Slim Content using torch.load
            # Map location CPU to avoid CUDA errors if on CPU only machine
            slim_state = torch.load(style_model, map_location="cpu", weights_only=True)

            # Slim state should NOT contain kc_decoders for families with is_slim_decoder=True
            # But it MAY contain gender/formality/grammar_point/register decoders (is_slim_decoder=False)
            # slim_state is the state dict directly
            allowed_slim_families = {"gender", "formality", "grammar_point", "register"}
            allowed_shared_layers = {
                "label_hidden1",
                "label_hidden2",  # Label pathway (for grammar_point)
                "mse_hidden1",
                "mse_hidden2",  # MSE pathway (for gender/formality)
                "activation",
                "tanh",  # Activation layers
            }
            unexpected_slim_decoders = [
                k
                for k in slim_state
                if k.startswith("kc_decoders.")
                and not any(f"decoders.{fam}." in k for fam in allowed_slim_families)
                and not any(
                    f"mse_decoders.{fam}." in k for fam in allowed_slim_families
                )
                and not any(
                    k.startswith(f"kc_decoders.{layer}")
                    for layer in allowed_shared_layers
                )
            ]
            self.assertEqual(
                len(unexpected_slim_decoders),
                0,
                f"Exported model should not contain is_slim_decoder=True decoders: {unexpected_slim_decoders}",
            )

            # 3. Assert Config Slimming
            with open(style_config_path, "r", encoding="utf-8") as f:
                style_config = json.load(f)

            self.assertNotIn(
                "kc_target_specs",
                style_config,
                "Exported model.json should NOT have kc_target_specs (Slim Config)",
            )

            # 4. Model size verification is done in save_model() during training.
            # We don't re-verify here because:
            # - load_model returns InferenceClassifier without kc_decoders module
            # - The saved file includes KC decoder weights (for families with is_slim_decoder=False)
            # - This mismatch would cause false verification failures

            # 5. Verify history in epochs.json
            history = bottle.get_epoch_history()
            self.assertTrue(len(history) > 0, "epochs.json should not be empty")

            # Filter to just epoch events for sequence verification
            epoch_events = [
                e for e in history if e.get_type_name() in ["KC_EPOCH", "STYLE_EPOCH"]
            ]

            # Expect at least 4 events: 2 KC epochs + 2 style epochs (interleaved)
            self.assertGreaterEqual(len(epoch_events), 4)

            # With interleaving, pattern is: KC, Style, KC, Style, ...
            self.assertEqual(epoch_events[0].get_type_name(), "KC_EPOCH")
            self.assertEqual(epoch_events[1].get_type_name(), "STYLE_EPOCH")
            self.assertEqual(epoch_events[2].get_type_name(), "KC_EPOCH")
            self.assertEqual(epoch_events[3].get_type_name(), "STYLE_EPOCH")

            # Verify KC pretraining uses ALL grammatical sentences
            self.assertEqual(
                epoch_events[0].metrics["sentence_count"],
                expected_counts["total_grammatic_sentences"],
            )

            # KC metrics check
            self.assertIn("avg_struct_loss", epoch_events[0].metrics)

            # Verify style epochs use full training split (gram + agram)
            for e in epoch_events:
                if e.get_type_name() == "STYLE_EPOCH":
                    self.assertEqual(
                        e.metrics["sentence_count"],
                        expected_counts["total_train_split_size"],
                    )

            # Verify performance profile coherence (clean jsonl, present txt)
            bottle.assert_coherent_performance_profile()


if __name__ == "__main__":
    unittest.main()
