import sys
import os
import unittest
import json

# Add tests-py directory to path to allow importing utility module
sys.path.append(os.path.dirname(__file__)) 
from training_test_utils import Bottle

@unittest.skipIf(os.environ.get("GITHUB_ACTIONS") == "true", "Skipping on GitHub CI")
class TestTrainStyleScript(unittest.TestCase):
    def test_train(self):
        # Test both regular training and training with MLM pretraining
        test_configs = [
            {"name": "regular", "extra_args": ""},
            {"name": "pretrain-mlm", "extra_args": "--pretrain-mlm --pretrain-epochs 1"},
        ]
        
        for config in test_configs:
            with self.subTest(config=config["name"]):
                with Bottle(self) as bottle:
                    bottle.populate_test_data()
                    
                    # Take initial snapshot (also resets profile counters)
                    from kotogram.profile import get_profile_report
                    profile_dir = bottle.get_file(".profile")
                    bottle.snapshot("initial")
                    
                    # Step 1: Pre-run labeling to setup cache/data
                    bottle.train_style("--label")
                    
                    # Verify label phase output files using glob patterns
                    EXPECTED_LABEL_MANIFEST = [
                        # Generated Cache Files
                        '[.cache]/agrammatic_combined.tsv',
                        '[.cache]/grammatic_combined.tsv',
                        
                        # Shards (using glob to be robust against hashing changes)
                        '[.cache]/kotogram_shards/*.db',
                        
                        # Style Data
                        '[models]/style-support/register_samples.csv',
                        '[models]/style-support/timing.yml',
                        '[.cache]/style_dataset/label_metadata.json',
                        '[.cache]/style_dataset/vocab.json',
                        
                        # Source Data (using globs to be robust against adding new data files)
                        '[data]/jpn_agrammatic_*.tsv',
                        '[data]/jpn_sentences*.tsv',

                        # Empty directories created by scripts
                        '[models]/style',
                    ]
                    bottle.assert_dir_layout(EXPECTED_LABEL_MANIFEST)
                    
                    # Verify profiling captured japanese_to_kotogram calls
                    report = get_profile_report(profile_dir=profile_dir)
                    # Exact count based on test data: 5 lines from each TSV file
                    self.assertEqual(
                        report.get_counter("japanese_to_kotogram"), 95,
                        f"Expected exactly 95 japanese_to_kotogram calls, got: {report.counters}"
                    )
                    
                    bottle.snapshot("after_label")
                    
                    # Step 2: Run train_style.sh for 1 epoch (with optional pretrain-mlm)
                    train_args = f"--epochs 1 {config['extra_args']}".strip()
                    result1 = bottle.train_style(train_args)
                    
                    # Verify epochs trained
                    # For pretrain-mlm: expect [1, 1] (1 MLM pretrain + 1 fine-tune)
                    # For regular: expect [1] (just 1 fine-tune)
                    expected_epochs = [1, 1] if "pretrain-mlm" in config['extra_args'] else [1]
                    bottle.assertEpochsTrained(result1, expected_epochs)
                    
                    # Verify changes since snapshot (should only be training artifacts)
                    EXPECTED_TRAIN_DIFFERENCES = [
                        # Style Data & Confusion Matrices (Training artifacts)
                        '[models]/style-support/timing.yml MODIFIED',
                        '[models]/style-support/*_confusion.csv ADDED',
                        '[models]/style-support/confusion_matrices/*.tsv ADDED',
                        
                        # Model Output
                        '[models]/style-support/training.log ADDED',
                        '[models]/style-support/checkpoint.pt ADDED',
                        '[models]/style-support/config.json ADDED',
                        '[models]/style-support/tokenizer.json ADDED',
                        '[models]/style/model.pt ADDED', 
                        '[models]/style/config.json ADDED',
                        '[models]/style/labels.json ADDED',
                        '[models]/style/model_type.txt ADDED',
                        '[models]/style/tokenizer.json ADDED',
                    ]
                    
                    bottle.assert_dir_diff("after_label", EXPECTED_TRAIN_DIFFERENCES)
                    
                    # Verify profiling captured japanese_to_kotogram calls during training
                    # Should be 0 because training uses the cached dataset and doesn't re-parse raw text
                    report = get_profile_report(profile_dir=profile_dir)
                    self.assertEqual(
                        report.get_counter("japanese_to_kotogram"), 0,
                        f"Expected 0 japanese_to_kotogram calls during training (should use cache), got: {report.counters}"
                    )
                    
                    # Step 3: Take snapshot after first training pass
                    bottle.snapshot("after_epoch_1")
                    
                    # Step 4: Resume training with --epochs 2 (should train only epoch 2)
                    result2 = bottle.train_style("--resume --epochs 2")
                    
                    # Verify only epoch 2 was trained (resume from epoch 1)
                    bottle.assertEpochsTrained(result2, [2])
                    
                    # Verify changes since first training pass
                    EXPECTED_RESUME_DIFFERENCES = [
                        # Training artifacts should be modified
                        '[models]/style-support/training.log MODIFIED',
                        '[models]/style-support/checkpoint.pt MODIFIED',
                        '[models]/style-support/timing.yml MODIFIED',
                        '[models]/style-support/config.json MODIFIED',
                        '[models]/style-support/tokenizer.json MODIFIED',
                        '[models]/style-support/*_confusion.csv MODIFIED',
                        '[models]/style-support/confusion_matrices/*.tsv MODIFIED',
                        
                        # Model output should be updated
                        '[models]/style/model.pt MODIFIED',
                        '[models]/style/config.json MODIFIED',
                        '[models]/style/labels.json MODIFIED',
                        '[models]/style/model_type.txt MODIFIED',
                        '[models]/style/tokenizer.json MODIFIED',
                    ]
                    
                    bottle.assert_dir_diff("after_epoch_1", EXPECTED_RESUME_DIFFERENCES)
                    
                    # Verify profiling captured japanese_to_kotogram calls during resume training
                    # Should be 0 because training uses the cached dataset
                    report = get_profile_report(profile_dir=profile_dir)
                    self.assertEqual(
                        report.get_counter("japanese_to_kotogram"), 0,
                        f"Expected 0 japanese_to_kotogram calls during resume (should use cache), got: {report.counters}"
                    )
                    
                    # Verify CLI tool works with the newly trained model
                    result = bottle.kotogram_cli("grammar", "こんにちは")
                    
                    # 1. Verify stderr is empty (no reload/path error messages)
                    self.assertEqual(result.stderr.strip(), "", msg=f"CLI stderr should be empty, but got:\n{result.stderr}")
                    
                    # 2. Verify result is valid JSON
                    try:
                        data = json.loads(result.stdout)
                        self.assertIn("kotogram", data)
                        self.assertIn("formality", data)
                    except json.JSONDecodeError as e:
                        self.fail(f"CLI stdout was not valid JSON: {e}\nSTDOUT:\n{result.stdout}")

                    # 3. Verify model.pt is in FP8 format
                    model_path = bottle.get_file("[models]/style/model.pt")
                    bottle.assertModelIsFp8(model_path)


if __name__ == "__main__":
    unittest.main()
