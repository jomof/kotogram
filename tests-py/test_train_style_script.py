import sys
import os
import unittest
import json

# Add tests-py directory to path to allow importing utility module
sys.path.append(os.path.dirname(__file__)) 
from training_test_utils import Bottle

class TestTrainStyleScript(unittest.TestCase):
    def test_label_only(self):
        with Bottle(self) as bottle:
            bottle.populate_test_data()
            
            # Run train_style.sh with TRAIN_ROOT handled by Bottle
            bottle.train_style("--label")
            
            # Verify output files using glob patterns for robustness
            EXPECTED_MANIFEST = [
                # Generated Cache Files
                '[.cache]/agrammatic_combined.tsv',
                '[.cache]/grammatic_combined.tsv',
                
                # Shards (using glob as requested to be robust against hashing changes)
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
            
            
            bottle.assert_dir_layout(EXPECTED_MANIFEST)

    def test_train_epochs_1(self):
        with Bottle(self) as bottle:
            bottle.populate_test_data()
            
            # Step 1: Pre-run labeling to setup cache/data
            bottle.train_style("--label")
            bottle.snapshot("after_label")
            
            # Step 2: Run train_style.sh for 1 epoch
            bottle.train_style("--epochs 1")
            
            # Verify changes since snapshot (should only be training artifacts)
            EXPECTED_DIFFERENCES = [
                # Style Data & Confusion Matrices (Training artifacts)
                '[models]/style-support/timing.yml MODIFIED',
                '[models]/style-support/*_confusion.csv ADDED', # Confusion matrix output from scripts.confusion
                '[models]/style-support/confusion_matrices/*.tsv ADDED', # Confusion matrices from Trainer.evaluate
                
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
            
            bottle.assert_dir_diff("after_label", EXPECTED_DIFFERENCES)
            
            # Verify CLI tool works with the newly trained model
            # This verifies the bin/kotogram respects TRAIN_ROOT (via analysis.py update)
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
