import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from kotogram.tokenizer import Tokenizer
from train import io as train_io
from train.dataset import StyleDataset


class TestDataflowV2(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        # We don't set TRAIN_ROOT globally in process,
        # because StyleDataset might verify paths at import time?
        # No, defaults are functions.
        # But we set it for subprocess.

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        # Verify clean up?

    def _create_data(self):
        data_dir = os.path.join(self.test_dir, "input_data")
        os.makedirs(data_dir)
        tsv_path = os.path.join(data_dir, "test.tsv")
        with open(tsv_path, "w", encoding="utf-8") as f:
            for i in range(100):
                f.write(f"{i}\t1\tこれはテスト文{i}です。\n")
        return tsv_path

    def _run_label(self, tsv_path):
        script_path = os.path.join(os.getcwd(), "scripts", "label.py")
        env = os.environ.copy()
        env["TRAIN_ROOT"] = self.test_dir
        env["PYTHONPATH"] = os.getcwd()
        subprocess.check_call(
            [sys.executable, script_path, "--grammatic-pattern", tsv_path],
            env=env,
        )

    def _verify_cache(self):
        cache_dir = os.path.join(self.test_dir, ".cache", "style_dataset")
        self.assertTrue(os.path.exists(cache_dir), "Cache dir not created")
        expected_files = [
            "feat_surface.bin",
            "feat_lemma.bin",
            "offsets.bin",
            "sentences.txt",
            "kotograms.txt",
            "vocab.json",
        ]
        for f in expected_files:
            self.assertTrue(os.path.exists(os.path.join(cache_dir, f)), f"{f} missing")
        return cache_dir

    def test_end_to_end_pipeline(self):
        """Run label.py and then load with StyleDataset."""
        # 1. Create Dummy Data
        tsv_path = self._create_data()

        # 2. Run label.py
        self._run_label(tsv_path)

        # 3. Verify Outputs
        cache_dir = self._verify_cache()

        # 4. Load with StyleDataset
        model_dir = os.path.join(self.test_dir, "models", "style")
        os.makedirs(model_dir)

        # Create dummy tokenizer.json
        tok = Tokenizer()

        train_io.save_tokenizer(tok, os.path.join(model_dir, "tokenizer.json"))

        dataset = StyleDataset(
            data_dir=cache_dir,
            tokenizer=tok,
        )

        # Verify
        self.assertEqual(len(dataset), 100)

        # Check sample
        sample = dataset[0]
        self.assertIsInstance(sample.feature_ids, dict)
        self.assertIn("lemma", sample.feature_ids)
        self.assertTrue(len(sample.feature_ids["lemma"]) > 0)

        # Inspect surface IDs
        surface_ids = sample.feature_ids["surface"]
        valid_ids = [fid for fid in surface_ids if fid > 2]
        self.assertTrue(len(valid_ids) > 0, f"All tokens special: {surface_ids}")

        # Check Register Labels (ragged)
        # "これはテスト文です0" has override KYOSHIGO (assumed ID 4)
        self.assertEqual(sample.register_labels, [4])

        # Split
        train_ds, val_ds = dataset.split(train_ratio=0.8)
        self.assertEqual(len(train_ds), 80)
        self.assertEqual(len(val_ds), 20)
        self.assertTrue(train_ds[0].feature_ids)


if __name__ == "__main__":
    unittest.main()
