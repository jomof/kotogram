import csv
import os
import unittest

from training_test_utils import Bottle

# V2: We check for binary shards and text files instead of SQLite cache
from train import paths as train_paths


# pylint: disable=protected-access, unnecessary-dunder-call
class TestLabelScript(unittest.TestCase):
    def setUp(self):
        self.bottle = Bottle(self)
        self.bottle.__enter__()

        # Create dummy data in bottle root
        self.data_file = os.path.join(self.bottle.root_dir, "test_data.tsv")
        with open(self.data_file, "w", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["id1", "label1", "これはテストです。"])
            writer.writerow(["id2", "label2", "美味しいですね。"])
            writer.writerow(["id3", "label3", "走る。"])

    def tearDown(self):
        self.bottle.__exit__(None, None, None)

    def test_label_output_artifacts(self):
        # Run label.py via subprocess using Bottle
        # V2 label.py writes to locations.get_style_dataset_cache_dir()
        args = [
            "--grammatic-pattern",
            self.data_file,
        ]

        self.bottle.run_script("scripts/label.py", args)

        with self.bottle.environment():
            # Verify V2 artifacts
            cache_dir = train_paths.get_style_dataset_cache_dir()
            print(f"Checking results in {cache_dir}...")

            # Check for core text files
            sentences_path = os.path.join(cache_dir, "sentences.txt")
            kotograms_path = os.path.join(cache_dir, "kotograms.txt")

            self.assertTrue(os.path.exists(sentences_path), "sentences.txt missing")
            self.assertTrue(os.path.exists(kotograms_path), "kotograms.txt missing")

            # Verify content
            with open(sentences_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f]
                self.assertIn("これはテストです。", lines)
                self.assertIn("美味しいですね。", lines)
                self.assertIn("走る。", lines)

            with open(kotograms_path, "r", encoding="utf-8") as f:
                k_lines = [line.strip() for line in f]
                self.assertEqual(len(k_lines), 3)
                self.assertTrue(all(len(k) > 0 for k in k_lines))

            # Check for binary shards (offsets, labels)
            # We expect at least shard_0... or unified binaries if merged?
            # scripts/label.py merges them into binaries in the root of cache_dir
            exts = [
                "labels.bin_f_val",
                "labels.bin_f_prag",
                "labels.bin_g_val",
                "labels.bin_g_prag",
                "labels.bin_gram",
                "labels.bin_reg_ids.bin",
            ]
            for ext in exts:
                path = os.path.join(cache_dir, ext)
                self.assertTrue(os.path.exists(path), f"Missing {ext}")

            # Check JSON metadata
            vocab_path = os.path.join(cache_dir, "vocab.json")
            self.assertTrue(os.path.exists(vocab_path), "vocab.json missing")

    def test_incremental_labeling_v2_artifacts(self):
        # First run
        args = [
            "--grammatic-pattern",
            self.data_file,
        ]

        self.bottle.run_script("scripts/label.py", args)

        with self.bottle.environment():
            cache_dir = train_paths.get_style_dataset_cache_dir()
            s_path = os.path.join(cache_dir, "sentences.txt")
            self.assertTrue(os.path.exists(s_path))

        # Add a new file
        new_data_file = os.path.join(self.bottle.root_dir, "new_data.tsv")
        with open(new_data_file, "w", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["id4", "label4", "新しい文です。"])

        # Run with both
        args_new = [
            "--grammatic-pattern",
            self.data_file,
            "--agrammatic-pattern",
            new_data_file,
        ]

        self.bottle.run_script("scripts/label.py", args_new)

        with self.bottle.environment():
            cache_dir = train_paths.get_style_dataset_cache_dir()
            s_path = os.path.join(cache_dir, "sentences.txt")

            with open(s_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f]
                self.assertIn("新しい文です。", lines)
                # It might have duplicates if not deduped, but V2 label.py dedups globally in main logic
                self.assertIn("これはテストです。", lines)

    def test_force_relabel(self):
        # First run to populate cache
        args = ["--grammatic-pattern", self.data_file]
        self.bottle.run_script("scripts/label.py", args)

        with self.bottle.environment():
            cache_dir = train_paths.get_style_dataset_cache_dir()
            s_path = os.path.join(cache_dir, "sentences.txt")
            self.assertTrue(os.path.exists(s_path))

            # Create a sentinel file that should be deleted
            sentinel = os.path.join(cache_dir, "sentinel.txt")
            with open(sentinel, "w", encoding="utf-8") as f:
                f.write("I should be deleted")

        # Run with --force-relabel
        args_force = ["--grammatic-pattern", self.data_file, "--force-relabel"]
        self.bottle.run_script("scripts/label.py", args_force)

        with self.bottle.environment():
            cache_dir = train_paths.get_style_dataset_cache_dir()
            s_path = os.path.join(cache_dir, "sentences.txt")
            self.assertTrue(os.path.exists(s_path))
            self.assertFalse(os.path.exists(os.path.join(cache_dir, "sentinel.txt")))


if __name__ == "__main__":
    unittest.main()
