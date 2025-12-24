import csv
import os
import shutil
import sys
import tempfile
import unittest

# Add project root to path so we can import scripts and kotogram
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch

from kotogram import locations
from scripts.cache import get_kotogram_cache
from scripts.label import main as label_main


class TestLabelScript(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        with patch.dict(os.environ, {"TRAIN_ROOT": self.test_dir}):
            self.shards_dir = locations.get_shards_cache_dir()
        self.data_file = os.path.join(self.test_dir, "test_data.tsv")

        # Create dummy data
        with open(self.data_file, "w", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["id1", "label1", "これはテストです。"])
            writer.writerow(["id2", "label2", "美味しいですね。"])
            writer.writerow(["id3", "label3", "走る。"])

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        # Reset global cache instance if needed
        import scripts.cache

        scripts.cache._kotogram_cache = None

    def test_label_and_cache(self):
        # Run label.py via main
        import sys
        from unittest.mock import patch

        # Mock sys.argv
        test_args = [
            "scripts/label.py",
            "--grammatic-pattern",
            self.data_file,
            # "--cache-dir", self.test_dir # Removed
        ]

        # Point the cache to our temp dir
        # Ensure the global cache is reset
        import scripts.cache

        scripts.cache._kotogram_cache = None

        with patch.dict(os.environ, {"TRAIN_ROOT": self.test_dir}):
            with patch.object(sys, "argv", test_args):
                label_main()

            # Verify cache was created and populated
            # Re-initialize to see what happened
            scripts.cache._kotogram_cache = None
            cache = get_kotogram_cache()

            print(f"Checking results in {self.shards_dir}...")
            results = cache.get_batch(
                ["これはテストです。", "美味しいですね。", "走る。"]
            )

            for k_sent, v in results.items():
                if v is None:
                    print(f"MISSING: {k_sent}")

            self.assertIsNotNone(results["これはテストです。"])
            self.assertIsNotNone(results["美味しいですね。"])
            self.assertIsNotNone(results["走る。"])

            # Check if fields are populated
            k, f, g_val, g_prag, r_lbls, g_lbl, f_ids = results["これはテストです。"]
            self.assertTrue(len(k) > 0)
            self.assertIsNotNone(f)
            self.assertIsNotNone(g_val)
            self.assertIsNotNone(g_prag)
            self.assertIsNotNone(r_lbls)
            self.assertIsNotNone(g_lbl)

    def test_incremental_labeling(self):
        # First run
        import sys
        from unittest.mock import patch

        import scripts.cache

        scripts.cache._kotogram_cache = None

        test_args = [
            "scripts/label.py",
            "--grammatic-pattern",
            self.data_file,
            # "--cache-dir", self.test_dir # Removed
        ]

        with patch.dict(os.environ, {"TRAIN_ROOT": self.test_dir}):
            with patch.object(sys, "argv", test_args):
                label_main()

            # Verify something was written
            files = os.listdir(self.shards_dir)
            print(f"Shard files: {files}")
            if not files:
                self.fail("No shard files created")

            shard_path = os.path.join(self.shards_dir, files[0])
            os.path.getmtime(shard_path)

            # Second run with same data
            with patch.object(sys, "argv", test_args):
                label_main()

            # Add a new file
            new_data_file = os.path.join(self.test_dir, "new_data.tsv")
            with open(new_data_file, "w", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["id4", "label4", "新しい文です。"])

            test_args_new = [
                "scripts/label.py",
                "--grammatic-pattern",
                self.data_file,
                "--agrammatic-pattern",
                new_data_file,
                # "--cache-dir", self.test_dir
            ]
            with patch.object(sys, "argv", test_args_new):
                label_main()

            scripts.cache._kotogram_cache = None
            cache = get_kotogram_cache()
            results = cache.get_batch(["新しい文です。"])
            self.assertIsNotNone(results["新しい文です。"])


if __name__ == "__main__":
    unittest.main()
