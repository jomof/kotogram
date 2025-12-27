import csv
import os
import unittest

from training_test_utils import Bottle

from train.cache import get_kotogram_cache


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
        # Reset global cache instance if needed
        import train.cache

        train.cache._KOTOGRAM_CACHE = None

    def test_label_and_cache(self):
        # Run label.py via subprocess using Bottle
        args = [
            "--grammatic-pattern",
            self.data_file,
        ]

        self.bottle.run_script("scripts/label.py", args)

        # Verify cache was created and populated
        # Re-initialize to see what happened
        import train.cache

        train.cache._KOTOGRAM_CACHE = None

        with self.bottle.environment():
            # In-process verification needs mocked environment
            shards_dir = get_kotogram_cache().shards_dir
            print(f"Checking results in {shards_dir}...")

            cache = get_kotogram_cache()
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
            k, f, g_val, g_prag, r_lbls, g_lbl, _ = results["これはテストです。"]
            self.assertTrue(len(k) > 0)
            self.assertIsNotNone(f)
            self.assertIsNotNone(g_val)
            self.assertIsNotNone(g_prag)
            self.assertIsNotNone(r_lbls)
            self.assertIsNotNone(g_lbl)

    def test_incremental_labeling(self):
        # First run
        import train.cache

        train.cache._KOTOGRAM_CACHE = None

        args = [
            "--grammatic-pattern",
            self.data_file,
        ]

        self.bottle.run_script("scripts/label.py", args)

        with self.bottle.environment():
            # Verify something was written
            shards_dir = get_kotogram_cache().shards_dir
            files = os.listdir(shards_dir)
            print(f"Shard files: {files}")
            if not files:
                self.fail("No shard files created")

            shard_path = os.path.join(shards_dir, files[0])
            os.path.getmtime(shard_path)

        # Second run with same data
        self.bottle.run_script("scripts/label.py", args)

        # Add a new file
        new_data_file = os.path.join(self.bottle.root_dir, "new_data.tsv")
        with open(new_data_file, "w", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["id4", "label4", "新しい文です。"])

        args_new = [
            "--grammatic-pattern",
            self.data_file,
            "--agrammatic-pattern",
            new_data_file,
        ]

        self.bottle.run_script("scripts/label.py", args_new)

        train.cache._KOTOGRAM_CACHE = None
        with self.bottle.environment():
            cache = get_kotogram_cache()
            results = cache.get_batch(["新しい文です。"])
            self.assertIsNotNone(results["新しい文です。"])


if __name__ == "__main__":
    unittest.main()
