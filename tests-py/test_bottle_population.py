import os
import unittest

from training_test_utils import Bottle


class TestBottlePopulation(unittest.TestCase):
    def test_populate_samples(self):
        """Verifies that populate_test_data produces the expected diversity and quantity of samples."""
        with Bottle(self) as bottle:
            bottle.populate_test_data()

            data_dir = bottle.resolve_path("[data]")

            # 1. Verify Agrammatic Samples (5 expected)
            agram_path = os.path.join(data_dir, "jpn_agrammatic_sampled.tsv")
            self.assertTrue(os.path.exists(agram_path), f"Missing {agram_path}")
            with open(agram_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            self.assertGreaterEqual(
                len(lines), 5, "Expected at least 5 agrammatic samples"
            )

            # 2. Verify Register Specific Files (5 each for all registers)
            # Read registers from corpus.db instead of hardcoded constant
            import sqlite3

            corpus_db_path = os.path.join(bottle.project_root, "data", "corpus.db")
            self.assertTrue(
                os.path.exists(corpus_db_path),
                f"Corpus DB not found at {corpus_db_path}",
            )

            conn = sqlite3.connect(corpus_db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT label FROM register")
                db_registers = [row[0] for row in cursor.fetchall()]
            finally:
                conn.close()

            for reg_label_str in db_registers:
                reg_name = reg_label_str.lower()
                reg_path = os.path.join(data_dir, f"jpn_sentences_{reg_name}.tsv")

                self.assertTrue(
                    os.path.exists(reg_path), f"Missing register file: {reg_path}"
                )

                with open(reg_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]

                # We expect exactly 5 samples per register query
                self.assertEqual(
                    len(lines),
                    5,
                    f"Expected 5 samples for register {reg_name}, got {len(lines)}",
                )

            # 3. Verify Generic Grammatic Samples
            # Should contain samples for Gender (4 types * 5), Formality (6 types * 5), Generic Grammatic (5)
            # Total expected distinct sentences roughly 20 + 30 + 5 = 55 (assuming no overlap)
            # Overlap is possible but low probability with random sampling from large corpus?
            # Actually, `sample_test_data` uses `LIMIT 5`, queries might return overlapping sentences if corpus is small/biased?
            # But the set() dedups them.
            # Let's ensure a robust minimum count.

            gram_path = os.path.join(data_dir, "jpn_sentences_sampled.tsv")
            self.assertTrue(os.path.exists(gram_path), f"Missing {gram_path}")
            with open(gram_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            # A strict count might be flaky if the DB is small or queries return same sentences.
            # But we expect significant diversity.
            # Let's assert at least 35 to be safe (allowing some overlap/deduplication).
            self.assertGreater(
                len(lines),
                35,
                f"Expected >35 diverse samples in {gram_path}, got {len(lines)}",
            )


if __name__ == "__main__":
    unittest.main()
