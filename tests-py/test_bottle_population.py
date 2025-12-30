import os
import sqlite3
import unittest

from training_test_utils import Bottle


class TestBottlePopulation(unittest.TestCase):
    def test_populate_samples(self):
        """Verifies that populate_test_data produces the expected diversity and quantity of samples in corpus.db."""
        with Bottle(self) as bottle:
            bottle.populate_test_data()

            data_dir = bottle.resolve_path("[data]")
            db_path = os.path.join(data_dir, "corpus.db")

            self.assertTrue(
                os.path.exists(db_path),
                f"Corpus DB missing at {db_path}",
            )

            conn = sqlite3.connect(db_path)
            try:
                cursor = conn.cursor()

                # 1. Verify Agrammatic Samples (grammatic=0)
                # Expected: 5 agrammatic samples
                cursor.execute("SELECT count(*) FROM corpus WHERE grammatic = 0")
                (agram_count,) = cursor.fetchone()
                self.assertGreaterEqual(
                    agram_count,
                    5,
                    f"Expected at least 5 agrammatic samples, got {agram_count}",
                )

                # 2. Verify Register Specific Samples
                # We need to map register label -> id first from the SOURCE corpus db
                # or just check that we have sentences with populated register_ids.
                # Since populate_test_data queries *all* registers, we should see diversity.

                cursor.execute(
                    "SELECT count(*) FROM corpus WHERE register_ids IS NOT NULL AND register_ids != ''"
                )
                (reg_count,) = cursor.fetchone()
                self.assertGreater(
                    reg_count,
                    0,
                    "Expected some usage of register_ids",
                )

                # Check specific registers?
                # The populate_test_data iterates over all registers and adds 5 each.
                # So we expect roughly 5 * num_registers samples that have register_ids.
                # Let's verify we have at least 15 (assuming >= 3 registers).
                self.assertGreaterEqual(
                    reg_count,
                    15,
                    f"Expected >= 15 register-bearing sentences, got {reg_count}",
                )

                # 3. Verify Grammatic Samples (grammatic=1)
                # Includes Gender, Formality, and Generic samples.
                # Rough expectation: > 35 samples.
                cursor.execute("SELECT count(*) FROM corpus WHERE grammatic = 1")
                (gram_count,) = cursor.fetchone()

                self.assertGreater(
                    gram_count,
                    25,
                    f"Expected >25 grammatic samples, got {gram_count}",
                )

                # 4. Total Count Check
                cursor.execute("SELECT count(*) FROM corpus")
                (total_count,) = cursor.fetchone()
                self.assertEqual(
                    total_count,
                    agram_count + gram_count,
                    "Total count verification failed",
                )

                print(
                    f"Verified Corpus DB: {total_count} total rows "
                    f"({gram_count} grammatic, {agram_count} agrammatic)"
                )

            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
