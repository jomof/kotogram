import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts.confusion import generate_reports


class TestConfusionReports(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.mock_data = {
            "sentences": ["s1", "s2"],
            "kotograms": ["k1", "k2"],
            "formality_prag_preds": [0, 1],
            "formality_prag_labels": [0, 1],
            "formality_val_preds": [0.0, 1.0],
            "formality_val_labels": [0.0, 1.0],
            "gender_prag_preds": [0, 1],
            "gender_prag_labels": [0, 1],
            "gender_val_preds": [0.0, 1.0],
            "gender_val_labels": [0.0, 1.0],
            "grammaticality_preds": [1, 0],
            "grammaticality_labels": [1, 0],
            # NUM_REGISTER_CLASSES is 14
            "register_preds": [[1] + [0] * 13, [0] * 14],
            "register_labels": [[1] + [0] * 13, [0] * 14],
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("scripts.confusion.console")
    def test_generate_reports_basic(self, _):
        """Test that generate_reports runs without error and makes expected calls."""
        # Create output directory
        output_dir = os.path.join(self.test_dir, "output")
        generate_reports(self.mock_data, output_dir)

        # Check files created
        self.assertTrue(os.path.exists(output_dir))
        # We expect no mismatches in perfect prediction scenario
        # So no mismatch files should be created, or maybe empty ones?
        # The logic says: if mismatches: save.
        # Let's introduce a mismatch to ensure files are created.

        mismatch_data = self.mock_data.copy()
        # Formality mismatch
        mismatch_data["formality_prag_preds"] = [1, 0]  # flipped
        mismatch_data["formality_prag_labels"] = [0, 1]

        generate_reports(mismatch_data, output_dir)

        # Check for formality confusion files
        self.assertTrue(
            os.path.exists(os.path.join(output_dir, "formality_confusion.csv"))
        )

    @patch("scripts.confusion.console")
    def test_generate_reports_perfect_match(self, _):
        """Test with perfect predictions (no mismatch files)."""
        output_dir = os.path.join(self.test_dir, "output_perfect")
        generate_reports(self.mock_data, output_dir)

        # No mismatch CSVs should exist for perfectly matched data
        self.assertFalse(
            os.path.exists(os.path.join(output_dir, "formality_confusion.csv"))
        )
        self.assertFalse(
            os.path.exists(os.path.join(output_dir, "grammaticality_confusion.csv"))
        )


if __name__ == "__main__":
    unittest.main()
