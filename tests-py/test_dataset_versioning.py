import json
import os
import sys
import unittest
from unittest.mock import MagicMock, mock_open, patch

# Ensure projects root is in path
sys.path.append(os.getcwd())

# pylint: disable=wrong-import-position
from train.dataset import CACHE_VERSION, StyleDataset


class TestDatasetVersioning(unittest.TestCase):
    def test_cache_version_mismatch(self):
        """Verify mismatched cache version raises ValueError immediately."""

        # Create a dummy vocab content with WRONG version
        dummy_vocab = {
            "version": CACHE_VERSION - 1,  # Mismatch
            "surface": {},
            "lemma": {},
        }

        # Mock open to read our dummy vocab
        with patch("builtins.open", mock_open(read_data=json.dumps(dummy_vocab))):
            with patch("os.path.exists", return_value=True):
                mock_tokenizer = MagicMock()

                # pylint: disable=protected-access
                with self.assertRaises(ValueError) as cm:
                    StyleDataset._load_vocab("dummy_vocab.json", mock_tokenizer)

            self.assertIn("Cache version mismatch", str(cm.exception))
            self.assertIn(f"Expected {CACHE_VERSION}", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
