import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

from kotogram.analysis import _ANALYZER
from kotogram.cli import main


class TestCliModelDir(unittest.TestCase):
    def setUp(self):
        # Create a temp dir to act as custom model dir
        self.test_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.test_dir, "model.pt")
        self.config_path = os.path.join(self.test_dir, "model.json")
        self.tokenizer_path = os.path.join(self.test_dir, "tokenizer.json")

        # Create dummy artifacts
        # We can't easily create a full valid model.pt that loads without dependencies,
        # so we will rely on mocking load_model or just checking if the path is passed correctly.
        # But wait, we want integration test.
        # We can create a dummy file and mock load_model.

        # Reset analyzer
        _ANALYZER.set_model_dir(None)

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        _ANALYZER.set_model_dir(None)

    @patch("kotogram.model.load_model")
    def test_custom_model_dir_used(self, mock_load_model):
        """Verify that --model-dir triggers loading from that path."""
        # Create dummy file so exists() check passes
        with open(self.model_path, "w", encoding="utf-8") as f:
            f.write("dummy")

        # Mock return value of load_model to avoid actual loading
        mock_model = MagicMock()
        # KC is always enabled; mock predict_kcs_top to return list with one empty element per batch
        mock_model.predict_kcs_top.return_value = [[]]  # One batch sample, no KCs
        mock_model.config.kc_threshold = 0.5  # Adaptive threshold default
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # Basic mocks for prediction to avoid crash
        mock_prediction = MagicMock()
        # Mocking return values for prediction attributes to simple float tensors
        mock_prediction.formality_value = torch.tensor([0.0])
        mock_prediction.formality_pragmatic_probs = torch.tensor(
            [[0.0, 1.0]]
        )  # Pragmatic
        mock_prediction.gender_value = torch.tensor([0.0])
        mock_prediction.gender_pragmatic_probs = torch.tensor([[0.0, 1.0]])
        mock_prediction.register_probs = torch.tensor([[0.0] * 10])
        mock_prediction.grammaticality_probs = torch.tensor([[0.0, 1.0]])
        mock_model.predict.return_value = mock_prediction
        # Mock predict_grammar_points to return None (no decoder available)
        mock_model.predict_grammar_points.return_value = None

        # Mock tokenizer encode
        mock_tokenizer.encode.return_value = {
            "input_ids": [1, 2, 3]
        }  # Assuming field name

        # We need to mock tokenizer.encode more realistically or mock FEATURE_FIELDS
        with (
            patch("kotogram.tokenizer.FEATURE_FIELDS", ["input_ids"]),
            patch("kotogram.tokenizer.ENCODER_FEATURE_FIELDS", ["input_ids"]),
        ):
            mock_tokenizer.encode.return_value = {"input_ids": [1, 2, 3]}

            # Run CLI
            test_args = [
                "kotogram",
                "--model-dir",
                self.test_dir,
                "grammar",
                "こんにちは",
            ]
            with patch("sys.argv", test_args):
                with patch("sys.stdout"):  # Suppress output
                    main()

        # Verify load_model was called with our custom dir
        mock_load_model.assert_called_with(self.test_dir)

    @patch("kotogram.model.load_model")
    def test_custom_model_dir_not_found(self, _mock_load_model):
        """Verify error if custom model dir does not exist/have model."""
        # Don't create model.pt

        test_args = [
            "kotogram",
            "--model-dir",
            self.test_dir,
            "grammar",
            "こんにちは",
        ]
        with patch("sys.argv", test_args):
            with patch("sys.stderr") as mock_stderr:
                ret = main()
                self.assertEqual(ret, 1)

                # Check that some error was printed
                # sys.stderr.write is called.
                # mock_stderr.write.assert_called() # This might be called multiple times

                # We can check if any call args contain "Error"
                writes = [
                    args[0] for name, args, kwargs in mock_stderr.write.mock_calls
                ]
                combined = "".join(writes)
                self.assertIn("Error: Model file not found", combined)


if __name__ == "__main__":
    unittest.main()
