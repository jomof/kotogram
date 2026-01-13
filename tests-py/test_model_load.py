import os
import shutil
import tempfile
import unittest

import torch

from kotogram.model import ModelConfig, StyleClassifier, load_model
from kotogram.tokenizer import Tokenizer
from train import io as train_io


class TestModelLoadUnit(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.test_dir, "model")
        os.makedirs(self.model_path)

        # Create dummy model - use ENCODER_FEATURE_FIELDS vocab sizes
        config = ModelConfig(
            vocab_sizes={
                "pos": 100,
                "pos_detail_1": 100,
                "pos_detail_2": 100,
                "pos_detail_3": 100,
                "conjugated_form": 100,
                "conjugated_type": 100,
                "reading": 100,
            }
        )
        model = StyleClassifier(config)

        # Save dummy model using io
        train_io.save_model(model, self.model_path, config)

        # Save dummy tokenizer
        tok = Tokenizer()
        train_io.save_tokenizer(tok, os.path.join(self.model_path, "tokenizer.json"))

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_load_model_device_variation(self):
        # Call 1: Default device (None)
        m1, _ = load_model(self.model_path)
        self.assertIsInstance(m1, StyleClassifier)

        # Call 2: Specific device (CPU)
        m2, _ = load_model(self.model_path, device=torch.device("cpu"))
        self.assertIsInstance(m2, StyleClassifier)


if __name__ == "__main__":
    unittest.main()
