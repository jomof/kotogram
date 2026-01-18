import os
import shutil
import tempfile
import unittest

from kotogram.model import InferenceClassifier, ModelConfig
from train import io as train_io


class TestIOUnit(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_save_model_variations(self):
        config = ModelConfig(vocab_sizes={})
        model = InferenceClassifier(config)

        # Call 1
        path1 = os.path.join(self.test_dir, "model1")
        train_io.save_model(model, path1, config)
        self.assertTrue(os.path.exists(path1))

        # Call 2 (Different path)
        path2 = os.path.join(self.test_dir, "model2")
        train_io.save_model(model, path2, config)
        self.assertTrue(os.path.exists(path2))


if __name__ == "__main__":
    unittest.main()
