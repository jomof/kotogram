import os
import shutil
import tempfile
import unittest

import torch

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

    def test_training_state_variations(self):
        config = ModelConfig(vocab_sizes={})
        model = InferenceClassifier(config)
        optimizer = torch.optim.Adam(model.parameters())

        # Call 1
        path1 = os.path.join(self.test_dir, "state1")
        train_io.save_training_state(
            path1, model, optimizer, epoch=1, history={}, config=config
        )
        self.assertTrue(os.path.exists(os.path.join(path1, "checkpoint.pt")))

        train_io.load_training_state(path1, model, optimizer)

        # Call 2 (Different path, filename, step, scheduler)
        path2 = os.path.join(self.test_dir, "state2")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        train_io.save_training_state(
            path2,
            model,
            optimizer,
            epoch=2,
            history={},
            config=config,
            global_step=10,
            scheduler=scheduler,
            filename="ckpt.pt",
        )
        self.assertTrue(os.path.exists(os.path.join(path2, "ckpt.pt")))

        train_io.load_training_state(
            path2, model, optimizer, scheduler=scheduler, filename="ckpt.pt"
        )


if __name__ == "__main__":
    unittest.main()
