import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from kotogram.model import ModelConfig, StyleClassifier
from train.trainer import KCTrainer
from train.types import TrainingMetrics


class TestInternalVariations(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_restore_from_checkpoint_variations(self):
        # Mock dependencies for KCTrainer
        model = MagicMock(spec=StyleClassifier)
        model.config = ModelConfig(vocab_sizes={})

        # Explicitly attach sub-mocks that spec might miss or to be safe
        model.kc_head = MagicMock()
        model.kc_head.parameters.return_value = []
        model.parameters.return_value = []
        model.embedding = MagicMock()
        model.embedding.parameters.return_value = []
        model.encoder = MagicMock()
        model.encoder.parameters.return_value = []

        # Optional decoders
        model.kc_decoders = MagicMock()
        model.kc_decoders.parameters.return_value = []

        dataset = MagicMock()
        dataset.tokenizer.field_vocabs = {}
        dataset.filter_by_grammaticality.return_value = dataset
        dataset.__len__.return_value = 10

        config = MagicMock()
        config.device = "cpu"
        config.batch_size = 2
        # fix set_num_threads issue
        config.hardware = MagicMock()
        config.hardware.cpu_threads = 1
        config.hardware.interop_threads = 1

        config.resolve_dataloader_config.return_value = MagicMock(
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None,
        )

        dl_config = MagicMock()
        dl_config.num_workers = 0
        dl_config.prefetch_factor = None
        dl_config.pin_memory = False
        dl_config.persistent_workers = False

        kc_config = MagicMock()
        kc_config.sparsity_weight = 0.0
        kc_config.freeze_encoder_epochs = 0

        kc_config.kc_grad_cap = 1.0

        # Instantiate trainer
        trainer = KCTrainer(model, dataset, config, dl_config, kc_config)

        # Call 1
        path1 = os.path.join(self.test_dir, "m1")
        os.makedirs(path1)
        trainer.restore_from_checkpoint(path1)

        # Call 2
        path2 = os.path.join(self.test_dir, "m2")
        os.makedirs(path2)
        trainer.restore_from_checkpoint(path2)

    def test_init_structural_decoder_biases_variations(self):
        # pylint: disable=protected-access
        # Create minimal mock trainer
        model = MagicMock(spec=StyleClassifier)
        model.config = ModelConfig(vocab_sizes={})
        model.config.kc_target_specs = {}

        # Explicitly attach kc_head and decoders
        model.kc_head = MagicMock()
        model.kc_head.parameters.return_value = []
        model.kc_decoders = MagicMock()
        model.embedding = MagicMock()
        model.embedding.parameters.return_value = []
        model.encoder = MagicMock()
        model.encoder.parameters.return_value = []

        dataset = MagicMock()
        dataset.tokenizer = MagicMock()
        dataset.filter_by_grammaticality.return_value = dataset
        dataset.__len__.return_value = 10

        config = MagicMock()
        config.device = "cpu"
        config.batch_size = 2
        config.hardware = MagicMock()
        config.hardware.cpu_threads = 1
        config.hardware.interop_threads = 1
        config.resolve_dataloader_config.return_value = MagicMock(num_workers=0)

        dl_config = MagicMock()
        dl_config.num_workers = 0
        dl_config.prefetch_factor = None
        dl_config.pin_memory = False
        dl_config.persistent_workers = False

        kc_config = MagicMock()
        kc_config.sparsity_weight = 0.0
        kc_config.freeze_encoder_epochs = 1

        kc_config.kc_grad_cap = 1.0

        trainer = KCTrainer(model, dataset, config, dl_config, kc_config)

        # Mock data loader to be empty so loop finishes immediately
        trainer.data_loader = []

        # Call 1
        trainer._init_structural_decoder_biases(num_batches=10)

        # Call 2
        trainer._init_structural_decoder_biases(num_batches=20)

    def test_training_metrics_update(self):
        stats = TrainingMetrics()
        loss_dict1 = {
            "loss": 1.0,
            "formality_loss": 0.1,
            "gender_loss": 0.1,
            "grammaticality_loss": 0.1,
            "register_loss": 0.1,
        }
        stats.update(loss_dict1, count=1)

        loss_dict2 = {
            "loss": 0.5,
            "formality_loss": 0.05,
            "gender_loss": 0.05,
            "grammaticality_loss": 0.05,
            "register_loss": 0.05,
        }
        stats.update(loss_dict2, count=2)


if __name__ == "__main__":
    unittest.main()
