import unittest

import torch

from kotogram.model import ModelConfig
from kotogram.tokenizer import Tokenizer
from train.config import DataLoaderSettings, KCConfig, TrainerConfig
from train.dataset import StyleDataset
from train.models import TrainingClassifier
from train.trainer import KCTrainer
from train.types import Sample


class MockDataset(StyleDataset):
    def __init__(self, samples):
        # pylint: disable=super-init-not-called
        self._samples = samples
        self.tokenizer = Tokenizer()
        # Add required attributes for style oversampling
        self.indices = torch.arange(len(samples))
        self.labels = {
            "f_val": torch.tensor(
                [s.formality_value for s in samples], dtype=torch.float32
            ),
            "g_val": torch.tensor(
                [s.gender_value for s in samples], dtype=torch.float32
            ),
        }

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]

    def filter_by_grammaticality(self, label: int = 1):
        filtered = [s for s in self._samples if s.grammaticality_label == label]
        return MockDataset(filtered)

    @property
    def samples(self):
        # Expose samples for test verification
        return self._samples


class TestPretrainDataFiltering(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()

        # Create mock samples
        self.grammatic_sample = Sample(
            feature_ids={
                f: [1] * 5
                for f in [
                    "surface",
                    "pos",
                    "compound_1",
                    "compound_2",
                    "conjugated_type",
                    "conjugated_form",
                    "lemma",
                ]
            },
            formality_value=1.0,
            formality_pragmatic=1,
            gender_value=0.0,
            gender_pragmatic=1,
            grammaticality_label=1,  # Grammatic
            register_labels=[1],
            original_sentence="こんにちは",
            kotogram=[],
            idx=0,
            kc_targets={},
        )

        self.agrammatic_sample = Sample(
            feature_ids={
                f: [1] * 5
                for f in [
                    "surface",
                    "pos",
                    "compound_1",
                    "compound_2",
                    "conjugated_type",
                    "conjugated_form",
                    "lemma",
                ]
            },
            formality_value=1.0,
            formality_pragmatic=1,
            gender_value=0.0,
            gender_pragmatic=1,
            grammaticality_label=0,  # Agrammatic
            register_labels=[1],
            original_sentence="こんにちは *",
            kotogram=[],
            idx=1,
            kc_targets={},
        )

        self.config = TrainerConfig(
            learning_rate=1e-4,
            batch_size=2,
            epochs=1,
            dataloader=DataLoaderSettings(num_workers=0, prefetch_factor=None),
            device="cpu",
        )

        self.model_config = ModelConfig(
            vocab_sizes={
                f: 100
                for f in [
                    "surface",
                    "pos",
                    "compound_1",
                    "compound_2",
                    "conjugated_type",
                    "conjugated_form",
                    "lemma",
                ]
            },
            num_formality_pragmatic_classes=3,
            num_gender_pragmatic_classes=3,
            num_grammaticality_classes=2,
            d_model=64,
            hidden_dim=128,
            num_layers=1,
            num_heads=2,
        )
        self.model = TrainingClassifier(self.model_config)

    def test_kc_trainer_retains_agrammatic(self):
        dataset = MockDataset([self.grammatic_sample, self.agrammatic_sample])
        kc_config = KCConfig(kl_sparse_weight=0.01, freeze_encoder_epochs=1)
        dl_config = self.config.resolve_dataloader_config(torch.device("cpu"))

        agrammatic_count = sum(
            1 for s in dataset.samples if s.grammaticality_label == 0
        )
        self.assertEqual(
            agrammatic_count,
            1,
            "Dataset should contain one agrammatic sample for testing",
        )

        trainer = KCTrainer(
            self.model,
            dataset,
            self.config,
            dl_config=dl_config,
            kc_config=kc_config,
        )

        def has_agrammatic(dataset):
            return any(s.grammaticality_label == 0 for s in dataset.samples)

        self.assertTrue(
            has_agrammatic(trainer.dataset),
            "KCTrainer's dataset should contain agrammatic samples",
        )
        self.assertEqual(
            len(trainer.dataset.samples),
            2,
            "KCTrainer should retain both grammatic and agrammatic samples",
        )


if __name__ == "__main__":
    unittest.main()
