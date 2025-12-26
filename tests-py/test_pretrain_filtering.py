import unittest

from kotogram.model import ModelConfig
from kotogram.tokenizer import Tokenizer
from train.config import TrainerConfig
from train.dataset import StyleDataset
from train.trainer import (
    KCTrainer,
    MLMTrainer,
    StyleClassifierWithMLM,
)
from train.types import Sample


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
                    "pos_detail1",
                    "pos_detail2",
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
        )

        self.agrammatic_sample = Sample(
            feature_ids={
                f: [1] * 5
                for f in [
                    "surface",
                    "pos",
                    "pos_detail1",
                    "pos_detail2",
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
        )

        self.config = TrainerConfig(
            learning_rate=1e-4,
            batch_size=2,
            epochs=1,
            device="cpu",
            world_size=1,
            local_rank=0,
        )

        self.model_config = ModelConfig(
            vocab_sizes={
                f: 100
                for f in [
                    "surface",
                    "pos",
                    "pos_detail1",
                    "pos_detail2",
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
            kc_enabled=True,
            kc_target_specs={"lemma": 100},
        )
        self.model = StyleClassifierWithMLM(self.model_config)

    def test_mlm_trainer_filtering(self):
        dataset = StyleDataset(
            [self.grammatic_sample, self.agrammatic_sample], self.tokenizer
        )

        agrammatic_count = sum(
            1 for s in dataset.samples if s.grammaticality_label == 0
        )
        self.assertEqual(
            agrammatic_count,
            1,
            "Dataset should contain one agrammatic sample for testing",
        )

        trainer = MLMTrainer(self.model, dataset, self.config)

        def has_agrammatic(dataset):
            return any(s.grammaticality_label == 0 for s in dataset.samples)

        self.assertFalse(
            has_agrammatic(trainer.dataset),
            "MLMTrainer's dataset should NOT contain agrammatic samples",
        )
        self.assertEqual(
            len(trainer.dataset.samples),
            1,
            "MLMTrainer should have filtered out the one agrammatic sample",
        )

    def test_kc_trainer_filtering(self):
        dataset = StyleDataset(
            [self.grammatic_sample, self.agrammatic_sample], self.tokenizer
        )
        kc_config = {"sparsity_weight": 0.01, "freeze_encoder_epochs": 1}

        agrammatic_count = sum(
            1 for s in dataset.samples if s.grammaticality_label == 0
        )
        self.assertEqual(
            agrammatic_count,
            1,
            "Dataset should contain one agrammatic sample for testing",
        )

        trainer = KCTrainer(self.model, dataset, self.config, kc_config=kc_config)

        def has_agrammatic(dataset):
            return any(s.grammaticality_label == 0 for s in dataset.samples)

        self.assertFalse(
            has_agrammatic(trainer.dataset),
            "KCTrainer's dataset should NOT contain agrammatic samples",
        )
        self.assertEqual(
            len(trainer.dataset.samples),
            1,
            "KCTrainer should have filtered out the one agrammatic sample",
        )


if __name__ == "__main__":
    unittest.main()
