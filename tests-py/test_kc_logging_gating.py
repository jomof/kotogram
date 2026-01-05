import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

from rich.console import Console

from kotogram.model import ModelConfig
from train import display
from train.config import CheckpointConfig, DataLoaderConfig, KCConfig, TrainerConfig
from train.trainer import KCTrainer


class MockFamilyStats:
    def __init__(self, **kwargs):
        self.num_pos = kwargs.get("num_pos", 0)
        self.num_total_labels = 1000
        p50 = kwargs.get("p50", 0)
        p90 = kwargs.get("p90", 0)

        self.p50 = p50
        self.p90 = p90

        # Reservoir sampling
        self.card_reservoir = [0] * 10
        self.card_reservoir[5] = p50
        self.card_reservoir[9] = p90

        self.sum_nll = kwargs.get("sum_nll", 0)
        self.count_nll = kwargs.get("count_nll", 0)


class TestKCLoggingGatingReal(unittest.TestCase):
    def setUp(self):
        self.kc_config = KCConfig(log_level="minimal", freeze_encoder_epochs=1)
        self.trainer_config = TrainerConfig(checkpoint=CheckpointConfig())
        self.dl_config = DataLoaderConfig(
            num_workers=0, pin_memory=False, persistent_workers=False
        )

        model_config = ModelConfig(vocab_sizes={"surface": 100}, kc_topk=10)
        self.model = MagicMock()
        self.model.config = model_config

        self.dataset = MagicMock()
        self.dataset.tokenizer.field_vocabs = {}
        self.dataset.__len__.return_value = 1
        self.dataset.filter_by_grammaticality.return_value = self.dataset

        with patch("train.trainer.KCTrainer._create_optimizer"):
            self.trainer = KCTrainer(
                model=self.model,
                dataset=self.dataset,
                config=self.trainer_config,
                dl_config=self.dl_config,
                kc_config=self.kc_config,
            )
            self.trainer.optimizer = MagicMock()
            self.trainer.optimizer.param_groups = [{"lr": 0.001}, {"lr": 0.001}]

    def run_epoch_with_stats(self, families, epoch=0):
        with patch("train.trainer.KCEpochDiag") as mock_diag_cls:
            diag_instance = mock_diag_cls.return_value
            diag_instance.families = families

            # Empty data loader so no updates happen
            self.trainer.data_loader = []

            # Mock display.console
            mock_out = StringIO()
            with patch.object(
                display,
                "console",
                Console(file=mock_out, force_terminal=False, color_system=None),
            ):
                self.trainer.train_epoch(epoch)

            return mock_out.getvalue()

    def test_support_gating_suppression(self):
        # Low support family
        stats = MockFamilyStats(num_pos=5, p50=2, p90=5, sum_nll=0)
        fams = {"test_fam": stats}

        out = self.run_epoch_with_stats(fams, epoch=2)

        self.assertNotIn("test_fam: pp/pn 0/0", out)
        self.assertNotIn("KC WARN (actionable)", out)

    def test_persistence_logic(self):
        # Good support, Fail dNLL
        # rate=0.1, bias~0.32. nll=2.0 -> dnll=1.68 > 0.2
        stats = MockFamilyStats(num_pos=100, p50=20, p90=30, sum_nll=200, count_nll=100)
        fams = {"pos_fam": stats}

        # Epoch 2 (1st fail) -> Persistence=1. No warn.
        out1 = self.run_epoch_with_stats(fams, epoch=2)
        self.assertNotIn("pos_fam: dNLL", out1)

        # Epoch 3 (2nd fail) -> Persistence=2. Warn.
        out2 = self.run_epoch_with_stats(fams, epoch=3)
        self.assertIn("pos_fam: dNLL", out2)
        self.assertIn("KC WARN (actionable)", out2)

    def test_sparse_family_gating(self):
        # Sparse family logic: Warn only if thawed + 1 (Conservative)
        stats = MockFamilyStats(num_pos=100, p50=20, p90=30, sum_nll=200, count_nll=100)
        fams = {"ngram_test": stats}

        # Epoch 1 (Thawed but early) -> Suppress
        out1 = self.run_epoch_with_stats(fams, epoch=1)
        self.assertNotIn("ngram_test", out1)

        # Epoch 2 -> Eligible. Run twice for persistence.
        self.run_epoch_with_stats(fams, epoch=2)  # Fail 1
        out2 = self.run_epoch_with_stats(fams, epoch=2)  # Fail 2

        self.assertIn("ngram_test: dNLL", out2)


if __name__ == "__main__":
    unittest.main()
