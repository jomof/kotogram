import re
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

import torch
from rich.console import Console
from torch import nn

# pylint: disable=no-name-in-module
from kotogram.model import ModelConfig
from train import display
from train.config import (
    CheckpointConfig,
    DataLoaderConfig,
    KCConfig,
    KcFamilyId,
    TrainerConfig,
)
from train.trainer import KCTrainer


class DiagMockBatch:
    def __init__(self, feature_inputs, attention_mask):
        self.feature_inputs = feature_inputs
        self.attention_mask = attention_mask

        # Add dummy attributes accessed by trainer (reordered)
        self.grammaticality_labels = torch.zeros(
            attention_mask.size(0), dtype=torch.long
        )
        self.formality_pragmatic = torch.zeros(attention_mask.size(0), dtype=torch.long)
        self.formality_value = torch.zeros(attention_mask.size(0))  # 1D for CE loss
        self.gender_pragmatic = torch.zeros(attention_mask.size(0), dtype=torch.long)
        self.gender_value = torch.zeros(attention_mask.size(0))  # 1D for CE loss

        self.register_labels = torch.zeros(attention_mask.size(0), 14)  # [B, 14]
        batch_size = attention_mask.size(0)
        # Simulate list of Sample.kc_targets (Dict[KcFamilyId, List[int]])
        self.kc_targets = [{KcFamilyId.BAG_POS: [0]} for _ in range(batch_size)]

    def __getitem__(self, key):
        return getattr(self, key)


class MockDecoders(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Need decoders dict for bias delta tracking in trainer
        self.decoders: dict = {}

    def forward(self, x: torch.Tensor) -> dict:
        return {
            KcFamilyId.BAG_POS.name.lower(): torch.zeros(
                x.size(0), 10, requires_grad=True
            )
        }


class MockModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.kc_decoders = MockDecoders()
        self.kc_head = nn.Linear(10, 100)
        self.embedding = nn.Linear(10, 10)
        self.encoder = nn.Linear(10, 10)

    def forward(self, *args, **kwargs):
        if kwargs.get("mode") == "kc":
            batch_size = args[0]["input_ids_surface"].size(0)
            vocab_size = self.config.kc_vocab_size

            # Create predictable outputs for testing metrics
            # 1. k_eff check: set sparse_activations > 0 based on sentence index
            # Index 0: len=2 -> make 1 activation
            # Index 1: len=10 -> make 4 activations
            # Index 2: len=30 -> make 8 activations

            sparse = torch.zeros(batch_size, vocab_size)
            topk_vals = torch.zeros(batch_size, vocab_size)
            topk_inds = torch.zeros(batch_size, vocab_size, dtype=torch.long)

            # We can't easily know index, but we can use batch size if fixed.
            # Let's assume the batch order matches the setup.
            # row 0
            sparse[0, 0] = 0.9
            topk_vals[0, 0] = 0.9
            topk_inds[0, 0] = 0

            # row 1
            if batch_size > 1:
                sparse[1, 0:4] = 0.8
                topk_vals[1, 0:4] = 0.8
                topk_inds[1, 0:4] = torch.arange(4)

            # row 2
            if batch_size > 2:
                sparse[2, 0:8] = 0.7
                topk_vals[2, 0:8] = 0.7
                topk_inds[2, 0:8] = torch.arange(8)

            return {
                "kc_logits_raw": torch.zeros(
                    batch_size, vocab_size, requires_grad=True
                ),
                "kc_logits_effective": torch.zeros(
                    batch_size, vocab_size, requires_grad=True
                ),
                "kc_logits": torch.zeros(batch_size, vocab_size, requires_grad=True),
                "kc_probs": torch.sigmoid(torch.zeros(batch_size, vocab_size)),
                "topk_vals": topk_vals,
                "topk_inds": topk_inds,
                "sparse_activations": sparse,
                "target_logits": {
                    KcFamilyId.BAG_POS.name.lower(): torch.zeros(
                        batch_size, 10, requires_grad=True
                    )
                },
                "logits_usage": torch.zeros(batch_size, vocab_size, requires_grad=True),
            }
        return {}


class TestKCLoggingDiagnostics(unittest.TestCase):
    def setUp(self):
        self.kc_config = KCConfig(
            freeze_encoder_epochs=0,
        )
        self.trainer_config = TrainerConfig(
            batch_size=3,
            device="cpu",
            checkpoint=CheckpointConfig(),
            kc_target_specs={KcFamilyId.BAG_POS: 10},
        )
        self.dl_config = DataLoaderConfig(
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None,
        )

    def test_epoch_logging_format(self):
        print("\n[Test] Running test_epoch_logging_format...")
        m_cfg = ModelConfig(
            vocab_sizes={"surface": 100},
            kc_topk=10,
            kc_vocab_size=100,
        )
        m_cfg.kc_temperature = 1.0

        mdl = MockModel(m_cfg)
        dataset = MagicMock()
        dataset.tokenizer = MagicMock()
        dataset.__len__.return_value = 3
        dataset.filter_by_grammaticality.return_value = dataset

        # input_ids surface
        inputs = torch.zeros(3, 40, dtype=torch.long)
        # Helper mask initialization
        mask = torch.zeros(3, 40)

        # Detailed lengths setup:
        # lengths: 2, 10, 30
        mask[0, :2] = 1
        mask[1, :10] = 1
        mask[2, :30] = 1

        # Create batch
        batch = DiagMockBatch(
            feature_inputs={"input_ids_surface": inputs},
            attention_mask=mask,
        )

        with patch.object(KCTrainer, "_create_optimizer"):
            trainer = KCTrainer(
                dataset=dataset,
                model=mdl,
                config=self.trainer_config,
                dl_config=self.dl_config,
                kc_config=self.kc_config,
            )

        mock_loader = MagicMock()
        mock_loader.__iter__.return_value = iter([batch])
        mock_loader.batch_size = 3
        trainer.data_loader = mock_loader
        trainer.optimizer = MagicMock()
        trainer.optimizer.param_groups = [{"lr": 0.0}, {"lr": 0.0}]

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            # Patch rich console to be plain text and use mock_stdout
            with patch.object(
                display,
                "console",
                Console(file=mock_stdout, force_terminal=False, color_system=None),
            ):
                with patch.object(trainer, "_perform_optimizer_step"):
                    trainer.train_epoch(0)

            output = mock_stdout.getvalue()
            # Strip ANSI codes for regex matching
            output = re.sub(r"\x1b\[[0-9;]*m", "", output)

        print(f"[Captured Output]\n{output}")

        # Verify the One-Line Summary exists and follows format
        # KC EP1 Thawed loss=
        # kEff=4.33[1.0,4.0,8.0] (approx)
        # len=14.0[2.0,10.0,30.0]
        # corrLxK should be high (perfect correlation 1.0)

        # Regex to match the core structure
        # KC EP1 Thawed loss=.* kEff=.*\[.*\] len=.*\[.*\] corrLxK=.*

        # Verify the Block 0 Header (new format: loss breakdown with epoch info)
        self.assertIn("KC EP1 Thawed Loss Breakdown:", output)
        # Loss breakdown is now on separate lines
        self.assertIn("struct", output)
        self.assertIn("gap", output)
        # Prior KC losses (formality, gender, register) removed - handled by style classifier
        self.assertIn("diversity", output)
        self.assertIn("load_bal", output)

        # Verify Block 1 Sizing Table Header
        # Verify Block 1 Sizing Table Header
        self.assertRegex(
            output, r"Bin.*N.*Len.*K\(Avg\|P10/50/90\).*K/Len.*TailMask.*Keff.*Diff"
        )
        # Verify some row content (e.g. 1-3 bin)
        # Note: bin logic depends on input length.
        # Mock batch had lengths 2, 10, 30.
        # Bins: "1-3" (for len 2), "8-15" (for len 10), "16-31" (for len 30).
        self.assertRegex(output, r"1-3\s+1\s+2\.0")
        self.assertRegex(output, r"8-15\s+1\s+10\.0")

        # Verify Block 2 Activation stats are now in loss table rows
        # AvgP is on diversity line
        self.assertRegex(output, r"diversity.*AvgP=")
        # PMax is on collapse line
        self.assertRegex(output, r"collapse.*PMax=")
        # sparsity shows density and K
        self.assertRegex(output, r"sparsity.*Dens=.*K=")
        # saturation shows sc and pen
        self.assertRegex(output, r"saturation.*sc=.*pen=")

        # Verify Block 3 Families
        # "Family ... Loss ... Pos% ..."
        # Verify Block 3 Families
        self.assertRegex(output, r"Family.*Loss.*Pos%.*Logit")
        self.assertRegex(output, r"Gap.*Msk%")

        # Verify Labels Line
        # "Labels: ..."
        # self.assertRegex(output, r"Labels: .*") # Might be empty if no labels in mock

        # Old stuff should be gone
        self.assertNotRegex(output, r"\[KC\] epoch=")
        self.assertNotRegex(output, r"KCdiag fam=")
        self.assertNotRegex(output, r"KC Health:")


if __name__ == "__main__":
    unittest.main()
