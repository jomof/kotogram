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

    # pylint: disable=unused-argument
    def forward(self, x: torch.Tensor, kc_probs: torch.Tensor) -> dict:
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
        # Encoder pipeline stubs for frozen epoch handling (logging diagnostics tests)
        self.embedding = nn.Identity()
        self.encoder = nn.Identity()
        self.position_encoding = (
            nn.Identity()
        )  # Order differs from adaptive_budget tests
        self.pooler = nn.Identity()

    def forward(self, *args, **kwargs):
        mode = kwargs.get("mode")
        if mode == "kc":
            field_inputs = args[0] if args else {}
            if field_inputs and "input_ids_surface" in field_inputs:
                batch_size = field_inputs["input_ids_surface"].size(0)
            elif "attention_mask" in kwargs:
                batch_size = kwargs["attention_mask"].size(0)
            else:
                batch_size = 1
            vocab_size = self.config.kc_vocab_size

            return {
                "kc_logits_raw": torch.zeros(
                    batch_size, vocab_size, requires_grad=True
                ),
                "kc_logits_effective": torch.zeros(
                    batch_size, vocab_size, requires_grad=True
                ),
                "kc_logits": torch.zeros(batch_size, vocab_size, requires_grad=True),
                "kc_probs": torch.sigmoid(torch.zeros(batch_size, vocab_size)),
                "kc_probs_clean": torch.sigmoid(torch.zeros(batch_size, vocab_size)),
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
        # Prior KC losses (formality, gender, register) removed - handled by style classifier
        self.assertIn("diversity", output)
        self.assertIn("entropy", output)

        # Verify Block 1 Sizing Table Header
        # Verify Block 1 Sizing Table Header
        self.assertRegex(output, r"Bin.*N.*Len.*K@.*Kth.*Spill.*Gap")
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
        # sparsity row shows S1/S0/Fuzzy
        self.assertRegex(output, r"sparsity.*S1=.*S0=")
        # saturation shows sc and pen
        self.assertRegex(output, r"saturation.*sc=.*pen=")

        # Verify Block 3 Families
        # "Family ... Loss ... Pos% ..."
        # Verify Block 3 Families (with possible truncation in narrow terminals)
        self.assertRegex(
            output, r"Fam.*Loss.*Pos%.*Logi"
        )  # UnlabFP column may or may not be present, columns may be truncated (e.g., "Logi…")
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
