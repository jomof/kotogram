import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

# pylint: disable=no-name-in-module
from kotogram.model import ModelConfig
from train.config import CheckpointConfig, DataLoaderConfig, KCConfig, TrainerConfig
from train.kc import KcFamilyId
from train.models import StyleClassifierWithKC
from train.trainer import KCTrainer


class MockBatch:
    def __init__(self, feature_inputs, attention_mask):
        self.feature_inputs = feature_inputs
        self.attention_mask = attention_mask

        # Add dummy attributes accessed by trainer
        self.formality_value = torch.zeros(attention_mask.size(0))  # 1D for CE loss
        self.formality_pragmatic = torch.zeros(attention_mask.size(0), dtype=torch.long)
        self.gender_value = torch.zeros(attention_mask.size(0))  # 1D for CE loss
        self.gender_pragmatic = torch.zeros(attention_mask.size(0), dtype=torch.long)
        self.grammaticality_labels = torch.zeros(
            attention_mask.size(0), dtype=torch.long
        )
        self.register_labels = torch.zeros(attention_mask.size(0), 14)  # [B, 14]
        batch_size = attention_mask.size(0)
        self.kc_targets = [{KcFamilyId.BAG_READING_GRAM: []} for _ in range(batch_size)]


class MockKCDecoders(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Need decoders dict for bias delta tracking in trainer
        self.decoders = nn.ModuleDict(
            {
                KcFamilyId.BAG_READING_GRAM.name.lower(): nn.Linear(
                    in_features, out_features
                )
            }
        )

    def forward(self, x: torch.Tensor) -> dict:
        return {name: decoder(x) for name, decoder in self.decoders.items()}


class MockModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.last_k_budget = None
        self.last_long_mask = None
        self.kc_decoders = MockKCDecoders(config.kc_vocab_size, 10)  # dummy wrapper
        self.kc_head = nn.Linear(10, config.kc_vocab_size)  # dummy

    def forward(self, *args, **kwargs):
        if kwargs.get("mode") == "kc":
            self.last_k_budget = kwargs.get("k_budget")
            self.last_long_mask = kwargs.get("long_sentence_mask")

            # Return dummy outputs required by trainer
            batch_size = args[0]["input_ids_surface"].size(0)
            top_k = self.config.kc_topk
            vocab_size = self.config.kc_vocab_size

            # Ensure consistency for diagnostics
            kc_logits_raw = torch.randn(batch_size, vocab_size, requires_grad=True)
            kc_probs = torch.sigmoid(kc_logits_raw)

            topk_vals, topk_inds = torch.topk(kc_probs, top_k)

            # Zero out beyond budget if k_budget provided (mimic behavior for consistency)
            if self.last_k_budget is not None:
                mask = torch.arange(top_k).expand(
                    batch_size, top_k
                ) < self.last_k_budget.unsqueeze(1)
                topk_vals = topk_vals * mask.float()

            sparse = torch.zeros_like(kc_probs)
            sparse.scatter_(1, topk_inds, topk_vals)

            return {
                "kc_logits_raw": kc_logits_raw,
                "kc_logits_effective": kc_logits_raw,
                "kc_logits": kc_logits_raw,  # same
                "kc_probs": kc_probs,
                "topk_vals": topk_vals,
                "topk_inds": topk_inds,
                "sparse_activations": sparse,
                "target_logits": {
                    KcFamilyId.BAG_READING_GRAM.name.lower(): torch.randn(
                        batch_size, 10, requires_grad=True
                    )
                },
                "logits_usage": torch.randn(batch_size, vocab_size, requires_grad=True),
            }
        return {}

    # pylint: disable=unused-argument
    def to(self, device, *args, **kwargs):
        return self


class TestKCAdaptiveBudget(unittest.TestCase):
    def setUp(self):
        self.kc_config = KCConfig(
            sparsity_weight=0.01,
            entropy_floor=0.5,
            collapse_weight_thawed=0.1,
            freeze_encoder_epochs=1,
        )

        self.trainer_config = TrainerConfig(
            batch_size=4,
            grad_accum_steps=1,
            device="cpu",
            checkpoint=CheckpointConfig(),
            kc_target_specs={KcFamilyId.BAG_READING_GRAM: 10},
        )
        self.dl_config = DataLoaderConfig(
            num_workers=0, pin_memory=False, persistent_workers=False
        )

    def test_adaptive_k_calculation(self):
        # ModelConfig needs vocab_sizes
        model_config = ModelConfig(
            vocab_sizes={"surface": 100},
            kc_topk=8,
            kc_vocab_size=100,
        )
        model_config.kc_temperature = 1.0  # Needed by trainer

        model = MockModel(model_config)
        dataset = MagicMock()
        dataset.tokenizer = MagicMock()

        # Setup len BEFORE KCTrainer init which calls DataLoader which calls len()
        dataset.__len__.return_value = 3

        # Mock filter_by_grammaticality return
        dataset.filter_by_grammaticality.return_value = dataset

        with patch.object(KCTrainer, "_create_optimizer"):
            trainer = KCTrainer(
                model=model,
                dataset=dataset,
                config=self.trainer_config,
                dl_config=self.dl_config,
                kc_config=self.kc_config,
            )

        # Create a batch with varying lengths
        # 1. Very short (<=3) -> min_k=2
        # 2. Medium (10) -> alpha*10 = 4
        # 3. Long (30) -> alpha*30 = 12 -> clamped to max_k=8

        input_ids = torch.zeros(3, 40, dtype=torch.long)

        # Short: 2 tokens (indices 1, 2)
        input_ids[0, :2] = 1

        # Medium: 10 tokens
        input_ids[1, :10] = 1

        # Long: 30 tokens
        input_ids[2, :30] = 1

        mask = torch.zeros(3, 40)
        mask[0, :2] = 1
        mask[1, :10] = 1
        mask[2, :30] = 1

        batch = MockBatch(
            feature_inputs={"input_ids_surface": input_ids},
            attention_mask=mask,
        )

        # Mock loader
        loader = MagicMock()
        loader.__iter__.return_value = iter([batch])
        loader.batch_size = 4
        trainer.data_loader = loader
        # dataset.__len__ is already set above

        # Patch optimizer and backward to avoid errors
        trainer.optimizer = MagicMock()
        trainer.optimizer.param_groups = [{"lr": 0.0}, {"lr": 0.0}]

        # Run 1 epoch
        with patch.object(trainer, "_perform_optimizer_step"):
            trainer.train_epoch(0)

        # Check last k_budget
        k_budget = model.last_k_budget
        self.assertIsNotNone(k_budget)

        # Expected k:
        # 0: len=2. alpha=0.4. ceil=1. min_k=2. -> 2
        # 1: len=10. alpha=0.4. ceil=4. -> 4
        # 2: len=30. alpha=0.55. ceil=17. max_k=16. -> 16

        expected = torch.tensor([2, 4, 16], dtype=torch.long)
        self.assertTrue(
            torch.equal(k_budget, expected), f"Expected {expected}, got {k_budget}"
        )

        # Check long_sentence_mask
        # 0: len=2 -> False
        # 1: len=10 -> False
        # 2: len=30 -> True (>=20)

        long_mask = model.last_long_mask
        expected_mask = torch.tensor([False, False, True], dtype=torch.bool)
        self.assertTrue(
            torch.equal(long_mask, expected_mask),
            f"Expected {expected_mask}, got {long_mask}",
        )

    def test_model_forward_kc_masking(self):
        # Test real logic logic for variable k
        config = ModelConfig(
            vocab_sizes={"surface": 100},
            kc_topk=8,
            kc_vocab_size=100,
            field_embed_dims={"surface": 16},
        )

        # Patch FEATURE_FIELDS to only allow 'surface' to suffice for get_embeddings
        with patch("kotogram.model.FEATURE_FIELDS", ["surface"]):
            model = StyleClassifierWithKC(config)
            model.to("cpu")

            batch_size = 2
            inputs = {"input_ids_surface": torch.ones(batch_size, 10, dtype=torch.long)}
            mask = torch.ones(batch_size, 10)
            k_budget = torch.tensor([2, 5], dtype=torch.long)  # Two different budgets

            outputs = model.forward_kc(
                inputs, attention_mask=mask, k_budget=k_budget, temperature=1.0
            )

            topk_vals = outputs["topk_vals"]  # (B, 8)

            # Check sample 0 (budget 2)
            # Indices 2..7 should be zero
            self.assertTrue((topk_vals[0, 2:] == 0).all())
            self.assertTrue((topk_vals[0, :2] > 0).all())  # Assuming >0 prob, likely

            # Check sample 1 (budget 5)
            # Indices 5..7 should be zero
            self.assertTrue((topk_vals[1, 5:] == 0).all())
            self.assertTrue((topk_vals[1, :5] > 0).all())


if __name__ == "__main__":
    unittest.main()
