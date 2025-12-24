import os
import sys

import torch

# Add project root to path to allow importing scripts
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kotogram.model import KCHead, ModelConfig
from kotogram.tokenizer import ALL_FEATURE_FIELDS
from scripts.train_style import (
    KCDecoder,
    KCTrainer,
    StyleClassifierWithMLM,
    TrainerConfig,
    create_kc_batch,
)


# Mock classes for Trainer dependencies
class MockTokenizer:
    def __init__(self):
        self.pad_id = 0
        self.mask_id = 3

    def get_vocab_sizes(self):
        return {f: 100 for f in ALL_FEATURE_FIELDS}


class MockDataset:
    def __init__(self):
        self.tokenizer = MockTokenizer()
        # Add mock samples for KCTrainer init check
        from dataclasses import dataclass

        @dataclass
        class MockSample:
            grammaticality_label: int

        self.samples = [MockSample(grammaticality_label=1) for _ in range(10)]

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        item = {
            "attention_mask": torch.tensor([1, 1, 1, 1], dtype=torch.long),
            "formality_value": 0.0,
            "formality_pragmatic": 0,
            "gender_value": 0.0,
            "gender_pragmatic": 0,
            "grammaticality_labels": 0,
            "register_labels": torch.zeros(5),
            "original_sentence": "test",
            "kotogram": None,
        }
        for f in ALL_FEATURE_FIELDS:
            item[f"input_ids_{f}"] = torch.tensor([1, 4, 5, 2], dtype=torch.long)
        return item


def test_kc_head_shapes():
    config = ModelConfig(
        vocab_sizes={"surface": 100}, kc_enabled=True, kc_vocab_size=64, kc_topk=4
    )
    kc_head = KCHead(config)

    batch_size = 2
    # KCHead likely expects config.d_model input size as per standard transformer heads?
    # Let's verify: In StyleClassifier, pooled output comes from get_encoder_output which is d_model (256).
    # Wait, KCHead definition uses config.hidden_dim usually?
    # kotogram/model.py says:
    # self.kc_head = KCHead(config)
    # And KCHead usually is nn.Linear(config.hidden_dim, ...).
    # But StyleClassifier hidden size is 512, d_model is 256.
    # The error "mat1 and mat2 shapes cannot be multiplied (2x512 and 256x64)"
    # means input was 512 (hidden_dim), but mat2 (weights) expects 256 (d_model).
    # So KCHead expects d_model (256).
    # So we should provide input of size d_model.

    input_dim = config.d_model  # 256
    pooled_output = torch.randn(batch_size, input_dim)

    output = kc_head(pooled_output)
    assert output.shape == (batch_size, 64)


def test_kc_decoder_shapes():
    kc_vocab_size = 64
    target_specs = {"pos": 50, "lemma": 200}
    decoder = KCDecoder(kc_vocab_size, target_specs)

    batch_size = 2
    kc_activations = torch.randn(batch_size, kc_vocab_size)

    logits_dict = decoder(kc_activations)
    assert "pos" in logits_dict
    assert "lemma" in logits_dict
    assert logits_dict["pos"].shape == (batch_size, 50)
    assert logits_dict["lemma"].shape == (batch_size, 200)


def test_style_classifier_with_mlm_kc_mode():
    vocab_sizes = {f: 100 for f in ALL_FEATURE_FIELDS}
    config = ModelConfig(
        vocab_sizes=vocab_sizes,
        kc_enabled=True,
        kc_vocab_size=64,
        kc_target_specs={"pos": 50},
    )
    model = StyleClassifierWithMLM(config)

    batch_size = 2
    seq_len = 10
    field_inputs = {}
    for f in ALL_FEATURE_FIELDS:
        field_inputs[f"input_ids_{f}"] = torch.randint(0, 100, (batch_size, seq_len))

    attention_mask = torch.ones(batch_size, seq_len)

    # Cast to ensure mypy knows it's the right type if needed, though runtime doesn't care
    output = model(field_inputs, attention_mask, mode="kc")

    # Check that output contains expected keys
    # Note: TypedDict might not support 'in' checks at runtime if it was a real TypedDict,
    # but here output is usually just a Dict.
    assert "kc_probs" in output
    assert "target_logits" in output
    assert output["kc_probs"].shape == (batch_size, 64)
    # Target head "pos" will use the decoder
    assert output["target_logits"]["pos"].shape == (batch_size, 50)


def test_create_kc_batch():
    tokenizer = MockTokenizer()
    batch = {
        "input_ids_pos": torch.tensor(
            [[1, 10, 11, 2], [1, 12, 0, 0]]
        ),  # Example with padding
        "input_ids_surface": torch.tensor(
            [[1, 100, 101, 2], [1, 102, 0, 0]]
        ),  # Surface needed for batch size
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
    }
    target_specs = {"pos": 50}

    targets = create_kc_batch(batch, tokenizer, target_specs)

    assert "kc_targets_pos" in targets
    target = targets["kc_targets_pos"]
    assert target.shape == (2, 50)

    assert target[0, 10] == 1.0
    assert target[0, 11] == 1.0
    assert target[0, 1] == 0.0  # Special token ignored

    assert target[1, 12] == 1.0
    assert target[1, 0] == 0.0  # Padding ignored


def test_kc_trainer_init():
    config = ModelConfig(
        vocab_sizes={f: 100 for f in ALL_FEATURE_FIELDS},
        kc_enabled=True,
        kc_vocab_size=32,
        kc_target_specs={},
    )
    model = StyleClassifierWithMLM(config)
    dataset = MockDataset()
    trainer_config = TrainerConfig(batch_size=2, device="cpu")
    kc_config = {"sparsity_weight": 0.01}

    trainer = KCTrainer(model, dataset, trainer_config, kc_config)
    assert trainer.kc_sparsity_weight == 0.01
