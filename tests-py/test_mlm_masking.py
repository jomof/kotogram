import os
import sys

import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.trainer import create_mlm_batch


def test_mlm_grammar_only():
    """Test that surface/lemma are potentially hidden and others are standard MLM."""
    torch.manual_seed(42)

    # Mock batch
    batch_size = 2
    seq_len = 50
    feature_fields = [
        "surface",
        "pos",
        "pos_detail1",
        "pos_detail2",
        "pos_detail3",
        "conjugated_type",
        "conjugated_form",
        "lemma",
        "base_orth",
        "reading",
    ]

    batch = {"attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long)}

    # Initialize random inputs
    for field in feature_fields:
        batch[f"input_ids_{field}"] = torch.randint(4, 100, (batch_size, seq_len))

    # Run masking with standard prob
    mask_prob = 0.15
    mask_token_id = 3
    masked_batch = create_mlm_batch(
        batch,
        mask_prob=mask_prob,
        mask_token_id=mask_token_id,
        vocab_sizes={f: 1000 for f in feature_fields},
    )

    # 1. Verify HIDDEN fields (surface, lemma)
    for field in ["surface", "lemma"]:
        # Inputs should be ALL masked (ID 3) where attention mask is 1
        current_ids = masked_batch[f"input_ids_{field}"]
        attention_mask = batch["attention_mask"].bool()

        # Check that ALL attended tokens are MASKED (3)
        # Note: In the plan we said "Override input_ids to be MASK_TOKEN at ALL positions".
        # Let's assume we respect attention mask (padding stays padding).
        assert (current_ids[attention_mask] == mask_token_id).all(), (
            f"Field {field} should be 100% MASKED"
        )

        # Labels should be ALL -100 (ignored)
        labels = masked_batch[f"mlm_labels_{field}"]
        assert (labels == -100).all(), (
            f"Field {field} labels should be all -100 (ignored)"
        )

    # 2. Verify TRAINED fields (pos, etc.)
    for field in ["pos", "pos_detail1", "conjugated_type"]:
        # Should have SOME valid labels targeting original values
        labels = masked_batch[f"mlm_labels_{field}"]
        valid_labels = labels != -100
        assert valid_labels.any(), f"Field {field} should have some targets"

        # Inputs should be mostly original, some masked
        original = batch[f"input_ids_{field}"]
        current = masked_batch[f"input_ids_{field}"]

        # Check that it's NOT 100% masked
        assert not (current[attention_mask] == mask_token_id).all(), (
            f"Field {field} should NOT be 100% masked"
        )

        # Check that some masking/change happened (statistically likely with 15% of 20 tokens)
        # batch 2 * 10 = 20 tokens. 15% = 3 tokens.
        if (original != current).sum() == 0:
            print(f"Warning: Field {field} had no changes (unlikely but possible)")


if __name__ == "__main__":
    test_mlm_grammar_only()
    print("Test passed!")
