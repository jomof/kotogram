"""Test that dense/sparse KC training paths are correctly routed."""

import torch

from train.dataset import create_kc_batch
from train.kc import ALL_KC_FAMILIES, KcFamilyId, is_family_sparse
from train.types import TrainingBatch


class DummyTokenizer:
    """Minimal tokenizer for testing."""

    unk_id = 1
    cls_id = 2


def _make_dummy_batch(
    batch_size: int, target_specs: dict[KcFamilyId, int]
) -> TrainingBatch:
    """Create a minimal batch with KC targets for testing."""
    # Create fake KC targets - each sample has a few positive IDs per family
    kc_targets = []
    for _ in range(batch_size):
        sample_targets: dict[KcFamilyId, list[int]] = {}
        for fid, _ in target_specs.items():
            # Put some random positive IDs (avoiding special IDs 0,1,2)
            sample_targets[fid] = [3, 4, 5]
        kc_targets.append(sample_targets)

    return TrainingBatch(
        feature_inputs={},
        attention_mask=torch.ones((batch_size, 10)),
        formality_value=torch.zeros(batch_size),
        formality_pragmatic=torch.zeros(batch_size, dtype=torch.long),
        gender_value=torch.zeros(batch_size),
        gender_pragmatic=torch.zeros(batch_size, dtype=torch.long),
        grammaticality_labels=torch.zeros(batch_size, dtype=torch.long),
        register_labels=torch.zeros((batch_size, 5)),
        original_sentence=[""] * batch_size,
        kotogram=[""] * batch_size,
        indices=torch.arange(batch_size),
        kc_targets=kc_targets,
    )


class TestDenseSparseRouting:
    """Tests for dense/sparse KC training path routing."""

    def test_dense_families_have_kc_targets_key(self) -> None:
        """Verify that is_family_sparse=False families get kc_targets_* keys."""
        # Setup with a mix of dense and sparse families
        target_specs = {
            KcFamilyId.BAG_POS: 50,  # Dense
            KcFamilyId.NGRAM_POS: 16384,  # Sparse
        }

        batch = _make_dummy_batch(batch_size=4, target_specs=target_specs)
        tokenizer = DummyTokenizer()

        result = create_kc_batch(batch, tokenizer, target_specs)  # type: ignore[arg-type]

        # Dense families should have kc_targets_* keys
        assert "kc_targets_bag_pos" in result, (
            "Dense family should have kc_targets_* key"
        )

        # Sparse families should NOT have kc_targets_* keys
        assert "kc_targets_ngram_pos" not in result, (
            "Sparse family should NOT have kc_targets_* key"
        )

        # Both should have sparse format keys
        assert "kc_pos_inds_bag_pos" in result
        assert "kc_pos_mask_bag_pos" in result
        assert "kc_pos_inds_ngram_pos" in result
        assert "kc_pos_mask_ngram_pos" in result

    def test_all_dense_families_have_targets(self) -> None:
        """Verify all 8 dense families are correctly routed."""
        dense_families = [fid for fid in ALL_KC_FAMILIES if not is_family_sparse(fid)]
        sparse_families = [fid for fid in ALL_KC_FAMILIES if is_family_sparse(fid)]

        # Verify we have exactly 8 of each
        assert len(dense_families) == 8, (
            f"Expected 8 dense families, got {len(dense_families)}"
        )
        assert len(sparse_families) == 8, (
            f"Expected 8 sparse families, got {len(sparse_families)}"
        )

        # Create specs for all families
        target_specs = {}
        for fid in dense_families:
            target_specs[fid] = 100  # Small vocab for dense
        for fid in sparse_families:
            target_specs[fid] = 16384  # Large vocab for sparse

        batch = _make_dummy_batch(batch_size=4, target_specs=target_specs)
        tokenizer = DummyTokenizer()

        result = create_kc_batch(batch, tokenizer, target_specs)  # type: ignore[arg-type]

        # Verify all dense families have dense keys
        for fid in dense_families:
            key = f"kc_targets_{fid.name.lower()}"
            assert key in result, f"Dense family {fid.name} should have {key}"

        # Verify no sparse families have dense keys
        for fid in sparse_families:
            key = f"kc_targets_{fid.name.lower()}"
            assert key not in result, f"Sparse family {fid.name} should NOT have {key}"

    def test_dense_target_shape_and_values(self) -> None:
        """Verify dense targets have correct shape and multi-hot encoding."""
        vocab_size = 50
        target_specs = {KcFamilyId.BAG_POS: vocab_size}

        batch = _make_dummy_batch(batch_size=4, target_specs=target_specs)
        tokenizer = DummyTokenizer()

        result = create_kc_batch(batch, tokenizer, target_specs)  # type: ignore[arg-type]

        dense_targets = result["kc_targets_bag_pos"]

        # Check shape
        assert dense_targets.shape == (4, vocab_size), (
            f"Wrong shape: {dense_targets.shape}"
        )

        # Check values are multi-hot (0s and 1s)
        assert ((dense_targets == 0) | (dense_targets == 1)).all(), (
            "Values should be 0 or 1"
        )

        # Check that positive IDs (3,4,5) are set
        for i in range(4):
            assert dense_targets[i, 3] == 1, "ID 3 should be positive"
            assert dense_targets[i, 4] == 1, "ID 4 should be positive"
            assert dense_targets[i, 5] == 1, "ID 5 should be positive"
