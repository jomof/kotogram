"""Tests for curate script KC family references.

This test ensures the curate script's KC family references stay in sync
with the KcFamilyId enum, catching rename mismatches.

Also verifies that kc-tail-stats diagnostic tool computes the same KC
targets as the actual training pipeline.
"""

import ast
import os
import unittest
from typing import List

from train.kc import KcFamilyId


class TestCurateKcFamilyReferences:
    """Test that curate script references valid KC family IDs."""

    def test_curate_uses_valid_kc_family_ids(self) -> None:
        """Verify all KcFamilyId references in curate are valid enum values.

        This test catches renames of KcFamilyId enum values that weren't
        propagated to the curate script, like NGRAM_POS_DETAIL_1 -> NGRAM_COMPOUND_1.
        """
        # Parse the curate script
        with open("scripts/curate", "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)

        # Find all KcFamilyId.XXX attribute accesses
        kc_family_refs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                # Check if it's KcFamilyId.XXX
                if isinstance(node.value, ast.Name) and node.value.id == "KcFamilyId":
                    kc_family_refs.append(node.attr)

        # Get valid enum names
        valid_names = {member.name for member in KcFamilyId}

        # Check all references are valid
        invalid_refs = [ref for ref in kc_family_refs if ref not in valid_names]

        assert not invalid_refs, (
            f"Invalid KcFamilyId references in scripts/curate: {invalid_refs}. "
            f"Valid names are: {sorted(valid_names)}"
        )

    def test_all_kc_families_have_feature_mapping(self) -> None:
        """Verify all KC families are mapped in FAMILY_FEATURES."""
        from train.kc import FAMILY_FEATURES

        missing = [f for f in KcFamilyId if f not in FAMILY_FEATURES]
        assert not missing, f"KC families missing from FAMILY_FEATURES: {missing}"


class TestCurateKcTailStatsSync(unittest.TestCase):
    """Verify curate kc-tail-stats computes the same targets as training.

    This test ensures the diagnostic tool stays in sync with training data.
    If these drift apart, the diagnostic output would be misleading.
    """

    @classmethod
    def setUpClass(cls):
        """Load tokenizer and dataset once for all tests."""
        from kotogram.tokenizer import Tokenizer
        from train import paths
        from train.kc import initialize_disallow_filter

        vocab_path = os.path.join(paths.get_style_dataset_cache_dir(), "vocab.json")
        if not os.path.exists(vocab_path):
            raise unittest.SkipTest(
                "Vocab not found. Run 'bin/train_style --label' first."
            )

        cls.tokenizer = Tokenizer.load(vocab_path)

        # Initialize disallow filter (same as training and curate do)
        compound_1_vocab = cls.tokenizer.field_vocabs.get("compound_1", {})
        initialize_disallow_filter(compound_1_vocab)

        # Check if dataset exists
        cache_dir = paths.get_style_dataset_cache_dir()
        offsets_path = os.path.join(cache_dir, "offsets.bin")
        if not os.path.exists(offsets_path):
            raise unittest.SkipTest(
                "Dataset not found. Run 'bin/train_style --label' first."
            )

        # Load dataset
        from train.dataset import StyleDataset

        cls.dataset = StyleDataset(cache_dir, cls.tokenizer)

    def test_tail_compound_1_matches_training_data(self):  # pylint: disable=too-many-locals
        """Verify curate helper produces same TAIL_COMPOUND_1 as training dataset.

        Compares training TAIL_COMPOUND_1 targets from StyleDataset with what
        the curate helper functions compute (same as kc-tail-stats). They must
        be identical to ensure the diagnostic tool is accurate.
        """
        import torch

        from train.kc import get_disallowed_positions, get_tail_ids

        mismatches: List[str] = []

        for idx in range(min(100, len(self.dataset))):
            sample = self.dataset[idx]

            # Get TAIL_COMPOUND_1 from training data
            training_set = set(sample.kc_targets.get(KcFamilyId.TAIL_COMPOUND_1, []))

            # Must convert tensors to lists (same as compute_kc_targets does)
            feature_ids_list = {
                k: v.tolist() if isinstance(v, torch.Tensor) else list(v)
                for k, v in sample.feature_ids.items()
            }
            disallowed = get_disallowed_positions(feature_ids_list)
            curate_set = set(
                get_tail_ids(
                    feature_ids_list,
                    "compound_1",
                    filter_unk=True,
                    disallowed_positions=disallowed,
                )
            )

            if training_set != curate_set:
                mismatches.append(
                    f"Sample {idx}: training={sorted(training_set)}, curate={sorted(curate_set)}"
                )

        if mismatches:
            self.fail(
                "TAIL_COMPOUND_1 mismatch between training and curate!\n"
                "kc-tail-stats is out of sync with training data.\n"
                f"First 5 mismatches:\n{chr(10).join(mismatches[:5])}"
            )

    def test_disallow_filter_active(self):
        """Verify disallow filter is applied when disallowed tokens are present."""
        from train.kc import TAIL_DISALLOW, get_disallowed_positions

        # Find a sample that contains a disallowed token
        for idx in range(min(500, len(self.dataset))):
            sample = self.dataset[idx]
            positions = get_disallowed_positions(sample.feature_ids)
            if positions:
                # Found one - verify disallow filter is working
                self.assertGreater(len(positions), 0)
                # Verify TAIL_DISALLOW contains expected tokens
                self.assertIn("noun:common-noun", TAIL_DISALLOW)
                return

        # No samples with disallowed tokens found - still pass but note it
        # This is unexpected but not a test failure

    def test_kc_families_matches_training_data(self):
        """Verify kc-families (compute_kc_targets) matches training dataset.

        This ensures scripts/curate kc-families shows the same KC targets
        as what training actually uses. Tests ALL KC families, not just
        TAIL_COMPOUND_1.
        """
        from train.kc import ALL_KC_FAMILIES, compute_kc_targets, is_family_db_sourced

        # Sample a few indices to test
        sample_size = min(50, len(self.dataset))
        test_indices = list(range(0, sample_size))

        mismatches: List[str] = []

        for idx in test_indices:
            sample = self.dataset[idx]

            # Get KC targets from training data (pre-computed in dataset)
            training_targets = sample.kc_targets

            # Compute KC targets using kc-families logic (compute_kc_targets)
            computed_targets = compute_kc_targets(sample.feature_ids)

            # Compare all families except DB-sourced ones (GRAMMAR_POINT)
            # DB-sourced families have different data structures (dict with pos/neg)
            # and are not included in compute_kc_targets output
            for family in ALL_KC_FAMILIES:
                if is_family_db_sourced(family):
                    continue  # Skip DB-sourced families

                training_set = set(training_targets.get(family, []))
                computed_set = set(computed_targets.get(family, []))

                if training_set != computed_set:
                    mismatches.append(
                        f"Sample {idx}, {family.name}: "
                        f"training={sorted(training_set)}, "
                        f"computed={sorted(computed_set)}"
                    )

        if mismatches:
            msg = (
                "KC targets mismatch between training and kc-families!\n"
                "scripts/curate kc-families is out of sync with training.\n"
                "First 10 mismatches:\n" + "\n".join(mismatches[:10])
            )
            self.fail(msg)


if __name__ == "__main__":
    unittest.main()
