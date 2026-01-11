"""Tests for curate script KC family references.

This test ensures the curate script's KC family references stay in sync
with the KcFamilyId enum, catching rename mismatches.
"""

import ast

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
