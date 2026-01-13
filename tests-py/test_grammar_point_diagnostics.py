"""Test that GRAMMAR_POINT family appears in KC epoch diagnostics."""

import unittest

from train.kc import ALL_KC_FAMILIES, KcFamilyId


class TestGrammarPointDiagnostics(unittest.TestCase):
    """Tests for GRAMMAR_POINT KC family diagnostics visibility."""

    def test_grammar_point_in_all_kc_families(self):
        """GRAMMAR_POINT should be a member of ALL_KC_FAMILIES."""
        self.assertIn(
            KcFamilyId.GRAMMAR_POINT,
            ALL_KC_FAMILIES,
            "GRAMMAR_POINT must be in ALL_KC_FAMILIES",
        )

    def test_all_kc_families_count_matches_enum(self):
        """ALL_KC_FAMILIES should contain all KcFamilyId enum members."""
        # Data-driven: all enum members should be in ALL_KC_FAMILIES
        all_family_ids = set(KcFamilyId)
        self.assertEqual(
            set(ALL_KC_FAMILIES),
            all_family_ids,
            f"ALL_KC_FAMILIES mismatch with KcFamilyId enum. "
            f"Missing: {all_family_ids - set(ALL_KC_FAMILIES)}, "
            f"Extra: {set(ALL_KC_FAMILIES) - all_family_ids}",
        )


if __name__ == "__main__":
    unittest.main()
