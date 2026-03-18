"""Tests for curate script KC family references and KC target computation.

This test ensures the curate script's KC family references stay in sync
with the KcFamilyId enum, catching rename mismatches.

Also verifies that compute_kc_targets is deterministic and correct using
synthetic inputs — no dependency on production labeling artifacts.
"""

import ast
import unittest
from typing import Any, Dict, List, Set

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


# ---------------------------------------------------------------------------
# Synthetic fixtures for compute_kc_targets tests
# ---------------------------------------------------------------------------

_DISALLOW_VOCAB: Dict[str, int] = {
    "noun:common-noun": 4,
    "verb:general": 5,
    "particle:case-particle": 6,
    "aux-verb": 7,
    "aux-symbol:period": 8,
    "adj-i:general": 9,
    "adverb:general": 10,
    "suffix:nominal": 11,
    "noun:proper-noun": 12,
}

# 10-token sentence; positions 2, 5, 9 are disallowed (compound_1 ∈ {4, 8}).
_FIXTURE_10: Dict[str, List[int]] = {
    "surface": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    "pos": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    "compound_1": [5, 6, 4, 7, 9, 4, 10, 5, 6, 8],
    "conjugated_type": [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    "reading_gram": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
}

# 3-token sentence with CLS (id=2) at position 0, no disallowed tokens.
_FIXTURE_CLS: Dict[str, List[int]] = {
    "surface": [2, 50, 51],
    "pos": [2, 15, 16],
    "compound_1": [2, 5, 6],
    "conjugated_type": [2, 33, 34],
    "reading_gram": [2, 60, 61],
}

# Expected outputs for _FIXTURE_10 (with disallow filter active).
# If compute_kc_targets logic changes, update these to match and re-label.
_EXPECTED_10: Dict[KcFamilyId, List[int]] = {
    KcFamilyId.BAG_READING_GRAM: [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
    KcFamilyId.BAG_POS: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    KcFamilyId.BAG_COMPOUND_1: [4, 5, 6, 7, 8, 9, 10],
    KcFamilyId.BAG_CONJUGATED_TYPE: [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    KcFamilyId.TAIL_READING_GRAM: [46, 47, 48],
    KcFamilyId.TAIL_POS: [26, 27, 28],
    KcFamilyId.TAIL_COMPOUND_1: [5, 6, 10],
    KcFamilyId.TAIL_CONJUGATED_TYPE: [36, 37, 38],
    KcFamilyId.NGRAM_POS: [48, 125, 439, 570, 725, 1174, 1186, 1202, 1215, 1364, 1999],
    KcFamilyId.NGRAM_COMPOUND_1: [
        464,
        4441,
        6983,
        15714,
        23032,
        23088,
        26745,
        28418,
        30602,
        31502,
    ],
    KcFamilyId.NGRAM_CONJUGATED_TYPE: [
        2355,
        2738,
        3571,
        3762,
        3801,
        4478,
        4769,
        5959,
        6055,
        6644,
        7492,
    ],
    KcFamilyId.NGRAM_READING_GRAM: [
        45379,
        50373,
        60292,
        62627,
        74431,
        97858,
        141901,
        167240,
        168458,
        226569,
        231823,
    ],
    KcFamilyId.TAIL_NGRAM_POS: [265, 270, 574],
    KcFamilyId.TAIL_NGRAM_COMPOUND_1: [4, 3183, 3699],
    KcFamilyId.TAIL_NGRAM_CONJUGATED_TYPE: [3655, 6539, 7034],
    KcFamilyId.TAIL_NGRAM_READING_GRAM: [15107, 90496, 114337],
}

# Expected outputs for _FIXTURE_CLS (CLS excluded, no disallowed tokens).
_EXPECTED_CLS: Dict[KcFamilyId, List[int]] = {
    KcFamilyId.BAG_READING_GRAM: [60, 61],
    KcFamilyId.BAG_POS: [15, 16],
    KcFamilyId.BAG_COMPOUND_1: [5, 6],
    KcFamilyId.BAG_CONJUGATED_TYPE: [33, 34],
    KcFamilyId.TAIL_READING_GRAM: [60, 61],
    KcFamilyId.TAIL_POS: [15, 16],
    KcFamilyId.TAIL_COMPOUND_1: [5, 6],
    KcFamilyId.TAIL_CONJUGATED_TYPE: [33, 34],
    KcFamilyId.NGRAM_POS: [1854],
    KcFamilyId.NGRAM_COMPOUND_1: [26745],
    KcFamilyId.NGRAM_CONJUGATED_TYPE: [6055],
    KcFamilyId.NGRAM_READING_GRAM: [92882],
    KcFamilyId.TAIL_NGRAM_POS: [256],
    KcFamilyId.TAIL_NGRAM_COMPOUND_1: [3183],
    KcFamilyId.TAIL_NGRAM_CONJUGATED_TYPE: [2658],
    KcFamilyId.TAIL_NGRAM_READING_GRAM: [85680],
}


def _computed_families(
    feature_ids: Dict[str, List[int]],
) -> Dict[KcFamilyId, Any]:
    """Thin wrapper: compute non-DB-sourced KC families."""
    from train.kc import compute_kc_targets, is_family_db_sourced

    raw = compute_kc_targets(feature_ids)
    return {k: v for k, v in raw.items() if not is_family_db_sourced(k)}


class TestComputeKcTargets(unittest.TestCase):
    """Verify compute_kc_targets correctness using synthetic fixtures.

    These tests are fully self-contained — no production labeling artifacts
    required.  If the expected values need updating (because compute_kc_targets
    was intentionally changed), update the _EXPECTED_* dicts above and re-run
    the labeling pipeline.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from train.kc import initialize_disallow_filter

        initialize_disallow_filter(_DISALLOW_VOCAB)

    def test_determinism(self) -> None:
        """Repeated calls produce identical output."""
        a = _computed_families(_FIXTURE_10)
        b = _computed_families(_FIXTURE_10)
        for fam_id, vals_a in a.items():
            self.assertEqual(vals_a, b[fam_id], f"{fam_id.name} not deterministic")

    def test_fixture_10_all_families(self) -> None:
        """Verify all non-DB families against known expected values."""
        actual = _computed_families(_FIXTURE_10)
        for fam_id, expected in _EXPECTED_10.items():
            self.assertEqual(
                actual.get(fam_id, []),
                expected,
                f"{fam_id.name} mismatch",
            )

    def test_fixture_cls_all_families(self) -> None:
        """Verify CLS token is excluded from all families."""
        actual = _computed_families(_FIXTURE_CLS)
        for fam_id, expected in _EXPECTED_CLS.items():
            self.assertEqual(
                actual.get(fam_id, []),
                expected,
                f"{fam_id.name} mismatch",
            )

    def test_cls_excluded_from_bags(self) -> None:
        """CLS_ID must never appear in bag family targets."""
        from kotogram.tokenizer import CLS_ID

        actual = _computed_families(_FIXTURE_CLS)
        for fam_id in [
            KcFamilyId.BAG_READING_GRAM,
            KcFamilyId.BAG_POS,
            KcFamilyId.BAG_COMPOUND_1,
            KcFamilyId.BAG_CONJUGATED_TYPE,
        ]:
            self.assertNotIn(CLS_ID, actual.get(fam_id, []), fam_id.name)

    def test_disallow_positions(self) -> None:
        """Positions with disallowed compound_1 IDs are identified."""
        from train.kc import get_disallowed_positions

        positions = get_disallowed_positions(_FIXTURE_10)
        self.assertEqual(positions, {2, 5, 9})

    def test_disallow_reduces_tail_but_not_bag(self) -> None:
        """Disallow filter removes tokens from tail/ngram families but not bag."""
        from train.kc import initialize_disallow_filter

        # Compute WITH disallow filter (already active from setUpClass)
        with_filter = _computed_families(_FIXTURE_10)

        # Compute WITHOUT disallow filter (empty vocab → no IDs to disallow)
        initialize_disallow_filter({})
        try:
            without_filter = _computed_families(_FIXTURE_10)
        finally:
            initialize_disallow_filter(_DISALLOW_VOCAB)

        bag_families: Set[KcFamilyId] = {
            KcFamilyId.BAG_READING_GRAM,
            KcFamilyId.BAG_POS,
            KcFamilyId.BAG_COMPOUND_1,
            KcFamilyId.BAG_CONJUGATED_TYPE,
        }

        for fam_id in with_filter:
            with_set = set(with_filter[fam_id])
            without_set = set(without_filter[fam_id])
            if fam_id in bag_families:
                self.assertEqual(
                    with_set, without_set, f"{fam_id.name} bags should be unaffected"
                )
            else:
                self.assertTrue(
                    len(with_set) < len(without_set),
                    f"{fam_id.name}: disallow filter should reduce targets "
                    f"(with={len(with_set)}, without={len(without_set)})",
                )

    def test_tail_window_respected(self) -> None:
        """Tail families only include tokens from the last KC_POS_BIASED_WINDOW positions."""
        from train.kc import (
            KC_POS_BIASED_WINDOW,
            get_disallowed_positions,
            get_tail_ids,
        )

        disallowed = get_disallowed_positions(_FIXTURE_10)
        tail = get_tail_ids(
            _FIXTURE_10, "compound_1", filter_unk=True, disallowed_positions=disallowed
        )
        self.assertEqual(tail, [5, 6, 10])

        seq_len = len(_FIXTURE_10["compound_1"])
        tail_start = max(0, seq_len - KC_POS_BIASED_WINDOW)
        self.assertEqual(tail_start, 5)

    def test_all_computed_families_have_expectations(self) -> None:
        """Every non-DB family returned by compute_kc_targets is covered by fixtures."""
        actual = _computed_families(_FIXTURE_10)
        untested = set(actual.keys()) - set(_EXPECTED_10.keys())
        self.assertFalse(
            untested,
            f"New KC families without test expectations: {[f.name for f in untested]}. "
            "Add entries to _EXPECTED_10 and _EXPECTED_CLS.",
        )


if __name__ == "__main__":
    unittest.main()
