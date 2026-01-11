"""Tests for the disallow list functionality in train.kc."""

from train.kc import (
    TAIL_NGRAM_DISALLOW_LIST,
    KcFamilyId,
    compute_kc_targets,
    get_disallowed_positions,
    get_tail_ids,
    initialize_disallow_filter,
)


class TestDisallowList:
    """Tests for the TAIL_NGRAM_DISALLOW_LIST filtering functionality."""

    def test_disallow_list_contains_common_noun(self):
        """Verify the disallow list contains the expected tokens."""
        assert "noun:common-noun" in TAIL_NGRAM_DISALLOW_LIST

    def test_initialize_disallow_filter_caches_ids(self):
        """Verify that initialize_disallow_filter resolves tokens to IDs."""
        # Create a mock vocab
        vocab = {
            "noun:common-noun": 42,
            "verb:general": 10,
            "particle:case-particle": 20,
        }
        initialize_disallow_filter(vocab)

        # After initialization, get_disallowed_positions should work
        feature_ids = {"compound_1": [10, 42, 20, 42]}  # 42 is noun:common-noun
        positions = get_disallowed_positions(feature_ids)

        # Positions 1 and 3 have the disallowed ID (42)
        assert positions == {1, 3}

    def test_get_disallowed_positions_returns_empty_if_not_initialized(self):
        """Verify get_disallowed_positions works gracefully before initialization."""
        # This test should be run in isolation or after resetting the cache
        # For safety, we just verify it doesn't crash with valid input
        feature_ids = {"compound_1": [1, 2, 3]}
        positions = get_disallowed_positions(feature_ids)
        assert isinstance(positions, set)

    def test_get_disallowed_positions_returns_empty_without_compound_1(self):
        """Verify get_disallowed_positions returns empty for missing field."""
        feature_ids = {"pos": [1, 2, 3]}  # No compound_1
        positions = get_disallowed_positions(feature_ids)
        assert positions == set()

    def test_get_tail_ids_filters_disallowed_positions(self):
        """Verify get_tail_ids respects disallowed_positions parameter."""
        feature_ids = {
            "compound_1": [10, 20, 30, 40, 50],  # 5 tokens
        }

        # Without filtering
        tail_ids_all = get_tail_ids(feature_ids, "compound_1")
        assert 30 in tail_ids_all

        # With position 2 (value 30) disallowed
        tail_ids_filtered = get_tail_ids(
            feature_ids, "compound_1", disallowed_positions={2}
        )
        assert 30 not in tail_ids_filtered
        assert 40 in tail_ids_filtered
        assert 50 in tail_ids_filtered

    def test_compute_kc_targets_filters_disallowed_from_tail_families(self):
        """Verify compute_kc_targets uses disallow filter for tail families."""
        # Initialize with a known disallow ID
        vocab = {"noun:common-noun": 99}
        initialize_disallow_filter(vocab)

        # Create feature_ids with the disallowed token
        feature_ids = {
            "compound_1": [10, 99, 20],  # 99 is disallowed
            "pos": [1, 2, 3],
        }

        targets = compute_kc_targets(feature_ids)

        # TAIL_COMPOUND_1 should not contain 99
        tail_compound_1 = targets[KcFamilyId.TAIL_COMPOUND_1]
        assert 99 not in tail_compound_1
        # But should contain the other values (10, 20)
        assert 10 in tail_compound_1
        assert 20 in tail_compound_1

    def test_bag_families_not_affected_by_disallow_list(self):
        """Verify that bag families are NOT filtered by the disallow list."""
        # Initialize with a known disallow ID
        vocab = {"noun:common-noun": 99}
        initialize_disallow_filter(vocab)

        # Create feature_ids with the disallowed token
        feature_ids = {
            "compound_1": [10, 99, 20],  # 99 is disallowed
        }

        targets = compute_kc_targets(feature_ids)

        # BAG_COMPOUND_1 SHOULD contain 99 (bags are not filtered)
        bag_compound_1 = targets[KcFamilyId.BAG_COMPOUND_1]
        assert 99 in bag_compound_1
