import unittest

from kotogram.exceptions import MissingMappingError
from train.kc import (
    KcFamilyId,
    compute_kc_targets,
    get_family,
)


class TestKCSaltBehavior(unittest.TestCase):
    """Test salt/hashing behavior through the public API."""

    def test_ngram_family_has_salt(self):
        """Test that ngram families have salt values for domain separation."""
        # Ngram families should have salt values
        family = get_family(KcFamilyId.NGRAM_COMPOUND_1)
        self.assertIsNotNone(family.salt)
        self.assertIsInstance(family.salt, int)
        self.assertGreater(family.salt, 0)

    def test_dense_family_has_no_salt(self):
        """Test that dense (bag/tail) families have no salt values."""
        # Bag families don't have salt values
        family = get_family(KcFamilyId.BAG_POS)
        self.assertIsNone(family.salt)

    def test_db_sourced_family_has_no_salt(self):
        """Test that DB-sourced families have no salt values."""
        family = get_family(KcFamilyId.GRAMMAR_POINT)
        self.assertIsNone(family.salt)

    def test_compute_kc_targets_works(self):
        """Test that compute_kc_targets works with proper feature data."""
        feature_ids = {
            "reading": [13, 14, 15, 16],
            "pos": [1, 1, 1, 1],
            "compound_1": [1, 2, 3, 4],
            "conjugated_form": [5, 6, 7, 8],
            "conjugated_type": [9, 10, 11, 12],
        }
        out = compute_kc_targets(feature_ids)
        # Sanity: expect at least some ngram and tail_ngram outputs
        self.assertIn(KcFamilyId.NGRAM_COMPOUND_1, out)
        self.assertIn(KcFamilyId.TAIL_NGRAM_COMPOUND_1, out)
        self.assertIsInstance(out[KcFamilyId.NGRAM_COMPOUND_1], list)
        self.assertIsInstance(out[KcFamilyId.TAIL_NGRAM_COMPOUND_1], list)

    def test_custom_map_name_error(self):
        """Test MissingMappingError with a custom map name."""
        with self.assertRaises(MissingMappingError) as ctx:
            raise MissingMappingError("CUSTOM_MAP", "test_key", "Optional context")

        msg = str(ctx.exception)
        self.assertIn("Missing mapping in CUSTOM_MAP", msg)
        self.assertIn("test_key", msg)


if __name__ == "__main__":
    unittest.main()
