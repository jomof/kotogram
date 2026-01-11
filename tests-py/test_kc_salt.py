import unittest

from train import kc
from train.kc import KcFamilyId


class TestKCSaltKeys(unittest.TestCase):
    def setUp(self):
        # Keep an exact copy so we can restore after each test
        self._orig_salt = dict(kc.SALT)

    def tearDown(self):
        kc.SALT.clear()
        kc.SALT.update(self._orig_salt)

    def test_missing_salt_key_for_ngram_field_is_descriptive(self):
        # Remove a key that compute_kc_targets will look up via f"ngram_{field}"
        # We now look up by KcFamilyId
        missing_key = KcFamilyId.NGRAM_COMPOUND_1
        self.assertIn(missing_key, kc.SALT, "Test assumes SALT initially has this key")
        del kc.SALT[missing_key]

        feature_ids = {
            "reading": [13, 14, 15, 16],
            "pos": [1, 1, 1, 1],
            "compound_1": [1, 2, 3, 4],  # triggers SALT["ngram_compound_1"]
            "conjugated_form": [5, 6, 7, 8],
            "conjugated_type": [9, 10, 11, 12],
        }

        with self.assertRaises(KeyError) as ctx:
            kc.compute_kc_targets(feature_ids)

        msg = str(ctx.exception)
        # MissingMappingError inherits from KeyError and includes standard message
        self.assertIn("Missing mapping in SALT", msg)
        self.assertIn(str(missing_key), msg)

    def test_missing_different_salt_key(self):
        # Vary the missing key to prevent 'key' arg constant value
        missing_key = KcFamilyId.NGRAM_POS
        self.assertIn(missing_key, kc.SALT)
        del kc.SALT[missing_key]

        feature_ids = {
            "reading": [13],
            "pos": [1],  # triggers SALT["ngram_pos"]
            "compound_1": [1],
            "conjugated_form": [5],
            "conjugated_type": [9],
        }

        with self.assertRaises(KeyError) as ctx:
            kc.compute_kc_targets(feature_ids)

        self.assertIn(str(missing_key), str(ctx.exception))

    def test_salt_present_allows_compute(self):
        feature_ids = {
            "reading": [13, 14, 15, 16],
            "pos": [1, 1, 1, 1],
            "compound_1": [1, 2, 3, 4],
            "conjugated_form": [5, 6, 7, 8],
            "conjugated_type": [9, 10, 11, 12],
        }
        out = kc.compute_kc_targets(feature_ids)
        # Sanity: expect at least some ngram and tail_ngram outputs
        self.assertIn(KcFamilyId.NGRAM_COMPOUND_1, out)
        self.assertIn(KcFamilyId.TAIL_NGRAM_COMPOUND_1, out)
        self.assertIsInstance(out[KcFamilyId.NGRAM_COMPOUND_1], list)
        self.assertIsInstance(out[KcFamilyId.TAIL_NGRAM_COMPOUND_1], list)

    def test_custom_map_name_error(self):
        """Test MissingMappingError with a custom map name."""
        from kotogram.exceptions import MissingMappingError

        with self.assertRaises(MissingMappingError) as ctx:
            raise MissingMappingError("CUSTOM_MAP", "test_key", "Optional context")

        msg = str(ctx.exception)
        self.assertIn("Missing mapping in CUSTOM_MAP", msg)
        self.assertIn("test_key", msg)


if __name__ == "__main__":
    unittest.main()
