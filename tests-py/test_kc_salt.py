import unittest

from train import kc


class TestKCSaltKeys(unittest.TestCase):
    def setUp(self):
        # Keep an exact copy so we can restore after each test
        self._orig_salt = dict(kc.SALT)

    def tearDown(self):
        kc.SALT.clear()
        kc.SALT.update(self._orig_salt)

    def test_missing_salt_key_for_ngram_field_is_descriptive(self):
        # Remove a key that compute_kc_targets will look up via f"ngram_{field}"
        missing_key = "ngram_pos_detail_1"
        self.assertIn(missing_key, kc.SALT, "Test assumes SALT initially has this key")
        del kc.SALT[missing_key]

        feature_ids = {
            "reading": [13, 14, 15, 16],
            "pos": [1, 1, 1, 1],
            "pos_detail_1": [1, 2, 3, 4],  # triggers SALT["ngram_pos_detail_1"]
            "conjugated_form": [5, 6, 7, 8],
            "conjugated_type": [9, 10, 11, 12],
        }

        with self.assertRaises(KeyError) as ctx:
            kc.compute_kc_targets(feature_ids)

        msg = str(ctx.exception)
        # MissingMappingError inherits from KeyError and includes standard message
        self.assertIn("Missing mapping in SALT", msg)
        self.assertIn(missing_key, msg)

    def test_salt_present_allows_compute(self):
        feature_ids = {
            "reading": [13, 14, 15, 16],
            "pos": [1, 1, 1, 1],
            "pos_detail_1": [1, 2, 3, 4],
            "conjugated_form": [5, 6, 7, 8],
            "conjugated_type": [9, 10, 11, 12],
        }
        out = kc.compute_kc_targets(feature_ids)
        # Sanity: expect at least some ngram and tail_ngram outputs
        self.assertIn("ngram_pos_detail_1", out)
        self.assertIn("tail_ngram_pos_detail_1", out)
        self.assertIsInstance(out["ngram_pos_detail_1"], list)
        self.assertIsInstance(out["tail_ngram_pos_detail_1"], list)


if __name__ == "__main__":
    unittest.main()
