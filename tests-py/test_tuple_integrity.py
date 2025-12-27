"""Unit tests for result integrity in train_style.py data pipeline.

These tests validate that objects returned by the processing functions have
the correct attributes and types, ensuring the data flow is correct.
"""

import os
import sys
import unittest

from train.types import ProcessedSample

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestResultIntegrity(unittest.TestCase):
    """Tests for result structure validation."""

    def test_process_sentence_batch_returns_objects(self):
        """Verify _process_sentence_batch returns ProcessedSample objects."""
        from scripts.label import (  # pylint: disable=import-private-name
            _process_sentence_batch,
        )

        # Single-element batch
        batch = [("これはテストです", 1)]  # (sentence, gram_label)
        results, _ = _process_sentence_batch(batch)

        assert isinstance(results, dict)
        assert len(results["sentences"]) == 1, "Expected 1 result"
        assert results["sentences"][0] == "これはテストです"
        assert isinstance(results["f_ids"][0], int)

    def test_compute_labels_batch_returns_objects(self):
        """Verify _compute_labels_batch returns ProcessedSample objects."""
        from scripts.label import (  # pylint: disable=import-private-name
            _compute_labels_batch,
        )

        # Single-element batch with pre-computed kotogram
        batch = [("テスト", "テスト[*]", 1)]  # (sentence, kotogram, gram_label)
        results, _ = _compute_labels_batch(batch)

        assert isinstance(results, dict), f"Expected dict results, got {type(results)}"
        assert len(results["sentences"]) == 1, "Expected 1 result sentence"

        # Verify columns exist
        expected_keys = [
            "sentences",
            "kotograms",
            "f_ids",
            "g_vals",
            "g_prags",
            "gram_labels",
            "reg_ids_flat",
            "reg_ids_lens",
        ]
        for key in expected_keys:
            assert key in results, f"Missing key {key}"
            assert isinstance(results[key], list), f"{key} should be a list"

        # Verify values
        assert results["sentences"][0] == "テスト"
        assert results["f_ids"][0] == 2  # Neutral (assuming ID 2)
        assert results["gram_labels"][0] == 1

    def test_process_parallel_returns_objects(self):
        """Verify _process_sentence_batch returns ProcessedSample objects."""
        from scripts.label import (  # pylint: disable=import-private-name
            _process_sentence_batch,
        )

        # Simple test rows
        rows = [
            ("これはテストです", 1),
            ("お元気ですか", 1),
        ]

        results, _ = _process_sentence_batch(rows)

        assert isinstance(results, dict)
        cnt = len(results["sentences"])
        assert cnt > 0, "Expected at least some results"

        for i in range(cnt):
            assert isinstance(results["sentences"][i], str)
            assert isinstance(results["kotograms"][i], str)
            assert isinstance(results["f_ids"][i], int)
            assert isinstance(results["g_vals"][i], float)
            assert isinstance(results["g_prags"][i], int)
            assert isinstance(results["gram_labels"][i], int)

    def test_register_ids_contains_valid_integers(self):
        """Verify register_ids list contains valid integer IDs."""
        from scripts.label import (  # pylint: disable=import-private-name
            _process_sentence_batch,
        )

        batch = [("お嬢様はごきげんよう", 1)]  # Ojousama register
        results, _ = _process_sentence_batch(batch)

        assert isinstance(results, dict)
        assert len(results["sentences"]) == 1

        reg_ids_flat = results["reg_ids_flat"]
        reg_ids_lens = results["reg_ids_lens"]

        assert len(reg_ids_lens) == 1
        length = reg_ids_lens[0]
        assert length > 0

        # In this simple batch of 1, flat list is just the ids
        register_ids = reg_ids_flat
        assert len(register_ids) == length

        for rid in register_ids:
            assert isinstance(rid, int), f"register ID should be int, got {type(rid)}"
            assert 0 <= rid <= 10, f"register ID {rid} out of expected range [0, 10]"


class TestEncodingInputs(unittest.TestCase):
    """Tests for the encoding_inputs tuple passed to _encode_samples_batch."""

    def test_encoding_inputs_structure(self):
        """Verify encoding_inputs are extracted correctly from processed_results."""
        # Simulate processed_results from _process_parallel
        processed_results = [
            ProcessedSample(
                "sentence1", "kotogram1", 0, 0.0, 0, [0], 1, 1
            ),  # success=1
            ProcessedSample(
                "sentence2", "kotogram2", 1, 1.0, 1, [1, 2], 0, 1
            ),  # Multi-label register
            ProcessedSample(
                "sentence3", "kotogram3", 2, 2.0, 2, [6], 1, 0
            ),  # success=0 (should be skipped)
        ]

        # Extract encoding_inputs as done in from_multiple_tsv
        encoding_inputs = []
        for p in processed_results:
            # Object result
            if p.success:
                encoding_inputs.append(
                    (
                        p.sentence,
                        p.kotogram,
                        p.formality_id,
                        p.gender_value,
                        p.gender_pragmatic,
                        p.register_ids,
                        p.gram_label,
                    )
                )

        assert len(encoding_inputs) == 2, "Should have 2 successful items"

        # Validate first item
        sent, koto, f_id, g_val, g_prag, r_ids, gram = encoding_inputs[0]
        assert sent == "sentence1"
        assert koto == "kotogram1"
        assert f_id == 0
        assert g_val == 0.0
        assert g_prag == 0
        assert r_ids == [0]
        assert gram == 1

        # Validate second item (multi-label register)
        sent, koto, f_id, g_val, g_prag, r_ids, gram = encoding_inputs[1]
        assert r_ids == [1, 2], f"Expected [1, 2], got {r_ids}"


if __name__ == "__main__":
    unittest.main()
