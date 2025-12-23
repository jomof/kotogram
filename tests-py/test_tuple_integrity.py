"""Unit tests for result integrity in train_style.py data pipeline.

These tests validate that objects returned by the processing functions have
the correct attributes and types, ensuring the data flow is correct.
"""

import os
import sys
import unittest

from scripts.style_data import ProcessedSample

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestResultIntegrity(unittest.TestCase):
    """Tests for result structure validation."""

    def test_process_sentence_batch_returns_objects(self):
        """Verify _process_sentence_batch returns ProcessedSample objects."""
        from scripts.label import _process_sentence_batch

        # Single-element batch
        batch = [("これはテストです", 1)]  # (sentence, gram_label)
        results, counters = _process_sentence_batch(batch)

        assert len(results) == 1, "Expected 1 result"
        result = results[0]

        # Validate type
        assert isinstance(result, ProcessedSample), (
            f"Result should be ProcessedSample, got {type(result)}"
        )

        # Validate attributes
        self.assertTrue(hasattr(result, "sentence"), "Missing sentence attribute")
        assert isinstance(result.sentence, str), (
            f"sentence should be str, got {type(result.sentence)}"
        )
        assert isinstance(result.kotogram, str), (
            f"kotogram should be str, got {type(result.kotogram)}"
        )
        assert isinstance(result.formality_id, int), (
            f"formality_id should be int, got {type(result.formality_id)}"
        )
        assert isinstance(result.gender_value, float), (
            f"gender_value should be float, got {type(result.gender_value)}"
        )
        assert isinstance(result.gender_pragmatic, int), (
            f"gender_pragmatic should be int, got {type(result.gender_pragmatic)}"
        )
        assert isinstance(result.register_ids, list), (
            f"register_ids should be list, got {type(result.register_ids)}"
        )
        assert isinstance(result.gram_label, int), (
            f"gram_label should be int, got {type(result.gram_label)}"
        )
        assert isinstance(result.success, int), (
            f"success should be int, got {type(result.success)}"
        )

    def test_compute_labels_batch_returns_objects(self):
        """Verify _compute_labels_batch returns ProcessedSample objects."""
        from scripts.label import _compute_labels_batch

        # Single-element batch with pre-computed kotogram
        batch = [("テスト", "テスト[*]", 1)]  # (sentence, kotogram, gram_label)
        results, counters = _compute_labels_batch(batch)

        assert len(results) == 1, "Expected 1 result"
        result = results[0]

        # Validate type
        assert isinstance(result, ProcessedSample), (
            f"Result should be ProcessedSample, got {type(result)}"
        )

        # Validate attributes
        self.assertTrue(hasattr(result, "sentence"))
        assert isinstance(result.sentence, str), (
            f"sentence should be str, got {type(result.sentence)}"
        )
        assert isinstance(result.kotogram, str), (
            f"kotogram should be str, got {type(result.kotogram)}"
        )
        assert isinstance(result.formality_id, int), (
            f"formality_id should be int, got {type(result.formality_id)}"
        )
        assert isinstance(result.gender_value, float), (
            f"gender_value should be float, got {type(result.gender_value)}"
        )
        assert isinstance(result.gender_pragmatic, int), (
            f"gender_pragmatic should be int, got {type(result.gender_pragmatic)}"
        )
        assert isinstance(result.register_ids, list), (
            f"register_ids should be list, got {type(result.register_ids)}"
        )
        assert isinstance(result.gram_label, int), (
            f"gram_label should be int, got {type(result.gram_label)}"
        )
        assert isinstance(result.success, int), (
            f"success should be int, got {type(result.success)}"
        )

    def test_process_parallel_returns_objects(self):
        """Verify _process_sentence_batch returns ProcessedSample objects."""
        from scripts.label import _process_sentence_batch

        # Simple test rows
        rows = [
            ("これはテストです", 1),
            ("お元気ですか", 1),
        ]

        results, counters = _process_sentence_batch(rows)

        assert len(results) > 0, "Expected at least some results"

        for i, result in enumerate(results):
            # Validate type
            assert isinstance(result, ProcessedSample), (
                f"Result {i} should be ProcessedSample"
            )

            # Validate attributes
            self.assertTrue(hasattr(result, "sentence"))
            assert isinstance(result.sentence, str), (
                f"Result {i}: sentence should be str"
            )
            assert isinstance(result.kotogram, str), (
                f"Result {i}: kotogram should be str"
            )
            assert isinstance(result.formality_id, int), (
                f"Result {i}: f_id should be int"
            )
            assert isinstance(result.gender_value, float), (
                f"Result {i}: g_val should be float"
            )
            assert isinstance(result.gender_pragmatic, int), (
                f"Result {i}: g_prag should be int"
            )
            assert isinstance(result.register_ids, list), (
                f"Result {i}: r_ids should be list, got {type(result.register_ids)}"
            )
            assert isinstance(result.gram_label, int), (
                f"Result {i}: gram_label should be int"
            )
            assert isinstance(result.success, int), f"Result {i}: success should be int"

    def test_register_ids_contains_valid_integers(self):
        """Verify register_ids list contains valid integer IDs."""
        from scripts.label import _process_sentence_batch

        batch = [("お嬢様はごきげんよう", 1)]  # Ojousama register
        results, counters = _process_sentence_batch(batch)

        assert len(results) == 1
        result = results[0]
        register_ids = result.register_ids

        assert isinstance(register_ids, list), (
            f"register_ids should be list, got {type(register_ids)}"
        )
        assert len(register_ids) > 0, "register_ids should not be empty"
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
