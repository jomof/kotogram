"""Unit tests for tuple integrity in train_style.py data pipeline.

These tests validate that tuples returned by the processing functions have
the correct cardinality and field types, ensuring the data flow is correct.
"""

import pytest
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTupleCardinality:
    """Tests for tuple structure validation."""

    def test_process_sentence_batch_returns_8_tuple(self):
        """Verify _process_sentence_batch returns 8-element tuples."""
        from scripts.train_style import _process_sentence_batch

        # Single-element batch
        batch = [("これはテストです", "id_001", 1)]  # (sentence, sentence_id, gram_label)
        results = _process_sentence_batch(batch)

        assert len(results) == 1, "Expected 1 result"
        result = results[0]
        
        # Validate cardinality
        assert len(result) == 8, f"Expected 8-tuple, got {len(result)}-tuple: {result}"

        # Validate types
        sentence, sentence_id, kotogram, formality_id, gender_id, register_ids, gram_label, success = result
        assert isinstance(sentence, str), f"sentence should be str, got {type(sentence)}"
        assert isinstance(sentence_id, str), f"sentence_id should be str, got {type(sentence_id)}"
        assert isinstance(kotogram, str), f"kotogram should be str, got {type(kotogram)}"
        assert isinstance(formality_id, int), f"formality_id should be int, got {type(formality_id)}"
        assert isinstance(gender_id, int), f"gender_id should be int, got {type(gender_id)}"
        assert isinstance(register_ids, list), f"register_ids should be list, got {type(register_ids)}"
        assert isinstance(gram_label, int), f"gram_label should be int, got {type(gram_label)}"
        assert isinstance(success, int), f"success should be int, got {type(success)}"

    def test_compute_labels_batch_returns_7_tuple(self):
        """Verify _compute_labels_batch returns 7-element tuples."""
        from scripts.train_style import _compute_labels_batch

        # Single-element batch with pre-computed kotogram
        batch = [("テスト", "テスト[*]", 1)]  # (sentence, kotogram, gram_label)
        results = _compute_labels_batch(batch)

        assert len(results) == 1, "Expected 1 result"
        result = results[0]
        
        # Validate cardinality
        assert len(result) == 7, f"Expected 7-tuple, got {len(result)}-tuple: {result}"

        # Validate types
        sentence, kotogram, formality_id, gender_id, register_ids, gram_label, success = result
        assert isinstance(sentence, str), f"sentence should be str, got {type(sentence)}"
        assert isinstance(kotogram, str), f"kotogram should be str, got {type(kotogram)}"
        assert isinstance(formality_id, int), f"formality_id should be int, got {type(formality_id)}"
        assert isinstance(gender_id, int), f"gender_id should be int, got {type(gender_id)}"
        assert isinstance(register_ids, list), f"register_ids should be list, got {type(register_ids)}"
        assert isinstance(gram_label, int), f"gram_label should be int, got {type(gram_label)}"
        assert isinstance(success, int), f"success should be int, got {type(success)}"

    def test_process_parallel_returns_7_tuple(self):
        """Verify _process_parallel returns 7-element tuples."""
        from scripts.train_style import StyleDataset

        # Simple test rows
        rows = [
            ("これはテストです", "id_001", 1),
            ("お元気ですか", "id_002", 1),
        ]

        results = StyleDataset._process_parallel(
            rows, 
            num_workers=1, 
            batch_size=100, 
            verbose=False,
            use_kotogram_cache=False  # Don't use cache for test isolation
        )

        assert len(results) > 0, "Expected at least some results"
        
        for i, result in enumerate(results):
            # Validate cardinality
            assert len(result) == 7, f"Result {i}: Expected 7-tuple, got {len(result)}-tuple: {result}"

            # Validate types
            sentence, kotogram, f_id, g_id, r_ids, gram_label, success = result
            assert isinstance(sentence, str), f"Result {i}: sentence should be str"
            assert isinstance(kotogram, str), f"Result {i}: kotogram should be str"
            assert isinstance(f_id, int), f"Result {i}: f_id should be int"
            assert isinstance(g_id, int), f"Result {i}: g_id should be int"
            assert isinstance(r_ids, list), f"Result {i}: r_ids should be list, got {type(r_ids)}"
            assert isinstance(gram_label, int), f"Result {i}: gram_label should be int"
            assert isinstance(success, int), f"Result {i}: success should be int"

    def test_register_ids_contains_valid_integers(self):
        """Verify register_ids list contains valid integer IDs."""
        from scripts.train_style import _process_sentence_batch

        batch = [("お嬢様はごきげんよう", "id_003", 1)]  # Ojousama register
        results = _process_sentence_batch(batch)

        assert len(results) == 1
        _, _, _, _, _, register_ids, _, _ = results[0]

        assert isinstance(register_ids, list), f"register_ids should be list, got {type(register_ids)}"
        assert len(register_ids) > 0, "register_ids should not be empty"
        for rid in register_ids:
            assert isinstance(rid, int), f"register ID should be int, got {type(rid)}"
            assert 0 <= rid <= 10, f"register ID {rid} out of expected range [0, 10]"


class TestEncodingInputs:
    """Tests for the encoding_inputs tuple passed to _encode_samples_batch."""

    def test_encoding_inputs_structure(self):
        """Verify encoding_inputs are extracted correctly from processed_results."""
        # Simulate processed_results from _process_parallel
        processed_results = [
            ("sentence1", "kotogram1", 0, 0, [0], 1, 1),  # 7-tuple: success=1
            ("sentence2", "kotogram2", 1, 1, [1, 2], 0, 1),  # Multi-label register
            ("sentence3", "kotogram3", 2, 2, [6], 1, 0),  # success=0 (should be skipped)
        ]

        # Extract encoding_inputs as done in from_multiple_tsv
        encoding_inputs = []
        for p in processed_results:
            assert len(p) == 7, f"Expected 7-tuple, got {len(p)}"
            if p[6]:  # success
                encoding_inputs.append((p[0], p[1], p[2], p[3], p[4], p[5]))

        assert len(encoding_inputs) == 2, "Should have 2 successful items"
        
        # Validate first item
        sent, koto, f_id, g_id, r_ids, gram = encoding_inputs[0]
        assert sent == "sentence1"
        assert koto == "kotogram1"
        assert f_id == 0
        assert g_id == 0
        assert r_ids == [0]
        assert gram == 1

        # Validate second item (multi-label register)
        sent, koto, f_id, g_id, r_ids, gram = encoding_inputs[1]
        assert r_ids == [1, 2], f"Expected [1, 2], got {r_ids}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
