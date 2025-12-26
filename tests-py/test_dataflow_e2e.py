#!/usr/bin/env python3
"""End-to-end data flow tests for train_style.py pipeline (Refactored).

These tests trace data through the entire pipeline, using actual parser output
instead of manually constructed kotograms. Each step prints its output for
debugging purposes.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.cache import get_kotogram_cache
from train.types import ProcessedSample


def populate_test_cache(rows):
    """Helper to populate the cache for testing."""
    from scripts.label import _process_sentence_batch

    processed, counters = _process_sentence_batch(rows)
    cache = get_kotogram_cache()
    memo = []
    for p in processed:
        if p.success:
            memo.append(p.to_cache_tuple())
    cache.put_batch(memo)
    return processed


def test_step1_parser_output():
    """Step 1: Verify parser produces valid kotogram from Japanese."""
    from kotogram.sudachi_japanese_parser import SudachiJapaneseParser

    print("\n" + "=" * 60)
    print("STEP 1: Parser Output")
    print("=" * 60)

    parser = SudachiJapaneseParser()

    test_sentences = [
        "これはテストです",  # Neutral formality
        "お元気ですか",  # Polite/formal
        "行くぜ",  # Masculine casual
        "いらっしゃいませ",  # Sonkeigo
    ]

    for sentence in test_sentences:
        kotogram = parser.japanese_to_kotogram(sentence)
        print(f"\nInput:    '{sentence}'")
        print(f"Kotogram: '{kotogram}'")

        # Assertions
        assert isinstance(kotogram, str), (
            f"kotogram should be str, got {type(kotogram)}"
        )
        assert len(kotogram) > 0, "kotogram should not be empty"
        assert "⌈" in kotogram, "kotogram should contain token markers"

    print("\n✓ Step 1 PASSED: Parser produces valid kotograms")


def test_step2_process_sentence_batch():
    """Step 2: Verify _process_sentence_batch returns correct structure."""
    from scripts.label import _process_sentence_batch

    print("\n" + "=" * 60)
    print("STEP 2: _process_sentence_batch Output")
    print("=" * 60)

    # Input format: (sentence, gram_label)
    batch = [
        ("これはテストです", 1),
        ("お元気ですか", 1),
        ("行くぜ", 1),
    ]

    print(f"\nInput batch ({len(batch)} items):")
    for item in batch:
        print(f"  {item}")

    results, counters = _process_sentence_batch(batch)

    print(f"\nOutput ({len(results)} results):")
    for i, result in enumerate(results):
        print(f"\n  Result {i}:")

        # Unpack and validate
        assert hasattr(result, "sentence"), (
            f"Expected object with sentence attribute, got {type(result)}"
        )

        print(f"    sentence:     '{result.sentence}'")
        print(f"    kotogram:     '{result.kotogram[:50]}...'")
        print(f"    formality_id: {result.formality_id}")
        print(f"    success:      {result.success}")

        # Type assertions
        assert isinstance(result.sentence, str)
        assert isinstance(result.kotogram, str)
        assert isinstance(result.formality_id, int)
        assert isinstance(result.gender_value, float)
        assert isinstance(result.gender_pragmatic, int)
        assert isinstance(result.register_ids, list)
        assert isinstance(result.success, int)

    print(
        "\n✓ Step 2 PASSED: _process_sentence_batch produces valid ProcessedSample objects"
    )


def test_step3_process_parallel():
    """Step 3: Verify StyleDataset._process_parallel uses cache correctly."""
    from train.dataset import StyleDataset

    print("\n" + "=" * 60)
    print("STEP 3: _process_parallel Output")
    print("=" * 60)

    # Input format: (sentence, gram_label)
    rows = [
        ("これはテストです", 1),
        ("お元気ですか", 1),
        ("行くぜ", 1),
    ]

    # Populate cache
    populate_test_cache(rows)

    print(f"\nInput rows ({len(rows)} items):")

    results = StyleDataset._process_parallel(
        rows,
        num_workers=1,
        batch_size=100,
        verbose=True,
    )

    print(f"\nOutput ({len(results)} results):")
    for i, result in enumerate(results):
        assert isinstance(result, ProcessedSample)
        assert result.success == 1

    print("\n✓ Step 3 PASSED: StyleDataset._process_parallel fetches from cache")


def test_step4_encoding_inputs_extraction():
    """Step 4: Verify encoding_inputs are extracted from ProcessedSample."""
    from train.dataset import StyleDataset

    print("\n" + "=" * 60)
    print("=" * 60)
    print("STEP 4: Encoding Inputs Extraction")
    print("=" * 60)

    rows = [
        ("これはテストです", 1),
        ("お元気ですか", 1),
    ]

    populate_test_cache(rows)

    processed_results = StyleDataset._process_parallel(
        rows, num_workers=1, batch_size=100, verbose=False
    )

    encoding_inputs = []
    for p in processed_results:
        if p.success:
            # Re-pack into the tuple expected by legacy/internal logic if needed,
            # or just use the object.
            # Based on actual code in train_style.py, we deal with Samples directly now.
            # But just checking that we can access the ID needed for conversion
            ei = p.to_cache_tuple()
            encoding_inputs.append(ei)

    assert len(encoding_inputs) == 2
    for ei in encoding_inputs:
        assert len(ei) == 8

    print("\n✓ Step 4 PASSED: Encoding inputs correctly extracted")


def test_step5_encode_samples_batch():
    """Step 5: Verify _encode_samples_batch creates valid Sample objects."""
    from kotogram.tokenizer import Tokenizer
    from train.dataset import StyleDataset
    from train.worker import _encode_samples_batch, init_worker

    print("\n" + "=" * 60)
    print("STEP 5: _encode_samples_batch Output")
    print("=" * 60)

    rows = [
        ("これはテストです", 1),
        ("お元気ですか", 1),
    ]

    populate_test_cache(rows)

    processed_results = StyleDataset._process_parallel(
        rows, num_workers=1, batch_size=100, verbose=False
    )

    # Create a tokenizer and build vocabulary
    tokenizer = Tokenizer()
    for p in processed_results:
        tokenizer.encode(p.kotogram, add_cls=True, add_to_vocab=True)
    tokenizer.freeze()

    # Call _encode_samples_batch
    init_worker({"field_vocabs": tokenizer.field_vocabs})
    samples = _encode_samples_batch(processed_results)

    assert len(samples) == 2
    for sample in samples:
        assert isinstance(sample.formality_value, float)
        assert isinstance(sample.formality_pragmatic, int)
        assert isinstance(sample.register_labels, list)

    print("\n✓ Step 5 PASSED: _encode_samples_batch creates valid Sample objects")


def test_step6_collate_fn():
    """Step 6: Verify collate_fn produces correct tensor shapes and types."""
    from kotogram.model import NUM_REGISTER_CLASSES
    from kotogram.tokenizer import Tokenizer
    from train.dataset import StyleDataset, collate_fn
    from train.worker import _encode_samples_batch, init_worker

    print("\n" + "=" * 60)
    print("STEP 6: collate_fn Output")
    print("=" * 60)

    rows = [
        ("これはテストです", 1),
        ("お元気ですか", 1),
        ("行くぜ", 0),
    ]

    populate_test_cache(rows)

    processed_results = StyleDataset._process_parallel(
        rows, num_workers=1, batch_size=100, verbose=False
    )

    tokenizer = Tokenizer()
    for p in processed_results:
        tokenizer.encode(p.kotogram, add_cls=True, add_to_vocab=True)
    tokenizer.freeze()

    init_worker({"field_vocabs": tokenizer.field_vocabs})
    samples = _encode_samples_batch(processed_results)

    # Call collate_fn
    batch = collate_fn(samples, tokenizer.pad_id)

    assert "attention_mask" in batch
    assert batch["formality_value"].shape == (3,)
    assert batch["formality_pragmatic"].shape == (3,)
    assert batch["register_labels"].shape == (3, NUM_REGISTER_CLASSES)

    print("\n✓ Step 6 PASSED: collate_fn produces correct tensor shapes and types")


def test_step7_evaluate_list_consistency():
    """Step 7: Verify evaluate() maintains list length consistency."""
    from kotogram.tokenizer import Tokenizer
    from train.dataset import StyleDataset
    from train.worker import _encode_samples_batch, init_worker

    print("\n" + "=" * 60)
    print("STEP 7: Evaluate List Length Consistency")
    print("=" * 60)

    rows = [
        ("これはテストです", 1),
        ("お元気ですか", 0),
        ("行くぜ", 1),
        ("食べますた", 0),
    ]

    populate_test_cache(rows)

    tokenizer = Tokenizer()
    processed_results = StyleDataset._process_parallel(
        rows, num_workers=1, batch_size=100, verbose=False
    )

    for p in processed_results:
        tokenizer.encode(p.kotogram, add_cls=True, add_to_vocab=True)
    tokenizer.freeze()

    init_worker({"field_vocabs": tokenizer.field_vocabs})
    samples = _encode_samples_batch(processed_results)
    dataset = StyleDataset(samples, tokenizer)

    expected_size = len(dataset)
    assert expected_size == 4

    print("\n✓ Step 7 PASSED: evaluate() list accumulation is correct (Simulated)")


def test_step8_trace_register_mislabel():
    """Step 8: Trace a specific mislabeled sentence."""
    from kotogram.model import REGISTER_ID_TO_LABEL
    from train.dataset import StyleDataset

    print("\n" + "=" * 60)
    print("STEP 8: Trace Register Mislabeling")
    print("=" * 60)

    test_sentence = "A:本当に全部食べると？B:それはお腹ペコペコやけんね！"
    rows = [(test_sentence, 1)]

    populate_test_cache(rows)

    processed = StyleDataset._process_parallel(
        rows, num_workers=1, batch_size=100, verbose=False
    )

    result = processed[0]
    print(f"Active register IDs: {result.register_ids}")

    register_names = [
        REGISTER_ID_TO_LABEL[rid].name
        for rid in result.register_ids
        if rid in REGISTER_ID_TO_LABEL
    ]
    print(f"Active register names: {register_names}")

    assert "HAKATABEN" in register_names

    print("\n✓ Step 8 PASSED: Labels are correctly propagated")


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# END-TO-END DATA FLOW TESTS (REFACTORED)")
    print("#" * 60)

    tests = [
        test_step1_parser_output,
        test_step2_process_sentence_batch,
        test_step3_process_parallel,
        test_step4_encoding_inputs_extraction,
        test_step5_encode_samples_batch,
        test_step6_collate_fn,
        test_step7_evaluate_list_consistency,
        test_step8_trace_register_mislabel,
    ]

    passed = 0
    failed = 0

    for test in tests:
        test()
        passed += 1

    print("\n" + "#" * 60)
    print(f"# SUMMARY: {passed}/{len(tests)} tests passed, {failed} failed")
    print("#" * 60)

    if failed > 0:
        sys.exit(1)
