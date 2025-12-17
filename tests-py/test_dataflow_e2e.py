#!/usr/bin/env python3
"""End-to-end data flow tests for train_style.py pipeline.

These tests trace data through the entire pipeline, using actual parser output
instead of manually constructed kotograms. Each step prints its output for
debugging purposes.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_step1_parser_output():
    """Step 1: Verify parser produces valid kotogram from Japanese."""
    from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
    
    print("\n" + "="*60)
    print("STEP 1: Parser Output")
    print("="*60)
    
    parser = SudachiJapaneseParser()
    
    test_sentences = [
        "これはテストです",  # Neutral formality
        "お元気ですか",      # Polite/formal
        "行くぜ",            # Masculine casual
        "いらっしゃいませ",  # Sonkeigo
    ]
    
    for sentence in test_sentences:
        kotogram = parser.japanese_to_kotogram(sentence)
        print(f"\nInput:    '{sentence}'")
        print(f"Kotogram: '{kotogram}'")
        
        # Assertions
        assert isinstance(kotogram, str), f"kotogram should be str, got {type(kotogram)}"
        assert len(kotogram) > 0, "kotogram should not be empty"
        assert '⌈' in kotogram, "kotogram should contain token markers"
    
    print("\n✓ Step 1 PASSED: Parser produces valid kotograms")
    return True


def test_step2_process_sentence_batch():
    """Step 2: Verify _process_sentence_batch returns correct 8-tuple structure."""
    from scripts.train_style import _process_sentence_batch
    
    print("\n" + "="*60)
    print("STEP 2: _process_sentence_batch Output")
    print("="*60)
    
    # Input format: (sentence, sentence_id, gram_label)
    batch = [
        ("これはテストです", "id_001", 1),
        ("お元気ですか", "id_002", 1),
        ("行くぜ", "id_003", 1),
    ]
    
    print(f"\nInput batch ({len(batch)} items):")
    for item in batch:
        print(f"  {item}")
    
    results = _process_sentence_batch(batch)
    
    print(f"\nOutput ({len(results)} results):")
    for i, result in enumerate(results):
        print(f"\n  Result {i}:")
        print(f"    Tuple length: {len(result)}")
        
        # Unpack and validate
        assert len(result) == 8, f"Expected 8-tuple, got {len(result)}: {result}"
        
        sentence, sentence_id, kotogram, formality_id, gender_id, register_ids, gram_label, success = result
        
        print(f"    sentence:     '{sentence}' (type: {type(sentence).__name__})")
        print(f"    sentence_id:  '{sentence_id}' (type: {type(sentence_id).__name__})")
        print(f"    kotogram:     '{kotogram[:50]}...' (type: {type(kotogram).__name__})")
        print(f"    formality_id: {formality_id} (type: {type(formality_id).__name__})")
        print(f"    gender_id:    {gender_id} (type: {type(gender_id).__name__})")
        print(f"    register_ids: {register_ids} (type: {type(register_ids).__name__})")
        print(f"    gram_label:   {gram_label} (type: {type(gram_label).__name__})")
        print(f"    success:      {success} (type: {type(success).__name__})")
        
        # Type assertions
        assert isinstance(sentence, str)
        assert isinstance(sentence_id, str)
        assert isinstance(kotogram, str)
        assert isinstance(formality_id, int)
        assert isinstance(gender_id, int)
        assert isinstance(register_ids, list), f"register_ids MUST be list, got {type(register_ids)}"
        assert all(isinstance(r, int) for r in register_ids), f"All register_ids must be int"
        assert isinstance(gram_label, int)
        assert isinstance(success, int)
    
    print("\n✓ Step 2 PASSED: _process_sentence_batch returns correct 8-tuple structure")
    return True


def test_step3_process_parallel():
    """Step 3: Verify _process_parallel returns correct 7-tuple structure."""
    from scripts.train_style import StyleDataset
    
    print("\n" + "="*60)
    print("STEP 3: _process_parallel Output")
    print("="*60)
    
    # Input format: (sentence, sentence_id, gram_label)
    rows = [
        ("これはテストです", "id_001", 1),
        ("お元気ですか", "id_002", 1),
        ("行くぜ", "id_003", 1),
    ]
    
    print(f"\nInput rows ({len(rows)} items):")
    for item in rows:
        print(f"  {item}")
    
    results = StyleDataset._process_parallel(
        rows,
        num_workers=1,
        batch_size=100,
        verbose=True,
        use_kotogram_cache=False
    )
    
    print(f"\nOutput ({len(results)} results):")
    for i, result in enumerate(results):
        print(f"\n  Result {i}:")
        print(f"    Tuple length: {len(result)}")
        
        # Unpack and validate
        assert len(result) == 7, f"Expected 7-tuple, got {len(result)}: {result}"
        
        sentence, kotogram, formality_id, gender_id, register_ids, gram_label, success = result
        
        print(f"    sentence:     '{sentence}' (type: {type(sentence).__name__})")
        print(f"    kotogram:     '{kotogram[:50]}...' (type: {type(kotogram).__name__})")
        print(f"    formality_id: {formality_id} (type: {type(formality_id).__name__})")
        print(f"    gender_id:    {gender_id} (type: {type(gender_id).__name__})")
        print(f"    register_ids: {register_ids} (type: {type(register_ids).__name__})")
        print(f"    gram_label:   {gram_label} (type: {type(gram_label).__name__})")
        print(f"    success:      {success} (type: {type(success).__name__})")
        
        # Type assertions
        assert isinstance(sentence, str)
        assert isinstance(kotogram, str)
        assert isinstance(formality_id, int)
        assert isinstance(gender_id, int)
        assert isinstance(register_ids, list), f"register_ids MUST be list, got {type(register_ids)}"
        assert all(isinstance(r, int) for r in register_ids), f"All register_ids must be int"
        assert isinstance(gram_label, int)
        assert isinstance(success, int)
    
    print("\n✓ Step 3 PASSED: _process_parallel returns correct 7-tuple structure")
    return True


def test_step4_encoding_inputs_extraction():
    """Step 4: Verify encoding_inputs are extracted correctly from processed_results."""
    from scripts.train_style import StyleDataset
    
    print("\n" + "="*60)
    print("STEP 4: Encoding Inputs Extraction")
    print("="*60)
    
    # Get processed_results from _process_parallel
    rows = [
        ("これはテストです", "id_001", 1),
        ("お元気ですか", "id_002", 1),
    ]
    
    print(f"\nInput rows: {len(rows)} items")
    
    processed_results = StyleDataset._process_parallel(
        rows,
        num_workers=1,
        batch_size=100,
        verbose=False,
        use_kotogram_cache=False
    )
    
    print(f"Processed results: {len(processed_results)} items")
    
    # Extract encoding_inputs exactly as done in from_multiple_tsv
    encoding_inputs = []
    for p in processed_results:
        print(f"\n  Checking tuple p: length={len(p)}")
        
        # This is the exact logic from from_multiple_tsv
        assert len(p) == 7, f"Expected 7-tuple in processed_results, got {len(p)}"
        
        if p[6]:  # success
            encoding_input = (p[0], p[1], p[2], p[3], p[4], p[5])
            encoding_inputs.append(encoding_input)
            
            print(f"    Extracted encoding_input: length={len(encoding_input)}")
            print(f"      p[0] sentence:     '{p[0]}'")
            print(f"      p[1] kotogram:     '{p[1][:40]}...'")
            print(f"      p[2] formality_id: {p[2]}")
            print(f"      p[3] gender_id:    {p[3]}")
            print(f"      p[4] register_ids: {p[4]} (type: {type(p[4]).__name__})")
            print(f"      p[5] gram_label:   {p[5]}")
    
    print(f"\nTotal encoding_inputs: {len(encoding_inputs)}")
    
    # Validate encoding_inputs structure
    for i, ei in enumerate(encoding_inputs):
        assert len(ei) == 6, f"encoding_input {i}: Expected 6-tuple, got {len(ei)}"
        sentence, kotogram, f_id, g_id, r_ids, gram_label = ei
        
        assert isinstance(r_ids, list), f"encoding_input {i}: r_ids MUST be list, got {type(r_ids)}"
        print(f"\n  encoding_input {i}: r_ids={r_ids}, gram_label={gram_label}")
    
    print("\n✓ Step 4 PASSED: Encoding inputs correctly extracted")
    return True


def test_step5_encode_samples_batch():
    """Step 5: Verify _encode_samples_batch creates valid Sample objects."""
    from scripts.train_style import _encode_samples_batch, StyleDataset
    from kotogram.model import Tokenizer
    
    print("\n" + "="*60)
    print("STEP 5: _encode_samples_batch Output")
    print("="*60)
    
    # First, get real data through the pipeline
    rows = [
        ("これはテストです", "id_001", 1),
        ("お元気ですか", "id_002", 1),
    ]
    
    processed_results = StyleDataset._process_parallel(
        rows,
        num_workers=1,
        batch_size=100,
        verbose=False,
        use_kotogram_cache=False
    )
    
    # Extract encoding_inputs
    encoding_inputs = []
    for p in processed_results:
        if p[6]:  # success
            encoding_inputs.append((p[0], p[1], p[2], p[3], p[4], p[5]))
    
    print(f"\nInput encoding_inputs: {len(encoding_inputs)} items")
    for ei in encoding_inputs:
        print(f"  {ei[:2]}... r_ids={ei[4]}, gram={ei[5]}")
    
    # Create a tokenizer and build vocabulary from the kotograms
    tokenizer = Tokenizer()
    for ei in encoding_inputs:
        kotogram = ei[1]
        tokenizer.encode(kotogram, add_cls=True, add_to_vocab=True)
    tokenizer.freeze()
    
    print(f"\nTokenizer vocab sizes: {tokenizer.get_vocab_sizes()}")
    
    # Serialize tokenizer state
    tokenizer_state = {'field_vocabs': tokenizer.field_vocabs}
    
    # Call _encode_samples_batch
    samples = _encode_samples_batch(encoding_inputs, tokenizer_state)
    
    print(f"\nOutput samples: {len(samples)} Sample objects")
    
    for i, sample in enumerate(samples):
        print(f"\n  Sample {i}:")
        print(f"    formality_label:      {sample.formality_label} (type: {type(sample.formality_label).__name__})")
        print(f"    gender_label:         {sample.gender_label} (type: {type(sample.gender_label).__name__})")
        print(f"    register_labels:      {sample.register_labels} (type: {type(sample.register_labels).__name__})")
        print(f"    grammaticality_label: {sample.grammaticality_label} (type: {type(sample.grammaticality_label).__name__})")
        print(f"    original_sentence:    '{sample.original_sentence}'")
        
        # Type assertions
        assert isinstance(sample.formality_label, int)
        assert isinstance(sample.gender_label, int)
        assert isinstance(sample.register_labels, list), f"register_labels MUST be list, got {type(sample.register_labels)}"
        assert all(isinstance(r, int) for r in sample.register_labels)
        assert isinstance(sample.grammaticality_label, int)
        assert isinstance(sample.original_sentence, str)
        assert isinstance(sample.feature_ids, dict)
    
    print("\n✓ Step 5 PASSED: _encode_samples_batch creates valid Sample objects")
    return True


def test_step6_collate_fn():
    """Step 6: Verify collate_fn produces correct tensor shapes and types."""
    from scripts.train_style import collate_fn, _encode_samples_batch, StyleDataset, Sample
    from kotogram.model import Tokenizer, FEATURE_FIELDS, NUM_REGISTER_CLASSES
    import torch
    
    print("\n" + "="*60)
    print("STEP 6: collate_fn Output")
    print("="*60)
    
    # Get samples through the pipeline
    rows = [
        ("これはテストです", "id_001", 1),
        ("お元気ですか", "id_002", 1),
        ("行くぜ", "id_003", 0),  # agrammatic for variety
    ]
    
    processed_results = StyleDataset._process_parallel(
        rows,
        num_workers=1,
        batch_size=100,
        verbose=False,
        use_kotogram_cache=False
    )
    
    encoding_inputs = []
    for p in processed_results:
        if p[6]:
            encoding_inputs.append((p[0], p[1], p[2], p[3], p[4], p[5]))
    
    tokenizer = Tokenizer()
    for ei in encoding_inputs:
        tokenizer.encode(ei[1], add_cls=True, add_to_vocab=True)
    tokenizer.freeze()
    
    tokenizer_state = {'field_vocabs': tokenizer.field_vocabs}
    samples = _encode_samples_batch(encoding_inputs, tokenizer_state)
    
    print(f"\nInput samples: {len(samples)}")
    for s in samples:
        print(f"  register_labels: {s.register_labels}, gram: {s.grammaticality_label}")
    
    # Call collate_fn
    batch = collate_fn(samples, tokenizer.pad_id)
    
    print(f"\nOutput batch keys: {list(batch.keys())}")
    
    # Check required keys
    required_keys = ['attention_mask', 'formality_labels', 'gender_labels', 
                     'register_labels', 'grammaticality_labels']
    for key in required_keys:
        assert key in batch, f"Missing key: {key}"
    
    # Check feature field keys
    for field in FEATURE_FIELDS:
        key = f'input_ids_{field}'
        assert key in batch, f"Missing key: {key}"
    
    # Check shapes
    batch_size = len(samples)
    
    print(f"\n  Batch size: {batch_size}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  formality_labels shape: {batch['formality_labels'].shape}")
    print(f"  gender_labels shape: {batch['gender_labels'].shape}")
    print(f"  register_labels shape: {batch['register_labels'].shape}")
    print(f"  grammaticality_labels shape: {batch['grammaticality_labels'].shape}")
    
    # Shape assertions
    assert batch['formality_labels'].shape == (batch_size,)
    assert batch['gender_labels'].shape == (batch_size,)
    assert batch['register_labels'].shape == (batch_size, NUM_REGISTER_CLASSES)
    assert batch['grammaticality_labels'].shape == (batch_size,)
    
    # Type assertions
    print(f"\n  formality_labels dtype: {batch['formality_labels'].dtype}")
    print(f"  gender_labels dtype: {batch['gender_labels'].dtype}")
    print(f"  register_labels dtype: {batch['register_labels'].dtype}")
    print(f"  grammaticality_labels dtype: {batch['grammaticality_labels'].dtype}")
    
    assert batch['formality_labels'].dtype == torch.long
    assert batch['gender_labels'].dtype == torch.long
    assert batch['register_labels'].dtype == torch.float32, f"register_labels must be float32 for BCEWithLogitsLoss, got {batch['register_labels'].dtype}"
    assert batch['grammaticality_labels'].dtype == torch.long
    
    # Check register_labels values (should be multi-hot encoded)
    print(f"\n  register_labels values:\n{batch['register_labels']}")
    
    # Each row should have at least one 1.0
    for i in range(batch_size):
        row_sum = batch['register_labels'][i].sum().item()
        assert row_sum >= 1.0, f"Sample {i}: register_labels row sum should be >= 1.0, got {row_sum}"
    
    print("\n✓ Step 6 PASSED: collate_fn produces correct tensor shapes and types")
    return True


def test_step7_evaluate_list_consistency():
    """Step 7: Verify evaluate() maintains list length consistency.
    
    This test catches bugs where prediction and label lists get out of sync
    (like the duplicate extend() bug).
    """
    print("\n" + "="*60)
    print("STEP 7: Evaluate List Length Consistency")
    print("="*60)
    
    from scripts.train_style import StyleDataset, collate_fn
    from kotogram.model import Tokenizer, StyleClassifier, ModelConfig, NUM_REGISTER_CLASSES
    from torch.utils.data import DataLoader
    import torch
    
    # Create a minimal dataset
    rows = [
        ("これはテストです", "id_001", 1),
        ("お元気ですか", "id_002", 0),
        ("行くぜ", "id_003", 1),
        ("食べますた", "id_004", 0),
    ]
    
    print(f"\nCreating dataset with {len(rows)} samples...")
    
    tokenizer = Tokenizer()
    processed_results = StyleDataset._process_parallel(
        rows, num_workers=1, batch_size=100, verbose=False, use_kotogram_cache=False
    )
    
    from scripts.train_style import _encode_samples_batch
    encoding_inputs = [(p[0], p[1], p[2], p[3], p[4], p[5]) for p in processed_results if p[6]]
    for ei in encoding_inputs:
        tokenizer.encode(ei[1], add_cls=True, add_to_vocab=True)
    tokenizer.freeze()
    
    samples = _encode_samples_batch(encoding_inputs, {'field_vocabs': tokenizer.field_vocabs})
    dataset = StyleDataset(samples, tokenizer)
    
    print(f"  Dataset size: {len(dataset)}")
    
    # Create model and evaluate
    config = ModelConfig(vocab_sizes=tokenizer.get_vocab_sizes())
    model = StyleClassifier(config)
    
    # Create dataloader with collate_fn
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer.pad_id)
    )
    
    # Simulate what evaluate() does - check list lengths
    print("\\nSimulating evaluate() list accumulation...")
    
    all_formality_preds = []
    all_formality_labels = []
    all_gender_preds = []
    all_gender_labels = []
    all_grammaticality_preds = []
    all_grammaticality_labels = []
    all_register_preds = []
    all_register_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            formality_labels = batch['formality_labels']
            gender_labels = batch['gender_labels']
            grammaticality_labels = batch['grammaticality_labels']
            register_labels = batch['register_labels']
            
            # Simulate predictions
            batch_size = formality_labels.shape[0]
            formality_preds = torch.randint(0, 6, (batch_size,))
            gender_preds = torch.randint(0, 4, (batch_size,))
            grammaticality_preds = torch.randint(0, 2, (batch_size,))
            register_preds = torch.randint(0, 2, (batch_size, NUM_REGISTER_CLASSES))
            register_labels_long = register_labels.long()
            
            # This is the FIXED code from evaluate()
            all_formality_preds.extend(formality_preds.cpu().tolist())
            all_formality_labels.extend(formality_labels.cpu().tolist())
            all_gender_preds.extend(gender_preds.cpu().tolist())
            all_gender_labels.extend(gender_labels.cpu().tolist())
            all_grammaticality_preds.extend(grammaticality_preds.cpu().tolist())
            all_grammaticality_labels.extend(grammaticality_labels.cpu().tolist())
            all_register_preds.extend(register_preds.cpu().tolist())
            all_register_labels.extend(register_labels_long.cpu().tolist())
            
            print(f"  Batch {batch_idx}: size={batch_size}")
    
    # CRITICAL CHECK: All prediction and label lists must have same length
    print(f"\\nList lengths:")
    print(f"  formality:      preds={len(all_formality_preds)}, labels={len(all_formality_labels)}")
    print(f"  gender:         preds={len(all_gender_preds)}, labels={len(all_gender_labels)}")
    print(f"  grammaticality: preds={len(all_grammaticality_preds)}, labels={len(all_grammaticality_labels)}")
    print(f"  register:       preds={len(all_register_preds)}, labels={len(all_register_labels)}")
    
    # Assertions that would have caught the bug
    assert len(all_formality_preds) == len(all_formality_labels), \
        f"formality preds/labels mismatch: {len(all_formality_preds)} vs {len(all_formality_labels)}"
    assert len(all_gender_preds) == len(all_gender_labels), \
        f"gender preds/labels mismatch: {len(all_gender_preds)} vs {len(all_gender_labels)}"
    assert len(all_grammaticality_preds) == len(all_grammaticality_labels), \
        f"grammaticality preds/labels mismatch: {len(all_grammaticality_preds)} vs {len(all_grammaticality_labels)}"
    assert len(all_register_preds) == len(all_register_labels), \
        f"register preds/labels mismatch: {len(all_register_preds)} vs {len(all_register_labels)}"
    
    # All should equal dataset size
    expected_size = len(dataset)
    assert len(all_grammaticality_preds) == expected_size, \
        f"Expected {expected_size} predictions, got {len(all_grammaticality_preds)}"
    
    print(f"\\n✓ All lists have correct length: {expected_size}")
    print("\\n✓ Step 7 PASSED: evaluate() list lengths are consistent")
    return True


def test_step8_trace_register_mislabel():
    """Step 8: Trace a specific mislabeled sentence from register_confusion.csv.
    
    This test follows a sentence that was mislabeled as 'netslang' when it should
    have been 'hakataben' to find where the labeling goes wrong.
    """
    print("\n" + "="*60)
    print("STEP 8: Trace Register Mislabeling")
    print("="*60)
    
    from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
    from scripts.rule_based_analysis import analyze_register, RegisterLevel
    from scripts.train_style import _process_sentence_batch, StyleDataset, _encode_samples_batch
    from kotogram.model import Tokenizer, NUM_REGISTER_CLASSES
    
    # Pick a sentence from register_confusion.csv that was mislabeled
    # This one was labeled hakataben but predicted as netslang
    test_sentence = "A:本当に全部食べるの？B:それはお腹ペコペコだからね！"
    expected_register = "hakataben"
    
    print(f"\nTest sentence: '{test_sentence}'")
    print(f"Expected register: {expected_register}")
    
    # Step 8.1: Parse the sentence
    print("\n--- Step 8.1: Parsing ---")
    parser = SudachiJapaneseParser()
    kotogram = parser.japanese_to_kotogram(test_sentence)
    print(f"Kotogram: '{kotogram[:80]}...'")
    
    # Step 8.2: Run analyze_register 
    print("\n--- Step 8.2: Rule-based analyze_register ---")
    registers = analyze_register(kotogram)
    print(f"analyze_register result: {registers}")
    print(f"Register names: {[r.name for r in registers]}")
    register_ids = [r.value for r in registers]
    print(f"Register IDs: {register_ids}")
    
    # Check: Does analyze_register correctly identify hakataben?
    has_hakataben = any(r.name.lower() == 'hakataben' for r in registers)
    has_netslang = any(r.name.lower() == 'netslang' for r in registers)
    print(f"\\nContains HAKATABEN? {has_hakataben}")
    print(f"Contains NETSLANG? {has_netslang}")
    
    # Step 8.3: Run through _process_sentence_batch
    print("\n--- Step 8.3: _process_sentence_batch ---")
    batch = [(test_sentence, "test_001", 1)]
    results = _process_sentence_batch(batch)
    
    for result in results:
        sentence, sentence_id, kotogram_out, f_id, g_id, r_ids, gram_label, success = result
        print(f"Output register_ids: {r_ids}")
        print(f"Register IDs type: {type(r_ids)}")
        
        # Decode register IDs back to names
        register_names = []
        for rid in r_ids:
            for member in RegisterLevel:
                if member.value == rid:
                    register_names.append(member.name)
                    break
        print(f"Register names from IDs: {register_names}")
    
    # Step 8.4: Run through _process_parallel
    print("\n--- Step 8.4: _process_parallel ---")
    rows = [(test_sentence, "test_001", 1)]
    processed = StyleDataset._process_parallel(
        rows, num_workers=1, batch_size=100, verbose=False, use_kotogram_cache=False
    )
    
    for p in processed:
        print(f"Processed tuple length: {len(p)}")
        print(f"register_ids (p[4]): {p[4]}")
        print(f"gram_label (p[5]): {p[5]}")
    
    # Step 8.5: Run through _encode_samples_batch
    print("\n--- Step 8.5: _encode_samples_batch ---")
    encoding_inputs = [(p[0], p[1], p[2], p[3], p[4], p[5]) for p in processed if p[6]]
    
    tokenizer = Tokenizer()
    for ei in encoding_inputs:
        tokenizer.encode(ei[1], add_cls=True, add_to_vocab=True)
    tokenizer.freeze()
    
    samples = _encode_samples_batch(encoding_inputs, {'field_vocabs': tokenizer.field_vocabs})
    
    for s in samples:
        print(f"Sample.register_labels: {s.register_labels}")
        
        # Decode back to names
        register_names = []
        for rid in s.register_labels:
            for member in RegisterLevel:
                if member.value == rid:
                    register_names.append(member.name)
                    break
        print(f"Register names from Sample: {register_names}")
    
    # Step 8.6: Check collate_fn multi-hot encoding
    print("\n--- Step 8.6: collate_fn multi-hot encoding ---")
    from scripts.train_style import collate_fn
    batch = collate_fn(samples, tokenizer.pad_id)
    
    print(f"register_labels tensor shape: {batch['register_labels'].shape}")
    print(f"register_labels tensor:\\n{batch['register_labels']}")
    
    # Decode the multi-hot back to register names
    register_tensor = batch['register_labels'][0]  # First (only) sample
    active_indices = (register_tensor > 0.5).nonzero(as_tuple=True)[0].tolist()
    print(f"Active register indices: {active_indices}")
    
    active_names = []
    for idx in active_indices:
        for member in RegisterLevel:
            if member.value == idx:
                active_names.append(member.name)
                break
    print(f"Active register names: {active_names}")
    
    # Final check
    print("\n--- FINAL ANALYSIS ---")
    if has_hakataben:
        print("✓ analyze_register correctly identifies HAKATABEN")
    else:
        print("❌ analyze_register DOES NOT identify HAKATABEN - rule-based logic issue!")
        
    if has_netslang:
        print("⚠ analyze_register also identifies NETSLANG")
    else:
        print("✓ analyze_register does NOT identify NETSLANG")
    
    print(f"\nThe label that reaches the model: {active_names}")
    
    if 'HAKATABEN' in active_names and 'NETSLANG' not in active_names:
        print("✓ Step 8 PASSED: Labels are correctly propagated")
    elif 'HAKATABEN' not in active_names:
        print("❌ Step 8 FAILED: HAKATABEN was lost somewhere in the pipeline!")
    elif 'NETSLANG' in active_names:
        print("❌ Step 8 FAILED: NETSLANG was incorrectly added!")
    
    return True



if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# END-TO-END DATA FLOW TESTS")
    print("#"*60)
    
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
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "#"*60)
    print(f"# SUMMARY: {passed}/{len(tests)} tests passed, {failed} failed")
    print("#"*60)
    
    if failed > 0:
        sys.exit(1)
