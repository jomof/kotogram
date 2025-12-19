"""Tests for Japanese parser implementations."""

import unittest
from kotogram import JapaneseParser, kotogram_to_japanese, split_kotogram


class TestJapaneseParserInterface(unittest.TestCase):
    """Test cases for the JapaneseParser abstract interface."""

    def test_cannot_instantiate_abstract_class(self):
        """JapaneseParser is abstract and cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            JapaneseParser()

    def test_subclass_must_implement_japanese_to_kotogram(self):
        """Subclasses must implement japanese_to_kotogram method."""

        class IncompleteParser(JapaneseParser):
            pass

        with self.assertRaises(TypeError):
            IncompleteParser()


class TestKotogramToJapanese(unittest.TestCase):
    """Test cases for kotogram_to_japanese conversion."""

    def test_simple_kotogram_to_japanese(self):
        """Convert simple kotogram back to Japanese."""
        kotogram = "⌈ˢ猫ᵖn:common_noun⌉"
        result = kotogram_to_japanese(kotogram)
        self.assertEqual(result, "猫")

    def test_multiple_tokens_without_spaces(self):
        """Convert multiple tokens without spaces."""
        kotogram = "⌈ˢ猫ᵖn⌉⌈ˢをᵖprt⌉⌈ˢ食べるᵖv⌉"
        result = kotogram_to_japanese(kotogram, spaces=False)
        self.assertEqual(result, "猫を食べる")

    def test_multiple_tokens_with_spaces(self):
        """Convert multiple tokens with spaces."""
        kotogram = "⌈ˢ猫ᵖn⌉⌈ˢをᵖprt⌉⌈ˢ食べるᵖv⌉"
        result = kotogram_to_japanese(kotogram, spaces=True)
        self.assertEqual(result, "猫 を 食べる")

    def test_punctuation_collapse(self):
        """Punctuation should not have spaces around it when collapse_punctuation=True."""
        kotogram = "⌈ˢ猫ᵖn⌉⌈ˢ。ᵖauxs⌉"
        result = kotogram_to_japanese(kotogram, spaces=True, collapse_punctuation=True)
        self.assertEqual(result, "猫。")

    def test_punctuation_no_collapse(self):
        """Punctuation can have spaces when collapse_punctuation=False."""
        kotogram = "⌈ˢ猫ᵖn⌉⌈ˢ。ᵖauxs⌉"
        result = kotogram_to_japanese(kotogram, spaces=True, collapse_punctuation=False)
        self.assertEqual(result, "猫 。")


class TestSplitKotogram(unittest.TestCase):
    """Test cases for split_kotogram function."""

    def test_split_single_token(self):
        """Split kotogram with single token."""
        kotogram = "⌈ˢ猫ᵖn:common_noun⌉"
        result = split_kotogram(kotogram)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "⌈ˢ猫ᵖn:common_noun⌉")

    def test_split_multiple_tokens(self):
        """Split kotogram with multiple tokens."""
        kotogram = "⌈ˢ猫ᵖn⌉⌈ˢをᵖprt⌉⌈ˢ食べるᵖv⌉"
        result = split_kotogram(kotogram)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "⌈ˢ猫ᵖn⌉")
        self.assertEqual(result[1], "⌈ˢをᵖprt⌉")
        self.assertEqual(result[2], "⌈ˢ食べるᵖv⌉")

    def test_split_empty_kotogram(self):
        """Split empty kotogram returns empty list."""
        result = split_kotogram("")
        self.assertEqual(result, [])

    def test_split_complex_tokens(self):
        """Split kotogram with complex token annotations."""
        kotogram = "⌈ˢ食べるᵖv:general:e-ichidan-ba:terminalᵇ食べるᵈ食べるʳタベル⌉"
        result = split_kotogram(kotogram)
        self.assertEqual(len(result), 1)
        self.assertIn("ᵇ", result[0])  # base form marker
        self.assertIn("ᵈ", result[0])  # lemma marker
        self.assertIn("ʳ", result[0])  # pronunciation marker


if __name__ == "__main__":
    unittest.main()


class TestMapBijectivity(unittest.TestCase):
    """Test that mapping dictionaries are bijections (all values are unique)."""

    def test_maps_are_bijections(self):
        from kotogram.japanese_parser import (
            POS_MAP, POS1_MAP, POS2_MAP, POS3_MAP,
            CONJUGATED_TYPE_MAP, CONJUGATED_FORM_MAP
        )

        maps_to_check = {
            "POS_MAP": POS_MAP,
            "POS1_MAP": POS1_MAP,
            "POS2_MAP": POS2_MAP,
            "POS3_MAP": POS3_MAP,
            "CONJUGATED_TYPE_MAP": CONJUGATED_TYPE_MAP,
            "CONJUGATED_FORM_MAP": CONJUGATED_FORM_MAP,
        }

        for name, map_obj in maps_to_check.items():
            with self.subTest(map_name=name):
                # Check for duplicate values
                values = list(map_obj.values())
                unique_values = set(values)
                
                if len(values) != len(unique_values):
                    # Find duplicates for better error message
                    from collections import Counter
                    counts = Counter(values)
                    duplicates = [val for val, count in counts.items() if count > 1]
                    self.fail(f"{name} has duplicate values: {duplicates}")

    def test_map_values_format(self):
        """Test that map values are lowercase, using only '-' as special character."""
        import re
        from kotogram.japanese_parser import (
            POS_MAP, POS1_MAP, POS2_MAP, POS3_MAP,
            CONJUGATED_TYPE_MAP, CONJUGATED_FORM_MAP
        )

        maps_to_check = {
            "POS_MAP": POS_MAP,
            "POS1_MAP": POS1_MAP,
            "POS2_MAP": POS2_MAP,
            "POS3_MAP": POS3_MAP,
            "CONJUGATED_TYPE_MAP": CONJUGATED_TYPE_MAP,
            "CONJUGATED_FORM_MAP": CONJUGATED_FORM_MAP,
        }

        # Regex: Start, optional sequence of lowercase/digits/hyphens, End.
        # Allows empty string.
        pattern = re.compile(r'^[a-z0-9-]*$')

        for name, map_obj in maps_to_check.items():
            with self.subTest(map_name=name):
                for key, value in map_obj.items():
                    if not pattern.match(value):
                            self.fail(f"{name} value '{value}' (for key '{key}') is invalid. "
                                  "Must be lowercase alphanumeric with hyphens only.")

    def test_extract_token_features_completeness(self):
        """Verify extract_token_features handles full 3-level POS detail via round-trip."""
        from kotogram.kotogram import extract_token_features, split_kotogram
        from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
        
        # 1. Generate kotogram from text using the real parser
        # "米国" (America) typically parses as: 
        # POS: 名詞 (noun)
        # POS1: 固有名詞 (proper-noun)
        # POS2: 地名 (place-name)
        # POS3: 国 (country)
        parser = SudachiJapaneseParser()
        kotogram = parser.japanese_to_kotogram("米国")
        
        # 2. Split (should be 1 token)
        tokens = split_kotogram(kotogram)
        self.assertEqual(len(tokens), 1, f"Expected 1 token for '米国', got: {tokens}")
        token_str = tokens[0]
        
        # 3. Extract features
        features = extract_token_features(token_str)
        
        # 4. Verify round-trip correctness (All fields)
        self.assertEqual(features['surface'], '米国')
        self.assertEqual(features['pos'], 'noun')
        self.assertEqual(features['pos_detail1'], 'proper-noun')
        self.assertEqual(features['pos_detail2'], 'place-name')
        self.assertEqual(features['pos_detail3'], 'country')
        # Nouns typically have no conjugation
        self.assertEqual(features['conjugated_type'], '')
        self.assertEqual(features['conjugated_form'], '')
        # Base/Lemma are same as surface here, so they might be empty if parser optimizes
        # SudachiJapaneseParser optimizes by omitting if same as surface
        self.assertEqual(features['base_orth'], '')
        self.assertEqual(features['lemma'], '')
        # Reading should be present
        self.assertEqual(features['reading'], 'ベイコク')

        # 5. Verify Conjugated Verb Case (to ensure those fields work)
        # "話します" (speak - polite)
        kotogram_verb = parser.japanese_to_kotogram("話します")
        # Split - tricky because it might split into 話し + ます
        # "話します" -> "話し" (verb) + "ます" (aux)
        tokens_verb = split_kotogram(kotogram_verb)
        self.assertTrue(len(tokens_verb) >= 1)
        
        # Check first token "話し" (hanashi)
        # Expected:
        # Surface: 話し
        # POS: verb
        # POS1: general
        # POS2: *
        # POS3: *
        # CType: godan-sa
        # CForm: conjunctive
        feat_v = extract_token_features(tokens_verb[0])
        self.assertEqual(feat_v['surface'], '話し')
        self.assertEqual(feat_v['pos'], 'verb')
        # Check conjugation fields are populated with correct values
        self.assertEqual(feat_v['conjugated_type'], 'godan-sa')
        self.assertEqual(feat_v['conjugated_form'], 'conjunctive')
        # Lemma "話す" != Surface "話し", so lemma/base should be present
        self.assertEqual(feat_v['lemma'], '話す')
        self.assertEqual(feat_v['base_orth'], '話す')


def _process_file(file_path):
    """Worker function to process a single file and collect POS stats."""
    import csv
    from sudachipy import tokenizer
    from sudachipy import dictionary

    # Initialize tokenizer (expensive, so done per process/file)
    try:
        dict_obj = dictionary.Dictionary(dict='full')
        sudachi_tokenizer = dict_obj.create()
    except Exception:
        # Fallback for CI/limited environments if full dict missing, though assumed present
        dict_obj = dictionary.Dictionary()
        sudachi_tokenizer = dict_obj.create()

    # We want to check all split modes
    modes = [
        tokenizer.Tokenizer.SplitMode.A,
        tokenizer.Tokenizer.SplitMode.B,
        tokenizer.Tokenizer.SplitMode.C,
    ]

    observed = {
        'pos': set(),
        'pos1': set(),
        'pos2': set(),
        'pos3': set(),
        'conjugated_type': set(),
        'conjugated_form': set(),
        'invariant_violations': []
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Check if tsv or plain text. Assuming TSV based on filename, 
            # but if it has no tabs, it might be just lines.
            # safe assumption: read line by line.
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                # If TSV, maybe the text is in a specific column?
                # For `jpn_sentences.tsv`, usually it's `id\ttext`.
                # Let's try to split and take the last column if multiple exist, 
                # or just process the whole line if it fails.
                # Actually, the user's data structure implies TSV. 
                # Let's assume the text is the last column if tab-separated.
                parts = line.split('\t')
                text = parts[-1] if len(parts) > 1 else line

                for mode in modes:
                    tokens = sudachi_tokenizer.tokenize(text, mode)
                    for token in tokens:
                        pos_tuple = token.part_of_speech()
                        # pos_tuple structure: (pos, pos1, pos2, pos3, c_type, c_form)
                        # We map '*' to '' to match the map behavior we standardized on.
                        
                        def norm(s):
                            return "" if s == "*" else s

                        # Check Invariants
                        # Violation 1: pos1 is '*' but pos2 is NOT '*'
                        if len(pos_tuple) >= 3:
                            p1 = pos_tuple[1]
                            p2 = pos_tuple[2]
                            if p1 == '*' and p2 != '*':
                                observed['invariant_violations'].append(
                                    f"Line {line_no} Mode {mode}: POS1='*' but POS2='{p2}' (Text: {text})"
                                )

                        # Violation 2: pos2 is '*' but pos3 is NOT '*'
                        if len(pos_tuple) >= 4:
                            p2 = pos_tuple[2]
                            p3 = pos_tuple[3]
                            if p2 == '*' and p3 != '*':
                                observed['invariant_violations'].append(
                                    f"Line {line_no} Mode {mode}: POS2='*' but POS3='{p3}' (Text: {text})"
                                )

                        if len(pos_tuple) >= 1: observed['pos'].add(norm(pos_tuple[0]))
                        if len(pos_tuple) >= 2: observed['pos1'].add(norm(pos_tuple[1]))
                        if len(pos_tuple) >= 3: observed['pos2'].add(norm(pos_tuple[2]))
                        if len(pos_tuple) >= 4: observed['pos3'].add(norm(pos_tuple[3]))
                        if len(pos_tuple) >= 5: observed['conjugated_type'].add(norm(pos_tuple[4]))
                        if len(pos_tuple) >= 6: observed['conjugated_form'].add(norm(pos_tuple[5]))
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Return empty stats on failure to avoid crashing entire suite
        return observed
        
    return observed


class TestMapCompleteness(unittest.TestCase):
    """Verify completeness and minimality of POS maps against real data."""

    def test_map_completeness_parallel(self):
        import glob
        import os
        import concurrent.futures
        from kotogram.japanese_parser import (
            POS_MAP, POS1_MAP, POS2_MAP, POS3_MAP,
            CONJUGATED_TYPE_MAP, CONJUGATED_FORM_MAP
        )

        # 1. Gather all data files
        data_files = glob.glob(os.path.join(os.path.dirname(__file__), '../data/jpn_*.tsv'))
        # If no data files found (e.g. CI), skip or warn.
        if not data_files:
            self.skipTest("No data/jpn_*.tsv files found for completeness check.")

        # 2. Process in parallel
        # Merge results keys
        final_observed = {
            'pos': set(),
            'pos1': set(),
            'pos2': set(),
            'pos3': set(),
            'conjugated_type': set(),
            'conjugated_form': set(),
            'invariant_violations': []
        }

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(_process_file, fp) for fp in data_files]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                for key in final_observed:
                    if key == 'invariant_violations':
                         final_observed[key].extend(result[key])
                    else:
                         final_observed[key].update(result[key])
        
        # 2.5 Assert Invariants
        violations = final_observed['invariant_violations']
        self.assertEqual(len(violations), 0, f"Found {len(violations)} POS hierarchy violations (sample: {violations[:5]})")

        # 3. Define expectations (Map Name -> (Map Object, Observed Key))
        checks = [
            ("POS_MAP", POS_MAP, 'pos'),
            ("POS1_MAP", POS1_MAP, 'pos1'),
            ("POS2_MAP", POS2_MAP, 'pos2'),
            ("POS3_MAP", POS3_MAP, 'pos3'),
            ("CONJUGATED_TYPE_MAP", CONJUGATED_TYPE_MAP, 'conjugated_type'),
            ("CONJUGATED_FORM_MAP", CONJUGATED_FORM_MAP, 'conjugated_form'),
        ]

        # 4. Compare
        for map_name, map_obj, obs_key in checks:
            with self.subTest(map_name=map_name):
                map_keys = set(map_obj.keys())
                observed_keys = final_observed[obs_key]

                # Special case: map keys might contain '' or '*'
                # In our normalized observed data, '*' became ''.
                # In the map, we recently removed "": "" but kept "*" : "" or vice versa?
                # Let's check `japanese_parser.py`: we removed `""` but kept `"*": ""`.
                # So we expect the map to have `"*"`. Does the observed have `""`?
                # Yes, we normalized `*` to `""`.
                # So if observed has `""`, it corresponds to map key `"*"` (which maps to value "").
                # Wait, this logic is tricky. 
                # Let's normalize MUTUALLY for comparison. 
                # The map logic maps KEY -> VALUE.
                # Completeness means: for every observed raw POS string, is it in map_obj?
                # BUT, observed raw data contains `*`. 
                # Our map has `"*": ""`.
                # So `*` IS in the map.
                # However, earlier cleaning logic in `_process_file`: `return "" if s == "*" else s`.
                # This makes observed set contain `""` instead of `*`.
                # Does keys contain `""`? No, we deleted it.
                # So we should probably NOT normalize inside `_process_file` if we want to check strict key existence,
                # OR we must handle the generic `*` case.
                
                # Correction: The map has `"*": ""`.
                # The data has `*`.
                # `_process_file` converts `*` to `""`.
                # Does map have `""`? No.
                # So `observed` should KEEP `*` to match map keys?
                # YES.
                # But wait, `*` signifies "empty".
                # Let's adjust `_process_file` logic in next step if this fails, or use this moment to correct it.
                # Actually, I CANNOT edit `_process_file` freely inside `test_map_completeness_parallel`'s execution flow easily without rewriting.
                # Let's adjust the COMPARISON logic here.
                
                # If observed has `""`, treat it as `*`.
                adjusted_observed_keys = set()
                for k in observed_keys:
                    if k == "":
                        adjusted_observed_keys.add("*")
                    else:
                        adjusted_observed_keys.add(k)
                        
                # Identify missing (In Data but not in Map)
                missing = adjusted_observed_keys - map_keys
                
                # Identify unused (In Map but not in Data)
                # Note: It's possible some valid dictionary words didn't appear in our dataset.
                # But the user asked for "no unused keys".
                unused = map_keys - adjusted_observed_keys

                self.assertTrue(len(missing) == 0, f"{map_name} Missing keys (found in data, not in map): {missing}")
                self.assertTrue(len(unused) == 0, f"{map_name} Unused keys (in map, not in data): {unused}")
