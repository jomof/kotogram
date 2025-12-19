
import pytest
import textwrap
from sudachipy import tokenizer, dictionary
from kotogram.japanese_parser import (
    POS_MAP, POS1_MAP, POS2_MAP, POS3_MAP, CONJUGATED_TYPE_MAP, CONJUGATED_FORM_MAP
)

"""
Sudachi Split Modes:

Sudachi provides three modes of tokenization (splitting):

1.  **SplitMode.A (Shortest/Unit):**
    -   Deepest splitting. Decomposes compound words into their smallest meaningful units.
    -   Example: "日本語" -> "日本" + "語"
    -   Use case: Search indexing, fine-grained analysis.

2.  **SplitMode.B (Middle):**
    -   Intermediate splitting. Splits longer compounds but keeps common compounds together.
    -   Example: "日本語" -> "日本語" (but "外国人参政権" -> "外国人" + "参政権")
    -   Use case: Named Entity Recognition (NER), general analysis where A is too granular.

3.  **SplitMode.C (Longest/Full):**
    -   Longest splitting. Treats compound words as single tokens whenever possible.
    -   Example: "日本語" -> "日本語"
    -   Use case: Text classification, specific dictionary term matching.

Part-of-Speech Hierarchy Invariant:
-----------------------------------
The POS fields (POS1, POS2, POS3) form a strict hierarchy of specificity.
We have empirically verified (via `tests-py/test_japanese_parser.py`) that:
- If a parent field is empty/unspecified ('*'), all child fields MUST be empty.
- Invariant 1: If POS1 is '*', then POS2 must be '*'.
- Invariant 2: If POS2 is '*', then POS3 must be '*'.

This confirms that these are not orthogonal attributes but represent increasing levels of detail.
"""

def test_splitmode_a():
    # Initialize Sudachi (similar to SudachiJapaneseParser)
    dict_obj = dictionary.Dictionary(dict='full')
    sudachi_tokenizer = dict_obj.create()
    mode = tokenizer.Tokenizer.SplitMode.A

    text = "日本語を話します"
    tokens = sudachi_tokenizer.tokenize(text, mode)

    result_string = _tokens_to_yaml(tokens)
    
    # Placeholder expected output (will be updated after run)
    check(result_string, """
        Token 0:
          begin: 0
          dictionary_form: 日本
          dictionary_id: 0
          end: 2
          is_oov: False
          normalized_form: 日本
          part_of_speech: 
          - raw: ('名詞', '固有名詞', '地名', '国', '*', '*')
          - pos: 名詞 -> noun
          - pos1: 固有名詞 -> proper-noun
          - pos2: 地名 -> place-name
          - pos3: 国 -> country
          - conjugated_type: * -> 
          - conjugated_form: * -> 
          part_of_speech_id: 23
          raw_surface: 日本
          reading_form: ニホン
          surface: 日本
          synonym_group_ids: [6418]
          word_id: 498214
        Token 1:
          begin: 2
          dictionary_form: 語
          dictionary_id: 0
          end: 3
          is_oov: False
          normalized_form: 語
          part_of_speech: 
          - raw: ('名詞', '普通名詞', '一般', '*', '*', '*')
          - pos: 名詞 -> noun
          - pos1: 普通名詞 -> common-noun
          - pos2: 一般 -> general
          - pos3: * -> 
          - conjugated_type: * -> 
          - conjugated_form: * -> 
          part_of_speech_id: 4
          raw_surface: 語
          reading_form: ゴ
          surface: 語
          synonym_group_ids: []
          word_id: 681202
        Token 2:
          begin: 3
          dictionary_form: を
          dictionary_id: 0
          end: 4
          is_oov: False
          normalized_form: を
          part_of_speech: 
          - raw: ('助詞', '格助詞', '*', '*', '*', '*')
          - pos: 助詞 -> particle
          - pos1: 格助詞 -> case-particle
          - pos2: * -> 
          - pos3: * -> 
          - conjugated_type: * -> 
          - conjugated_form: * -> 
          part_of_speech_id: 7
          raw_surface: を
          reading_form: ヲ
          surface: を
          synonym_group_ids: []
          word_id: 170871
        Token 3:
          begin: 4
          dictionary_form: 話す
          dictionary_id: 0
          end: 6
          is_oov: False
          normalized_form: 話す
          part_of_speech: 
          - raw: ('動詞', '一般', '*', '*', '五段-サ行', '連用形-一般')
          - pos: 動詞 -> verb
          - pos1: 一般 -> general
          - pos2: * -> 
          - pos3: * -> 
          - conjugated_type: 五段-サ行 -> godan-sa
          - conjugated_form: 連用形-一般 -> conjunctive
          part_of_speech_id: 82
          raw_surface: 話し
          reading_form: ハナシ
          surface: 話し
          synonym_group_ids: [7]
          word_id: 679836
        Token 4:
          begin: 6
          dictionary_form: ます
          dictionary_id: 0
          end: 8
          is_oov: False
          normalized_form: ます
          part_of_speech: 
          - raw: ('助動詞', '*', '*', '*', '助動詞-マス', '終止形-一般')
          - pos: 助動詞 -> auxiliary-verb
          - pos1: * -> 
          - pos2: * -> 
          - pos3: * -> 
          - conjugated_type: 助動詞-マス -> auxv-masu
          - conjugated_form: 終止形-一般 -> terminal
          part_of_speech_id: 55
          raw_surface: ます
          reading_form: マス
          surface: ます
          synonym_group_ids: []
          word_id: 148491
    """)

def test_splitmode_b():
    # Initialize Sudachi (similar to SudachiJapaneseParser)
    dict_obj = dictionary.Dictionary(dict='full')
    sudachi_tokenizer = dict_obj.create()
    mode = tokenizer.Tokenizer.SplitMode.B

    text = "日本語を話します"
    tokens = sudachi_tokenizer.tokenize(text, mode)

    result_string = _tokens_to_yaml(tokens)
    
    # Placeholder expected output (will be updated after run)
    check(result_string, """
        Token 0:
          begin: 0
          dictionary_form: 日本語
          dictionary_id: 0
          end: 3
          is_oov: False
          normalized_form: 日本語
          part_of_speech: 
          - raw: ('名詞', '普通名詞', '一般', '*', '*', '*')
          - pos: 名詞 -> noun
          - pos1: 普通名詞 -> common-noun
          - pos2: 一般 -> general
          - pos3: * -> 
          - conjugated_type: * -> 
          - conjugated_form: * -> 
          part_of_speech_id: 4
          raw_surface: 日本語
          reading_form: ニホンゴ
          surface: 日本語
          synonym_group_ids: []
          word_id: 1272370
        Token 1:
          begin: 3
          dictionary_form: を
          dictionary_id: 0
          end: 4
          is_oov: False
          normalized_form: を
          part_of_speech: 
          - raw: ('助詞', '格助詞', '*', '*', '*', '*')
          - pos: 助詞 -> particle
          - pos1: 格助詞 -> case-particle
          - pos2: * -> 
          - pos3: * -> 
          - conjugated_type: * -> 
          - conjugated_form: * -> 
          part_of_speech_id: 7
          raw_surface: を
          reading_form: ヲ
          surface: を
          synonym_group_ids: []
          word_id: 170871
        Token 2:
          begin: 4
          dictionary_form: 話す
          dictionary_id: 0
          end: 6
          is_oov: False
          normalized_form: 話す
          part_of_speech: 
          - raw: ('動詞', '一般', '*', '*', '五段-サ行', '連用形-一般')
          - pos: 動詞 -> verb
          - pos1: 一般 -> general
          - pos2: * -> 
          - pos3: * -> 
          - conjugated_type: 五段-サ行 -> godan-sa
          - conjugated_form: 連用形-一般 -> conjunctive
          part_of_speech_id: 82
          raw_surface: 話し
          reading_form: ハナシ
          surface: 話し
          synonym_group_ids: [7]
          word_id: 679836
        Token 3:
          begin: 6
          dictionary_form: ます
          dictionary_id: 0
          end: 8
          is_oov: False
          normalized_form: ます
          part_of_speech: 
          - raw: ('助動詞', '*', '*', '*', '助動詞-マス', '終止形-一般')
          - pos: 助動詞 -> auxiliary-verb
          - pos1: * -> 
          - pos2: * -> 
          - pos3: * -> 
          - conjugated_type: 助動詞-マス -> auxv-masu
          - conjugated_form: 終止形-一般 -> terminal
          part_of_speech_id: 55
          raw_surface: ます
          reading_form: マス
          surface: ます
          synonym_group_ids: []
          word_id: 148491
    """)

def test_splitmode_c():
    # Initialize Sudachi (similar to SudachiJapaneseParser)
    dict_obj = dictionary.Dictionary(dict='full')
    sudachi_tokenizer = dict_obj.create()
    mode = tokenizer.Tokenizer.SplitMode.C

    text = "日本語を話します"
    tokens = sudachi_tokenizer.tokenize(text, mode)

    result_string = _tokens_to_yaml(tokens)
    
    # Placeholder expected output (will be updated after run)
    check(result_string, """
        Token 0:
          begin: 0
          dictionary_form: 日本語
          dictionary_id: 0
          end: 3
          is_oov: False
          normalized_form: 日本語
          part_of_speech: 
          - raw: ('名詞', '普通名詞', '一般', '*', '*', '*')
          - pos: 名詞 -> noun
          - pos1: 普通名詞 -> common-noun
          - pos2: 一般 -> general
          - pos3: * -> 
          - conjugated_type: * -> 
          - conjugated_form: * -> 
          part_of_speech_id: 4
          raw_surface: 日本語
          reading_form: ニホンゴ
          surface: 日本語
          synonym_group_ids: []
          word_id: 1272370
        Token 1:
          begin: 3
          dictionary_form: を
          dictionary_id: 0
          end: 4
          is_oov: False
          normalized_form: を
          part_of_speech: 
          - raw: ('助詞', '格助詞', '*', '*', '*', '*')
          - pos: 助詞 -> particle
          - pos1: 格助詞 -> case-particle
          - pos2: * -> 
          - pos3: * -> 
          - conjugated_type: * -> 
          - conjugated_form: * -> 
          part_of_speech_id: 7
          raw_surface: を
          reading_form: ヲ
          surface: を
          synonym_group_ids: []
          word_id: 170871
        Token 2:
          begin: 4
          dictionary_form: 話す
          dictionary_id: 0
          end: 6
          is_oov: False
          normalized_form: 話す
          part_of_speech: 
          - raw: ('動詞', '一般', '*', '*', '五段-サ行', '連用形-一般')
          - pos: 動詞 -> verb
          - pos1: 一般 -> general
          - pos2: * -> 
          - pos3: * -> 
          - conjugated_type: 五段-サ行 -> godan-sa
          - conjugated_form: 連用形-一般 -> conjunctive
          part_of_speech_id: 82
          raw_surface: 話し
          reading_form: ハナシ
          surface: 話し
          synonym_group_ids: [7]
          word_id: 679836
        Token 3:
          begin: 6
          dictionary_form: ます
          dictionary_id: 0
          end: 8
          is_oov: False
          normalized_form: ます
          part_of_speech: 
          - raw: ('助動詞', '*', '*', '*', '助動詞-マス', '終止形-一般')
          - pos: 助動詞 -> auxiliary-verb
          - pos1: * -> 
          - pos2: * -> 
          - pos3: * -> 
          - conjugated_type: 助動詞-マス -> auxv-masu
          - conjugated_form: 終止形-一般 -> terminal
          part_of_speech_id: 55
          raw_surface: ます
          reading_form: マス
          surface: ます
          synonym_group_ids: []
          word_id: 148491
    """)


def token_to_yaml(token) -> str:
    """Reflect against the token object to dump all public no-arg properties."""
    lines = []
    
    # Get all public attributes
    attributes = [attr for attr in dir(token) if not attr.startswith("_")]
    
    collected_data = {}
    
    for attr_name in attributes:
        # Skip unstable or deprecated attributes
        if attr_name == 'get_word_info':
            continue
            
        attr = getattr(token, attr_name)
        
        # We only care about methods that return values (getters) or properties
        # SudachiPy tokens mostly use methods for accessors
        if callable(attr):
            try:
                # Try calling without arguments
                val = attr()
                collected_data[attr_name] = val
            except TypeError:
                # Requires arguments, skip
                pass 
            except Exception as e:
                collected_data[attr_name] = f"<Error: {e}>"
        else:
            # It's a property or field
            collected_data[attr_name] = attr

    # Special handling for decomposing part_of_speech
    if 'part_of_speech' in collected_data:
        pos_tuple = collected_data['part_of_speech']
        
        # Build the nested list representation
        pos_lines = []
        pos_lines.append(f"- raw: {pos_tuple}")
        
        # Decompose tuple elements:
        # (POS, POS1, POS2, POS3, conjugated_type, conjugated_form)
        if len(pos_tuple) >= 1:
            pos_lines.append(f"- pos: {pos_tuple[0]} -> {POS_MAP.get(pos_tuple[0], 'UNKNOWN')}")
        if len(pos_tuple) >= 2:
            pos_lines.append(f"- pos1: {pos_tuple[1]} -> {POS1_MAP.get(pos_tuple[1], 'UNKNOWN')}")
        if len(pos_tuple) >= 3:
            pos_lines.append(f"- pos2: {pos_tuple[2]} -> {POS2_MAP.get(pos_tuple[2], 'UNKNOWN')}")
        if len(pos_tuple) >= 4:
            pos_lines.append(f"- pos3: {pos_tuple[3]} -> {POS3_MAP.get(pos_tuple[3], 'UNKNOWN')}")
        if len(pos_tuple) >= 5:
            pos_lines.append(f"- conjugated_type: {pos_tuple[4]} -> {CONJUGATED_TYPE_MAP.get(pos_tuple[4], 'UNKNOWN')}")
        if len(pos_tuple) >= 6:
            pos_lines.append(f"- conjugated_form: {pos_tuple[5]} -> {CONJUGATED_FORM_MAP.get(pos_tuple[5], 'UNKNOWN')}")
            
        # Format as a multi-line string indented safely for the YAML-like output
        # The key is printed as "  {key}: {value}"
        # We want:
        #   part_of_speech: 
        #   - raw: ...
        # So we prepend a newline and indent the list items by 2 spaces (to match 'part_of_speech')
        collected_data['part_of_speech'] = "\n  " + "\n  ".join(pos_lines)


    # Sort keys for deterministic output
    for key in sorted(collected_data.keys()):
        value = collected_data[key]
        lines.append(f"  {key}: {value}")
        
    return "\n".join(lines)

def _tokens_to_yaml(tokens) -> str:
    output_lines = []
    for i, token in enumerate(tokens):
        output_lines.append(f"Token {i}:")
        output_lines.append(token_to_yaml(token))
    return "\n".join(output_lines)

def check(actual: str, expected: str):
    """Helper to verify output matches expected string (dedented)."""
    expected = textwrap.dedent(expected).strip()
    print("\n--- Actual Output ---\n" + actual + "\n---------------------")
    assert actual == expected

