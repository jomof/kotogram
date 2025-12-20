# Formality Analysis

The `kotogram.analysis` module provides formality analysis for Japanese sentences in kotogram format.

## Features

- Analyzes Japanese sentence formality levels
- Detects awkward formality mixing (unpragmatic formality)
- Works with Sudachi parser

## Formality Levels

- **VERY_FORMAL**: Honorific/keigo language (敬語)
- **FORMAL**: Polite forms with -ます/-です endings
- **NEUTRAL**: Plain/dictionary forms
- **CASUAL**: Informal with casual particles
- **VERY_CASUAL**: Highly casual/slang
- **UNPRAGMATIC_FORMALITY**: Awkward mixed formality

## Usage

```python
from kotogram import SudachiJapaneseParser, formality, FormalityLevel

# Initialize parser
parser = SudachiJapaneseParser(dict_type='full')

# Analyze formality
text = "食べます"  # "I eat" (polite)
kotogram = parser.japanese_to_kotogram(text)
level = formality(kotogram)

print(level)  # FormalityLevel.FORMAL
```

## Examples

### Formal Sentences
```python
formality(parser.japanese_to_kotogram("食べます"))        # FORMAL (ます form)
formality(parser.japanese_to_kotogram("学生です"))        # FORMAL (です form)
formality(parser.japanese_to_kotogram("私は学生です。"))   # FORMAL (full sentence)
```

### Neutral Sentences
```python
formality(parser.japanese_to_kotogram("食べる"))   # NEUTRAL (plain form)
formality(parser.japanese_to_kotogram("猫を見る")) # NEUTRAL (plain sentence)
formality(parser.japanese_to_kotogram("高い"))     # NEUTRAL (plain adjective)
```

### Casual Sentences
```python
formality(parser.japanese_to_kotogram("食べるよ"))  # NEUTRAL/CASUAL (with particle)
formality(parser.japanese_to_kotogram("学生だ"))    # NEUTRAL/CASUAL (plain copula)
```

### Unpragmatic Formality
The function detects awkward combinations of formal and casual markers:
- Formal verbs with very casual particles (except よ/ね which are acceptable)
- Mixing of keigo with casual forms

## Advanced Usage: Token Feature Extraction

For lower-level analysis, you can extract linguistic features from individual tokens:

```python
from kotogram import SudachiJapaneseParser, extract_token_features
from kotogram.kotogram import split_kotogram

parser = SudachiJapaneseParser(dict_type='full')
text = "食べます"
kotogram = parser.japanese_to_kotogram(text)

# Split into individual tokens
tokens = split_kotogram(kotogram)

# Extract features from each token
for token in tokens:
    features = extract_token_features(token)
    print(f"Surface: {features['surface']}")
    print(f"POS: {features['pos']}")
    print(f"Conjugated Type: {features['conjugated_type']}")
    print(f"Conjugated Form: {features['conjugated_form']}")
```

Output:
```
Surface: 食べ
POS: v
Conjugated Type: e-ichidan-ba
Conjugated Form: conjunctive

Surface: ます
POS: auxv
Conjugated Type: auxv-masu
Conjugated Form: terminal
```

### Extracted Features

The `extract_token_features()` function returns a dictionary with:
- `surface`: The actual text form
- `pos`: Part of speech (v, n, auxv, prt, adj, etc.)
- `pos_detail1`: First detail level (e.g., 'general', 'common_noun')
- `pos_detail2`: Second detail level
- `conjugated_type`: Conjugation type (e.g., 'e-ichidan-ba', 'auxv-masu')
- `conjugated_form`: Conjugation form (e.g., 'terminal', 'conjunctive')
- `base_orth`: Base orthography (dictionary form spelling)
- `lemma`: Lemma/dictionary form
- `reading`: Reading/pronunciation

## Testing

Comprehensive tests are provided:
- `tests-py/test_formality.py`: Formality analysis tests
  - Tests with Sudachi parser
  - Edge cases (questions, negatives, past tense)
  - Unpragmatic formality detection tests
- `tests-py/test_extract_token_features.py`: Token feature extraction tests
  - Semantic parsing of variable-length POS format
  - Edge cases and malformed tokens

Run tests:
```bash
python -m unittest tests-py.test_formality
python -m unittest tests-py.test_extract_token_features
# Or run all tests
python -m unittest discover tests-py
```

## Detection Logic

The analysis examines:
1. **Auxiliary verbs**: ます, です, じゃ, なんだ, etc.
2. **Verb forms**: Honorific/humble forms (いらっしゃる, etc.)
3. **Particles**: Sentence-ending particles (よ, ね, な, etc.)
4. **Copulas**: Plain だ vs. polite です
5. **Consistency**: Mixing of formal and casual markers
