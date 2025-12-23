# Kotogram

[![Python Canary](https://github.com/jomof/kotogram/actions/workflows/python_canary.yml/badge.svg?branch=main)](https://github.com/jomof/kotogram/actions/workflows/python_canary.yml)
[![TypeScript Canary](https://github.com/jomof/kotogram/actions/workflows/typescript_canary.yml/badge.svg?branch=main)](https://github.com/jomof/kotogram/actions/workflows/typescript_canary.yml)
[![PyPI Version](https://img.shields.io/pypi/v/kotogram.svg)](https://pypi.org/project/kotogram/)
[![npm Version](https://img.shields.io/npm/v/kotogram.svg)](https://www.npmjs.com/package/kotogram)
[![Python Support](https://img.shields.io/pypi/pyversions/kotogram.svg)](https://pypi.org/project/kotogram/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

Kotogram is a lightweight Japanese NLP library that analyzes grammatical style, formality, gender markers, and register. It goes beyond traditional morphological analyzers like MeCab and Sudachi by providing **high-level linguistic analysis** powered by a compact 7MB neural model.

**Key Features:**
- **Grammar Analysis**: Detect formality levels, gender markers, dialectal registers (Kansai-ben, Hakata-ben, etc.), and grammaticality
- **Compact Representation**: Kotogram format encodes rich linguistic features (POS, conjugation, pronunciation) in a space-efficient format
- **Small Neural Model**: 7MB PyTorch model trained on curated Japanese corpora for style classification
- **Dual-Language Support**: Python for model inference and analysis; TypeScript for kotogram manipulation and rendering
- **Production-Ready**: Comprehensive CI/CD with testing across Python 3.9-3.12 and Node.js 18-22

**What Makes Kotogram Different:**

While tools like Sudachi and MeCab excel at morphological analysis (tokenization and POS tagging), Kotogram operates at a **higher linguistic level**, analyzing:
- **Formality** (casual ↔ formal)
- **Gender** (masculine ↔ feminine speech patterns)
- **Register** (dialect, honorifics, internet slang, etc.)
- **Grammaticality** (well-formed vs. malformed)

This makes Kotogram ideal for applications requiring nuanced understanding of Japanese text style and appropriateness.

## CLI Examples

The `kotogram` command-line tool provides instant access to grammar analysis:

### Analyzing Formality

```bash
$ bin/kotogram grammar "お疲れ様でございます"
{
  "kotogram": "⌈ˢお疲れ様ᵖnoun:common-noun:adjectival-noun-possibleʳオツカレサマ⌉⌈ˢでᵖaux-verb:aux-da:continuativeᵇだᵈだʳデ⌉⌈ˢございᵖverb:bound:godan-ra:continuative-i-euphonicᵇござるᵈござるʳゴザイ⌉⌈ˢますᵖaux-verb:aux-masu:terminalʳマス⌉",
  "formality": "formal",
  "formality_score": 0.5010958909988403,
  "formality_is_pragmatic": true,
  "gender": "neutral",
  "gender_score": 0.0007681779679842293,
  "gender_is_pragmatic": true,
  "registers": [
    "neutral"
  ],
  "register_scores": {
    "neutral": 0.9213598966598511
  },
  "is_grammatic": true,
  "grammaticality_score": 0.9999127388000488
}
```

The `kotogram` field shows the compact linguistic representation. Let's deconstruct one token:

```
⌈ˢございᵖverb:bound:godan-ra:continuative-i-euphonicᵇござるᵈござるʳゴザイ⌉
  │  │     │                                        │      │      │
  │  │     │                                        │      │      └─ pronunciation (ʳ)
  │  │     │                                        │      └─ lemma (ᵈ)
  │  │     │                                        └─ base form (ᵇ)
  │  │     └─ part-of-speech + conjugation details (ᵖ)
  │  └─ surface form (ˢ)
  └─ token boundary markers (⌈⌉)
```

### Detecting Gender Markers

```bash
$ bin/kotogram grammar "あら、素敵ですわ"
{
  "kotogram": "⌈ˢあらᵖinterj:generalʳアラ⌉⌈ˢ、ᵖaux-symbol:comma⌉⌈ˢ素敵ᵖadjectival-noun:generalʳステキ⌉⌈ˢですᵖaux-verb:aux-desu:terminalʳデス⌉⌈ˢわᵖparticle:sentence-final-particleʳワ⌉",
  "formality": "formal",
  "formality_score": 0.5490256547927856,
  "formality_is_pragmatic": true,
  "gender": "feminine",
  "gender_score": 0.9999998211860657,
  "gender_is_pragmatic": true,
  "registers": [
    "ojousama"
  ],
  "register_scores": {
    "ojousama": 0.9900707602500916
  },
  "is_grammatic": true,
  "grammaticality_score": 0.999970555305481
}
```

### Checking Grammaticality

```bash
$ bin/kotogram grammar "猫を食べる"
{
  "kotogram": "⌈ˢ猫ᵖnoun:common-nounʳネコ⌉⌈ˢをᵖparticle:case-particleʳヲ⌉⌈ˢ食べるᵖverb:general:lower-ichidan-ba:terminalʳタベル⌉",
  "formality": "neutral",
  "formality_score": -0.005069365259259939,
  "formality_is_pragmatic": true,
  "gender": "neutral",
  "gender_score": -0.009396958164870739,
  "gender_is_pragmatic": true,
  "registers": [
    "neutral"
  ],
  "register_scores": {
    "neutral": 0.9672769904136658
  },
  "is_grammatic": true,
  "grammaticality_score": 0.9998162388801575
}
```

**Non-grammatic Example** (common intermediate learner mistake):

```bash
$ bin/kotogram grammar "食べるました"
{
  "kotogram": "⌈ˢ食べるᵖverb:general:lower-ichidan-ba:terminalʳタベル⌉⌈ˢましᵖaux-verb:aux-masu:continuativeᵇますᵈますʳマシ⌉⌈ˢたᵖaux-verb:aux-ta:terminalʳタ⌉",
  "formality": "formal",
  "formality_score": 0.4452589154243469,
  "formality_is_pragmatic": true,
  "gender": "neutral",
  "gender_score": 0.008438648656010628,
  "gender_is_pragmatic": true,
  "registers": [
    "neutral"
  ],
  "register_scores": {
    "neutral": 0.9035218954086304
  },
  "is_grammatic": false,
  "grammaticality_score": 0.009288009256124496
}
```

**Why this is ungrammatical:** This sentence attempts "I ate" but incorrectly mixes the **dictionary form** (食べる *taberu*) with the **polite past suffix** (ました *mashita*). In Japanese, you must conjugate the verb stem before adding polite forms. The correct form is either:
- **Polite**: 食べました (*tabemashita*) — verb stem (食べ) + polite past (ました)
- **Casual**: 食べた (*tabeta*) — verb stem (食べ) + plain past (た)

This error is common among learners who memorize the dictionary form (食べる) but forget to drop the る before adding conjugations.

**Unpragmatic Formality Example** (mixing casual and formal inappropriately):

```bash
$ bin/kotogram grammar "食べたんだぜです"
{
  "kotogram": "⌈ˢ食べᵖverb:general:lower-ichidan-ba:continuativeᵇ食べるᵈ食べるʳタベ⌉⌈ˢたᵖaux-verb:aux-ta:attributiveʳタ⌉⌈ˢんᵖparticle:nominal-particleʳン⌉⌈ˢだᵖaux-verb:aux-da:terminalʳダ⌉⌈ˢぜᵖparticle:sentence-final-particleʳゼ⌉⌈ˢですᵖaux-verb:aux-desu:terminalʳデス⌉",
  "formality": "unpragmatic_formality",
  "formality_score": 0.3184594213962555,
  "formality_is_pragmatic": false,
  "gender": "masculine",
  "gender_score": -0.9999995827674866,
  "gender_is_pragmatic": true,
  "registers": [
    "danseigo"
  ],
  "register_scores": {
    "danseigo": 0.9998853206634521
  },
  "is_grammatic": false,
  "grammaticality_score": 2.01202964879299e-12
}
```

**Why formality is unpragmatic:** This sentence mixes **casual masculine speech** (ぜ *ze* - a rough, masculine sentence-ending particle) with a **formal copula** (です *desu*). In Japanese, formality must be consistent throughout an utterance. You cannot combine:
- Casual markers like ぜ, よ (casual), だ (plain copula)
- With formal endings like です, ます

The correct forms maintain consistency:
- **Casual masculine**: 食べたんだぜ (*tabetan da ze*) - "I ate, you know!" (rough/masculine)
- **Formal neutral**: 食べたんです (*tabetan desu*) - "I ate." (polite explanation)

This kind of formality clash sounds jarring to native speakers, like mixing "ain't" with "indeed" in English.

### Parsing to Kotogram Format

```bash
$ kotogram parse "猫を食べる"
⌈ˢ猫ᵖn:common_noun⌉⌈ˢをᵖprt:case_particle⌉⌈ˢ食べるᵖv:general:e-ichidan-ba:terminal⌉
```

## Install and Use Python Library

### Installation

```bash
pip install kotogram
```

### API Usage

```python
from kotogram import SudachiJapaneseParser, grammar

# Initialize parser
parser = SudachiJapaneseParser()

# Parse Japanese text to kotogram
text = "お疲れ様でございます"
kotogram_str = parser.japanese_to_kotogram(text)

# Analyze grammar
analysis = grammar(kotogram_str)

print(f"Formality: {analysis.formality}")
print(f"Gender: {analysis.gender}")
print(f"Registers: {analysis.registers}")
print(f"Is Grammatic: {analysis.is_grammatic}")

# Access detailed scores
print(f"Formality Score: {analysis.formality_score:.2f}")
print(f"Grammaticality Score: {analysis.grammaticality_score:.2f}")
```

### Working with Kotograms

```python
from kotogram import kotogram_to_japanese, split_kotogram

# Convert back to Japanese
japanese = kotogram_to_japanese(kotogram_str)

# Add furigana (readings)
with_furigana = kotogram_to_japanese(kotogram_str, furigana=True)
# Result: "お疲れ様[おつかれさま]で御座います[ございます]"

# Add spaces between tokens
spaced = kotogram_to_japanese(kotogram_str, spaces=True)

# Split into individual tokens
tokens = split_kotogram(kotogram_str)
for token in tokens:
    print(token)
```

## Install and Use TypeScript Library

### Installation

```bash
npm install kotogram
```

### API Usage

```typescript
import { kotogramToJapanese, splitKotogram, GrammarAnalysis } from 'kotogram';

// Work with kotogram format (parsing requires Python library)
const kotogram = "⌈ˢ猫ᵖn:common_noun⌉⌈ˢをᵖprt:case_particle⌉⌈ˢ食べるᵖv:general:e-ichidan-ba:terminal⌉";

// Convert to Japanese
const japanese = kotogramToJapanese(kotogram);
console.log(japanese);  // "猫を食べる"

// Add furigana
const withFurigana = kotogramToJapanese(kotogram, { furigana: true });
console.log(withFurigana);  // "猫[ねこ]を食べる[たべる]"

// Split into tokens
const tokens = splitKotogram(kotogram);
tokens.forEach(token => console.log(token));

// Deserialize grammar analysis from JSON
const analysisJson = '{"formality":"Formal","gender":"Neutral",...}';
const analysis = GrammarAnalysis.fromJson(analysisJson);
console.log(analysis.formality);  // FormalityLevel.Formal
```

## Neural Architecture

Kotogram's grammar analysis leverages a **compact transformer-based architecture** optimized for Japanese style classification. The model demonstrates strong performance despite its constrained parameter budget:

**Model Characteristics:**
- **Architecture**: Multi-head attention over linguistic feature embeddings
- **Size**: ~7MB (PyTorch checkpoint)
- **Input**: Kotogram token sequences (POS, conjugation, surface forms)
- **Tasks**: Multi-task learning over formality, gender, register detection, and grammaticality classification

**Training Approach:**
- **Data**: Curated corpus of ~270+ hand-verified examples per register, focusing on edge cases and dialectal variations
- **Features**: Extracted from kotogram tokens including POS tags, conjugation forms, and lexical markers
- **Loss**: Weighted multi-task objective balancing formality regression, gender regression, register multi-label classification, and grammaticality binary classification
- **Optimization**: AdamW with linear warmup and cosine annealing

**Design Philosophy:**

The model prioritizes **sample efficiency** and **generalization** over raw scale. By operating on structured linguistic features (kotogram format) rather than raw text, the model learns meaningful abstractions with minimal data. The compact architecture enables:
- **Fast inference**: < 10ms on CPU for typical sentences
- **Deployment-friendly**: Embeddable in web applications, mobile apps, and serverless functions
- **Interpretable**: Feature-based representations facilitate debugging and error analysis

This approach reflects modern NLP best practices: leveraging domain knowledge (linguistic structure) to build efficient, task-specific models rather than relying solely on massive pre-training.

## Citation

If you use Kotogram in your research or project, please cite:

```bibtex
@software{kotogram2024,
  author = {Fisher, Jomo},
  title = {Kotogram: A Lightweight Japanese NLP Library for Grammar Analysis},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/jomof/kotogram},
  note = {Python and TypeScript library for Japanese formality, gender, and register detection}
}
```

---

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
