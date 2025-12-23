# Kotogram

[![Python Canary](https://github.com/jomof/kotogram/actions/workflows/python_canary.yml/badge.svg?branch=main)](https://github.com/jomof/kotogram/actions/workflows/python_canary.yml)
[![TypeScript Canary](https://github.com/jomof/kotogram/actions/workflows/typescript_canary.yml/badge.svg?branch=main)](https://github.com/jomof/kotogram/actions/workflows/typescript_canary.yml)
[![PyPI Version](https://img.shields.io/pypi/v/kotogram.svg)](https://pypi.org/project/kotogram/)
[![npm Version](https://img.shields.io/npm/v/kotogram.svg)](https://www.npmjs.com/package/kotogram)
[![Python Support](https://img.shields.io/pypi/pyversions/kotogram.svg)](https://pypi.org/project/kotogram/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## What is this?

Ever wondered if a Japanese sentence sounds too formal, or whether that sentence-ending particle makes it sound masculine? Kotogram is a lightweight NLP library that analyzes Japanese grammatical style, formality, gender markers, and register detection.

While excellent tools like MeCab and Sudachi focus on morphological analysis (breaking text into tokens and identifying parts of speech), Kotogram takes things a step further by analyzing the **social and stylistic dimensions** of Japanese text:

- **Formality**: Is this casual banter or keigo? (And is it mixing them inappropriately?)
- **Gender**: Does this use masculine (俺だぜ), feminine (〜わ), or neutral speech patterns?
- **Register**: Kansai-ben? Internet slang? Honorific language? Military commands?
- **Grammaticality**: Is this sentence well-formed, or a common learner mistake?

The whole thing runs on a compact 7MB neural model and works in both Python (for the ML inference) and TypeScript (for working with the kotogram format).

## Quick Examples

Let's see it in action! The `bin/kotogram grammar` command analyzes any Japanese text:

### Detecting Formality

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

The `kotogram` field in the output shows how the sentence gets internally represented. Here's what one token looks like when you break it down:

```
⌈ˢございᵖverb:bound:godan-ra:continuative-i-euphonicᵇござるᵈござるʳゴザイ⌉
  │  │     │                                        │      │      │
  │  │     │                                        │      │      └─ pronunciation (ʳ)
  │  │     │                                        │      └─ lemma (ᵈ)
  │  │     │                                        └─ base form (ᵇ)
  │  │     └─ part-of-speech + conjugation (ᵖ)
  │  └─ surface form (ˢ)
  └─ token boundaries (⌈⌉)
```

Pretty neat how much linguistic information we can pack into a compact format, right?

### Gender Detection

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

The model picks up on that sentence-final わ (*wa*) and correctly identifies this as ojousama-style speech (refined, upper-class feminine Japanese). The gender score of 0.9999998 means the model is extremely confident about the feminine markers.

### Catching Subtle Awkwardness

Here's a more subtle issue — a sentence that's technically parseable but semantically awkward:

```bash
$ bin/kotogram grammar "大きくない小さい"
{
  "kotogram": "⌈ˢ大きくᵖadj:general:i-adjective:continuativeᵇ大きいᵈ大きいʳオオキク⌉⌈ˢないᵖadj:bound:i-adjective:terminalʳナイ⌉⌈ˢ小さいᵖadj:general:i-adjective:terminalʳチイサイ⌉",
  "formality": "neutral",
  "formality_score": -0.00582164479419589,
  "formality_is_pragmatic": true,
  "gender": "neutral",
  "gender_score": -0.0024029570631682873,
  "gender_is_pragmatic": true,
  "registers": [
    "neutral"
  ],
  "register_scores": {
    "neutral": 0.9790019989013672
  },
  "is_grammatic": false,
  "grammaticality_score": 0.1085873544216156
}
```

**Why this is awkward:** This literally means "not-big small" — grammatically parseable, but semantically redundant. While you *can* stack adjectives in Japanese, saying "not big small" is unnatural because 小さい (*chiisai*, small) already implies "not big." 

Japanese highly values **concision** (簡潔さ). The natural way to express this would be simply:
- **Concise**: 小さい (*chiisai*) — "small"  
- **Or with emphasis**: 大きくない (*ookikunai*) — "not big"

This kind of redundant negation occasionally appears in learner speech when they're trying to be emphatic but end up being unnecessarily verbose. The model's grammaticality score of 0.108 (pretty low, but not zero) reflects that while the syntax parses, the semantic redundancy makes it sound distinctly non-native.

### Detecting Unpragmatic Mixing

Here's an interesting one — a sentence that's grammatically parseable but stylistically bizarre:

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

**Why is this unpragmatic?** It mixes ぜ (*ze*, a rough masculine sentence-ender) with です (*desu*, formal copula). In Japanese, you need to pick a formality register and stick with it throughout the sentence. This would sound as jarring to a native speaker as mixing "ain't" with "indeed" in English.

Correct versions:
- **Casual masculine**: 食べたんだぜ (*tabetan da ze*) — "I ate, y'know!" (rough)
- **Formal neutral**: 食べたんです (*tabetan desu*) — "I ate." (polite)

## Installation & Usage

### Python

```bash
pip install kotogram
```

```python
from kotogram import SudachiJapaneseParser, grammar

# Parse Japanese to kotogram format
parser = SudachiJapaneseParser()
text = "お疲れ様でございます"
kotogram_str = parser.japanese_to_kotogram(text)

# Analyze the grammar
analysis = grammar(kotogram_str)

print(f"Formality: {analysis.formality}")
print(f"Gender: {analysis.gender}")
print(f"Registers: {analysis.registers}")
print(f"Grammatic? {analysis.is_grammatic}")
print(f"Grammaticality confidence: {analysis.grammaticality_score:.4f}")
```

You can also work with kotograms directly:

```python
from kotogram import kotogram_to_japanese, split_kotogram

# Convert back to readable Japanese
japanese = kotogram_to_japanese(kotogram_str)

# Add furigana readings (great for learners!)
with_furigana = kotogram_to_japanese(kotogram_str, furigana=True)
# Output: "お疲れ様[おつかれさま]で御座います[ございます]"

# Split into tokens for detailed analysis
tokens = split_kotogram(kotogram_str)
```

### TypeScript

```bash
npm install kotogram
```

```typescript
import { kotogramToJapanese, splitKotogram } from 'kotogram';

// Work with pre-computed kotograms (Python handles the parsing)
const kotogram = "⌈ˢ猫ᵖnoun:common-nounʳネコ⌉⌈ˢをᵖparticle:case-particleʳヲ⌉...";

// Convert to Japanese
const japanese = kotogramToJapanese(kotogram);
console.log(japanese);  // "猫を食べる"

// Add furigana
const withFurigana = kotogramToJapanese(kotogram, { furigana: true });
console.log(withFurigana);  // "猫[ねこ]を食べる[たべる]"

// Split into tokens
const tokens = splitKotogram(kotogram);
```

## How It Works

The core of Kotogram is a compact transformer-based neural model (only 7MB!) trained on a carefully curated dataset. Rather than feeding it raw text, we use the **kotogram representation** — a structured format that explicitly encodes morphological features like POS tags, conjugation forms, and lemmas.

### Why this approach?

By working with structured linguistic features instead of raw characters, the model can learn meaningful patterns from relatively small amounts of data. Think of it like the difference between learning grammar rules versus memorizing every possible sentence.

**Training data:**
- **~265K grammatic sentences** with formality/gender labels (applied via heuristics)
- **1,115 hand-curated register examples** across 13 categories (sonkeigo, kenjogo, dialects, internet slang, etc.)
- **~593K agrammatic examples** for error detection
- **~270K unpragmatic examples** showing inappropriate formality/gender mixing

**What the model learns:**
- **Formality** as a continuous scale (-1.0 = very casual → +1.0 = very formal)
- **Gender** as a continuous scale (-1.0 = masculine → +1.0 = feminine)
- **Register detection** as a multi-label problem (sentences can have multiple registers!)
- **Grammaticality** as binary classification
- **Pragmatic consistency** — does this sentence maintain appropriate formality/gender?

The architecture uses multi-head attention over linguistic feature embeddings, trained with AdamW and cosine annealing — pretty standard modern NLP techniques, but applied to a focused domain-specific problem.

### Design Philosophy

I built Kotogram around the idea that **domain knowledge + efficient models > massive pre-training**. Instead of throwing a huge transformer at raw text, we leverage what we know about Japanese linguistics to create structured representations that make the learning problem tractable.

Benefits:
- **Fast**: < 10ms inference on CPU for typical sentences
- **Lightweight**: 7MB model fits easily in web apps, mobile apps, serverless functions
- **Interpretable**: Feature-based representations make it easier to debug and understand predictions

## Citation

If you use Kotogram in your research or project, feel free to cite:

```bibtex
@software{kotogram2024,
  author = {Fisher, Jomo},
  title = {Kotogram: A Lightweight Japanese NLP Library for Grammar Analysis},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/jomof/kotogram}
}
```

## Contributing

This started as a weekend project to explore Japanese linguistics and small-scale NLP. If you're interested in Japanese grammar, machine learning, or both — I'd love to hear from you! Feel free to open issues, submit PRs, or just say hi.

## License

MIT — use it for whatever you like!
