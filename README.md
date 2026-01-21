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
- **Formality**: Modeled as a continuous scale (-1.0 to +1.0) via regression heads on top of the KCs.
- **Gender**: Modeled as a continuous scale (-1.0 to +1.0).
- **Register**: Multi-label classification (detecting specific dialects/styles).
- **Grammaticality**: Binary classification for error detection.

### Knowledge Components (KCs)

Most style classifiers learn a direct mapping from “sentence in” → “label out.” That works, but it also tends to become a black box that’s hard to *reason about*, hard to *debug*, and sometimes oddly brittle when you change domains or introduce new constructions.

Kotogram’s KC learner is my attempt to build something closer to how I actually learn Japanese: accumulate reusable little “grammar instincts,” then combine them to explain what a sentence is doing.

At a high level, KCs are **sparse latent units** (think: a small set of “on” switches per sentence) that the model learns *without* being told what each unit means. The trick is that we train those units to be useful by asking them to reconstruct sentence structure.

#### The KC learner, layer by layer

1) **Feature embeddings (kotogram fields)**
- **What**: Each token is represented by explicit linguistic fields (POS, conjugation type/form, lemma, etc.), each with its own embedding table.
- **Why**: Japanese style cues often live in *morphology* (auxiliaries, endings, particles). Feeding structured features makes the learning problem dramatically more tractable than raw characters.

2) **Transformer encoder**
- **What**: Multi-head attention over the feature embeddings to build contextual token representations.
- **Why**: Style and register are often non-local. A single です might be fine, until it collides with a ぜ at the end. The encoder is where “this token in this context” gets formed.

3) **Sentence pooling**
- **What**: A pooled sentence vector from the encoder output.
- **Why**: KCs are sentence-level “atoms,” so we need a stable sentence representation to decide which atoms should fire.

4) **KC head (logits → probabilities)**
- **What**: A linear projection produces **KC logits** over a KC vocabulary. During training we optionally add Gumbel noise, then apply a temperatured sigmoid.
- **Why**: This is the “proposal” step: which KCs might explain this sentence? Temperature + (optional) noise helps avoid early lock-in where a few KCs win forever.

5) **Top-K sparsifier (the key move)**
- **What**: Keep only the top *k* KC activations (e.g., k=8), zero out the rest.
- **Why it helped**: Sparsity forces *specialization*. If only 8 KCs can speak for a sentence, each KC has to become a reliable, reusable pattern instead of a mushy average.

6) **Structural decoders (reconstruction heads)**
- **What**: From the sparse KC activations, lightweight linear decoders predict multi-hot structural targets like lemma/POS/conjugation “bags.”
- **Why it helped**: This is how the model learns meaningfully structured KCs without hand-labeling them. If a KC repeatedly helps reconstruct “polite auxiliaries” or “sentence-final particles,” it has a reason to exist and to stay stable.

7) **Regularization: diversity, load-balance, collapse control**
- **What**: We add gentle pressure so KC usage doesn’t collapse into a tiny set.
- **Why**: Without this, the model can discover one “do-everything” KC and starve the rest. The regularizers encourage a healthier ecosystem of KCs that actually cover the space of Japanese constructions.

#### So why build KCs at all?

Because I want Kotogram’s predictions to be grounded in something closer to *linguistic structure* than pure label-fitting.

KCs give me:
- **Interpretability hooks**: Even when KCs are learned unsupervised, they tend to cluster around recognizable patterns (politeness auxiliaries, dialect markers, final particles, etc.). That’s exactly the level where Japanese “style” lives.
- **Robustness**: The classifier heads sit on top of reusable components instead of memorizing superficial correlations.
- **A teaching tool**: Ultimately, I want to surface explanations like: “This sentence reads masculine because these markers fired,” not just “masculine: 0.99.”

In other words: KCs are the model’s internal set of “grammar instincts,” and the downstream heads (formality/gender/register/grammaticality) learn to read those instincts consistently.


The architecture uses multi-head attention over linguistic feature embeddings, trained with AdamW and cosine annealing — pretty standard modern NLP techniques, but applied to a focused domain-specific problem.

### Design Philosophy

I built Kotogram around the idea that **domain knowledge + efficient models > massive pre-training**. Instead of throwing a huge transformer at raw text, we leverage what we know about Japanese linguistics to create structured representations that make the learning problem tractable.

Benefits:
- **Fast**: < 10ms inference on CPU for typical sentences
- **Lightweight**: 7MB model fits easily in web apps, mobile apps, serverless functions
- **Interpretable**: Feature-based representations make it easier to debug and understand predictions

## Training

### Mixed Precision Training

Both KC and style trainers support fp16 mixed precision training for faster training on GPUs:

- **CUDA**: Full support with automatic gradient scaling (~2x speedup)
- **MPS** (Apple Silicon): Autocast support without gradient scaling (~1.3x speedup)
- **CPU**: Automatically disabled (no benefit)

Mixed precision training is **enabled by default**. The trainer automatically detects your device and configures the appropriate settings.

#### Disabling Mixed Precision

If you encounter numerical stability issues or want to train in full fp32:

```bash
# For KC pretraining
./train_kc --use-amp false

# For style fine-tuning
./train_style --use-amp false
```

#### Requirements

- **PyTorch 2.5.0+** for MPS autocast support
- **CUDA**: Any recent version with autocast support
- **MPS**: macOS 12.3+ with Apple Silicon

#### Technical Details

The implementation uses device-aware autocast contexts that wrap forward passes and loss computation:

- **Forward pass**: All model operations (embedding, attention, decoders) run in fp16 where beneficial
- **Loss computation**: All loss calculations (structural, diversity, load balancing, etc.) use fp16
- **Gradient accumulation**: Handled automatically with proper scaling
- **Optimizer states**: Always maintained in fp32 for numerical stability

The extensive metrics gathering and validation checks in `kc_trainer.py` remain in fp32 to preserve accuracy of diagnostic information.

For more details, see `train/amp_utils.py`.

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
