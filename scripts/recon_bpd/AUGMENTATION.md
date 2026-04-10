# recon_bpd Training Pipeline

This document describes the full data transformation and loss composition
pipeline for `scratch/recon_bpd.py`.  It covers each stage from raw dataset
to gradient, the goal of each component, and notes for debugging.

## Data Flow Overview

```
                      ┌─────────────────────────────────────┐
                      │  Dataset Bundle (.pt)               │
                      │  • surface token IDs  [N, T]        │
                      │  • vocab  {str → int}               │
                      │  • content_mask  [V]                 │
                      │  • token_gram_freq  [V]              │
                      │  • grammaticality labels             │
                      └──────────────┬──────────────────────┘
                                     │
              ┌──────────────────────▼──────────────────────┐
              │  Stage 1: Token Percentile Remap            │
              │  (once, before any training)                │
              │  182K → 169K vocab                          │
              └──────────────────────┬──────────────────────┘
                                     │
              ┌──────────────────────▼──────────────────────┐
              │  Stage 2: Pristine Static Mapping           │
              │  (once, after remap)                        │
              │  Build [V] id→id mapping for clean punct    │
              └──────────────────────┬──────────────────────┘
                                     │
              ┌──────────────────────▼──────────────────────┐
              │  Stage 3: Grammaticality Filter             │
              │  Keep only gram=1 sentences                 │
              └──────────────────────┬──────────────────────┘
                                     │
          ┌──────────────────────────▼──────────────────────────┐
          │  Per-sample (__getitem__)                            │
          │                                                     │
          │  3a. apply_pristine(dirty_ids)  → pristine_ids      │
          │      (context-dep: sentence-final '.', '"' parity)  │
          │                                                     │
          │  3b. Content drop (p=0.5)                            │
          │      Randomly remove non-content tokens from BOTH   │
          │      dirty input and pristine target                │
          └──────────────────────────┬──────────────────────────┘
                                     │
          ┌──────────────────────────▼──────────────────────────┐
          │  Per-batch (training loop)                           │
          │                                                     │
          │  4a. Input masking: 15% of dirty IDs → 0 (BERT)     │
          │  4b. Encoder: masked_dirty → pooled [B, H]          │
          │  4c. KC head: pooled → kc_probs [B, K] (Gumbel-σ)   │
          │  4d. Recon decoder: kc_probs → h_recon [B, T, H']   │
          │  4e. Output head: h_recon → logits [B, T, V]        │
          │  4f. CE loss: logits vs pristine_ids (or dirty_ids) │
          │  4g. Semantic distillation: h_recon → 300D vs chiVe │
          │  4h. Regularizers: MDL, rank, VICReg, consistency   │
          └─────────────────────────────────────────────────────┘
```

---

## Stage 1: Token Percentile Remap

**File:** `scripts/recon_bpd/token_remap.py` → `apply_remap_to_bundle()`

**Goal:** Reduce vocabulary size to accelerate the `[B, T, V]` output
projection and CE loss.  Keeping tokens covering 99% of gram token-position
mass eliminates ~55% of vocab while affecting only 1% of training positions.

**How it works:**
1. Sort tokens by frequency (from `token_gram_freq`).
2. Cumulative sum until the target percentile is reached.
3. Map all remaining rare tokens to `<UNK_REDUCED>` (last slot in new vocab).
4. Force-keep: special tokens (PAD/UNK/CLS/MASK), pristine source/target
   tokens, and top-quartile chiVe tokens.
5. Build a `[V_old]` int64 `old_to_new` mapping tensor.
6. Remap `bundle["features"]["surface"]`, rebuild `bundle["vocab"]["surface"]`,
   slice `content_mask` and chiVe to the new vocab.

UNK chiVe embedding = frequency-weighted mean of all removed tokens' vectors.

**Config:** `token_percentile: float = 99.0` (100.0 = disabled)

---

## Stage 2: Pristine Target Mapping

**File:** `scripts/recon_bpd/token_remap.py` → `build_pristine_id_mapping()`,
`apply_pristine()`

**Goal:** Train the model as a denoiser. The encoder sees dirty tokens
(ASCII `!`, `?`, etc.) and the CE target is the canonical Japanese
form (`！`, `？`, etc.). At inference, reconstruction output is
automatically pristine.

### Static rules (token → token, no context needed)

| Dirty | Pristine | Notes |
|-------|----------|-------|
| `!`   | `！`     | Fullwidth exclamation |
| `?`   | `？`     | Fullwidth question |
| `,`   | `、`     | Ideographic comma |
| `:`   | `：`     | Fullwidth colon |
| `~`   | `～`     | Fullwidth tilde |
| `...` | `…`      | Horizontal ellipsis (single token) |
| `．`  | `。`     | Fullwidth stop → ideographic stop |
| `｡`   | `。`     | Halfwidth → fullwidth |
| `｢`   | `「`     | Halfwidth bracket |
| `｣`   | `」`     | Halfwidth bracket |
| `､`   | `、`     | Halfwidth comma |

### Context-dependent rules (applied per-sequence)

| Rule | Condition |
|------|-----------|
| `.` → `。` | Only when `.` is the **last token** in the sequence |
| `"` → `「`/`」` | Alternating by occurrence: odd = `「`, even = `」` |

### Non-content tokens

Non-content tokens (e.g. `-`, `%`, `*`, `/`) are left as **identity** in
the pristine target. They are removed from both input and target by the
content drop mechanism (Stage 3b).

**Critical invariant:** Pristine **never introduces PAD** (token ID 0).
See [Debugging → Historical bugs](#historical-bugs-resolved) for why.

**Config:** `use_pristine_targets: bool = True`

---

## Stage 3: Per-Sample Augmentation (Dataloader)

**File:** `scripts/dataset.py` → `BundledStyleDataset.__getitem__()`

### 3a. Apply pristine

Called **before** content drop so that context-dependent rules (quote
parity, sentence-final period) see the full unmodified sequence.

### 3b. Content drop

Randomly removes non-content tokens with probability `content_drop_ratio`
(default 0.5). Applied to **both** dirty input and pristine target using
the **same boolean mask**, maintaining positional alignment.

Content is defined by `content_mask[token_id]` — a binary vector built
from `is_content_token()` in `kotogram/masking.py`. Kanji, kana, digits,
and standard Japanese punctuation are content; ASCII noise, symbols, and
Western punctuation are non-content.

**Effect:** The model never sees most non-content tokens at training time,
so it doesn't waste capacity learning to reconstruct them. The tokens that
survive the 50% drop are treated as identity (dirty = pristine for
non-content, so the CE loss just asks the model to copy them).

---

## Stage 4: Per-Batch Processing (Training Loop)

**File:** `scratch/recon_bpd.py` → `train()`

### 4a. Input masking

15% of dirty surface tokens are randomly zeroed (set to PAD ID 0),
BERT-style. This forces the model to reconstruct masked positions from
context and KC probs, not just copy the input.

**Config:** `input_mask_ratio: float = 0.15`

### 4b–4e. Model forward pass

```
dirty_masked → Encoder → pooled [B, H]
                            ↓
                    KC Head → kc_logits [B, K]
                            ↓  (+ Gumbel noise, temperature scaling, sigmoid)
                        kc_probs [B, K]  (soft binary KC selection)
                            ↓
                    Recon Decoder → h_recon [B, T, H']
                            ↓
                    Output Head → logits [B, T, V]
```

**Temperature annealing:** KC logits are divided by a temperature that
linearly decays from `temp * multiplier` → `temp` over
`temperature_anneal_epochs` effective epochs. High temperature → soft
KC probs (near 0.5) → exploration. Low temperature → sharp probs
(near 0/1) → exploitation.

### 4f. Bits-per-dimension (BPD) — Primary loss

```python
total_bits = cross_entropy(logits, recon_targets) / log(2)
bpd = total_bits / num_attended_tokens
```

`recon_targets` = pristine IDs (if enabled) or dirty IDs. Padding
positions use `ignore_index=-100`. BPD is the primary fitness metric
(lower = better).

### 4g. Semantic distillation loss

**Goal:** Ensure the model's reconstruction is **semantically close** to the
target, not just token-exact. Even if the model predicts a synonym or
inflection variant, the semantic loss stays low.

```python
pred_emb = normalize(semantic_head(h_recon))     # [B,T,300]
tgt_emb  = chive_normed[recon_targets]            # [B,T,300]
cos_sim  = (pred_emb * tgt_emb).sum(dim=-1)       # [B,T]
sem_loss = mean(1 - cos_sim)                       # over valid positions
```

The `semantic_head` projects from `H'` → 300D to match chiVe embedding
space. Positions where `recon_targets == 0` (PAD) are excluded from the
semantic loss since PAD has a zero-vector chiVe embedding.

**Semantic gating** (throughput optimization): When `semantic_gating_threshold`
< 1.0, tokens with `cos_sim >= threshold` are "easy" and their CE is
skipped. Only "hard" tokens (low cosine similarity) get the full CE
computation. A stochastic rescue term prevents the boundary from being
too sharp.

**Weight:** `loss += semantic_distillation_loss * 5.0`

### 4h. Regularizers

#### MDL (Minimum Description Length)

```python
mdl_cost = mean(kc_load / sentence_length)
loss += mdl_weight * mdl_warmup * mdl_cost
```

Charges each active KC a per-token information cost. Short sentences pay
more per KC, naturally suppressing over-allocation. Quadratic warmup ramps
from 0 → full over `temperature_anneal_epochs`.

**Config:** `mdl_weight: float = 0.1`

#### Pairwise ranking margin

```python
# For each pair (a,b) where len_a < len_b:
# hinge_loss = max(0, margin * log(len_b/len_a) - (load_b - load_a))
```

Enforces monotonic relationship: longer sentences should activate more KCs.
Uses log-ratio margin so the required separation scales with length ratio.

**Config:** `rank_margin_weight: float = 3.0`, `rank_margin: float = 1.0`

#### VICReg (Variance-Invariance-Covariance)

Applied to encoder output `pooled` (before KC head). Prevents dimensional
collapse in the encoder representation.

- **Variance:** Hinge loss on per-dimension std (must exceed gamma)
- **Covariance:** Penalizes off-diagonal correlations

**Config:** `vicreg_var_weight: float = 11.0`, `vicreg_cov_weight: float = 5.0`

#### Consistency (dual-mask)

Two different random masks are applied to the same sentence. The KC logits
from both views should agree (stop-gradient symmetrized cosine).

**Config:** `consistency_weight: float = 0.0001`

#### KC covariance penalty

Penalizes off-diagonal covariance of KC activation matrix. Encourages
different KCs to activate on different sentences.

**Config:** `cov_penalty_weight: float = 5.0`

#### Length prediction (diagnostic)

Predicts sentence length from KC probs. Diagnostic head to verify
the KC vector carries structural information.

**Config:** `length_pred_weight: float = 0.01`

---

## Warmup Schedules

All warmup schedules use `eff_epoch = epoch * sample_ratio` to normalize
by dataset subsampling.

| Component | Schedule | Period |
|-----------|----------|--------|
| Temperature | Linear: `start*mult` → `start` | `temperature_anneal_epochs` |
| MDL weight | Quadratic: `(eff/anneal)²` | `temperature_anneal_epochs` |
| Semantic threshold | Linear: `1.0` → `base_threshold` | Dynamic (0.005/epoch) |

---

## Key Files

| File | Role |
|------|------|
| `scripts/recon_bpd/token_remap.py` | Token percentile remap + pristine mapping |
| `scripts/dataset.py` | Dataloader: pristine application + content drop |
| `scratch/recon_bpd.py` | Training loop: all losses and regularizers |
| `kotogram/masking.py` | `is_content_char()`, `is_content_token()` |
| `scripts/label.py` | Generates `content_mask.bin` from vocabulary |
| `train/chive.py` | chiVe embedding loading and alignment |
| `scratch/recon_bpd_optuna.py` | Hyperparameter search wrapper |

---

## Loss Composition Summary

```
loss = bpd                                           # primary
     + semantic_distillation_loss * 5.0               # semantic
     + consistency_weight * consistency_loss           # dual-mask
     + vicreg_loss                                     # encoder health
     + mdl_weight * mdl_warmup * mdl_cost              # KC budget
     + rank_margin_weight * rank_loss                  # length ordering
     + cov_penalty_weight * cov_term                   # KC diversity
     + length_pred_weight * length_pred_loss           # diagnostic
```

---

## MLflow Metrics Reference

| Metric | Meaning | Healthy range |
|--------|---------|---------------|
| `bpd/bpd` | Bits per attended token | Decreasing (3–10) |
| `bpd/loss` | Total composite loss | Decreasing |
| `bpd/cos` | Cosine similarity (predicted vs target chiVe) | > 0.6 |
| `bpd/t1` | Top-1 token accuracy | > 0.3 |
| `bpd/s0` | Fraction of dead KCs (prob < 0.1) | 0.2–0.8 |
| `bpd/s1` | Fraction of sharp KCs (prob > 0.9) | 0.02–0.1 |
| `bpd/logit_std` | Std of KC logits | 2–15 (>20 = collapse) |
| `bpd/mean_abs_logit` | Mean |logit| of KC head | < 15 |
| `bpd/pooled_std` | Std of encoder pooled output | ~2.0 (stable) |
| `bpd/temperature` | Current Gumbel-softmax temperature | Decreasing |
| `bpd/mdl_warmup` | MDL weight multiplier (0→1) | Increasing |
| `bpd/semantic_threshold` | Current semantic gating threshold | Decreasing |
| `bpd/pristine_pad_ratio` | Fraction of targets == PAD | ~0 |

---

## Debugging

### Diagnostic script

```bash
PYTHONPATH=. .venv/bin/python3 scratch/debug_pristine.py
```

Checks the pristine augmentation pipeline end-to-end: static mapping
coverage, missing targets, per-sentence PAD/UNK rates, CE target alignment.

### Collapse indicators

If the model is collapsing, check these metrics:

1. **`s0` → 1.0:** All KCs dead. The KC head has collapsed to a constant.
2. **`logit_std` exploding (>20):** KC logits blowing up.
3. **`mean_abs_logit` exploding:** Same cause, different view.
4. **`bpd` → NaN:** Numerical overflow in CE or KC head.
5. **`cos` dropping:** Semantic quality degrading, loss is not learning.
6. **`pristine_pad_ratio` > 0:** PAD in targets = corrupted training signal.

### Common failure modes

| Symptom | Likely cause | Check |
|---------|-------------|-------|
| KC collapse at epoch 5-7 | PAD in pristine targets | Run `debug_pristine.py` |
| BPD stuck high | Semantic gating too aggressive | Check `semantic_threshold` |
| s0 = 1.0 immediately | Temperature too low / mdl too high | Check warmup schedules |
| NaN loss | Gradient explosion in KC logits | Check `logit_std`, `grad_cap` |

### Historical bugs (resolved)

**PAD in pristine targets** (commit 6f458fe, fixed in 1c1b0ef):
Three sources injected PAD into pristine CE targets, creating trivially
predictable positions and zero-vector chiVe targets that corrupted the
semantic loss gradient. ~23% of all pristine change positions were PAD.
Fixed by: (1) removing non-content→PAD static mapping, (2) removing
dot-run collapsing logic, (3) removing content_mask post-filter from
dataloader.
