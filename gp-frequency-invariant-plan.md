# GP Loss: Frequency Invariance Across Dataset Sizes

## Problem

With separate labeled/unlabeled normalization, each **batch's** GP loss magnitude is invariant.
But the **frequency** of labeled batches per GP changes with dataset size:

- **28K dataset**: GP with 77 labels → labels appear in ~8.8% of batches
- **282K dataset**: Same GP → labels appear in ~0.87% of batches (10x rarer)

Per-epoch gradient totals:
- **Labeled**: `(n_labeled/B) × bce_labeled` — invariant ✅
- **Unlabeled**: `(n_total/B) × bce_unlabeled` — scales with dataset size ❌

The labeled/unlabeled ratio shifts ~10x between 28K and 282K, explaining AvgPos 7.4 (local) vs 31.1 (GPU) at EP2.

## Options

### Option A: Label-Conditioned Loss (no precomputation)

Only apply unlabeled loss for a GP when that GP has ≥1 label in the batch:

```python
has_labels = (labeled_count > 0).float()  # (vocab_size,)
loss_per_gp = labeled_loss + w * unlabeled_loss * has_labels
```

Per-epoch, both labeled and unlabeled gradient accumulate in `n_labeled/B` batches → invariant.

- ✅ No precomputation, no tensors, minimal code change
- ✅ Automatically invariant to dataset size
- ⚠️ GPs with 0 labels get zero loss → no prior anchoring at all
  - ~1200 GPs have priors but 0 labels; they'd hallucinate freely
  - Mitigation: add a small unconditional anchor term for 0-label GPs only

### Option B: Label Density Scaling (precomputed tensor)

Scale unlabeled loss by each GP's label density `d_i = n_labeled_i / n_total`:

```python
density = gp_label_density.unsqueeze(0)  # (1, vocab_size)
loss_per_gp = labeled_loss + w * density * unlabeled_loss
```

Per-epoch unlabeled gradient: `(n_total/B) × d_i × bce = (n_labeled_i/B) × bce` → invariant.

- ✅ Fully invariant, handles all GPs including 0-label
- ✅ Clear semantics: "unlabeled weight scales with how much evidence we have"
- ⚠️ Requires precomputing `gp_label_density` tensor (reintroduce `gp_label_counts.bin`)
- ⚠️ 0-label GPs have density=0 → same problem as Option A

### Option C: Hybrid — Option A + Anchor Floor

```python
has_labels = (labeled_count > 0).float()
# Primary scaling: invariant for labeled GPs
scale = has_labels.clamp_min(anchor_floor)
loss_per_gp = labeled_loss + w * scale * unlabeled_loss
```

Where `anchor_floor` is a small constant (e.g., `0.01`).

- ✅ Invariant for labeled GPs (has_labels=1 in proportion to n_labeled)
- ✅ 0-label GPs still get small anchoring via `anchor_floor`
- ✅ No precomputation — fully online
- ⚠️ `anchor_floor` adds one more hyperparameter
- ⚠️ Anchor floor is NOT frequency-invariant (applied every batch)

### Option D: Accumulator + Reset (online tracking)

Track per-GP labeled encounter counts during the epoch. Scale unlabeled
contribution by the running fraction of batches that have seen labels:

```python
# Maintain a counter: batches_with_label[gp] / batches_so_far
# Use this as unlabeled scale
```

- ✅ Online, no precomputation
- ⚠️ Complex implementation, warmup instability
- ⚠️ Adds mutable state to tracking

## Recommendation

**Option A** is the simplest and achieves true invariance for all GPs that
have labels (which are the only ones where PosP/AvgPos matter). The 0-label
GP concern is minor: those GPs have no training signal anyway, so their
predictions are driven purely by initialization + struct backprop, not by
GP-specific loss. The prior anchoring for 0-label GPs was arguably a fiction
— without labels, we can't know if the anchor is correct.

If 0-label GP hallucination becomes a problem in practice, Option C adds a
small fallback at minimal complexity cost.

## Implementation (Option A)

One line change in `_multilabel_pnu_loss`:

```diff
+        has_labels = (labeled_count > 1e-6).float()  # per-GP: 1 if any label in batch
         loss_per_gp = labeled_loss + unlabeled_weight * unlabeled_loss
+        loss_per_gp = labeled_loss + unlabeled_weight * unlabeled_loss * has_labels
```

No config changes, no new files, no precomputation.
