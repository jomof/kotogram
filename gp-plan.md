# GP-Specific Bottleneck Plan

## Problem

The `grammar_point` family tends to over-predict labels per sentence. As training
progresses and `Logit(+)` approaches/exceeds 0, the model predicts too many grammar
points for a given sentence. This may be partially caused by the shared hidden
representation in the label pathway — features learned for dense surface-pattern
families (n-grams, conjugated types) create "positive leakage" that lifts GP logits
for labels that should be inactive.

## Current Architecture

File: [train/models.py](file:///Users/jomofisher/projects/kotogram/train/models.py)
Class: `KCDecoder` (lines 12-109)

The decoder has two pathways:

```
MSE pathway (gender, formality):
  kc_probs (1024) → mse_hidden1 (1024→256) → ReLU → mse_hidden2 (256→256) → ReLU
                     → per-family Linear(256→1) → Tanh

Label pathway (grammar_point, n-grams, conjugated_type, etc.):
  kc_probs (1024) → label_hidden1 (1024→256) → ReLU → label_hidden2 (256→256) → ReLU
                     → per-family Linear(256→vocab_size)
```

All label families share `label_hidden1` and `label_hidden2`. The grammar_point
output head is `nn.Linear(256, 1374)` stored in `self.decoders["grammar_point"]`.

Key code in `__init__` (line 66-73):
```python
for fid, vocab_size in target_specs.items():
    name = fid.name.lower()
    if name in self._mse_families:
        self.mse_decoders[name] = nn.Linear(hidden_dim, vocab_size)
    else:
        self.decoders[name] = nn.Linear(hidden_dim, vocab_size)
```

Key code in `forward` (line 101-107):
```python
if self.decoders:
    h_label = self.activation(self.label_hidden1(kc_probs))
    h_label = self.activation(self.label_hidden2(h_label))

    for name, decoder in self.decoders.items():
        result[name] = decoder(h_label)
```

## Proposed Change: GP-Specific Bottleneck

Add a small nonlinear bottleneck between the shared label hidden layers and the
grammar_point output head. Other label families continue to use `h_label` directly.

### New architecture for grammar_point only:

```
kc_probs → label_hidden1 → ReLU → label_hidden2 → ReLU → h_label (256)
                                                            │
                                                            ├─→ n-gram heads (unchanged)
                                                            │
                                                            └─→ gp_bottleneck (256→64) → ReLU
                                                                   → grammar_point head (64→1374)
```

### Implementation Steps

#### 1. Add bottleneck parameter to `KCDecoder.__init__`

In [train/models.py](file:///Users/jomofisher/projects/kotogram/train/models.py), line 35-39:

- Add `gp_bottleneck_dim: int = 64` parameter to `__init__`
- Store `self.gp_bottleneck_dim = gp_bottleneck_dim`

#### 2. Create bottleneck layers when grammar_point is present

After the `for fid, vocab_size in target_specs.items()` loop (line 66-73):

```python
# GP-specific bottleneck
self.gp_bottleneck: Optional[nn.Linear] = None
if KcFamilyId.GRAMMAR_POINT in target_specs and gp_bottleneck_dim > 0:
    self.gp_bottleneck = nn.Linear(hidden_dim, gp_bottleneck_dim)
    # Replace the grammar_point decoder with narrower input
    gp_vocab = target_specs[KcFamilyId.GRAMMAR_POINT]
    self.decoders["grammar_point"] = nn.Linear(gp_bottleneck_dim, gp_vocab)
```

Note: `KcFamilyId` is imported from `train.kc` (already imported at line 10).

#### 3. Route grammar_point through bottleneck in `forward`

In the `forward` method, replace the label pathway loop (lines 101-107):

```python
if self.decoders:
    h_label = self.activation(self.label_hidden1(kc_probs))
    h_label = self.activation(self.label_hidden2(h_label))

    for name, decoder in self.decoders.items():
        if name == "grammar_point" and self.gp_bottleneck is not None:
            h_gp = self.activation(self.gp_bottleneck(h_label))
            result[name] = decoder(h_gp)
        else:
            result[name] = decoder(h_label)
```

#### 4. Parameter count impact

- Old GP path: `Linear(256, 1374)` = 256×1374 + 1374 = **352,638** params
- New GP path: `Linear(256, 64)` + `Linear(64, 1374)` = 16,384 + 64 + 87,936 + 1374 = **105,758** params
- Net change: **saves ~247K parameters** while adding a nonlinear transformation

The bottleneck actually _reduces_ total parameters because 256→64→1374 has fewer
weights than 256→1374. The benefit is the nonlinear ReLU between the two projections,
which lets the model learn GP-specific feature combinations that differ from what
the n-gram families need.

### Testing

No existing tests directly instantiate `KCDecoder` in isolation (verified by grep).
The decoder is created inside `TrainingClassifier.__init__` at
[train/models.py line 129](file:///Users/jomofisher/projects/kotogram/train/models.py#L129):

```python
self.kc_decoders = KCDecoder(config.kc_vocab_size, kc_target_specs)
```

Integration tests that run `forward_kc` will exercise the decoder automatically.
Run `test.sh` to verify no regressions. The change is backward-incompatible with
saved checkpoints (different weight shapes), so training must restart from scratch.

### Configuration

Consider adding `gp_bottleneck_dim` to `KcConfig` in
[train/config.py](file:///Users/jomofisher/projects/kotogram/train/config.py)
(around line 140-150 where other GP-related config lives):

```python
gp_bottleneck_dim: int = 64  # 0 = no bottleneck (legacy behavior)
```

Then pass it through when constructing `KCDecoder`.

### PNU Loss Context

The grammar_point family uses PNU (Positive-Negative-Unlabeled) loss defined in
[train/kc_trainer.py](file:///Users/jomofisher/projects/kotogram/train/kc_trainer.py)
method `_multilabel_pnu_loss` (lines 394-548). The loss weights are:

- `gp_pos_weight: 1.0` — positive labels
- `gp_neg_weight: 1000.0` — explicit negatives (very strong)
- `gp_unlabeled_weight: 1.0` — unlabeled slots (treated as weak negatives)

Per-GP prior-based scaling is computed in `_init_gp_prior_weights` (lines 550-603),
which adjusts unlabeled and negative weights based on each GP's corpus frequency.

The over-prediction problem may also benefit from adjusting these weights, but the
bottleneck addresses the architectural issue (shared hidden layer interference)
independently of the loss function.
