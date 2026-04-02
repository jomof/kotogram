# Branch: `collapse-firewall` — Feature Summary

Branch created: **2026-03-30**  
Branched from: `main`  
Focus: Preventing representation collapse in deep (9+ layer) transformer encoders during `recon_bpd` training.

---

## Feature 1: Collapse Firewall Regularization Trio
**Commits:** `01717db` → `e5d470e` (Mar 30)  
**Files:** `scratch/recon_bpd.py`

The founding commit of the branch. Three independent regularization mechanisms were added simultaneously to combat depth-induced representation collapse when scaling from 6 to 9+ encoder layers. **No architectural changes.**

### 1a. Stop-Gradient Consistency Loss (BYOL/SimSiam)
- Symmetrized `.detach()` on one branch of the dual-mask cosine similarity loss.
- Breaks the constant-collapse attractor that pulls all token embeddings toward a single fixed point.
- Without this, the consistency loss gradient descends toward trivial solutions (all embeddings identical).

### 1b. VICReg Variance-Covariance Regularization
- Added hinge loss on per-dimension std of pooled encoder output to prevent any dimension from collapsing to near-zero variance.
- Added off-diagonal covariance penalty to discourage correlated (redundant) dimensions.
- Maintains high-rank representations across depth.
- Follow-up tuning: `var_weight 25→11`, `gamma 1.0→0.3`, VICReg weight adjustments (`86b2e5b`, `e5d470e`).
- Bug fix (`5be3873`): accumulated epoch average instead of noisy last-batch snapshot; freed computation graph to prevent memory leak; upcasted to float32 before variance/covariance math to avoid TF32/FP16 noise.

### 1c. Stochastic Depth (LayerDrop)
- Randomly skips entire transformer layers during training.
- Prevents over-smoothing at depth and ensures robust representations at every effective depth.
- Follow-up (`a6da09d`): raised `layer_drop_prob` from `0.1 → 0.5` for stronger regularization; protected the first encoder layer from being skipped.

---

## Feature 2: Stochastic Token Dropout & Rescue Gate
**Commits:** `2d95449`, `9e11e1b`, `f3b2a8f` (Mar 30–31)  
**Files:** `scratch/recon_bpd.py`, `scratch/recon_bpd_optuna.py`

Modifies how tokens are selected for the hard-token reconstruction loss.

### 2a. Stochastic Token Dropout (`2d95449`)
- Replaced deterministic cosine-similarity gate with uniform random sampling at probability = `current_threshold`.
- Goal: avoid hard semantic boundaries between easy/hard tokens, which create sharp gradient boundaries.

### 2b. Stochastic Rescue Gate (`9e11e1b`)
- Refined dropout: always keep tokens with `cos_sim < threshold` (deterministically hard), and *randomly rescue* easy tokens with probability `(1 - threshold)`.
- Softens the boundary without abandoning the semantic signal: hard tokens always train, most easy tokens skip, a fraction of easy tokens are rescued stochastically.
- Applied to 9-layer config in Optuna (`f3b2a8f`).

---

## Feature 3: Reconstruction Spot-Check Test Suite
**Commits:** `7c2af8c`, `152042f`, `3ba0a65`, `8273817`, `7906f9b`, `7c7b52c`, `98338dd`, `28d17ec`, `b529e0d`, `29730b6` (Mar 31)  
**Files:** `scratch/recon_bpd_test.py` [NEW], `scratch/recon_bpd_test.txt` [NEW], `scratch/recon_bpd.py`, `scratch/recon_bpd_optuna.py`

A post-epoch behavioral evaluation suite for the reconstruction model.

- **`recon_bpd_test.py`**: runs masked reconstruction through the live model and reports pass/fail per probe. Supports `--check` mode to validate tokenization alignment without a model.
- **`recon_bpd_test.txt`**: initial 13-probe set (particles, content words, multi-mask). Grew to **80+ probes across 20 grammar categories** (`7906f9b`), then expanded further.
- **Integration**: `recon_bpd.py` calls the test suite after each epoch and merges metrics. Optuna logs `recon_test_pct/pass/fail` to MLflow and uploads `epoch N failures.txt` as an artifact.
- **Diagnostics**: masked variant included in failure report; `verbose.txt` uploaded to MLflow; `strict` and `alt` metrics added.
- Sudachi tokenizer fix, non-numeric checkpoint string stripping (`152042f`).

---

## Feature 4: Deterministic Resume & Checkpoint Infrastructure
**Commits:** `d28d0b3`, `61e3d4c`, `ee9a114` (revert), `c737d04` (Mar 31)  
**Files:** `scratch/recon_bpd.py`, `scratch/recon_bpd_optuna.py`, `scratch/recon_bpd_checkpoint.py` [NEW]

Makes training resumption exactly reproducible and extracts checkpoint logic into its own module.

### 4a. RNG State Checkpointing (`d28d0b3`)
- Saves `torch`, CUDA, and DataLoader generator RNG states into the checkpoint.
- Restores on resume so batch order and all stochastic ops (LayerDrop, rescue gate, token dropout) are bit-for-bit identical.
- Guards MLflow replay loops to skip backfill for metrics that already exist in a resumed run.

### 4b. Per-Epoch Reseeding — attempted then reverted (`61e3d4c`, `ee9a114`)
- Tried `manual_seed(seed + epoch)` as an alternative to full RNG state serialization.
- Reverted after evaluation.

### 4c. Checkpoint I/O Extraction (`c737d04`)
- New file `recon_bpd_checkpoint.py`: `TrainCheckpoint` dataclass, `save_checkpoint` / `load_checkpoint`, `EpochContext` with an `artifact_paths` bag.
- Changes to the checkpoint module don't invalidate the config hash (important for Optuna trial identity).
- `recon_bpd_test.py` becomes self-service: receives `EpochContext`, merges its own metrics, registers its own artifacts.
- Optuna callback generically drains `ctx.artifact_paths` without knowing test internals.

---

## Feature 5: Temperature Annealing & KL Warmup
**Commits:** `a01949e`, `1fefb65`, `18327c0`, `93a6720`, `7b0bd57` (Mar 31)  
**Files:** `scratch/recon_bpd.py`, `scratch/recon_bpd_optuna.py`

Staged training schedule to prevent premature commitment of KC probability assignments.

### Temperature Annealing
- New `TrainConfig` fields: `temperature_start_multiplier` (default 3.0) and `temperature_anneal_epochs` (default 30).
- Training starts with `temperature = target × 3.0` (soft, high-entropy KC probs), linearly cooling to the config target.
- Motivation (cited: Jang/Gu/Poole ICLR 2017, Maddison/Mnih/Teh ICLR 2017): the length-proportional KL sparsity needs time to negotiate logit allocation across sentence lengths *before* sigmoid sharpens assignments into irreversible binary decisions.
- Temperature metric logged to MLflow each epoch.

### KL Sparsity Warmup
- KL sparsity weight ramped from ~0 to full strength on the same schedule as temperature annealing, with a **quadratic ramp** (near-zero for early half, accelerates as temperature cools).
- Ensures KC allocation is negotiated under soft probabilities before the sparsity penalty bites.
- `kl_warmup` metric logged to MLflow each epoch.

### Layer count experiments
- `93a6720` / `7b0bd57`: switched back to 6 layers for direct comparison under annealing.

---

## Feature 6: Bidirectional Decoding (Diagnostic Head)
**Commit:** `814e314` (Mar 31)  
**Files:** `scratch/recon_bpd.py`

Extends the reconstruction decoder and adds a diagnostic head.

### Bidirectional Positional Encoding
- `ReconDecoder` previously used only end-relative position embeddings (position 0 = last content token).
- Added a second start-relative embedding (position 0 = first content token).
- Together they implicitly encode both **absolute position** and **sentence length**: a token at `start_rel=2, end_rel=5` is the third token in an 8-token sentence.
- Decoder input dimension grows: `kc_vocab_size + 2 × pos_embed_dim`.

### Sentence Length Prediction Head (Diagnostic)
- New `model.length_head`: MLP (1024 → 128 → 1) predicting sentence length from KC probs alone.
- Weight `length_pred_weight = 0.01` — intentionally low; this is a diagnostic, not a training driver.
- Normalized MSE (divided by mean length²) so the loss scale is independent of sentence length.
- Reports `length_pred_mae` and `length_pred_loss` metrics.

---

## Feature 7: Non-Content Masking (Data Augmentation)
**Commits:** `3612ff8`, `be46ba7` (Mar 31 – Apr 1)  
**Files:** `scratch/recon_bpd.py`, `scripts/label.py`, `scratch/recon_bpd_optuna.py`

Content-aware data augmentation that randomly drops grammatical/particle tokens during training.

### Content Mask Generation (`scripts/label.py`)
- New `_is_content_char`: classifies Unicode codepoints as content (Hiragana, Katakana, CJK, digits, Latin) or non-content (particles, punctuation, special tokens).
- New `_compute_and_write_content_mask`: iterates the surface vocab and writes a per-token-ID binary mask to `content_mask.bin` in the dataset cache dir.
- A token is "content" if ALL its characters are in content ranges (multi-character tokens are always content).

### Training Augmentation (`scratch/recon_bpd.py`)
- Loads `content_mask.bin` via `torch.from_file` (shared memory, zero-copy).
- Before each batch: identifies non-content tokens (`is_non_content_valid`), randomly drops 50% of them by zeroing their `input_ids` (→ PAD) and clearing their `attention_mask` bit.
- Special tokens (IDs < 4) are always protected.
- Forces the encoder to generalize across varied particle/function-word configurations, reducing memorization of surface-level grammatical patterns.
- Optuna integration (`be46ba7`): Optuna config exposes `non_content_mask_ratio` as a tunable hyperparameter.

---

## Infrastructure & Tooling Changes

| Commit | Change |
|--------|--------|
| `cbfe9ed` | Adhoc trial name auto-derived from last git commit touching `recon_bpd.py` (`git log -1 --format=%s`) |
| `40bd17e` | Suppressed spurious `LambdaLR.step()` warning (called internally before any optimizer step) |
| `8273817` | 6-layer test config for direct comparison |

---

## Training Configuration Evolution (key hyperparameters)

| Parameter | Initial | After tuning |
|-----------|---------|-------------|
| `layer_drop_prob` | 0.1 | 0.5 (first layer protected) |
| `vicreg_var_weight` | 25 | 11 |
| `vicreg_gamma` | 1.0 | 0.3 |
| `temperature_start_multiplier` | — | 3.0 |
| `temperature_anneal_epochs` | — | 30 |
| `length_pred_weight` | — | 0.01 |
| `non_content_mask_ratio` | — | 0.5 |

---

## Summary of Training Changes

The branch addresses a single core problem — **representation collapse at depth** — through a layered sequence of interventions:

1. **Structural regularization** (VICReg, stop-gradient, LayerDrop) to maintain high-rank encoder output.
2. **Soft token selection** (stochastic rescue gate) to avoid sharp semantic boundaries in supervision.
3. **Behavioral regression testing** (spot-check suite) to detect model degradation per epoch.
4. **Reproducible training** (RNG checkpointing, checkpoint module extraction) for reliable comparisons.
5. **Staged training schedule** (temperature annealing + KL warmup) to give the KC bottleneck time to negotiate assignments before sparsity pressure hardens them.
6. **Richer positional signal** (bidirectional encoding) and a diagnostic head to detect length leakage.
7. **Content-aware augmentation** (non-content masking) to prevent memorization of grammatical surface patterns.
