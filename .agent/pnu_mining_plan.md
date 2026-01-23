# Hard Negative Mining for Grammar Points - PNU Learning System

## Problem Statement

Grammar point labels in the corpus are extremely sparse in a Positive-Negative-Unlabeled (PNU) setting:
- **gp0888 example**: 31 positive, 35 negative, 1,077,390 unlabeled (~99.99% unlabeled)
- Most grammar points have similarly sparse labels
- Need to mine hard negatives/positives to improve training data quality

## Research-Based Approach

### PNU Learning Framework

Given the extreme sparsity, we'll use a hybrid approach combining:

1. **PU Learning Foundation** (Positive-Unlabeled Learning)
   - Treat labeled negatives as "reliable negatives"
   - Use unlabeled as mixture of hidden positives and negatives
   - Apply class prior estimation

2. **Loss-Based Hard Negative Mining**
   - Track per-sample loss during training
   - High loss on "negative" predictions → candidate hard negative
   - High loss on "positive" predictions → candidate hard positive

3. **Uncertainty-Based Sampling**
   - Samples near decision boundary are most informative
   - Use prediction uncertainty (entropy) as signal
   - Combine with loss for robust ranking

4. **Cost-Sensitive Learning**
   - Weight positive examples higher (due to extreme rarity)
   - Use focal loss to focus on hard examples
   - Asymmetric loss for false positive/negative

## Technical Design

### Model Architecture

**Lightweight Grammar-Specific Classifier:**
```
Input: Token features from cached .cache/style_dataset/
  ↓
Shared Embedding Layer (from main model, frozen initially)
  ↓
2-Layer Transformer Encoder (lightweight, trainable)
  ↓
Attention Pooling
  ↓
Binary Classification Head: [grammar_present, grammar_absent]
```

**Key Design Decisions:**
- Use pre-computed token features (no re-tokenization)
- Smaller model than main style classifier (2 layers vs 4)
- d_model=256, hidden_dim=1024
- Focus on one grammar point at a time
- Train on MPS with fp16 for speed

### Training Strategy

**Phase 1: Warm-up with Labeled Data (5-10 epochs)**
- Train only on labeled positive/negative examples
- Extreme class weighting (positive_weight = neg_count / pos_count)
- Focal loss with γ=2 to focus on hard examples
- Track baseline performance

**Phase 2: PNU Training with Unlabeled (15-20 epochs)**
- Sample unlabeled data in batches
- Use current model to pseudo-label unlabeled
- Apply confidence-based reweighting
- Continue tracking per-sample losses

**Phase 3: Loss Accumulation (5 epochs)**
- Freeze model or use minimal learning rate
- Run through entire dataset
- Accumulate loss statistics per sample
- Track prediction confidence and uncertainty

### Hard Negative Mining Metrics

For each sentence, compute:

1. **Loss Contribution**: Total accumulated loss across epochs
2. **Prediction Confidence**: `max(P(positive), P(negative))`
3. **Uncertainty**: `H = -Σ p_i log(p_i)` (entropy)
4. **Boundary Score**: `|P(positive) - 0.5|` (smaller = closer to boundary)
5. **False Prediction Score**: 
   - For unlabeled: High loss + model predicts negative → hard positive candidate
   - For unlabeled: High loss + model predicts positive → hard negative candidate

**Hard Negative Candidate Score:**
```python
score = (loss_contribution * 0.4) + 
        (uncertainty * 0.3) + 
        (boundary_proximity * 0.2) +
        (prediction_confidence * 0.1)
```

Filter to sentences where model predicts **positive** (but likely false positive).

**Hard Positive Candidate Score:**
```python
score = (loss_contribution * 0.4) + 
        (uncertainty * 0.3) + 
        (boundary_proximity * 0.2) +
        (prediction_confidence * 0.1)
```

Filter to sentences where model predicts **negative** (but likely false negative).

### Implementation Structure

**Main Script**: `scripts/curate_study_grammar_label.py`
```python
def main(grammar_label: str, apply: bool = False):
    if not apply:
        # Training and mining mode
        run_pnu_training(grammar_label)
        generate_candidates(grammar_label)
    else:
        # Apply curated labels to database
        apply_curated_labels(grammar_label)
```

**CLI Integration**: `scripts/curate`
```bash
# Usage:
scripts/curate study gp0888              # Train and generate candidates
scripts/curate study gp0888 --apply      # Apply curated labels
```

### Output Format

**`.cache/curate/study/gp0888/best-hard-negative-candidates.txt`:**
```
# Hard Negative Candidates for gp0888
# Score | Sentence
# ------|----------
0.9234 | 今日はとても暑いです。
0.8891 | 彼は学生ではありません。
0.8654 | この本を読みました。
...
```

User manually reviews and **removes** lines for actual positives (false alarms).

**`.cache/curate/study/gp0888/best-hard-positive-candidates.txt`:**
```
# Hard Positive Candidates for gp0888
# Score | Sentence
# ------|----------
0.9456 | このラーメン、うまいぜ！
0.9123 | 今日も頑張るぜ！
0.8934 | 絶対に勝つぜ！
...
```

User manually reviews and **removes** lines for actual negatives (false alarms).

### Apply Mode Logic

After manual curation:

1. Read `best-hard-negative-candidates.txt`
2. For each remaining sentence (not removed by user):
   - Add grammar_label to `grammar_negative` column
   - If was in `grammar` column, remove it (was hard positive, now known negative)
3. Read `best-hard-positive-candidates.txt`
4. For each remaining sentence:
   - Add grammar_label to `grammar` column
   - If was in `grammar_negative` column, remove it
5. Ensure no duplicates in each column
6. Update `corpus.db` with transaction

### Progress and Monitoring

- Use `rich` progress bars for all phases
- Show loss curves during training
- Display candidate statistics
- Save training metrics to `.cache/curate/study/gp0888/metrics.json`

## Success Criteria

- Generates 100-500 hard negative candidates ranked by usefulness
- Generates 100-500 hard positive candidates ranked by usefulness
- Top candidates should have >50% true positive rate when manually reviewed
- Training completes in <30 minutes on MPS
- Full pipeline is reusable for any grammar point label

## Dataset Statistics Needed

Before training, collect and display:
- Total sentences in corpus
- Positive count for grammar point
- Negative count for grammar point
- Unlabeled count
- Positive/negative ratio
- Label density (labeled / total)
