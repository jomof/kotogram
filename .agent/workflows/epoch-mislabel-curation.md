---
description: Curate mislabels from worst-*.tsv after each training epoch
---

After each epoch, a mislabel curation workflow runs on the worst-*.tsv files in `.cache/training/`.

## Workflow

1. **Snapshot**: Copy `worst-*.tsv` into `.cache/training/snapshots/epNN/`.
2. **Analyze** these five files: worst-formality, worst-gender, worst-grammar_point, worst-grammatic, worst-register.
3. **Apply fixes** per category (see below).

### Batch mode (recommended)

Run `scripts/curate fix-epoch N` to apply fixes in batch and get change counts:

```bash
scripts/curate fix-epoch 35                    # formality, gender, grammatic, register
scripts/curate fix-epoch 35 --grammar-point     # + grammar_point (verify negatives first)
```

Reports only **actual changes** (no-ops excluded) in the summary.

## Grammar Point Protocol

For each row in `worst-grammar_point.tsv` (target: none = currently unlabeled):

### (1) Add top-1 predicted as negative — **only after verification**

**You must verify the sentence does NOT contain the grammar point before adding it as negative.**

- Look up the grammar point: `sqlite3 data/corpus.db "SELECT name FROM grammar WHERE id = 'gpXXXX';"`
- Read the YAML definition: `data/grammar/<name>.yaml` (find via `ls data/grammar/ | grep -i <pattern>`)
- Check whether the sentence actually contains that grammar pattern. If it does, do **not** add it as negative; add it as positive instead.
- Only run `scripts/curate upsert 'sentence' --grammar='-gpXXXX'` when verified negative.

### (2) Add positives when they exist

- Identify any grammar points that actually appear in the sentence.
- Add them: `scripts/curate upsert 'sentence' --grammar='+gpXXXX'` (or `+gpXXXX,+gpYYYY` for multiple).

## Other Categories

- **Formality, gender, grammatic, grammar_point**: `scripts/curate upsert 'sentence' --formality=... --gender=... --grammatic=... --grammar=...`

## Summary

When reporting curation results, **only count fixes where an actual change was made**. If the corpus already had the correct value (upsert output shows "unchanged"), do not include it in the count.
- **Register**: Direct SQL on `data/corpus.db`:
  ```sql
  UPDATE sentences SET register_ids = '0' WHERE sentence = '...';
  ```
  Register IDs: 0=neutral, 1=sonkeigo, 2=kenjogo, 3=kansaiben, 4=hakataben, 5=kyoshigo, 7=ojousama, 8=guntai, 9=joseigo, 10=danseigo, 12=tohoku.
