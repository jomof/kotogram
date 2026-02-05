---
description: Investigate a grammar point and make improvements to the training corpus
---

The goal is to improve the accuracy of the training corpus for a specific grammar point (gpXXXX) by adding high-quality positive validation data, challenging adversarial negative data, and verifying existing labels.

## Phase 1: Research
1.  **Identify Definition**: Run `sqlite3 data/corpus.db "SELECT name FROM grammar WHERE id = 'gpXXXX';"` to get the official name.
2.  **Review Study Material**: Look in `.cache/curate/study/gpXXXX`. Read the YAML definition file to understand nuances, false friends, and existing examples.
3.  **Find Related Grammar**: Look in `data/grammar/*.yaml` for similar grammar points that might be confused with the target. You will need to list these in your implementation plan.

## Phase 2: Planning
1.  **Create Implementation Plan**: Create `implementation_plan.md` with:
    *   **Goal**: Augment gpXXXX.
    *   **Related Grammar**: List the similar grammar points you found in `data/grammar/*.yaml`.
    *   **User Review**: Explicitly ask for review of your adversarial negative strategy.
    *   **Adversarial Strategy**: List specific patterns you will generate to confuse the model (e.g., similar keywords, partial matches, correct usage of confounding grammar).
    *   **Data Cleaning**: Plan to review and verify existing data.
2.  **Get Approval**: detailed `notify_user` call.

## Phase 3: Execution
**Important Rule**: All incidental sentence files you create (new positives, corrections, mined lists) MUST be saved in the `.cache/curate/study/gpXXXX/` directory. Do not clutter the root directory.

1.  **Generate Data**:
    *   **30 New Positives**: Diverse examples covering all usages defined in the YAML.
    *   **30 New Adversarial Negatives**: Sentences that *look* like the grammar point but aren't (e.g., if target is `～ている`, generate `～てある`, `～ておく`, `～た`, `～る` sentences).
    *   *Save to*: `.cache/curate/study/gpXXXX/new_positives.txt` and `.cache/curate/study/gpXXXX/new_negatives.txt`.
2.  **Verify Existing Data (Crucial)**:
    *   **Confirm Positives**: Review `existing-hard-positive.txt`. Ensure all sentences are *undisputedly* positive for gpXXXX. If any are wrong, add them to `.cache/curate/study/gpXXXX/correction_negatives.txt`.
    *   **Confirm Negatives**: Review `existing-hard-negative.txt`. Ensure all sentences are *undisputedly* negative (do not contain gpXXXX). If any are actually positive, add them to `.cache/curate/study/gpXXXX/correction_positives.txt`.
    *   **Sanitize**: If you find awkward, unnatural, or ungrammatic sentences in ANY file, mark them for removal (set grammatic=0).
3.  **Mine Predictions**:
    *   Review `high-certainty-positives.txt`. **Crucial**: Only look for False Positives (sentences that are wrong). Do *not* extract valid positives from this list.
    *   Review `most-uncertain.txt` (rescue true positives).
    *   *Save mined lists to*: `.cache/curate/study/gpXXXX/mined_false_positives.txt` (for false positives found in high-certainty) or `.cache/curate/study/gpXXXX/mined_positives.txt` (for true positives found in uncertain).
4.  **Apply Changes**:
    *   Use `scripts/curate upsert` to apply changes.
    *   **New Data**: `scripts/curate upsert --sentences='.cache/curate/study/gpXXXX/new_positives.txt' --grammar='+gpXXXX' --allow-insert`
    *   **Corrections**: `scripts/curate upsert --sentences='.cache/curate/study/gpXXXX/correction_negatives.txt' --grammar='-gpXXXX'`
    *   **Ungrammatic**: `scripts/curate upsert 'Bad sentence here' --grammatic=0`

## Phase 4: Verification
1.  **Verify Counts**:
    *   Run sqlite3: `SELECT 'Positives:', COUNT(*) FROM corpus_gp_pos WHERE gp_id = 'gpXXXX' UNION ALL SELECT 'Negatives:', COUNT(*) FROM corpus_gp_neg WHERE gp_id = 'gpXXXX';`
    *   Ensure counts have increased appropriately and corrections are reflected.
2.  **Create Walkthrough**:
    *   Create `walkthrough.md` documenting added data, corrections made to existing data, and final counts.

## Final Report
Notify the user with the final counts and a link to the walkthrough.