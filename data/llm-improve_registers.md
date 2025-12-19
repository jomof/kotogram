# Instruction: Refining Register Overrides

The objective is to iteratively audit the style classification model's grounding data and provide high-quality manual overrides to correct misclassifications.

## The Iterative Loop (Learnings from Dec 2025)

1.  **Examine Samples**: Open `models/style/register_samples.csv`. This file contains ~36 sampled sentences across 12 registers.
2.  **Evaluate Accuracy**: Check if the `register` column correctly describes the `sentence`.
3.  **Create Paired Overrides**:
    -   **Neutralization (The Primary Fix)**: Most "False Positives" should be moved to `data/jpn_sentences_neutral.tsv`.
        -   *Example*: If a standard formal sentence is labeled as `hakataben`, verify it contains NO dialect markers (like `~to`, `~ken`). If it's standard Japanese, move it to Neutral.
    -   **Positive Discriminators**: When adding to `data/jpn_sentences_<register>.tsv`, use **unambiguous** examples.
        -   *Good Hakataben*: "なんばしよっと？" (Strong dialect grammar)
        -   *Bad Hakataben*: "雨が降るけん" (Standard causal "ken" vs dialect "ken", ambiguous without context)
    -   **Balance**: Always add an `_a` and `_b` pair to keep training data balanced.

4.  **Verification**:
    ```bash
    rm -rf .cache/kotogram_shards && ./train_style.sh --label
    ```

## Specific Linguistic Challenges

### 1. Passive vs. Sonkeigo (Critical)
The model consistently confuses the Passive verb form (`~reru`/`~rareru`) with Sonkeigo (Honorific) because they are morphologically identical.
-   **Symptom**: "彼は説得された" (He was persuaded) -> Classifies as Sonkeigo.
-   **Fix**: Add these specifically to `jpn_sentences_neutral.tsv`.
-   **Advice**: You cannot fix this completely without semantic understanding. Focus on "Neutralizing" obvious Passives.

### 2. Dialect vs. Casual
The model often hallucinates Kansaiben or Hakataben on standard Casual speech (`~janai ka`, `~darou`, `~ya`).
-   **Fix**: Move these "False Dialect" samples to `jpn_sentences_neutral.tsv`.
-   **Positive Examples**: Ensure the *actual* dialect files contain only strongest dialect markers (`~yan`, `~hen` for Kansai; `~to`, `~bai`, `~tai` for Hakata).

### 3. Character Registers (Success Story)
Registers like **Ojousama**, **Kyoshigo**, **Netslang**, **Burikko** are highly distinct.
-   **Strategy**: These are driven by sentence-final particles and specific vocabulary.
-   *Ojousama*: `~desu wa`, `~masu no` (NOT just polite `desu/masu`).
-   *Kyoshigo*: `~nasai`, `~ikemasen` (Imperative/Prohibitive).
-   *Burikko*: `~mon`, `~cham`, `~o`.

## Implementing New Registers (Learnings from Round 7)

If asked to implement a new register from `register-catalog.txt`:

1.  **Code Changes Required**:
    -   `kotogram/analysis.py`: Add to `RegisterLevel` Enum.
    -   `kotogram/model.py`: Update `NUM_REGISTER_CLASSES` and ID mappings.
    -   `scripts/rule_based_analysis.py`: Add detection rules in `rule_based_register`.

2.  **Dataset Creation Pitfall (CRITICAL)**:
    -   The TSV format for `data/jpn_sentences_<register>.tsv` MUST be:
        `sentence_id` TAB `jpn` TAB `sentence`
    -   *Do NOT forget the `jpn` column!*

3.  **Rule-Based Logic Tip**:
    -   **Avoid Over-Triggering**: Standard polite forms often overlap with archaic/formal registers.
        -   *Fail*: `if 'gozaru' in surface: return BUSHI` (Triggers on "arigatou gozaimasu")
        -   *Success*: Check *following* tokens. If `gozaru` is followed by polite aux (`masu`, `mase`), it is NOT Bushi.

## Summary of Commands
-   **Labeling**: `./train_style.sh --label` (clears cache and re-processes)
-   **Check Samples**: `head -n 40 models/style/register_samples.csv`
-   **Confusion Matrix**: `./train_style.sh --confusion` (use after large data updates to see impact)
