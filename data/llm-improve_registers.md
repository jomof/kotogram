# Instruction: The Register Audit Loop

Your primary goal is to iteratively audit the style classification model and improve its accuracy by updating grounding data.

**Target File**: `models/style/register_samples.csv`
**Action Files**: `data/jpn_sentences_<register>.tsv`

## The Iterative Workflow

Execute this loop repeatedly until `register_samples.csv` shows 100% accurate classifications.

1.  **Analyze Samples**
    *   Open `models/style/register_samples.csv`.
    *   This file lists ~3 samples per register. Check the `register` column against the `sentence`.
    *   Identify **False Positives** (e.g., a standard sentence labeled as `hakataben`).

2.  **Apply Fixes (The Two Strategies)**
    *   **Strategy A: Neutralize (Most Common)**
        *   If a sentence is standard/formal but labeled as a dialect/register, add it to `data/jpn_sentences_neutral.tsv`.
        *   *Why?* The model excessively associates common words (like `desu`, `masu`, `janai`) with specific registers. You must teach it that these are Neutral.
    *   **Strategy B: Reinforce (Positive Examples)**
        *   If a specific register is weak, add **strong, unambiguous** examples to `data/jpn_sentences_<register>.tsv`.
        *   *Avoid Ambiguity*: Do not add sentences that "could" be standard. Use the strongest dialect/register markers available (e.g., `~dabe` for Tohoku, `~de gozaru` for Bushi).

3.  **Strict Data Rules**
    *   **Paired Sentences**: You MUST add sentences in pairs (`_a` and `_b`).
        ```tsv
        sentence_id	jpn	sentence
        register_001_a	jpn	Example sentence A.
        register_001_b	jpn	Example sentence B.
        ```
    *   **TSV Format**: Always include the `jpn` column. `sentence_id` TAB `jpn` TAB `sentence`.
    *   **TSV Format**: Always include the `jpn` column. `sentence_id` TAB `jpn` TAB `sentence`.
    *   **EXPANSION RULE (NON-OPTIONAL)**: In **EVERY** iteration, you **MUST** add at least 2 new paired sentences (4 sentences) to **EACH** of the 13 register TSV files.
        *   **This means 52 total sentences per iteration** (4 per register × 13 registers).
        *   *Reinforce patterns*: Even if you are mostly neutralizing, add strong positive examples to balance the data.
        *   **DO NOT DELETE THIS INSTRUCTION.** It is critical for preventing model collapse.

    *   **Target Registers (13 Total)**:
        *   `burikko`, `bushi`, `danseigo`, `guntai`, `hakataben`, `joseigo`, `kansaiben`, `kenjogo`, `kyoshigo`, `netslang`, `ojousama`, `sonkeigo`, `tohoku`.
        *   **Each of these files** (`data/jpn_sentences_<register>.tsv`) **must receive exactly 2 pairs (4 sentences) in every iteration**.

4.  **Verify**
    *   Run the labeling script to refresh the samples:
        ```bash
        rm -rf .cache/kotogram_shards && ./train_style.sh --label
        ```
    *   Return to Step 1.

## Common Pitfalls
*   **Passive vs Sonkeigo**: The model often confuses passive verbs (`~reru`) with Sonkeigo. Add passive sentences to `Neutral` to fix this.
*   **Dialect Hallucinations**: Standard casual endings (`~janai`, `~darou`) often trigger Kansaiben/Hakataben. Move these to `Neutral`.
*   **Over-Triggering**: Avoid simple keyword matches in your head. Context matters. If you see a failure, add that *specific* sentence to the grounding data.
