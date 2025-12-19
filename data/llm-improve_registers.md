# Instruction: Refining Register Overrides

The objective is to iteratively audit the style classification model's grounding data and provide high-quality manual overrides to correct misclassifications.

## The Iterative Loop

1.  **Examine Samples**: Open `models/style/register_samples.csv`. This file contains ~36 sampled sentences across 12 registers.
2.  **Evaluate Accuracy**: Check if the `register` column correctly describes the `sentence`.
    -   *Example*: If a standard formal sentence is labeled as `hakataben`, it's a False Positive.
3.  **Create Paired Overrides**: For every misclassification, add a "Paired Set" to the corresponding `data/jpn_sentences_<register>.tsv` file.
    -   **CRITICAL RULE**: Every correction MUST be paired with a second, grammatically similar augmented sentence (Line A and Line B). This keeps the training data balanced.
    -   *Naming Convention*: `<register>_<next_id>_a` and `<register>_<next_id>_b`.
    -   *Target Files*:
        -   `data/jpn_sentences_neutral.tsv` (Most common for neutralizing false positives)
        -   `data/jpn_sentences_sonkeigo.tsv`
        -   `data/jpn_sentences_kansaiben.tsv`
        -   ... etc.
4.  **Verification**: After updating the TSV files, run the labeling script to refresh the samples:
    ```bash
    rm -rf .cache/kotogram_shards && ./train_style.sh --label
    ```
5.  **Audit Again**: Check the refreshed `models/style/register_samples.csv` and repeat until the samples are 100% accurate.

## Expert Context
- **Rule-Based Logic**: Many misclassifications stem from heuristics in `scripts/rule_based_analysis.py`. While you can patch the logic there for systemic issues, manual overrides in the TSV files take absolute precedence and are often safer for specific outliers.
- **Previous Work**: See `expert_analysis.md` (if available in the brain directory) for the latest audit results of the 36 samples.
- **Unification**: All overrides have been moved out of the legacy `jpn_sentences_register.tsv` (now deleted) into modular register-specific files.

## Summary of Commands
- **Labeling**: `./train_style.sh --label` (clears cache and re-processes)
- **Check Samples**: `head -n 40 models/style/register_samples.csv`
- **Confusion Matrix**: `./train_style.sh --confusion` (use after large data updates to see impact)
