---
description: Add positive and negative sentences for a grammar point
---

I'll give you a grammar point in the form gpXXXX.

### 1. Identify the Grammar Point
1. Run `sqlite3 data/corpus.db "SELECT name FROM grammar WHERE id = 'gpXXXX'"` to get the human-readable name.
2. Locate the definition file: `ls "data/grammar/<name>"*`.
3. Read the file using `view_file`.
   - **Goal**: Understand the nuance, strict formation rules, and existing false friends.
   - **Check**: Are there existing "competing grammar" or "false friend" sections? Use them as inspiration for adversarial negatives.


### 2. Generate Sentences (Draft)
- **Positive Sentences (30)**:
  - Create 30 diverse examples (varied contexts, casual/formal, specific/generic nouns).
  - **Complexity**: Mix in *other* grammar points.
  - **Hygiene**: Must be natural, grammatic Japanese.

- **Adversarial Negative Sentences (30)**:
  - **False Friends**: Use grammar that looks/sounds similar but isn't it (e.g., for `それどころか`, use `どころか` without `それ`).
  - **Phonetic/Visual Traps**: Characters present but unconnected grammar (e.g. `それ、どこから` vs `それどころか`). 
  - **Semantic Traps**: Similar meaning, different grammar.

- (MANDATORY) ALL SENTENCES ADDED MUST BE GRAMMATIC AND NATURAL JAPANESE.
- (MANDATORY) ALL SENTENCES MUST BE NEW. VERIFY THEY DON'T ALREADY EXIST IN THE DATABASE.

### 3. Review Plan (MANDATORY)
Write a plan to an artifact file and `notify_user` before making any database changes.
Present the full list of 60 sentences (30 positive, 30 negative) in the implementation plan or message.
- Double check: ALL SENTENCES ADDED MUST BE GRAMMATIC AND NATURAL JAPANESE.
- Explain the logic behind the adversarial traps.
- Format a checklist of the steps you will take, including the final report below.

### 4. Batch Upsert
1. Write positive sentences to `positive_list-gpXXXX.txt`.
2. Write negative sentences to `negative_list-gpXXXX.txt`.
3. Run the upsert commands.
   - **CRITICAL**: Use `--grammar="-gpXXXX"` (with **equals sign**) for negative labels.
   
```bash
source ./requirements.sh
# Positive Upsert
./scripts/curate upsert --sentences positive_list.txt --grammar "+gpXXXX" --allow-insert --grammatic 1 --formality neutral --gender neutral

# Negative Upsert
./scripts/curate upsert --sentences negative_list.txt --grammar="-gpXXXX" --allow-insert --grammatic 1 --formality neutral --gender neutral
```

### 5. Sample Random Negatives
Sample 30 random grammatic sentences that are not already labeled for this grammar point and mark them as negative:

1. Query random sentences:
   ```bash
   sqlite3 data/corpus.db "SELECT sentence FROM corpus WHERE grammatic = 1 AND (grammar IS NULL OR grammar NOT LIKE '%gpXXXX%') AND (grammar_negative IS NULL OR grammar_negative NOT LIKE '%gpXXXX%') ORDER BY RANDOM() LIMIT 30;"
   ```
2. Verify none contain the target grammar pattern.
3. Write to `random_negative.txt` and upsert:
   ```bash
   source ./requirements.sh
   ./scripts/curate upsert --sentences random_negative.txt --grammar="-gpXXXX" --grammatic 1 --formality neutral --gender neutral
   ```
4. Cleanup: `rm random_negative.txt`

### 6. Verify & Cleanup
1. Verify database counts:
   ```bash
   sqlite3 data/corpus.db "SELECT count(*) FROM corpus WHERE grammar LIKE '%gpXXXX%'; SELECT count(*) FROM corpus WHERE grammar_negative LIKE '%gpXXXX%';"
   ```
2. Cleanup: `rm positive_list.txt negative_list.txt`
3. Cleanup: remove the Step 3 plan artifact after completion (e.g. `rm compare/augment-gpXXXX-plan.md`)

### 7. Final Report
Confirm the operation and final counts. Write to an artifact file and 'notify_user'.