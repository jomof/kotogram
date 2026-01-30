---
trigger: always_on
---

[Using 'curate' to modify the training corpus]

First call 'source requirements.sh' to set up venv.

Run 'scripts/curate upsert' to modify the data/corpus.db. Use 'scripts/curate upsert --help' for details.

Recipes:
- Remove a label from all sentences (positive and negative).
  curate upsert --select-sentences='!gpXXXX' --grammar='!gpXXXX'
  Note: You must use '=' between parameter and parameter value.

- Add a positive label to sentences in a file.
  curate upsert --sentences='path/to/file.txt' --grammar='+gpXXXX'
  Note: The file must be one Japanese sentence alone per line.

- Add a negative label to a single sentence.
  curate upsert '一体何をしたいの？' --grammar='-gpXXXX'

- Insert sentence into corpus.
  curate upsert '先生の本だぜ' --gender='masculine' --allow-insert

- Mark a sentences as ungrammatic.
  curate upsert 'せえへんわ、そなん' --grammatic=0
  Note: This is destructive. It will remove fields like gender, formality, grammar, etc.

General:
- There are three ways to specify sentences to operate on:
  --select-sentences <query>: Query the corpus.
  --sentences <file>: Specify sentences in a file.
  Raw, single sentence passed at the command-line.
