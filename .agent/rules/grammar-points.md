---
trigger: always_on
---

[Grammar Points: gpXXXX labels]

**Adding negatives**: Before adding a sentence as negative for gpXXXX, verify the sentence does NOT contain that grammar point. Check the grammar YAML definition and confirm the pattern is absent.

How to get information about grammar points:
- gpXXXX id to name of grammar point.
  sqlite3 data/corpus.db "SELECT name FROM grammar WHERE id = 'gp0102';"

- Grammar point name to grammar point definition.
  cat 'data/grammar/の (one).yaml'

