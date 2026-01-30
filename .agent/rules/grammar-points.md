---
trigger: always_on
---

[Grammar Points: gpXXXX labels]

How do get information about grammar points:
- gpXXXX id to name of grammar point.
  sqlite3 data/corpus.db "SELECT name FROM grammar WHERE id = 'gp0102';"

- Grammar point name to grammar point definition.
  cat 'data/grammar/の (one).yaml'

