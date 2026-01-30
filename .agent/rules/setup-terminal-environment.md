---
trigger: always_on
---

[Set up python venv]

In order to set up an environment to run ./train_style, bin/kotogram, scripts/curate, and one-off scripts that use this codebase, run `source ./requirements.sh` _once_ at the start of a new shell session.

That script will install all dependencies, and nothing needs to be conditionally imported.

You should NOT add conditional import logic like this:

```
# Conditional import using find_spec to avoid forbidden ImportError try-except
if importlib.util.find_spec("kotogram.japanese_parser"):
    from kotogram.japanese_parser import POS_MAP
else:
    POS_MAP = {}  # pylint: disable=invalid-name
```