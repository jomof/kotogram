---
trigger: always_on
---

[How to run unittests]


To run tests:
- 'source requirements.sh'
- 'test.sh'

Check code hygiene separately: 'test.sh --hygiene'

Run 'test.sh --help' to see how to run individual tests.

The test driver files are locked because you prefer to disable things over writing correct, well-factored code. If you think you need to modify test.sh or test_runner.py, think about it, and construct a plan for why that change would lead to a healthier codebase. Present the plan to me and I'll unlock those files temporarily. However, I encourage you to write better-factored code.

Vulture is known to be configured correctly. If you think you need to bypass vulture for some reason, then present a plan for why it's necessary.