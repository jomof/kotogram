---
trigger: always_on
---

- Don't circumvent Vulture dead code analysis. Remove unused code.
- Don't circumvent or suppress MyPy analysis. Make code typesafe.
- Don't use Dict[str, <something>] when a data class would be more type safe.
- Don't use Optional[] just to make writing unittests. There should be a production-level reason for the use of Optional[].
- If, as you're refactoring code, you see an opportunity for increasing type safety later, but it's not central to the current refactoring, then add a comment like:
```# UNDONE(type safety): Describe the refactoring that should be done```.