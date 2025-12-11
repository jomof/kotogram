# Claude Code Goal: Improve Grammaticality Detection

## Current Status
- **Detection rate**: 85.40% (125,047/146,427)
- **False positive rate**: 0%
- **Tests**: 187 passing
- **Patterns**: 43 implemented

## Quick Start: Single Iteration

```bash
# 1. Find missed sentences (look for patterns in the output)
python scripts/analyze_missed.py --limit 20

# 2. Analyze a specific sentence to understand token structure
python scripts/analyze_missed.py --analyze "問題の文"

# 3. Filter by surface pattern to find related errors
python scripts/analyze_missed.py --surface "でいる"

# 4. Test your pattern BEFORE implementing (REQUIRED - must have 0 FP)
python scripts/check_pattern.py "curr.get('pos')=='v' and nxt.get('surface')=='Y'"

# 5. Use --analyze for detailed token breakdown of matches
python scripts/check_pattern.py "PATTERN" --analyze

# 6. If safe (0 false positives): add test + implement + verify
python -m pytest tests-py/test_grammaticality.py -v
```

## Pattern Discovery Workflow

### Method 1: Manual Analysis (Most Reliable)
```bash
# Browse missed sentences
python scripts/analyze_missed.py --limit 50

# Look for patterns like:
# - Wrong particle (で vs て)
# - Missing conjugation
# - Wrong form (terminal vs conjunctive)

# When you spot a pattern, analyze the sentence:
python scripts/analyze_missed.py --analyze "その文"
```

### Method 2: Surface Pattern Search
```bash
# Search for specific error patterns
python scripts/analyze_missed.py --surface "ない"    # Negation errors
python scripts/analyze_missed.py --surface "だ"      # Copula errors
python scripts/analyze_missed.py --surface "て"      # Te-form errors
```

### Method 3: Automated Suggestion
```bash
# Find high-frequency patterns in missed sentences
python scripts/suggest_patterns.py --min-detections 30

# Options:
#   --min-detections N   Minimum detections to consider (default: 30)
#   --max-suggestions N  Maximum patterns to suggest (default: 10)
#   --verify-fp N        Number of grammatical sentences to check for FP (default: 20000)

# Example output shows pattern code ready to implement:
#   ✓ 552 detections | v[terminal] + auxv(auxv-masu)
#   Code: curr.get('pos')=='v' and curr.get('conjugated_form')=='terminal' ...

# Note: Many suggested patterns are too broad - always verify with check_pattern.py
```

## Pattern Validation (REQUIRED)

Always test patterns before implementing:

```bash
# Basic check
python scripts/check_pattern.py "curr.get('pos')=='adj' and nxt.get('surface')=='ない'"

# Detailed analysis of matches
python scripts/check_pattern.py "PATTERN" --analyze --show 10

# Check more grammatical sentences for FP
python scripts/check_pattern.py "PATTERN" --fp-limit 50000
```

Available variables in patterns:
- `curr` - current token features
- `nxt` - next token features
- `nxt2` - token after next
- `prev` - previous token

## Pattern Implementation Template

```python
# In analysis.py, add detection function:
def _is_PATTERN_NAME(curr: Dict[str, str], nxt: Dict[str, str]) -> bool:
    """Detect DESCRIPTION. Example: XXX (should be YYY)."""
    if curr.get('KEY') != 'VALUE':
        return False
    if nxt.get('KEY') != 'VALUE':
        return False
    return True

# Add to _rule_based_grammaticality():
if _is_PATTERN_NAME(curr, nxt):
    return False
```

## Key Token Features

| Feature | Description | Examples |
|---------|-------------|----------|
| `pos` | Part of speech | `v`, `n`, `adj`, `auxv`, `prt`, `shp`, `adv`, `pron`, `auxs` |
| `conjugated_type` | Verb/adj class | `godan-ka`, `ichidan`, `sa-irregular`, `auxv-ta`, `auxv-da`, `ka-irregular`, `i-ichidan-ka` |
| `conjugated_form` | Conjugation form | `terminal`, `conjunctive`, `attributive`, `imperfective`, `conjunctive-geminate`, `conjunctive-nasal` |
| `surface` | Actual text | Use sparingly, only for particles/function words |

## Common Error Patterns (by category)

### Te-form errors (て/で confusion)
- Godan nasal verbs need で: 死んて→死んで, 読んて→読んで
- Godan sa-row verbs need て: 話しで→話して
- Missing っ: かかて→かかって, 向かて→向かって
- ka-irregular 来る + で: 来でくれる→来てくれる

### Copula/auxiliary errors
- i-adj + だ: 美しいだ (wrong)
- た + だ: なかっただ (wrong, but た+だろう is OK)
- ます + ない: ますない→ません
- ました + こと: しましたことがない→したことがない (style mixing)

### Negation errors
- i-adj terminal + ない: 高いない→高くない
- na-adj + ない: 好きない→好きではない

### Attributive errors
- i-adj conjunctive + な: 早くな出発→早い出発/早く出発

### Doubled elements
- Particles: がが, をを, にに, でで, とと
- Past: たた, た+たこ

### Question/quotation errors
- godan-ka conjunctive + か: 行きか→行くか
- と + い: といて→といって

### Missing て errors
- verb conjunctive + できる: 忘れできた→忘れてきた

## Files

| File | Purpose |
|------|---------|
| `kotogram/analysis.py` | `_rule_based_grammaticality()` - main detection logic |
| `tests-py/test_grammaticality.py` | `TestAgrammaticTruePositives` class |
| `scripts/analyze_missed.py` | Find and analyze missed sentences |
| `scripts/check_pattern.py` | Validate pattern before implementing |
| `scripts/suggest_patterns.py` | Auto-suggest patterns (use with caution) |

## Rules

1. **Test pattern with check_pattern.py BEFORE implementing** - must have 0 false positives
2. **Use POS/conjugation features, not surface forms** (except particles)
3. **Write failing test FIRST**, then implement
4. **One pattern at a time** - verify each before moving to next
5. **Update this status section** after each iteration

## Troubleshooting

### Pattern has false positives
- Make the pattern more specific (add more conditions)
- Check if there's a grammatical exception (e.g., た+だろう is OK)
- Use `--analyze` to understand why FP sentences match

### Can't find the error in token analysis
- The parser may analyze the error differently than expected
- Try different test sentences with the same error
- Look at adjacent tokens for context

### Pattern doesn't match expected sentences
- Check conjugation forms carefully (terminal vs conjunctive)
- Verify the POS tag is correct
- Use `--analyze` to see actual token features
