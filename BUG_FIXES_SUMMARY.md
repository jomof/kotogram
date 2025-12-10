# Formality Analysis Bug Fixes Summary

## Testing Process
Analyzed sentences from `data/jpn_sentences.tsv` using Sudachi parser to find sentences flagged as UNPRAGMATIC_FORMALITY and validate against real Japanese usage.

## Bugs Found and Fixed

### Bug #1: Feminine-Formal Register (お嬢様言葉) Incorrectly Flagged
**Problem**: The function flagged sentences like "ですわ", "ですの", "ませんわ" as unpragmatic.

**Root Cause**: The analyzer didn't recognize that particles わ, の, and な are acceptable with formal forms in feminine-formal speech (お嬢様言葉 - ojou-sama kotoba).

**Examples of False Positives**:
- `もうよろしくてよ。実のない会話にはうんざりですわ。`
- `何十年も鍛え続けた強者が、ほんの一瞬の油断で弱者に倒されることがあるんですの。`
- `車？ああ・・・あのリムジンでしたら、私がチャーターした物ですわ。`

**Fix**: Updated particle checking logic to allow わ, の, and な with formal forms:
```python
# Before: Only よ and ね were acceptable
if surface not in ['よ', 'ね']:
    casual_count += 2

# After: Allow feminine-formal particles
if surface not in ['よ', 'ね', 'わ', 'の', 'な']:
    casual_count += 2
```

**Tests Added**: 6 tests in `TestFeminineFormalRegister` class
- `test_desu_wa_is_formal_*`
- `test_desu_no_is_formal_*`
- `test_masu_wa_is_formal_*`

### Bug #2: Polite Imperatives (ください/なさい) Not Recognized
**Problem**: Sentences like "連絡してくださいね" were flagged as unpragmatic.

**Root Cause**: The function only recognized です/ます as formal sentence-enders, but ください and なさい are also polite forms that establish formal register.

**Examples of False Positives**:
- `連絡してくださいね。`
- `まあまあ、そう怒らないでくださいな。`
- `ふふふ、ごめんあそばせ。気になさらないで下さいな！`

**Fix**: Added detection for polite imperative verbs:
```python
# Check for polite imperative verbs (ください, なさい)
# These are v:non_self_reliant:godan-ra:imperative
if pos == 'v' and detail3 == 'imperative':
    if surface in ['ください', 'なさい']:
        formal_count += 2
        sentence_ender_formality = 'formal'
```

**Tests Added**: 4 tests in `TestFeminineFormalRegister` class
- `test_kudasai_ne_is_formal_*`
- `test_kudasai_na_is_formal_*`

## Test Results

**Before Fixes**: 45 tests passing
**After Fixes**: 55 tests passing (10 new tests added)

All tests pass with Sudachi parser.

## Remaining Considerations

### Dialogue Detection
The TSV dataset contains dialogue entries where multiple speakers use different formality levels within a single line:
- Example: `「具合はどうなんですか？」「ああ、うん・・・少し脱水症状が出ているかな」`

This is not actually unpragmatic - it's just two different speakers. The function is designed for single-speaker analysis, which is the correct approach.

### Honorifics with Casual Particles
Some borderline cases like "いらっしゃったよ" (honorific + よ) are still being flagged. This could be:
- Acceptable in casual conversation among equals when referencing a superior
- Actually unpragmatic depending on context

The current behavior errs on the side of flagging these as potential issues, which seems reasonable.

### Compound Sentences
Sentences with multiple clauses may legitimately have different formality levels:
- Example: "一生懸命勉強しなさい、さもないと試験に落ちるぞ。"
  (Polite command followed by casual warning)

This is a design limitation - analyzing formality at the clause level would require more sophisticated parsing.

## Files Modified

1. `kotogram/analysis.py` - Core formality analysis logic
2. `tests-py/test_formality.py` - Comprehensive test suite
3. `kotogram/__init__.py` - Export formality functions
4. `FORMALITY_ANALYSIS.md` - Documentation

## Conclusion

The formality analysis function now correctly handles:
- ✅ Feminine-formal register (ですわ, ますの, etc.)
- ✅ Polite imperatives (ください, なさい)
- ✅ Standard formal forms (です, ます)
- ✅ Plain/casual forms

The function is designed for single-speaker sentence analysis and correctly identifies unpragmatic mixing within that scope.
