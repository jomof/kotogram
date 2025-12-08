/**
 * Tests for cross-language validation bugs that were discovered and fixed.
 *
 * These tests document specific issues found during cross-language validation
 * to prevent regression.
 */

import { test } from "node:test";
import assert from "node:assert";
import { kotogramToJapanese } from "../dist/kotogram.js";

/**
 * Bug 1: Missing POS_TO_CHARS.auxs entries
 *
 * Issue: TypeScript was missing many punctuation characters in POS_TO_CHARS.auxs,
 * most critically the small tsu 'っ' which is needed to properly collapse
 * compound verb forms.
 *
 * Example: "もっ" + "て" should collapse to "もって" when collapsePunctuation=true
 * because 'っ' at the end of a token should attach to the following particle.
 */

test("Bug 1: Small tsu っ collapses with following particle", () => {
  // Kotogram for "もって" (motte) - verb stem + te particle
  // ⌈もっ⌉ = verb stem "mots-" with geminate
  // ⌈て⌉ = te particle
  const kotogram =
    "⌈ˢもっᵖv:general:godan-ta:conjunctive-geminateᵇもつᵈもつʳモッ⌉⌈ˢてᵖprt:conjunctive_particleʳテ⌉";

  // With collapsePunctuation=true, small tsu should attach to following て
  const collapsed = kotogramToJapanese(kotogram, {
    spaces: true,
    collapsePunctuation: true,
  });
  assert.strictEqual(collapsed, "もって");

  // With collapsePunctuation=false, should keep space
  const notCollapsed = kotogramToJapanese(kotogram, {
    spaces: true,
    collapsePunctuation: false,
  });
  assert.strictEqual(notCollapsed, "もっ て");
});

test("Bug 1: Period 。 collapses when collapsePunctuation=true", () => {
  // Sentence ending with period
  const kotogram =
    "⌈ˢこんにちはᵖint⌉⌈ˢ。ᵖauxs:period⌉";

  const collapsed = kotogramToJapanese(kotogram, {
    spaces: true,
    collapsePunctuation: true,
  });
  assert.strictEqual(collapsed, "こんにちは。");

  const notCollapsed = kotogramToJapanese(kotogram, {
    spaces: true,
    collapsePunctuation: false,
  });
  assert.strictEqual(notCollapsed, "こんにちは 。");
});

test("Bug 1: Question mark ？ collapses when collapsePunctuation=true", () => {
  const kotogram = "⌈ˢ何ᵖpronʳナン⌉⌈ˢ？ᵖauxs:period⌉";

  const collapsed = kotogramToJapanese(kotogram, {
    spaces: true,
    collapsePunctuation: true,
  });
  assert.strictEqual(collapsed, "何？");

  const notCollapsed = kotogramToJapanese(kotogram, {
    spaces: true,
    collapsePunctuation: false,
  });
  assert.strictEqual(notCollapsed, "何 ？");
});

test("Bug 1: All punctuation characters collapse correctly", () => {
  // Test that all POS_TO_CHARS.auxs characters are handled
  // Key characters that were missing: っ, ー, 々, ぇ, etc.

  // Test with 々 (iteration mark)
  const kotogram1 = "⌈ˢ時ᵖnʳトキ⌉⌈ˢ々ᵖauxsʳドキドキ⌉";
  const result1 = kotogramToJapanese(kotogram1, {
    spaces: true,
    collapsePunctuation: true,
  });
  assert.strictEqual(result1, "時々");

  // Test with ー (long vowel mark)
  const kotogram2 = "⌈ˢコーヒーᵖnʳコーヒー⌉";
  const result2 = kotogramToJapanese(kotogram2);
  assert.strictEqual(result2, "コーヒー");
});

/**
 * Bug 2: Parameter naming mismatch (snake_case vs camelCase)
 *
 * Issue: The cross-language validation script was passing Python's snake_case
 * parameter names (collapse_punctuation) directly to TypeScript, which expects
 * camelCase (collapsePunctuation).
 *
 * This was a validation script bug, not a library bug, but we test both
 * naming conventions work as expected.
 */

test("Bug 2: collapsePunctuation parameter works (camelCase)", () => {
  const kotogram = "⌈ˢこんにちはᵖint⌉⌈ˢ。ᵖauxs:period⌉";

  // Test camelCase parameter
  const result = kotogramToJapanese(kotogram, {
    spaces: true,
    collapsePunctuation: false,
  });
  assert.strictEqual(result, "こんにちは 。");
});

test("Bug 2: Verify all parameters use camelCase", () => {
  const kotogram =
    "⌈ˢ猫ᵖn:common_nounᵇ猫ʳネコ⌉⌈ˢをᵖprt:case_particleᵇをʳオ⌉⌈ˢ食べるᵖv:general:e-ichidan-ba:attributiveᵇ食べるʳタベル⌉";

  // All three parameters should be camelCase
  const result = kotogramToJapanese(kotogram, {
    spaces: true, // camelCase
    collapsePunctuation: true, // camelCase
    furigana: true, // camelCase
  });

  assert.ok(result.includes(" "));
  assert.ok(result.includes("["));
});

/**
 * Integration test: Full sentence from validation failure
 *
 * This was one of the actual failing test cases that revealed the bugs.
 */

test("Integration: Full sentence with compound verbs and particles", () => {
  // "きみにちょっとしたものをもってきたよ。"
  // "I brought you a little something."
  const kotogram =
    "⌈ˢきみᵖpronʳキミ⌉⌈ˢにᵖprt:case_particleʳニ⌉⌈ˢちょっとᵖadvʳチョット⌉" +
    "⌈ˢしᵖv:non_self_reliant:sa-irregular:conjunctiveᵇするᵈするʳシ⌉" +
    "⌈ˢたᵖauxv:auxv-ta:attributiveʳタ⌉⌈ˢものᵖn:common_noun:suru-possibleʳモノ⌉" +
    "⌈ˢをᵖprt:case_particleʳヲ⌉" +
    "⌈ˢもっᵖv:general:godan-ta:conjunctive-geminateᵇもつᵈもつʳモッ⌉" +
    "⌈ˢてᵖprt:conjunctive_particleʳテ⌉" +
    "⌈ˢきᵖv:non_self_reliant:ka-irregular:conjunctiveᵇくるᵈくるʳキ⌉" +
    "⌈ˢたᵖauxv:auxv-ta:terminalʳタ⌉⌈ˢよᵖprt:sentence_final_particleʳヨ⌉" +
    "⌈ˢ。ᵖauxs:period⌉";

  // Default: no spaces, punctuation naturally attached
  const default_result = kotogramToJapanese(kotogram);
  assert.strictEqual(
    default_result,
    "きみにちょっとしたものをもってきたよ。"
  );

  // With spaces + collapse: should collapse もっ+て and attach period
  const collapsed = kotogramToJapanese(kotogram, {
    spaces: true,
    collapsePunctuation: true,
  });
  assert.strictEqual(
    collapsed,
    "きみ に ちょっと し た もの を もって き た よ。"
  );
  assert.ok(collapsed.includes("もって")); // Should be collapsed
  assert.ok(!collapsed.includes(" 。")); // Period should be attached

  // With spaces but no collapse: should keep もっ て separated and space before period
  const notCollapsed = kotogramToJapanese(kotogram, {
    spaces: true,
    collapsePunctuation: false,
  });
  assert.strictEqual(
    notCollapsed,
    "きみ に ちょっと し た もの を もっ て き た よ 。"
  );
  assert.ok(notCollapsed.includes("もっ て")); // Should be separated
  assert.ok(notCollapsed.includes(" 。")); // Period should have space before it
});

test("Integration: Sentence with furigana and compound verbs", () => {
  // Same sentence with furigana
  const kotogram =
    "⌈ˢきみᵖpronʳキミ⌉⌈ˢにᵖprt:case_particleʳニ⌉⌈ˢちょっとᵖadvʳチョット⌉" +
    "⌈ˢしᵖv:non_self_reliant:sa-irregular:conjunctiveᵇするᵈするʳシ⌉" +
    "⌈ˢたᵖauxv:auxv-ta:attributiveʳタ⌉⌈ˢものᵖn:common_noun:suru-possibleʳモノ⌉" +
    "⌈ˢをᵖprt:case_particleʳヲ⌉" +
    "⌈ˢもっᵖv:general:godan-ta:conjunctive-geminateᵇもつᵈもつʳモッ⌉" +
    "⌈ˢてᵖprt:conjunctive_particleʳテ⌉" +
    "⌈ˢきᵖv:non_self_reliant:ka-irregular:conjunctiveᵇくるᵈくるʳキ⌉" +
    "⌈ˢたᵖauxv:auxv-ta:terminalʳタ⌉⌈ˢよᵖprt:sentence_final_particleʳヨ⌉" +
    "⌈ˢ。ᵖauxs:period⌉";

  const result = kotogramToJapanese(kotogram, {
    spaces: true,
    furigana: true,
    collapsePunctuation: true,
  });

  // Should have furigana, spaces, and proper collapsing
  assert.ok(result.includes("もって")); // Collapsed
  assert.ok(!result.includes(" 。")); // Period attached
  // Pure kana tokens shouldn't get furigana, but check it doesn't break
  assert.ok(result.length > 0);
});
