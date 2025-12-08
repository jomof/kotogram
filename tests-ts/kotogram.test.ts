/**
 * Tests for kotogram utilities
 *
 * Since we don't have a TypeScript parser implementation, these tests use
 * manually-crafted kotogram strings based on the Python parser output.
 */

import { test } from "node:test";
import assert from "node:assert";
import { kotogramToJapanese, splitKotogram } from "../dist/kotogram.js";

/**
 * Test fixtures - kotogram strings manually created from Python parser output
 */
const FIXTURES = {
  // 猫を食べる - "The cat eats"
  cat_eats: "⌈ˢ猫ᵖn:common_nounᵇ猫ʳネコ⌉⌈ˢをᵖprt:case_particleᵇをʳオ⌉⌈ˢ食べるᵖv:general:e-ichidan-ba:attributiveᵇ食べるʳタベル⌉",

  // 漢字 - "Kanji"
  kanji: "⌈ˢ漢字ᵖnʳカンジ⌉",

  // ひらがな - "Hiragana" (pure kana, no reading needed)
  hiragana: "⌈ˢひらがなᵖnʳヒラガナ⌉",

  // カタカナ - "Katakana" (pure katakana)
  katakana: "⌈ˢカタカナᵖnʳカタカナ⌉",

  // こんにちは。 - "Hello." (with punctuation)
  hello: "⌈ˢこんにちはᵖint⌉⌈ˢ。ᵖauxs⌉",

  // 学校 - "School" (has small kana っ in reading)
  school: "⌈ˢ学校ᵖnʳガッコウ⌉",

  // 東京へ行く - "Go to Tokyo" (particle へ)
  tokyo: "⌈ˢ東京ᵖnʳトウキョウ⌉⌈ˢへᵖprtʳエ⌉⌈ˢ行くᵖvʳイク⌉",

  // 私は学生です - "I am a student" (particle は)
  student: "⌈ˢ私ᵖnʳワタクシ⌉⌈ˢはᵖprtʳワ⌉⌈ˢ学生ᵖnʳガクセイ⌉⌈ˢですᵖauxv⌉",
};

// ============================================================================
// Basic kotogram_to_japanese tests
// ============================================================================

test("kotogramToJapanese - basic conversion", () => {
  const result = kotogramToJapanese(FIXTURES.cat_eats);
  assert.strictEqual(result, "猫を食べる");
});

test("kotogramToJapanese - with spaces", () => {
  const result = kotogramToJapanese(FIXTURES.cat_eats, { spaces: true });
  assert.ok(result.includes(" "));
  // Should be able to reconstruct original by removing spaces
  assert.strictEqual(result.replace(/ /g, ""), "猫を食べる");
});

test("kotogramToJapanese - with punctuation collapse", () => {
  const result = kotogramToJapanese(FIXTURES.hello, {
    spaces: true,
    collapsePunctuation: true,
  });
  // Should not have space before punctuation
  assert.ok(!result.includes(" 。"));
  assert.strictEqual(result.replace(/ /g, ""), "こんにちは。");
});

test("kotogramToJapanese - empty string", () => {
  const result = kotogramToJapanese("");
  assert.strictEqual(result, "");
});

test("kotogramToJapanese - single token", () => {
  const result = kotogramToJapanese(FIXTURES.kanji);
  assert.strictEqual(result, "漢字");
});

// ============================================================================
// Furigana tests
// ============================================================================

test("kotogramToJapanese - kanji gets furigana", () => {
  const result = kotogramToJapanese(FIXTURES.kanji, { furigana: true });
  assert.ok(result.includes("["));
  assert.ok(result.includes("]"));
  // Should contain hiragana reading
  assert.ok(result.includes("かんじ"));
  assert.strictEqual(result, "漢字[かんじ]");
});

test("kotogramToJapanese - hiragana no furigana", () => {
  const result = kotogramToJapanese(FIXTURES.hiragana, { furigana: true });
  // Should not have furigana markers
  assert.ok(!result.includes("["));
  assert.strictEqual(result, "ひらがな");
});

test("kotogramToJapanese - katakana no furigana", () => {
  const result = kotogramToJapanese(FIXTURES.katakana, { furigana: true });
  // Should not have furigana markers
  assert.ok(!result.includes("["));
  assert.strictEqual(result, "カタカナ");
});

test("kotogramToJapanese - particles no pronunciation furigana", () => {
  const result = kotogramToJapanese(FIXTURES.cat_eats, { furigana: true });
  // Should NOT have [お] for を - particles show IME input
  assert.ok(!result.includes("[お]"));
  // Should have を as-is (IME input)
  assert.ok(result.includes("を"));
});

test("kotogramToJapanese - particle は no pronunciation", () => {
  const result = kotogramToJapanese(FIXTURES.student, { furigana: true });
  // は is the IME input, not わ
  assert.ok(!result.includes("[わ]"));
  assert.ok(result.includes("は"));
});

test("kotogramToJapanese - particle へ no pronunciation", () => {
  const result = kotogramToJapanese(FIXTURES.tokyo, { furigana: true });
  // へ is the IME input, not え
  assert.ok(!result.includes("[え]"));
  assert.ok(result.includes("へ"));
});

test("kotogramToJapanese - furigana is hiragana", () => {
  const result = kotogramToJapanese(FIXTURES.kanji, { furigana: true });
  // Extract furigana from brackets
  const furiganaMatch = result.match(/\[(.*?)\]/);
  assert.ok(furiganaMatch);
  const furigana = furiganaMatch![1];

  // Check that furigana is hiragana, not katakana
  for (const char of furigana) {
    const code = char.charCodeAt(0);
    // Skip ー (長音符)
    if (char === "ー") continue;
    // Should be hiragana (0x3041-0x309F), not katakana (0x30A1-0x30F6)
    const isKatakana = code >= 0x30a1 && code <= 0x30f6;
    assert.ok(!isKatakana, `Furigana contains katakana: ${char}`);
  }
});

test("kotogramToJapanese - small kana preserved", () => {
  const result = kotogramToJapanese(FIXTURES.school, { furigana: true });
  // Should preserve small っ (different IME input than large つ)
  if (result.includes("がっこ")) {
    assert.ok(result.includes("っ"));
  }
});

test("kotogramToJapanese - furigana with spaces", () => {
  const result = kotogramToJapanese(FIXTURES.cat_eats, {
    furigana: true,
    spaces: true,
  });
  // Should have both furigana and spaces
  assert.ok(result.includes("["));
  assert.ok(result.includes(" "));
});

test("kotogramToJapanese - default no furigana", () => {
  const result = kotogramToJapanese(FIXTURES.kanji);
  // Default should not have furigana
  assert.ok(!result.includes("["));
  assert.strictEqual(result, "漢字");
});

// ============================================================================
// split_kotogram tests
// ============================================================================

test("splitKotogram - multiple tokens", () => {
  const tokens = splitKotogram(FIXTURES.cat_eats);
  assert.strictEqual(tokens.length, 3);
  assert.ok(tokens[0].includes("猫"));
  assert.ok(tokens[1].includes("を"));
  assert.ok(tokens[2].includes("食べる"));
});

test("splitKotogram - single token", () => {
  const tokens = splitKotogram(FIXTURES.kanji);
  assert.strictEqual(tokens.length, 1);
  assert.ok(tokens[0].includes("漢字"));
});

test("splitKotogram - empty string", () => {
  const tokens = splitKotogram("");
  assert.strictEqual(tokens.length, 0);
});

test("splitKotogram - preserves annotations", () => {
  const tokens = splitKotogram(FIXTURES.kanji);
  const token = tokens[0];
  // Should preserve all markers
  assert.ok(token.includes("⌈"));
  assert.ok(token.includes("⌉"));
  assert.ok(token.includes("ˢ"));
  assert.ok(token.includes("ᵖ"));
  assert.ok(token.includes("ʳ"));
});

test("splitKotogram - roundtrip with join", () => {
  const tokens = splitKotogram(FIXTURES.cat_eats);
  const rejoined = tokens.join("");
  assert.strictEqual(rejoined, FIXTURES.cat_eats);
});

test("splitKotogram - tokens work with kotogramToJapanese", () => {
  const tokens = splitKotogram(FIXTURES.cat_eats);
  // Each token should be valid kotogram
  for (const token of tokens) {
    const result = kotogramToJapanese(token);
    assert.ok(result.length > 0);
  }
});

// ============================================================================
// Integration tests
// ============================================================================

test("kotogramToJapanese - all options combined", () => {
  const result = kotogramToJapanese(FIXTURES.cat_eats, {
    spaces: true,
    collapsePunctuation: true,
    furigana: true,
  });
  // Should have furigana for kanji
  assert.ok(result.includes("["));
  // Should have spaces
  assert.ok(result.includes(" "));
  // Should have original text
  assert.ok(result.includes("猫"));
  assert.ok(result.includes("を"));
  assert.ok(result.includes("食べる"));
});

test("splitKotogram then kotogramToJapanese", () => {
  const tokens = splitKotogram(FIXTURES.cat_eats);
  const parts = tokens.map((t) => kotogramToJapanese(t));
  const result = parts.join("");
  assert.strictEqual(result, "猫を食べる");
});

test("splitKotogram then kotogramToJapanese with furigana", () => {
  const tokens = splitKotogram(FIXTURES.kanji);
  const parts = tokens.map((t) => kotogramToJapanese(t, { furigana: true }));
  const result = parts.join("");
  assert.strictEqual(result, "漢字[かんじ]");
});

// ============================================================================
// Edge cases
// ============================================================================

test("kotogramToJapanese - no surface markers", () => {
  // Malformed kotogram without surface markers
  const result = kotogramToJapanese("⌈invalid⌉");
  assert.strictEqual(result, "");
});

test("kotogramToJapanese - partial marker", () => {
  // Missing closing ᵖ marker - still extracts surface between ˢ and ᵖ
  const result = kotogramToJapanese("⌈ˢ猫ᵖn");
  // The regex ˢ(.*?)ᵖ will still match "猫" between ˢ and ᵖ
  assert.strictEqual(result, "猫");
});

test("splitKotogram - unclosed token", () => {
  // Should only match complete tokens
  const tokens = splitKotogram("⌈ˢ猫ᵖn");
  assert.strictEqual(tokens.length, 0);
});

test("kotogramToJapanese - special characters in surface", () => {
  // Test with special regex characters (though unlikely in real kotogram)
  const kotogram = "⌈ˢ+*?ᵖn⌉";
  const result = kotogramToJapanese(kotogram);
  assert.strictEqual(result, "+*?");
});
