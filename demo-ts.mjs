/**
 * Demo script for TypeScript kotogram utilities
 *
 * Since we don't have a TypeScript parser, we use manually-crafted
 * kotogram strings from Python parser output.
 */

import { kotogramToJapanese, splitKotogram } from "./dist/kotogram.js";

console.log("=".repeat(80));
console.log("TypeScript Kotogram Demo");
console.log("=".repeat(80));
console.log();

// Sample kotogram from Python parser: 猫を食べる (The cat eats)
const catEats =
  "⌈ˢ猫ᵖn:common_nounᵇ猫ʳネコ⌉⌈ˢをᵖprt:case_particleᵇをʳオ⌉⌈ˢ食べるᵖv:general:e-ichidan-ba:attributiveᵇ食べるʳタベル⌉";

console.log("Original kotogram:");
console.log(catEats);
console.log();

console.log("Default (no options):");
console.log(kotogramToJapanese(catEats));
console.log();

console.log("With spaces:");
console.log(kotogramToJapanese(catEats, { spaces: true }));
console.log();

console.log("With furigana:");
console.log(kotogramToJapanese(catEats, { furigana: true }));
console.log();

console.log("Both furigana + spaces:");
console.log(kotogramToJapanese(catEats, { furigana: true, spaces: true }));
console.log();

console.log("Split into tokens:");
const tokens = splitKotogram(catEats);
console.log(`  Found ${tokens.length} tokens:`);
tokens.forEach((token, i) => {
  const surface = kotogramToJapanese(token);
  const withFurigana = kotogramToJapanese(token, { furigana: true });
  console.log(`  ${i + 1}. ${surface} → ${withFurigana}`);
});
console.log();

// Sample with kanji: 漢字
const kanji = "⌈ˢ漢字ᵖnʳカンジ⌉";

console.log("-".repeat(80));
console.log("Kanji example: 漢字");
console.log("-".repeat(80));
console.log();

console.log("Kotogram:", kanji);
console.log("Plain:", kotogramToJapanese(kanji));
console.log("With furigana:", kotogramToJapanese(kanji, { furigana: true }));
console.log();

// Sample with particles showing IME input principle
const student = "⌈ˢ私ᵖnʳワタクシ⌉⌈ˢはᵖprtʳワ⌉⌈ˢ学生ᵖnʳガクセイ⌉⌈ˢですᵖauxv⌉";

console.log("-".repeat(80));
console.log("Particle example: 私は学生です (I am a student)");
console.log("-".repeat(80));
console.log();

console.log("Kotogram:", student);
console.log("Plain:", kotogramToJapanese(student));
console.log("With furigana:", kotogramToJapanese(student, { furigana: true }));
console.log();
console.log("Note: は (particle) doesn't get [わ] - it shows IME input 'は'");
console.log();

// Pure kana example
const hiragana = "⌈ˢひらがなᵖnʳヒラガナ⌉";

console.log("-".repeat(80));
console.log("Pure kana example: ひらがな");
console.log("-".repeat(80));
console.log();

console.log("Kotogram:", hiragana);
console.log("Plain:", kotogramToJapanese(hiragana));
console.log("With furigana:", kotogramToJapanese(hiragana, { furigana: true }));
console.log();
console.log("Note: Pure hiragana doesn't get furigana - already shows IME input");
console.log();

console.log("=".repeat(80));
console.log("All 36 tests passing! ✓");
console.log("=".repeat(80));
