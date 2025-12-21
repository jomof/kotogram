/**
 * Tests for extractTokenFeatures function
 *
 * These tests use manually-crafted kotogram strings based on the Python parser output
 * to verify that the TypeScript implementation matches the Python behavior.
 */

import { test } from "node:test";
import assert from "node:assert";
import { extractTokenFeatures, splitKotogram } from "../dist/kotogram.js";

/**
 * Test fixtures - kotogram strings from Python parser
 */
const FIXTURES = {
    // 食べる verb token
    verb_taberu:
        "⌈ˢ食べるᵖverb:general:lower-ichidan-ba:terminalᵇ食べるʳタベル⌉",

    // 食べ continuative form
    verb_tabe:
        "⌈ˢ食べᵖverb:general:lower-ichidan-ba:continuativeᵇ食べるᵈ食べるʳタベ⌉",

    // ます auxiliary verb
    aux_masu: "⌈ˢますᵖaux-verb:aux-masu:terminalᵇますʳマス⌉",

    // です polite copula
    aux_desu: "⌈ˢですᵖaux-verb:aux-desu:terminalᵇですʳデス⌉",

    // だ plain copula
    aux_da: "⌈ˢだᵖaux-verb:aux-da:terminalᵇだʳダ⌉",

    // は binding particle
    particle_wa: "⌈ˢはᵖparticle:binding-particleᵇはʳワ⌉",

    // を case particle
    particle_wo: "⌈ˢをᵖparticle:case-particleᵇをʳオ⌉",

    // 学生 common noun
    noun_gakusei: "⌈ˢ学生ᵖnoun:common-noun:generalᵇ学生ʳガクセイ⌉",

    // 高い i-adjective
    adj_takai: "⌈ˢ高いᵖadj:general:i-adjective:terminalᵇ高いʳタカイ⌉",

    // テスト noun (simple)
    noun_test: "⌈ˢテストᵖnoun:common-noun:generalᵇテストʳテスト⌉",

    // Sentence: 食べます (2 tokens)
    sentence_tabemasu:
        "⌈ˢ食べᵖverb:general:lower-ichidan-ba:continuativeᵇ食べるᵈ食べるʳタベ⌉⌈ˢますᵖaux-verb:aux-masu:terminalᵇますʳマス⌉",
};

// ============================================================================
// Verb extraction tests
// ============================================================================

test("extractTokenFeatures - verb basic extraction", () => {
    const features = extractTokenFeatures(FIXTURES.verb_taberu);

    assert.strictEqual(features.surface, "食べる");
    assert.strictEqual(features.pos, "verb");
    assert.strictEqual(features.posDetail1, "general");
    assert.strictEqual(features.conjugatedType, "lower-ichidan-ba");
    assert.strictEqual(features.conjugatedForm, "terminal");
    assert.strictEqual(features.baseOrth, "食べる");
    assert.strictEqual(features.reading, "タベル");
});

test("extractTokenFeatures - verb continuative form", () => {
    const features = extractTokenFeatures(FIXTURES.verb_tabe);

    assert.strictEqual(features.surface, "食べ");
    assert.strictEqual(features.pos, "verb");
    assert.strictEqual(features.conjugatedForm, "continuative");
    assert.strictEqual(features.lemma, "食べる");
});

// ============================================================================
// Auxiliary verb tests
// ============================================================================

test("extractTokenFeatures - aux-masu", () => {
    const features = extractTokenFeatures(FIXTURES.aux_masu);

    assert.strictEqual(features.surface, "ます");
    assert.strictEqual(features.pos, "aux-verb");
    assert.strictEqual(features.conjugatedType, "aux-masu");
    assert.strictEqual(features.conjugatedForm, "terminal");
});

test("extractTokenFeatures - aux-desu", () => {
    const features = extractTokenFeatures(FIXTURES.aux_desu);

    assert.strictEqual(features.surface, "です");
    assert.strictEqual(features.pos, "aux-verb");
    assert.strictEqual(features.conjugatedType, "aux-desu");
    assert.strictEqual(features.conjugatedForm, "terminal");
});

test("extractTokenFeatures - aux-da", () => {
    const features = extractTokenFeatures(FIXTURES.aux_da);

    assert.strictEqual(features.surface, "だ");
    assert.strictEqual(features.pos, "aux-verb");
    assert.strictEqual(features.conjugatedType, "aux-da");
    assert.strictEqual(features.conjugatedForm, "terminal");
});

// ============================================================================
// Particle tests
// ============================================================================

test("extractTokenFeatures - particle wa", () => {
    const features = extractTokenFeatures(FIXTURES.particle_wa);

    assert.strictEqual(features.surface, "は");
    assert.strictEqual(features.pos, "particle");
    assert.strictEqual(features.posDetail1, "binding-particle");
});

test("extractTokenFeatures - particle wo", () => {
    const features = extractTokenFeatures(FIXTURES.particle_wo);

    assert.strictEqual(features.surface, "を");
    assert.strictEqual(features.pos, "particle");
    assert.strictEqual(features.posDetail1, "case-particle");
});

// ============================================================================
// Noun tests
// ============================================================================

test("extractTokenFeatures - noun common", () => {
    const features = extractTokenFeatures(FIXTURES.noun_gakusei);

    assert.strictEqual(features.surface, "学生");
    assert.strictEqual(features.pos, "noun");
    assert.strictEqual(features.posDetail1, "common-noun");
    assert.strictEqual(features.posDetail2, "general");
    // Nouns don't have conjugation
    assert.strictEqual(features.conjugatedType, "");
    assert.strictEqual(features.conjugatedForm, "");
});

test("extractTokenFeatures - noun simple", () => {
    const features = extractTokenFeatures(FIXTURES.noun_test);

    assert.strictEqual(features.surface, "テスト");
    assert.strictEqual(features.pos, "noun");
    assert.strictEqual(features.baseOrth, "テスト");
});

// ============================================================================
// Adjective tests
// ============================================================================

test("extractTokenFeatures - i-adjective", () => {
    const features = extractTokenFeatures(FIXTURES.adj_takai);

    assert.strictEqual(features.surface, "高い");
    assert.strictEqual(features.pos, "adj");
    assert.strictEqual(features.posDetail1, "general");
    assert.strictEqual(features.conjugatedType, "i-adjective");
});

// ============================================================================
// Edge cases
// ============================================================================

test("extractTokenFeatures - empty token", () => {
    const features = extractTokenFeatures("");

    assert.strictEqual(features.surface, "");
    assert.strictEqual(features.pos, "");
    assert.strictEqual(features.posDetail1, "");
    assert.strictEqual(features.conjugatedType, "");
});

test("extractTokenFeatures - malformed token no markers", () => {
    const features = extractTokenFeatures("テスト");

    assert.strictEqual(features.surface, "");
    assert.strictEqual(features.pos, "");
});

test("extractTokenFeatures - token without boundaries", () => {
    // Token content but no ⌈⌉ boundaries
    const features = extractTokenFeatures("ˢ猫ᵖnoun");

    assert.strictEqual(features.surface, "猫");
    assert.strictEqual(features.pos, "noun");
});

// ============================================================================
// Integration with splitKotogram
// ============================================================================

test("extractTokenFeatures - works with splitKotogram", () => {
    const tokens = splitKotogram(FIXTURES.sentence_tabemasu);
    assert.strictEqual(tokens.length, 2);

    const features1 = extractTokenFeatures(tokens[0]);
    assert.strictEqual(features1.surface, "食べ");
    assert.strictEqual(features1.pos, "verb");

    const features2 = extractTokenFeatures(tokens[1]);
    assert.strictEqual(features2.surface, "ます");
    assert.strictEqual(features2.pos, "aux-verb");
    assert.strictEqual(features2.conjugatedType, "aux-masu");
});

// ============================================================================
// Default values
// ============================================================================

test("extractTokenFeatures - all fields default to empty string", () => {
    const features = extractTokenFeatures("");

    // All fields should be empty strings, not undefined
    assert.strictEqual(features.surface, "");
    assert.strictEqual(features.pos, "");
    assert.strictEqual(features.posDetail1, "");
    assert.strictEqual(features.posDetail2, "");
    assert.strictEqual(features.posDetail3, "");
    assert.strictEqual(features.conjugatedType, "");
    assert.strictEqual(features.conjugatedForm, "");
    assert.strictEqual(features.baseOrth, "");
    assert.strictEqual(features.lemma, "");
    assert.strictEqual(features.reading, "");
});
