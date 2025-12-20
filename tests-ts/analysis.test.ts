import { test } from "node:test";
import assert from "node:assert";
import {
    GrammarAnalysis,
    FormalityLevel,
    GenderLevel,
    RegisterLevel
} from "../dist/analysis.js";

test("GrammarAnalysis serialization roundtrip", () => {
    const analysis = new GrammarAnalysis(
        "dummy_kotogram",
        FormalityLevel.FORMAL,
        0.9,
        true,
        GenderLevel.NEUTRAL,
        0.1,
        true,
        new Set([RegisterLevel.KANSAIBEN, RegisterLevel.SONKEIGO]),
        new Map([
            [RegisterLevel.KANSAIBEN, 0.8],
            [RegisterLevel.SONKEIGO, 0.7],
            [RegisterLevel.NEUTRAL, 0.2]
        ]),
        true,
        0.95
    );

    // To JSON
    const jsonStr = analysis.toJson();
    const data = JSON.parse(jsonStr);

    // Verify JSON content (raw keys matching GrammarAnalysisData)
    assert.strictEqual(data.formality, "formal");
    assert.strictEqual(data.gender, "neutral");
    assert.deepStrictEqual(data.registers.sort(), ["kansaiben", "sonkeigo"]);
    assert.strictEqual(data.register_scores.kansaiben, 0.8);
    assert.strictEqual(data.register_scores.neutral, 0.2);

    // From JSON
    const restored = GrammarAnalysis.fromJson(jsonStr);

    // Verify restored object
    assert.strictEqual(restored.kotogram, analysis.kotogram);
    assert.strictEqual(restored.formality, analysis.formality);
    assert.strictEqual(restored.gender, analysis.gender);
    assert.deepStrictEqual(restored.registers, analysis.registers);
    assert.strictEqual(restored.register_scores.get(RegisterLevel.KANSAIBEN), 0.8);
    assert.strictEqual(restored.register_scores.get(RegisterLevel.NEUTRAL), 0.2);
    assert.strictEqual(restored.is_grammatic, analysis.is_grammatic);
    assert.strictEqual(restored.grammaticality_score, analysis.grammaticality_score);
});
