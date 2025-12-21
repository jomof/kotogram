/**
 * Kotogram - A dual Python/TypeScript library for Japanese text parsing and encoding
 */

export {
  kotogramToJapanese,
  splitKotogram,
  extractTokenFeatures,
  type KotogramToJapaneseOptions,
  type TokenFeatures,
} from "./kotogram.js";

export {
  FormalityLevel,
  GenderLevel,
  RegisterLevel,
  GrammarAnalysis,
  type GrammarAnalysisData,
} from "./analysis.js";
