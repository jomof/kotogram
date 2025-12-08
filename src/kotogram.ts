/**
 * Kotogram format utilities for parsing and reconstructing Japanese text.
 *
 * This module provides core utilities for working with kotogram compact format,
 * a specialized encoding for Japanese text that preserves linguistic annotations
 * alongside the original text.
 *
 * Kotogram Format Structure:
 *     The kotogram format uses Unicode markers to encode linguistic information:
 *     - ⌈⌉ : Token boundaries
 *     - ˢ : Surface form (the actual text)
 *     - ᵖ : Part of speech and grammatical features
 *     - ᵇ : Base orthography (dictionary form spelling)
 *     - ᵈ : Lemma (dictionary form)
 *     - ʳ : Reading/pronunciation
 *
 *     Example:
 *         "猫を食べる" (The cat eats) becomes:
 *         "⌈ˢ猫ᵖn⌉⌈ˢをᵖprt:case_particle⌉⌈ˢ食べるᵖv:e-ichidan-ba⌉"
 */

/**
 * Options for kotogram_to_japanese conversion
 */
export interface KotogramToJapaneseOptions {
  /**
   * If true, insert spaces between tokens to preserve word boundaries.
   * Useful for debugging or analysis. Default is false for natural
   * Japanese text without spaces.
   */
  spaces?: boolean;

  /**
   * If true (default), remove spaces around punctuation marks to ensure
   * natural Japanese formatting. Only applies when spaces=true. Handles
   * common Japanese punctuation including 。、・etc.
   */
  collapsePunctuation?: boolean;

  /**
   * If true, append IME-style readings in hiragana brackets after each token
   * when available and different from the surface form. Shows what you would
   * type in a Japanese IME to input the text. For example, "漢字[かんじ]" for
   * kanji. Default is false. Redundant readings (same as surface) are omitted.
   */
  furigana?: boolean;
}

// Part-of-speech to character mappings for punctuation
// Must match Python's POS_TO_CHARS['auxs'] exactly for cross-language compatibility
const POS_TO_CHARS: { [key: string]: string[] } = {
  auxs: [
    "。",
    "、",
    "・",
    "：",
    "；",
    "？",
    "！",
    "…",
    "「",
    "」",
    "『",
    "』",
    "{",
    "}",
    ".",
    "ー",
    ":",
    "?",
    "っ",
    "-",
    "々",
    "(",
    ")",
    "[",
    "]",
    "<",
    ">",
    "／",
    "＼",
    "＊",
    "＋",
    "＝",
    "＠",
    "＃",
    "％",
    "＆",
    "＊",
    "ぇ",
    "〇",
    "（",
    "）",
    "* ",
    "*",
    "～",
    '"',
    "◯",
  ],
};

/**
 * Convert kotogram compact representation back to Japanese text.
 *
 * This function extracts the surface forms (ˢ markers) from a kotogram string
 * and reconstructs the original Japanese text. It can optionally preserve
 * token boundaries with spaces, handle punctuation spacing intelligently, and
 * include furigana readings in brackets.
 *
 * @param kotogram - Kotogram compact sentence representation containing encoded
 *                   linguistic information. Must follow the standard kotogram format
 *                   with ⌈⌉ token boundaries and ˢ surface markers.
 * @param options - Options for conversion
 * @returns Japanese text string reconstructed from the kotogram representation.
 *          Preserves the original character sequence and can optionally show
 *          token boundaries with spaces and/or furigana readings.
 *
 * @example
 * ```typescript
 * const kotogram = "⌈ˢ猫ᵖn⌉⌈ˢをᵖprt:case_particle⌉⌈ˢ食べるᵖv⌉";
 * kotogramToJapanese(kotogram);
 * // => '猫を食べる'
 *
 * kotogramToJapanese(kotogram, { spaces: true });
 * // => '猫 を 食べる'
 *
 * const kotogram2 = "⌈ˢこんにちはᵖint⌉⌈ˢ。ᵖauxs⌉";
 * kotogramToJapanese(kotogram2, { spaces: true, collapsePunctuation: true });
 * // => 'こんにちは。'
 *
 * const kotogram3 = "⌈ˢ漢字ᵖnʳカンジ⌉⌈ˢですᵖauxv⌉";
 * kotogramToJapanese(kotogram3, { furigana: true });
 * // => '漢字[かんじ]です'
 *
 * // Redundant readings are omitted (hiragana surface = hiragana reading)
 * const kotogram4 = "⌈ˢひらがなᵖnʳヒラガナ⌉";
 * kotogramToJapanese(kotogram4, { furigana: true });
 * // => 'ひらがな'
 * ```
 *
 * @remarks
 * Without furigana=true, this function is lossy - it only preserves the
 * surface forms and discards all linguistic annotations (POS tags, readings,
 * etc.). To preserve full information, keep the original kotogram string.
 */
export function kotogramToJapanese(
  kotogram: string,
  options: KotogramToJapaneseOptions = {}
): string {
  const {
    spaces = false,
    collapsePunctuation = true,
    furigana = false,
  } = options;

  if (!furigana) {
    // Original implementation - extract surface forms only
    const pattern = /ˢ(.*?)ᵖ/gs;
    const matches: string[] = [];
    let match: RegExpExecArray | null;

    while ((match = pattern.exec(kotogram)) !== null) {
      matches.push(match[1]);
    }

    if (spaces) {
      // Join tokens with spaces
      let result = matches.join(" ").replace(/{ /g, "{").replace(/ }/g, "}");

      if (collapsePunctuation) {
        // Remove spaces around Japanese punctuation for natural formatting
        for (const punc of POS_TO_CHARS.auxs) {
          // Skip braces as they're handled above
          if (punc === "{" || punc === "}") {
            continue;
          }
          // Remove space before and after punctuation
          result = result.replace(new RegExp(` ${escapeRegExp(punc)}`, "g"), punc);
          result = result.replace(new RegExp(`${escapeRegExp(punc)} `, "g"), punc);
        }
      }

      return result;
    } else {
      // Concatenate all surface forms without spaces (natural Japanese)
      return matches.join("");
    }
  } else {
    // Furigana mode - extract surface forms and IME readings (hiragana)
    const tokens = splitKotogram(kotogram);
    const resultParts: string[] = [];

    /**
     * Convert katakana to hiragana for IME-style furigana.
     */
    function toHiragana(text: string): string {
      const result: string[] = [];
      for (const char of text) {
        const code = char.charCodeAt(0);
        // Katakana range: 0x30A1-0x30F6
        if (code >= 0x30a1 && code <= 0x30f6) {
          // Convert to hiragana by subtracting offset
          result.push(String.fromCharCode(code - 0x60));
        }
        // Keep katakana length marker as hiragana equivalent
        else if (char === "ー") {
          result.push("ー");
        } else {
          result.push(char);
        }
      }
      return result.join("");
    }

    /**
     * Check if text contains only hiragana and katakana characters.
     */
    function isKanaOnly(text: string): boolean {
      for (const char of text) {
        const code = char.charCodeAt(0);
        // Check if it's hiragana (0x3041-0x309F) or katakana (0x30A0-0x30FF)
        const isHiragana = code >= 0x3041 && code <= 0x309f;
        const isKatakana = code >= 0x30a0 && code <= 0x30ff;

        if (!isHiragana && !isKatakana) {
          return false;
        }
      }
      return true;
    }

    for (const token of tokens) {
      // Extract surface form
      const surfaceMatch = token.match(/ˢ(.*?)ᵖ/s);
      if (!surfaceMatch) {
        continue;
      }
      const surface = surfaceMatch[1];

      // For IME-style furigana, we only add readings for kanji or mixed text
      // Pure kana (hiragana/katakana) already shows the IME input
      if (isKanaOnly(surface)) {
        // Surface is already in kana - no furigana needed
        resultParts.push(surface);
      } else {
        // Surface contains kanji - extract reading for IME input
        const readingMatch = token.match(/ʳ(.*?)(?:⌉|ᵇ|ᵈ)/);
        const readingKatakana = readingMatch ? readingMatch[1] : null;

        if (readingKatakana) {
          // Convert pronunciation to hiragana for IME-style furigana
          const readingHiragana = toHiragana(readingKatakana);
          resultParts.push(`${surface}[${readingHiragana}]`);
        } else {
          // No reading available
          resultParts.push(surface);
        }
      }
    }

    if (spaces) {
      let result = resultParts
        .join(" ")
        .replace(/{ /g, "{")
        .replace(/ }/g, "}");

      if (collapsePunctuation) {
        // Remove spaces around Japanese punctuation for natural formatting
        for (const punc of POS_TO_CHARS.auxs) {
          if (punc === "{" || punc === "}") {
            continue;
          }
          result = result.replace(new RegExp(` ${escapeRegExp(punc)}`, "g"), punc);
          result = result.replace(new RegExp(`${escapeRegExp(punc)} `, "g"), punc);
        }
      }

      return result;
    } else {
      return resultParts.join("");
    }
  }
}

/**
 * Split a kotogram sentence into individual token representations.
 *
 * This function segments a complete kotogram string into a list of individual
 * token kotograms, each representing one morphological unit. Each token
 * retains its full linguistic annotation.
 *
 * @param kotogram - Kotogram compact sentence representation. Should be a valid
 *                   kotogram string with properly matched ⌈⌉ token boundaries.
 * @returns Array of individual token kotogram strings, each containing one complete
 *          token with its full annotation enclosed in ⌈⌉ boundaries. Returns empty
 *          array if no tokens are found.
 *
 * @example
 * ```typescript
 * const kotogram = "⌈ˢ猫ᵖn⌉⌈ˢをᵖprt:case_particle⌉⌈ˢ食べるᵖv⌉";
 * splitKotogram(kotogram);
 * // => ['⌈ˢ猫ᵖn⌉', '⌈ˢをᵖprt:case_particle⌉', '⌈ˢ食べるᵖv⌉']
 *
 * const kotogram2 = "⌈ˢこんにちはᵖintᵈこんにち‐はʳコンニチワ⌉⌈ˢ。ᵖauxs⌉";
 * const tokens = splitKotogram(kotogram2);
 * // tokens.length => 2
 * // tokens[0] => '⌈ˢこんにちはᵖintᵈこんにち‐はʳコンニチワ⌉'
 * ```
 *
 * @remarks
 * This function assumes well-formed kotogram input with balanced ⌈⌉ markers.
 * Malformed input may produce unexpected results. Each returned token is
 * a complete, standalone kotogram representation that can be further analyzed.
 *
 * @see {@link kotogramToJapanese} - Extract surface forms from tokens
 */
export function splitKotogram(kotogram: string): string[] {
  // Find all complete token annotations enclosed in ⌈⌉
  // Pattern matches: ⌈ followed by any chars (non-greedy) until ⌉
  const pattern = /⌈[^⌉]*⌉/g;
  const matches = kotogram.match(pattern);
  return matches || [];
}

/**
 * Escape special regex characters in a string
 */
function escapeRegExp(string: string): string {
  return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
