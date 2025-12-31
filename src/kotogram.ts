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
const POS_TO_CHARS: {[key: string]: string[]} = {
  auxs: [
    '。',
    '、',
    '・',
    '：',
    '；',
    '？',
    '！',
    '…',
    '「',
    '」',
    '『',
    '』',
    '{',
    '}',
    '.',
    'ー',
    ':',
    '?',
    'っ',
    '-',
    '々',
    '(',
    ')',
    '[',
    ']',
    '<',
    '>',
    '／',
    '＼',
    '＊',
    '＋',
    '＝',
    '＠',
    '＃',
    '％',
    '＆',
    '＊',
    'ぇ',
    '〇',
    '（',
    '）',
    '* ',
    '*',
    '～',
    '"',
    '◯',
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
  options: KotogramToJapaneseOptions = {},
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
      let result = matches.join(' ').replace(/{ /g, '{').replace(/ }/g, '}');

      if (collapsePunctuation) {
        // Remove spaces around Japanese punctuation for natural formatting
        for (const punc of POS_TO_CHARS.auxs) {
          // Skip braces as they're handled above
          if (punc === '{' || punc === '}') {
            continue;
          }
          // Remove space before and after punctuation
          result = result.replace(
            new RegExp(` ${escapeRegExp(punc)}`, 'g'),
            punc,
          );
          result = result.replace(
            new RegExp(`${escapeRegExp(punc)} `, 'g'),
            punc,
          );
        }
      }

      return result;
    } else {
      // Concatenate all surface forms without spaces (natural Japanese)
      return matches.join('');
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
        else if (char === 'ー') {
          result.push('ー');
        } else {
          result.push(char);
        }
      }
      return result.join('');
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
        .join(' ')
        .replace(/{ /g, '{')
        .replace(/ }/g, '}');

      if (collapsePunctuation) {
        // Remove spaces around Japanese punctuation for natural formatting
        for (const punc of POS_TO_CHARS.auxs) {
          if (punc === '{' || punc === '}') {
            continue;
          }
          result = result.replace(
            new RegExp(` ${escapeRegExp(punc)}`, 'g'),
            punc,
          );
          result = result.replace(
            new RegExp(`${escapeRegExp(punc)} `, 'g'),
            punc,
          );
        }
      }

      return result;
    } else {
      return resultParts.join('');
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

function escapeRegExp(string: string): string {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Linguistic features extracted from a kotogram token.
 */
export interface TokenFeatures {
  surface: string;
  pos: string;
  pos_detail_1: string;
  pos_detail_2: string;
  pos_detail_3: string;
  conjugatedType: string;
  conjugatedForm: string;
  baseOrth: string;
  lemma: string;
  reading: string;
}

// POS detail level 1 values (from Python POS1_MAP.values())
const POS1_MAP_VALUES = new Set([
  'general',
  'proper-noun',
  'common-noun',
  'numeral',
  'case-particle',
  'binding-particle',
  'adverbial-particle',
  'conjunctive-particle',
  'sentence-final-particle',
  'nominal-particle',
  'aux-verb-stem',
  'bound',
  'verbal',
  'adjectival',
  'adjectival-noun-like',
  'nominal',
  'tari',
  'filler',
  'letter',
  'ascii-art',
  'period',
  'comma',
  'open-bracket',
  'close-bracket',
]);

// POS detail level 2 values (from Python POS2_MAP.values())
const POS2_MAP_VALUES = new Set([
  'general',
  'verbal-suru',
  'verbal-suru-adj',
  'adverbial',
  'adjectival-noun-possible',
  'counter',
  'counter-possible',
  'place-name',
  'person-name',
  'kaomoji',
]);

// POS detail level 3 values (from Python POS3_MAP.values())
const POS3_MAP_VALUES = new Set([
  'general',
  'country',
  'given-name',
  'surname',
]);

// Conjugation type values (from Python CONJUGATED_TYPE_MAP.values())
const CONJUGATED_TYPE_MAP_VALUES = new Set([
  // Auxiliary verbs
  'aux-ta',
  'aux-da',
  'aux-desu',
  'aux-masu',
  'aux-nai',
  'aux-nu',
  'aux-reru',
  'aux-tai',
  'aux-rashii',
  'aux-mai',
  'aux-ja',
  'aux-ya',
  'aux-nanda',
  'aux-hen',
  // Godan verbs
  'godan-ra',
  'godan-ka',
  'godan-ga',
  'godan-sa',
  'godan-ta',
  'godan-na',
  'godan-ba',
  'godan-ma',
  'godan-waa',
  // Ichidan verbs
  'upper-ichidan-a',
  'upper-ichidan-ka',
  'upper-ichidan-ga',
  'upper-ichidan-za',
  'upper-ichidan-ta',
  'upper-ichidan-na',
  'upper-ichidan-ha',
  'upper-ichidan-ba',
  'upper-ichidan-ma',
  'upper-ichidan-ra',
  'lower-ichidan-a',
  'lower-ichidan-ka',
  'lower-ichidan-ga',
  'lower-ichidan-sa',
  'lower-ichidan-za',
  'lower-ichidan-ta',
  'lower-ichidan-da',
  'lower-ichidan-na',
  'lower-ichidan-ha',
  'lower-ichidan-ba',
  'lower-ichidan-ma',
  'lower-ichidan-ra',
  // Irregular verbs
  'ka-irregular',
  'sa-irregular',
  // Adjectives
  'i-adjective',
  // Classical Japanese
  'classical-sa-irregular',
  'classical-ra-irregular',
  'classical-adj-ku',
  'classical-adj-shiku',
  'classical-aux-tari-perfective',
  'classical-aux-tari-assertive',
  'classical-aux-nari',
  'classical-aux-ri',
  'classical-aux-beshi',
  'classical-aux-zu',
  'classical-aux-ki',
  'classical-aux-keri',
  'classical-aux-gotoshi',
  'classical-aux-maji',
  'classical-aux-mu',
  'classical-aux-ji',
  'classical-aux-nu',
  'classical-aux-rashi',
  'classical-aux-ramu',
  'classical-aux-zamasu',
  'classical-upper-nidan-ta',
  'classical-upper-nidan-da',
  'classical-upper-nidan-ba',
  'classical-lower-nidan-a',
  'classical-lower-nidan-ka',
  'classical-lower-nidan-ga',
  'classical-lower-nidan-sa',
  'classical-lower-nidan-da',
  'classical-lower-nidan-na',
  'classical-lower-nidan-ha',
  'classical-lower-nidan-ma',
  'classical-lower-nidan-ra',
  'classical-yodan-ka',
  'classical-yodan-sa',
  'classical-yodan-ta',
  'classical-yodan-ha',
  'classical-yodan-ma',
  'classical-yodan-ra',
]);

// Conjugation form values (from Python CONJUGATED_FORM_MAP.values())
const CONJUGATED_FORM_MAP_VALUES = new Set([
  'terminal',
  'terminal-nasal',
  'terminal-geminate',
  'terminal-fused',
  'terminal-u-euphonic',
  'continuative',
  'continuative-geminate',
  'continuative-nasal',
  'continuative-i-euphonic',
  'continuative-u-euphonic',
  'continuative-ni',
  'continuative-abbreviated',
  'continuative-fused',
  'continuative-auxiliary',
  'attributive',
  'attributive-nasal',
  'attributive-abbreviated',
  'attributive-auxiliary',
  'irrealis',
  'irrealis-sa',
  'irrealis-se',
  'irrealis-nasal',
  'irrealis-auxiliary',
  'conditional',
  'conditional-fused',
  'imperative',
  'volitional-presumptive',
  'realis',
  'stem',
  'stem-sa',
  'ku-form',
]);

/**
 * Extract linguistic features from a single kotogram token.
 *
 * Parses a kotogram token to extract all encoded linguistic information including
 * part of speech, conjugation details, and orthographic forms. This function handles
 * the variable-length POS format where empty fields are omitted by the parser.
 *
 * Kotogram format uses Unicode markers to encode linguistic information:
 * - ⌈⌉ : Token boundaries
 * - ˢ : Surface form (the actual text)
 * - ᵖ : Part of speech and grammatical features (colon-separated)
 * - ᵇ : Base orthography (dictionary form spelling)
 * - ᵈ : Lemma (dictionary form)
 * - ʳ : Reading/pronunciation
 *
 * The POS field (ᵖ) contains colon-separated values in a specific semantic order:
 * `pos:pos_detail_1:pos_detail_2:conjugated_type:conjugated_form`
 *
 * However, the parser omits empty fields, so this function identifies each field
 * semantically by checking which mapping it belongs to, rather than relying on
 * positional indices.
 *
 * @param token - A single kotogram token string (⌈...⌉)
 * @returns TokenFeatures object with extracted features
 *
 * @example
 * ```typescript
 * // Extract features from a verb token
 * const token = "⌈ˢ食べᵖverb:general:lower-ichidan-ba:continuativeᵇ食べるᵈ食べるʳタベ⌉";
 * const features = extractTokenFeatures(token);
 * // features.pos === 'verb'
 * // features.conjugatedType === 'lower-ichidan-ba'
 * // features.conjugatedForm === 'continuative'
 *
 * // Extract features from an auxiliary verb
 * const token2 = "⌈ˢますᵖaux-verb:aux-masu:terminalᵇますʳマス⌉";
 * const features2 = extractTokenFeatures(token2);
 * // features2.pos === 'aux-verb'
 * // features2.conjugatedType === 'aux-masu'
 * // features2.conjugatedForm === 'terminal'
 * ```
 */
export function extractTokenFeatures(token: string): TokenFeatures {
  const feature: TokenFeatures = {
    surface: '',
    pos: '',
    pos_detail_1: '',
    pos_detail_2: '',
    pos_detail_3: '',
    conjugatedType: '',
    conjugatedForm: '',
    baseOrth: '',
    lemma: '',
    reading: '',
  };

  // Extract surface form (ˢ...ᵖ)
  const surfaceMatch = token.match(/ˢ(.*?)ᵖ/s);
  if (surfaceMatch) {
    feature.surface = surfaceMatch[1];
  }

  // Extract POS data (ᵖ...ᵇ|ᵈ|ʳ|⌉)
  const posMatch = token.match(/ᵖ([^⌉ᵇᵈʳ]+)/);
  if (posMatch) {
    const posData = posMatch[1];
    const parts = posData.split(':');

    // Main POS code (always first)
    feature.pos = parts.length > 0 ? parts[0] : '';

    // Parse remaining fields semantically by checking which map they belong to
    // The parser skips empty fields, so we can't rely on position alone
    for (let i = 1; i < parts.length; i++) {
      const value = parts[i];
      if (!value) {
        continue;
      }

      // Check which map this value belongs to
      if (CONJUGATED_FORM_MAP_VALUES.has(value)) {
        feature.conjugatedForm = value;
      } else if (CONJUGATED_TYPE_MAP_VALUES.has(value)) {
        feature.conjugatedType = value;
      } else if (POS2_MAP_VALUES.has(value)) {
        // pos_detail_2 comes after pos_detail_1, so check if we already have pos_detail_1
        if (feature.pos_detail_1) {
          feature.pos_detail_2 = value;
        } else {
          feature.pos_detail_1 = value;
        }
      } else if (POS3_MAP_VALUES.has(value)) {
        // pos_detail_3 usually comes last for details
        feature.pos_detail_3 = value;
      } else if (POS1_MAP_VALUES.has(value)) {
        // pos_detail_1 comes before pos_detail_2
        if (!feature.pos_detail_1) {
          feature.pos_detail_1 = value;
        } else {
          feature.pos_detail_2 = value;
        }
      } else {
        // Unknown value - try to assign by position as fallback
        if (!feature.pos_detail_1) {
          feature.pos_detail_1 = value;
        } else if (!feature.pos_detail_2) {
          feature.pos_detail_2 = value;
        } else if (!feature.pos_detail_3) {
          feature.pos_detail_3 = value;
        } else if (!feature.conjugatedType) {
          feature.conjugatedType = value;
        } else if (!feature.conjugatedForm) {
          feature.conjugatedForm = value;
        }
      }
    }
  }

  // Extract base orthography (ᵇ...ᵈ|ʳ|⌉)
  const baseMatch = token.match(/ᵇ([^⌉ᵈʳ]+)/);
  if (baseMatch) {
    feature.baseOrth = baseMatch[1];
  }

  // Extract lemma/dictionary form (ᵈ...ʳ|⌉)
  const lemmaMatch = token.match(/ᵈ([^⌉ʳ]+)/);
  if (lemmaMatch) {
    feature.lemma = lemmaMatch[1];
  }

  // Extract reading (ʳ...⌉)
  const readingMatch = token.match(/ʳ([^⌉]+)/);
  if (readingMatch) {
    feature.reading = readingMatch[1];
  }

  return feature;
}
