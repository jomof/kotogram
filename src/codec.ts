/**
 * Base codec interface for encoding and decoding strings.
 */
export interface Codec {
  /**
   * Encode a string.
   * @param text - The string to encode
   * @returns The encoded string
   */
  encode(text: string): string;

  /**
   * Decode a string.
   * @param text - The string to decode
   * @returns The decoded string
   */
  decode(text: string): string;
}
