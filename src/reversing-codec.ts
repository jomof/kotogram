import { Codec } from "./codec.js";

/**
 * A codec that reverses strings for both encoding and decoding.
 */
export class ReversingCodec implements Codec {
  /**
   * Encode a string by reversing it.
   * @param text - The string to encode
   * @returns The reversed string
   */
  encode(text: string): string {
    return text.split("").reverse().join("");
  }

  /**
   * Decode a string by reversing it.
   * @param text - The string to decode
   * @returns The reversed string
   */
  decode(text: string): string {
    return text.split("").reverse().join("");
  }
}
