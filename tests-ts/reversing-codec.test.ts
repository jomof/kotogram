/**
 * Tests for ReversingCodec
 */

import { test } from "node:test";
import assert from "node:assert";
import { ReversingCodec } from "../dist/reversing-codec.js";

test("ReversingCodec - encode simple string", () => {
  const codec = new ReversingCodec();
  const result = codec.encode("hello");
  assert.strictEqual(result, "olleh");
});

test("ReversingCodec - decode simple string", () => {
  const codec = new ReversingCodec();
  const result = codec.decode("olleh");
  assert.strictEqual(result, "hello");
});

test("ReversingCodec - encode/decode roundtrip", () => {
  const codec = new ReversingCodec();
  const original = "kotogram";
  const encoded = codec.encode(original);
  const decoded = codec.decode(encoded);
  assert.strictEqual(decoded, original);
});

test("ReversingCodec - encode empty string", () => {
  const codec = new ReversingCodec();
  const result = codec.encode("");
  assert.strictEqual(result, "");
});

test("ReversingCodec - decode empty string", () => {
  const codec = new ReversingCodec();
  const result = codec.decode("");
  assert.strictEqual(result, "");
});

test("ReversingCodec - encode palindrome", () => {
  const codec = new ReversingCodec();
  const palindrome = "racecar";
  const result = codec.encode(palindrome);
  assert.strictEqual(result, palindrome);
});

test("ReversingCodec - encode with spaces", () => {
  const codec = new ReversingCodec();
  const result = codec.encode("hello world");
  assert.strictEqual(result, "dlrow olleh");
});

test("ReversingCodec - encode with unicode", () => {
  const codec = new ReversingCodec();
  const result = codec.encode("hello 世界");
  assert.strictEqual(result, "界世 olleh");
});
