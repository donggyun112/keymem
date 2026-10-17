import { test } from "node:test";
import assert from "node:assert/strict";
import { ensureLexicalTokenizer, tokenizeLexical } from "../src/lexical.js";

test("Kiwi initialization is shared and Korean content morphemes match English normalization", async () => {
  await Promise.all([ensureLexicalTokenizer(), ensureLexicalTokenizer()]);
  assert.deepEqual(tokenizeLexical("메모리를"), ["메모리"]);
  assert.deepEqual(tokenizeLexical("메모리가"), ["메모리"]);
  assert.deepEqual(tokenizeLexical("Redis MEMORY 123"), ["redis", "memory", "123"]);
  assert.deepEqual(tokenizeLexical(""), []);
  assert.deepEqual(tokenizeLexical("!?."), []);
});
