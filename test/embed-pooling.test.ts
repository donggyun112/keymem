// bge-m3 embeds through our own ONNX session (see embedding.ts): CLS-pool the first token
// of last_hidden_state, then L2-normalize. Taking the wrong slice still yields a plausible
// 1024-float unit vector, so nothing downstream would complain — it would just quietly
// retrieve worse. This pins the slice and the normalization.
import { test } from "node:test";
import assert from "node:assert/strict";
import { clsPool } from "../src/embedding.ts";

test("clsPool takes the FIRST token, not a later one or the whole tensor", () => {
  // [1, 3 tokens, 2 dims] — CLS is [3, 4], the rest must be ignored.
  const hidden = [3, 4, 100, 100, 200, 200];
  assert.deepEqual(clsPool(hidden, 2), [0.6, 0.8]); // 3-4-5 triangle, normalized
});

test("clsPool returns a unit vector", () => {
  const hidden = Float32Array.from([1, 2, 3, 4, 9, 9, 9, 9]);
  const v = clsPool(hidden, 4);
  const norm = Math.sqrt(v.reduce((a, x) => a + x * x, 0));
  assert.ok(Math.abs(norm - 1) < 1e-9, `expected unit norm, got ${norm}`);
  assert.equal(v.length, 4);
});

test("clsPool survives an all-zero vector instead of dividing by zero", () => {
  const v = clsPool([0, 0, 0], 3);
  assert.deepEqual(v, [0, 0, 0]);
  assert.ok(v.every(Number.isFinite));
});
