// The rerank thread cap trades latency for a lower CPU peak so a per-turn daemon stays
// unobtrusive (measured on an M4 Pro: 6.9 cores at ORT's own default, 4.0 capped). It also
// keeps the count off the E-cores, where ORT collapses. This guards the sizing rule only.
import { test } from "node:test";
import assert from "node:assert/strict";
import { modelThreads as rerankThreads } from "../src/env.ts";

test("rerank thread cap takes a quarter of the machine and never the whole of it", () => {
  assert.equal(rerankThreads(4), 1); // small laptop: one core, three left for the user
  assert.equal(rerankThreads(8), 2);
  assert.equal(rerankThreads(14), 4); // M4 Pro (10P + 4E) — the measured good point
  assert.equal(rerankThreads(24), 6);
  assert.equal(rerankThreads(128), 6); // capped: a big host is not an excuse
  assert.equal(rerankThreads(1), 1); // never 0, which ONNX reads as "pick for me"
});

test("KEYMEM_RERANK_THREADS overrides, and garbage falls back to the derived value", () => {
  assert.equal(rerankThreads(14, "8"), 8);
  assert.equal(rerankThreads(14, "1"), 1);
  assert.equal(rerankThreads(14, ""), 4);
  assert.equal(rerankThreads(14, "wat"), 4);
  assert.equal(rerankThreads(14, "0"), 4);
  assert.equal(rerankThreads(14, "-2"), 1);
});
