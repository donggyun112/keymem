import assert from "node:assert/strict";
import test from "node:test";
import { selectInject, type InjectCandidate } from "../src/inject.ts";

// candidates in relevance order (best first); depths vary
const C: InjectCandidate[] = [
  { id: "a", depth: 0.10 },
  { id: "b", depth: 0.80 },
  { id: "c", depth: 0.30 },
  { id: "d", depth: 0.05 },
];

test("no opts: keeps relevance order", () => {
  assert.deepEqual(selectInject(C, 2), ["a", "b"]);
});

test("preferDepth: surfaces the deepest (confirmed) candidates first", () => {
  assert.deepEqual(selectInject(C, 2, { preferDepth: true }), ["b", "c"]); // depth 0.80, 0.30
});

test("exploreShallow: reserves a slot for the shallowest relevant candidate", () => {
  // preferDepth picks [b,c]; explore must surface the global shallowest (d, 0.05)
  const r = selectInject(C, 2, { preferDepth: true, exploreShallow: true });
  assert.ok(r.includes("d"), `shallowest 'd' should be surfaced, got ${r.join(",")}`);
  assert.ok(r.includes("b"), `deepest 'b' should remain, got ${r.join(",")}`);
  assert.equal(r.length, 2);
});

test("exploreShallow is a no-op when all candidates already fit", () => {
  assert.equal(selectInject(C, 4, { exploreShallow: true }).length, 4);
});
