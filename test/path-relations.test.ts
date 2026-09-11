import assert from "node:assert/strict";
import test from "node:test";
import { DEFAULT_PATH_RELATION_CONFIG, PathRelationGraph } from "../src/pathRelations.js";

const observation = (targetKeyId: string | null, evidenceId: string) => ({
  namespace: "ns",
  sourceKeyId: "source",
  bridgeMemoryId: "bridge",
  targetKeyId,
  evidenceId,
});

test("path relations stay latent until two traversal events", () => {
  const graph = new PathRelationGraph(() => 0);
  graph.observe(observation("target", "event-1"));
  assert.deepEqual(graph.routes("ns", "source", "bridge"), []);
  assert.equal(graph.serialize()[0].active, false);

  graph.observe(observation("target", "event-2"));
  assert.deepEqual(graph.routes("ns", "source", "bridge").map((route) => route.target_key_id), ["target"]);
  assert.equal(graph.serialize()[0].evidence_count, 2);
  assert.equal(graph.serialize()[0].weight, 2);
});

test("path relations are directional and scoped to the bridge memory", () => {
  const graph = new PathRelationGraph(() => 0);
  graph.observe(observation("target", "event-1"));
  graph.observe(observation("target", "event-2"));
  assert.equal(graph.routes("ns", "target", "bridge").length, 0);
  assert.equal(graph.routes("ns", "source", "other-bridge").length, 0);
  assert.equal(graph.routes("other", "source", "bridge").length, 0);
});

test("unused weak paths demote after eight opportunities while established paths remain", () => {
  const weak = new PathRelationGraph(() => 0);
  weak.observe(observation("target", "event-1"));
  weak.observe(observation("target", "event-2"));
  for (let i = 0; i < 8; i++) weak.observe(observation(null, `miss-${i}`));
  assert.equal(weak.routes("ns", "source", "bridge").length, 0);

  const established = new PathRelationGraph(() => 0);
  for (let i = 0; i < 4; i++) established.observe(observation("target", `event-${i}`));
  for (let i = 0; i < 8; i++) established.observe(observation(null, `miss-${i}`));
  const [route] = established.routes("ns", "source", "bridge");
  assert.ok(route);
  assert.ok(Math.abs(route.weight - 2) < 1e-9);
});

test("inactive paths decay with wall time and malformed persisted paths are repaired", () => {
  let now = 0;
  const graph = new PathRelationGraph(() => now, { ...DEFAULT_PATH_RELATION_CONFIG, wallHalfLifeSeconds: 100 });
  graph.observe(observation("target", "event-1"));
  graph.observe(observation("target", "event-2"));
  now = 100;
  assert.equal(graph.routes("ns", "source", "bridge").length, 0);

  const stored = graph.serialize()[0];
  const loaded = new PathRelationGraph(() => now);
  const repaired = loaded.load([
    stored,
    { ...stored, target_key_id: "source" },
  ], (record) => record.bridge_memory_id === "bridge");
  assert.equal(repaired, true);
  assert.equal(loaded.serialize().length, 1);
});

test("key rewrites preserve direction and pruning removes dangling paths", () => {
  const graph = new PathRelationGraph(() => 0);
  graph.observe(observation("target", "event-1"));
  graph.observe(observation("target", "event-2"));
  graph.rewriteKey("target", "canonical");
  assert.equal(graph.routes("ns", "source", "bridge")[0].target_key_id, "canonical");
  assert.equal(graph.prune((record) => record.bridge_memory_id !== "bridge"), true);
  assert.equal(graph.serialize().length, 0);
});
