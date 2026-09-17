import assert from "node:assert/strict";
import test from "node:test";
import { findStrictHopCandidate } from "../src/memoryGraph.ts";

// Helper: build a Map<string, unknown> from a list of ids (mirrors _memToKeys/_keyToMems shape).
function map(ids: string[]): Map<string, unknown> {
  return new Map(ids.map((id) => [id, true]));
}

test("findStrictHopCandidate picks a hop-2 neighbor reachable via a narrow key", () => {
  // anchor and target share "narrowKey" (2 members total) — a real, specific connector.
  const gated: Array<[string, number]> = [
    ["anchor", 0.9],
    ["noise1", 0.05],
    ["target", 0.01], // hop-2, buried at the bottom by fused score — the bug this fixes
  ];
  const memHop = { anchor: 1, noise1: 1, target: 2 };
  const memToKeys = {
    anchor: map(["narrowKey"]),
    noise1: map(["hubKey"]),
    target: map(["narrowKey"]),
  };
  const keyToMems = {
    narrowKey: map(["anchor", "target"]), // 2 members
    hubKey: map(["anchor", "noise1", "other1", "other2", "other3", "other4"]), // 6 members
  };
  const result = findStrictHopCandidate(gated, memHop, memToKeys, keyToMems, 3);
  assert.equal(result, "target");
});

test("findStrictHopCandidate ignores a hop-2 neighbor reachable only via a hub key", () => {
  const gated: Array<[string, number]> = [
    ["anchor", 0.9],
    ["target", 0.01],
  ];
  const memHop = { anchor: 1, target: 2 };
  const memToKeys = {
    anchor: map(["hubKey"]),
    target: map(["hubKey"]),
  };
  const keyToMems = {
    // hubKey has 6 members -> exceeds maxMembers=3, must NOT be treated as a narrow connector.
    hubKey: map(["anchor", "target", "other1", "other2", "other3", "other4"]),
  };
  const result = findStrictHopCandidate(gated, memHop, memToKeys, keyToMems, 3);
  assert.equal(result, null);
});

test("findStrictHopCandidate returns null when there is no hop-2 candidate at all", () => {
  const gated: Array<[string, number]> = [["anchor", 0.9], ["other", 0.5]];
  const memHop = { anchor: 1, other: 1 };
  const memToKeys = { anchor: map(["k"]), other: map(["k"]) };
  const keyToMems = { k: map(["anchor", "other"]) };
  const result = findStrictHopCandidate(gated, memHop, memToKeys, keyToMems, 3);
  assert.equal(result, null);
});

test("findStrictHopCandidate picks the highest-scoring eligible candidate when several exist", () => {
  const gated: Array<[string, number]> = [
    ["anchor", 0.9],
    ["target_weak", 0.01],
    ["target_strong", 0.03],
  ];
  const memHop = { anchor: 1, target_weak: 2, target_strong: 2 };
  const memToKeys = {
    anchor: map(["k1", "k2"]),
    target_weak: map(["k1"]),
    target_strong: map(["k2"]),
  };
  const keyToMems = {
    k1: map(["anchor", "target_weak"]),
    k2: map(["anchor", "target_strong"]),
  };
  const result = findStrictHopCandidate(gated, memHop, memToKeys, keyToMems, 3);
  assert.equal(result, "target_strong");
});
