import assert from "node:assert/strict";
import test from "node:test";
import { freshDecayGraph } from "./decay-test-utils.js";

test("supersede rejects an expired source and explicit TTL starts at correction time", async (t) => {
  let now = 1_800_000_000;
  const { graph } = await freshDecayGraph(t, () => now);
  const [expiredId] = await graph.add("temporary", ["temporary"], { ttlSeconds: 1 });
  now += 2;
  await assert.rejects(() => graph.supersede(expiredId, "updated"), /not found/);

  const [liveId] = await graph.add("live", ["live"], { ttlSeconds: 100 });
  now += 10;
  const correctedId = await graph.supersede(liveId, "live updated", { ttlSeconds: 200 });
  assert.equal(graph.memories[correctedId].ttl, now + 200);
});

test("expired memory cannot become a live contradiction", async (t) => {
  let now = 1_800_000_000;
  const vectors: Record<string, number[]> = {
    "expired proposition": [1, 0],
    "current proposition": [0.85, Math.sqrt(1 - 0.85 ** 2)],
    topic: [0, 1],
  };
  const { graph } = await freshDecayGraph(t, () => now, (text) => vectors[text]);
  const [expiredId] = await graph.add("expired proposition", ["topic"], { ttlSeconds: 1 });
  now += 2;
  const [currentId] = await graph.add("current proposition", ["topic"]);
  assert.deepEqual(graph.memories[currentId].contradicts, []);
  assert.ok(!graph.memories[expiredId].contradicts.includes(currentId));
});
