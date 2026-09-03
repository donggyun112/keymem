import assert from "node:assert/strict";
import test, { type TestContext } from "node:test";
import { freshDecayGraph } from "./decay-test-utils.js";

const DAY = 24 * 3600;

async function rankedFixture(t: TestContext) {
  let now = 1_800_000_000;
  const vectors: Record<string, number[]> = {
    "same relevance": [1, 0, 0, 0],
    "stale fact": [0.8, 0.6, 0, 0],
    "fresh fact": [0.8, -0.6, 0, 0],
    shared: [0, 0, 1, 0],
  };
  const { graph } = await freshDecayGraph(t, () => now, (text) => vectors[text] ?? [0, 0, 0, 1]);
  const [staleId] = await graph.add("stale fact", ["shared"]);
  const [freshId] = await graph.add("fresh fact", ["shared"]);
  graph.memories[staleId].last_confirmed_at = now - 4 * 90 * DAY;
  return { graph, ids: { staleId, freshId } };
}

async function keyEntryFixture(t: TestContext) {
  let now = 1_800_000_000;
  const vectors: Record<string, number[]> = {
    "query cue": [1, 0, 0, 0],
    "stale entry": [0.8, 0.6, 0, 0],
    "fresh entry": [0.8, -0.6, 0, 0],
    "stale-key": [0, 0, 1, 0],
    "fresh-key": [0, 0, 0, 1],
  };
  const { graph } = await freshDecayGraph(t, () => now, (text) => vectors[text] ?? [0.5, 0.5, 0.5, 0.5]);
  const [staleId] = await graph.add("stale entry", ["stale-key"]);
  await graph.add("fresh entry", ["fresh-key"]);
  graph.memories[staleId].last_confirmed_at = now - 4 * 90 * DAY;
  const staleKeyId = Object.values(graph.keys).find((key) => key.concept === "stale-key")!.id;
  const freshKeyId = Object.values(graph.keys).find((key) => key.concept === "fresh-key")!.id;
  return { graph, ids: { staleKeyId, freshKeyId } };
}

async function hopFixture(t: TestContext) {
  let now = 1_800_000_000;
  const vectors: Record<string, number[]> = {
    anchor: [1, 0, 0, 0, 0],
    "stale hop": [0, 1, 0, 0, 0],
    "fresh hop": [0, 0, 1, 0, 0],
    "bridge-stale": [0, 0, 0, 1, 0],
    "bridge-fresh": [0, 0, 0, 0, 1],
  };
  const { graph } = await freshDecayGraph(t, () => now, (text) => vectors[text] ?? [0.4, 0.4, 0.4, 0.4, 0.4]);
  const [source] = await graph.add("anchor", ["bridge-stale", "bridge-fresh"]);
  const [stale] = await graph.add("stale hop", ["bridge-stale"]);
  const [fresh] = await graph.add("fresh hop", ["bridge-fresh"]);
  graph.memories[stale].last_confirmed_at = now - 4 * 90 * DAY;
  return { graph, ids: { source, stale, fresh } };
}

test("readKey ranks fresh before stale when other evidence is equal", async (t) => {
  const { graph, ids } = await rankedFixture(t);
  const key = Object.values(graph.keys).find((candidate) => candidate.concept === "shared")!;
  const page = await graph.readKey(key.id, { query: "same relevance" }) as any;
  assert.deepEqual(page.memories.map((memory: any) => memory.memory_id), [ids.freshId, ids.staleId]);
  assert.deepEqual(page.memories.map((memory: any) => memory.validity.status), ["fresh", "stale"]);
});

test("searchKeys discounts stale member content but keeps literal key reachability", async (t) => {
  const { graph, ids } = await keyEntryFixture(t);
  const result = await graph.searchKeys("query cue", 10) as any[];
  assert.equal(result[0].key_id, ids.freshKeyId);
  const literal = await graph.searchKeys("stale-key", 10) as any[];
  assert.ok(literal.some((key) => key.key_id === ids.staleKeyId));
});

test("direct recall ranks fresh before stale when semantic evidence is equal", async (t) => {
  const { graph, ids } = await rankedFixture(t);
  const result = await graph.recall("same relevance", 10, null, false, 1, 0, 0, 0, 0, 0, false) as any[];
  assert.ok(result.findIndex((memory) => memory.id === ids.freshId) < result.findIndex((memory) => memory.id === ids.staleId));
});

test("second-hop expansion applies target freshness", async (t) => {
  const { graph, ids } = await hopFixture(t);
  const result = await graph.recall("anchor", 10, null, false, 2, 0, 0, 0, 0, 0, false) as any[];
  assert.ok(result.findIndex((memory) => memory.id === ids.fresh) < result.findIndex((memory) => memory.id === ids.stale));
});

test("related ranks fresh before stale when graph evidence is equal", async (t) => {
  const { graph, ids } = await hopFixture(t);
  const result = graph.getRelated(ids.source) as any[];
  assert.deepEqual(result.slice(0, 2).map((memory) => memory.id), [ids.fresh, ids.stale]);
  assert.ok(result.every((memory) => "validity" in memory));
});

test("list, read, recall, and injection expose validity", async (t) => {
  const { graph, ids } = await rankedFixture(t);
  assert.ok((graph.listAll() as any[]).every((memory) => "validity" in memory));
  assert.ok("validity" in ((await graph.readMemory(ids.freshId)) as any).memory);
  assert.ok(((await graph.recall("same relevance", 5, null, false, 1, 0, 0, 0, 0, 0, false)) as any[]).every((memory) => "validity" in memory));
  await graph.add(`injected payload ${"detail ".repeat(50)}`, ["injected"]);
  const injected = (await graph.recallInject("injected payload", 1, null, { maxChars: 256 })) as any;
  assert.ok(injected.memories.length > 0);
  assert.equal(injected.memories[0].content_truncated, true);
  assert.ok(injected.memories.every((memory: any) => "validity" in memory));
});
