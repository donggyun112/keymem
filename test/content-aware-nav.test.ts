// Content signal at navigation time — the cure for key-coining dependence. Pure key-space
// search is blind to content, so a query whose vocabulary misses the coined key (or a generic
// hub key) finds nothing. Two fixes, both asserted here deterministically:
//   (a) searchKeys admits/ranks a key by the best content match among its memories, so a key
//       surfaces when its CONTENT matches even if the concept does not hit the query.
//   (b) readKey ranks a key's memories by content relevance to the query, so the target rises
//       to the top of a hub instead of being buried.
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;
// Query is orthogonal to the (generic) key concept but aligned with the drink memory content.
const vecs: Record<string, number[]> = {
  DRINKQ: [1, 0, 0, 0, 0],
  취향: [0, 1, 0, 0, 0], // keySim to query = 0  -> only content can surface it
  drinkContent: [0.9, 0.436, 0, 0, 0], // cos(query) ~= 0.9
  foodContent: [0, 0, 1, 0, 0], // cos(query) = 0
};
const vec = (t: string): number[] => vecs[t] ?? [0, 0, 0, 0, 1];

async function freshGraph(t: any) {
  const dir = await mkdtemp(join(tmpdir(), "sm-cnav-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((tx: string) => vec(tx));
  t.after(() => emb.__clearTestEmbedder());
  const mg = await import(`../src/memoryGraph.ts?cnav=${n++}`);
  const g = new mg.MemoryGraph();
  await g.load();
  return g;
}

test("searchKeys surfaces a key whose content matches even if the concept misses the query", async (t) => {
  const g = await freshGraph(t);
  await g.add("drinkContent", ["취향"], {}); // generic key, but content matches the query
  await g.add("foodContent", ["취향"], {});

  const keys = (await g.searchKeys("DRINKQ", 10)) as Array<{ concept: string; match_type: string }>;
  const hit = keys.find((k) => k.concept === "취향");
  // keySim is 0 and there is no literal overlap; pre-cure this key would be excluded.
  assert.ok(hit, `generic key "취향" must surface via content match, got ${keys.map((k) => k.concept).join(",")}`);
  assert.equal(hit!.match_type, "content", "match should be attributed to content signal");
});

test("readKey ranks a hub's memories by content relevance when given the query", async (t) => {
  const g = await freshGraph(t);
  const [drinkId] = await g.add("drinkContent", ["취향"], {});
  await g.add("foodContent", ["취향"], {});
  const keyId = Object.keys(g.keys).find((k: string) => g.keys[k].concept === "취향")!;

  const withQuery = (await g.readKey(keyId, { query: "DRINKQ" })) as { memories: Array<{ memory_id: string }> };
  assert.equal(withQuery.memories[0].memory_id, drinkId, "query-relevant memory must rank first in the hub");

  // Backward compat: no query -> prior link-weight/recency ordering (no content rank applied).
  const noQuery = (await g.readKey(keyId, {})) as { memories: Array<{ memory_id: string }> };
  assert.equal(noQuery.memories.length, 2, "no-query read still returns the hub members");
});
