import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { compactRecallKeys } from "../src/recallView.js";

let imports = 0;

async function setup(t: test.TestContext) {
  const dir = await mkdtemp(join(tmpdir(), "keymem-path-nav-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => {
    const vector = new Array(8).fill(0);
    for (const char of text) vector[char.codePointAt(0)! % vector.length] += 1;
    return vector;
  });
  t.after(() => embedding.__clearTestEmbedder());
  const { MemoryGraph } = await import(`../src/memoryGraph.ts?path-nav=${imports++}`);
  const graph = new MemoryGraph({ now: () => 1_000 });
  await graph.load();
  const [bridge] = await graph.add("source bridge fact", ["source-key", "target-key"], { namespace: "ns" });
  const [destination] = await graph.add("target destination fact", ["target-key", "destination-key"], { namespace: "ns" });
  const source = Object.values(graph.keys).find((key) => key.concept === "source-key")!.id;
  const target = Object.values(graph.keys).find((key) => key.concept === "target-key")!.id;
  return { dir, graph, MemoryGraph, bridge, destination, source, target };
}

function relationFor(result: any, target: string) {
  return (result.keys ?? result.connected_keys).find((key: any) => key.key_id === target)?.relation_strength;
}

test("a path stays hidden until repeated, then routes past its bridge and persists", async (t) => {
  const { dir, graph, MemoryGraph, bridge, destination, source, target } = await setup(t);
  const nav = "session-a";

  const firstBridge: any = await graph.readMemory(bridge, source, "ns", nav);
  assert.equal(relationFor(firstBridge, target), undefined);
  await graph.readMemory(destination, target, "ns", nav);

  const secondBridge: any = await graph.readMemory(bridge, source, "ns", nav);
  assert.equal(relationFor(secondBridge, target), undefined, "one completed traversal must remain latent");
  const oneShotKeys: any[] = await graph.searchKeys("source-key", 8, "ns", null, "one-shot-search");
  assert.equal(oneShotKeys.some((key) => key.relation_strength !== undefined), false);
  await graph.readMemory(destination, target, "ns", nav);

  const promoted: any = await graph.readMemory(bridge, source, "ns", nav);
  assert.equal(relationFor(promoted, target), 2);
  const routed: any = await graph.readKey(target, { namespace: "ns", query: "source bridge", navigationId: nav });
  assert.deepEqual(routed.memories.map((memory: any) => memory.memory_id), [destination]);
  assert.equal(routed.relation_path.bridge_memory_id, bridge);
  assert.equal(routed.relation_path.bridge_excluded, true);
  const promotedKeys: any[] = await graph.searchKeys("source-key", 8, "ns", null, "promoted-search");
  const relationIndex = promotedKeys.findIndex((key) => key.key_id === target && key.relation_strength !== undefined);
  assert.ok(relationIndex >= 0 && relationIndex < 5, "active path must occupy a Top-5 slot");
  const compact = compactRecallKeys(promotedKeys as any);
  assert.equal(compact[relationIndex].relation_path?.bridge_memory_id, bridge);

  await graph.flush();
  const stored = JSON.parse(await readFile(join(dir, "graph.json"), "utf8"));
  assert.equal(stored.meta.schemaVersion, 3);
  assert.equal(stored.path_relations.length, 1);
  assert.equal(stored.path_relations[0].active, true);

  const reloaded = new MemoryGraph({ now: () => 1_000 });
  await reloaded.load();
  const afterReload: any = await reloaded.readMemory(bridge, source, "ns", "session-reload");
  assert.equal(relationFor(afterReload, target), 2);

  const [topKey]: any[] = await reloaded.searchKeys("source-key", 8, "ns", null, "session-passive");
  const passive: any = await reloaded.directHydrateTop1(topKey, "source-key", "ns", undefined, "session-passive");
  assert.equal(relationFor(passive.candidate.memory, target), 2);
  const passiveRoute: any = await reloaded.readKey(target, {
    namespace: "ns",
    query: "source bridge",
    navigationId: "session-passive",
  });
  assert.deepEqual(passiveRoute.memories.map((memory: any) => memory.memory_id), [destination]);

  assert.equal(await reloaded.delete(bridge), true);
  const afterDelete = JSON.parse(await readFile(join(dir, "graph.json"), "utf8"));
  assert.deepEqual(afterDelete.path_relations, []);
});

test("pending paths are isolated by navigation session", async (t) => {
  const { dir, graph, bridge, destination, source, target } = await setup(t);
  await graph.readMemory(bridge, source, "ns", "session-a");
  await graph.readMemory(destination, target, "ns", "session-b");
  await graph.searchKeys("different topic", 8, "ns", null, "session-a");
  await graph.flush();
  const stored = JSON.parse(await readFile(join(dir, "graph.json"), "utf8"));
  assert.deepEqual(stored.path_relations, []);
});
