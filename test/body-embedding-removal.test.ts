import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { __setTestEmbedder, __clearTestEmbedder } from "../src/embedding.js";
import { __setTestReranker, __clearTestReranker } from "../src/reranker.js";
import { readVectors, writeVectors } from "../src/vectorStore.js";

let sequence = 0;
async function graphAt(dir: string) {
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const { MemoryGraph } = await import(`../src/memoryGraph.ts?no-body=${sequence++}`);
  return new MemoryGraph();
}

test("writes, corrections and retrieval never embed memory bodies", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-no-body-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const bodies = new Set(["The deployment uses a blue canary.", "The deployment uses a green canary."]);
  const calls: string[] = [];
  __setTestEmbedder((text) => {
    assert(!bodies.has(text), "memory body reached the embedder");
    calls.push(text);
    return text.includes("canary") ? [1, 0] : [0, 1];
  });
  __setTestReranker((_q, texts) => texts.map((s) => s.includes("green") ? 2 : 1));
  t.after(() => { __clearTestEmbedder(); __clearTestReranker(); });
  const graph = await graphAt(dir);
  const [first] = await graph.add([...bodies][0], ["deployment"]);
  const next = await graph.supersede(first, [...bodies][1]);
  assert(!("embedding" in graph.memories[next]));
  const hits = await graph.recall("green canary", 5, null, false, 1, 0, undefined, undefined, undefined, 0, false) as any[];
  assert(hits.some((hit) => hit.id === next), "BM25-only candidates must survive without a key cosine anchor");
  const keys = await graph.searchKeys("green canary") as any[];
  assert(keys.length > 0, "body lexical match should expose its navigation key");
  await graph.readKey(keys[0].key_id, { query: "green canary" });
  const injected = await graph.recallInject("green canary", 1, null, { minRelScore: 0 });
  assert(injected.memories.length > 0, "lexical-only evidence must remain injectable");
  assert(calls.includes("deployment"), "key embedding is retained");
  const raw = JSON.parse(await readFile(join(dir, "graph.json"), "utf8"));
  assert(Object.values(raw.memories).every((m: any) => !("embedding" in m)));
  assert([...(await readVectors(dir))!.keys()].every((id) => id.startsWith("k:")));
});

test("legacy body vectors are discarded without re-embedding or losing memory metadata", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-no-body-legacy-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const body = "A legacy body that must never reach the embedder";
  __setTestEmbedder((text) => { assert.notEqual(text, body); return [1, 0]; });
  t.after(() => __clearTestEmbedder());
  await writeFile(join(dir, "graph.json"), JSON.stringify({
    keys: { k: { id: "k", concept: "legacy", embedding: [], aliases: [], key_type: "concept" } },
    memories: { m: { id: "m", content: body, embedding: [0, 1], created_at: 1, source: { ticket: 42 }, namespace: "project", links: [], contradicts: [], ttl: null, supersedes: null, depth: 0.4 } },
    links: [{ key_id: "k", memory_id: "m", weight: 0.7 }],
    meta: { schemaVersion: 2 },
  }));
  await writeVectors(dir, new Map([["k:k", [[1, 0]]], ["m:m", [[0, 1]]]]));
  const graph = await graphAt(dir);
  await graph.load();
  assert(!("embedding" in graph.memories.m));
  assert.equal(graph.memories.m.content, body);
  assert.deepEqual(graph.memories.m.source, { ticket: 42 });
  assert.equal(graph.linkCount, 1);
  assert.deepEqual(graph.keys.k.embedding, [1, 0]);
  await graph.flush();
  assert.deepEqual([...(await readVectors(dir))!.keys()], ["k:k"]);
  const raw = JSON.parse(await readFile(join(dir, "graph.json"), "utf8"));
  assert(!("embedding" in raw.memories.m));
  assert.equal(raw.memories.m.namespace, "project");
  assert.equal(raw.links[0].weight, 0.7);
});

test("readKey revalidates handles after an unlocked rerank", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-lexical-read-race-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  __setTestEmbedder(() => [1, 0]);
  t.after(() => { __clearTestEmbedder(); __clearTestReranker(); });
  const graph = await graphAt(dir);
  const [deleted] = await graph.add("red canary", ["deployment"]);
  const [kept] = await graph.add("blue canary", ["deployment"]);
  let started!: () => void;
  let release!: () => void;
  const running = new Promise<void>((resolve) => { started = resolve; });
  const gate = new Promise<void>((resolve) => { release = resolve; });
  __setTestReranker(async (_q, texts) => { started(); await gate; return texts.map(() => 1); });
  const pending = graph.readKey(Object.keys(graph.keys)[0], { query: "canary" });
  await running;
  await graph.delete(deleted);
  release();
  const result = await pending as any;
  assert.deepEqual(result.memories.map((m: any) => m.memory_id), [kept]);
  assert.equal(result.total, 1);
});

test("passive Top-1 runs the cross-encoder only once", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-single-rerank-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  __setTestEmbedder(() => [1, 0]);
  t.after(() => { __clearTestEmbedder(); __clearTestReranker(); });
  const graph = await graphAt(dir);
  await graph.add("red canary", ["deployment"]);
  const [best] = await graph.add("blue canary", ["deployment"]);
  let calls = 0;
  __setTestReranker((_q, texts) => { calls++; return texts.map((text) => text.includes("blue") ? 3 : 1); });
  const key = Object.keys(graph.keys)[0];
  const result = await graph.directHydrateTop1({ key_id: key, concept: "deployment", score: 1, match_type: "concept" }, "canary");
  assert.equal(calls, 1);
  assert.equal(result.status === "candidate" && result.candidate.memory.id, best);
});

test("readKey pagination uses a stable rerank pool", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-rerank-page-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  __setTestEmbedder(() => [1, 0]);
  t.after(() => { __clearTestEmbedder(); __clearTestReranker(); });
  const graph = await graphAt(dir);
  for (let i = 0; i < 31; i++) await graph.add(`canary item ${i}`, ["deployment"]);
  const pools: number[] = [];
  __setTestReranker((_q, texts) => { pools.push(texts.length); return texts.map((s) => -Number(s.match(/\d+/)![0])); });
  const keyId = Object.keys(graph.keys)[0];
  const first = await graph.readKey(keyId, { query: "canary", limit: 30 }) as any;
  const second = await graph.readKey(keyId, { query: "canary", limit: 1, offset: 30 }) as any;
  assert.deepEqual(pools, [30, 30]);
  assert.equal(new Set([...first.memories, ...second.memories].map((m: any) => m.memory_id)).size, 31);
});

test("other namespaces cannot exhaust the lexical candidate budget", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-lexical-scope-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  __setTestEmbedder((text) => text === "needle" ? [1, 0] : [0, 1]);
  t.after(() => __clearTestEmbedder());
  const graph = await graphAt(dir);
  for (let i = 0; i < 55; i++) await graph.add(`needle needle needle needle needle foreign ${i}`, ["foreign"], { namespace: "other" });
  const [target] = await graph.add("needle target", ["anchor"], { namespace: "project" });
  const hits = await graph.recall("needle", 5, "project", false, 1, 0, 0, 0, 0, 0, false, null, false, false, 0) as any[];
  assert(hits.some((hit) => hit.id === target));
  assert(hits.every((hit) => hit.namespace === "project"));
});
