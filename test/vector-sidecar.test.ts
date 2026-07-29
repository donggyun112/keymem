import assert from "node:assert/strict";
import { mkdtemp, rm, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

function vec(text: string): number[] {
  const t = text.toLowerCase();
  if (t.includes("알파")) return [1, 0, 0];
  if (t.includes("베타")) return [0, 1, 0];
  return [0, 0, 1];
}

async function freshGraph(dir: string) {
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const { MemoryGraph } = await import(`../src/memoryGraph.ts?sidecar=${n++}`);
  const g = new MemoryGraph();
  await g.load();
  return g;
}

test("save writes vectors to a binary sidecar and strips them from graph.json", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-sidecar-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vec(text));
  t.after(() => embedding.__clearTestEmbedder());

  const g1 = await freshGraph(dir);
  await g1.add("알파 사실", ["알파키"], { namespace: "default" });
  await g1.add("베타 사실", ["베타키"], { namespace: "default" });
  await g1.save();

  assert.ok(existsSync(join(dir, "vectors.bin")), "vectors.bin missing");
  assert.ok(existsSync(join(dir, "vectors.idx.json")), "vectors.idx.json missing");
  const raw = JSON.parse(await readFile(join(dir, "graph.json"), "utf-8"));
  for (const m of Object.values(raw.memories) as Array<{ embedding: number[] }>) {
    assert.equal(m.embedding.length, 0, "memory embedding should be stripped from JSON");
  }
  for (const k of Object.values(raw.keys) as Array<{ embedding: number[] }>) {
    assert.equal(k.embedding.length, 0, "key embedding should be stripped from JSON");
  }

  // reload hydrates from the sidecar and recall still works
  const g2 = await freshGraph(dir);
  const alpha = Object.values(g2.memories).find((m: any) => m.content.includes("알파")) as any;
  assert.deepEqual(Array.from(alpha.embedding), [1, 0, 0]);
  const hits = (await g2.searchKeys("알파키", 8, "default")) as any[];
  assert.ok(hits.length >= 1);
  assert.equal(hits[0].concept, "알파키");
});

test("legacy graph.json with inline embeddings loads and migrates on next save", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-legacy-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vec(text));
  t.after(() => embedding.__clearTestEmbedder());

  const legacy = {
    keys: {
      k1: { id: "k1", concept: "알파키", embedding: [1, 0, 0], key_type: "concept", aliases: [] },
    },
    memories: {
      m1: {
        id: "m1", content: "알파 사실", embedding: [1, 0, 0], depth: 0, access_count: 0,
        last_accessed: 0, created_at: 1785000000, namespace: "default", ttl: null,
        links: [], contradicts: [], source: null, supersedes: null,
      },
    },
    links: [{ key_id: "k1", memory_id: "m1", weight: 1.0 }],
    meta: { embeddingFingerprint: "local:bge-m3" },
  };
  await writeFile(join(dir, "graph.json"), JSON.stringify(legacy), "utf-8");

  const g = await freshGraph(dir);
  assert.deepEqual(Array.from((g.memories as any).m1.embedding), [1, 0, 0]);
  await g.flush(); // legacy load marks dirty → save migrates
  assert.ok(existsSync(join(dir, "vectors.bin")), "migration should create sidecar");
  const raw = JSON.parse(await readFile(join(dir, "graph.json"), "utf-8"));
  assert.equal(raw.memories.m1.embedding.length, 0);
});
