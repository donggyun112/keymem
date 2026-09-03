import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { freshDecayGraph } from "./decay-test-utils.js";

let n = 0;

test("v1 memories migrate confirmation state and persist schema version 2", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-decay-migration-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder(() => [1, 0]);
  t.after(() => embedding.__clearTestEmbedder());

  await writeFile(join(dir, "graph.json"), JSON.stringify({
    keys: {},
    links: [],
    memories: {
      m1: {
        id: "m1", content: "legacy", embedding: [1, 0], created_at: 100,
        source: null, supersedes: null, depth: 0.4, access_count: 4,
        last_accessed: 250, namespace: "default", ttl: null, links: [], contradicts: [],
      },
    },
    meta: {},
  }), "utf-8");

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?decay-migration=${n++}`);
  const graph = new MemoryGraph({ now: () => 500 });
  await graph.load();
  assert.deepEqual(
    {
      last_confirmed_at: graph.memories.m1.last_confirmed_at,
      confirmation_count: graph.memories.m1.confirmation_count,
      decay_profile: graph.memories.m1.decay_profile,
      last_confirmation_evidence: graph.memories.m1.last_confirmation_evidence,
      last_confirmation_source: graph.memories.m1.last_confirmation_source,
      last_confirmation_id: graph.memories.m1.last_confirmation_id,
    },
    {
      last_confirmed_at: 250,
      confirmation_count: 4,
      decay_profile: "standard",
      last_confirmation_evidence: null,
      last_confirmation_source: null,
      last_confirmation_id: null,
    }
  );
  await graph.flush();
  const saved = JSON.parse(await readFile(join(dir, "graph.json"), "utf-8"));
  assert.equal(saved.meta.schemaVersion, 2);
  assert.equal(saved.memories.m1.last_confirmed_at, 250);
});

test("new memory timestamps come from the injected epoch-seconds clock", async (t) => {
  const { graph } = await freshDecayGraph(t, () => 1_800_000_000);
  const [id] = await graph.add("clock fact", ["clock"]);
  assert.equal(graph.memories[id].created_at, 1_800_000_000);
  assert.equal(graph.memories[id].last_confirmed_at, 1_800_000_000);
  assert.equal(graph.memories[id].last_accessed, 1_800_000_000);
});
