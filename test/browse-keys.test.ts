import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

function vec(text: string): number[] {
  const terms = [
    "alpha one", "alpha two", "alpha three", "beta only",
    "shared", "first", "second", "third", "beta",
  ];
  const out = new Array(terms.length + 1).fill(0);
  const index = terms.indexOf(text.toLowerCase());
  out[index >= 0 ? index : terms.length] = 1;
  return out;
}

test("browseKeys is namespace-scoped, hub-first, paginated, and distinguishes an empty namespace", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-browse-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.KEYMEM_SHORT_KEY_MERGE = "0";
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  t.after(() => delete process.env.KEYMEM_SHORT_KEY_MERGE);

  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vec(text));
  t.after(() => embedding.__clearTestEmbedder());

  const { MemoryGraph, classifyRecallStatus } =
    await import(`../src/memoryGraph.ts?browse=${n++}`);
  const graph = new MemoryGraph();
  await graph.load();
  await graph.add("alpha one", ["shared", "first"], { namespace: "alpha" });
  await graph.add("alpha two", ["shared", "second"], { namespace: "alpha" });
  await graph.add("alpha three", ["shared", "third"], { namespace: "alpha" });
  await graph.add("beta only", ["beta"], { namespace: "beta" });

  const alpha = await graph.browseKeys("alpha", { limit: 2 }) as any;
  assert.equal(alpha.status, "ok");
  assert.equal(alpha.memory_count, 3);
  assert.equal(alpha.keys[0].concept, "shared");
  assert.equal(alpha.keys[0].is_hub, true);
  assert.equal(alpha.keys[0].memory_count, 3);
  assert.equal(alpha.keys.some((key: any) => key.concept === "beta"), false);
  assert.equal(alpha.next_offset, 2);

  const hubs = await graph.browseKeys("alpha", { hubsOnly: true }) as any;
  assert.deepEqual(hubs.keys.map((key: any) => key.concept), ["shared"]);

  const empty = await graph.browseKeys("missing") as any;
  assert.equal(empty.status, "empty_namespace");
  assert.equal(empty.memory_count, 0);
  assert.deepEqual(empty.keys, []);
  assert.equal(classifyRecallStatus(1, 3), "found");
  assert.equal(classifyRecallStatus(0, 3), "no_match");
  assert.equal(classifyRecallStatus(0, 0), "empty_namespace");
});

test("retrieval payloads label incompatible score scales", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-scores-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.KEYMEM_SHORT_KEY_MERGE = "0";
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  t.after(() => delete process.env.KEYMEM_SHORT_KEY_MERGE);

  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vec(text));
  t.after(() => embedding.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?scores=${n++}`);
  const graph = new MemoryGraph();
  await graph.load();
  await graph.add("alpha one", ["shared"], { namespace: "alpha" });
  const [key] = await graph.searchKeys("shared", 1, "alpha") as any[];
  assert.equal(key.score_kind, "key_relevance");

  const read = await graph.readKey(key.key_id, { query: "alpha one", namespace: "alpha" }) as any;
  assert.equal(read.memories[0].score_kind, "within_key_rank");
  assert.equal(read.memories[0].content_relevance, 1);

  const [direct] = await graph.recall(
    "alpha one", 1, "alpha", false, 1, 0, 0, 0, 0, 0, false
  ) as any[];
  assert.equal(direct.score_kind, "rrf_fused_rank");
  assert.equal(direct.rank_score, direct.score);
  assert.equal(direct.relevance_score, 1);
});
