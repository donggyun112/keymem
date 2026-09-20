// recall() surfaces batch siblings (memories written in the same remember_batch call, via a
// shared source.batch_id) as a plain suggestion — batch_sibling_ids. This never touches
// ranking, score, hop, or link weight: it's read-time discoverability, not a graph edge.
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

function vec(t: string): number[] {
  if (t.includes("첫번째") || t.includes("쿼리")) return [1, 0, 0];
  // Orthogonal to m1's content — avoids write-time dedup (cos >= ~0.94 supersedes instead of
  // adding). Still reachable via the shared "쿼리" key, which matches on the KEY's own
  // embedding, not content similarity.
  if (t.includes("두번째")) return [0, 0, 1];
  return [0, 1, 0];
}

test("recall() attaches batch_sibling_ids for memories written in the same remember_batch call", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-batchsib-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder(vec);
  t.after(() => emb.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?batchsib=${n++}`);
  const g = new MemoryGraph();
  await g.load();

  const [m1] = await g.add("첫번째 항목", ["쿼리"], { source: { batch_id: "b1" } });
  const [m2] = await g.add("두번째 항목", ["쿼리"], { source: { batch_id: "b1" } });

  const res = (await g.recall("쿼리", 5)) as Array<{ id: string; batch_sibling_ids?: string[] }>;
  const r1 = res.find((r) => r.id === m1)!;
  assert.ok(r1, "m1 must be returned");
  assert.deepEqual(r1.batch_sibling_ids, [m2], "m1's result must list m2 as its batch sibling");

  const r2 = res.find((r) => r.id === m2)!;
  assert.deepEqual(r2.batch_sibling_ids, [m1], "m2's result must list m1 as its batch sibling");
});

test("recall() omits batch_sibling_ids entirely when there is no sibling", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-batchsib-none-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder(vec);
  t.after(() => emb.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?batchsib-none=${n++}`);
  const g = new MemoryGraph();
  await g.load();

  // No batch_id at all (plain remember).
  await g.add("첫번째 항목", ["쿼리"]);

  const res = (await g.recall("쿼리", 5)) as Array<Record<string, unknown>>;
  assert.equal(res.length, 1);
  assert.ok(!("batch_sibling_ids" in res[0]), "batch_sibling_ids must be omitted, not an empty array");
});
