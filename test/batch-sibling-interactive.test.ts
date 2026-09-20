// batch_sibling_ids on the paths agents actually exercise: directHydrateTop1 (what the
// `recall` MCP tool returns) and readKey (what `read_key` returns). graph.recall() is not
// on either path — see recall's own batch-sibling test for that narrower, passive-hook-only
// surface.
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

function vec(t: string): number[] {
  const s = t.toLowerCase();
  if (s.includes("첫번째") || s.includes("쿼리")) return [1, 0, 0];
  if (s.includes("두번째")) return [0, 0, 1]; // orthogonal — avoids write-time dedup
  return [0, 1, 0];
}

async function freshGraph(prefix: string, t: any) {
  const dir = await mkdtemp(join(tmpdir(), prefix));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder(vec);
  t.after(() => emb.__clearTestEmbedder());
  const { MemoryGraph } = await import(`../src/memoryGraph.ts?batchsib-interactive=${n++}`);
  const g = new MemoryGraph();
  await g.load();
  return g;
}

test("directHydrateTop1's candidate memory carries batch_sibling_ids", async (t) => {
  const g = await freshGraph("sm-batchsib-top1-", t);
  const [m1] = await g.add("첫번째 항목", ["쿼리"], { source: { batch_id: "b1" } });
  const [m2] = await g.add("두번째 항목", ["다른키"], { source: { batch_id: "b1" } });

  const keys = await g.searchKeys("쿼리", 8, null, "쿼리");
  const topKey = keys[0] as { key_id: string };
  const decision = await g.directHydrateTop1(topKey, "쿼리", null) as {
    status: string;
    candidate: { memory: { id: string; batch_sibling_ids?: string[] } } | null;
  };

  assert.equal(decision.status, "candidate");
  assert.equal(decision.candidate?.memory.id, m1);
  assert.deepEqual(decision.candidate?.memory.batch_sibling_ids, [m2]);
});

test("readKey's memory entries carry batch_sibling_ids only when a sibling exists", async (t) => {
  const g = await freshGraph("sm-batchsib-readkey-", t);
  const [m1] = await g.add("첫번째 항목", ["쿼리"], { source: { batch_id: "b1" } });
  const [m2] = await g.add("두번째 항목", ["쿼리"], { source: { batch_id: "b1" } });
  const [m3] = await g.add("무관한 항목", ["쿼리"]); // shares the key, no batch_id

  const keyId = Object.keys(g.keys).find((k) => g.keys[k].concept === "쿼리")!;
  const page = await g.readKey(keyId, { limit: 10 }) as {
    memories: Array<{ memory_id: string; batch_sibling_ids?: string[] }>;
  };

  const e1 = page.memories.find((e) => e.memory_id === m1)!;
  const e2 = page.memories.find((e) => e.memory_id === m2)!;
  const e3 = page.memories.find((e) => e.memory_id === m3)!;
  assert.deepEqual(e1.batch_sibling_ids, [m2]);
  assert.deepEqual(e2.batch_sibling_ids, [m1]);
  assert.ok(!("batch_sibling_ids" in e3), "a memory with no batch_id must not get the field");
});
