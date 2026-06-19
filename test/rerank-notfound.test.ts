// Rerank not-found gate (SUPER_MEMORY_RERANK_MIN_SCORE): when the cross-encoder's top
// relevance logit is below the floor, the query is treated as unanswerable → recall returns
// []. This catches distractor queries that pass the bi-encoder cosine gate (e.g. "what car
// does X drive?" against an X-job memory). Driven by a test reranker (no model load).
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;
function vec(t: string): number[] {
  const m: Record<string, number[]> = {
    QQ: [1, 0, 0, 0, 0],
    A: [0.9, 0.4359, 0, 0, 0],   // cos(query)=0.90 (in pool, not a definite anchor)
    B: [0.85, 0, 0.5268, 0, 0],  // cos(query)=0.85
    ka: [0, 0, 0, 1, 0], kb: [0, 0, 0, 0, 1],
  };
  return m[t] ?? [0, 0, 0, 1, 0];
}

test("rerank not-found gate returns [] when top relevance is below the floor", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-rrnf-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  process.env.SUPER_MEMORY_RERANK_MIN_SCORE = "0"; // reject when top rerank logit < 0
  t.after(() => { delete process.env.SUPER_MEMORY_RERANK_MIN_SCORE; });

  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((tx: string) => vec(tx));
  t.after(() => emb.__clearTestEmbedder());
  const rer = await import("../src/reranker.ts");
  t.after(() => rer.__clearTestReranker());

  const mg = await import(`../src/memoryGraph.ts?rrnf=${n++}`);
  const g = new mg.MemoryGraph();
  await g.load();
  await g.add("A", ["ka"], {});
  await g.add("B", ["kb"], {});

  // All candidates score below the floor → not-found.
  rer.__setTestReranker((_q, texts) => texts.map(() => -1));
  const none = (await g.recall("QQ", 5, null, false, 2, 0, 0, 0, 0)) as any[];
  assert.equal(none.length, 0, `low rerank logits must yield [], got ${none.map((m) => m.content).join(",")}`);

  // One candidate clears the floor → returned.
  rer.__setTestReranker((_q, texts) => texts.map((_t, i) => (i === 0 ? 2 : -1)));
  const some = (await g.recall("QQ", 5, null, false, 2, 0, 0, 0, 0)) as any[];
  assert.ok(some.length > 0, "a candidate above the floor must be returned");

  // Reranker unavailable (scores === null, e.g. model missing) → the gate must NOT fire,
  // recall falls back to the fused order. Otherwise every query would become not-found.
  rer.__setTestReranker(() => null as unknown as number[]);
  const fallback = (await g.recall("QQ", 5, null, false, 2, 0, 0, 0, 0)) as any[];
  assert.ok(fallback.length > 0, "null rerank scores must NOT trigger the not-found gate");
});
