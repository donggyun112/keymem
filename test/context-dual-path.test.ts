import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

// The keyword query is orthogonal to everything; the sentence context aligns with the
// fact content (cos 0.95). Mirrors the measured bge-m3 behavior: sentence-shaped cues
// score higher against sentence-shaped content than noun keywords do.
function vec(text: string): number[] {
  const t = text.toLowerCase();
  if (t.includes("취향질문")) return [1, 0, 0]; // keyword query (misses everything)
  if (t.includes("어떤 음료를 좋아하")) return [0, 1, 0]; // sentence context
  if (t.includes("아메리카노")) return [0, 0.95, 0.312]; // fact content (cos 0.95 to context)
  if (t.includes("음료취향키")) return [0, 0, 1]; // its key, orthogonal to both
  // Weak content: cos 0.50 to the context, orthogonal to the query, and far from the other
  // key/memory so auto-link and dedup leave it alone.
  if (t.includes("라떼")) return [0, 0.5, -0.866];
  if (t.includes("약한키")) return [0.5, -0.6, -0.624]; // its key: below the key gate for the query
  return [0.577, 0.577, 0.577];
}

test("context parameter feeds the content path only", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-ctx-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vec(text));
  t.after(() => embedding.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?ctx=${n++}`);
  const graph = new MemoryGraph();
  await graph.load();
  await graph.add("아이스 아메리카노를 즐긴다", ["음료취향키"], { namespace: "default" });
  // Sits between the short-keyword content gate (0.46) and the sentence gate (0.55): a
  // sentence context must be judged by the sentence gate, or this leaks through as a hit.
  await graph.add("라떼는 싫어한다", ["약한키"], { namespace: "default" });

  const withoutCtx = await graph.searchKeys("취향질문", 8, "default");
  assert.equal(withoutCtx.length, 0);

  const withCtx = await graph.searchKeys("취향질문", 8, "default", "사용자가 어떤 음료를 좋아하는지 알려줘");
  assert.equal(withCtx.length, 1, "cos 0.50 content must not pass the sentence gate");
  assert.equal((withCtx[0] as { match_type: string; concept: string }).concept, "음료취향키");
  assert.equal((withCtx[0] as { match_type: string }).match_type, "content");

  // recall(): same dual-path behavior on the memory-level path.
  const memsWithout = await graph.recall("취향질문", 5, "default", false, 2, 0, undefined, undefined, undefined, 0, false);
  assert.equal(memsWithout.length, 0);
  const memsWith = await graph.recall(
    "취향질문", 5, "default", false, 2, 0, undefined, undefined, undefined, 0, false,
    "사용자가 어떤 음료를 좋아하는지 알려줘"
  );
  assert.equal(memsWith.length, 1);
  assert.match((memsWith[0] as { content: string }).content, /아메리카노/);
});
