import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

// Keep key vectors orthogonal to query vectors to isolate lexical retrieval.
function vec(t: string): number[] {
  const m: Record<string, number[]> = {
    "취향": [1, 0, 0],
    "강한기억 문장": [0.6, 0.8, 0],
    "약한기억 문장": [0.5, 0, 0.866],
    "키A": [0, 1, 0],
    "키B": [0, 0, 1],
  };
  return m[t] ?? [0, 0.7071, 0.7071];
}

async function freshGraph(t: any) {
  const dir = await mkdtemp(join(tmpdir(), "sm-weak-fallback-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((tx: string) => vec(tx));
  t.after(() => emb.__clearTestEmbedder());
  const { MemoryGraph } = await import(`../src/memoryGraph.ts?weakfb=${n++}`);
  const graph = new MemoryGraph();
  await graph.load();
  return graph;
}

const ids = (rs: unknown) => (rs as Array<{ content: string }>).map((r) => r.content);

test("lexical content remains recallable without a body vector", async (t) => {
  const graph = await freshGraph(t);
  await graph.add("약한기억 문장", ["키B"], {});
  const res = ids(await graph.recall("약한기억", 5, null, false, 1, 0, undefined, undefined, undefined, 0, false));
  assert.deepEqual(res, ["약한기억 문장"], "lexical content supplies an independent recall path");
});

test("graph expansion includes a memory without lexical overlap through a shared key", async (t) => {
  const graph = await freshGraph(t);
  const [anchorId] = await graph.add("coffee preference", ["키A"]);
  const [associatedId] = await graph.add("morning routine", ["키A"]);
  const direct = await graph.recall("coffee", 5, null, false, 1, 0, undefined, undefined, undefined, 0, false) as Array<{id:string}>;
  assert.ok(direct.some(m => m.id === anchorId), "lexical anchor remains reachable");
  const expanded = await graph.recall("coffee", 5, null, true, 2, 0, undefined, undefined, undefined, 0, false) as Array<{id:string}>;
  assert.ok(expanded.some(m => m.id === anchorId));
  assert.ok(expanded.some(m => m.id === associatedId));
});
