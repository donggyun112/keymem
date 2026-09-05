// The calibrated short-query content gate (bge-m3: 0.46 vs 0.55) admits weak content matches
// so a keyword query whose only related memories sit at 0.46–0.55 is not a no_match. But RRF
// fusion ranks every hop-1 candidate above every hop-2 association, so each weak admit pushes
// one association out of the top-K (the assoc2 reach regression, bisected to af884a0). Weak
// admits are therefore a FALLBACK: they enter only when nothing clears the full gate.
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

// Query "취향" (short) → [1,0,0]. STRONG memory cos 0.60 (≥0.55), WEAK memory cos 0.50
// (in the short band [0.46, 0.55)), keys orthogonal so only the content path admits anything.
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

test("a weak content match is admitted when nothing clears the full gate", async (t) => {
  const graph = await freshGraph(t);
  await graph.add("약한기억 문장", ["키B"], {});
  const res = ids(await graph.recall("취향", 5, null, false, 1, 0, undefined, undefined, undefined, 0, false));
  assert.deepEqual(res, ["약한기억 문장"], "short-query gate must still rescue the weak-only case");
});

test("a weak content match stays returned but ranks after a strong match and after a hop-2 association", async (t) => {
  const graph = await freshGraph(t);
  await graph.add("약한기억 문장", ["키B"], {});
  const [strongId] = await graph.add("강한기억 문장", ["키A"], {});
  // A memory reachable only through the strong memory's key (hop 2): orthogonal to the query.
  await graph.add("연상기억 문장", ["키A"], {});
  const direct = ids(await graph.recall("취향", 5, null, false, 1, 0, undefined, undefined, undefined, 0, false));
  assert.deepEqual(direct, ["강한기억 문장", "약한기억 문장"], "weak admit still returned, after the strong match");
  const expanded = ids(await graph.recall("취향", 5, null, true, 2, 0, undefined, undefined, undefined, 0, false));
  assert.deepEqual(
    expanded,
    ["강한기억 문장", "연상기억 문장", "약한기억 문장"],
    "a graph association outranks a weak-only content admit",
  );
  assert.ok(strongId);
});
