import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

// The multi-fact note's WHOLE vector is a centroid far from the query (cos ≈ 0.41),
// below every content gate; its second sentence aligns with the query (cos 0.9).
function vec(text: string): number[] {
  const t = text.toLowerCase();
  if (t.includes("힌트질의")) return [0, 1, 0]; // query
  if (t.includes("첫째 기능은 격리다")) return [1, 0, 0];
  if (t.includes("둘째 기능은 힌트다")) return [0, 0.9, 0.436]; // cos 0.9 to query
  if (t.includes("첫째 기능") && t.includes("둘째 기능")) return [0.71, 0.41, 0.57]; // whole note, cos 0.41
  if (t.includes("노트키")) return [1, 0, 0]; // key, orthogonal to query
  return [0.577, 0.577, 0.577];
}

test("max-sim over sentence vectors rescues sub-fact queries", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-maxsim-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vec(text));
  t.after(() => embedding.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?maxsim=${n++}`);
  const graph = new MemoryGraph();
  await graph.load();
  const [mid] = await graph.add("첫째 기능은 격리다. 둘째 기능은 힌트다.", ["노트키"], {
    namespace: "default",
  });
  assert.equal(graph._sentVecs[mid]?.length, 2);

  // searchKeys: whole-vec (0.41) misses both gates; best sentence (0.9) passes.
  const keys = (await graph.searchKeys("힌트질의", 8, "default")) as any[];
  assert.equal(keys.length, 1);
  assert.equal(keys[0].match_type, "content");

  // recall: same rescue on the memory-level path.
  const mems = (await graph.recall(
    "힌트질의", 5, "default", false, 2, 0, undefined, undefined, undefined, 0, false
  )) as any[];
  assert.equal(mems.length, 1);
  assert.match(mems[0].content, /둘째 기능/);

  // nearestKeys reports the sentence-level score, not the diluted whole score.
  const near = await graph.nearestKeys("힌트질의", "default", 3);
  assert.ok(near[0].score > 0.85);

  // Without the sentence pack the whole-vec centroid stays gated out (fallback parity).
  delete graph._sentVecs[mid];
  const keysNoPack = (await graph.searchKeys("힌트질의", 8, "default")) as any[];
  assert.equal(keysNoPack.length, 0);
});
