import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

// Short concept keys sit at cos ~0.5 to same-topic sentence content on bge-m3
// (measured band: same-topic 0.477–0.643, off-topic ≤0.444). The old drift-drop
// bar (KEY_RECALL 0.62) therefore dropped ALL inherited keys on a typical
// correction, leaving the new version an unreachable keyless orphan.
function vec(text: string): number[] {
  const t = text.toLowerCase();
  if (t.includes("거주지키")) return [1, 0, 0]; // concept key
  if (t.includes("강남에 산다")) return [0.55, 0.835, 0]; // original content (cos 0.55)
  if (t.includes("부산에 산다")) return [0.5, 0.866, 0]; // same-topic correction (cos 0.50)
  if (t.includes("땅콩 알러지")) return [0, 0, 1]; // off-topic correction (cos 0)
  return [0, 1, 0];
}

test("correct() without keys keeps same-topic keys and never orphans", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-correct-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vec(text));
  t.after(() => embedding.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?correct=${n++}`);
  const graph = new MemoryGraph();
  await graph.load();

  // Same-topic correction (Seoul → Busan): cos 0.50 to the key — must inherit it.
  const [mid1] = await graph.add("사용자는 강남에 산다", ["거주지키"], { namespace: "default" });
  const nid1 = await graph.supersede(mid1, "사용자는 부산에 산다");
  const keys1 = graph.getKeysForMemory(nid1);
  assert.ok(keys1.includes("거주지키"), `same-topic key must be inherited, got: ${keys1}`);

  // Off-topic correction: the key fails the drift bar, but a keyless orphan is
  // strictly worse than a stale tag — the zero-key guard must keep it.
  const nid2 = await graph.supersede(nid1, "사용자는 땅콩 알러지가 있다");
  const keys2 = graph.getKeysForMemory(nid2);
  assert.ok(keys2.length > 0, "correction must NEVER produce a keyless orphan");

  // And the orphan symptom the evaluator hit: the memory must be recall-reachable.
  const found = (await graph.searchKeys("거주지키", 8, "default")) as Array<{ concept: string }>;
  assert.ok(found.length >= 1, "corrected memory must stay reachable via recall");
});
