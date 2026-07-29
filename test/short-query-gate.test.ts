import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

// The fact content sits at cos 0.50 to the query direction: passes the 0.46 short-query
// gate, fails the 0.55 sentence gate. The key is orthogonal so only the content path can hit.
function vec(text: string): number[] {
  const t = text.toLowerCase();
  if (t === "거주지") return [1, 0, 0]; // short keyword query
  if (t.includes("이 사람이 어디에 사는지 궁금")) return [1, 0, 0]; // long query, same direction
  if (t.includes("강남에 산다")) return [0.5, 0.866, 0]; // fact content (cos 0.50)
  if (t.includes("집주소키")) return [0, 0, 1]; // key, orthogonal
  return [0, 1, 0];
}

test("short keyword queries use the calibrated lower content gate", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-shortgate-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vec(text));
  t.after(() => embedding.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?shortgate=${n++}`);
  const graph = new MemoryGraph();
  await graph.load();
  await graph.add("사용자는 강남에 산다", ["집주소키"], { namespace: "default" });

  // searchKeys: short query admits the 0.50-cosine content match via the 0.46 gate...
  const short = await graph.searchKeys("거주지", 8, "default");
  assert.equal(short.length, 1);
  assert.equal((short[0] as { match_type: string }).match_type, "content");

  // ...while a sentence-shaped query keeps the calibrated 0.55 sentence gate.
  const long = await graph.searchKeys("이 사람이 어디에 사는지 궁금하다", 8, "default");
  assert.equal(long.length, 0);

  // recall(): same short-query gate on the memory-level content path (default minScore
  // is lowered alongside so the anchor gate does not re-drop what the gate admitted).
  const mems = await graph.recall("거주지", 5, "default", false, 2, 0, undefined, undefined, undefined, 0, false);
  assert.equal(mems.length, 1);
  const memsLong = await graph.recall(
    "이 사람이 어디에 사는지 궁금하다", 5, "default", false, 2, 0, undefined, undefined, undefined, 0, false
  );
  assert.equal(memsLong.length, 0);
});
