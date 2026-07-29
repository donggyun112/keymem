import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

// residence key sits at cos 0.5 to the query — below the bgem3 gates (key 0.62 /
// content 0.55) but well above noise; beverage is orthogonal.
function vec(text: string): number[] {
  const t = text.toLowerCase();
  if (t.includes("주소")) return [1, 0, 0]; // query
  if (t.includes("거주지")) return [0.5, 0.866, 0]; // near-miss key (cos 0.5)
  if (t.includes("강남")) return [0.45, 0.893, 0]; // its memory content (cos 0.45)
  return [0, 0, 1]; // unrelated key/content
}

test("empty recall surfaces nearest ungated keys", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-nearmiss-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vec(text));
  t.after(() => embedding.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?nearmiss=${n++}`);
  const graph = new MemoryGraph();
  await graph.load();
  await graph.add("사용자는 강남에 산다", ["거주지"], { namespace: "default" });
  await graph.add("음료는 아메리카노", ["음료"], { namespace: "default" });

  const gated = await graph.searchKeys("주소", 8, "default");
  assert.equal(gated.length, 0); // below both gates → recall is empty

  const near = await graph.nearestKeys("주소", "default", 5);
  assert.ok(near.length >= 1);
  assert.equal(near[0].concept, "거주지");
  assert.ok(near[0].score > 0.4 && near[0].score < 0.62);
  assert.equal(near[0].memory_count, 1);

  // namespace scoping: nothing active in an empty namespace
  const other = await graph.nearestKeys("주소", "elsewhere", 5);
  assert.equal(other.length, 0);
});
