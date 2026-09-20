// correct()/supersede() inheriting a key from the old memory carries a penalized weight,
// not the old weight verbatim — the key path led to something that needed fixing once, so
// it starts the new memory below a fresh-add's trust level.
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

function vec(t: string): number[] {
  if (t.includes("주제키")) return [1, 0, 0];
  if (t.includes("원본 내용") || t.includes("수정된 내용")) return [1, 0, 0]; // stays on-topic
  return [0, 1, 0];
}

test("supersede() carries an inherited key at a penalized weight, below the old link", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-correctpenalty-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder(vec);
  t.after(() => embedding.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?correctpenalty=${n++}`);
  const g = new MemoryGraph();
  await g.load();

  const [mid1] = await g.add("원본 내용", ["주제키"]);
  const kid = Object.keys(g.keys).find((k) => g.keys[k].concept === "주제키")!;
  const gw = g as unknown as { _getLinkWeight(k: string, m: string): number };
  const oldWeight = gw._getLinkWeight(kid, mid1);

  const nid = await g.supersede(mid1, "수정된 내용");
  const newWeight = gw._getLinkWeight(kid, nid);

  assert.ok(newWeight < oldWeight, `carried key link must start below the old weight (old=${oldWeight}, new=${newWeight})`);
});
