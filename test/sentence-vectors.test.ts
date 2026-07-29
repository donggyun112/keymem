import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

function vec(text: string): number[] {
  const t = text.toLowerCase();
  if (t.includes("첫째 사실")) return [1, 0, 0];
  if (t.includes("둘째 사실")) return [0, 1, 0];
  if (t.includes("셋째 사실")) return [0, 0, 1];
  return [0.577, 0.577, 0.577]; // whole content / keys / everything else
}

test("splitSentences splits multi-fact notes and skips single facts", async () => {
  const { splitSentences } = await import(`../src/memoryGraph.ts?split=${n++}`);
  const multi = splitSentences(
    "첫째 사실은 이러이러하다. 둘째 사실은 저러저러하다.\n셋째 사실은 요러요러하다."
  );
  assert.equal(multi.length, 3);
  assert.match(multi[0], /첫째/);
  assert.match(multi[2], /셋째/);
  // single short fact → no sentence pack
  assert.deepEqual(splitSentences("하나의 사실만 있다."), []);
  // tiny fragments dropped
  assert.deepEqual(splitSentences("음. 네."), []);
});

test("add stores per-sentence vectors, delete removes them, save/load round-trips", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-sentvec-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vec(text));
  t.after(() => embedding.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?sentvec=${n++}`);
  const g1 = new MemoryGraph();
  await g1.load();
  const [mid] = await g1.add(
    "첫째 사실은 이러이러하다. 둘째 사실은 저러저러하다. 셋째 사실은 요러요러하다.",
    ["묶음노트"],
    { namespace: "default" }
  );
  assert.equal(g1._sentVecs[mid]?.length, 3, "three sentence vectors expected");
  assert.deepEqual(g1._sentVecs[mid][1], [0, 1, 0]);

  // single-fact memory gets no pack
  const [mid2] = await g1.add("하나의 사실만 있다.", ["단일"], { namespace: "default" });
  assert.equal(g1._sentVecs[mid2], undefined);

  await g1.save();
  const g2 = new MemoryGraph();
  await g2.load();
  assert.equal(g2._sentVecs[mid]?.length, 3, "sentence pack must round-trip via sidecar");
  assert.deepEqual(g2._sentVecs[mid][2], [0, 0, 1]);

  await g2.delete(mid);
  assert.equal(g2._sentVecs[mid], undefined, "delete must drop the sentence pack");
});
