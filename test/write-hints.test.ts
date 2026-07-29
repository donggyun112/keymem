import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

// "적중률 개선" sits at cos 0.70 to the existing key "recall 적중률": above keyRecall
// (0.62, close enough to be the same topic) but below keyMerge (0.86, so the write-time
// auto-merge rightly left them separate) — exactly the band writeHints must surface.
function vec(text: string): number[] {
  const t = text.toLowerCase();
  if (t.includes("recall 적중률")) return [1, 0, 0];
  if (t.includes("적중률 개선")) return [0.7, 0.7141428, 0];
  if (t.includes("기존내용")) return [0, 0, 1];
  if (t.includes("새내용")) return [0, 0.6, 0.8];
  if (t.includes("orthogonal") || t.includes("직교")) return [0, 0, 1]; // far from every key
  return [0, 1, 0];
}

test("remember surfaces near-neighbor keys and single-language warning", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-hints-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.KEYMEM_SHORT_KEY_MERGE = "0";
  t.after(() => delete process.env.KEYMEM_SHORT_KEY_MERGE);
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vec(text));
  t.after(() => embedding.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?hints=${n++}`);
  const graph = new MemoryGraph();
  await graph.load();
  await graph.add("기존내용", ["recall 적중률"], { namespace: "default" });
  const [mid] = await graph.add("새내용", ["적중률 개선"], { namespace: "default" });

  const hints = await graph.writeHints(mid, ["적중률 개선"]);
  assert.ok(hints);
  assert.equal(hints!.near_keys.length, 1);
  assert.equal(hints!.near_keys[0].your_key, "적중률 개선");
  assert.equal(hints!.near_keys[0].existing_concept, "recall 적중률");
  assert.ok(hints!.near_keys[0].similarity > 0.62 && hints!.near_keys[0].similarity < 0.86);
  assert.match(hints!.language_note ?? "", /cross-lingual/i);

  // bilingual keys → no language note; distant keys → no near_keys → null
  const [mid2] = await graph.add("전혀 다른 사실", ["orthogonal topic", "직교 주제"], {
    namespace: "default",
  });
  const hints2 = await graph.writeHints(mid2, ["orthogonal topic", "직교 주제"]);
  assert.equal(hints2, null);
});
