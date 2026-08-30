import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

// Explicit vectors so the gate, not the embedder, decides: the 403 memory sits at
// cosine 0.6 from "git push" (bridgeable, below the 0.62 write-time auto-link bar),
// the budget memory is orthogonal to "review" (a shared token, not a shared topic).
const VECTORS: Record<string, number[]> = {
  "git push": [1, 0, 0, 0],
  review: [0, 0, 1, 0],
  "git push 403 fix": [0.8, 0.6, 0, 0],
  "quarterly budget review": [0, 0.6, 0.8, 0],
  "release pushed": [1, 0, 0, 0],
  "403 on the release push": [0.6, 0.8, 0, 0],
  "budget numbers for q3": [0, 1, 0, 0],
  "code review notes": [0, 0, 1, 0],
};

function vec(text: string): number[] {
  return VECTORS[text.toLowerCase()] ?? [0, 0, 0, 1];
}

test("legacy phrase keys bridge onto contained atomic keys, but only on real topic overlap", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-phrase-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.KEYMEM_SHORT_KEY_MERGE = "0";
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  t.after(() => delete process.env.KEYMEM_SHORT_KEY_MERGE);

  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vec(text));
  t.after(() => embedding.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?phrase=${n++}`);
  const graph = new MemoryGraph();
  await graph.load();
  // Atomic hubs the store already built.
  await graph.add("release pushed", ["git push"]);
  await graph.add("code review notes", ["review"]);
  // Two memories filed under phrase keys nothing else reaches.
  const [onTopic] = await graph.add("403 on the release push", ["git push 403 fix"]);
  const [offTopic] = await graph.add("budget numbers for q3", ["quarterly budget review"]);
  await graph.flush();

  // Bridging runs on load; flush persists the added links.
  const bridged = new MemoryGraph();
  await bridged.load();
  await bridged.flush();

  const reloaded = new MemoryGraph();
  await reloaded.load();

  const onTopicKeys = reloaded.getKeysForMemory(onTopic);
  assert.ok(onTopicKeys.includes("git push"), `expected bridge onto "git push", got ${onTopicKeys.join(", ")}`);
  assert.ok(onTopicKeys.includes("git push 403 fix"), "phrase key must survive for literal recall");

  const offTopicKeys = reloaded.getKeysForMemory(offTopic);
  assert.ok(
    !offTopicKeys.includes("review"),
    `a shared token is not a shared topic — should not bridge, got ${offTopicKeys.join(", ")}`
  );

  // Idempotent: a second load adds nothing.
  const linkCount = reloaded.linkCount;
  await reloaded.flush();
  const again = new MemoryGraph();
  await again.load();
  assert.equal(again.linkCount, linkCount);
});
