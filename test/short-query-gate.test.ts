import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

// Only the query and key vectors matter; body strings must never be embedded.
function vec(text: string): number[] {
  const t = text.toLowerCase();
  if (t === "거주지") return [1, 0, 0]; // short keyword query
  if (t.includes("이 사람이 어디에 사는지 궁금")) return [1, 0, 0]; // long query, same direction
  if (t.includes("강남에 산다")) return [0.5, 0.866, 0]; // fact content (cos 0.50)
  if (t.includes("집주소키")) return [0, 0, 1]; // key, orthogonal
  return [0, 1, 0];
}

test("body-vector similarity alone cannot surface a key or memory", async (t) => {
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

  // Semantically similar body text has no lexical or key-vector match.
  const short = await graph.searchKeys("거주지", 8, "default");
  assert.equal(short.length, 0);

  // Sentence-shaped queries likewise have no matching key or body token.
  const long = await graph.searchKeys("이 사람이 어디에 사는지 궁금하다", 8, "default");
  assert.equal(long.length, 0);

  // Direct recall must not recover the removed body-vector path.
  const mems = await graph.recall("거주지", 5, "default", false, 2, 0, undefined, undefined, undefined, 0, false);
  assert.equal(mems.length, 0);
  const memsLong = await graph.recall(
    "이 사람이 어디에 사는지 궁금하다", 5, "default", false, 2, 0, undefined, undefined, undefined, 0, false
  );
  assert.equal(memsLong.length, 0);
});

test("lexical content matches survive an orthogonal key embedding", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-lexicalgate-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => text === "coffee" ? [1, 0] : [0, 1]);
  t.after(() => embedding.__clearTestEmbedder());
  const { MemoryGraph } = await import(`../src/memoryGraph.ts?shortgate=${n++}`);
  const graph = new MemoryGraph();
  await graph.load();
  await graph.add("coffee preference", ["집주소키"]);
  const hits = await graph.searchKeys("coffee", 8, "default");
  assert.equal(hits.length, 1);
  const memories = await graph.recall("coffee", 5, "default", false, 1, 0, undefined, undefined, undefined, 0, false);
  assert.equal(memories.length, 1);
});
