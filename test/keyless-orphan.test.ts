import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

// A memory with zero keys is unreachable by key traversal forever, yet it still
// counts in list_memories/memory_stats and still occupies a vector slot — the
// worst failure mode this store has. supersede() has carried an explicit
// "must NEVER produce a keyless orphan" invariant since the correct() regression,
// but it only covered the inherit path (keys omitted). These cases cover the
// three routes that still reached a keyless write:
//   A  add() with an empty key list
//   A2 add() whose keys are all dropped by sanitizeKeys (1-syllable Hangul)
//   B  supersede() whose keys are all dropped — branches on the PRE-sanitize
//      array, so it skipped both the explicit-key path and the inherit fallback
//
// Live damage before the fix: one keyless memory in the author's graph
// (id 5a53157cdfc7, ns=default, created 2026-07-31), invisible to recall.

let n = 0;

// Distinct vectors per key: two keys sharing the stub's fallback vector score
// cos 1.0 and get folded together by the short-key merge, which would mask
// whether sanitizeKeys kept them.
function vec(text: string): number[] {
  const t = text.toLowerCase();
  if (t.includes("거주지키")) return [1, 0, 0];
  if (t === "집") return [0, 0, 1];
  if (t === "돈") return [0.6, 0.8, 0];
  if (t.includes("강남에 산다")) return [0.55, 0.835, 0];
  if (t.includes("부산에 산다")) return [0.5, 0.866, 0];
  return [0, 1, 0];
}

async function newGraph(t: { after: (fn: () => unknown) => void }) {
  const dir = await mkdtemp(join(tmpdir(), "keymem-orphan-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vec(text));
  t.after(() => embedding.__clearTestEmbedder());
  const { MemoryGraph } = await import(`../src/memoryGraph.ts?orphan=${n++}`);
  const graph = new MemoryGraph();
  await graph.load();
  return graph;
}

test("add() rejects a write with no usable keys instead of orphaning it", async (t) => {
  const graph = await newGraph(t);
  await assert.rejects(
    () => graph.add("사용자는 강남에 산다", [], { namespace: "default" }),
    /at least one usable key/,
    "add() must refuse a keyless write"
  );
  assert.equal(graph.listAll().length, 0, "the rejected write must not be persisted");
});

test("add() keeps 1-syllable Hangul keys (집, 돈) instead of dropping them", async (t) => {
  const graph = await newGraph(t);
  const [mid] = await graph.add("사용자는 강남에 산다", ["집", "돈"], { namespace: "default" });
  const keys = graph.getKeysForMemory(mid);
  assert.deepEqual(new Set(keys), new Set(["집", "돈"]), `1-char CJK keys must survive, got: ${keys}`);
});

test("add() still drops 1-char latin noise but keeps the usable rest", async (t) => {
  const graph = await newGraph(t);
  const [mid] = await graph.add("사용자는 강남에 산다", ["a", "거주지키"], { namespace: "default" });
  assert.deepEqual(graph.getKeysForMemory(mid), ["거주지키"]);
});

test("correct() with only-dropped keys falls back to inheriting, never orphans", async (t) => {
  const graph = await newGraph(t);
  const [mid] = await graph.add("사용자는 강남에 산다", ["거주지키"], { namespace: "default" });
  // "a" is dropped by sanitizeKeys -> the explicit-key path links nothing. The
  // inherit fallback (and its zero-key guard) must take over.
  const nid = await graph.supersede(mid, "사용자는 부산에 산다", { keyConcepts: ["a"] });
  const keys = graph.getKeysForMemory(nid);
  assert.ok(keys.length > 0, `correct() must NEVER produce a keyless orphan, got: ${keys}`);

  const found = (await graph.searchKeys("거주지키", 8, "default")) as Array<{ concept: string }>;
  assert.ok(found.length >= 1, "the corrected fact must stay reachable via recall");
});
