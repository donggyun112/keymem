// Silent-supersede surfacing. When a new memory's content cosine is >= the dedup
// threshold, add() treats it as a duplicate and supersedes the existing one. The
// DECISION (supersede vs keep) is intentionally unchanged here — distinguishing a
// true paraphrase from a high-similarity CONFLICT ("회의는 월요일" vs "회의는 금요일")
// needs the calibration corpus. What this fixes is the *silence*: add() must surface
// (a) WHICH memory was superseded, so it is recoverable, and (b) whether the
// superseded memory SHARED A KEY with the incoming one — the cheap, threshold-free
// signal that this looks like a conflict rather than a benign restatement.
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

// Content embeddings live so that A and B are cos=0.97 (>= bge-m3 dedup 0.94) -> B
// supersedes A. Keys are embedded too; their vectors are irrelevant to dedup (which
// is content-only) and to key sharing (exact string match), so any 2-D vector works.
function vec(t: string): number[] {
  const m: Record<string, number[]> = {
    "회의는 월요일이다": [1, 0],
    "회의는 금요일이다": [0.97, 0.2431], // cos with [1,0] = 0.97 -> dedup fires
    "점심은 김치찌개였다": [0.97, 0.2431], // same cosine, but a DIFFERENT key
    "회의": [0, 1],
    "점심": [0, 1],
  };
  return m[t] ?? [0, 1];
}

async function freshGraph(t: any) {
  const dir = await mkdtemp(join(tmpdir(), "sm-supersede-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((tx: string) => vec(tx));
  t.after(() => emb.__clearTestEmbedder());

  const mg = await import(`../src/memoryGraph.ts?supersede=${n++}`);
  const g = new mg.MemoryGraph();
  await g.load();
  return g;
}

test("add() surfaces the superseded id and flags a key-sharing conflict", async (t) => {
  const g = await freshGraph(t);

  const [m1, dup1, sup1, conflict1] = await g.add("회의는 월요일이다", ["회의"], {});
  assert.equal(dup1, false, "first insert is not a dedup");
  assert.equal(sup1, null, "nothing superseded on a fresh insert");
  assert.equal(conflict1, false, "no conflict on a fresh insert");

  // Same key "회의", content cosine 0.97 >= dedup -> superseded. Distinct fact, not a
  // paraphrase: the caller MUST be able to see what was replaced and that it conflicts.
  const [, dup2, sup2, conflict2] = await g.add("회의는 금요일이다", ["회의"], {});
  assert.equal(dup2, true, "second insert deduped against the first");
  assert.equal(sup2, m1, "add() must return the id of the superseded memory");
  assert.equal(conflict2, true, "sharing a key with the superseded memory flags a conflict");
});

test("add() reports a key-disjoint dedup as a non-conflict supersede", async (t) => {
  const g = await freshGraph(t);

  const [m1] = await g.add("회의는 월요일이다", ["회의"], {});
  // Content cosine 0.97 -> still deduped, but no shared key -> benign, not a conflict.
  const [, dup2, sup2, conflict2] = await g.add("점심은 김치찌개였다", ["점심"], {});
  assert.equal(dup2, true, "deduped on content similarity");
  assert.equal(sup2, m1, "superseded id still surfaced");
  assert.equal(conflict2, false, "no shared key -> not flagged as a conflict");
});
