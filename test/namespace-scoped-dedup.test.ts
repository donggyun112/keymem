// Dedup / supersede and contradiction detection must stay inside one namespace. The same
// sentence saved under two projects is two facts; a note in project A must never be silently
// superseded (or flagged as contradicting) by a look-alike written in project B.
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

function vec(t: string): number[] {
  const m: Record<string, number[]> = {
    "회의는 월요일이다": [1, 0],
    "회의는 금요일이다": [0.97, 0.2431], // cos 0.97 to Monday → dedup band on bge-m3 (0.94)
    "회의는 수요일이다": [0.85, 0.5268], // cos 0.85 → contradiction band [0.80, 0.94)
    "회의": [0, 1],
  };
  return m[t] ?? [0, 1];
}

async function freshGraph(t: any) {
  const dir = await mkdtemp(join(tmpdir(), "sm-ns-dedup-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((tx: string) => vec(tx));
  t.after(() => emb.__clearTestEmbedder());
  const { MemoryGraph } = await import(`../src/memoryGraph.ts?nsdedup=${n++}`);
  const graph = new MemoryGraph();
  await graph.load();
  return graph;
}

test("an identical sentence in another namespace is a new memory, not a supersede", async (t) => {
  const graph = await freshGraph(t);
  const [aId, , aSup] = await graph.add("회의는 월요일이다", ["회의"], { namespace: "proj-a" });
  const [bId, , bSup] = await graph.add("회의는 월요일이다", ["회의"], { namespace: "proj-b" });
  assert.equal(aSup, null);
  assert.equal(bSup, null, "must not supersede across namespaces");
  assert.notEqual(aId, bId);
  assert.ok(graph.memories[aId], "project A's memory is still live");
  assert.ok(graph.memories[bId]);

  // Same namespace still dedups: the near-paraphrase supersedes A.
  const [cId, , cSup] = await graph.add("회의는 금요일이다", ["회의"], { namespace: "proj-a" });
  assert.equal(cSup, aId);
  assert.equal(await graph.readMemory(aId).then(() => "live", (e: Error) => e.message.includes("superseded") ? "superseded" : "other"), "superseded");
  assert.ok(graph.memories[cId]);
  assert.ok(graph.memories[bId], "project B untouched by project A's supersede");
});

test("contradiction detection does not cross namespaces", async (t) => {
  const graph = await freshGraph(t);
  const [aId] = await graph.add("회의는 월요일이다", ["회의"], { namespace: "proj-a" });
  const [bId] = await graph.add("회의는 수요일이다", ["회의"], { namespace: "proj-b" });
  assert.deepEqual(graph.memories[aId].contradicts, []);
  assert.deepEqual(graph.memories[bId].contradicts, []);

  const [cId] = await graph.add("회의는 수요일이다", ["회의"], { namespace: "proj-a" });
  assert.deepEqual(graph.memories[cId].contradicts, [aId], "same-namespace conflict is still flagged");
  assert.deepEqual(graph.memories[aId].contradicts, [cId]);
});

test("read_key with a concept name instead of a key_id explains what a key_id is", async (t) => {
  const graph = await freshGraph(t);
  await graph.add("회의는 월요일이다", ["회의"], { namespace: "proj-a" });
  await assert.rejects(graph.readKey("회의", { namespace: "proj-a" }), /12-character id.*recall\(query\)/s);
  await assert.rejects(graph.readKey("0123456789ab", { namespace: "proj-a" }), /^Error: Key 0123456789ab not found$/);
});
