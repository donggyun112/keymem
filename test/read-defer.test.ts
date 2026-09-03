// read_memory access/link signals are soft: they should accumulate in RAM and be
// flushed, not force a full-file write per read (measured 263ms @ 3k memories).
// Behavior test (no timing): flush() persists access without turning reads into
// confirmation.
import assert from "node:assert/strict";
import { mkdtemp, rm, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

test("read_memory defers persistence; flush() writes it", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-defer-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((tx: string) => { const v = new Array(8).fill(0); v[tx.length % 8] = 1; return v; });
  t.after(() => emb.__clearTestEmbedder());

  const mg = await import(`../src/memoryGraph.ts?defer=${n++}`);
  const g = new mg.MemoryGraph();
  await g.load();

  const [mid] = await g.add("a fact to reinforce by reading", ["reinforce-key"], {});
  const diskMemory = async () => JSON.parse(await readFile(join(dir, "graph.json"), "utf-8")).memories[mid];

  assert.equal((await diskMemory()).depth, 0, "new memory persists at depth 0");

  await g.readMemory(mid, null, null);
  await g.readMemory(mid, null, null);
  assert.equal((await diskMemory()).access_count, 0, "read_memory must NOT rewrite graph.json on every read");

  await g.flush();
  const persisted = await diskMemory();
  assert.equal(persisted.access_count, 2, "flush() persists accumulated access");
  assert.equal(persisted.depth, 0, "reads must not deepen memory when flushed");
});
