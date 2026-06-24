// Key fragmentation: the SAME surface string ("동균") must resolve to ONE key cluster regardless of
// key_type. Before cross-type reconciliation, a `name` hub and a `concept` key for the same string
// could coexist — so a pivot onto the 1-memory concept twin was a dead end. These tests cover both
// (A) preventing new twins at creation time, and (B) healing already-fragmented keys on load.
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

// Deterministic embedder: same string → identical vector (cosine 1.0); different strings → distinct
// bigram profile (cosine well below the 0.86 key-merge / 0.94 dedup thresholds), so nothing merges
// or dedups by accident and the tests isolate the cross-type reconciliation logic.
function vec(tx: string): number[] {
  const v = new Array(64).fill(0);
  const t = tx.toLowerCase();
  if (t.length === 1) v[t.charCodeAt(0) % 64] += 1;
  for (let i = 0; i < t.length - 1; i++) v[(t.charCodeAt(i) * 256 + t.charCodeAt(i + 1)) % 64] += 1;
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
  return v.map((x) => x / norm);
}

async function freshGraph(t: { after: (fn: () => unknown) => void }) {
  const dir = await mkdtemp(join(tmpdir(), "sm-frag-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((tx: string) => vec(tx));
  t.after(() => emb.__clearTestEmbedder());
  const mg = await import(`../src/memoryGraph.ts?frag=${n++}`);
  return mg.MemoryGraph;
}

test("A1: a concept request for an existing name string joins the name hub (no twin)", async (t) => {
  const MemoryGraph = await freshGraph(t);
  const g = new MemoryGraph();
  await g.load();

  const nameKid = await g.findOrCreateKey("동균", "name");
  const conceptKid = await g.findOrCreateKey("동균", "concept");

  assert.equal(conceptKid, nameKid, "same string must reuse the same key id");
  const donggyun = Object.values(g.keys).filter((k: { concept: string }) => k.concept === "동균");
  assert.equal(donggyun.length, 1, "exactly one '동균' key, not a name+concept pair");
  assert.equal(g.keys[nameKid].key_type, "name", "canonical stays the stronger name type");
});

test("A2: a name request for an existing concept string promotes it in place (no twin)", async (t) => {
  const MemoryGraph = await freshGraph(t);
  const g = new MemoryGraph();
  await g.load();

  const conceptKid = await g.findOrCreateKey("Curie", "concept");
  assert.equal(g.keys[conceptKid].key_type, "concept");
  const nameKid = await g.findOrCreateKey("Curie", "name");

  assert.equal(nameKid, conceptKid, "name request must reuse the existing concept key's id");
  assert.equal(g.keys[conceptKid].key_type, "name", "the concept key is promoted to name in place");
  assert.equal(
    Object.values(g.keys).filter((k: { concept: string }) => k.concept === "Curie").length,
    1,
    "exactly one 'Curie' key"
  );
});

test("B: load() heals an already-fragmented name+concept pair, relinking the twin's memory", async (t) => {
  const MemoryGraph = await freshGraph(t);

  // Build the broken state directly (the fix prevents creating it via the API): a name hub with a
  // memory, plus a legacy concept twin of the same string carrying its own memory.
  const g1 = new MemoryGraph();
  await g1.load();
  const [mHub] = await g1.add("동균은 밤에 작업하는 걸 선호한다", ["동균"], { keyTypes: { 동균: "name" } });
  const nameKid = Object.keys(g1.keys).find((k) => g1.keys[k].concept === "동균");
  assert.ok(nameKid, "name hub created");

  const twinKid = "legacytwin01";
  g1.keys[twinKid] = { id: twinKid, concept: "동균", aliases: [], embedding: g1.keys[nameKid!].embedding, key_type: "concept" };
  const [mTwin] = await g1.add("동균의 거주지는 서울이다", ["거주지"], {});
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (g1 as any)._link(twinKid, mTwin);
  await g1.save();

  // Two '동균' keys now exist on disk. A fresh load must heal them.
  const g2 = new MemoryGraph();
  await g2.load();

  const donggyun = Object.keys(g2.keys).filter((k) => g2.keys[k].concept === "동균");
  assert.equal(donggyun.length, 1, `fragmented twins must merge into one, got ${donggyun.length}`);
  const survivor = donggyun[0];
  assert.equal(g2.keys[survivor].key_type, "name", "canonical is the name hub (more memories + stronger type)");
  assert.equal(g2.keys[twinKid], undefined, "the concept twin key is gone");

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const linked = (g2 as any)._keyToMems[survivor] as Map<string, number>;
  assert.ok(linked.has(mHub), "hub memory still linked");
  assert.ok(linked.has(mTwin), "the twin's memory was relinked onto the canonical key");
});
