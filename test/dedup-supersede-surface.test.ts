import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

function vec(): number[] { return [0, 1]; }

async function freshGraph(t: any, now?: () => number) {
  const dir = await mkdtemp(join(tmpdir(), "sm-supersede-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((tx: string) => vec());
  t.after(() => emb.__clearTestEmbedder());

  const mg = await import(`../src/memoryGraph.ts?supersede=${n++}`);
  const g = new mg.MemoryGraph({ now });
  await g.load();
  return g;
}

test("add() surfaces the superseded id for an exact restatement", async (t) => {
  const g = await freshGraph(t);

  const [m1, dup1, sup1, conflict1] = await g.add("회의는 월요일이다", ["회의"], {});
  assert.equal(dup1, false, "first insert is not a dedup");
  assert.equal(sup1, null, "nothing superseded on a fresh insert");
  assert.equal(conflict1, false, "no conflict on a fresh insert");

  const [, dup2, sup2, conflict2] = await g.add("회의는 월요일이다", ["회의"], {});
  assert.equal(dup2, true);
  assert.equal(sup2, m1, "add() must return the id of the superseded memory");
  assert.equal(conflict2, false, "an exact restatement is not a semantic conflict");
});

test("add() keeps distinct facts even when key vectors are identical", async (t) => {
  const g = await freshGraph(t);

  const [m1] = await g.add("회의는 월요일이다", ["회의"], {});
  const [, dup2, sup2, conflict2] = await g.add("점심은 김치찌개였다", ["점심"], {});
  assert.equal(dup2, false, "distinct text is not a duplicate");
  assert.equal(sup2, null);
  assert.ok(g.memories[m1]);
  assert.equal(conflict2, false, "no shared key -> not flagged as a conflict");
});

test("expired duplicate cannot supersede a newly remembered fact", async (t) => {
  let now = 1_800_000_000;
  const g = await freshGraph(t, () => now);
  const [expiredId] = await g.add("회의는 월요일이다", ["회의"], { ttlSeconds: 1 });
  now += 2;
  const [newId, deduped, superseded] = await g.add(
    "회의는 월요일이다",
    ["회의"],
    { ttlSeconds: 60 }
  );
  assert.notEqual(newId, expiredId);
  assert.equal(deduped, false);
  assert.equal(superseded, null);
  assert.ok(g.listAll().some((m: any) => m.id === newId));
});

test("duplicate add forwards supplied expiry and profile to its successor", async (t) => {
  let now = 1_800_000_000;
  const g = await freshGraph(t, () => now);
  await g.add("회의는 월요일이다", ["회의"], {
    ttlSeconds: 100,
    decayProfile: "stable",
  });
  now += 10;
  const [id, deduped] = await g.add("회의는 월요일이다", ["회의"], {
    ttlSeconds: 200,
    decayProfile: "transient",
  });
  assert.equal(deduped, true);
  assert.equal(g.memories[id].ttl, now + 200);
  assert.equal(g.memories[id].decay_profile, "transient");
});

test("duplicate add without policies uses normal add defaults", async (t) => {
  let now = 1_800_000_000;
  const g = await freshGraph(t, () => now);
  await g.add("회의는 월요일이다", ["회의"], {
    ttlSeconds: 100,
    decayProfile: "stable",
  });
  now += 10;
  const [id, deduped] = await g.add("회의는 월요일이다", ["회의"]);
  assert.equal(deduped, true);
  assert.equal(g.memories[id].ttl, null);
  assert.equal(g.memories[id].decay_profile, "standard");
});

test("dedup normalizes Unicode and whitespace while preserving case", async (t) => {
  const g = await freshGraph(t);
  const [original] = await g.add("café meeting", ["회의"]);
  const [, duplicate, superseded, conflict] = await g.add("  cafe\u0301   meeting\n", ["회의"]);
  assert.equal(duplicate, true);
  assert.equal(superseded, original);
  assert.equal(conflict, false);
  const [, caseDuplicate, caseSuperseded] = await g.add("CAFÉ meeting", ["회의"]);
  assert.equal(caseDuplicate, false);
  assert.equal(caseSuperseded, null);
});
