import assert from "node:assert/strict";
import test from "node:test";
import { findStrictHopCandidate } from "../src/memoryGraph.ts";

// Helper: build a Map<string, unknown> from a list of ids (mirrors _memToKeys/_keyToMems shape).
function map(ids: string[]): Map<string, unknown> {
  return new Map(ids.map((id) => [id, true]));
}

test("findStrictHopCandidate picks a hop-2 neighbor reachable via a narrow key", () => {
  // anchor and target share "narrowKey" (2 members total) — a real, specific connector.
  const gated: Array<[string, number]> = [
    ["anchor", 0.9],
    ["noise1", 0.05],
    ["target", 0.01], // hop-2, buried at the bottom by fused score — the bug this fixes
  ];
  const memHop = { anchor: 1, noise1: 1, target: 2 };
  const memToKeys = {
    anchor: map(["narrowKey"]),
    noise1: map(["hubKey"]),
    target: map(["narrowKey"]),
  };
  const keyToMems = {
    narrowKey: map(["anchor", "target"]), // 2 members
    hubKey: map(["anchor", "noise1", "other1", "other2", "other3", "other4"]), // 6 members
  };
  const result = findStrictHopCandidate(gated, memHop, memToKeys, keyToMems, 3);
  assert.equal(result, "target");
});

test("findStrictHopCandidate ignores a hop-2 neighbor reachable only via a hub key", () => {
  const gated: Array<[string, number]> = [
    ["anchor", 0.9],
    ["target", 0.01],
  ];
  const memHop = { anchor: 1, target: 2 };
  const memToKeys = {
    anchor: map(["hubKey"]),
    target: map(["hubKey"]),
  };
  const keyToMems = {
    // hubKey has 6 members -> exceeds maxMembers=3, must NOT be treated as a narrow connector.
    hubKey: map(["anchor", "target", "other1", "other2", "other3", "other4"]),
  };
  const result = findStrictHopCandidate(gated, memHop, memToKeys, keyToMems, 3);
  assert.equal(result, null);
});

test("findStrictHopCandidate returns null when there is no hop-2 candidate at all", () => {
  const gated: Array<[string, number]> = [["anchor", 0.9], ["other", 0.5]];
  const memHop = { anchor: 1, other: 1 };
  const memToKeys = { anchor: map(["k"]), other: map(["k"]) };
  const keyToMems = { k: map(["anchor", "other"]) };
  const result = findStrictHopCandidate(gated, memHop, memToKeys, keyToMems, 3);
  assert.equal(result, null);
});

test("findStrictHopCandidate picks the highest-scoring eligible candidate when several exist", () => {
  const gated: Array<[string, number]> = [
    ["anchor", 0.9],
    ["target_weak", 0.01],
    ["target_strong", 0.03],
  ];
  const memHop = { anchor: 1, target_weak: 2, target_strong: 2 };
  const memToKeys = {
    anchor: map(["k1", "k2"]),
    target_weak: map(["k1"]),
    target_strong: map(["k2"]),
  };
  const keyToMems = {
    k1: map(["anchor", "target_weak"]),
    k2: map(["anchor", "target_strong"]),
  };
  const result = findStrictHopCandidate(gated, memHop, memToKeys, keyToMems, 3);
  assert.equal(result, "target_strong");
});

test("recall() promotes a narrow-key hop-2 candidate past a flood of same-topic noise", async (t) => {
  const { mkdtemp, rm } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const dir = await mkdtemp(join(tmpdir(), "sm-stricthop-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  // Deterministic vectors: QQ/ANCHOR are identical (cos=1.0, definite fused #1). TARGET is
  // orthogonal to the query (cos=0, never found by content alone). NOISEn all sit at
  // cos≈0.60-0.72 with the query — enough to clear bge-m3's contentRecall/minScore admit
  // gate on topic overlap alone, exactly the corpus-density flooding this feature targets.
  // Each NOISEn gets its OWN orthogonal axis (not a repeating cycle) so no two are ever
  // >=memoryDedup(0.94) similar to each other — a repeating cycle would make write-time
  // dedup silently supersede most of them, defeating the point of this test.
  const DIM = 52;
  function zeros(): number[] { return new Array(DIM).fill(0); }
  function vec(t: string): number[] {
    if (t === "QQ" || t === "ANCHOR") { const v = zeros(); v[0] = 1; return v; }
    if (t === "TARGET") { const v = zeros(); v[1] = 1; return v; }
    const m = /^NOISE(\d+)$/.exec(t);
    if (m) {
      const i = Number(m[1]);
      const c = 0.6 + (i % 5) * 0.03;
      const v = zeros();
      v[0] = c;
      v[2 + i] = Math.sqrt(1 - c * c);
      return v;
    }
    return zeros();
  }
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((text: string) => vec(text));
  t.after(() => emb.__clearTestEmbedder());
  const rer = await import("../src/reranker.ts");
  rer.__setTestReranker((_q: string, texts: string[]) => texts.map(() => 0)); // no reordering
  t.after(() => rer.__clearTestReranker());

  const mg = await import(`../src/memoryGraph.ts?stricthop=${Date.now()}`);
  const g = new mg.MemoryGraph();
  await g.load();
  await g.add("ANCHOR", ["anchorNarrowKey"], {});
  await g.add("TARGET", ["anchorNarrowKey"], {}); // shares a 2-member key with ANCHOR only
  for (let i = 0; i < 50; i++) {
    await g.add(`NOISE${i}`, [`noiseKey${i}`], {}); // each on its own key — pure content noise
  }

  const before = (await g.recall("QQ", 10, null, true, 2, 0, 0, 0, 0)) as Array<{ content: string }>;
  // Must land within the top5 confidence window specifically, not merely appear somewhere in
  // the padded actualTopK=20 list — that weaker check silently passed an earlier, broken
  // version of this feature that only ever swapped into the LAST slot of that padded list
  // (rank ~18, no closer to a real Hit@5/Hit@10 than before the fix at all).
  assert.equal(
    before.slice(0, 5).some((m) => m.content === "TARGET"),
    true,
    `expected TARGET within the top5, got: ${before.map((m) => m.content).join(",")}`
  );
});

test("recall() moves a strict-hop candidate into top5 without duplicating it when it already appears later in the results", async (t) => {
  // A smaller noise set than the "flood" test above: TARGET already lands somewhere in the
  // (unpromoted) results on its own — the promotion must MOVE it into top5, not add a second
  // copy alongside the one already further down the list.
  const { mkdtemp, rm } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const dir = await mkdtemp(join(tmpdir(), "sm-stricthop-dedup-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const DIM = 12;
  function zeros(): number[] { return new Array(DIM).fill(0); }
  function vec(t: string): number[] {
    if (t === "QQ" || t === "ANCHOR") { const v = zeros(); v[0] = 1; return v; }
    if (t === "TARGET") { const v = zeros(); v[1] = 1; return v; }
    const m = /^NOISE(\d+)$/.exec(t);
    if (m) {
      const i = Number(m[1]);
      const c = 0.6 + (i % 5) * 0.03;
      const v = zeros();
      v[0] = c;
      v[2 + i] = Math.sqrt(1 - c * c);
      return v;
    }
    return zeros();
  }
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((text: string) => vec(text));
  t.after(() => emb.__clearTestEmbedder());
  const rer = await import("../src/reranker.ts");
  rer.__setTestReranker((_q: string, texts: string[]) => texts.map(() => 0));
  t.after(() => rer.__clearTestReranker());

  const mg = await import(`../src/memoryGraph.ts?stricthop-dedup=${Date.now()}`);
  const g = new mg.MemoryGraph();
  await g.load();
  await g.add("ANCHOR", ["anchorNarrowKey"], {});
  await g.add("TARGET", ["anchorNarrowKey"], {});
  for (let i = 0; i < 8; i++) {
    await g.add(`NOISE${i}`, [`noiseKey${i}`], {});
  }

  const result = (await g.recall("QQ", 10, null, true, 2, 0, 0, 0, 0)) as Array<{ content: string }>;
  const occurrences = result.filter((m) => m.content === "TARGET").length;
  assert.equal(occurrences, 1, `TARGET must appear exactly once, got ${occurrences} in: ${result.map((m) => m.content).join(",")}`);
  assert.equal(
    result.slice(0, 5).some((m) => m.content === "TARGET"),
    true,
    `expected TARGET within the top5, got: ${result.map((m) => m.content).join(",")}`
  );
});

test("recall() never evicts a confident direct hit to make room for a strict-hop candidate", async (t) => {
  const { mkdtemp, rm } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const dir = await mkdtemp(join(tmpdir(), "sm-stricthop-safe-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  function vec(t: string): number[] {
    if (t === "QQ") return [1, 0, 0, 0];
    if (t === "ANCHOR") return [1, 0, 0, 0]; // definite fused anchor, cos=1.0
    if (t === "CONFIDENT") return [0.99, 0.1, 0, 0]; // strong, near-definite direct hit
    if (t === "TARGET") return [0, 1, 0, 0];
    return [0, 0, 1, 0];
  }
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((text: string) => vec(text));
  t.after(() => emb.__clearTestEmbedder());
  const rer = await import("../src/reranker.ts");
  rer.__setTestReranker((_q: string, texts: string[]) => texts.map(() => 0));
  t.after(() => rer.__clearTestReranker());

  const mg = await import(`../src/memoryGraph.ts?stricthop-safe=${Date.now()}`);
  const g = new mg.MemoryGraph();
  await g.load();
  await g.add("ANCHOR", ["anchorNarrowKey"], {});
  await g.add("TARGET", ["anchorNarrowKey"], {});
  await g.add("CONFIDENT", ["confidentKey"], {}); // top5 filler, but a genuinely strong hit

  const result = (await g.recall("QQ", 3, null, true, 2, 0, 0, 0, 0)) as Array<{ content: string }>;
  assert.equal(
    result.some((m) => m.content === "CONFIDENT"),
    true,
    `CONFIDENT must not be evicted, got: ${result.map((m) => m.content).join(",")}`
  );
});
