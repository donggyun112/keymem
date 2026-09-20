// Widen Hebbian reinforcement from "only the single top-1 result" to "every RETURNED
// hop>=2 result's connecting key" — scaled down vs. the top-1 amount so a large result set
// doesn't inflate the graph as fast as a confirmed top hit would.
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

// Query/hub/M1-content share axis 0 (cos=1.0, direct match). "conn" sits on axis 1 —
// orthogonal to the query, so it is never matched directly and only reachable by
// traversing from M1. M2/M3 content sits on its own orthogonal axis so neither is
// admitted by content similarity either — their only path in is the hop from M1.
function vec(tx: string): number[] {
  if (tx === "hub" || tx === "hub query" || tx === "M1CONTENT") return [1, 0, 0, 0];
  if (tx === "conn") return [0, 1, 0, 0];
  if (tx === "M2CONTENT") return [0, 0, 1, 0];
  if (tx === "M3CONTENT") return [0, 0, 0, 1];
  return [0, 0, 0, 0];
}

test("recall reinforces the connecting key of every returned hop>=2 result, scaled below the top-1 amount", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-hopreinf-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder(vec);
  t.after(() => emb.__clearTestEmbedder());

  const mg = await import(`../src/memoryGraph.ts?hopreinf=${n++}`);
  const g = new mg.MemoryGraph({ now: () => 1_800_000_000 });
  await g.load();

  const [m1] = await g.add("M1CONTENT", ["hub", "conn"]);
  const [m2] = await g.add("M2CONTENT", ["conn"]);
  const [m3] = await g.add("M3CONTENT", ["conn"]);
  const hubId = Object.keys(g.keys).find((k) => g.keys[k].concept === "hub")!;
  const connId = Object.keys(g.keys).find((k) => g.keys[k].concept === "conn")!;

  const gw = g as unknown as { _getLinkWeight(k: string, m: string): number };
  const hubM1Before = gw._getLinkWeight(hubId, m1);
  const connM1Before = gw._getLinkWeight(connId, m1);
  const connM2Before = gw._getLinkWeight(connId, m2);
  const connM3Before = gw._getLinkWeight(connId, m3);

  const res = (await g.recall("hub query", 5, null, false, 2)) as Array<{ id: string; hop: number }>;
  assert.equal(res[0].id, m1, "M1 (direct hit) should rank first");
  assert.ok(res.some((r) => r.id === m2 && r.hop === 2), "M2 should be returned at hop 2");
  assert.ok(res.some((r) => r.id === m3 && r.hop === 2), "M3 should be returned at hop 2");

  const hubM1After = gw._getLinkWeight(hubId, m1);
  const connM1After = gw._getLinkWeight(connId, m1);
  const connM2After = gw._getLinkWeight(connId, m2);
  const connM3After = gw._getLinkWeight(connId, m3);

  assert.ok(hubM1After > hubM1Before, "top-1's own matched key must still be reinforced");
  assert.equal(connM1After, connM1Before, "top-1's non-matched key must not double-dip");

  const m2Delta = connM2After - connM2Before;
  const m3Delta = connM3After - connM3Before;
  assert.ok(m2Delta > 0, "hop-2 result M2's connecting key must be reinforced");
  assert.ok(m3Delta > 0, "hop-2 result M3's connecting key must be reinforced");
  const topDelta = hubM1After - hubM1Before;
  assert.ok(m2Delta < topDelta, "hop-2 reinforcement must be scaled below the top-1 amount");
  assert.ok(m3Delta < topDelta, "hop-2 reinforcement must be scaled below the top-1 amount");
});
