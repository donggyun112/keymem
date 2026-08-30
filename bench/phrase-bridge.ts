// Phrase-key bridging ablation. Same data, same embeddings, one difference: whether
// legacy phrase keys get bridged onto the atomic keys they contain (_bridgePhraseKeys,
// disabled with KEYMEM_PHRASE_BRIDGE=false).
//
// The associative ablation (§1 of BENCHMARKS.md) found the graph's gain is in REACH,
// not rank: 2-hop hits land at ranks 9-11 and hit@5 shows no gain. Bridging turns a
// 2-hop path into a 1-hop link, so the hypothesis under test is that it moves those
// hits INTO the top 5 — and the cost to check is whether it drags token-sharing but
// off-topic memories in with them.
//
//   tsx bench/phrase-bridge.ts        # real bge-m3
import { mkdtemp, rm, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

process.env.EMBEDDING_BACKEND ??= "local";
process.env.LOCAL_EMBEDDING_MODEL ??= "bge-m3";

type Mem = { id: string; content: string; keys: string[] };
type Q = { q: string; expect: string[]; category: string };
const fixture = JSON.parse(await readFile(resolve("bench/phrase-fixture.json"), "utf-8")) as {
  memories: Mem[];
  queries: Q[];
};

// Memories filed under a phrase key that shares a TOKEN with an atomic hub but is
// off-topic for it ("예산 리뷰 회의 결과" vs the review hub). Bridging must not pull
// these into a topical query's results; counting them is the precision cost.
const OFF_TOPIC = new Set(["far-budget", "far-hiring"]);

const dir = await mkdtemp(join(tmpdir(), "km-phrase-"));
process.env.KEYMEM_DATA_DIR = dir;
const { MemoryGraph } = await import("../src/memoryGraph.ts");
const { LOCAL_EMBEDDING_MODEL } = await import("../src/embedding.ts");

const revMap: Record<string, string> = {};
{
  const builder = new MemoryGraph();
  await builder.load();
  for (const m of fixture.memories) {
    const [gid] = await builder.add(m.content, m.keys);
    revMap[gid] = m.id;
  }
  await builder.flush();
}

const TOPK = 10;
type Agg = { n: number; hit5: number; hit10: number; mrr: number; nf_ok: number; nf_n: number };
const mk = (): Agg => ({ n: 0, hit5: 0, hit10: 0, mrr: 0, nf_ok: 0, nf_n: 0 });

// Third condition: an experimental alternative gate, implemented here rather than as a
// shipped knob. COSINE (production) requires the atomic key to be topically close to the
// memory; STRUCTURAL instead asks whether the phrase key is a dead-end label (singleton)
// and the atomic key is a real hub, then links at a deliberately low weight — buying
// reach at the cost of rank noise. The point of the run is to decide which gate ships.
function unbridge(g: any, graphJson: string): void {
  const onDisk = new Set(
    (JSON.parse(graphJson).links as Array<{ key_id: string; memory_id: string }>).map(
      (l) => `${l.key_id}|${l.memory_id}`
    )
  );
  for (const [kid, mids] of Object.entries(g._keyToMems) as Array<[string, Map<string, number>]>) {
    for (const mid of [...mids.keys()]) {
      if (onDisk.has(`${kid}|${mid}`)) continue;
      mids.delete(mid);
      g._memToKeys[mid]?.delete(kid);
    }
  }
}

const STRUCTURAL_WEIGHT = 0.3;
function bridgeStructural(g: any): number {
  const degree = (kid: string): number => g._keyToMems[kid]?.size ?? 0;
  const byConcept = new Map<string, string>();
  for (const [kid, key] of Object.entries(g.keys) as Array<[string, any]>) {
    for (const surface of [key.concept, ...(key.aliases ?? [])]) {
      const norm = surface.trim().toLowerCase();
      const prev = byConcept.get(norm);
      if (!prev || degree(kid) > degree(prev)) byConcept.set(norm, kid);
    }
  }
  let n = 0;
  for (const [phraseId, key] of Object.entries(g.keys) as Array<[string, any]>) {
    const tokens = key.concept.trim().split(/\s+/);
    if (tokens.length < 3) continue;
    const mids = [...(g._keyToMems[phraseId]?.keys() ?? [])];
    if (mids.length !== 1) continue; // singleton phrase key = a label, not a concept
    const subs = new Set<string>();
    for (const k of [1, 2]) for (let i = 0; i + k <= tokens.length; i++) subs.add(tokens.slice(i, i + k).join(" ").toLowerCase());
    for (const sub of subs) {
      const atomicId = byConcept.get(sub);
      if (!atomicId || atomicId === phraseId) continue;
      if (degree(atomicId) < 3) continue; // KEY_HUB_MIN_LINKS — bridge onto real hubs only
      for (const mid of mids) {
        if (g._hasLink(atomicId, mid)) continue;
        g._link(atomicId, mid, STRUCTURAL_WEIGHT);
        n++;
      }
    }
  }
  return n;
}

const CONDITIONS = ["NO-BRIDGE", "BRIDGE", "STRUCTURAL"] as const;
const out: Record<string, Record<string, Agg>> = {};
const pollution: Record<string, number> = {};
const bridgedLinks: Record<string, number> = {};
const perQuery: Array<Record<string, unknown>> = [];

for (const cond of CONDITIONS) {
  // Both conditions load the same on-disk graph; neither flushes, so the bridged
  // links never leak into the other condition's run.
  const g = new MemoryGraph();
  const before = g.linkCount;
  await g.load();
  // Bridging is core and runs on every load, so NO-BRIDGE is reconstructed here rather
  // than switched off in production code: graph.json still holds the pre-bridge links
  // (this bench never flushes), so anything extra is exactly what the heal added.
  if (cond !== "BRIDGE") unbridge(g, await readFile(join(dir, "graph.json"), "utf-8"));
  if (cond === "STRUCTURAL") bridgeStructural(g);
  bridgedLinks[cond] = g.linkCount - before;

  const cats: Record<string, Agg> = {};
  let polluted = 0;
  for (const q of fixture.queries) {
    const res = (await g.recall(q.q, TOPK, null, true, 2)) as Array<{ id: string }>;
    const ids = res.map((r) => revMap[r.id]).filter(Boolean);
    const cat = (cats[q.category] ??= mk());
    if (q.expect.length === 0) {
      cat.nf_n++;
      cat.nf_ok += ids.length === 0 ? 1 : 0;
      continue;
    }
    cat.n++;
    const top5 = ids.slice(0, 5);
    const hit5 = q.expect.some((e) => top5.includes(e));
    const hit10 = q.expect.some((e) => ids.slice(0, 10).includes(e));
    let rank = 0;
    for (let i = 0; i < ids.length; i++) if (q.expect.includes(ids[i])) { rank = i + 1; break; }
    cat.hit5 += hit5 ? 1 : 0;
    cat.hit10 += hit10 ? 1 : 0;
    cat.mrr += rank ? 1 / rank : 0;
    if (q.category !== "precision") polluted += top5.filter((id) => OFF_TOPIC.has(id)).length;
    perQuery.push({ q: q.q, category: q.category, condition: cond, hit5, hit10, rank: rank || null });
  }
  out[cond] = cats;
  pollution[cond] = polluted;
}

const pct = (x: number) => `${(x * 100).toFixed(0)}%`;
console.log(`\nkeymem phrase-key bridging ablation — model=${LOCAL_EMBEDDING_MODEL}`);
console.log("─".repeat(62));
console.log(`${"category".padEnd(11)} ${"n".padStart(2)}  ${"metric".padEnd(12)} ${CONDITIONS.map((c) => c.padStart(10)).join("  ")}`);
for (const cat of Object.keys(out["BRIDGE"])) {
  const a = (c: string) => out[c][cat];
  if (cat === "notfound") {
    console.log(`${cat.padEnd(11)} ${String(a("BRIDGE").nf_n).padStart(2)}  ${"not-found".padEnd(12)} ${CONDITIONS.map((c) => `${a(c).nf_ok}/${a(c).nf_n}`.padStart(10)).join("  ")}`);
    continue;
  }
  const metrics: Array<[string, (g: Agg) => string]> = [
    ["reach@10", (x) => pct(x.hit10 / x.n)],
    ["hit@5", (x) => pct(x.hit5 / x.n)],
    ["MRR", (x) => (x.mrr / x.n).toFixed(2)],
  ];
  metrics.forEach(([label, f], i) => {
    console.log(`${(i === 0 ? cat : "").padEnd(11)} ${(i === 0 ? String(a("BRIDGE").n) : "").padStart(2)}  ${label.padEnd(12)} ${CONDITIONS.map((c) => f(a(c)).padStart(10)).join("  ")}`);
  });
  console.log("");
}
console.log("─".repeat(62));
console.log(`links added on load    ${CONDITIONS.map((c) => String(bridgedLinks[c]).padStart(10)).join("  ")}`);
console.log(`off-topic in top-5     ${CONDITIONS.map((c) => String(pollution[c]).padStart(10)).join("  ")}`);

await writeFile(
  resolve("bench/phrase-bridge-results.json"),
  JSON.stringify({ model: LOCAL_EMBEDDING_MODEL, conditions: CONDITIONS, topK: TOPK, aggregates: out, pollution, bridgedLinks, perQuery }, null, 2)
);
console.log("\nresults written to bench/phrase-bridge-results.json");
await rm(dir, { recursive: true, force: true });
