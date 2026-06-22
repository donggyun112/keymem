// Associative-recall ablation: isolates the contribution of keymem's key-graph
// traversal by running the SAME engine + SAME real embeddings under two conditions:
//   DIRECT  = recall(expand=false, hops=1)  → 1-hop semantic match, no associative
//             expansion. Approximates a flat semantic store's reach.
//   GRAPH   = recall(expand=true,  hops=2)  → keymem's multi-hop key-graph traversal.
// The only difference is graph expansion, so the metric delta IS the graph's gain.
//
//   tsx bench/ablation.ts            # real bge-m3 (auto-downloads ~570MB first run)
import { mkdtemp, rm, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

process.env.EMBEDDING_BACKEND ??= "local";
process.env.LOCAL_EMBEDDING_MODEL ??= "bge-m3";

type Mem = { id: string; content: string; keys: string[]; key_types?: Record<string, string> };
type Q = { q: string; expect: string[]; category: string };
const fixture = JSON.parse(await readFile(resolve("bench/assoc-fixture.json"), "utf-8")) as { memories: Mem[]; queries: Q[] };

const dir = await mkdtemp(join(tmpdir(), "km-ablation-"));
process.env.KEYMEM_DATA_DIR = dir;
const { MemoryGraph } = await import("../src/memoryGraph.ts");
const { LOCAL_EMBEDDING_MODEL } = await import("../src/embedding.ts");

const g = new MemoryGraph();
await g.load();
const idMap: Record<string, string> = {};
for (const m of fixture.memories) {
  const [gid] = await g.add(m.content, m.keys, { keyTypes: m.key_types ?? null });
  idMap[m.id] = gid;
}

const CONDITIONS = [
  { name: "DIRECT", expand: false, hops: 1 },
  { name: "GRAPH", expand: true, hops: 2 },
];

type Agg = { n: number; hit5: number; hit10: number; mrr: number; nf_ok: number; nf_n: number };
const mk = (): Agg => ({ n: 0, hit5: 0, hit10: 0, mrr: 0, nf_ok: 0, nf_n: 0 });
const TOPK = 10; // request 10 so reach@10 captures graph hits that HOP_DECAY ranks low

const out: Record<string, Record<string, Agg>> = {};
const perQuery: Array<Record<string, unknown>> = [];

for (const cond of CONDITIONS) {
  const cats: Record<string, Agg> = {};
  for (const q of fixture.queries) {
    const res = (await g.recall(q.q, TOPK, null, cond.expand, cond.hops)) as Array<{ id: string }>;
    const ids = res.map((r) => r.id);
    const cat = (cats[q.category] ??= mk());
    const expectIds = q.expect.map((e) => idMap[e]);

    if (q.expect.length === 0) {
      const ok = ids.length === 0;
      cat.nf_n++; cat.nf_ok += ok ? 1 : 0;
      if (cond.name === "GRAPH") perQuery.push({ q: q.q, category: q.category, condition: cond.name, returned: ids.length });
      continue;
    }
    cat.n++;
    const top5 = new Set(ids.slice(0, 5));
    const top10 = new Set(ids.slice(0, 10));
    const hit5 = expectIds.some((e) => top5.has(e));
    const hit10 = expectIds.some((e) => top10.has(e));
    let rank = 0; for (let i = 0; i < ids.length; i++) if (expectIds.includes(ids[i])) { rank = i + 1; break; }
    cat.hit5 += hit5 ? 1 : 0; cat.hit10 += hit10 ? 1 : 0; cat.mrr += rank ? 1 / rank : 0;
    perQuery.push({ q: q.q, category: q.category, condition: cond.name, hit5, hit10, rank: rank || null });
  }
  out[cond.name] = cats;
}

const pct = (x: number) => `${(x * 100).toFixed(0)}%`;
const allCats = [...new Set(fixture.queries.map((q) => q.category))];

console.log(`\nkeymem associative-recall ablation — model=${LOCAL_EMBEDDING_MODEL}`);
console.log(`(DIRECT = expand=false,hops=1 ≈ flat semantic reach;  GRAPH = expand=true,hops=2 = key-graph traversal)`);
console.log("─".repeat(78));
console.log(`category     n   metric        DIRECT     GRAPH      Δ (gain)`);
for (const cat of allCats) {
  const d = out.DIRECT[cat], gph = out.GRAPH[cat];
  if (!d) continue;
  if (cat === "notfound") {
    console.log(`${cat.padEnd(11)} ${String(d.nf_n).padStart(2)}  not-found acc  ${(`${d.nf_ok}/${d.nf_n}`).padStart(7)}    ${(`${gph.nf_ok}/${gph.nf_n}`).padStart(7)}`);
    continue;
  }
  const lines: Array<[string, number, number]> = [
    ["reach@10", d.hit10 / d.n, gph.hit10 / gph.n],
    ["hit@5", d.hit5 / d.n, gph.hit5 / gph.n],
    ["MRR", d.mrr / d.n, gph.mrr / gph.n],
  ];
  lines.forEach(([label, dv, gv], i) => {
    const fmt = (v: number) => (label === "MRR" ? v.toFixed(2) : pct(v));
    const delta = label === "MRR" ? (gv - dv).toFixed(2) : `${((gv - dv) * 100).toFixed(0)}pp`;
    console.log(`${(i === 0 ? cat : "").padEnd(11)} ${(i === 0 ? String(d.n) : "").padStart(2)}  ${label.padEnd(12)}  ${fmt(dv).padStart(6)}    ${fmt(gv).padStart(6)}    ${delta.padStart(7)}`);
  });
  console.log("");
}
console.log("─".repeat(78));

await writeFile(resolve("bench/assoc-results.json"), JSON.stringify({ model: LOCAL_EMBEDDING_MODEL, conditions: CONDITIONS, aggregates: out, perQuery }, null, 2));
console.log("results written to bench/assoc-results.json");
await rm(dir, { recursive: true, force: true });
