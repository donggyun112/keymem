// Associative-recall comparison. Same data, same real embeddings (bge-m3), three retrievers:
//   BM25    = standalone lexical search over memory content (MiniSearch) — the classic
//             "flat keyword store" baseline many simple memory layers use.
//   DIRECT  = keymem recall(expand=false, hops=1) — 1-hop semantic match, no graph expansion.
//   GRAPH   = keymem recall(expand=true,  hops=2) — multi-hop key-graph traversal.
// BM25 and DIRECT are the flat baselines; GRAPH adds graph traversal. The metric delta isolates
// what the key-graph buys over flat lexical / flat semantic retrieval.
//
//   tsx bench/ablation.ts            # real bge-m3 (auto-downloads ~570MB first run)
import { mkdtemp, rm, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import MiniSearch from "minisearch";

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
const idMap: Record<string, string> = {};      // logical -> graph id
const revMap: Record<string, string> = {};      // graph id -> logical
for (const m of fixture.memories) {
  const [gid] = await g.add(m.content, m.keys, { keyTypes: m.key_types ?? null });
  idMap[m.id] = gid; revMap[gid] = m.id;
}

// Standalone BM25 baseline (lexical, no embeddings, no graph).
const bm25 = new MiniSearch({ fields: ["content"], storeFields: ["id"], idField: "id" });
bm25.addAll(fixture.memories.map((m) => ({ id: m.id, content: m.content })));

const TOPK = 10;
// Each retriever returns ranked LOGICAL ids so all conditions score against q.expect uniformly.
async function retrieve(cond: string, query: string): Promise<string[]> {
  if (cond === "BM25") {
    return (bm25.search(query, { fuzzy: 0.2, prefix: true }) as Array<{ id: string }>).slice(0, TOPK).map((r) => r.id);
  }
  const expand = cond === "GRAPH";
  const hops = cond === "GRAPH" ? 2 : 1;
  const res = (await g.recall(query, TOPK, null, expand, hops)) as Array<{ id: string }>;
  return res.map((r) => revMap[r.id]).filter(Boolean);
}

const CONDITIONS = ["BM25", "DIRECT", "GRAPH"];
type Agg = { n: number; hit5: number; hit10: number; mrr: number; nf_ok: number; nf_n: number };
const mk = (): Agg => ({ n: 0, hit5: 0, hit10: 0, mrr: 0, nf_ok: 0, nf_n: 0 });
const out: Record<string, Record<string, Agg>> = {};
const perQuery: Array<Record<string, unknown>> = [];

for (const cond of CONDITIONS) {
  const cats: Record<string, Agg> = {};
  for (const q of fixture.queries) {
    const ids = await retrieve(cond, q.q);
    const cat = (cats[q.category] ??= mk());
    if (q.expect.length === 0) {
      cat.nf_n++; cat.nf_ok += ids.length === 0 ? 1 : 0;
      continue;
    }
    cat.n++;
    const top5 = new Set(ids.slice(0, 5));
    const top10 = new Set(ids.slice(0, 10));
    const hit5 = q.expect.some((e) => top5.has(e));
    const hit10 = q.expect.some((e) => top10.has(e));
    let rank = 0; for (let i = 0; i < ids.length; i++) if (q.expect.includes(ids[i])) { rank = i + 1; break; }
    cat.hit5 += hit5 ? 1 : 0; cat.hit10 += hit10 ? 1 : 0; cat.mrr += rank ? 1 / rank : 0;
    perQuery.push({ q: q.q, category: q.category, condition: cond, hit5, hit10, rank: rank || null });
  }
  out[cond] = cats;
}

const pct = (x: number) => `${(x * 100).toFixed(0)}%`;
const allCats = [...new Set(fixture.queries.map((q) => q.category))];

console.log(`\nkeymem associative-recall comparison — model=${LOCAL_EMBEDDING_MODEL}`);
console.log(`BM25=flat lexical · DIRECT=keymem 1-hop semantic · GRAPH=keymem key-graph traversal`);
console.log("─".repeat(74));
console.log(`category     n   metric        BM25    DIRECT    GRAPH`);
for (const cat of allCats) {
  const a = (c: string) => out[c][cat];
  if (cat === "notfound") {
    console.log(`${cat.padEnd(11)} ${String(a("GRAPH").nf_n).padStart(2)}  not-found acc ${CONDITIONS.map((c) => `${a(c).nf_ok}/${a(c).nf_n}`.padStart(7)).join("  ")}`);
    continue;
  }
  const metrics: Array<[string, (g: Agg) => string]> = [
    ["reach@10", (x) => pct(x.hit10 / x.n)],
    ["hit@5", (x) => pct(x.hit5 / x.n)],
    ["MRR", (x) => (x.mrr / x.n).toFixed(2)],
  ];
  metrics.forEach(([label, f], i) => {
    console.log(`${(i === 0 ? cat : "").padEnd(11)} ${(i === 0 ? String(a("GRAPH").n) : "").padStart(2)}  ${label.padEnd(12)} ${CONDITIONS.map((c) => f(a(c)).padStart(6)).join("   ")}`);
  });
  console.log("");
}
console.log("─".repeat(74));

await writeFile(resolve("bench/assoc-results.json"), JSON.stringify({ model: LOCAL_EMBEDDING_MODEL, conditions: CONDITIONS, topK: TOPK, aggregates: out, perQuery }, null, 2));
console.log("results written to bench/assoc-results.json");
await rm(dir, { recursive: true, force: true });
