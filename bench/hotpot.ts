// External multi-hop benchmark on HotpotQA (distractor). Each question ships 10 paragraphs
// (2 gold supporting + 8 distractors) and gold supporting-fact TITLES — so we measure
// retrieval recall of the supports with NO LLM judge. Mapping to keymem: each paragraph is a
// memory keyed by its own title + any other paragraph title it mentions in-text — so a "bridge"
// entity (the thing that links the question's paragraph to the answer's paragraph) becomes a
// shared key. For BRIDGE questions the answer paragraph is connected-but-dissimilar to the
// query (the query doesn't mention it) → flat retrieval should miss it, graph traversal reach it.
//
//   tsx bench/hotpot.ts [N]     # default 120 questions; real bge-m3
import { mkdtemp, rm, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import MiniSearch from "minisearch";

process.env.EMBEDDING_BACKEND ??= "local";
process.env.LOCAL_EMBEDDING_MODEL ??= "bge-m3";

type Row = { id: string; question: string; type: string; support: string[]; titles: string[]; paras: string[] };
const all = JSON.parse(await readFile(resolve("bench/hotpot-slice.json"), "utf-8")) as Row[];
const N = Number(process.argv[2]) || 120;
const rows = all.slice(0, N);

const { LOCAL_EMBEDDING_MODEL } = await import("../src/embedding.ts");

const TOPK = 5;
const CONDITIONS = ["BM25", "DIRECT", "GRAPH"];
type Agg = { n: number; recall: number; both: number };
const mk = (): Agg => ({ n: 0, recall: 0, both: 0 });
const out: Record<string, Record<string, Agg>> = {};
for (const c of CONDITIONS) out[c] = { bridge: mk(), comparison: mk(), all: mk() };

const dir = await mkdtemp(join(tmpdir(), "km-hotpot-"));
let done = 0;
for (const r of rows) {
  // Fresh per-question graph: DATA_DIR is captured at module-load, so re-import memoryGraph
  // with a cache-busting query AFTER setting the dir — otherwise every question piles into one
  // graph (contaminating the 10-paragraph distractor protocol).
  process.env.KEYMEM_DATA_DIR = await mkdtemp(join(dir, "q-"));
  const mg = await import(`../src/memoryGraph.ts?h=${done}`);
  const g = new mg.MemoryGraph();
  await g.load();
  const gidToTitle: Record<string, string> = {};
  for (let i = 0; i < r.titles.length; i++) {
    const title = r.titles[i];
    const text = r.paras[i] ?? "";
    const lc = text.toLowerCase();
    const keys = [title, ...r.titles.filter((t) => t !== title && lc.includes(t.toLowerCase()))];
    const [gid] = await g.add(text, keys, {});
    gidToTitle[gid] = title;
  }
  const bm25 = new MiniSearch({ fields: ["content"], storeFields: ["id"], idField: "id" });
  bm25.addAll(r.titles.map((t, i) => ({ id: t, content: r.paras[i] ?? "" })));

  for (const cond of CONDITIONS) {
    let titles: string[];
    if (cond === "BM25") {
      titles = (bm25.search(r.question, { fuzzy: 0.2, prefix: true }) as Array<{ id: string }>).map((x) => x.id);
    } else {
      const res = await g.recall(r.question, TOPK, null, cond === "GRAPH", cond === "GRAPH" ? 2 : 1, 0, 0);
      titles = res.map((x) => gidToTitle[x.id]).filter(Boolean);
    }
    const top = titles.slice(0, TOPK);
    const found = r.support.filter((s) => top.includes(s)).length;
    for (const cat of [r.type, "all"]) {
      const a = out[cond][cat]; a.n++; a.recall += found / r.support.length; a.both += found === r.support.length ? 1 : 0;
    }
  }
  done++;
}
await rm(dir, { recursive: true, force: true });

const pct = (x: number) => `${(x * 100).toFixed(0)}%`;
console.log(`\nkeymem × HotpotQA (distractor) — model=${LOCAL_EMBEDDING_MODEL}, N=${done} questions`);
console.log(`support-recall@${TOPK} (fraction of gold paragraphs retrieved) / both@${TOPK} (got BOTH golds)`);
console.log("─".repeat(70));
console.log(`category        n    metric          BM25    DIRECT    GRAPH`);
for (const cat of ["bridge", "comparison", "all"]) {
  const a = (c: string) => out[c][cat];
  const n = a("GRAPH").n; if (!n) continue;
  console.log(`${cat.padEnd(13)} ${String(n).padStart(3)}   support-recall  ${CONDITIONS.map((c) => pct(a(c).recall / a(c).n).padStart(6)).join("  ")}`);
  console.log(`${"".padEnd(13)} ${"".padStart(3)}   both@${TOPK}          ${CONDITIONS.map((c) => pct(a(c).both / a(c).n).padStart(6)).join("  ")}`);
  console.log("");
}
console.log("─".repeat(70));
await writeFile(resolve("bench/hotpot-results.json"), JSON.stringify({ model: LOCAL_EMBEDDING_MODEL, N: done, topK: TOPK, aggregates: out }, null, 2));
console.log("results → bench/hotpot-results.json");
