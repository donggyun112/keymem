// Search-quality benchmark (committed regression guard).
//
//   npm run bench                 # uses LOCAL_EMBEDDING_MODEL (default bge-m3, auto-downloads)
//   LOCAL_EMBEDDING_MODEL=fast-multilingual-e5-large npm run bench   # compare a model
//
// Builds a fresh persona graph from bench/fixture.json, runs the labeled queries, and prints
// per-category scores: recall@1/@5 + MRR for answerable queries, and not-found accuracy for
// distractors. Deterministic per (model, fixture) — diff the scorecard across recall changes.
import { mkdtemp, rm, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

process.env.EMBEDDING_BACKEND ??= "local";
process.env.LOCAL_EMBEDDING_MODEL ??= "bge-m3";

type Mem = { id: string; content: string; keys: string[]; key_types?: Record<string, string>; namespace?: string };
type Q = { q: string; expect: string[]; category: string };

const fixture = JSON.parse(await readFile(resolve("bench/fixture.json"), "utf-8")) as { memories: Mem[]; queries: Q[] };

const dir = await mkdtemp(join(tmpdir(), "sm-bench-"));
process.env.SUPER_MEMORY_DATA_DIR = dir;
const { MemoryGraph } = await import("../src/memoryGraph.ts");
const { LOCAL_EMBEDDING_MODEL } = await import("../src/embedding.ts");

const g = new MemoryGraph();
await g.load();
const idMap: Record<string, string> = {}; // fixture id -> graph id
for (const m of fixture.memories) {
  const [gid] = await g.add(m.content, m.keys, { keyTypes: m.key_types ?? null, namespace: m.namespace });
  idMap[m.id] = gid;
}

type Agg = { n: number; r1: number; r5: number; mrr: number; nf_ok: number; nf_n: number };
const cats: Record<string, Agg> = {};
const mk = (): Agg => ({ n: 0, r1: 0, r5: 0, mrr: 0, nf_ok: 0, nf_n: 0 });
const overall = mk();

for (const q of fixture.queries) {
  const res = (await g.recall(q.q, 5)) as Array<{ id: string }>;
  const ids = res.map((r) => r.id);
  const cat = (cats[q.category] ??= mk());
  const expectIds = q.expect.map((e) => idMap[e]);

  if (q.expect.length === 0) {
    const ok = ids.length === 0;
    cat.nf_n++; cat.nf_ok += ok ? 1 : 0; overall.nf_n++; overall.nf_ok += ok ? 1 : 0;
    continue;
  }
  cat.n++; overall.n++;
  const inTop1 = expectIds.includes(ids[0]);
  const top5 = new Set(ids.slice(0, 5));
  const r5 = expectIds.filter((e) => top5.has(e)).length / expectIds.length;
  let fr = 0; for (let i = 0; i < ids.length; i++) if (expectIds.includes(ids[i])) { fr = i + 1; break; }
  for (const a of [cat, overall]) { a.r1 += inTop1 ? 1 : 0; a.r5 += r5; a.mrr += fr ? 1 / fr : 0; }
}

const pct = (x: number) => `${(x * 100).toFixed(0)}%`;
console.log(`\nsuper-memory search-quality benchmark — model=${LOCAL_EMBEDDING_MODEL}`);
console.log("─".repeat(72));
console.log("category        n   recall@1  recall@5   MRR     notfound");
for (const [name, a] of Object.entries(cats)) {
  const ans = a.n > 0 ? `${pct(a.r1 / a.n).padStart(7)}  ${pct(a.r5 / a.n).padStart(7)}  ${(a.mrr / a.n).toFixed(2).padStart(5)}` : "      -        -      -";
  const nf = a.nf_n > 0 ? `${a.nf_ok}/${a.nf_n}` : "-";
  console.log(`${name.padEnd(14)} ${String(a.n + a.nf_n).padStart(2)}   ${ans}   ${nf}`);
}
console.log("─".repeat(72));
const oAns = `${pct(overall.r1 / overall.n).padStart(7)}  ${pct(overall.r5 / overall.n).padStart(7)}  ${(overall.mrr / overall.n).toFixed(2).padStart(5)}`;
console.log(`${"OVERALL".padEnd(14)} ${String(overall.n + overall.nf_n).padStart(2)}   ${oAns}   ${overall.nf_ok}/${overall.nf_n}`);
console.log(`\nanswerable: ${overall.n}  | not-found: ${overall.nf_n}  | recall@1=${pct(overall.r1 / overall.n)}  not-found-acc=${pct(overall.nf_ok / overall.nf_n)}`);

await rm(dir, { recursive: true, force: true });
