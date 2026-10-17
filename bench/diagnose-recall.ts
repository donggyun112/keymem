// Why did `recall` return [] for a paraphrase? Shows key cosines and lexical body matches,
// and how (and at what hop) the expand+ungated path finds the target. Read-only on your store:
// it COPIES graph.json into a temp dir and works there, so your real memories are untouched.
//
//   KM_SRC=~/.keymem EMBEDDING_BACKEND=local LOCAL_EMBEDDING_MODEL=bge-m3 \
//     tsx bench/diagnose-recall.ts "your paraphrase query" "another query"
import { mkdtemp, rm, copyFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir, homedir } from "node:os";
import { join } from "node:path";

process.env.KEYMEM_AUTO_MIGRATE = "false"; // don't re-embed the copied graph
const SRC = process.env.KM_SRC || join(homedir(), ".keymem");
const tmp = await mkdtemp(join(tmpdir(), "km-diag-"));
await copyFile(join(SRC, "graph.json"), join(tmp, "graph.json"));
for (const file of ["vectors.bin", "vectors.idx.json"]) {
  if (existsSync(join(SRC, file))) await copyFile(join(SRC, file), join(tmp, file));
}
process.env.KEYMEM_DATA_DIR = tmp;

const { MemoryGraph } = await import("../src/memoryGraph.ts");
const { embedTextAsync, getThresholdProfile, LOCAL_EMBEDDING_MODEL } = await import("../src/embedding.ts");

const cos = (a: number[], b: number[]) => {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return na && nb ? d / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const g: any = new MemoryGraph();
await g.load();
const p = getThresholdProfile();
const queries = process.argv.slice(2);
const KR = p.keyRecall;
const ANON = process.env.KM_ANON === "1"; // suppress memory/key text; show only ids/cosines

console.log(`key model=${LOCAL_EMBEDDING_MODEL}  KEY_RECALL=${KR}; body retrieval=Kiwi BM25`);
console.log(`store: ${Object.keys(g.keys).length} keys, ${Object.keys(g.memories).length} memories\n`);

for (const q of queries) {
  const qe = await embedTextAsync(q);
  const lbl = (text: string, id: string) => (ANON ? `mem(${id.slice(0, 6)})` : text.slice(0, 46));
  const keyHits = Object.values(g.keys)
    .map((k: any) => ({ c: ANON ? "" : k.concept, s: cos(qe, k.embedding), t: k.key_type }))
    .sort((a, b) => b.s - a.s).slice(0, 6);
  const lexical = g._lexicalScores(q) as Map<string, number>;
  const memHits = [...lexical].map(([id, s]) => ({ id, c: g.memories[id]?.content ?? "", s }))
    .sort((a, b) => b.s - a.s).slice(0, 6);

  console.log(`════ query: "${q}"`);
  console.log(`  top keys (query↔key cosine vs KEY_RECALL ${KR}):`);
  for (const k of keyHits) console.log(`    ${k.s.toFixed(3)} ${k.s >= KR ? "✓" : "✗"} [${k.t}]${ANON ? "" : " " + k.c}`);
  console.log("  top memories (normalized BM25 lexical score):");
  for (const m of memHits) console.log(`    ${m.s.toFixed(3)} ${lbl(m.c, m.id)}`);

  const sk = (await g.searchKeys(q, 8, null)) as unknown[];
  console.log(`  → recall (searchKeys) returns: ${sk.length === 0 ? "[]  (gated out)" : sk.length + " key(s)"}`);

  const r = (await g.recall(q, 10, null, true, 2, 0, 0)) as Array<{ id: string; hop?: number; content?: string }>;
  console.log(`  → recall(expand, minScore=0) returns ${r.length}:`);
  for (const m of r.slice(0, 5)) console.log(`      hop=${m.hop ?? "?"}  ${lbl(g.memories[m.id]?.content || m.content || "", m.id)}`);
  console.log("");
}
await rm(tmp, { recursive: true, force: true });
