// Hebbian association cost/benefit. The unit tests already prove the edge forms and
// traverses; the open question is what turning it on COSTS, because it is the one path
// that can invent a connection the data never had.
//
// Runs the §6 fixture and metrics so the columns are directly comparable, after a
// training phase of realistic use (recall -> confirmed read) that lets associations form.
// HEBBIAN_ENABLED is latched when autokey.ts is first imported, so one process = one
// condition. Run both and diff:
//
//   tsx bench/hebbian.ts                        # OFF (default)
//   KEYMEM_HEBBIAN=true tsx bench/hebbian.ts    # ON
import { mkdtemp, rm, readFile } from "node:fs/promises";
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
const OFF_TOPIC = new Set(["far-budget", "far-hiring"]);

const dir = await mkdtemp(join(tmpdir(), "km-hebb-"));
process.env.KEYMEM_DATA_DIR = dir;
const { MemoryGraph } = await import("../src/memoryGraph.ts");
const { LOCAL_EMBEDDING_MODEL } = await import("../src/embedding.ts");
const { HEBBIAN_ENABLED, HEBBIAN_PROMOTE_N } = await import("../src/autokey.ts");

const g = new MemoryGraph();
await g.load();
const idMap: Record<string, string> = {};
const revMap: Record<string, string> = {};
for (const m of fixture.memories) {
  const [gid] = await g.add(m.content, m.keys);
  idMap[m.id] = gid;
  revMap[gid] = m.id;
}

// Training: realistic use, not a scripted co-occurrence. For each answerable query the
// agent recalls, then confirms by reading the memory it was after — exactly the
// recall -> read_key -> read_memory flow the protocol asks for. Associations form (or
// not) as a side effect, from whichever keys happened to co-match.
const TRAIN_ROUNDS = HEBBIAN_PROMOTE_N;
for (let round = 0; round < TRAIN_ROUNDS; round++) {
  for (const q of fixture.queries) {
    if (q.expect.length === 0) continue;
    await g.searchKeys(q.q, 8, null);
    const gid = idMap[q.expect[0]];
    const refs = g.getKeyRefsForMemory(gid) as Array<{ concept: string; key_id: string }>;
    if (refs.length > 0) await g.readMemory(gid, refs[0].key_id);
  }
}

const TOPK = 10;
type Agg = { n: number; hit5: number; hit10: number; mrr: number; nf_ok: number; nf_n: number };
const mk = (): Agg => ({ n: 0, hit5: 0, hit10: 0, mrr: 0, nf_ok: 0, nf_n: 0 });
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
  cat.hit5 += q.expect.some((e) => top5.includes(e)) ? 1 : 0;
  cat.hit10 += q.expect.some((e) => ids.slice(0, 10).includes(e)) ? 1 : 0;
  let rank = 0;
  for (let i = 0; i < ids.length; i++) if (q.expect.includes(ids[i])) { rank = i + 1; break; }
  cat.mrr += rank ? 1 / rank : 0;
  if (q.category !== "precision") polluted += top5.filter((id) => OFF_TOPIC.has(id)).length;
}

let edges = 0;
let promoted = 0;
for (const key of Object.values(g.keys) as Array<{ assoc?: Record<string, number> }>) {
  for (const count of Object.values(key.assoc ?? {})) {
    edges++;
    if (count >= HEBBIAN_PROMOTE_N) promoted++;
  }
}

const pct = (x: number) => `${(x * 100).toFixed(0)}%`;
console.log(`\nkeymem Hebbian association — model=${LOCAL_EMBEDDING_MODEL} | HEBBIAN=${HEBBIAN_ENABLED ? "ON" : "OFF"}`);
console.log("─".repeat(58));
for (const [cat, a] of Object.entries(cats)) {
  if (cat === "notfound") {
    console.log(`${cat.padEnd(11)} ${String(a.nf_n).padStart(2)}  not-found    ${a.nf_ok}/${a.nf_n}`);
    continue;
  }
  console.log(
    `${cat.padEnd(11)} ${String(a.n).padStart(2)}  reach@10 ${pct(a.hit10 / a.n).padStart(5)}` +
      `  hit@5 ${pct(a.hit5 / a.n).padStart(5)}  MRR ${(a.mrr / a.n).toFixed(2)}`
  );
}
console.log("─".repeat(58));
console.log(`off-topic in top-5    ${polluted}`);
console.log(`assoc pairs accrued   ${edges} (traversable at >=${HEBBIAN_PROMOTE_N}: ${promoted})`);
await rm(dir, { recursive: true, force: true });
