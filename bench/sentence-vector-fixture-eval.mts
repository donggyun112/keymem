// Binary acceptance test for _embedSentences (SENTENCE_VECTORS_ENABLED): does it clear a real
// bar on a held-out "buried fact" fixture, enough to justify its measured ~14.6x write-time
// cost? No opt-in outcome -- if ON doesn't measurably beat OFF here, the feature is deleted.
import { mkdtemp, rm, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

process.env.EMBEDDING_BACKEND ??= "local";
process.env.LOCAL_EMBEDDING_MODEL ??= "bge-m3";

type Mem = { id: string; content: string; keys: string[] };
type Q = { q: string; expect: string[]; category: string };
const fixture = JSON.parse(await readFile(resolve("bench/sentence-vector-fixture.json"), "utf-8")) as {
  memories: Mem[]; queries: Q[];
};

async function run(sentenceVectorsEnabled: boolean) {
  const dir = await mkdtemp(join(tmpdir(), `km-sv-${sentenceVectorsEnabled}-`));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.KEYMEM_SENTENCE_VECTORS = sentenceVectorsEnabled ? "1" : "0";
  process.env.KEYMEM_RERANK_POOL = String(fixture.memories.length);
  const { MemoryGraph } = await import(`../src/memoryGraph.ts?sv-eval=${sentenceVectorsEnabled}`);
  const g: any = new MemoryGraph();
  await g.load();

  const writeStart = performance.now();
  for (const m of fixture.memories) await g.add(m.content, m.keys);
  const writeMs = performance.now() - writeStart;

  let hit5 = 0, hit10 = 0, mrrSum = 0;
  const perQuery: Array<{ q: string; rank: number | null }> = [];
  for (const q of fixture.queries) {
    const res = (await g.recall(q.q, 10, null, false, 1)) as Array<{ content: string }>;
    // Match by content since ids here are graph-internal, not the fixture's logical ids.
    const expectContent = fixture.memories.find((m) => m.id === q.expect[0])?.content;
    let rank: number | null = null;
    for (let i = 0; i < res.length; i++) if (res[i].content === expectContent) { rank = i + 1; break; }
    if (rank !== null && rank <= 5) hit5++;
    if (rank !== null && rank <= 10) hit10++;
    if (rank !== null) mrrSum += 1 / rank;
    perQuery.push({ q: q.q, rank });
  }

  await rm(dir, { recursive: true, force: true });
  return {
    writeMs,
    hit5: hit5 / fixture.queries.length,
    hit10: hit10 / fixture.queries.length,
    mrr: mrrSum / fixture.queries.length,
    perQuery,
  };
}

const on = await run(true);
const off = await run(false);

console.log("\nsentence-vector fixture eval (10 buried-fact queries, 30 memories)");
console.log("─".repeat(60));
console.log(`                  ON        OFF`);
console.log(`Hit@5           ${(on.hit5 * 100).toFixed(0).padStart(3)}%       ${(off.hit5 * 100).toFixed(0).padStart(3)}%`);
console.log(`Hit@10          ${(on.hit10 * 100).toFixed(0).padStart(3)}%       ${(off.hit10 * 100).toFixed(0).padStart(3)}%`);
console.log(`MRR             ${on.mrr.toFixed(2).padStart(4)}      ${off.mrr.toFixed(2).padStart(4)}`);
console.log(`write time    ${on.writeMs.toFixed(0).padStart(6)}ms   ${off.writeMs.toFixed(0).padStart(6)}ms  (30 memories)`);
console.log("─".repeat(60));

console.log("\nper-query rank (ON | OFF):");
for (let i = 0; i < on.perQuery.length; i++) {
  console.log(`  ${on.perQuery[i].q.padEnd(20)} ${String(on.perQuery[i].rank).padStart(4)} | ${String(off.perQuery[i].rank).padStart(4)}`);
}
