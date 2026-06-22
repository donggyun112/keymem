// Manual calibration + verification of the distribution not-found gate on real e5.
// Run: EMBEDDING_BACKEND=local npx tsx test/distribution-gate.live.ts
// (Optionally LOCAL_EMBEDDING_MODEL=bge-m3 LOCAL_EMBEDDING_MODEL_PATH=/abs/dir to compare.)
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

process.env.EMBEDDING_BACKEND ??= "local";
const dataDir = await mkdtemp(join(tmpdir(), "sm-distgate-live-"));
process.env.SUPER_MEMORY_DATA_DIR = dataDir;

const { MemoryGraph, robustZScore } = await import("../src/memoryGraph.ts");
const { embedTextAsync, LOCAL_EMBEDDING_MODEL, getThresholdProfile } = await import("../src/embedding.ts");

function cos(a: number[], b: number[]): number {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return d / (Math.sqrt(na) * Math.sqrt(nb));
}

const g = new MemoryGraph();
await g.load();

// Seed a realistic-ish set.
const seed: [string, string[]][] = [
  ["사용자 이름은 동균이다", ["이름", "동균"]],
  ["동균은 고양이 두 마리를 키운다", ["고양이", "동균"]],
  ["동균의 목표는 keymem를 표준 메모리 레이어로 만드는 것", ["keymem", "목표"]],
  ["keymem는 TypeScript로 작성된 LLM 장기 메모리 시스템", ["keymem", "개발"]],
  ["사용자는 커피를 좋아한다", ["커피", "음료"]],
  ["회의는 매주 월요일 오전 10시", ["회의", "일정"]],
  ["프로젝트는 npm에 배포되어 있다", ["배포", "npm"]],
  ["사용자는 서울에 거주한다", ["거주지", "서울"]],
];
for (const [c, k] of seed) await g.add(c, k);

console.log(`model=${LOCAL_EMBEDDING_MODEL}  profile.gateZ=${getThresholdProfile().gateZ}\n`);

// Measure robust-z of the top content sim for FOUND vs NOT-FOUND queries.
const allMems = Object.values((g as any).memories).map((m: any) => m.embedding);
async function topZ(q: string): Promise<number> {
  const qe = await embedTextAsync(q, "query");
  const sims = allMems.map((e: number[]) => cos(qe, e));
  const top = Math.max(...sims);
  return robustZScore(top, sims);
}
const found = ["이름", "고양이", "keymem 목표", "커피"];
const notFound = ["블록체인 합의 알고리즘", "양자역학 블랙홀", "축구 월드컵 결승", "비트코인 시세"];
console.log("FOUND queries (robust-z of top):");
for (const q of found) console.log(`  ${(await topZ(q)).toFixed(2).padStart(6)}  ${q}`);
console.log("NOT-FOUND queries (robust-z of top):");
for (const q of notFound) console.log(`  ${(await topZ(q)).toFixed(2).padStart(6)}  ${q}`);

// End-to-end: with the active e5 gateZ, not-found should be [] and found non-empty.
let pass = 0, fail = 0;
const ck = (n: string, c: boolean, d = "") => { c ? pass++ : fail++; console.log(`  ${c ? "✅" : "❌"} ${n}  ${d}`); };
console.log("\nend-to-end (active profile gateZ):");
for (const q of found) { const r = (await g.recall(q, 5)) as any[]; ck(`found: ${q}`, r.length >= 1, `n=${r.length}`); }
for (const q of notFound) { const r = (await g.recall(q, 5)) as any[]; ck(`not-found: ${q} -> []`, r.length === 0, `n=${r.length}`); }

console.log(`\nRESULT: ${pass} passed, ${fail} failed`);
await rm(dataDir, { recursive: true, force: true });
process.exitCode = fail ? 1 : 0;
