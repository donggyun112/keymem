// Manual end-to-end verification on the REAL bge-m3 model (fastembed CUSTOM).
// Run:
//   EMBEDDING_BACKEND=local LOCAL_EMBEDDING_MODEL=bge-m3 \
//   LOCAL_EMBEDDING_MODEL_PATH=/abs/dir/with/model.onnx \
//   npx tsx test/bge-m3-verify.live.ts
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

process.env.EMBEDDING_BACKEND ??= "local";
process.env.LOCAL_EMBEDDING_MODEL ??= "bge-m3";
const dataDir = await mkdtemp(join(tmpdir(), "sm-bgem3-"));
process.env.SUPER_MEMORY_DATA_DIR = dataDir;

const { MemoryGraph } = await import("../src/memoryGraph.ts");
const { embedTextAsync, LOCAL_EMBEDDING_MODEL, getThresholdProfile } = await import(
  "../src/embedding.ts"
);

let pass = 0,
  fail = 0;
const fails: string[] = [];
const check = (name: string, cond: boolean, detail = "") => {
  if (cond) {
    pass++;
    console.log(`  ✅ ${name}${detail ? "  " + detail : ""}`);
  } else {
    fail++;
    fails.push(name);
    console.log(`  ❌ ${name}  ${detail}`);
  }
};

function cos(a: number[], b: number[]): number {
  let d = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return d / (Math.sqrt(na) * Math.sqrt(nb));
}

const p = getThresholdProfile();
console.log(`model=${LOCAL_EMBEDDING_MODEL}`);
console.log(
  `key profile: keyRecall=${p.keyRecall} keyMerge=${p.keyMerge}\n`
);

// ── 1. dimension + no-prefix embedding ──
const v = await embedTextAsync("커피", "passage");
check("bge-m3 embeds to 1024-dim", v.length === 1024, `dim=${v.length}`);

const g = new MemoryGraph();
await g.load();

// ── 2. absolute score gate (relevant vs unrelated) ──
await g.add("사용자는 커피를 좋아한다", ["커피", "음료"]);
await g.add("사용자의 이름은 동건이다", ["이름", "동건"], { keyTypes: { 동건: "name" } });
const rel = (await g.recall("좋아하는 음료", 5)) as any[];
check("gate: relevant query returns hits", rel.length >= 1, `n=${rel.length}`);
const noise = (await g.recall("양자역학 블랙홀 우주론", 5)) as any[];
check("gate: unrelated query returns []", noise.length === 0, `n=${noise.length}`);

// ── 3. short-key A/B defense ──
const a1 = await g.findOrCreateKey("Agent A");
const a2 = await g.findOrCreateKey("Agent A");
const b = await g.findOrCreateKey("Agent B");
const ka = await embedTextAsync("Agent A", "passage");
const kb = await embedTextAsync("Agent B", "passage");
check("short-key: 'Agent A' exact repeat reuses key", a1 === a2);
check(
  "short-key: 'Agent A' vs 'Agent B' stay distinct",
  a1 !== b,
  `cos(A,B)=${cos(ka, kb).toFixed(4)} (would merge at keyMerge=${p.keyMerge} without guard)`
);

// ── 4. exact dedup and no inferred contradiction ──
const [first] = await g.add("프로젝트A는 PostgreSQL을 쓴다", ["프로젝트A", "DB"], { namespace: "dedup" });
const [, exactDup] = await g.add("프로젝트A는 PostgreSQL을 쓴다", ["프로젝트A", "DB"], { namespace: "dedup" });
check("exact normalized repeat is deduplicated", exactDup);
const [second, differentDup] = await g.add("프로젝트A는 MongoDB를 쓴다", ["프로젝트A", "DB"], { namespace: "dedup" });
check("different facts remain distinct", !differentDup && second !== first);
check("body vector absent in every memory", Object.values(g.memories).every(m => !("embedding" in m)));
check("no body-driven contradiction inference", g.memories[second].contradicts.length === 0);

console.log(`\n${"=".repeat(50)}`);
console.log(`RESULT: ${pass} passed, ${fail} failed`);
if (fail) console.log(`FAILED: ${fails.join(", ")}`);
await rm(dataDir, { recursive: true, force: true });
process.exitCode = fail ? 1 : 0;
