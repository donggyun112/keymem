// Tuning sweep: SUPER_MEMORY_CONTENT_RECALL (+ matching minScore) vs.
//   cross-lingual recall  ↑   /   distractor noise  ↑
//
// Run: npx tsx test/contentrecall-sweep.ts
// Requires the bge-m3 model present at local_cache/bge-m3-custom.
//
// CONTENT_RECALL_THRESHOLD is read ONCE at memoryGraph module load, so we
// cache-bust-reimport memoryGraph per sweep value. embedding.ts is NOT
// cache-busted, so the 570MB model loads exactly once and is shared.
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

process.env.EMBEDDING_BACKEND = "local";
process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
process.env.LOCAL_EMBEDDING_MODEL_PATH =
  process.env.LOCAL_EMBEDDING_MODEL_PATH ?? "local_cache/bge-m3-custom";
// Rerank only REORDERS the candidate pool; it cannot change pool membership.
// We measure pool membership (retrieved within top-10), so keep rerank OFF for a
// clean read of what contentRecall lets in.
delete process.env.SUPER_MEMORY_RERANK;

const dataDir = await mkdtemp(join(tmpdir(), "sm-cr-sweep-"));
process.env.SUPER_MEMORY_DATA_DIR = dataDir;

// ── Golden dataset (mixed KR / EN) ──
const GOLDEN: Record<string, { content: string; keys: string[] }> = {
  job: { content: "Jiwoo works as a backend engineer at a fintech company called PayStream, mainly building payment settlement systems in Go.", keys: ["Jiwoo", "job", "PayStream", "backend engineer", "Go"] },
  lang: { content: "Jiwoo's favorite programming language is Rust, but he uses Go at work for historical reasons.", keys: ["Jiwoo", "favorite language", "Rust", "Go", "preference"] },
  commute: { content: "Jiwoo lives in Busan and commutes to the Seoul office twice a week by KTX train.", keys: ["Jiwoo", "Busan", "Seoul", "commute", "KTX"] },
  db: { content: "PayStream migrated its core ledger from PostgreSQL to CockroachDB in 2024 to handle multi-region writes.", keys: ["PayStream", "ledger", "PostgreSQL", "CockroachDB", "multi-region"] },
  allergy: { content: "Jiwoo is allergic to peanuts and always carries an epinephrine auto-injector.", keys: ["Jiwoo", "allergy", "peanuts", "epinephrine", "health"] },
  dog: { content: "Jiwoo adopted a Korean Jindo dog named Bori in 2022; Bori is afraid of thunderstorms.", keys: ["Jiwoo", "dog", "Jindo", "Bori", "pet"] },
  oncall: { content: "The PayStream on-call rotation uses PagerDuty and the SLO for settlement latency is 99.9% under 500ms.", keys: ["PayStream", "on-call", "PagerDuty", "SLO", "latency"] },
  bass: { content: "Jiwoo plays bass guitar in an amateur jazz band that rehearses every Thursday evening.", keys: ["Jiwoo", "bass guitar", "jazz", "band", "hobby"] },
  running: { content: "지우는 매주 토요일 아침에 광안리 해변에서 5킬로미터 러닝을 한다.", keys: ["지우", "러닝", "광안리", "운동", "토요일"] },
  food: { content: "지우가 가장 좋아하는 음식은 부산 돼지국밥이며 매운 다대기를 듬뿍 넣어 먹는다.", keys: ["지우", "음식", "돼지국밥", "부산", "매운맛"] },
  cto: { content: "PayStream's CTO is Mina Park, who previously led infrastructure at a large e-commerce firm.", keys: ["PayStream", "CTO", "Mina Park", "infrastructure"] },
  aws: { content: "Jiwoo is studying for the AWS Solutions Architect Professional exam and plans to take it in the fall.", keys: ["Jiwoo", "AWS", "certification", "Solutions Architect", "exam"] },
};

// ── Build the graph once (default thresholds) ──
const boot = await import("../src/memoryGraph.ts");
const gboot = new boot.MemoryGraph();
await gboot.load();
const ids: Record<string, string> = {};
for (const [label, m] of Object.entries(GOLDEN)) {
  const [id] = await gboot.add(m.content, m.keys);
  ids[label] = id;
}
console.log(`built ${Object.keys(ids).length} memories in ${dataDir}\n`);

// ── Labeled probes ──
// Cross-lingual positives: query language != stored language. Expected memory label.
const POSITIVES: { q: string; expect: string; tag: string }[] = [
  { q: "지우의 직업과 회사는?", expect: "job", tag: "KR→EN job (short)" },
  { q: "지우가 키우는 강아지가 무서워하는 것은?", expect: "dog", tag: "KR→EN dog" },
  { q: "지우의 알레르기와 건강 문제는?", expect: "allergy", tag: "KR→EN allergy" },
  { q: "페이스트림은 어떤 데이터베이스를 사용하나?", expect: "db", tag: "KR→EN database" },
  { q: "What is Jiwoo's favorite food?", expect: "food", tag: "EN→KR food" },
  { q: "What does Jiwoo do for exercise on weekends?", expect: "running", tag: "EN→KR running" },
];
// Pure distractors: NO stored fact, NO known entity → ideal result is [].
const PURE_DISTRACTORS = [
  "양자역학 블랙홀 우주론 상대성이론",
  "best sourdough bread recipe and fermentation technique",
  "비트코인 시세 전망과 장기 투자 전략",
];
// Entity distractors: mention a known entity but ask an unstored attribute.
const ENTITY_DISTRACTORS = ["What car does Jiwoo drive?", "지우의 혈액형은 무엇인가?"];

const SWEEP = [0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3];
const TOPK = 10;

const idToLabel: Record<string, string> = {};
for (const [l, id] of Object.entries(ids)) idToLabel[id] = l;
const labelsOf = (r: any[]) => r.map((x) => idToLabel[x.id] ?? "?");

console.log("T = contentRecall = minScore");
console.log(
  "T".padEnd(6) +
    "xRecall".padEnd(10) +
    "xTop1".padEnd(9) +
    "pureFP".padEnd(9) +
    "entFP".padEnd(8) +
    "detail"
);
console.log("-".repeat(78));

for (const T of SWEEP) {
  process.env.SUPER_MEMORY_CONTENT_RECALL = String(T);
  const mod = await import(`../src/memoryGraph.ts?cr=${T}`);
  const g = new mod.MemoryGraph();
  await g.load();

  let recalled = 0;
  let top1 = 0;
  const missed: string[] = [];
  for (const p of POSITIVES) {
    const r = (await g.recall(p.q, TOPK, null, false, 2, 0, T)) as any[];
    const labels = labelsOf(r);
    if (labels.includes(p.expect)) recalled++;
    else missed.push(p.tag.split(" ")[0]);
    if (labels[0] === p.expect) top1++;
  }

  let pureFP = 0;
  for (const q of PURE_DISTRACTORS) {
    const r = (await g.recall(q, TOPK, null, false, 2, 0, T)) as any[];
    if (r.length > 0) pureFP++;
  }
  let entFP = 0;
  for (const q of ENTITY_DISTRACTORS) {
    const r = (await g.recall(q, TOPK, null, false, 2, 0, T)) as any[];
    if (r.length > 0) entFP++;
  }

  console.log(
    String(T).padEnd(6) +
      `${recalled}/${POSITIVES.length}`.padEnd(10) +
      `${top1}/${POSITIVES.length}`.padEnd(9) +
      `${pureFP}/${PURE_DISTRACTORS.length}`.padEnd(9) +
      `${entFP}/${ENTITY_DISTRACTORS.length}`.padEnd(8) +
      (missed.length ? `miss: ${missed.join(",")}` : "all xling hit")
  );
}

console.log("\nLegend:");
console.log("  xRecall = cross-lingual positives whose correct memory is in top-10");
console.log("  xTop1   = cross-lingual positives where correct memory ranks #1");
console.log("  pureFP  = pure distractors that wrongly returned >=1 result (want 0)");
console.log("  entFP   = entity distractors that returned >=1 result (key-leak; want 0)");

await rm(dataDir, { recursive: true, force: true });
