# Keymem Recall Hit-Rate V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve keymem's first injected-memory accuracy through measured adaptive graph expansion and evidence-aware abstention without increasing false injections or runtime-model cost.

**Architecture:** Add a portable quality harness over the checked-in bge-m3 graph, then isolate retrieval policy in a pure module. Test conditional graph expansion and injection gating as separate hypotheses, tune only on a declared tune split, evaluate once on a memory-disjoint holdout, and retain only hypotheses that clear every guardrail.

**Tech Stack:** TypeScript, Node.js 20+, `node:test` through `tsx`, existing `MemoryGraph`, MiniSearch/BM25, local bge-m3 embeddings; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-04-recall-hit-rate-v2-design.md`

## Global Constraints

- Preserve all public MCP schemas and the normal `recall` result shape.
- Keep passive injection non-reinforcing.
- Do not enable the optional cross-encoder by default or add a runtime-trained model.
- Tune policy only on cases marked `tune`; do not inspect per-case holdout results before selecting the winning profile.
- Split positive paraphrases by target memory so one fact never appears in both tune and holdout.
- Adopt a candidate only when positive inject hit@1 improves, recall hit@3 does not fall, negative false-inject rate does not rise, associative/comparison controls do not regress, and median-of-five p95 latency remains within 10% of baseline.
- If baseline positive inject hit@1 is already 100%, preserve it while improving recall hit@3 or negative false-inject rate.
- Route file reads, searches, tests, and benchmark commands through lean-ctx.

---

### Task 1: Portable recall-quality baseline

**Files:**
- Create: `bench/recall-quality-fixture.json`
- Create: `bench/recall-quality-lib.ts`
- Create: `bench/recall-quality.ts`
- Create: `test/recall-quality-lib.test.ts`
- Create after measurement: `bench/recall-quality-baseline.json`

**Interfaces:**
- Consumes: checked-in `eval/real-graph.bgem3.json` and `MemoryGraph.recall()` / `MemoryGraph.recallInject()`.
- Produces: `QualityCase`, `CaseResult`, `QualityMetrics`, `scoreCase()`, and `aggregateQuality()` from `bench/recall-quality-lib.ts`; JSON baseline used by Tasks 4–6.

- [ ] **Step 1: Add the labeled fixture before changing retrieval behavior**

Create `bench/recall-quality-fixture.json` with exact, memory-disjoint tune/holdout labels:

```json
{
  "graph": "eval/real-graph.bgem3.json",
  "cases": [
    {"id":"name-ko","split":"tune","kind":"positive","query":"이름","context":"내 이름이 뭐였지?","namespace":"default","expect":"사용자 이름은 동균이다"},
    {"id":"coffee-ko","split":"tune","kind":"positive","query":"커피 취향","context":"내가 평소 어떤 커피를 마시지?","namespace":"default","expect":"아이스 아메리카노"},
    {"id":"language-ko","split":"tune","kind":"positive","query":"주력 언어","context":"내가 주력으로 쓰는 프로그래밍 언어가 뭐지?","namespace":"default","expect":"TypeScript를 주력"},
    {"id":"food-ko","split":"tune","kind":"positive","query":"좋아하는 음식","context":"내가 좋아하는 음식이 뭐였지?","namespace":"default","expect":"마라탕과 초밥"},
    {"id":"worktime-en","split":"tune","kind":"positive","query":"work time","context":"When does the user prefer to work?","namespace":"default","expect":"밤에 작업"},
    {"id":"missing-phone","split":"tune","kind":"negative","query":"전화번호","context":"내 전화번호가 뭐였지?","namespace":"default"},
    {"id":"missing-birthday","split":"tune","kind":"negative","query":"생일","context":"사용자 생일이 언제야?","namespace":"default"},
    {"id":"missing-company","split":"tune","kind":"negative","query":"회사 이름","context":"Which company does the user work for?","namespace":"default"},
    {"id":"missing-movie","split":"tune","kind":"negative","query":"favorite movie","context":"What is the user's favorite movie?","namespace":"default"},
    {"id":"missing-wallet","split":"tune","kind":"negative","query":"crypto wallet","context":"사용자의 암호화폐 지갑 주소는?","namespace":"default"},

    {"id":"residence-ko","split":"holdout","kind":"positive","query":"거주지","context":"지금 어디에 살고 있지?","namespace":"default","expect":"서울 성수동"},
    {"id":"cats-en","split":"holdout","kind":"positive","query":"cat names","context":"What are the user's cats called?","namespace":"default","expect":"콩(수컷)과 팥(암컷)"},
    {"id":"project-en","split":"holdout","kind":"positive","query":"current project","context":"What project is the user currently developing?","namespace":"default","expect":"super-memory 프로젝트를 개발"},
    {"id":"exercise-ko","split":"holdout","kind":"positive","query":"운동 취향","context":"사용자가 선호하는 운동은 뭐야?","namespace":"default","expect":"산책 정도"},
    {"id":"goal-en","split":"holdout","kind":"positive","query":"super-memory goal","context":"What is the user's goal for super-memory?","namespace":"default","expect":"Claude 생태계 표준 메모리 레이어"},
    {"id":"missing-email","split":"holdout","kind":"negative","query":"이메일","context":"사용자의 이메일 주소가 뭐야?","namespace":"default"},
    {"id":"missing-weather","split":"holdout","kind":"negative","query":"weather","context":"What weather does the user prefer?","namespace":"default"},
    {"id":"missing-deploy","split":"holdout","kind":"negative","query":"배포 일정","context":"다음 배포 일정은 언제야?","namespace":"default"},
    {"id":"missing-school","split":"holdout","kind":"negative","query":"학교","context":"사용자가 나온 학교는 어디야?","namespace":"default"},
    {"id":"missing-color","split":"holdout","kind":"negative","query":"favorite color","context":"What is the user's favorite color?","namespace":"default"}
  ]
}
```

- [ ] **Step 2: Write failing metric tests**

Create `test/recall-quality-lib.test.ts`:

```typescript
import assert from "node:assert/strict";
import test from "node:test";
import { aggregateQuality, scoreCase, type CaseResult, type QualityCase } from "../bench/recall-quality-lib.ts";

const positive: QualityCase = {
  id: "p", split: "tune", kind: "positive", query: "coffee",
  context: "what coffee", namespace: "default", expect: "americano",
};
const negative: QualityCase = {
  id: "n", split: "tune", kind: "negative", query: "phone",
  context: "what phone", namespace: "default",
};

test("scoreCase distinguishes inject hit, recall hit, and false injection", () => {
  assert.deepEqual(scoreCase(positive, ["iced americano"], ["iced americano"]), {
    injectHit: true, recallHit3: true, falseInject: false, injectCovered: true,
  });
  assert.deepEqual(scoreCase(negative, ["unrelated"], []), {
    injectHit: false, recallHit3: false, falseInject: true, injectCovered: true,
  });
});

test("aggregateQuality reports denominators explicitly", () => {
  const rows: CaseResult[] = [
    { id: "p", split: "tune", kind: "positive", injectHit: true, recallHit3: true, falseInject: false, injectCovered: true, latencyMs: 10 },
    { id: "n", split: "tune", kind: "negative", injectHit: false, recallHit3: false, falseInject: true, injectCovered: true, latencyMs: 20 },
  ];
  const m = aggregateQuality(rows);
  assert.equal(m.positiveInjectHit1, 1);
  assert.equal(m.positiveRecallHit3, 1);
  assert.equal(m.negativeFalseInjectRate, 1);
  assert.equal(m.injectCoverage, 1);
  assert.equal(m.p95LatencyMs, 20);
});
```

- [ ] **Step 3: Run the metric tests and verify RED**

Run: `pnpm exec tsx --test test/recall-quality-lib.test.ts`

Expected: FAIL because `bench/recall-quality-lib.ts` does not exist.

- [ ] **Step 4: Implement the metric library**

Create `bench/recall-quality-lib.ts`:

```typescript
export type QualitySplit = "tune" | "holdout";
export type QualityKind = "positive" | "negative";

export interface QualityCase {
  id: string;
  split: QualitySplit;
  kind: QualityKind;
  query: string;
  context: string;
  namespace: string;
  expect?: string;
}

export interface CaseScore {
  injectHit: boolean;
  recallHit3: boolean;
  falseInject: boolean;
  injectCovered: boolean;
}

export interface CaseResult extends CaseScore {
  id: string;
  split: QualitySplit;
  kind: QualityKind;
  latencyMs: number;
}

export interface QualityMetrics {
  positives: number;
  negatives: number;
  positiveInjectHit1: number;
  positiveRecallHit3: number;
  negativeFalseInjectRate: number;
  injectCoverage: number;
  p50LatencyMs: number;
  p95LatencyMs: number;
}

export function scoreCase(c: QualityCase, injected: string[], recalled: string[]): CaseScore {
  const injectCovered = injected.length > 0;
  const expected = c.expect ?? "";
  return {
    injectHit: c.kind === "positive" && injected[0]?.includes(expected) === true,
    recallHit3: c.kind === "positive" && recalled.slice(0, 3).some((text) => text.includes(expected)),
    falseInject: c.kind === "negative" && injectCovered,
    injectCovered,
  };
}

const quantile = (xs: number[], q: number): number => {
  if (xs.length === 0) return 0;
  const sorted = [...xs].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * q) - 1)];
};

export function aggregateQuality(rows: CaseResult[]): QualityMetrics {
  const pos = rows.filter((r) => r.kind === "positive");
  const neg = rows.filter((r) => r.kind === "negative");
  const ratio = (n: number, d: number) => d === 0 ? 0 : n / d;
  const latencies = rows.map((r) => r.latencyMs);
  return {
    positives: pos.length,
    negatives: neg.length,
    positiveInjectHit1: ratio(pos.filter((r) => r.injectHit).length, pos.length),
    positiveRecallHit3: ratio(pos.filter((r) => r.recallHit3).length, pos.length),
    negativeFalseInjectRate: ratio(neg.filter((r) => r.falseInject).length, neg.length),
    injectCoverage: ratio(rows.filter((r) => r.injectCovered).length, rows.length),
    p50LatencyMs: quantile(latencies, 0.50),
    p95LatencyMs: quantile(latencies, 0.95),
  };
}
```

- [ ] **Step 5: Run the metric tests and verify GREEN**

Run: `pnpm exec tsx --test test/recall-quality-lib.test.ts`

Expected: 2 tests PASS.

- [ ] **Step 6: Implement the baseline runner**

Create `bench/recall-quality.ts`. It must:

```typescript
import { copyFile, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { performance } from "node:perf_hooks";
import { aggregateQuality, scoreCase, type CaseResult, type QualityCase } from "./recall-quality-lib.ts";

process.env.EMBEDDING_BACKEND = "local";
process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
process.env.KEYMEM_AUTO_MIGRATE = "false";

const fixturePath = resolve("bench/recall-quality-fixture.json");
const fixture = JSON.parse(await readFile(fixturePath, "utf8")) as { graph: string; cases: QualityCase[] };
let qualityRun = 0;
const splitArg = process.argv.find((x) => x.startsWith("--split="))?.split("=")[1] ?? "tune";
if (splitArg !== "tune" && splitArg !== "holdout") throw new Error(`invalid split: ${splitArg}`);
const outArg = process.argv.find((x) => x.startsWith("--out="))?.slice("--out=".length);
const repeat = Number(process.argv.find((x) => x.startsWith("--repeat="))?.split("=")[1] ?? "1");
if (!Number.isInteger(repeat) || repeat < 1 || repeat > 10) throw new Error(`invalid repeat: ${repeat}`);
const dir = await mkdtemp(join(tmpdir(), "keymem-quality-"));
await copyFile(resolve(fixture.graph), join(dir, "graph.json"));
process.env.KEYMEM_DATA_DIR = dir;

const { MemoryGraph } = await import(`../src/memoryGraph.ts?quality=${qualityRun++}`);
const graph = new MemoryGraph();
await graph.load();
const rows: CaseResult[] = [];
try {
  for (let run = 0; run < repeat; run++) {
    for (const c of fixture.cases.filter((x) => x.split === splitArg)) {
      const started = performance.now();
      const injected = await graph.recallInject(c.query, 1, c.namespace, {}, c.context) as { memories: Array<{ content: string }> };
      const recalled = await graph.recall(c.query, 3, c.namespace, false, 1, 0, undefined, undefined, undefined, 0, false, c.context) as Array<{ content: string }>;
      const score = scoreCase(c, injected.memories.map((m) => m.content), recalled.map((m) => m.content));
      rows.push({ id: `${c.id}#${run + 1}`, split: c.split, kind: c.kind, ...score, latencyMs: performance.now() - started });
    }
  }
} finally {
  await rm(dir, { recursive: true, force: true });
}
const report = { model: "bge-m3", split: splitArg, repeat, rows, metrics: aggregateQuality(rows) };
console.log(JSON.stringify(report, null, 2));
if (outArg) {
  await writeFile(resolve(outArg), `${JSON.stringify(report, null, 2)}\n`);
}
```

- [ ] **Step 7: Type-check and capture the tune baseline**

Run: `pnpm run build`

Expected: PASS.

Run: `pnpm exec tsx bench/recall-quality.ts --split=tune --out=bench/recall-quality-baseline.json`

Expected: exits 0 and writes per-case results plus `positiveInjectHit1`, `positiveRecallHit3`, `negativeFalseInjectRate`, coverage, p50, and p95. Do not run holdout yet.

- [ ] **Step 8: Commit the frozen labels and baseline**

```bash
git add bench/recall-quality-fixture.json bench/recall-quality-lib.ts bench/recall-quality.ts bench/recall-quality-baseline.json test/recall-quality-lib.test.ts
git commit -m "bench: add recall quality baseline"
```

---

### Task 2: Pure adaptive-recall policy

**Files:**
- Create: `src/recallPolicy.ts`
- Create: `test/recall-policy.test.ts`

**Interfaces:**
- Produces: `RecallPolicyProfile`, `DirectEvidenceSummary`, `InjectEvidence`, `PolicyDecision`, `RecallPolicyTrace`, `DEFAULT_RECALL_POLICY`, `signalsFromMatchedVia()`, `shouldExpand()`, and `shouldInject()`.
- Consumed by: `src/memoryGraph.ts` and calibration code in later tasks.

- [ ] **Step 1: Write failing pure-policy tests**

Create `test/recall-policy.test.ts`:

```typescript
import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_RECALL_POLICY, shouldExpand, shouldInject, signalsFromMatchedVia,
  type DirectEvidenceSummary,
} from "../src/recallPolicy.ts";

const strong: DirectEvidenceSummary = {
  definiteAnchor: false,
  contentGate: 0.55,
  topRankScore: 0.032,
  runnerUpRankScore: 0.020,
  topRelevanceScore: 0.70,
  topSignals: ["key", "content"],
  candidateCount: 3,
};

test("matched_via is normalized into independent evidence kinds", () => {
  assert.deepEqual(signalsFromMatchedVia(["coffee", "(content)", "(bm25)", "person(via)"]), ["key", "content", "bm25", "graph"]);
});

test("strong, clearly separated direct evidence skips graph expansion", () => {
  assert.deepEqual(shouldExpand(strong, DEFAULT_RECALL_POLICY), { accept: false, reason: "strong_direct" });
});

test("weak or ambiguous direct evidence keeps graph expansion", () => {
  assert.equal(shouldExpand({ ...strong, topRelevanceScore: 0.56 }, DEFAULT_RECALL_POLICY).accept, true);
  assert.equal(shouldExpand({ ...strong, runnerUpRankScore: 0.031 }, DEFAULT_RECALL_POLICY).accept, true);
});

test("inject rejects BM25-only and structured-token mismatches", () => {
  const base = {
    definiteAnchor: false, contentGate: 0.55, anchorRelevanceScore: 0.70,
    rankScore: 0.032, runnerUpRankScore: 0.020, relevanceScore: 0.70,
    hop: 1, matchedVia: ["(bm25)"], structuredCoverage: true, lexicalCoverage: true,
  };
  assert.equal(shouldInject(base, DEFAULT_RECALL_POLICY).accept, false);
  assert.equal(shouldInject({ ...base, matchedVia: ["coffee", "(content)"], structuredCoverage: false }, DEFAULT_RECALL_POLICY).accept, false);
});

test("inject accepts strong direct evidence and anchored graph evidence", () => {
  const direct = {
    definiteAnchor: false, contentGate: 0.55, anchorRelevanceScore: 0.70,
    rankScore: 0.032, runnerUpRankScore: 0.020, relevanceScore: 0.70,
    hop: 1, matchedVia: ["coffee", "(content)"], structuredCoverage: true, lexicalCoverage: true,
  };
  assert.equal(shouldInject(direct, DEFAULT_RECALL_POLICY).accept, true);
  assert.equal(shouldInject({ ...direct, hop: 2, relevanceScore: 0.10, matchedVia: ["person(via)"] }, DEFAULT_RECALL_POLICY).accept, true);
});
```

- [ ] **Step 2: Run the policy tests and verify RED**

Run: `pnpm exec tsx --test test/recall-policy.test.ts`

Expected: FAIL because `src/recallPolicy.ts` does not exist.

- [ ] **Step 3: Implement the pure policy module**

Create `src/recallPolicy.ts` with these exact public types and deterministic rules:

```typescript
export type RetrievalSignal = "bm25" | "content" | "key" | "graph";

export interface RecallPolicyProfile {
  directGateRatio: number;
  directMarginRatio: number;
  directMinSignals: number;
  injectGateRatio: number;
  injectMarginRatio: number;
  injectMinSignals: number;
}

export interface DirectEvidenceSummary {
  definiteAnchor: boolean;
  contentGate: number;
  topRankScore: number;
  runnerUpRankScore: number;
  topRelevanceScore: number;
  topSignals: RetrievalSignal[];
  candidateCount: number;
}

export interface InjectEvidence {
  definiteAnchor: boolean;
  contentGate: number;
  anchorRelevanceScore: number;
  rankScore: number;
  runnerUpRankScore: number;
  relevanceScore: number;
  hop: number;
  matchedVia: string[];
  structuredCoverage: boolean;
  lexicalCoverage: boolean;
}

export interface PolicyDecision { accept: boolean; reason: string }

export interface RecallPolicyTrace {
  direct?: DirectEvidenceSummary;
  expanded?: boolean;
  expansionReason?: string;
  injectAccepted?: boolean;
  injectReason?: string;
}

export const DEFAULT_RECALL_POLICY: RecallPolicyProfile = {
  directGateRatio: 1.15,
  directMarginRatio: 0.10,
  directMinSignals: 2,
  injectGateRatio: 1.05,
  injectMarginRatio: 0.05,
  injectMinSignals: 1,
};

export function signalsFromMatchedVia(values: string[]): RetrievalSignal[] {
  const out: RetrievalSignal[] = [];
  for (const value of values) {
    const kind: RetrievalSignal = value === "(bm25)" ? "bm25"
      : value === "(content)" ? "content"
      : value === "(linked)" || value.endsWith("(via)") ? "graph"
      : "key";
    if (!out.includes(kind)) out.push(kind);
  }
  return out;
}

const marginRatio = (top: number, runnerUp: number): number =>
  top <= 0 ? 0 : Math.max(0, (top - runnerUp) / top);

export function shouldExpand(s: DirectEvidenceSummary, p: RecallPolicyProfile): PolicyDecision {
  if (s.definiteAnchor) return { accept: false, reason: "definite_anchor" };
  const independent = s.topSignals.filter((x) => x !== "bm25" && x !== "graph").length;
  const strong = s.candidateCount > 0
    && s.topRelevanceScore >= s.contentGate * p.directGateRatio
    && marginRatio(s.topRankScore, s.runnerUpRankScore) >= p.directMarginRatio
    && independent >= p.directMinSignals;
  return strong
    ? { accept: false, reason: "strong_direct" }
    : { accept: true, reason: s.candidateCount === 0 ? "no_direct_candidate" : "weak_or_ambiguous_direct" };
}

export function shouldInject(e: InjectEvidence, p: RecallPolicyProfile): PolicyDecision {
  if (!e.structuredCoverage) return { accept: false, reason: "structured_token_mismatch" };
  const signals = signalsFromMatchedVia(e.matchedVia);
  const supported = signals.filter((x) => x !== "bm25").length;
  if (supported === 0) return { accept: false, reason: "bm25_only" };
  if (marginRatio(e.rankScore, e.runnerUpRankScore) < p.injectMarginRatio) {
    return { accept: false, reason: "ambiguous_margin" };
  }
  const graphSupported = e.hop > 1 && signals.includes("graph")
    && e.anchorRelevanceScore >= e.contentGate;
  const directSupported = e.hop === 1
    && (e.definiteAnchor || e.relevanceScore >= e.contentGate * p.injectGateRatio)
    && (e.lexicalCoverage || supported >= p.injectMinSignals + 1);
  return directSupported || graphSupported
    ? { accept: true, reason: graphSupported ? "anchored_graph" : "strong_direct" }
    : { accept: false, reason: "insufficient_evidence" };
}
```

- [ ] **Step 4: Run focused and full tests**

Run: `pnpm exec tsx --test test/recall-policy.test.ts`

Expected: 5 tests PASS.

Run: `pnpm run build`

Expected: PASS.

- [ ] **Step 5: Commit the isolated policy**

```bash
git add src/recallPolicy.ts test/recall-policy.test.ts
git commit -m "feat(recall): add testable retrieval policy"
```

---

### Task 3: H1 conditional graph-expansion experiment

**Files:**
- Modify: `src/memoryGraph.ts:2109-2171,2175-2505`
- Create: `test/adaptive-recall.test.ts`
- Modify: `bench/recall-quality.ts`

**Interfaces:**
- Adds internal-only `RecallPolicyMode = "legacy" | "adaptive-expand" | "adaptive"` and `RecallPolicyTrace`. `adaptive-expand` tests H1 alone; `adaptive` enables H1 and H2 together.
- Extends the final `MemoryGraph.recall()` argument with `internal?: { adaptiveExpand?: boolean; profile?: RecallPolicyProfile; trace?: RecallPolicyTrace }`.
- Extends `recallInject()` options with the same internal `policyMode`, `profile`, and `trace`; MCP handlers do not expose these fields.

- [ ] **Step 1: Write a failing deterministic integration test**

Create `test/adaptive-recall.test.ts` using the repository's `__setTestEmbedder` pattern. The fixture must create a direct memory and a graph-only neighbor joined by a shared key. Use an explicit permissive/strict profile in the test so it does not depend on the calibrated default:

```typescript
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import type { RecallPolicyProfile } from "../src/recallPolicy.ts";

let n = 0;
const vec = (text: string): number[] => {
  const t = text.toLowerCase();
  if (t.includes("directq") || t.includes("direct fact")) return [1, 0, 0, 0];
  if (t.includes("weakq")) return [0.58, 0.42, 0, 0];
  if (t.includes("bridge")) return [0, 1, 0, 0];
  if (t.includes("associated")) return [0, 0, 1, 0];
  return [0, 0, 0, 1];
};
const profile: RecallPolicyProfile = {
  directGateRatio: 1.05, directMarginRatio: 0, directMinSignals: 1,
  injectGateRatio: 1, injectMarginRatio: 0, injectMinSignals: 1,
};

test("adaptive mode skips graph neighbors for a strong direct anchor", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-adaptive-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder(vec);
  t.after(() => embedding.__clearTestEmbedder());
  const { MemoryGraph } = await import(`../src/memoryGraph.ts?adaptive=${n++}`);
  const g = new MemoryGraph();
  await g.load();
  await g.add("direct fact bridge", ["direct", "bridge"], {});
  const [neighbor] = await g.add("associated memory bridge", ["associated", "bridge"], {});
  const trace: { expanded?: boolean } = {};
  const out = await g.recall("directq", 5, null, true, 2, 0, 0, 0, 0, 0, false, null, { adaptiveExpand: true, profile, trace }) as Array<{ id: string; hop: number }>;
  assert.equal(trace.expanded, false);
  assert.ok(!out.some((m) => m.id === neighbor && m.hop > 1));
});

test("adaptive mode keeps expansion when direct evidence is weak", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-adaptive-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder(vec);
  t.after(() => embedding.__clearTestEmbedder());
  const { MemoryGraph } = await import(`../src/memoryGraph.ts?adaptive=${n++}`);
  const g = new MemoryGraph();
  await g.load();
  await g.add("direct fact bridge", ["direct", "bridge"], {});
  const [neighbor] = await g.add("associated memory bridge", ["associated", "bridge"], {});
  const strict = { ...profile, directGateRatio: 2 };
  const trace: { expanded?: boolean } = {};
  const out = await g.recall("weakq", 5, null, true, 2, 0, 0, 0, 0, 0, false, null, { adaptiveExpand: true, profile: strict, trace }) as Array<{ id: string; hop: number }>;
  assert.equal(trace.expanded, true);
  assert.ok(out.some((m) => m.id === neighbor));
});
```

- [ ] **Step 2: Run the integration test and verify RED**

Run: `pnpm exec tsx --test test/adaptive-recall.test.ts`

Expected: TypeScript/runtime failure because `recall()` has no internal policy argument and no trace.

- [ ] **Step 3: Add internal trace types and compute direct evidence before traversal**

In `src/memoryGraph.ts`, import the policy types/functions and add:

```typescript
import {
  DEFAULT_RECALL_POLICY, shouldExpand, shouldInject, signalsFromMatchedVia,
  type DirectEvidenceSummary, type InjectEvidence, type RecallPolicyProfile, type RecallPolicyTrace,
} from "./recallPolicy.js";

type RecallPolicyMode = "legacy" | "adaptive-expand" | "adaptive";
interface RecallInternalOptions {
  adaptiveExpand?: boolean;
  profile?: RecallPolicyProfile;
  trace?: RecallPolicyTrace;
}
```

Append `internal: RecallInternalOptions = {}` after `contextText` in `recall()`. Hoist `let effectiveExpand = expand;` beside the other Phase-1 outputs and remove the existing early `actualTopK` declaration. Immediately before associative traversal, sort a snapshot of direct scores, identify the first and second candidates, and build `DirectEvidenceSummary` from `memScores`, `memRawSim`, and `memMatchedKeys`. Call `shouldExpand()` only when both `expand` and `internal.adaptiveExpand` are true. Use `effectiveExpand` for traversal and the hop penalty; after Phase 1, define `const actualTopK = effectiveExpand ? topK * 2 : topK` for Phase 2. Leave `expand` behavior byte-compatible when `adaptiveExpand` is false.

The exact summary assembly is:

```typescript
const directSorted = Object.entries(memScores).sort(([, a], [, b]) => b - a);
const [topId, topRankScore = 0] = directSorted[0] ?? [];
const runnerUpRankScore = directSorted[1]?.[1] ?? 0;
const directSummary: DirectEvidenceSummary = {
  definiteAnchor: Boolean(topId) && (memRawSim[topId] ?? 0) >= 0.999,
  contentGate,
  topRankScore,
  runnerUpRankScore,
  topRelevanceScore: topId ? (memRawSim[topId] ?? 0) : 0,
  topSignals: signalsFromMatchedVia(topId ? (memMatchedKeys[topId] ?? []) : []),
  candidateCount: directSorted.length,
};
definiteAnchor = directSummary.definiteAnchor;
if (expand && internal.adaptiveExpand) {
  const decision = shouldExpand(directSummary, internal.profile ?? DEFAULT_RECALL_POLICY);
  effectiveExpand = decision.accept;
  if (internal.trace) {
    internal.trace.direct = directSummary;
    internal.trace.expanded = effectiveExpand;
    internal.trace.expansionReason = decision.reason;
  }
}
```

Remove the later `definiteAnchor = candidateIds.some(...)` recomputation and continue using the direct-only value above. A graph-expanded neighbor must never manufacture a query anchor.

- [ ] **Step 4: Wire experiment mode through `recallInject()` without changing its default**

Extend its options type with:

```typescript
policyMode?: RecallPolicyMode;
profile?: RecallPolicyProfile;
trace?: RecallPolicyTrace;
```

Create `const policyMode = opts.policyMode ?? "legacy";` and `const policyTrace = opts.trace ?? {};` near the start of `recallInject()` so experiments are explicit and adaptive injection can consume the direct summary even when the caller did not request diagnostics. Pass the final internal argument to `recall()`:

```typescript
{
  adaptiveExpand: policyMode !== "legacy",
  profile: opts.profile,
  trace: policyTrace,
}
```

The default remains legacy until Task 4's adoption gate passes.

- [ ] **Step 5: Run focused and regression tests**

Run: `pnpm exec tsx --test test/adaptive-recall.test.ts test/recall-inject.test.ts test/recall-quality-lib.test.ts`

Expected: all PASS.

Run: `pnpm run build`

Expected: PASS.

- [ ] **Step 6: Export the quality runner and add explicit mode selection**

Extract Task 1's runner body behind this interface, leaving the CLI as a thin caller:

```typescript
import { pathToFileURL } from "node:url";
import type { RecallPolicyProfile, RecallPolicyTrace } from "../src/recallPolicy.ts";
import type { CaseResult, QualityMetrics } from "./recall-quality-lib.ts";

export interface QualityRunOptions {
  split: "tune" | "holdout";
  mode: "legacy" | "adaptive-expand" | "adaptive";
  profile?: RecallPolicyProfile;
  repeat?: number;
}

export interface QualityReport {
  model: "bge-m3";
  split: "tune" | "holdout";
  mode: "legacy" | "adaptive-expand" | "adaptive";
  repeat: number;
  rows: Array<CaseResult & { trace: RecallPolicyTrace }>;
  metrics: QualityMetrics;
}

export async function runQuality(options: QualityRunOptions): Promise<QualityReport>;
```

The function performs the same graph-copy/load/cleanup loop already written in Task 1 and passes `options.profile` without reading process arguments. The CLI parses `--mode=legacy|adaptive-expand|adaptive`, `--split`, `--repeat`, and `--out`, calls `runQuality()`, then prints/writes the returned report. Create a `RecallPolicyTrace` per case and call:

```typescript
const injected = await graph.recallInject(
  c.query, 1, c.namespace,
  { policyMode: mode, profile: options.profile, trace },
  c.context,
) as { memories: Array<{ content: string }> };
```

Include `mode` and the trace in the report. Reject any other mode with a non-zero exit. Guard CLI execution with `import.meta.url === pathToFileURL(process.argv[1]).href` so importing `runQuality()` from the calibrator does not execute the CLI.

- [ ] **Step 7: Commit the dormant H1 experiment path**

```bash
git add src/memoryGraph.ts test/adaptive-recall.test.ts bench/recall-quality.ts
git commit -m "feat(recall): add adaptive expansion experiment"
```

---

### Task 4: Calibrate and decide H1

**Files:**
- Create: `bench/calibrate-recall-policy.ts`
- Create: `bench/recall-policy-tune.json`
- Modify: `bench/hotpot.ts`
- Modify: `bench/hotpot-agentkeys.ts`
- Create after measurement: `bench/hotpot-adaptive-results.json`
- Create after measurement: `bench/hotpot-agentkeys-adaptive-results.json`
- Modify on adoption: `src/recallPolicy.ts`
- Modify: `src/memoryGraph.ts`
- Modify: `BENCHMARKS.md`

**Interfaces:**
- Consumes: tune cases, `RecallPolicyProfile`, and explicit legacy/adaptive-expand runner modes.
- Produces: one frozen H1 profile selected without holdout labels and a provisional H1 result; final adoption waits for Task 5's single holdout run.

- [ ] **Step 1: Implement a bounded tuning grid**

Create `bench/calibrate-recall-policy.ts` with this exact grid:

```typescript
import { writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { DEFAULT_RECALL_POLICY, type RecallPolicyProfile } from "../src/recallPolicy.ts";
import { runQuality } from "./recall-quality.ts";

type TuneRow = {
  profile: RecallPolicyProfile;
  negativeFalseInjectRate: number;
  positiveInjectHit1: number;
  positiveRecallHit3: number;
  expandedQueries: number;
};
const profiles = [];
for (const directGateRatio of [1.00, 1.05, 1.10, 1.15, 1.20]) {
  for (const directMarginRatio of [0, 0.05, 0.10, 0.20]) {
    for (const directMinSignals of [1, 2]) {
      profiles.push({
        ...DEFAULT_RECALL_POLICY,
        directGateRatio, directMarginRatio, directMinSignals,
      });
    }
  }
}

const table: TuneRow[] = [];
for (const profile of profiles) {
  const report = await runQuality({ split: "tune", mode: "adaptive-expand", profile, repeat: 1 });
  table.push({
    profile,
    negativeFalseInjectRate: report.metrics.negativeFalseInjectRate,
    positiveInjectHit1: report.metrics.positiveInjectHit1,
    positiveRecallHit3: report.metrics.positiveRecallHit3,
    expandedQueries: report.rows.filter((row) => row.trace.expanded).length,
  });
}
table.sort((a, b) =>
  a.negativeFalseInjectRate - b.negativeFalseInjectRate
  || b.positiveInjectHit1 - a.positiveInjectHit1
  || b.positiveRecallHit3 - a.positiveRecallHit3
  || a.expandedQueries - b.expandedQueries
  || a.profile.directGateRatio - b.profile.directGateRatio
  || a.profile.directMarginRatio - b.profile.directMarginRatio
  || a.profile.directMinSignals - b.profile.directMinSignals
);
await writeFile(
  resolve("bench/recall-policy-tune.json"),
  `${JSON.stringify({ winner: table[0].profile, table }, null, 2)}\n`,
);
```

For each profile, run only the tune cases in `adaptive-expand` mode. Sort candidates lexicographically by:

1. lower `negativeFalseInjectRate`;
2. higher `positiveInjectHit1`;
3. higher `positiveRecallHit3`;
4. fewer expanded queries;
5. lower `directGateRatio`, then lower `directMarginRatio`, then lower `directMinSignals` for a stable tie-break.

Write the full tuning table and winner to `bench/recall-policy-tune.json`. Do not load, execute, or print holdout case details in this script.

- [ ] **Step 2: Run tune calibration**

Run: `pnpm exec tsx bench/calibrate-recall-policy.ts`

Expected: exits 0, evaluates 40 profiles on tune only, and writes a deterministic winner.

- [ ] **Step 3: Freeze the winning H1 values**

Update only `directGateRatio`, `directMarginRatio`, and `directMinSignals` in `DEFAULT_RECALL_POLICY` to the winner recorded in `bench/recall-policy-tune.json`. Leave inject fields unchanged.

- [ ] **Step 4: Measure H1 tune latency without opening holdout**

Run legacy and adaptive expansion five times on tune:

```bash
pnpm exec tsx bench/recall-quality.ts --split=tune --mode=legacy --repeat=5 --out=/tmp/keymem-h1-tune-legacy.json
pnpm exec tsx bench/recall-quality.ts --split=tune --mode=adaptive-expand --repeat=5 --out=/tmp/keymem-h1-tune-adaptive.json
```

Expected provisional result: adaptive expansion improves or preserves positive metrics, does not raise tune false-inject, and stays within the latency guardrail. Do not run any holdout case in Task 4.

- [ ] **Step 5: Run independent graph controls**

Before running the controls, add `const adaptive = process.env.KEYMEM_ADAPTIVE === "1";` to both Hotpot runners. Pass the internal options object as the final `recall()` argument only for GRAPH:

```typescript
const res = await g.recall(
  r.question, TOPK, null,
  cond === "GRAPH", cond === "GRAPH" ? 2 : 1,
  0, 0, 0, 0, 0, false, null,
  { adaptiveExpand: adaptive && cond === "GRAPH" },
);
```

When `adaptive` is true, write to `bench/hotpot-adaptive-results.json` and `bench/hotpot-agentkeys-adaptive-results.json`; otherwise preserve the existing output paths.

Run:

```bash
KEYMEM_ADAPTIVE=1 pnpm exec tsx bench/hotpot-agentkeys.ts bench/hotpot-agentkeys.json
```

Expected: bridge GRAPH both@5 is not below the documented 63% blind-key baseline.

Run: `KEYMEM_ADAPTIVE=1 pnpm exec tsx bench/hotpot.ts 100`

Expected: comparison GRAPH both@5 is not below the existing candidate's legacy result. Record the exact before/after values; do not substitute support-recall for both@5.

- [ ] **Step 6: Provisionally retain or reject H1**

If the tune and independent controls pass, freeze H1's profile and keep `adaptive-expand` available for the preregistered final holdout comparison. Do not change the runtime default from legacy yet.

If a tune or independent control fails, reject H1 before holdout: keep legacy as default, remove the production expansion branch and its production import, keep the benchmark result, and add a `BENCHMARKS.md` negative-result section explaining which metric failed. Do not keep a default-off production feature that has no measured value.

- [ ] **Step 7: Commit the H1 decision**

```bash
git add bench/calibrate-recall-policy.ts bench/recall-policy-tune.json bench/hotpot.ts bench/hotpot-agentkeys.ts bench/hotpot-adaptive-results.json bench/hotpot-agentkeys-adaptive-results.json src/recallPolicy.ts src/memoryGraph.ts test/adaptive-recall.test.ts BENCHMARKS.md
git commit -m "bench(recall): calibrate adaptive expansion"
```

If rejected, use: `git commit -m "bench: record adaptive expansion result"`.

---

### Task 5: H2 evidence-aware injection gate

**Files:**
- Modify: `src/memoryGraph.ts:2109-2171`
- Modify: `src/recallPolicy.ts`
- Modify: `test/recall-policy.test.ts`
- Modify: `test/recall-inject.test.ts`
- Modify: `bench/calibrate-recall-policy.ts`
- Create: `bench/inject-policy-tune.json`
- Create after decision: `bench/inject-policy-holdout-legacy.json`
- Create after decision when H1 is provisional: `bench/inject-policy-holdout-adaptive-expand.json`
- Create after decision: `bench/inject-policy-holdout-adaptive.json`

**Interfaces:**
- Consumes: existing recall result fields `matched_via`, `hop`, `rank_score`, and `relevance_score`; existing structured/lexical coverage helpers; query-level anchor summary.
- Produces: an optional final inject abstention decision plus benchmark-only `injectAccepted` / `injectReason` trace fields.

- [ ] **Step 1: Add failing inject integration tests**

Extend `test/recall-inject.test.ts` with two deterministic cases:

```typescript
function ambiguousInjectVec(text: string): number[] {
  const t = text.toLowerCase();
  if (t.includes("alpha")) return [0.8, 0.6, 0];
  if (t.includes("beta")) return [0.8, -0.6, 0];
  return [1, 0, 0];
}

test("adaptive injection abstains when top candidates are indistinguishable", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-inject-ambiguous-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder(ambiguousInjectVec);
  t.after(() => embedding.__clearTestEmbedder());
  const mg = await import(`../src/memoryGraph.ts?inject=${n++}`);
  const g = new mg.MemoryGraph();
  await g.load();
  await g.add("topic alpha", ["alpha"], {});
  await g.add("topic beta", ["beta"], {});
  const profile = {
    directGateRatio: 1, directMarginRatio: 0, directMinSignals: 1,
    injectGateRatio: 1, injectMarginRatio: 0.20, injectMinSignals: 1,
  };
  const legacy = await g.recallInject("topic", 1, null, { policyMode: "legacy", profile }) as { memories: unknown[] };
  const adaptive = await g.recallInject("topic", 1, null, { policyMode: "adaptive", profile }) as { memories: unknown[] };
  assert.equal(legacy.memories.length, 1);
  assert.equal(adaptive.memories.length, 0);
});

test("adaptive injection keeps an anchored graph-only memory", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-inject-graph-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((tx: string) => vec(tx));
  t.after(() => embedding.__clearTestEmbedder());
  const mg = await import(`../src/memoryGraph.ts?inject=${n++}`);
  const g = new mg.MemoryGraph();
  await g.load();
  await g.add("user works at Acme", ["job", "Acme"], {});
  const [historyId] = await g.add("Acme was founded in 1990", ["Acme", "history"], {});
  const profile = {
    directGateRatio: 2, directMarginRatio: 0, directMinSignals: 1,
    injectGateRatio: 1, injectMarginRatio: 0, injectMinSignals: 1,
  };
  const adaptive = await g.recallInject("user job", 5, null, { policyMode: "adaptive", profile }) as { memories: Array<{ id: string }> };
  assert.ok(adaptive.memories.some((m) => m.id === historyId));
});
```

- [ ] **Step 2: Run the inject tests and verify RED**

Run: `pnpm exec tsx --test test/recall-inject.test.ts`

Expected: the indistinguishable-candidate adaptive case fails because `recallInject()` does not call `shouldInject()`.

- [ ] **Step 3: Integrate the final policy gate**

In `recallInject()`, after `precise` is built and before `selectInject`, derive evidence for its first two candidates:

```typescript
const top = precise[0];
const runnerUp = precise[1];
const direct = policyTrace.direct;
const evidence: InjectEvidence | null = top ? {
  definiteAnchor: direct?.definiteAnchor ?? false,
  contentGate: direct?.contentGate ?? this._contentGateFor(query),
  anchorRelevanceScore: direct?.topRelevanceScore ?? top.relevance_score ?? 0,
  rankScore: top.rank_score ?? top.score,
  runnerUpRankScore: runnerUp?.rank_score ?? runnerUp?.score ?? 0,
  relevanceScore: top.relevance_score ?? 0,
  hop: top.hop ?? 1,
  matchedVia: top.matched_via ?? [],
  structuredCoverage: hasStructuredTokenCoverage(query, top),
  lexicalCoverage: hasLexicalQueryCoverage(query, top),
} : null;
const injectDecision = evidence && policyMode === "adaptive"
  ? shouldInject(evidence, opts.profile ?? DEFAULT_RECALL_POLICY)
  : { accept: Boolean(evidence), reason: evidence ? "legacy" : "no_candidate" };
policyTrace.injectAccepted = injectDecision.accept;
policyTrace.injectReason = injectDecision.reason;
const injectable = injectDecision.accept ? precise : [];
```

Use `injectable`, not `precise`, to build `cands` and `byId`. Add `rank_score`, `relevance_score`, and `hop` to the local `InjectMemory` type because those fields already exist in `recall()` results.

- [ ] **Step 4: Run focused tests and build**

Run: `pnpm exec tsx --test test/recall-policy.test.ts test/recall-inject.test.ts test/adaptive-recall.test.ts`

Expected: all PASS.

Run: `pnpm run build`

Expected: PASS.

- [ ] **Step 5: Tune H2 without reading holdout labels**

Extend the calibrator with the exact grid:

```typescript
const frozenH1Profile = DEFAULT_RECALL_POLICY;
const injectProfiles: RecallPolicyProfile[] = [];
for (const injectGateRatio of [1.00, 1.05, 1.10, 1.15]) {
  for (const injectMarginRatio of [0, 0.03, 0.05, 0.10, 0.20]) {
    for (const injectMinSignals of [1, 2]) {
      injectProfiles.push({
        ...frozenH1Profile,
        injectGateRatio, injectMarginRatio, injectMinSignals,
      });
    }
  }
}

const injectTable = [];
for (const profile of injectProfiles) {
  const report = await runQuality({ split: "tune", mode: "adaptive", profile, repeat: 1 });
  const correctCoverage = report.rows.filter((row) => row.kind === "positive" && row.injectHit).length;
  injectTable.push({ profile, metrics: report.metrics, correctCoverage });
}
injectTable.sort((a, b) =>
  a.metrics.negativeFalseInjectRate - b.metrics.negativeFalseInjectRate
  || b.metrics.positiveInjectHit1 - a.metrics.positiveInjectHit1
  || b.correctCoverage - a.correctCoverage
  || a.profile.injectGateRatio - b.profile.injectGateRatio
  || a.profile.injectMarginRatio - b.profile.injectMarginRatio
  || a.profile.injectMinSignals - b.profile.injectMinSignals
);
await writeFile(
  resolve("bench/inject-policy-tune.json"),
  `${JSON.stringify({ winner: injectTable[0].profile, table: injectTable }, null, 2)}\n`,
);
```

Select lexicographically by lower false-inject rate, higher positive inject hit@1, higher coverage among correctly injected positives, then lower gate/margin/min-signal values. Write all results and the winner to `bench/inject-policy-tune.json`.

- [ ] **Step 6: Freeze H2 and run holdout once**

Update the three inject fields in `DEFAULT_RECALL_POLICY`. At this point every policy is frozen. Run the holdout exactly once per preregistered mode, in one uninterrupted batch:

```bash
pnpm exec tsx bench/recall-quality.ts --split=holdout --mode=legacy --repeat=5 --out=bench/inject-policy-holdout-legacy.json
pnpm exec tsx bench/recall-quality.ts --split=holdout --mode=adaptive-expand --repeat=5 --out=bench/inject-policy-holdout-adaptive-expand.json
pnpm exec tsx bench/recall-quality.ts --split=holdout --mode=adaptive --repeat=5 --out=bench/inject-policy-holdout-adaptive.json
```

Do not edit policy code between these commands. If H1 was rejected in Task 4, omit the `adaptive-expand` command and record H1 as already rejected.

Make the final decisions from the preregistered comparisons:

- H1 passes when `adaptive-expand` beats legacy on positive inject hit@1 (or preserves 100% while improving hit@3/false-inject) and clears every global guardrail.
- H2 passes when `adaptive` beats the accepted H1 mode on positive inject hit@1 or false-inject rate, does not lower coverage of correctly injected positives, and the complete adaptive pipeline clears every global guardrail against legacy.
- If both pass, set `const policyMode = opts.policyMode ?? "adaptive";`.
- If only H1 passes, set the default to `"adaptive-expand"` and remove the unadopted injection-gate production call.
- If only H2 passes, keep unconditional expansion, set the default to `"adaptive"`, and remove the unadopted conditional-expansion branch.
- If neither passes, keep `"legacy"` and remove both production branches.

Retain result files and document rejected hypotheses. Never retune after viewing holdout.

- [ ] **Step 7: Commit the H2 decision**

```bash
git add src/memoryGraph.ts src/recallPolicy.ts test/recall-policy.test.ts test/recall-inject.test.ts bench/calibrate-recall-policy.ts bench/inject-policy-tune.json bench/inject-policy-holdout-legacy.json bench/inject-policy-holdout-adaptive-expand.json bench/inject-policy-holdout-adaptive.json BENCHMARKS.md
git commit -m "feat(inject): adopt measured evidence gate"
```

If H1 was rejected and the adaptive-expand report was therefore not produced, omit that one path from `git add`. If H2 is rejected, use: `git commit -m "bench: record evidence gate result"`.

---

### Task 6: Final regression, benchmark report, and compatibility proof

**Files:**
- Modify: `BENCHMARKS.md`
- Modify: `docs/superpowers/plans/2026-09-04-recall-hit-rate-v2.md` (check completed boxes and record decisions)

**Interfaces:**
- Consumes: baseline, tuning tables, holdout reports, independent graph benchmarks, build/test output.
- Produces: a reproducible benchmark section and final verification evidence.

- [ ] **Step 1: Document measured results**

Add a `Recall hit-rate v2` section to `BENCHMARKS.md` containing:

- fixture size and memory-disjoint split method;
- legacy versus adopted-candidate tune and holdout tables;
- positive inject hit@1, recall hit@3, false-inject rate, coverage, and median-of-five p50/p95;
- blind-key bridge both@5 and comparison both@5 controls;
- each hypothesis marked adopted or rejected with its disproof criterion;
- exact reproduction commands.

Do not describe an unadopted hypothesis as a feature. If both hypotheses fail, the correct result is a benchmark-only commit documenting why global threshold/policy changes were rejected.

- [ ] **Step 2: Run focused retrieval tests**

Run:

```bash
pnpm exec tsx --test test/recall-policy.test.ts test/adaptive-recall.test.ts test/recall-inject.test.ts test/inject-select.test.ts test/context-dual-path.test.ts test/searchkeys-ranking.test.ts test/short-query-gate.test.ts test/nearest-keys.test.ts test/decay-ranking.test.ts
```

Expected: all PASS.

- [ ] **Step 3: Run full verification**

Run: `pnpm run build`

Expected: PASS.

Run: `pnpm test`

Expected: all tests PASS with zero failures.

Run: `git diff --check`

Expected: no output.

- [ ] **Step 4: Verify public compatibility and passive behavior**

Run: `pnpm exec tsx --test test/recall-inject.test.ts test/inject-endpoint.test.ts test/hook-inject.test.ts test/decay-mcp.test.ts`

Expected: all PASS; MCP payload shapes remain compatible, injection still defaults to one memory, and passive injection does not increment access/depth or link weights.

- [ ] **Step 5: Commit final evidence**

```bash
git add BENCHMARKS.md docs/superpowers/plans/2026-09-04-recall-hit-rate-v2.md
git commit -m "docs: report recall hit-rate v2 experiments"
```
