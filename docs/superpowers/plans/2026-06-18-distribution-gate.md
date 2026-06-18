# Distribution-Based Not-Found Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make "not found" detection work on the default e5 backend by gating recall on whether the top hit is a robust-z (median/MAD) outlier of the query's similarity distribution, not just an absolute threshold.

**Architecture:** A new per-profile `gateZ` threshold drives a distribution gate composed into `recall()`'s existing anchor condition: `hasAnchor = definiteAnchor || (absoluteAnchor && passesDistributionGate(...))`. The gate computes a robust z-score of the top content similarity against the full eligible content-similarity population (already computed in Dense Path B). `gateZ = 0` disables it, preserving 0.7.0 behavior for every profile except e5. Pure stats helpers are unit-tested deterministically; the e5 `gateZ` value is calibrated against the real model.

**Tech Stack:** TypeScript (ESM, NodeNext), fastembed (local ONNX), minisearch, `node:test` + `tsx`.

## Global Constraints

- ESM with `.js` import specifiers in TypeScript source (NodeNext). Match existing style.
- Profiles with `gateZ = 0` must behave **byte-for-byte** like 0.7.0 (no regression). Only e5 ships a non-zero `gateZ`.
- The distribution gate is a single **query-level** found/not-found decision. It must NOT per-item filter results (in-result trimming stays `min_rel_score`).
- `gateZ` is a non-negative number, NOT in [0,1] — it must use a new `envNonNegative` parser, not the existing `envThreshold` (which clamps to [0,1]).
- `definiteAnchor` (literal name/proper_noun exact match, `memRawSim >= 0.999`) bypasses the distribution gate — such hits are always "found".
- `npm test` (test/*.test.ts) and `npm run build` (tsc) must pass. Fast deterministic tests go in `test/retriever-quality.test.ts` via the existing test embedder seam. Model-dependent calibration is a manual `*.live.ts` script, not in `npm test`.
- The test file `test/retriever-quality.test.ts` sets `LOCAL_EMBEDDING_MODEL=bge-m3` (profile `gateZ=0`), so recall tests must pass `min_z` explicitly to activate the gate.
- `GATE_MIN_POPULATION = 8`: with fewer eligible memories the gate is skipped (passes).
- Commit after every task. Run `npm run build` before committing.

---

### Task 1: Pure stats helpers (robust-z + distribution gate)

Self-contained, deterministic. No wiring into recall yet.

**Files:**
- Modify: `src/memoryGraph.ts` (add helpers near `passesAbsoluteGate`, ~line 63)
- Test: `test/retriever-quality.test.ts` (append)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `robustZScore(top: number, values: number[]): number` — `(top − median) / (1.4826 × MAD)`. Returns `Infinity` when `values` is empty or `MAD === 0` (degenerate — cannot compute a meaningful z). Returns `0` when `top === median` with non-zero MAD only via the formula.
  - `passesDistributionGate(top: number, values: number[], gateZ: number, minCount: number): boolean` — `true` (skip/pass) when `gateZ <= 0`, or `values.length < minCount`, or `robustZScore` is non-finite; otherwise `robustZScore(top, values) >= gateZ`.

- [ ] **Step 1: Write the failing tests**

Append to `test/retriever-quality.test.ts`:

```typescript
test("robustZScore: clear right-tail outlier scores high", async () => {
  const { robustZScore } = await import("../src/memoryGraph.ts");
  // background clustered near 0.5, one value 0.95 -> large positive z
  const z = robustZScore(0.95, [0.48, 0.5, 0.52, 0.49, 0.51, 0.5, 0.53, 0.95]);
  assert.ok(z > 4, `expected big z, got ${z}`);
});

test("robustZScore: non-outlier (uniform-ish band) scores low", async () => {
  const { robustZScore } = await import("../src/memoryGraph.ts");
  const z = robustZScore(0.95, [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.9, 0.92]);
  assert.ok(z < 3, `expected small z, got ${z}`);
});

test("robustZScore: degenerate (MAD 0) and empty -> Infinity", async () => {
  const { robustZScore } = await import("../src/memoryGraph.ts");
  assert.equal(robustZScore(0.9, [0.5, 0.5, 0.5, 0.5]), Infinity); // MAD 0
  assert.equal(robustZScore(0.9, []), Infinity);                    // empty
});

test("passesDistributionGate: disabled / small-N / degenerate all pass", async () => {
  const { passesDistributionGate } = await import("../src/memoryGraph.ts");
  // gateZ <= 0 -> disabled
  assert.equal(passesDistributionGate(0.95, [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.95], 0, 8), true);
  // population < minCount -> skip
  assert.equal(passesDistributionGate(0.95, [0.1, 0.2, 0.3], 3, 8), true);
  // MAD 0 (degenerate) -> skip
  assert.equal(passesDistributionGate(0.95, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 3, 8), true);
});

test("passesDistributionGate: outlier passes, non-outlier blocks", async () => {
  const { passesDistributionGate } = await import("../src/memoryGraph.ts");
  const bg = [0.48, 0.5, 0.52, 0.49, 0.51, 0.5, 0.53, 0.95];
  assert.equal(passesDistributionGate(0.95, bg, 3, 8), true);   // z>4 >= 3
  const flat = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.9, 0.92];
  assert.equal(passesDistributionGate(0.95, flat, 3, 8), false); // z<3
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx tsx --test test/retriever-quality.test.ts`
Expected: FAIL — `robustZScore` / `passesDistributionGate` are not exported.

- [ ] **Step 3: Implement the helpers**

In `src/memoryGraph.ts`, immediately after `passesAbsoluteGate` (~line 65), add:

```typescript
// Median of a numeric array (sorted copy; average of middle two for even length).
function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

// Robust z-score of `top` against the distribution of `values`, using median and
// MAD (median absolute deviation) scaled by 1.4826 so the result reads in
// sigma-like units. Robust to the skewed, packed cosine bands that e5 produces,
// where a true match is a right-tail outlier even though every value is "high".
// Returns Infinity when the distribution is degenerate (empty, or MAD == 0) —
// no meaningful z can be computed, so callers treat it as "do not block".
export function robustZScore(top: number, values: number[]): number {
  if (values.length === 0) return Infinity;
  const med = median(values);
  const mad = median(values.map((v) => Math.abs(v - med)));
  if (mad === 0) return Infinity;
  return (top - med) / (1.4826 * mad);
}

// Distribution "not-found" gate. Passes (true) when disabled (gateZ <= 0), when
// the population is too small to be reliable (< minCount), or when the z is
// non-finite (degenerate distribution). Otherwise the top hit must be at least a
// gateZ-sigma outlier of the similarity distribution to count as "found".
export function passesDistributionGate(
  top: number,
  values: number[],
  gateZ: number,
  minCount: number
): boolean {
  if (gateZ <= 0) return true;
  if (values.length < minCount) return true;
  const z = robustZScore(top, values);
  return Number.isFinite(z) ? z >= gateZ : true;
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx tsx --test test/retriever-quality.test.ts`
Expected: PASS (all 5 new tests + existing).

- [ ] **Step 5: Build + commit**

```bash
npm run build
git add src/memoryGraph.ts test/retriever-quality.test.ts
git commit -m "feat: robust-z (median/MAD) distribution helpers for recall gate"
```

---

### Task 2: Wire the distribution gate into recall

Add the `gateZ` profile field + env parser + recall `minZ` param, collect the background distribution, and compose the gate into the anchor condition.

**Files:**
- Modify: `src/embedding.ts` (`ThresholdProfile` ~line 90, `THRESHOLD_PROFILES` ~line 106, add `envNonNegative` near `envThreshold` ~line 176, `getThresholdProfile` ~line 189)
- Modify: `src/memoryGraph.ts` (thresholds block ~line 23, `recall()` signature ~line 746, Dense Path B ~line 833, anchor block ~line 956)
- Modify: `src/server.ts` (recall inputSchema + call site)
- Test: `test/retriever-quality.test.ts` (append)

**Interfaces:**
- Consumes: `robustZScore`, `passesDistributionGate` (Task 1); existing `passesAbsoluteGate`, `getThresholdProfile`.
- Produces:
  - `ThresholdProfile.gateZ: number` (0 = disabled).
  - `recall(query, topK?, namespace?, expand?, maxHops?, minRelScore?, minScore?, minZ?)` — new 8th param `minZ` (default = profile `gateZ`).
  - MCP recall tool param `min_z`.
  - Constant `GATE_MIN_POPULATION = 8` and `GATE_Z_THRESHOLD = _THRESHOLDS.gateZ` in memoryGraph.ts.

- [ ] **Step 1: Write the failing tests**

Append to `test/retriever-quality.test.ts`. The seam maps each text to a 2-D vector; `cos([1,0],[c, sqrt(1-c²)]) = c`, so we control every similarity exactly. Queries embed to `[1,0]`; keys embed orthogonally (`[0,1]`, keySim 0) so anchoring is purely content/distribution.

```typescript
function vecForCos(c: number): [number, number] {
  return [c, Math.sqrt(Math.max(0, 1 - c * c))];
}

async function gateGraph(t: any, contents: Record<string, number>, queryName: string) {
  const { mkdtemp, rm } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const dataDir = await mkdtemp(join(tmpdir(), "sm-distgate-"));
  t.after(() => rm(dataDir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dataDir;
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((text) => {
    if (text === queryName) return [1, 0];
    if (text in contents) return vecForCos(contents[text]);
    return [0, 1]; // keys + anything else: orthogonal to the query
  });
  t.after(() => emb.__clearTestEmbedder());
  const { MemoryGraph } = await import(`../src/memoryGraph.ts?distgate=${queryName}`);
  const g = new MemoryGraph();
  await g.load();
  return g;
}

test("distribution gate: uniform high-sim band returns [] (no outlier)", async (t) => {
  // 8 memories all clustered ~0.90-0.95; absolute gate would pass, but no outlier.
  const contents: Record<string, number> = {};
  const cs = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.9, 0.92];
  cs.forEach((c, i) => (contents[`m${i}`] = c));
  const g = await gateGraph(t, contents, "q");
  for (let i = 0; i < cs.length; i++) await g.add(`m${i}`, [`k${i}`]);
  // min_z=3 activates the gate (bge-m3 profile default is 0)
  const r = (await g.recall("q", 5, null, false, 2, 0, 0, 3)) as any[];
  assert.equal(r.length, 0, `expected [] for uniform band, got ${r.length}`);
});

test("distribution gate: one clear outlier returns hits", async (t) => {
  const contents: Record<string, number> = {};
  const cs = [0.48, 0.5, 0.52, 0.49, 0.51, 0.5, 0.53, 0.95]; // last is the outlier
  cs.forEach((c, i) => (contents[`m${i}`] = c));
  const g = await gateGraph(t, contents, "q");
  for (let i = 0; i < cs.length; i++) await g.add(`m${i}`, [`k${i}`]);
  const r = (await g.recall("q", 5, null, false, 2, 0, 0, 3)) as any[];
  assert.ok(r.length >= 1, "outlier query should return hits");
  assert.ok(r.some((x) => x.content === "m7"), "the 0.95 outlier should be returned");
});

test("distribution gate: min_z=0 reproduces 0.7.0 (gate off)", async (t) => {
  const contents: Record<string, number> = {};
  const cs = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.9, 0.92];
  cs.forEach((c, i) => (contents[`m${i}`] = c));
  const g = await gateGraph(t, contents, "q");
  for (let i = 0; i < cs.length; i++) await g.add(`m${i}`, [`k${i}`]);
  const r = (await g.recall("q", 5, null, false, 2, 0, 0, 0)) as any[]; // min_z=0
  assert.ok(r.length >= 1, "with gate off, the uniform band should still return hits");
});

test("distribution gate: literal name-key match bypasses the gate", async (t) => {
  // Content forms a uniform non-outlier band (gate would otherwise block), but an
  // exact name-key literal match is a definite anchor and must survive.
  const { mkdtemp, rm } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const dataDir = await mkdtemp(join(tmpdir(), "sm-distgate-name-"));
  t.after(() => rm(dataDir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dataDir;
  const contentCos: Record<string, number> = { "Dongkyun is the user": 0.92 };
  for (let i = 0; i < 8; i++) contentCos[`pad ${i}`] = 0.9 + i * 0.005; // 0.90..0.935, no outlier
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((text) => {
    if (text === "Dongkyun") return [1, 0];          // query + the name key embed here (key uses literal match anyway)
    if (text in contentCos) return vecForCos(contentCos[text]);
    return [0, 1];                                    // other (concept) keys: orthogonal, no key-path match
  });
  t.after(() => emb.__clearTestEmbedder());
  const { MemoryGraph } = await import(`../src/memoryGraph.ts?distgatename=1`);
  const g = new MemoryGraph();
  await g.load();
  for (let i = 0; i < 8; i++) await g.add(`pad ${i}`, [`p${i}`]); // pad population >= GATE_MIN_POPULATION
  await g.add("Dongkyun is the user", ["Dongkyun"], { keyTypes: { Dongkyun: "name" } });
  // query literally contains the name -> exact name-key match -> definiteAnchor (rawSim 1.0)
  const r = (await g.recall("Dongkyun", 5, null, false, 2, 0, 0, 3)) as any[];
  assert.ok(r.some((x) => x.content === "Dongkyun is the user"), "literal name match must survive the gate");
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx tsx --test test/retriever-quality.test.ts`
Expected: FAIL — `recall` ignores the 8th arg (gate not wired); uniform-band test returns results instead of `[]`.

- [ ] **Step 3: Add `gateZ` to the profile interface and all rows**

In `src/embedding.ts`, add to `ThresholdProfile` (after `contradiction`, ~line 103):

```typescript
  // Robust-z (median/MAD) distribution gate threshold: the top content similarity
  // must be at least this many MAD-sigmas above the median of the query's
  // similarity distribution for the query to count as "found". 0 disables the gate.
  // Built for e5, whose narrow packed cosine band defeats the absolute minScore gate.
  gateZ: number;
```

Add `gateZ` to every profile row (`THRESHOLD_PROFILES`, ~line 106). All non-e5 profiles use `0` (disabled — no behavior change). e5 uses a draft `3.0`, finalized by calibration in Task 3:

```typescript
  openai: { keyMerge: 0.85, memoryDedup: 0.9, keyAutoLink: 0.5, keyRecall: 0.28, contentRecall: 0.28, minScore: 0.28, contradiction: 0.85, gateZ: 0 },
  bge: { keyMerge: 0.85, memoryDedup: 0.9, keyAutoLink: 0.6, keyRecall: 0.6, contentRecall: 0.5, minScore: 0.5, contradiction: 0.85, gateZ: 0 },
  e5: { keyMerge: 0.97, memoryDedup: 0.985, keyAutoLink: 0.93, keyRecall: 0.85, contentRecall: 0.8, minScore: 0.8, contradiction: 0.95, gateZ: 3.0 },
  minilm: { keyMerge: 0.85, memoryDedup: 0.9, keyAutoLink: 0.6, keyRecall: 0.5, contentRecall: 0.45, minScore: 0.45, contradiction: 0.85, gateZ: 0 },
```

(The bgem3 row keeps `gateZ: 0` — append `, gateZ: 0` before its closing brace, preserving the existing calibrated comment above it.)

- [ ] **Step 4: Add `envNonNegative` and wire `SUPER_MEMORY_GATE_Z`**

In `src/embedding.ts`, immediately after the `envThreshold` function (~line 188), add a parser that allows values > 1 (z is typically 2–5):

```typescript
// Like envThreshold but for non-negative unbounded values (e.g. a z-score gate),
// which legitimately exceed 1. Rejects negative / non-finite input.
function envNonNegative(name: string): number | undefined {
  const raw = process.env[name];
  if (raw === undefined || raw.trim() === "") return undefined;
  const n = Number(raw);
  if (!Number.isFinite(n) || n < 0) {
    console.error(`[super-memory] WARNING: ignoring ${name}="${raw}" (must be a number >= 0).`);
    return undefined;
  }
  return n;
}
```

In `getThresholdProfile()` (~line 189), add the `gateZ` field to the returned object (after `contradiction`):

```typescript
    contradiction: envThreshold("SUPER_MEMORY_CONTRADICTION") ?? base.contradiction,
    gateZ: envNonNegative("SUPER_MEMORY_GATE_Z") ?? base.gateZ,
```

- [ ] **Step 5: Add constants + recall param + collect the distribution**

In `src/memoryGraph.ts`, after `MIN_SCORE_THRESHOLD` (~line 23) add:

```typescript
const GATE_Z_THRESHOLD = _THRESHOLDS.gateZ;
const GATE_MIN_POPULATION = 8;
```

Import the new helpers — they are defined in this same file (Task 1), so no import change is needed.

Extend the `recall()` signature (~line 746) with an 8th param and clamp it (no upper bound):

```typescript
    minScore = MIN_SCORE_THRESHOLD,
    minZ = GATE_Z_THRESHOLD
  ): Promise<object[]> {
```

Next to the existing `minScore` clamp (~line 755) add:

```typescript
    minZ = Math.max(0, minZ);
```

Declare the background-distribution array next to `memRawSim` (~line 765):

```typescript
      const allContentSims: number[] = [];
```

In Dense Path B (~line 833), record every eligible memory's content sim into the background population (before the `contentRecall` threshold check). Change:

```typescript
        for (let i = 0; i < memIds.length; i++) {
          const mid = memIds[i];
          if (skip(mid)) continue;
          const cSim = contentSims[i];
          allContentSims.push(cSim);
          if (cSim >= CONTENT_RECALL_THRESHOLD) {
```

(Only the `allContentSims.push(cSim);` line is new; everything else is unchanged.)

- [ ] **Step 6: Compose the gate into the anchor condition**

Replace the current anchor block (~line 956-962) — currently:

```typescript
      const hasAnchor = Object.keys(memScores).some(
        (mid) => passesAbsoluteGate(memRawSim[mid] ?? 0, minScore)
      );
```

with:

```typescript
      // Anchor: the query is "found" iff a definite literal-key hit exists, OR a
      // candidate clears the absolute gate AND the top content similarity is a
      // robust-z outlier of the similarity distribution. The distribution gate
      // catches the e5 failure mode where every cosine is uniformly high so the
      // absolute gate false-positives. minZ (gateZ) = 0 disables it, leaving the
      // 0.7.0 absolute-only behavior unchanged for bge-m3 and other profiles.
      const candidateIds = Object.keys(memScores);
      const definiteAnchor = candidateIds.some((mid) => (memRawSim[mid] ?? 0) >= 0.999);
      const absoluteAnchor = candidateIds.some(
        (mid) => passesAbsoluteGate(memRawSim[mid] ?? 0, minScore)
      );
      let maxContentSim = 0;
      for (const s of allContentSims) if (s > maxContentSim) maxContentSim = s;
      const distOK = passesDistributionGate(maxContentSim, allContentSims, minZ, GATE_MIN_POPULATION);
      const hasAnchor = definiteAnchor || (absoluteAnchor && distOK);
```

(The `floor` / `ranked` lines immediately below are unchanged.)

- [ ] **Step 7: Surface `min_z` in the MCP tool**

In `src/server.ts`, add to the recall `inputSchema.properties` (after `min_score`):

```typescript
          min_score: { type: "number" },
          min_z: { type: "number" },
```

And pass it as the 8th argument in the `case "recall"` call (after the `min_score` arg):

```typescript
          typeof a.min_score === "number" ? a.min_score : undefined,
          typeof a.min_z === "number" ? a.min_z : undefined
```

(`undefined` falls through to the parameter default `GATE_Z_THRESHOLD`, preserving the per-profile gate.)

- [ ] **Step 8: Run tests + build**

Run: `npx tsx --test test/retriever-quality.test.ts`
Expected: PASS (4 new gate tests + all prior).
Run: `npm run build`
Expected: tsc clean.
Run: `npm test`
Expected: full suite passes (memoryGraph.test.ts unaffected — `gateZ` defaults to 0 there since no model is configured → BGE fallback profile; recall callers that pass no `minZ` get the profile default).

- [ ] **Step 9: Commit**

```bash
git add src/embedding.ts src/memoryGraph.ts src/server.ts test/retriever-quality.test.ts
git commit -m "feat: distribution (robust-z) not-found gate wired into recall"
```

---

### Task 3: Calibrate e5 gateZ against the real model + document

Measure the real e5 robust-z separation between found and not-found queries, set the e5 `gateZ`, and document the gate.

**Files:**
- Create: `test/distribution-gate.live.ts` (manual calibration/verification script)
- Modify: `src/embedding.ts` (e5 `gateZ` value, if calibration shows the 3.0 draft is wrong)
- Modify: `README.md` (gate documentation)
- Modify: `test/retriever-quality.test.ts` (update the e5 `gateZ` profile-value assertion if one was added; otherwise add one)

**Interfaces:**
- Consumes: real e5 backend; `recall()` with `min_z`; `robustZScore`.

- [ ] **Step 1: Create the calibration/verification script**

Create `test/distribution-gate.live.ts` (run with the real e5 model):

```typescript
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
  ["동균의 목표는 super-memory를 표준 메모리 레이어로 만드는 것", ["super-memory", "목표"]],
  ["super-memory는 TypeScript로 작성된 LLM 장기 메모리 시스템", ["super-memory", "개발"]],
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
const found = ["이름", "고양이", "super-memory 목표", "커피"];
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
```

- [ ] **Step 2: Run the calibration script and read the z separation**

Run: `EMBEDDING_BACKEND=local npx tsx test/distribution-gate.live.ts`
Expected: a table of robust-z for FOUND vs NOT-FOUND queries. Read the two clusters: FOUND z should be clearly higher than NOT-FOUND z. The end-to-end section uses the current draft `gateZ=3.0`.

- [ ] **Step 3: Set the e5 gateZ between the clusters**

Pick a `gateZ` strictly between the highest NOT-FOUND z and the lowest FOUND z (favor not over-blocking real matches — bias toward the lower side of the gap). If the measured gap does not contain 3.0, edit the e5 row in `src/embedding.ts` `THRESHOLD_PROFILES` to the chosen value, and update the e5 `gateZ` comment to record the measured FOUND/NOT-FOUND ranges and the chosen value (mirroring the contradiction-floor comment style). If 3.0 already sits cleanly in the gap, leave it and record the measured ranges in the comment.

- [ ] **Step 4: Re-run to confirm separation holds**

Run: `EMBEDDING_BACKEND=local npx tsx test/distribution-gate.live.ts`
Expected: `RESULT: 8 passed, 0 failed` (all FOUND non-empty, all NOT-FOUND `[]`). If a borderline query fails, note it in the report — thresholds are calibration-pending and env-tunable (`SUPER_MEMORY_GATE_Z`); do not chase 100% by distorting the value into a real cluster.

- [ ] **Step 5: Add/Update the profile-value unit test**

In `test/retriever-quality.test.ts`, in the test that asserts bgem3 profile fields, add an assertion that every profile defines a numeric `gateZ` and that bge-m3 is `0`:

```typescript
  assert.equal(THRESHOLD_PROFILES.bgem3.gateZ, 0);
  for (const fam of ["openai", "e5", "bge", "minilm", "bgem3"]) {
    assert.equal(typeof THRESHOLD_PROFILES[fam].gateZ, "number", fam);
  }
```

Run: `npx tsx --test test/retriever-quality.test.ts`
Expected: PASS.

- [ ] **Step 6: Document in README**

In `README.md`, in the score-gate / threshold section, add: the absolute `min_score` gate works on well-separated models (bge-m3/bge/openai); e5's narrow cosine band defeats it, so e5 instead uses a **distribution gate** — the top hit must be a robust-z (median/MAD) outlier (`gateZ`, env `SUPER_MEMORY_GATE_Z`, recall param `min_z`; `0` disables). Note both gates compose (AND) and a literal name/proper_noun match always counts as found. Mention `GATE_MIN_POPULATION = 8` (gate skipped below that). Keep the existing `min_rel_score` guidance for in-result trimming.

- [ ] **Step 7: Build + full suite + commit**

Run: `npm run build` → clean.
Run: `npm test` → all pass.

```bash
git add src/embedding.ts README.md test/retriever-quality.test.ts test/distribution-gate.live.ts
git commit -m "feat: calibrate e5 distribution gate (gateZ) on real model + docs"
```

---

## Self-Review

**Spec coverage:**
- Robust-z / MAD computation → Task 1 (`robustZScore`). ✓
- `passesDistributionGate` skip rules (disabled / small-N / degenerate) → Task 1. ✓
- Anchor composition `definiteAnchor || (absoluteAnchor && distGate)` → Task 2 Step 6. ✓
- Background population = all eligible content sims (Dense Path B) → Task 2 Step 5. ✓
- `gateZ` profile field, every row, e5 non-zero, others 0 → Task 2 Step 3. ✓
- `envNonNegative` (not `envThreshold`, since z ∉ [0,1]) → Task 2 Step 4. ✓
- `recall(minZ)` param + MCP `min_z` → Task 2 Steps 5,7. ✓
- `GATE_MIN_POPULATION = 8` → Task 2 Step 5. ✓
- No regression when `gateZ = 0` → preserved: gate passes when disabled (Task 1), bge-m3/others ship 0 (Task 2 Step 3); `min_z=0` reproduction test (Task 2 Step 1). ✓
- Calibration on real e5 + record measured values → Task 3. ✓
- Documentation → Task 3 Step 6. ✓

**Placeholder scan:** No TBD/TODO. e5 `gateZ` ships a concrete draft (3.0) finalized by measurement in Task 3 (a calibration step, not a placeholder). Every code step shows full code; commands have expected output.

**Type consistency:** `robustZScore(top, values)` and `passesDistributionGate(top, values, gateZ, minCount)` signatures are identical where consumed (Task 2 Step 6). `recall`'s 8th param `minZ` matches the server call site and the test calls (`recall(q, 5, null, false, 2, 0, 0, 3)` → minRelScore=0, minScore=0, minZ=3). `ThresholdProfile.gateZ` is defined (Task 2 Step 3), env-wired (Step 4), consumed as `GATE_Z_THRESHOLD` (Step 5).
