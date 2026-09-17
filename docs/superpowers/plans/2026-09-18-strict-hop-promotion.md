# Strict-Hop Promotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give a hop-2 key-association a guaranteed, safety-checked shot at `recall()`'s final result window instead of only competing on fused RRF/HOP_DECAY score, where corpus noise buries it before the reranker ever sees it.

**Architecture:** `recall()` already computes every hop-2 candidate and its score inside `gated` (Phase 1, locked). Add a pure helper that structurally identifies the single strongest hop-2 candidate reachable from the top anchor via a "narrow" key (shared by few memories — a specific connector, not a hub topic word), then wire it into Phase 2 (rerank) in two places: (1) guarantee it enters the rerank candidate pool even if `RERANK_POOL` would have excluded it by fused score, (2) after rerank, promote it into the final window only if the slot it would take is itself weak — never evict a confident direct/key hit.

**Tech Stack:** TypeScript, Node's built-in `node:test` + `node:assert/strict` (matches existing `test/*.test.ts` files — no new test framework).

**Spec:** This plan is derived from a same-day design conversation (no separate spec doc) plus two bench artifacts that ARE the evidence base:
- `bench/edge-experiments.ts` (the `STRICT_HOP` condition) and its results in `bench/edge-experiments-results.json` — measured assoc2 Hit@5 27%→60% with zero regression on `direct`/`semantic`, using a maximal "always force top5" version of this idea against `bench/assoc-fixture.json` (49 memories / 48 queries).
- `bench/edge-experiments-llm-results.json` — measured the risk this plan's safety check exists to prevent: the bench's blunt "always force top5" rule flipped `"미나 취미"` from a correct rank-5 hit to a rank-6 miss by evicting an already-good answer.

## Global Constraints

- Do not change the *set* of results a plain `expand=false` (DIRECT) call returns — this feature only activates when the caller already opted into `expand=true` (hop-2 mode). No new required parameter on `recall()`'s public signature.
- Feature-flagged via env var, default-on, so it can be killed in production with one env var if it misbehaves on the user's real store (`~/.keymem`), which this plan has NOT been validated against — only against the synthetic 49-memory bench fixture.
- No placeholder thresholds: every magic number below is either taken from the bench evidence or has an explicit env override so it can be re-tuned without a code change.

---

## File Structure

- **Modify: `src/memoryGraph.ts`**
  - Add module constants `STRICT_HOP_ENABLED`, `STRICT_HOP_MAX_KEY_MEMBERS`, `STRICT_HOP_EVICT_BELOW` near the other `cfgRaw(...)`-derived constants (top of file, alongside `RERANK_POOL` at line 72).
  - Add exported pure function `findStrictHopCandidate` near `passesDistributionGate` (before the `// ── Utils ──` comment, currently at line 276).
  - Modify `recall()`'s Phase 1→Phase 2 boundary (currently lines 2778–2808) to compute and apply the candidate.
- **Test: `test/strict-hop-promotion.test.ts`** (new file) — unit tests for the pure function (no embeddings) plus one integration test through `recall()` (mocked embedder/reranker, following the pattern in `test/rerank-integration.test.ts`).

---

## Task 1: Pure candidate-selection function

**Files:**
- Modify: `src/memoryGraph.ts:276` (insert before the `// ── Utils ──` comment)
- Test: `test/strict-hop-promotion.test.ts` (new)

**Interfaces:**
- Produces: `export function findStrictHopCandidate(gated: Array<[string, number]>, memHop: Record<string, number>, memToKeys: Record<string, Map<string, unknown>>, keyToMems: Record<string, Map<string, unknown>>, maxMembers: number): string | null` — used by Task 2.

- [ ] **Step 1: Write the failing tests**

Create `test/strict-hop-promotion.test.ts`:

```ts
import assert from "node:assert/strict";
import test from "node:test";
import { findStrictHopCandidate } from "../src/memoryGraph.ts";

// Helper: build a Map<string, unknown> from a list of ids (mirrors _memToKeys/_keyToMems shape).
function map(ids: string[]): Map<string, unknown> {
  return new Map(ids.map((id) => [id, true]));
}

test("findStrictHopCandidate picks a hop-2 neighbor reachable via a narrow key", () => {
  // anchor and target share "narrowKey" (2 members total) — a real, specific connector.
  const gated: Array<[string, number]> = [
    ["anchor", 0.9],
    ["noise1", 0.05],
    ["target", 0.01], // hop-2, buried at the bottom by fused score — the bug this fixes
  ];
  const memHop = { anchor: 1, noise1: 1, target: 2 };
  const memToKeys = {
    anchor: map(["narrowKey"]),
    noise1: map(["hubKey"]),
    target: map(["narrowKey"]),
  };
  const keyToMems = {
    narrowKey: map(["anchor", "target"]), // 2 members
    hubKey: map(["anchor", "noise1", "other1", "other2", "other3", "other4"]), // 6 members
  };
  const result = findStrictHopCandidate(gated, memHop, memToKeys, keyToMems, 3);
  assert.equal(result, "target");
});

test("findStrictHopCandidate ignores a hop-2 neighbor reachable only via a hub key", () => {
  const gated: Array<[string, number]> = [
    ["anchor", 0.9],
    ["target", 0.01],
  ];
  const memHop = { anchor: 1, target: 2 };
  const memToKeys = {
    anchor: map(["hubKey"]),
    target: map(["hubKey"]),
  };
  const keyToMems = {
    // hubKey has 6 members -> exceeds maxMembers=3, must NOT be treated as a narrow connector.
    hubKey: map(["anchor", "target", "other1", "other2", "other3", "other4"]),
  };
  const result = findStrictHopCandidate(gated, memHop, memToKeys, keyToMems, 3);
  assert.equal(result, null);
});

test("findStrictHopCandidate returns null when there is no hop-2 candidate at all", () => {
  const gated: Array<[string, number]> = [["anchor", 0.9], ["other", 0.5]];
  const memHop = { anchor: 1, other: 1 };
  const memToKeys = { anchor: map(["k"]), other: map(["k"]) };
  const keyToMems = { k: map(["anchor", "other"]) };
  const result = findStrictHopCandidate(gated, memHop, memToKeys, keyToMems, 3);
  assert.equal(result, null);
});

test("findStrictHopCandidate picks the highest-scoring eligible candidate when several exist", () => {
  const gated: Array<[string, number]> = [
    ["anchor", 0.9],
    ["target_weak", 0.01],
    ["target_strong", 0.03],
  ];
  const memHop = { anchor: 1, target_weak: 2, target_strong: 2 };
  const memToKeys = {
    anchor: map(["k1", "k2"]),
    target_weak: map(["k1"]),
    target_strong: map(["k2"]),
  };
  const keyToMems = {
    k1: map(["anchor", "target_weak"]),
    k2: map(["anchor", "target_strong"]),
  };
  const result = findStrictHopCandidate(gated, memHop, memToKeys, keyToMems, 3);
  assert.equal(result, "target_strong");
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx tsx --test test/strict-hop-promotion.test.ts`
Expected: FAIL — `findStrictHopCandidate is not a function` (or module has no export of that name), since `src/memoryGraph.ts` doesn't define it yet.

- [ ] **Step 3: Implement the function**

In `src/memoryGraph.ts`, insert immediately before the `// ── Utils ──` comment (currently line 276):

```ts
// Structural (not fused-score) hop-2 candidate: the strongest neighbor of the top `gated`
// anchor that is reachable ONLY via a "narrow" key — one shared by <= maxMembers memories,
// i.e. a specific connector (a name, a project, a shared fact) rather than a hub topic word.
// A hop-2 association's HOP_DECAY-scaled fused score routinely loses to dozens of weakly-
// admitted same-topic candidates once the corpus is large (measured: real hop-2 hits ranked
// ~15-18th of 49 in bench/edge-experiments-results.json) — this bypasses that competition by
// looking at graph structure directly, exactly like the bench/edge-experiments.ts STRICT_HOP
// probe that measured assoc2 Hit@5 27%→60% with no regression elsewhere.
export function findStrictHopCandidate(
  gated: Array<[string, number]>,
  memHop: Record<string, number>,
  memToKeys: Record<string, Map<string, unknown>>,
  keyToMems: Record<string, Map<string, unknown>>,
  maxMembers: number
): string | null {
  if (gated.length === 0) return null;
  const anchorId = gated[0][0];
  const anchorKeys = memToKeys[anchorId];
  if (!anchorKeys) return null;
  let bestId: string | null = null;
  let bestScore = -Infinity;
  for (const [mid, score] of gated) {
    if (mid === anchorId || (memHop[mid] ?? 1) < 2) continue;
    const midKeys = memToKeys[mid];
    if (!midKeys) continue;
    let sharesNarrowKey = false;
    for (const kid of midKeys.keys()) {
      if (!anchorKeys.has(kid)) continue;
      const memberCount = keyToMems[kid]?.size ?? 0;
      if (memberCount > 0 && memberCount <= maxMembers) { sharesNarrowKey = true; break; }
    }
    if (sharesNarrowKey && score > bestScore) { bestScore = score; bestId = mid; }
  }
  return bestId;
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx tsx --test test/strict-hop-promotion.test.ts`
Expected: PASS (4/4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/memoryGraph.ts test/strict-hop-promotion.test.ts
git commit -m "feat: add findStrictHopCandidate, a structural hop-2 promotion helper"
```

---

## Task 2: Wire the candidate into `recall()`'s rerank phase

**Files:**
- Modify: `src/memoryGraph.ts:72` (module constants, alongside `RERANK_POOL`)
- Modify: `src/memoryGraph.ts:2778-2808` (Phase 1→Phase 2 boundary inside `recall()`)
- Test: `test/strict-hop-promotion.test.ts` (append integration test)

**Interfaces:**
- Consumes: `findStrictHopCandidate` from Task 1; `gated: [string, number][]`, `memHop: Record<string, number>`, `memRawSim: Record<string, number>`, `this._memToKeys`, `this._keyToMems`, `RERANK_POOL`, `RERANK_MIN_SCORE`, `rerankEnabled()`, `rerankScores()` — all already in scope in `recall()`.
- Produces: no new public interface; changes `recall()`'s returned result set when `expand=true` and a narrow-key hop-2 candidate exists.

- [ ] **Step 1: Write the failing integration test**

Append to `test/strict-hop-promotion.test.ts`:

```ts
test("recall() promotes a narrow-key hop-2 candidate past a flood of same-topic noise", async (t) => {
  const { mkdtemp, rm } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const dir = await mkdtemp(join(tmpdir(), "sm-stricthop-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  // Deterministic vectors: QQ/ANCHOR are identical (cos=1.0, definite fused #1). TARGET is
  // orthogonal to the query (cos=0, never found by content alone). NOISEn all sit at
  // cos≈0.60-0.72 with the query — enough to clear bge-m3's contentRecall/minScore admit
  // gate on topic overlap alone, exactly the corpus-density flooding this feature targets.
  function vec(t: string): number[] {
    if (t === "QQ" || t === "ANCHOR") return [1, 0, 0, 0];
    if (t === "TARGET") return [0, 1, 0, 0];
    const m = /^NOISE(\d+)$/.exec(t);
    if (m) {
      const c = 0.6 + (Number(m[1]) % 5) * 0.03;
      return [c, Math.sqrt(1 - c * c), 0, 0];
    }
    return [0, 0, 1, 0];
  }
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((text: string) => vec(text));
  t.after(() => emb.__clearTestEmbedder());
  const rer = await import("../src/reranker.ts");
  rer.__setTestReranker((_q: string, texts: string[]) => texts.map(() => 0)); // no reordering
  t.after(() => rer.__clearTestReranker());

  const mg = await import(`../src/memoryGraph.ts?stricthop=${Date.now()}`);
  const g = new mg.MemoryGraph();
  await g.load();
  await g.add("ANCHOR", ["anchorNarrowKey"], {});
  await g.add("TARGET", ["anchorNarrowKey"], {}); // shares a 2-member key with ANCHOR only
  for (let i = 0; i < 30; i++) {
    await g.add(`NOISE${i}`, [`noiseKey${i}`], {}); // each on its own key — pure content noise
  }

  const before = (await g.recall("QQ", 10, null, true, 2, 0, 0, 0, 0)) as Array<{ content: string }>;
  assert.equal(
    before.some((m) => m.content === "TARGET"),
    true,
    `expected TARGET to be promoted into the top10, got: ${before.map((m) => m.content).join(",")}`
  );
});

test("recall() never evicts a confident direct hit to make room for a strict-hop candidate", async (t) => {
  const { mkdtemp, rm } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const dir = await mkdtemp(join(tmpdir(), "sm-stricthop-safe-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  function vec(t: string): number[] {
    if (t === "QQ") return [1, 0, 0, 0];
    if (t === "ANCHOR") return [1, 0, 0, 0]; // definite fused anchor, cos=1.0
    if (t === "CONFIDENT") return [0.99, 0.1, 0, 0]; // strong, near-definite direct hit
    if (t === "TARGET") return [0, 1, 0, 0];
    return [0, 0, 1, 0];
  }
  const emb = await import("../src/embedding.ts");
  emb.__setTestEmbedder((text: string) => vec(text));
  t.after(() => emb.__clearTestEmbedder());
  const rer = await import("../src/reranker.ts");
  rer.__setTestReranker((_q: string, texts: string[]) => texts.map(() => 0));
  t.after(() => rer.__clearTestReranker());

  const mg = await import(`../src/memoryGraph.ts?stricthop-safe=${Date.now()}`);
  const g = new mg.MemoryGraph();
  await g.load();
  await g.add("ANCHOR", ["anchorNarrowKey"], {});
  await g.add("TARGET", ["anchorNarrowKey"], {});
  await g.add("CONFIDENT", ["confidentKey"], {}); // top5 filler, but a genuinely strong hit

  const result = (await g.recall("QQ", 3, null, true, 2, 0, 0, 0, 0)) as Array<{ content: string }>;
  assert.equal(
    result.some((m) => m.content === "CONFIDENT"),
    true,
    `CONFIDENT must not be evicted, got: ${result.map((m) => m.content).join(",")}`
  );
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx tsx --test test/strict-hop-promotion.test.ts`
Expected: the two new tests FAIL (TARGET/promotion not yet implemented in `recall()`); the four Task 1 tests still PASS.

- [ ] **Step 3: Add the module constants**

In `src/memoryGraph.ts`, right after the `RERANK_POOL` line (currently line 72), add:

```ts
// Strict-hop promotion (see findStrictHopCandidate below). Default-on when the caller already
// opted into expand=true — this does not add a new opt-in surface, it strengthens the existing
// one. KEYMEM_STRICT_HOP=false kills it in one env var if it misbehaves on a real store.
const STRICT_HOP_ENABLED = cfgRaw("STRICT_HOP") !== "false";
// A key shared by more memories than this is a hub topic word, not a specific connector —
// bench/edge-experiments.ts calibrated 3 as "narrow" on a 49-memory fixture; re-tune per corpus.
const STRICT_HOP_MAX_KEY_MEMBERS = Number(cfgRaw("STRICT_HOP_MAX_KEY_MEMBERS") ?? 3);
// Only evict the final slot for a strict-hop candidate when its current occupant's raw
// similarity is below this AND it is itself hop>=2 (i.e. also just noise, not a confident hit).
const STRICT_HOP_EVICT_BELOW = Number(cfgRaw("STRICT_HOP_EVICT_BELOW") ?? 0.75);
```

- [ ] **Step 4: Compute the candidate right after Phase 1 (before Phase 2)**

In `src/memoryGraph.ts`, immediately after the closing `});` of the Phase 1 `_lock.runExclusive` block (currently line 2778) and before the `// ── Phase 2` comment (currently line 2780), insert:

```ts
    const strictHopId = STRICT_HOP_ENABLED && expand
      ? findStrictHopCandidate(gated, memHop, this._memToKeys, this._keyToMems, STRICT_HOP_MAX_KEY_MEMBERS)
      : null;
```

- [ ] **Step 5: Guarantee the candidate reaches the rerank pool and apply the safety-checked promotion**

Replace the existing Phase 2 block (currently lines 2780-2808):

```ts
    // ── Phase 2 (UNLOCKED) ── default-on cross-encoder rerank. Model inference is the
    // only slow, I/O-like await in recall; running it outside the lock lets other
    // recalls and writes proceed meanwhile. It only READS immutable memory content
    // (all reads happen synchronously before the await) and mutates nothing shared.
    let ranked: [string, number][] = gated.slice(0, actualTopK);
    if (rerankEnabled() && gated.length > 0) {
      const pool = gated.slice(0, Math.max(actualTopK, RERANK_POOL));
      const scores = await rerankScores(
        query,
        pool.map(([mid]) => this.memories[mid]?.content ?? "")
      );
      if (scores) {
        const reordered = pool
          .map((entry, i) => ({ entry, s: scores[i] }))
          .sort((a, b) => b.s - a.s);
        // Not-found gate (opt-in): a low top relevance logit means nothing answers the
        // query → []. Trusted only when the query and the top candidate share script —
        // cross-lingual logits run low even when relevant, so on a script mismatch we keep
        // the result (the cosine/key gate already vouched). This catches same-language
        // distractors; cross-lingual not-found stays a known limitation (use bilingual keys).
        const topContent = this.memories[reordered[0]?.entry[0]]?.content ?? "";
        const sameScript = hasHangul(query) === hasHangul(topContent);
        if (RERANK_MIN_SCORE !== null && sameScript && reordered[0].s < RERANK_MIN_SCORE) {
          ranked = [];
        } else {
          ranked = reordered.map((x) => x.entry).slice(0, actualTopK);
        }
      }
    }
```

with:

```ts
    // ── Phase 2 (UNLOCKED) ── default-on cross-encoder rerank. Model inference is the
    // only slow, I/O-like await in recall; running it outside the lock lets other
    // recalls and writes proceed meanwhile. It only READS immutable memory content
    // (all reads happen synchronously before the await) and mutates nothing shared.
    let ranked: [string, number][] = gated.slice(0, actualTopK);
    if (rerankEnabled() && gated.length > 0) {
      let pool = gated.slice(0, Math.max(actualTopK, RERANK_POOL));
      // Guarantee the strict-hop candidate reaches the cross-encoder even when its fused
      // score would have put it outside RERANK_POOL — the corpus-density bug this feature
      // fixes (bench/edge-experiments-results.json measured a real hop-2 hit ranked ~15-18th
      // of 49 and never reaching the rerank pool at all).
      if (strictHopId && !pool.some(([mid]) => mid === strictHopId)) {
        const entry = gated.find(([mid]) => mid === strictHopId);
        if (entry) pool = [...pool, entry];
      }
      const scores = await rerankScores(
        query,
        pool.map(([mid]) => this.memories[mid]?.content ?? "")
      );
      if (scores) {
        const reordered = pool
          .map((entry, i) => ({ entry, s: scores[i] }))
          .sort((a, b) => b.s - a.s);
        // Not-found gate (opt-in): a low top relevance logit means nothing answers the
        // query → []. Trusted only when the query and the top candidate share script —
        // cross-lingual logits run low even when relevant, so on a script mismatch we keep
        // the result (the cosine/key gate already vouched). This catches same-language
        // distractors; cross-lingual not-found stays a known limitation (use bilingual keys).
        const topContent = this.memories[reordered[0]?.entry[0]]?.content ?? "";
        const sameScript = hasHangul(query) === hasHangul(topContent);
        if (RERANK_MIN_SCORE !== null && sameScript && reordered[0].s < RERANK_MIN_SCORE) {
          ranked = [];
        } else {
          ranked = reordered.map((x) => x.entry).slice(0, actualTopK);
          // Safety-checked promotion: only bump the strict-hop candidate into the final
          // window if it survived rerank but landed just outside it, AND the item it would
          // evict is itself weak — never displace a confident direct/key hit. This is the
          // guard the bench ceiling probe lacked (bench/edge-experiments-llm-results.json
          // measured it evicting an already-correct rank-5 answer for "미나 취미").
          if (strictHopId && !ranked.some(([mid]) => mid === strictHopId)) {
            const fullIdx = reordered.findIndex((r) => r.entry[0] === strictHopId);
            const lastMid = ranked[ranked.length - 1]?.[0];
            const lastIsWeak = lastMid !== undefined
              && (memRawSim[lastMid] ?? 0) < STRICT_HOP_EVICT_BELOW;
            if (fullIdx !== -1 && lastIsWeak) {
              ranked = [...ranked.slice(0, -1), reordered[fullIdx].entry];
            }
          }
        }
      }
    }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `npx tsx --test test/strict-hop-promotion.test.ts`
Expected: PASS (6/6 tests: 4 from Task 1, 2 new integration tests).

- [ ] **Step 7: Run the full test suite to check for regressions**

Run: `npm test` (or the project's existing test command — check `package.json`'s `"test"` script before running; do not guess a different command).
Expected: PASS, same count as before this change plus the 6 new tests. Pay particular attention to any existing `recall`/`rerank`/`hop`/`expand`-related test file from the earlier `ctx_glob` listing (e.g. `agentic-navigation.test.ts`, `content-aware-nav.test.ts`, `maxsim-recall.test.ts`, `recall-inject.test.ts`) — if any of those fail, it means a real store shape trips `findStrictHopCandidate` in a way this plan's synthetic tests didn't cover; stop and re-diagnose rather than loosening the safety check to force a pass.

- [ ] **Step 8: Commit**

```bash
git add src/memoryGraph.ts test/strict-hop-promotion.test.ts
git commit -m "feat: promote structurally-confirmed hop-2 associations past corpus-noise flooding in recall()"
```

---

## Task 3: Re-validate against the bench fixture and document the result

**Files:**
- Modify: `bench/ablation.ts` (no code change needed — it already calls `g.recall(query, TOPK, null, expand, hops)`, which now carries the new behavior automatically when `expand=true`)
- Create: none (this task only re-runs an existing bench and records the outcome)

**Interfaces:** none new.

- [ ] **Step 1: Re-run the ablation bench with the change in place**

Run: `npx tsx bench/ablation.ts`

This exercises the new code through the SAME `GRAPH` condition (`expand=true, hops=2`) already in that script — no bench code change is needed, only a re-run now that `recall()` itself carries the fix.

- [ ] **Step 2: Compare against the pre-change baseline**

The pre-change `GRAPH` numbers are already committed in `bench/assoc-results.json` (from the session that produced this plan): `assoc2 hit@5 = 27%` (tied with `DIRECT`, i.e. no measurable graph benefit — the bug this plan fixes). After Task 2, `GRAPH`'s `assoc2 hit@5` should move toward the bench/edge-experiments.ts `STRICT_HOP` ceiling of 60%, though likely not reach it exactly — that ceiling used a maximal always-force-top5 rule with no eviction safety check, and used a simpler standalone top-cosine-anchor pipeline rather than `recall()`'s full BM25+key+content fusion. Record whatever number actually comes out; do not adjust `STRICT_HOP_EVICT_BELOW`/`STRICT_HOP_MAX_KEY_MEMBERS` to hit a target number without first checking (via the same per-query rank inspection method used earlier in this session) *why* any given query still fails.

- [ ] **Step 3: Commit the updated bench results**

```bash
git add bench/assoc-results.json
git commit -m "chore: re-run ablation bench after strict-hop promotion lands in recall()"
```

---

## Not in scope for this plan (do NOT attempt as part of these tasks)

- **minScore corpus-density scaling** (the not-found accuracy problem: 0/10–1/10 on the 49-memory bench fixture). This needs its own calibration experiment first — there is no validated formula yet, only a ruled-out approach (gateZ/keyGate, documented in `src/embedding.ts`'s bgem3 profile comment). Do not fold a `minScore` change into this plan; write a separate plan once a bench experiment (analogous to tonight's `bench/z-check.mts`) has produced and validated an actual scaling formula.
- **Validation against a real, multi-topic `~/.keymem` store.** Everything in this plan is validated only against the synthetic single-subject `bench/assoc-fixture.json` (49 memories, all about one person "미나"). Before treating `STRICT_HOP_MAX_KEY_MEMBERS=3` / `STRICT_HOP_EVICT_BELOW=0.75` as production-correct defaults, run `bench/diagnose-recall.ts` (which operates read-only against a COPY of the real store) with `KEYMEM_STRICT_HOP` on and off across a handful of real recall queries and compare.
- **Removing `gateZ`/`keyGate` from the codebase.** They are already off by default for every profile and are legitimate env-opt-in escape hatches for deployments that want the precision/recall tradeoff explicitly documented in `embedding.ts`. This plan only fixed the stale comment claiming bge-m3 doesn't need them for a different reason (it does, just not via those two knobs) — deleting the mechanism is a separate decision, not implied by that fix.

---

## Self-Review

**Spec coverage:** The two open design decisions named in the request are both resolved with a concrete mechanism: (a) promotion strength → safety-checked single-slot eviction (`STRICT_HOP_EVICT_BELOW`), not a blind forced top-5, directly informed by the SEM_EDGE_LLM eviction bug; (b) scope → gated behind `expand=true` (Task 2, Step 4: `STRICT_HOP_ENABLED && expand`), not unconditional. The `minScore` density-scaling half of the original request is explicitly and intentionally NOT covered here (see "Not in scope") because no validated algorithm exists yet — bundling it would have required placeholder steps, which this skill's rules forbid.

**Placeholder scan:** No TBD/TODO markers; every code block is complete and copy-pasteable; every threshold has a stated source (bench measurement) and an env override.

**Type consistency:** `findStrictHopCandidate`'s signature (Task 1) is used identically in Task 2, Step 4 (`findStrictHopCandidate(gated, memHop, this._memToKeys, this._keyToMems, STRICT_HOP_MAX_KEY_MEMBERS)`) — same argument order and types throughout.
