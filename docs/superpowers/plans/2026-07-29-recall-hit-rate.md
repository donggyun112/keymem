# Recall Hit-Rate Improvement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise keymem recall hit rate by quarantining benchmark data pollution, adding near-miss key hints, dual-path (keyword+utterance) query embedding, a calibrated short-query content gate, and write-time key-quality feedback.

**Architecture:** All retrieval logic lives in `MemoryGraph` (src/memoryGraph.ts); the MCP surface is src/server.ts. The agent-facing `recall` tool calls `graph.searchKeys()` (key clusters), while `graph.recall()` serves recallInject/recall_memories. Data lives in `~/.super-memory/graph.json` (93 MB, embeddings inline), served by a daemon (`dist/daemon.js`) that the MCP shim respawns automatically after a kill.

**Tech Stack:** TypeScript (tsc build), node:test via `npm test` (tsx --test), fastembed bge-m3 local embeddings, `__setTestEmbedder` seam for deterministic tests.

## Global Constraints

- NEVER add Claude/AI attribution to commit messages (user rule).
- Before push: `npm run build` (tsc) and `npm test` must pass.
- bgem3 threshold changes must document measured calibration numbers in comments (repo convention, see THRESHOLD_PROFILES).
- Non-empty `recall` responses keep their current shape (array of key clusters) — only the EMPTY case may change shape.
- Backup before any mutation of `~/.super-memory/graph.json`; daemon must be stopped while the file is rewritten (it holds state in RAM and would clobber on save).
- Baseline measurements (Task 1) run against a COPY of graph.json via `KEYMEM_DATA_DIR`, never the live store.

---

### Task 1: Real-workload eval baseline (`bench/real-eval.ts`)

**Files:**
- Create: `bench/real-eval.ts`

**Interfaces:**
- Produces: CLI script `npx tsx bench/real-eval.ts <data-dir>` printing per-case hit/miss and totals for (a) memory-level `graph.recall()` hit@5 and (b) key-level `searchKeys→readKey` hit@(3 keys × 5 mems). Task 7 re-runs it unchanged for the after-measurement.

- [ ] **Step 1: Copy live store to scratch and write the eval script**

```bash
mkdir -p /tmp/keymem-eval && cp ~/.super-memory/graph.json /tmp/keymem-eval/graph.json
```

```typescript
// bench/real-eval.ts — hit-rate eval over REAL stored facts (run against a COPY).
// Usage: KEYMEM_DATA_DIR=/tmp/keymem-eval EMBEDDING_BACKEND=local LOCAL_EMBEDDING_MODEL=bge-m3 npx tsx bench/real-eval.ts
import { MemoryGraph } from "../src/memoryGraph.js";

type Case = { query: string; expect: string; ns?: string | null };
// expect = substring that must appear in the hit memory's content.
// Mix of keyword-style and sentence-style queries over facts known to exist.
const CASES: Case[] = [
  { query: "커밋 서명 규칙", expect: "서명" },
  { query: "recall 적중률", expect: "적중률" },
  { query: "벤치 데이터 오염", expect: "HotpotQA" },
  { query: "임베딩 모델", expect: "bge-m3" },
  { query: "Nexora suspend", expect: "suspend", ns: "Nexora" },
  { query: "CodeCanvas MCP 전환", expect: "MCP" },
  { query: "arcmemory 커넥터 큐", expect: "커넥터", ns: "arcmemory" },
  { query: "사용자가 어떤 임베딩 백엔드를 쓰는지", expect: "bge-m3" },
  { query: "recall 적중률이 낮은 이유", expect: "오염" },
  { query: "Nexora 원자적 resume 커밋", expect: "fcad585", ns: "Nexora" },
];

async function main() {
  const g = new MemoryGraph();
  await g.load();
  let recallHits = 0, keyHits = 0;
  for (const c of CASES) {
    const ns = c.ns ?? null;
    const mems = (await g.recall(c.query, 5, ns, false, 2, 0, undefined, undefined, undefined, 0, false)) as Array<{ content: string }>;
    const rHit = mems.some((m) => m.content.includes(c.expect));
    const keys = (await g.searchKeys(c.query, 8, ns)) as Array<{ key_id: string }>;
    let kHit = false;
    for (const k of keys.slice(0, 3)) {
      const page = (await g.readKey(k.key_id, { query: c.query, namespace: ns, limit: 5 })) as { memories: Array<{ memory_id: string }> };
      for (const h of page.memories) {
        const full = (await g.readMemory(h.memory_id, null, ns).catch(() => null)) as { content?: string } | null;
        if (full?.content?.includes(c.expect)) { kHit = true; break; }
      }
      if (kHit) break;
    }
    recallHits += rHit ? 1 : 0; keyHits += kHit ? 1 : 0;
    console.log(`${rHit ? "R✓" : "R✗"} ${kHit ? "K✓" : "K✗"}  ${c.query}`);
  }
  console.log(`\nrecall() hit@5: ${recallHits}/${CASES.length}   searchKeys→readKey hit: ${keyHits}/${CASES.length}`);
}
main();
```

Note: `readKey` result field names must be checked against the actual return shape at implementation time (`memories` vs `handles`); adjust the two property accesses if needed. `readMemory` mutates depth on the copy — irrelevant, it is a scratch copy.

- [ ] **Step 2: Run baseline and record output**

Run: `cd ~/Project/super-memory && KEYMEM_DATA_DIR=/tmp/keymem-eval EMBEDDING_BACKEND=local LOCAL_EMBEDDING_MODEL=bge-m3 npx tsx bench/real-eval.ts | tee /tmp/keymem-eval/baseline.txt`
Expected: script completes; totals printed. Whatever the numbers are, they are the baseline.

- [ ] **Step 3: Commit**

```bash
git add bench/real-eval.ts && git commit -m "bench: add real-workload recall hit-rate eval"
```

---

### Task 2: Quarantine benchmark data (`bench/quarantine-bench-data.ts`)

1,181 of 1,229 `default`-namespace memories are HotpotQA/Wikipedia passages (no Hangul, empty `source`). Move them to namespace `bench` — reversible, logged.

**Files:**
- Create: `bench/quarantine-bench-data.ts`

**Interfaces:**
- Consumes: `~/.super-memory/graph.json` structure `{keys, memories: {id: {content, namespace, source, ...}}, links, meta}`.
- Produces: rewritten graph.json (namespace `default`→`bench` for matched ids), `bench-moved-ids.json` rollback log next to it.

- [ ] **Step 1: Write the script**

```typescript
// bench/quarantine-bench-data.ts — move HotpotQA/wiki pollution out of `default`.
// Usage: npx tsx bench/quarantine-bench-data.ts <data-dir>   (daemon MUST be stopped)
import { readFileSync, writeFileSync, copyFileSync, renameSync } from "node:fs";
import { join } from "node:path";

const dir = process.argv[2];
if (!dir) { console.error("usage: quarantine-bench-data.ts <data-dir>"); process.exit(1); }
const file = join(dir, "graph.json");
const hasHangul = (s: string) => /[가-힣]/.test(s);

copyFileSync(file, `${file}.pre-quarantine.bak`);
const g = JSON.parse(readFileSync(file, "utf-8"));
const moved: string[] = [];
for (const [id, m] of Object.entries<any>(g.memories)) {
  const src = m.source ?? {};
  const emptySource = !src || Object.keys(src).length === 0;
  if (m.namespace === "default" && emptySource && !hasHangul(m.content ?? "")) {
    m.namespace = "bench";
    moved.push(id);
  }
}
writeFileSync(join(dir, "bench-moved-ids.json"), JSON.stringify(moved, null, 2));
const tmp = `${file}.quarantine.tmp`;
writeFileSync(tmp, JSON.stringify(g));
renameSync(tmp, file);
console.log(`moved ${moved.length} memories default→bench; backup at graph.json.pre-quarantine.bak`);
```

- [ ] **Step 2: Dry-verify on the scratch copy first**

Run: `npx tsx bench/quarantine-bench-data.ts /tmp/keymem-eval && python3 -c "import json;g=json.load(open('/tmp/keymem-eval/graph.json'));import collections;print(collections.Counter(m['namespace'] for m in g['memories'].values()).most_common(5))"`
Expected: `bench` ≈ 1181±30, `default` ≈ 48–60 (Korean + recent sourced memories stay).

- [ ] **Step 3: Stop daemon, run on live store, verify, let shim respawn daemon**

```bash
pkill -f "dist/daemon.js"
npx tsx bench/quarantine-bench-data.ts ~/.super-memory
python3 -c "import json,collections;g=json.load(open('$HOME/.super-memory/graph.json'));print(collections.Counter(m['namespace'] for m in g['memories'].values()).most_common(5))"
```
Expected: same counts as dry run; daemon respawns on next MCP call (0.17.x shim-reconnect fix).

- [ ] **Step 4: Commit script**

```bash
git add bench/quarantine-bench-data.ts && git commit -m "bench: add benchmark-data quarantine script"
```

---

### Task 3: Near-miss key hints on empty recall

**Files:**
- Modify: `src/memoryGraph.ts` (new method `nearestKeys`, after `searchKeys` ~L1317)
- Modify: `src/server.ts:542-560` (recall handler empty case), recall tool description
- Test: `test/nearest-keys.test.ts`

**Interfaces:**
- Produces: `MemoryGraph.nearestKeys(query: string, namespace?: string | null, limit = 5): Promise<Array<{key_id, concept, aliases, key_type, score, memory_count}>>` — top ungated keys by `max(keySim, bestMemberContentSim)`, namespace-active only.
- Server recall empty response becomes `{status: "no_match", nearest_keys: [...], note: string}` (non-empty responses unchanged).

- [ ] **Step 1: Write the failing test** (`test/nearest-keys.test.ts`, pattern from `test/browse-keys.test.ts`: env `KEYMEM_DATA_DIR`=tmpdir, `__setTestEmbedder`, cache-busted dynamic import)

```typescript
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;
// residence-fact ~0.5 to query (below bgem3 gates 0.62/0.55); beverage orthogonal.
function vec(text: string): number[] {
  const t = text.toLowerCase();
  if (t.includes("주소")) return [1, 0, 0];              // query
  if (t.includes("거주지")) return [0.5, 0.866, 0];      // near-miss key (cos 0.5)
  if (t.includes("강남")) return [0.45, 0.893, 0];       // its memory content
  return [0, 0, 1];                                       // unrelated
}

test("empty recall surfaces nearest ungated keys", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-nearmiss-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vec(text));
  t.after(() => embedding.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?nearmiss=${n++}`);
  const graph = new MemoryGraph();
  await graph.load();
  await graph.add("사용자는 강남에 산다", ["거주지"], { namespace: "default" });
  await graph.add("음료는 아메리카노", ["음료"], { namespace: "default" });

  const gated = await graph.searchKeys("주소", 8, "default");
  assert.equal(gated.length, 0); // below both gates → recall is empty

  const near = await graph.nearestKeys("주소", "default", 5);
  assert.ok(near.length >= 1);
  assert.equal(near[0].concept, "거주지");
  assert.ok(near[0].score > 0.4 && near[0].score < 0.62);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx tsx --test test/nearest-keys.test.ts`
Expected: FAIL — `graph.nearestKeys is not a function`

- [ ] **Step 3: Implement `nearestKeys` in memoryGraph.ts** (after `searchKeys`)

```typescript
  // Hint path for an empty recall: the best keys that FAILED the gate, so the agent
  // can retry with the store's actual vocabulary instead of dead-ending. Index
  // metadata only — same shape philosophy as searchKeys, no memory content.
  async nearestKeys(
    query: string,
    namespace?: string | null,
    limit = 5
  ): Promise<Array<{
    key_id: string; concept: string; aliases: string[];
    key_type: Key["key_type"]; score: number; memory_count: number;
  }>> {
    const cleanQuery = query.trim();
    if (!cleanQuery || Object.keys(this.keys).length === 0) return [];
    const qEmb = await embedTextAsync(cleanQuery, "query");
    this._checkDim(qEmb);
    const memIds = Object.keys(this.memories);
    const memSimArr = batchCosineSim(qEmb, memIds.map((mid) => this.memories[mid].embedding));
    const memSim = new Map<string, number>();
    for (let j = 0; j < memIds.length; j++) memSim.set(memIds[j], memSimArr[j]);
    return this._lock.runExclusive(async () => {
      const keyIds = Object.keys(this.keys);
      const sims = batchCosineSim(qEmb, keyIds.map((kid) => this.keys[kid].embedding));
      const out: Array<{ key_id: string; concept: string; aliases: string[]; key_type: Key["key_type"]; score: number; memory_count: number }> = [];
      for (let i = 0; i < keyIds.length; i++) {
        const kid = keyIds[i];
        const key = this.keys[kid];
        const activeIds = this._activeMemoryIdsForKey(kid, namespace);
        if (activeIds.length === 0) continue;
        let contentSim = 0;
        for (const mid of activeIds) contentSim = Math.max(contentSim, memSim.get(mid) ?? 0);
        out.push({
          key_id: kid, concept: key.concept, aliases: key.aliases ?? [],
          key_type: key.key_type,
          score: Math.round(Math.max(sims[i], contentSim) * 1000) / 1000,
          memory_count: activeIds.length,
        });
      }
      return out.sort((a, b) => b.score - a.score).slice(0, Math.max(1, Math.min(10, limit)));
    });
  }
```

- [ ] **Step 4: Wire the server empty case** (src/server.ts recall handler, replace the final plain return)

```typescript
          if (results.length === 0) {
            const nearest = await graph.nearestKeys(a.query as string, namespace, 5);
            const empty = {
              status: "no_match",
              nearest_keys: nearest,
              note: "No key cleared the recall gate. If a nearest_keys concept matches the topic, retry recall with that concept (or read_key it directly); otherwise browse_keys(namespace) to see the vocabulary.",
            };
            return { content: [{ type: "text", text: JSON.stringify(empty) }] };
          }
          return { content: [{ type: "text", text: JSON.stringify(results, null, 0) }] };
```

Also append to the recall tool description (same string, src/server.ts:240): `" An empty result returns {status:'no_match', nearest_keys} — the closest stored concepts below the gate; retry with one of those concepts when relevant."`

- [ ] **Step 5: Run tests**

Run: `npx tsx --test test/nearest-keys.test.ts` → PASS. Then `npm test` (full suite; recall-shape tests may assert `[]` for the MCP layer — if a server-level test breaks, update it to the new empty shape; graph-level `searchKeys` behavior is unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/memoryGraph.ts src/server.ts test/nearest-keys.test.ts
git commit -m "feat: return nearest ungated keys when recall has no match"
```

---

### Task 4: Dual-path query — `context` parameter

Keyword queries fit the KEY path (probe: exact concept 1.0); sentence/utterance cues fit the CONTENT path (+0.11–0.23 cosine). Let recall carry both.

**Files:**
- Modify: `src/memoryGraph.ts` — `searchKeys` (~L1177) and `recall` (~L1617) accept optional `contextText`
- Modify: `src/server.ts` — recall inputSchema + handler pass-through
- Test: `test/context-dual-path.test.ts`

**Interfaces:**
- Produces: `searchKeys(query, topK, namespace, contextText?: string | null)`; `recall(query, topK, namespace, expand, maxHops, minRelScore, minScore, minZ, minKeyGate, minDepth, reinforce, contextText?: string | null)`. When `contextText` is a non-empty string, the CONTENT similarity signal (memSim in searchKeys; Dense Path B `contentSims` in recall) is computed from `embedTextAsync(contextText, "query")`; key-path signals keep using the `query` embedding. Omitted/empty → byte-identical to today.

- [ ] **Step 1: Write the failing test** (`test/context-dual-path.test.ts`)

```typescript
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;
// keyword query orthogonal to content; sentence context aligned with content.
function vec(text: string): number[] {
  const t = text.toLowerCase();
  if (t.includes("취향질문")) return [1, 0, 0];          // keyword query (misses everything)
  if (t.includes("어떤 음료를 좋아하")) return [0, 1, 0]; // sentence context
  if (t.includes("아메리카노")) return [0, 0.95, 0.312]; // fact content (cos 0.95 to context)
  if (t.includes("음료취향키")) return [0, 0, 1];         // its key, orthogonal to both
  return [0.577, 0.577, 0.577];
}

test("context parameter feeds the content path only", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-ctx-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vec(text));
  t.after(() => embedding.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?ctx=${n++}`);
  const graph = new MemoryGraph();
  await graph.load();
  await graph.add("아이스 아메리카노를 즐긴다", ["음료취향키"], { namespace: "default" });

  const withoutCtx = await graph.searchKeys("취향질문", 8, "default");
  assert.equal(withoutCtx.length, 0);

  const withCtx = await graph.searchKeys("취향질문", 8, "default", "어떤 음료를 좋아하는지");
  assert.equal(withCtx.length, 1);
  assert.equal((withCtx[0] as any).match_type, "content");
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx tsx --test test/context-dual-path.test.ts`
Expected: FAIL — second assertion (searchKeys ignores 4th arg, still 0 results)

- [ ] **Step 3: Implement in searchKeys** — signature `async searchKeys(query: string, topK = 8, namespace?: string | null, contextText?: string | null)`; after `qEmb` is computed:

```typescript
    // Dual-path cue: the caller's raw utterance/sentence, when provided, drives the
    // CONTENT signal (sentence↔sentence cosine runs ~0.1–0.2 higher than keyword↔sentence
    // on bge-m3 — measured 2026-07-29), while the short keyword query keeps driving the
    // key/lexical signals it is optimal for.
    const ctx = contextText?.trim();
    const cEmb = ctx ? await embedTextAsync(ctx, "query") : qEmb;
    if (ctx) this._checkDim(cEmb);
```

and compute `memSimArr` from `cEmb` instead of `qEmb` (the `batchCosineSim(qEmb, ...)` over `memIds` becomes `batchCosineSim(cEmb, ...)`). Key sims (`sims`) keep `qEmb`.

- [ ] **Step 4: Implement in recall()** — append trailing param `contextText: string | null = null`; same `cEmb` computation next to the existing `qEmb` (both OUTSIDE the lock); in “Dense Path B” replace `batchCosineSim(qEmb, ...)` with `batchCosineSim(cEmb, ...)`. Keys/BM25/literal paths untouched (BM25 stays on `query`).

- [ ] **Step 5: Wire the server** — recall inputSchema gains:

```typescript
            context: {
              type: "string",
              description:
                "The raw user utterance or sentence this lookup serves. Keep query as short noun keywords; pass the sentence here — it drives content matching, which measures higher on sentence-shaped cues.",
            },
```

and both `graph.searchKeys(...)` and `graph.recallInject(...)` call sites pass `typeof a.context === "string" ? a.context : null` (for recallInject: thread it through to the internal `this.recall(...)` call as the new trailing arg; `recallInject` gains its own trailing `contextText` param).

- [ ] **Step 6: Run tests**

Run: `npx tsx --test test/context-dual-path.test.ts` → PASS; `npm test` → all pass (default-arg design keeps every existing call site byte-identical).

- [ ] **Step 7: Commit**

```bash
git add src/memoryGraph.ts src/server.ts test/context-dual-path.test.ts
git commit -m "feat: dual-path recall — optional context drives content matching"
```

---

### Task 5: Calibrated short-query content gate (bgem3)

Measured 2026-07-29 (bge-m3, 5 real-style cases): keyword-query related band 0.477–0.643, unrelated 0.294–0.444; the 0.55 `contentRecall` gate sits INSIDE the related band. Sentence queries: related 0.527–0.770, unrelated ≤0.570 — 0.55 stays right for them.

**Files:**
- Modify: `src/embedding.ts` — `ThresholdProfile` + profiles + `getThresholdProfile`
- Modify: `src/memoryGraph.ts` — gate selection in `searchKeys` and `recall`
- Test: `test/short-query-gate.test.ts`

**Interfaces:**
- Produces: `ThresholdProfile.contentRecallShort: number` (0 = disabled → falls back to `contentRecall`); env override `KEYMEM_CONTENT_RECALL_SHORT`; `MemoryGraph` private helper `_contentGateFor(query: string): number`.

- [ ] **Step 1: Expand the calibration probe before fixing the value.** Extend the existing scratch probe to ~12 related keyword-query/fact pairs drawn from real store facts plus all cross-pairs; confirm the related-min / unrelated-max gap still brackets 0.46. If the measured gap moves, use `(relatedMin + unrelatedMax) / 2` rounded to 2 decimals instead of 0.46, and record the numbers in the profile comment.

Run: `KEYMEM_DATA_DIR=/tmp/keymem-eval EMBEDDING_BACKEND=local LOCAL_EMBEDDING_MODEL=bge-m3 npx tsx <probe>` — record `관련 min` / `무관 max`.

- [ ] **Step 2: Write the failing test** (`test/short-query-gate.test.ts`)

```typescript
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;
// content sits at cos 0.50 to the query: passes a 0.46 short gate, fails 0.55.
function vec(text: string): number[] {
  const t = text.toLowerCase();
  if (t === "거주지") return [1, 0, 0];                          // short keyword query
  if (t.includes("이 사람이 어디에 사는지 궁금")) return [1, 0, 0]; // long query, same direction
  if (t.includes("강남에 산다")) return [0.5, 0.866, 0];          // fact content (cos 0.50)
  if (t.includes("집주소키")) return [0, 0, 1];                    // key, orthogonal
  return [0, 1, 0];
}

test("short keyword queries use the calibrated lower content gate", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-shortgate-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vec(text));
  t.after(() => embedding.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?shortgate=${n++}`);
  const graph = new MemoryGraph();
  await graph.load();
  await graph.add("사용자는 강남에 산다", ["집주소키"], { namespace: "default" });

  const short = await graph.searchKeys("거주지", 8, "default");
  assert.equal(short.length, 1); // 0.50 ≥ 0.46 short gate

  const long = await graph.searchKeys("이 사람이 어디에 사는지 궁금하다", 8, "default");
  assert.equal(long.length, 0); // long query keeps the 0.55 gate; 0.50 < 0.55
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `npx tsx --test test/short-query-gate.test.ts`
Expected: FAIL — first assertion (short query still gated at 0.55 → 0 results)

- [ ] **Step 4: Implement.** `embedding.ts`: add `contentRecallShort` to `ThresholdProfile` (doc comment: “Content gate for SHORT keyword queries (isShortConcept). Retrieval models are trained on sentence-shaped queries, so keyword↔content cosines run systematically lower; bgem3 measured 2026-07-29: related 0.477–0.643 vs unrelated ≤0.444 → gate between the bands. 0 = disabled (use contentRecall).”). Set `0.46` in the `bgem3` profile (or Step-1 midpoint), `0` in all others. In `getThresholdProfile()`: `contentRecallShort: envThreshold("CONTENT_RECALL_SHORT") ?? base.contentRecallShort`. `memoryGraph.ts`: 

```typescript
const CONTENT_RECALL_SHORT = _THRESHOLDS.contentRecallShort || CONTENT_RECALL_THRESHOLD;
```
```typescript
  private _contentGateFor(query: string): number {
    return isShortConcept(query) ? CONTENT_RECALL_SHORT : CONTENT_RECALL_THRESHOLD;
  }
```
(`isShortConcept` is already exported from embedding.ts.) In `searchKeys`, compute `const contentGate = this._contentGateFor(cleanQuery);` and use it in the L1247 gate in place of `CONTENT_RECALL_THRESHOLD`. In `recall()`, compute the same from `query` and use it in the Dense Path B threshold (L1751); additionally, when the caller left `minScore` at its default (`minScore === MIN_SCORE_THRESHOLD`) and the query is short, lower the absolute anchor gate to the same value (`minScore = Math.min(minScore, contentGate)`) so the anchor gate does not silently re-drop what the content gate admitted — explicit caller overrides are respected.

- [ ] **Step 5: Run tests**

Run: `npx tsx --test test/short-query-gate.test.ts` → PASS; `npm test` → all pass (other profiles have `contentRecallShort: 0` → fallback keeps them byte-identical).

- [ ] **Step 6: Commit**

```bash
git add src/embedding.ts src/memoryGraph.ts test/short-query-gate.test.ts
git commit -m "feat: calibrated content gate for short keyword queries (bgem3)"
```

---

### Task 6: Write-time key feedback in remember response

Server confronts the agent with index state at write time (judgment stays with the LLM; the server only forces the question): near-neighbor existing keys it should have reused, and a single-language warning (measured: EN-only keys lose up to 0.28 cosine against KO queries and can fall under the key gate).

**Files:**
- Modify: `src/memoryGraph.ts` — new method `writeHints`
- Modify: `src/server.ts` — remember handler appends `hints`
- Test: `test/write-hints.test.ts`

**Interfaces:**
- Produces: `MemoryGraph.writeHints(memoryId: string, providedKeys: string[]): Promise<{near_keys: Array<{your_key: string, existing_concept: string, key_id: string, similarity: number}>, language_note?: string} | null>` — null when nothing to say. `near_keys` lists provided keys whose nearest DISTINCT existing concept key has cosine in `[KEY_RECALL_THRESHOLD, keyMerge)` (close enough to be the same topic, not close enough to auto-merge).

- [ ] **Step 1: Write the failing test** (`test/write-hints.test.ts`)

```typescript
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;
// "적중률 개선" sits at cos 0.70 to existing "recall 적중률": ≥0.62 keyRecall, <0.86 keyMerge.
function vec(text: string): number[] {
  const t = text.toLowerCase();
  if (t.includes("recall 적중률")) return [1, 0, 0];
  if (t.includes("적중률 개선")) return [0.7, 0.7141428, 0];
  if (t.includes("기존내용")) return [0, 0, 1];
  if (t.includes("새내용")) return [0, 0.6, 0.8];
  return [0, 1, 0];
}

test("remember surfaces near-neighbor keys and single-language warning", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-hints-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.KEYMEM_SHORT_KEY_MERGE = "0";
  t.after(() => delete process.env.KEYMEM_SHORT_KEY_MERGE);
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vec(text));
  t.after(() => embedding.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?hints=${n++}`);
  const graph = new MemoryGraph();
  await graph.load();
  await graph.add("기존내용", ["recall 적중률"], { namespace: "default" });
  const [mid] = await graph.add("새내용", ["적중률 개선"], { namespace: "default" });

  const hints = await graph.writeHints(mid, ["적중률 개선"]);
  assert.ok(hints);
  assert.equal(hints!.near_keys.length, 1);
  assert.equal(hints!.near_keys[0].existing_concept, "recall 적중률");
  assert.match(hints!.language_note ?? "", /cross-lingual|양언어|alias/i);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx tsx --test test/write-hints.test.ts`
Expected: FAIL — `graph.writeHints is not a function`

- [ ] **Step 3: Implement `writeHints`** (memoryGraph.ts, near `nearestKeys`)

```typescript
  // Write-time confrontation: after a remember(), tell the agent which EXISTING keys
  // its freshly-coined keys nearly duplicate (reuse/alias candidates the auto-merge was
  // rightly too conservative to fold), and whether the key set is single-language.
  // Judgment stays with the calling LLM; this only forces the question. Read-only.
  async writeHints(
    memoryId: string,
    providedKeys: string[]
  ): Promise<{
    near_keys: Array<{ your_key: string; existing_concept: string; key_id: string; similarity: number }>;
    language_note?: string;
  } | null> {
    return this._lock.runExclusive(async () => {
      if (!(memoryId in this.memories)) return null;
      const ownKeyIds = new Set(this._memToKeys[memoryId]?.keys() ?? []);
      const nearKeys: Array<{ your_key: string; existing_concept: string; key_id: string; similarity: number }> = [];
      const upper = getThresholdProfile().keyMerge;
      for (const kid of ownKeyIds) {
        const key = this.keys[kid];
        if (!key || !providedKeys.some((p) => p.toLowerCase() === key.concept.toLowerCase())) continue;
        let best: { kid: string; sim: number } | null = null;
        for (const [otherId, other] of Object.entries(this.keys)) {
          if (ownKeyIds.has(otherId) || other.key_type === "name" || other.key_type === "proper_noun") continue;
          const sim = cosineSim(key.embedding, other.embedding);
          if (sim >= KEY_RECALL_THRESHOLD && sim < upper && (!best || sim > best.sim)) best = { kid: otherId, sim };
        }
        if (best) nearKeys.push({
          your_key: key.concept, existing_concept: this.keys[best.kid].concept,
          key_id: best.kid, similarity: Math.round(best.sim * 1000) / 1000,
        });
      }
      const allHangul = providedKeys.length > 0 && providedKeys.every((k) => hasHangul(k));
      const noneHangul = providedKeys.length > 0 && providedKeys.every((k) => !hasHangul(k));
      const languageNote = allHangul || noneHangul
        ? "All keys are single-language. Recall queries arrive in both Korean and English — include cross-lingual variants (measured: single-language keys lose up to 0.28 cosine cross-lingually and can fall below the recall gate)."
        : undefined;
      if (nearKeys.length === 0 && !languageNote) return null;
      return { near_keys: nearKeys, ...(languageNote ? { language_note: languageNote } : {}) };
    });
  }
```

(`getThresholdProfile` is already imported in memoryGraph.ts via `_THRESHOLDS`; reuse `_THRESHOLDS.keyMerge` instead of re-calling if simpler — use `const upper = _THRESHOLDS.keyMerge` at module scope access.)

- [ ] **Step 4: Wire the server** — in the remember handler after `result` is built:

```typescript
          const hints = await graph.writeHints(mid, keys);
          if (hints) result.hints = hints;
```

Append to the remember tool description: `" Keys MUST include both Korean and English variants. The response may include hints.near_keys (existing concepts your keys nearly duplicate — prefer reusing those concepts next time) and hints.language_note."`

- [ ] **Step 5: Run tests**

Run: `npx tsx --test test/write-hints.test.ts` → PASS; `npm test` → all pass.

- [ ] **Step 6: Commit**

```bash
git add src/memoryGraph.ts src/server.ts test/write-hints.test.ts
git commit -m "feat: write-time key hints — near-neighbor concepts and language warning"
```

---

### Task 7: Guidance, version, deploy, after-measurement

**Files:**
- Modify: `src/server.ts` — `SERVER_INSTRUCTIONS` (L196-216)
- Modify: `package.json` (0.18.0 → 0.19.0), `CHANGELOG.md`
- Modify: `~/.claude/CLAUDE.md` keymem section (outside repo)

- [ ] **Step 1: Update SERVER_INSTRUCTIONS** — in the “Recall first” paragraph, after the noun-keyword sentence add: `Pass the raw user utterance as context alongside the keyword query — keys match keywords, content matches sentences. On {status:'no_match'}, retry with a nearest_keys concept or browse_keys(namespace) before giving up.` In the “Remember durable facts” paragraph, change “include colloquial and cross-lingual variants” to `keys MUST span both Korean and English (single-language keys measurably fall below the cross-lingual recall gate)`.

- [ ] **Step 2: Update `~/.claude/CLAUDE.md` keymem section** to match: recall with `context`; on `no_match` use `nearest_keys`/`browse_keys`; bilingual keys mandatory; keep the existing remember/correct rules.

- [ ] **Step 3: Version + changelog** — package.json `0.19.0`; CHANGELOG entry:

```markdown
## [0.19.0] - 2026-07-29

### Added
- `recall` returns `{status:"no_match", nearest_keys}` (closest ungated concepts) instead of a bare empty array, so a miss is a retry hint rather than a dead end.
- Optional `context` on `recall`: the raw utterance drives content matching while the keyword query keeps driving key matching (dual-path cues).
- `contentRecallShort` threshold (bgem3 0.46, `KEYMEM_CONTENT_RECALL_SHORT`): calibrated content gate for short keyword queries, which embed systematically lower against sentence content.
- `remember` response `hints`: near-neighbor existing concepts and a single-language key warning.
- `bench/real-eval.ts` (real-workload hit-rate eval) and `bench/quarantine-bench-data.ts`.
```

Also bump the `Server` version string in `createMcpServer` (src/server.ts:227) to `0.19.0`.

- [ ] **Step 4: Full verification**

Run: `npm run build && npm test`
Expected: build clean, all tests pass.

- [ ] **Step 5: After-measurement** — re-copy the NOW-QUARANTINED live store and re-run the Task 1 eval:

```bash
cp ~/.super-memory/graph.json /tmp/keymem-eval/graph.json
KEYMEM_DATA_DIR=/tmp/keymem-eval EMBEDDING_BACKEND=local LOCAL_EMBEDDING_MODEL=bge-m3 npx tsx bench/real-eval.ts | tee /tmp/keymem-eval/after.txt
diff /tmp/keymem-eval/baseline.txt /tmp/keymem-eval/after.txt || true
```
Expected: hit totals ≥ baseline; record both numbers in the final report.

- [ ] **Step 6: Deploy** — restart daemon on the new build and verify over MCP:

```bash
pkill -f "dist/daemon.js"
```
Then call `memory_stats` and one `recall` with a nonsense query over MCP: daemon respawns, `no_match` + `nearest_keys` shape confirmed live.

- [ ] **Step 7: Commit + push**

```bash
git add -A && git commit -m "chore(release): v0.19.0 — recall hit-rate improvements"
git push origin main
```

---

## Self-Review Notes

- Spec coverage: pollution cleanup (T2), near-miss hints (T3), dual-path cue (T4), gate calibration (T5), write-side quality feedback (T6), guidance/deploy/eval (T1, T7). Deferred by design (YAGNI for this round): hook-based push injection, offline consolidation daemon, negative Hebbian feedback.
- Type consistency: `nearestKeys` element shape matches what the server empty-case embeds; `searchKeys`/`recall` context params are trailing optionals with `null` defaults so all existing call sites compile unchanged.
- Known checks at implementation time: `readKey` page field name in T1; exact `_THRESHOLDS` access in T6; a server-layer test may assert the old `[]` empty shape in T3 Step 5.
