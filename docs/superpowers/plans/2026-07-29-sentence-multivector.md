# Sentence Multi-Vector + Binary Vector Sidecar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix content-path dilution for multi-fact memories via per-sentence embeddings scored by max-sim, and move all vectors out of graph.json into a binary sidecar.

**Architecture:** `MemoryGraph` (src/memoryGraph.ts) keeps whole-content embeddings for dedup/fingerprint; a new in-RAM `_sentVecs: Record<mid, number[][]>` holds per-sentence vectors. All vectors persist in `vectors.bin` (Float32Array concat) + `vectors.idx.json` (id → {off,n} in float units), written atomically alongside graph.json (which stores `embedding: []` from now on). Content-similarity scoring in `recall` Dense Path B, `searchKeys`, and `nearestKeys` becomes `max(whole, best sentence)`.

**Tech Stack:** Node Buffer/Float32Array, existing bge-m3 local embeddings, node:test + `__setTestEmbedder`.

## Global Constraints

- NEVER add Claude/AI attribution to commit messages (user rule).
- Before push: `npm run build` (tsc) and `npm test` must pass.
- Loading a legacy graph.json (inline embeddings, no sidecar) must work unchanged; the next save migrates it.
- A missing/corrupt sidecar entry falls back to the existing re-embed path (`load()` already re-embeds empty embeddings) — never crash.
- Sentence vectors are an ADDITIVE signal: with `KEYMEM_SENTENCE_VECTORS=0` (or no sentence vecs stored), scoring is byte-identical to v0.19.1.
- Live-store operations: daemon stopped first; graph.json backed up before backfill.

---

### Task 1: Binary vector sidecar

**Files:**
- Create: `src/vectorStore.ts`
- Modify: `src/memoryGraph.ts` — `save()` (~L803), `load()` (~L700)
- Test: `test/vector-sidecar.test.ts`

**Interfaces (produces):**
```typescript
// src/vectorStore.ts
export type VectorIndex = { schema: 1; dim: number; entries: Record<string, { off: number; n: number }> };
export async function writeVectors(dir: string, vecs: Map<string, number[][]>): Promise<void>;
// writes vectors.bin + vectors.idx.json atomically (tmp+rename, bin first).
// Map value = list of vectors per id; entry key convention: "m:<mid>" whole memory,
// "k:<kid>" key, "s:<mid>" sentence pack (n = count*dim, vectors concatenated).
export async function readVectors(dir: string): Promise<Map<string, number[][]> | null>;
// null when no sidecar exists. Tolerates a missing bin/idx pair (returns null).
```

- [ ] **Step 1: Write failing test** (`test/vector-sidecar.test.ts`): build a graph with 2 memories + keys (test embedder), `save()`, assert graph.json contains `"embedding": []` for every key/memory and `vectors.bin`/`vectors.idx.json` exist; then `new MemoryGraph()` + `load()` in the same dir and assert embeddings restored (deep-equal to test embedder output) and recall still hits. Also: legacy load — write a graph.json WITH inline embeddings and NO sidecar by hand (JSON.stringify of the raw structure), `load()` succeeds, `save()` strips inline + creates sidecar.
- [ ] **Step 2: Run to verify failure** (`npx tsx --test test/vector-sidecar.test.ts` — fails: sidecar files absent / inline embeddings still present).
- [ ] **Step 3: Implement `vectorStore.ts`** (Buffer.concat of Float32Array buffers; idx JSON written after bin; reads validate `off+n <= totalFloats`, invalid → null).
- [ ] **Step 4: Wire `save()`**: build `vecs` Map from `this.keys` (`k:` entries), `this.memories` (`m:`), `this._sentVecs` (`s:`); serialize graph with `embedding: []` clones (do NOT mutate live objects); write sidecar inside the same `_saveLock` critical section before graph.json rename.
- [ ] **Step 5: Wire `load()`**: after parsing graph.json, `readVectors(DATA_DIR)`; for each key/memory with empty inline embedding, hydrate from `k:`/`m:` entries; hydrate `_sentVecs` from `s:` entries (split flat floats into `n/dim` vectors); anything still empty falls through to the existing re-embed fallback. If inline embeddings were present (legacy), `markDirty()` so the next flush migrates.
- [ ] **Step 6: Full suite + commit** (`npm test`; `git commit -m "feat: binary vector sidecar — strip embeddings from graph.json"`).

---

### Task 2: Sentence splitting + write-time sentence vectors

**Files:**
- Modify: `src/memoryGraph.ts` — new `splitSentences()` (module fn, exported for tests), `add()` embeds sentences; deletion paths (`forget`/supersede/expiry cleanup — find `delete this.memories[` call sites) drop `_sentVecs[mid]`
- Modify: `src/env.ts` consumers — read `KEYMEM_SENTENCE_VECTORS` via `cfgRaw`
- Test: `test/sentence-vectors.test.ts`

**Interfaces (produces):**
```typescript
export function splitSentences(content: string): string[];
// split on newlines and sentence enders ([.!?…] + whitespace), trim, drop < 10 chars,
// cap at 12. Returns [] when the content yields fewer than 2 sentences (no benefit).
const SENTENCE_VECTORS_ENABLED = (cfgRaw("SENTENCE_VECTORS") ?? "1") !== "0";
```

- [ ] **Step 1: Failing test**: `splitSentences` unit cases (multi-fact Korean note → N≥2 pieces; single short fact → []); `add()` with multi-sentence content populates `graph._sentVecs[mid]` with one vector per sentence (test embedder maps each sentence differently); `forget(mid)` removes them; save/load round-trips them (relies on Task 1).
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** — in `add()`, after the whole-content embedding (outside lock where the whole embedding is computed), when enabled and `splitSentences(content).length >= 2`, embed each sentence (`passage`) and stash under `_sentVecs[mid]` inside the locked commit. Dedup/supersede replacing a memory must replace its sentence pack too.
- [ ] **Step 4: Full suite + commit** (`git commit -m "feat: per-sentence vectors at write time"`).

---

### Task 3: Max-sim content scoring

**Files:**
- Modify: `src/memoryGraph.ts` — new private `_bestContentSims(qVec: number[]): Map<string, number>`; use it in `searchKeys` (replaces the plain memSim map build), `nearestKeys` (same), and `recall` Dense Path B (`contentSims[i]` → max with sentence sims)
- Test: `test/maxsim-recall.test.ts`

**Interfaces (produces):**
- `_bestContentSims`: one pass over whole-memory vectors + all sentence vectors, returning per-mid `max(whole, sentences…)`. Skips expired/superseded checks (callers keep their own `skip()` semantics).

- [ ] **Step 1: Failing test**: memory whose whole-content vector sits below the content gate vs the query but with one sentence vector above it (test embedder: whole → cos 0.30, sentence #2 → cos 0.80). Assert `searchKeys` surfaces its key with `match_type: "content"` and `recall` returns the memory; with `KEYMEM_SENTENCE_VECTORS=0` and no stored sentence vecs, both miss (byte-identical fallback).
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** `_bestContentSims` and swap the three call sites.
- [ ] **Step 4: Full suite + commit** (`git commit -m "feat: max-sim content scoring over sentence vectors"`).

---

### Task 4: Live-store migration + backfill

**Files:**
- Create: `bench/backfill-sentence-vectors.ts` — loads the graph from `<data-dir>` (arg), computes sentence vectors for memories with `namespace !== "bench"` lacking them, saves (which also migrates everything to the sidecar), prints counts.

- [ ] **Step 1: Write script** (constructs `MemoryGraph` with `KEYMEM_DATA_DIR` already pointing at the dir; embed real bge-m3 — NOT the test embedder).
- [ ] **Step 2: Dry-run on the scratch copy**; verify: graph.json shrinks to a few MB, sidecar ~15–25 MB, backfilled count ≈ non-bench multi-sentence memories, `bench/real-eval.ts` still runs against the migrated copy.
- [ ] **Step 3: Live run**: `pkill -f dist/daemon.js`; `cp ~/.super-memory/graph.json ~/.super-memory/graph.json.pre-sidecar.bak`; run script on `~/.super-memory`; verify sizes + memory_stats over MCP after respawn.
- [ ] **Step 4: Commit script.**

---

### Task 5: Release v0.20.0 + after-measurement

- [ ] **Step 1:** Add multi-fact sub-fact eval cases to `bench/real-eval.ts` CASES (e.g. `{ query: "단일언어 경고", expect: "writeHints", ns: "keymem" }`, `{ query: "no_match 힌트", expect: "nearest_keys", ns: "keymem" }`).
- [ ] **Step 2:** package.json + server.ts version `0.20.0`; CHANGELOG entry (sidecar, sentence max-sim, backfill script, measured dilution numbers: sub-fact queries +0.07~0.19 cosine via best-sentence).
- [ ] **Step 3:** `npm run build && npm test` — all green.
- [ ] **Step 4:** Re-run eval on a fresh copy of the migrated live store; compare to baseline; record.
- [ ] **Step 5:** Commit, push origin main, restart daemon, live-verify (memory_stats + one recall).

## Self-Review Notes
- Fallbacks: sidecar missing → inline → re-embed (all three paths exist after Task 1).
- Fingerprint swap (backend change) re-embeds whole vectors via existing machinery; sentence vecs must be DROPPED on mismatch — handled in `_ensureEmbeddingDim` region check during Task 2 Step 3.
- `KEYMEM_SENTENCE_VECTORS=0` + absent sentence packs ⇒ scoring identical to v0.19.1 (Task 3 test enforces).
