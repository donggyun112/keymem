# Default Latent Key Relations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add namespace-scoped latent key-to-key relations as an unconditional one-hop input to normal recall, learned only from confirmed `read_memory(via_key_id)` traversals and weakened by disuse.

**Architecture:** A focused `KeyRelationGraph` owns relation state, burst suppression, hysteresis, lazy wall-clock decay, missed-opportunity decay, persistence views, key rewrites, and bounded neighbor lookup. `MemoryGraph` integrates it under the existing mutex and atomic graph snapshot; `server.ts` supplies a host-turn evidence ID. `searchKeys` computes its existing direct candidates first, then merges at most one hop of active related keys without allowing graph output to count as evidence.

**Tech Stack:** TypeScript, Node.js 20+, `node:test`, existing `MemoryGraph` mutex/JSON persistence, existing bge-m3 benchmark harness.

**Spec:** `docs/superpowers/specs/2026-09-11-default-key-relations-design.md`

## Global Constraints

- This is a normal feature: no off, shadow, active, experimental, or opt-in mode exists.
- Learn only from successful `read_memory(memory_id, via_key_id)` calls.
- Exclude auto-linked keys from relation evidence.
- Scope every relation to one normalized namespace.
- Constants are wall half-life 3 days, opportunity half-life 8, promote 1.5, demote 1.0, minimum 2 distinct evidence events, maximum weight 8, score attenuation 0.85, and maximum 3 neighbors per direct key.
- Passive recall performs no mutation or persistence.
- Existing schema-v2 graphs load with an empty relation graph and save as schema v3.
- If any release gate in Task 6 fails, revert Tasks 1–5 rather than adding a feature flag.

---

### Task 1: Pure Key Relation State Machine

**Files:**
- Create: `src/keyRelations.ts`
- Create: `test/key-relations.test.ts`

**Interfaces:**
- Produces: `StoredKeyRelation`, `RelationNeighbor`, `RelationObservation`, `DEFAULT_KEY_RELATION_CONFIG`, and `KeyRelationGraph`.
- Produces: `load(records, isValid): boolean`, `serialize(): StoredKeyRelation[]`, `observe(input): void`, `neighbors(namespace, keyId): RelationNeighbor[]`, and `rewriteKey(fromId, intoId): void`.

- [ ] **Step 1: Write failing state-machine tests**

Create tests that instantiate the store with a fake clock and prove exact behavior:

```ts
const clock = { now: 1_000 };
const relations = new KeyRelationGraph(() => clock.now);
relations.observe({
  namespace: "alpha", viaKeyId: "a", selectedKeyIds: ["b"], evidenceId: "turn-1",
});
assert.equal(relations.neighbors("alpha", "a").length, 0);
relations.observe({
  namespace: "alpha", viaKeyId: "a", selectedKeyIds: ["b"], evidenceId: "turn-2",
});
assert.deepEqual(relations.neighbors("alpha", "a").map((x) => x.key_id), ["b"]);
```

Also assert: same evidence ID is capped; namespace beta cannot see alpha; a non-selected incident relation decays by `0.5 ** (1 / 8)`; four days of wall time hide a relation below the demotion threshold; weight never exceeds 8; self-relations are ignored; neighbor lookup is sorted and capped by the caller's limit.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pnpm exec tsx --test test/key-relations.test.ts`

Expected: FAIL because `src/keyRelations.ts` does not exist.

- [ ] **Step 3: Implement the minimal state machine**

Use an ordered composite ID and lazy materialization:

```ts
export interface StoredKeyRelation {
  key_a: string;
  key_b: string;
  namespace: string;
  weight: number;
  active: boolean;
  evidence_count: number;
  last_evidence_id: string | null;
  updated_at: number;
}

export interface RelationObservation {
  namespace: string;
  viaKeyId: string;
  selectedKeyIds: string[];
  evidenceId: string;
}

export interface RelationNeighbor {
  key_id: string;
  weight: number;
}
```

`observe` must materialize wall decay for every incident relation, opportunity-decay unselected relations, create/update selected relations once per evidence ID, then reconcile promotion/demotion. `neighbors` computes effective wall-decayed weight without mutating stored state and returns only relations whose persisted active state still clears the demotion threshold.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `pnpm exec tsx --test test/key-relations.test.ts`

Expected: all tests pass.

- [ ] **Step 5: Commit the state machine**

```bash
git add src/keyRelations.ts test/key-relations.test.ts
git commit -m "feat: add latent key relation state machine"
```

### Task 2: Persistence, Repair, and Key-Merge Integration

**Files:**
- Modify: `src/types.ts`
- Modify: `src/memoryGraph.ts`
- Create: `test/key-relation-persistence.test.ts`
- Modify: `test/decay-migration.test.ts`

**Interfaces:**
- Consumes: `KeyRelationGraph.load`, `serialize`, and `rewriteKey` from Task 1.
- Produces: `GraphData.key_relations?: StoredKeyRelation[]` and schema version 3 snapshots.

- [ ] **Step 1: Write failing persistence and migration tests**

Cover four fixtures:

1. schema v2 without `key_relations` loads and reports zero relations;
2. schema v3 round-trips every relation field while vectors remain in the sidecar;
3. malformed, dangling, self-linked, and namespace-invalid records are omitted on the next flush;
4. `_healFragmentedKeys` rewrites relation endpoints and conservatively merges collisions without summing weights into a promotion.

Update the existing assertion from schema version 2 to 3 only after the RED run proves the new tests fail.

- [ ] **Step 2: Run persistence tests and verify RED**

Run: `pnpm exec tsx --test test/key-relation-persistence.test.ts test/decay-migration.test.ts`

Expected: FAIL because `GraphData` and `MemoryGraph.save/load` do not carry relations.

- [ ] **Step 3: Integrate relation storage**

Add `key_relations?: StoredKeyRelation[]` to `GraphData`, construct one `KeyRelationGraph` inside `MemoryGraph`, load records after key-memory links exist, call `rewriteKey` inside `_mergeKeyInto`, serialize records in the same `graph.json` snapshot, and stamp `schemaVersion: 3`.

The load validator must require both endpoint keys to exist and each endpoint to retain at least one active memory in the relation namespace. A repair calls `markDirty()` but never aborts memory loading.

- [ ] **Step 4: Run persistence tests and verify GREEN**

Run: `pnpm exec tsx --test test/key-relation-persistence.test.ts test/decay-migration.test.ts test/model-migration.test.ts test/vector-sidecar.test.ts`

Expected: all tests pass.

- [ ] **Step 5: Commit persistence support**

```bash
git add src/types.ts src/memoryGraph.ts test/key-relation-persistence.test.ts test/decay-migration.test.ts
git commit -m "feat: persist namespace key relations"
```

### Task 3: Learn Relations from Confirmed Reads

**Files:**
- Modify: `src/memoryGraph.ts`
- Modify: `src/server.ts`
- Create: `test/key-relation-read.test.ts`
- Modify: `test/read-defer.test.ts`
- Modify: `test/host-link-source.test.ts`

**Interfaces:**
- Consumes: `KeyRelationGraph.observe` from Task 1.
- Produces: `MemoryGraph.readMemory(memoryId, viaKeyId?, namespace?, evidenceId?)`.
- Produces: `readEvidenceId(hostLink, now): string`, formatted as `agent:session_id:turn` or `local:<five-minute-bucket>`.

- [ ] **Step 1: Write failing confirmed-read tests**

Create a memory with explicit keys A, B, C and one auto-linked key D. Assert:

- a read without `via_key_id` creates no relations;
- a read via A observes A–B and A–C but not A–D;
- repeating one evidence ID cannot promote;
- a second evidence ID promotes both relations;
- reading via A into a memory selecting C opportunity-decays an existing unselected A–B relation;
- reading a memory in another namespace does not touch the first namespace;
- relation mutation stays in RAM until `flush()` and then round-trips.

Add server-level assertions that host headers produce a stable per-turn evidence ID and the fallback groups calls in the same five-minute bucket.

- [ ] **Step 2: Run confirmed-read tests and verify RED**

Run: `pnpm exec tsx --test test/key-relation-read.test.ts test/read-defer.test.ts test/host-link-source.test.ts`

Expected: FAIL because `readMemory` has no evidence ID or relation observation.

- [ ] **Step 3: Implement confirmed observation**

In `server.ts`, resolve the host link before `read_memory` and pass its evidence ID. In `MemoryGraph.readMemory`, after validating the traversal, collect explicit keys with:

```ts
const selectedKeyIds = [...(this._memToKeys[memoryId]?.keys() ?? [])]
  .filter((kid) => kid !== viaKeyId && !this._isAutoLink(kid, memoryId));
```

Call `observe` with the memory's normalized namespace. Preserve existing key-memory reinforcement and deferred `markDirty()` behavior. Do not return internal relation updates in the MCP response.

- [ ] **Step 4: Run confirmed-read tests and verify GREEN**

Run: `pnpm exec tsx --test test/key-relation-read.test.ts test/read-defer.test.ts test/host-link-source.test.ts test/agentic-navigation.test.ts`

Expected: all tests pass.

- [ ] **Step 5: Commit confirmed learning**

```bash
git add src/memoryGraph.ts src/server.ts test/key-relation-read.test.ts test/read-defer.test.ts test/host-link-source.test.ts
git commit -m "feat: learn key relations from confirmed reads"
```

### Task 4: Make Active Relations Part of Normal Recall

**Files:**
- Modify: `src/memoryGraph.ts`
- Modify: `src/recallView.ts`
- Create: `test/key-relation-recall.test.ts`
- Modify: `test/direct-hydrate-recall.test.ts`

**Interfaces:**
- Consumes: `KeyRelationGraph.neighbors` from Task 1.
- Produces: relation candidates with `match_type: "relation"`, `related_from: string`, and `relation_strength: number`.
- Preserves: existing direct candidate fields and `directHydrateTop1` input compatibility.

- [ ] **Step 1: Write failing recall tests**

Tests must prove:

- one read leaves a relation latent and search results byte-equivalent to baseline;
- two distinct reads promote it and normal `searchKeys` returns the neighbor without a flag;
- the related score equals `source_score * 0.85 * min(1, effective_weight / 2.5)` within rounding tolerance;
- the relation result ranks below its source and below a literal entity hit;
- a direct candidate wins deduplication over the same relation candidate;
- expansion is one hop only and capped at three neighbors per source;
- wall-decayed inactive relations disappear without changing serialized state;
- merely returning a relation candidate does not change its weight or evidence count;
- namespace beta cannot receive alpha relation candidates.

- [ ] **Step 2: Run recall tests and verify RED**

Run: `pnpm exec tsx --test test/key-relation-recall.test.ts test/direct-hydrate-recall.test.ts`

Expected: FAIL because `searchKeys` has no relation expansion.

- [ ] **Step 3: Merge one-hop relation candidates**

Keep the existing direct scoring loop unchanged. Sort direct candidates, seed expansion from at most the direct Top-K, request at most three neighbors per source, discard neighbors with no active memory in the requested namespace, and create relation candidates from the canonical key view. Merge by key ID, retaining the direct form on collision, sort once by the existing score/specificity rules, and slice to Top-K.

Extend `RecallKeyCandidate`/`compactRecallKeys` so the two relation fields survive normal MCP serialization. Do not add a mode argument or environment variable.

- [ ] **Step 4: Run recall tests and verify GREEN**

Run: `pnpm exec tsx --test test/key-relation-recall.test.ts test/direct-hydrate-recall.test.ts test/agentic-navigation.test.ts test/recall-inject.test.ts`

Expected: all tests pass.

- [ ] **Step 5: Commit default recall integration**

```bash
git add src/memoryGraph.ts src/recallView.ts test/key-relation-recall.test.ts test/direct-hydrate-recall.test.ts
git commit -m "feat: expand normal recall through active key relations"
```

### Task 5: Concurrency and End-to-End Compatibility

**Files:**
- Modify: `test/concurrency.test.ts`
- Modify: `test/recall-concurrency.test.ts`
- Modify: `test/e2e-two-shims.test.ts`
- Modify: `README.md`

**Interfaces:**
- Consumes: the complete default relation feature from Tasks 1–4.
- Produces: documented unconditional behavior and regression coverage across daemon/shim requests.

- [ ] **Step 1: Add failing integration assertions**

Extend concurrency storms to interleave confirmed reads, relation-expanded recalls, adds, and flushes. Parse the final `graph.json`; assert every relation has finite weight, ordered distinct endpoints, a valid namespace, and existing keys.

Extend the two-shim test so two calls in one host turn count once and a later host turn promotes. Assert the next normal recall can surface the related key with no request flag.

- [ ] **Step 2: Run integration tests and verify RED or coverage failure**

Run: `pnpm exec tsx --test test/concurrency.test.ts test/recall-concurrency.test.ts test/e2e-two-shims.test.ts`

Expected: the new relation assertions fail until all request and persistence paths are wired.

- [ ] **Step 3: Fix only integration defects and document behavior**

Keep lock order `_lock → _saveLock`; build relation snapshots synchronously while `_lock` protects mutations; perform disk I/O only through existing `save`/`flush`. Add a concise README section stating that confirmed key paths form latent relations, two distinct turns activate them, and disuse hides them again.

- [ ] **Step 4: Run integration tests and verify GREEN**

Run: `pnpm exec tsx --test test/concurrency.test.ts test/recall-concurrency.test.ts test/e2e-two-shims.test.ts`

Expected: all tests pass and the final JSON snapshot parses.

- [ ] **Step 5: Commit integration coverage**

```bash
git add test/concurrency.test.ts test/recall-concurrency.test.ts test/e2e-two-shims.test.ts README.md
git commit -m "test: cover default key relation integration"
```

### Task 6: Reproduce the Release Gates

**Files:**
- Create: `bench/key-relation-eval.ts`
- Create: `bench/key-relation-trace-replay.ts`
- Modify: `package.json`

**Interfaces:**
- Produces: `pnpm bench:key-relations` for the fixture gate.
- Produces: `pnpm bench:key-relation-trace` for aggregate-only local Codex replay; it prints counts and metrics, never memory or conversation content.

- [ ] **Step 1: Write the fixture evaluator and expected assertions**

Load `bench/assoc-fixture.json`, build the normal graph, issue the confirmed read paths that form ground-truth relations, then execute the same direct, assoc2, and notfound queries used by the existing benchmark. Exit nonzero unless assoc2 Hit@5 is at least 0.50, direct Hit@5 is exactly 1.00, and notfound accuracy is no lower than the checked-in baseline.

- [ ] **Step 2: Write the aggregate trace replay gate**

Reuse the Codex JSONL shapes already supported by `nativeTranscripts.ts`. Extract only recall key IDs and successful read outputs, exclude the current thread when `CODEX_THREAD_ID` is present, derive explicit non-auto star edges, and replay the production constants. Exit nonzero unless one-shot promotions equal zero and all 10 serveable repeat uses in the pinned local corpus remain active before reuse. Output aggregate counts only.

- [ ] **Step 3: Run both gates**

Run: `pnpm bench:key-relations`

Expected: assoc2 Hit@5 `>= 0.50`, direct Hit@5 `1.00`, notfound at or above baseline.

Run: `pnpm bench:key-relation-trace`

Expected: one-shot promotions `0`, serveable repeat hits `10/10` on the current pinned corpus.

- [ ] **Step 4: Run complete verification**

Run: `pnpm test`

Expected: zero failed tests.

Run: `pnpm build`

Expected: TypeScript compilation and asset synchronization exit 0.

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 5: Apply the binary release decision**

If every gate passes, commit the benchmark harness:

```bash
git add bench/key-relation-eval.ts bench/key-relation-trace-replay.ts package.json
git commit -m "bench: gate default key relation quality"
```

If any gate fails, revert the Task 1–5 feature commits and report the failing metric. Do not retain a disabled implementation.
