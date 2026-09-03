# Direct Hydrate Shadow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record the top memory that a deterministic `top key → top handle` policy would hydrate for every normal recall, without changing the public response or reinforcing memory state.

**Architecture:** `MemoryGraph.directHydrateTop1()` reuses the already-ranked top key, ranks that key's handles with the raw utterance, and passively copies the winning memory. An opt-in server path writes one local JSONL event per normal recall, including no-candidate decisions and host transcript coordinates, while returning the existing recall payload byte-for-byte.

**Tech Stack:** TypeScript, Node.js test runner, MCP in-memory transport, JSONL

**Spec:** `docs/superpowers/specs/2026-09-04-direct-hydrate-shadow-design.md`

## Global Constraints

- Gate the feature behind `KEYMEM_DIRECT_HYDRATE_SHADOW=true`; default behavior is unchanged.
- Hydration must not increment access/depth, reinforce links, learn aliases, or confirm freshness.
- Record both candidate and no-candidate decisions so offline coverage is measurable.
- Keep the normal `recall` response shape unchanged.
- Rank within the top key using the raw `context` utterance when present.
- Store at most 2,000 characters of candidate content in the local shadow log.
- Shadow logging failures must never fail the recall request.

---

### Task 1: Passive direct Top-1 selection

**Files:**
- Modify: `src/memoryGraph.ts`
- Test: `test/direct-hydrate-shadow.test.ts`

**Interfaces:**
- Consumes: the top item returned by `searchKeys()` and a full ranking query.
- Produces: `MemoryGraph.directHydrateTop1(topKey, query, namespace)` returning a candidate, `no_key`, or `no_memory` decision.

- [x] **Step 1: Write the failing graph test**

Create a deterministic two-memory fixture. Assert that the method selects the query-relevant memory under the supplied top key and that access count, depth, and link weight remain unchanged.

- [x] **Step 2: Run test to verify it fails**

Run: `pnpm exec tsx --test test/direct-hydrate-shadow.test.ts`
Expected: FAIL because `directHydrateTop1` does not exist.

- [x] **Step 3: Implement the minimal passive selector**

Call `readKey(topKey.key_id, {query, namespace, limit: 1})`, then copy the selected internal memory under the graph lock without calling `readMemory`. Reuse `truncateInjectedContent(..., 2000)` and include handle score, validity, and `reinforced: false`.

- [x] **Step 4: Run the focused test**

Run: `pnpm exec tsx --test test/direct-hydrate-shadow.test.ts`
Expected: graph test PASS.

### Task 2: Opt-in shadow JSONL integration

**Files:**
- Create: `src/directHydrateShadow.ts`
- Modify: `src/server.ts`
- Modify: `test/direct-hydrate-shadow.test.ts`
- Modify: `README.md`

**Interfaces:**
- Consumes: `DirectHydrateDecision`, recall query/context/namespace, and optional host transcript coordinates.
- Produces: `<data-dir>/direct-hydrate-shadow.jsonl`, schema version 1.

- [x] **Step 1: Write the failing MCP integration tests**

Enable the flag in an isolated data directory, call normal `recall`, assert the returned JSON still contains only key results, then assert the JSONL event contains the candidate. Call a no-match query and assert a `no_key` event is also written.

- [x] **Step 2: Run test to verify it fails**

Run: `pnpm exec tsx --test test/direct-hydrate-shadow.test.ts`
Expected: FAIL because the JSONL file is absent.

- [x] **Step 3: Implement recorder and server hook**

Add `directHydrateShadowEnabled()` and `recordDirectHydrateShadow()` in the focused module. In the normal recall branch, compute the decision from the existing key results, use `context ?? query` for within-key ranking, resolve the host link, and await the append inside a catch that logs but suppresses telemetry failures.

- [x] **Step 4: Document activation and local log semantics**

Document the environment flag, log path, privacy scope, passive/non-reinforcing guarantee, and JSONL fields. State explicitly that shadow data is not returned to the model.

- [x] **Step 5: Run focused and full verification**

Run: `pnpm exec tsx --test test/direct-hydrate-shadow.test.ts test/recall-inject.test.ts test/context-dual-path.test.ts test/recall-reinforce.test.ts`
Expected: PASS.

Run: `pnpm test && pnpm run build`
Expected: PASS.

Execution note: the local pnpm supply-chain hook rewrote `pnpm-workspace.yaml` with invalid placeholder values, so verification used the equivalent checked-in scripts via `npm test` and `npm run build`; the generated placeholder edit was removed.
