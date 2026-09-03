# Keymem Recall Hit-Rate V2 Design

**Date:** 2026-09-04  
**Status:** Proposed  
**Goal:** Improve the chance that keymem surfaces the right memory near the top, especially through passive injection, without increasing false injections or adding a default runtime model.

## Problem

Keymem already reaches known memories well, but reachability is not the same as a useful hit. The current real-workload check reached all 12 known facts, while two landed below the top three. Separate live checks also found unrelated queries scoring in the same broad cosine band as valid queries. Lowering a global threshold therefore trades misses for false positives instead of cleanly improving quality.

The current pipeline has three distinct decisions:

1. `searchKeys` decides which key clusters look related.
2. `recall` fuses BM25, key, content, literal, and graph-hop signals into a ranked memory list.
3. `recallInject` filters that list and injects a small number of memories automatically.

The change will optimize these decisions as a measured pipeline. It will not redefine model-file caching, change public response shapes, enable the optional cross-encoder by default, or add an online learned model.

## Success Metrics

The primary user-facing metric is **positive inject hit@1**: for an answerable prompt, the first injected memory is the expected memory. It is reported with **inject coverage** so abstaining on every prompt cannot look successful.

Guardrails are:

- **negative false-inject rate:** fraction of unanswerable prompts that inject any memory;
- **recall hit@3:** expected memory appears in the first three direct-recall results;
- **associative both@5:** both gold supports appear for blind-key HotpotQA bridge cases;
- **comparison both@5:** direct/comparison queries do not regress when graph expansion adds noise;
- **p50/p95 latency:** no material regression from the current local bge-m3 path.

The benchmark will record the baseline before tuning. A candidate ships only if the held-out set improves positive inject hit@1, does not reduce recall hit@3, does not increase negative false-inject rate, does not regress either associative or comparison controls, and keeps p95 latency within 10% of baseline. If positive inject hit@1 is already 100% on the expanded baseline, it must stay at 100% while recall hit@3 or negative false-inject rate improves. Latency is compared using the median p95 from five runs to reduce local CPU noise. Tune and holdout cases are split by target memory, so paraphrases of one memory cannot appear on both sides.

## Hypotheses

### H1: Conditional graph expansion improves direct-query ranking

**Claim:** When a query has a strong direct anchor, unconditional two-hop expansion introduces associated but less relevant memories. Skipping expansion for those queries will improve inject hit@1 and comparison-query ranking without hurting direct recall.

**Test:** Replay the positive real-workload and HotpotQA comparison cases with expansion always on versus adaptive expansion. The hypothesis passes only if direct/comparison metrics improve or remain equal and positive inject coverage does not fall.

**Disproof:** If adaptive expansion removes expected memories or produces no ranking gain, retain unconditional expansion.

### H2: Evidence-aware injection gating beats a global cosine threshold

**Claim:** A candidate supported by multiple independent signals is safer to inject than one with the same cosine supported only by a weak semantic match. A pure threshold cannot express that distinction because valid and invalid raw scores overlap.

**Signals:** literal key/entity match, key cosine, content cosine, structured-token coverage, dense versus BM25 provenance, hop count, link specificity, freshness, and the margin between the first two candidates.

**Test:** Compare the current inject filter with a deterministic, offline-calibrated confidence rule on tune and untouched holdout cases. The hypothesis passes only if positive hit@1 improves without increasing the negative false-inject rate.

**Disproof:** If the rule only improves its tuning split or reduces coverage without improving correct top-one injections, discard it.

### H3: Weak direct anchors still need graph expansion

**Claim:** Conditional expansion can retain keymem's distinctive associative recall by expanding whenever direct evidence is weak or ambiguous.

**Test:** Run the existing blind-agent-key HotpotQA bridge benchmark. Adaptive retrieval must preserve the existing GRAPH advantage over DIRECT and must not reduce both@5.

**Disproof:** If bridge recall falls, make the expansion decision more permissive or abandon adaptive expansion.

### H4: Existing feedback should remain explicit

**Claim:** `read_memory` and `dismiss` are useful feedback, but automatically learning from passive injection would reinforce errors and create a self-amplifying ranking loop.

**Test:** Regression tests assert that injection remains non-reinforcing and that only explicit reads/dismissals change link evidence.

## Proposed Architecture

### 1. Reproducible quality harness

Add a benchmark that evaluates positive, negative, direct, associative, Korean, English, cross-language, version/hash, and namespace-sensitive cases. Each case identifies an expected memory or explicitly declares that no memory should be injected. The runner emits per-case decisions plus aggregate metrics and latency.

The existing `bench/real-eval.ts`, blind-key HotpotQA data, inject sweep, and not-found fixtures remain independent controls. The new harness composes their relevant metrics rather than replacing them.

### 2. Pure retrieval-decision module

Add a small pure module that owns two decisions:

- `shouldExpand(summary)` decides whether direct evidence is strong enough to avoid graph expansion.
- `shouldInject(candidate, runnerUp, summary)` decides whether the top candidate has sufficient evidence to enter context.

The module receives already-computed numeric/provenance features and has no graph, embedding, filesystem, or model dependency. This makes every hypothesis testable with fixed fixtures and keeps policy out of the large `MemoryGraph` method.

Initial thresholds and weights are selected by a bounded grid search on the tuning split, then frozen as documented constants. No parameters are fitted at runtime.

### 3. Minimal pipeline integration

Candidate generation remains BM25 plus dense key/content retrieval. Public `recall` behavior remains unchanged by default.

For `recallInject`, retrieval records a direct-evidence summary before graph traversal. Strong direct evidence uses the direct candidate set; weak or ambiguous evidence allows the existing two-hop expansion. The selected pool retains current BM25-only, structured-token, relative-score, freshness, and validity filters. The new confidence rule is the final abstention gate before `selectInject` chooses the top item.

This preserves one query embedding pass and avoids calling `recall` twice. Optional cross-encoder reranking remains opt-in and composes after candidate generation as it does today.

### 4. Diagnostics

Benchmark-only diagnostics expose the expansion decision, inject decision, and contributing signals. Normal MCP payloads keep their current shape and do not expose internal tuning data.

## Data Flow

1. Embed the keyword query and optional sentence context once.
2. Build BM25, key, content, and literal direct candidates.
3. Summarize direct evidence and decide whether graph expansion is warranted.
4. If warranted, run the existing shared-key and explicit-link traversal.
5. Fuse and rank candidates using the existing retrieval machinery.
6. Apply existing inject safety filters.
7. Apply the calibrated evidence/margin gate; abstain when confidence is insufficient.
8. Inject at most the configured top-K, with the existing default of one.

## Error Handling and Compatibility

- Empty stores and empty candidate sets continue to return empty arrays.
- Invalid numeric settings continue to be clamped by existing code.
- Missing diagnostic features choose the conservative path: allow expansion, but do not inject solely because data is absent.
- Public MCP schemas and non-inject `recall` results remain backward compatible.
- Passive injection remains non-reinforcing.
- Existing namespace, expiry, supersession, correction, and freshness filters remain authoritative.

## Testing Strategy

Implementation follows TDD for each hypothesis:

1. Add failing pure decision tests for strong-direct, ambiguous, associative, BM25-only, structured-token, and close-margin cases.
2. Add failing integration tests showing direct-anchor noise and weak-anchor associative recovery.
3. Capture the unchanged baseline with the quality harness.
4. Implement the smallest policy needed to pass the tests.
5. Tune only on the declared tuning split.
6. Run the holdout once for the adoption decision.
7. Run the full build and test suite plus existing retrieval benchmarks.

If no tested policy clears every adoption guardrail, the implementation is reverted and the negative result is documented. A higher hit rate on a tuning fixture alone is not a successful change.

## Expected Files

- Add `src/recallPolicy.ts` for pure adaptive-expansion and injection decisions.
- Modify `src/memoryGraph.ts` only at the direct-evidence summary, expansion branch, and inject gate.
- Add `bench/recall-quality.ts` and a versioned labeled fixture.
- Add focused policy and integration tests under `test/`.
- Update `BENCHMARKS.md` with baseline, candidate, and holdout results.

## Deferred Work

- Default-on cross-encoder reranking.
- Runtime-trained classifiers or external inference calls.
- Learning from passive injections.
- New persistent telemetry or private prompt logging.
- Broad retrieval refactors unrelated to the measured hypotheses.
