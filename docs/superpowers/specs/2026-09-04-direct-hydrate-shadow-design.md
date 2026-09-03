# Direct Hydrate Shadow Design

**Date:** 2026-09-04

## Decision

Evaluate a deterministic `top recalled key → top query-ranked memory` policy in shadow mode before exposing it to agents. Shadow mode is opt-in, records one counterfactual decision for every normal recall, and never changes the existing MCP response.

## Evidence

A pilot over 30 real historical prompts and a disposable copy of the owner's 635-memory/2,027-key store compared no memory, deliberate navigation, direct Top-1, and direct Top-2 with the cross-encoder disabled.

- Direct Top-1 scored 11.533/12 versus 10.767 for deliberate navigation and 9.533 for no memory.
- Top-1 beat deliberate navigation on 9 prompts, tied 20, and lost 1; the paired mean delta was +0.767/12 with bootstrap 95% CI +0.233 to +1.433.
- Top-2's additional +0.167/12 over Top-1 was inconclusive; its 95% CI crossed zero.
- Direct Top-1 worsened 3/30 answers versus no memory, so pollution is defined as answer degradation rather than mere exposure to a tangential association.
- The existing `recall(inject:true)` returned no memory on all 30 prompts and therefore did not instantiate this policy.

The pilot supports collecting real Top-1 decisions. It does not justify default injection or Top-2 delivery.

## Shadow event

With `KEYMEM_DIRECT_HYDRATE_SHADOW=true`, every normal, non-inject recall appends one schema-v1 JSON object to `<data-dir>/direct-hydrate-shadow.jsonl`.

The event contains the query, raw context, namespace, optional host session coordinates, and a decision with status `candidate`, `no_key`, or `no_memory`. A candidate includes the selected key, handle scores, validity, and at most 2,000 characters of memory content.

The event is evaluation data only. It is not returned to the caller, does not increment access or depth, does not change link weight, does not learn aliases, and does not confirm freshness. Logging failure is reported to stderr but cannot fail recall.

## Evaluation gate

Join events to host transcripts by `host.session_id` and `host.turn`. Compare the real answer against a counterfactual answer with the recorded candidate, tracking answer quality, abstention, stale-memory misuse, answer-level degradation, tokens, and latency. Only after the shadow sample confirms the pilot should a separate change propose agent-visible delivery.
