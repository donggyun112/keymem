# Evidence-Based Memory Decay

**Date:** 2026-09-03
**Status:** Design approved, pending implementation plan
**Branch:** `main`

## Problem

The current decay model is a weak ranking modifier rather than a trustworthy signal about
whether a memory is still current:

- `_timeFactor()` uses `created_at`, even though reads update `last_accessed`.
- `read_memory()` treats inspection as confirmation and always deepens the memory.
- `depth` only rises; it does not represent when the fact was last verified.
- decay is applied by `read_key()` and direct one-hop recall, but not consistently by key entry,
  graph expansion, or `related()`.
- callers cannot see that a returned fact is old, so an LLM may state stale information as current.
- an expired, not-yet-cleaned memory can participate in duplicate detection; its already-expired
  absolute TTL is then inherited by the new version.

Age does not prove that a fact is false. The system therefore needs to preserve old memories while
making uncertainty explicit and giving the agent a deliberate way to record fresh evidence.

## Goal

Introduce evidence-based soft forgetting:

1. every returned memory states how recently it was confirmed and whether verification is needed;
2. reading and confirming are separate operations;
3. freshness influences ranking consistently without silently deleting memories;
4. explicit TTL remains the only automatic hard-expiration mechanism;
5. existing graph files migrate without losing prior reinforcement history; and
6. each behavioral change is introduced as a falsifiable hypothesis with a deterministic test.

## Non-goals

- No automatic deletion based on age or score.
- No claim that freshness is objective truth or source credibility.
- No LLM-based truth judging inside the server.
- No event-sourced history ledger in this iteration.
- No time-based decay of key-to-memory link weights. Association strength and content currency are
  separate concerns; the existing reinforcement/dismissal behavior remains unchanged.
- No benchmark-driven claim that the status metadata changes model behavior. This repository can
  prove the protocol and payload contract, but a separate model evaluation is required to measure
  compliance by a particular LLM.

## Terminology

- **Access**: the agent loaded a memory. It says nothing about truth.
- **Confirmation**: the user, an authoritative source, or a direct observation supplied current
  evidence that the memory is still valid.
- **Freshness**: a time-derived score based on the last confirmation and a decay profile.
- **Depth**: accumulated confirmation strength. After this change, access alone never changes it.
- **Expiration**: a TTL policy decision that makes a memory unavailable. It is not decay.
- **Supersession**: a newer fact replaced an older one. Superseded memories remain historical data
  but are unavailable through normal retrieval.

## Considered approaches

### A. Reinterpret the existing fields only

Compute decay from `last_accessed` and keep `read_memory()` as reinforcement. This has a small diff,
but repeatedly inspecting a questionable memory makes it appear current. It cannot represent
evidence separately from exposure, so it does not solve the core problem.

### B. Evidence-based confirmation state — selected

Add explicit confirmation metadata, make reads access-only, expose freshness/status in retrieval
payloads, and add `confirm_memory`. This is a moderate persisted-schema and API change, but it makes
the semantics inspectable and testable without adding an unbounded event log.

### C. Event-source every memory interaction

Persist read, confirmation, correction, dismissal, and source events and derive all state. This
would provide the strongest audit trail, but introduces storage compaction, replay, and migration
complexity disproportionate to the current local JSON store.

## Data model

Extend `Memory` with:

```ts
type DecayProfile = "transient" | "standard" | "stable" | "permanent";
type ConfirmationEvidence = "user" | "authoritative_source" | "observation";

interface Memory {
  // existing fields remain
  last_confirmed_at: number;
  confirmation_count: number;
  decay_profile: DecayProfile;
  last_confirmation_evidence: ConfirmationEvidence | null;
  last_confirmation_source: Record<string, unknown> | null;
  last_confirmation_id: string | null;
}
```

Profiles have explicit, configurable defaults:

| Profile | Half-life | Intended use |
|---|---:|---|
| `transient` | 7 days | task state, short-lived plans, operational context |
| `standard` | 90 days | mutable preferences and ordinary project facts; default |
| `stable` | 365 days | durable historical or identity context |
| `permanent` | none | immutable definitions/facts only; opt-in |

The environment overrides are `KEYMEM_DECAY_TRANSIENT_DAYS`,
`KEYMEM_DECAY_STANDARD_DAYS`, and `KEYMEM_DECAY_STABLE_DAYS`, resolved through `cfgRaw` so legacy
`SUPER_MEMORY_*` names continue to work.

`remember` and `remember_batch` accept optional `decay_profile`, defaulting to `standard`. A newly
stored user-provided fact starts with `confirmation_count = 1`, `last_confirmed_at = created_at`, and
`last_confirmation_evidence = "user"` because the write itself originates from the current turn.

`correct` accepts optional `decay_profile` and `ttl_seconds`. A correction is a new current assertion,
so its confirmation state starts fresh. An explicitly supplied TTL is computed from correction time;
when omitted, the prior absolute TTL is preserved for backward compatibility. Correcting an already
expired memory is rejected rather than producing an immediately expired successor.

## Freshness model

For non-permanent memories:

```text
age_seconds = max(0, now - last_confirmed_at)
freshness = 2 ^ -(age_seconds / profile_half_life_seconds)
```

This makes the configured value a real half-life. `permanent` always has freshness `1.0`.

Status is defined in half-life-relative terms:

| Freshness | Status | Agent guidance |
|---:|---|---|
| `>= 0.5` | `fresh` | may be used normally |
| `>= 0.125` | `aging` | qualify as previously known; verify when currentness matters |
| `< 0.125` | `stale` | do not assert as current; verify or ask the user |

Every memory view exposes:

```ts
interface ValidityView {
  freshness: number;
  status: "fresh" | "aging" | "stale";
  age_days: number;
  last_confirmed_at: number;
  confirmation_count: number;
  decay_profile: DecayProfile;
  verification_recommended: boolean;
  verification_required: boolean;
}
```

`verification_recommended` is true for `aging` and `stale`; `verification_required` is true only for
`stale`. These fields communicate currency, not objective truth.

## Ranking policy

Candidate inclusion remains relevance-driven so age cannot make a memory unreachable. Once a memory
is a candidate, apply:

```text
freshness_factor = 0.2 + 0.8 * freshness
```

The `0.2` floor allows a fresh fact to outrank an equally relevant stale one by at most 5x while
keeping old memories available. Exact key navigation and direct reads never fail only because a
memory is stale.

Apply the same factor at every memory-ranking boundary:

- `searchKeys()`: apply it to each member memory's content-similarity contribution. Literal and
  semantic key matches remain independent of memory age.
- `readKey()`: replace the current `_timeFactor()` with the shared freshness factor.
- optional direct recall: apply it to initial memory scores.
- graph expansion: apply the target memory's factor when the target is introduced, not only the
  source memory's factor.
- `related()`: apply the target memory's factor after shared-key/explicit-link evidence is summed.
- passive injection inherits the corrected direct-recall ranking and remains non-reinforcing.

The factor is computed lazily; retrieval does not rewrite the graph merely because time passed.

## Read and confirmation behavior

### `read_memory`

`read_memory()` becomes neutral with respect to content validity:

- increment `access_count`;
- set `last_accessed`;
- keep the existing traversed link reinforcement and alias-learning behavior, because selecting a
  path is evidence about association quality;
- do not change `depth`, `last_confirmed_at`, or `confirmation_count`; and
- return the `validity` view with the content.

This deliberately separates “the key led to something worth inspecting” from “the content is still
true.” `dismiss()` continues to reverse a bad association signal and does not alter content validity.

### `confirm_memory`

Add a consistently named MCP tool:

```ts
confirm_memory({
  memory_id: string,
  evidence: "user" | "authoritative_source" | "observation",
  namespace?: string,
  source?: Record<string, unknown>
})
```

On success it:

- rejects missing, expired, or superseded memories;
- sets `last_confirmed_at = now`;
- increments `confirmation_count`;
- increments `depth` by the existing bounded `DEPTH_INCREMENT`;
- records evidence and source provenance; and
- returns the updated validity view.

The server derives `last_confirmation_id` from host agent/session/turn provenance when available.
Repeating the same confirmation ID is idempotent: it returns the current view without refreshing the
timestamp or incrementing depth/count. Without host provenance, the call remains usable but cannot
provide server-enforced turn idempotency.

Tool descriptions and both server prompt variants must say:

- never call `confirm_memory` merely because `read_memory` returned content;
- call it only after an explicit current user assertion, an authoritative current source, or direct
  observation;
- qualify `aging` information when currentness matters; and
- do not assert `stale` information as current until it is confirmed or corrected.

## API exposure

The `validity` view is included wherever memory data or a memory handle is returned:

- `read_key`
- `read_memory`
- `related`
- `list_memories`
- optional `recall_memories`
- passive injected memories

`recall` continues to return key clusters, not memory content. The following `read_key` call exposes
the validity of each candidate before the agent chooses one.

## Expiration and duplicate handling

`_findDuplicate()` and `_findContradiction()` must exclude expired memories just as retrieval does.
This prevents an expired record from becoming a duplicate/supersession target or a live
contradiction. A new `remember` after expiry creates a new active memory with a newly computed TTL.

`cleanupExpired()` remains physical cleanup. Correctness must not depend on it having run recently.

## Migration

Loading a graph without the new fields assigns:

```text
last_confirmed_at = max(created_at, last_accessed || 0)
confirmation_count = max(1, access_count)
decay_profile = "standard"
last_confirmation_evidence = null
last_confirmation_source = null
last_confirmation_id = null
```

The migration preserves the old model's interpretation that historical full reads acted as
confirmations. New behavior separates them after migration. Add `schemaVersion: 2` to `GraphData.meta`
and persist the normalized graph through the existing atomic save path. Vector sidecar behavior is
unchanged.

## Clock and testability

Replace direct ranking/lifecycle calls to `Date.now()` with a graph-local clock dependency:

```ts
new MemoryGraph({ now?: () => number })
```

The default returns epoch seconds. Tests use a mutable fake clock, allowing exact boundary checks
without sleeps or global timer mocks.

## Hypotheses and falsification tests

### H1 — expired records cannot poison new writes

**Hypothesis:** duplicate and contradiction detection that ignores expired records lets the same
content be remembered again as an active memory.

**Falsification:** create an expired memory, remember identical content with a positive TTL, and
assert the new ID is not a supersession of the expired ID and remains visible after advancing less
than its new TTL. The test must fail before changing duplicate detection.

### H2 — freshness follows confirmation, not creation or access

**Hypothesis:** ranking from `last_confirmed_at` makes a recently reconfirmed old memory outrank an
equally relevant unconfirmed old memory, while a mere read changes neither freshness nor depth.

**Falsification:** use identical relevance/link/depth, advance a fake clock, read one memory, confirm
the other, and assert only the confirmed memory becomes fresh and ranks first.

### H3 — the half-life and status contract is exact

**Hypothesis:** an explicit half-life formula produces predictable model-visible uncertainty.

**Falsification:** at 0, 1, and 3 half-lives assert freshness `1`, `0.5`, and `0.125`, with statuses
`fresh`, `fresh`, and `aging`; just beyond 3 half-lives assert `stale` and
`verification_required=true`. Permanent memories remain `1/fresh` at every age.

### H4 — all retrieval paths agree

**Hypothesis:** applying one shared freshness function at every ranking boundary prevents stale
memories from regaining priority through graph expansion or `related()`.

**Falsification:** construct fresh/stale pairs with equal semantic, key, link, and graph evidence and
assert the fresh member ranks first in `searchKeys`, `readKey`, direct recall, a second-hop expansion,
and `related`. Exact lookup must still return the stale member.

### H5 — confirmation is deliberate and idempotent

**Hypothesis:** a dedicated confirmation operation prevents exposure loops from inflating content
confidence.

**Falsification:** repeated reads change access count but not depth/count/confirmation time; one
confirmation changes them once; replaying the same confirmation ID changes nothing.

### H6 — legacy graphs retain prior meaning

**Hypothesis:** deterministic migration preserves the best historical confirmation approximation and
does not change embeddings, keys, links, TTLs, or supersession chains.

**Falsification:** load a v1 fixture with differing creation/access times and depth, assert the exact
derived fields, save/reload, and assert byte-equivalent semantic state for all pre-existing fields.

### H7 — the LLM receives an actionable contract

**Hypothesis:** every memory-bearing MCP response and system instruction contains enough information
for a caller to distinguish fresh, aging, stale, expired, and superseded behavior.

**Falsification:** schema/handler tests assert `validity` presence and prompt tests assert the required
stale/confirm rules. Actual model compliance remains a separate evaluation and is not inferred from
these tests.

## Implementation boundaries

Keep freshness calculation in a small, pure module rather than adding more responsibility to the
already-large `memoryGraph.ts`. The module owns profile parsing, half-life math, status thresholds,
ranking factor, and the validity view. `MemoryGraph` owns persistence and state transitions;
`server.ts` owns MCP schemas and agent instructions.

Expected file responsibilities:

- `src/decay.ts`: pure types, configuration, freshness/status/ranking calculations.
- `src/types.ts`: persisted confirmation/profile fields and schema metadata.
- `src/memoryGraph.ts`: clock use, migration, confirmation transition, consistent ranking, expiry
  filtering.
- `src/server.ts`: `confirm_memory`, new write arguments, validity-aware descriptions/instructions.
- `test/decay.test.ts`: pure formula and status boundaries.
- focused graph/server tests: hypotheses H1, H2, H4, H5, H6, and H7.
- `README.md` and `CHANGELOG.md`: replace the old claim that reading confirms facts; document status
  fields, profiles, confirmation, and compatibility.

## Error handling

- Unknown profile values are rejected at the MCP boundary and by graph methods.
- Invalid/non-positive configured half-lives fall back to documented defaults with one startup
  warning.
- Future timestamps clamp age to zero.
- `confirm_memory` returns not-found semantics for expired, superseded, namespace-mismatched, or
  missing IDs, matching `read_memory`'s information-hiding behavior.
- Migration treats non-finite legacy timestamps/counts as absent and uses safe defaults.

## Rollout and compatibility

- Existing tool names and arguments remain valid.
- New fields are additive in JSON responses.
- `confirm_memory` is additive.
- `depth` remains in persisted data and responses, but its documented meaning changes from repeated
  read depth to confirmed evidence depth.
- `KEYMEM_DIRECT_RECALL` remains opt-in.
- No automatic rewrite occurs solely because time passes; only normal writes, confirmation, cleanup,
  or migration persist state.

## Success criteria

- All seven hypotheses have tests observed failing for the intended reason before production changes.
- The full test suite and TypeScript build pass after implementation.
- A stale memory remains retrievable by explicit key/ID but is visibly marked stale.
- Reading never refreshes currency or deepens content confidence.
- Confirming refreshes currency exactly once per confirmation identity.
- Expired memories cannot affect duplicate or contradiction decisions.
- Every memory-bearing retrieval path uses the shared freshness policy.
- Existing graph fixtures migrate without losing prior data.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Agent calls `confirm_memory` without real evidence | Strong tool/prompt contract; explicit evidence enum; idempotency from host turn provenance |
| Freshness suppresses old but still-correct facts | No age-based inclusion gate; 0.2 ranking floor; exact navigation remains available |
| Arbitrary defaults do not fit every domain | Four explicit profiles plus environment-overridable half-lives |
| Migration mistakes historical reads for confirmations | This intentionally preserves the old semantics; evidence is separated only after migration |
| Added metadata increases context size | Use one compact nested `validity` object and rounded numeric fields |
| Retrieval paths drift apart again | Central pure helper plus path-consistency regression tests |
| Scope expands into association forgetting | Keep link-weight time decay as a separately designed follow-up |
