# Default Latent Key Relations Design

## Decision

Key-to-key relations are a normal part of recall. There is no feature flag, shadow mode, or
opt-in path. A relation is learned only from a successful `read_memory(memory_id, via_key_id)`
traversal, remains latent until repeated evidence promotes it, participates in one-hop key
recall while active, and becomes invisible again after disuse.

The change ships only if the confirmed-relation association benchmark improves while direct
recall and the existing test suite do not regress. Otherwise the complete feature is removed;
it is not retained as dormant optional code.

## Evidence

The bge-m3 association fixture showed that learning all co-recalled keys polluted the graph:
top-5 all-pairs edge precision was 51.5%. Learning only the keys on a selected
`read_memory(via_key_id)` path produced 100% edge precision in the fixture, raised associative
Hit@5 from 33.3% to 50%, and kept direct Hit@5 at 100%.

A replay of 40 local keymem-related Codex transcripts, excluding the experiment session,
contained 115 recalls, 82 confirmed reads, and 373 derived explicit key relations. Of those,
347 occurred once. A two-evidence gate hid every one-shot relation and retained all 10 repeat
uses that could be served before their next confirmation. Adding opportunity decay plus a
three-day wall-clock half-life retained the same 10 uses while leaving no stale active relation
at the end of the replay. Thirty-five neighboring configurations retained all 10 uses; two also
finished with no active stale relation.

## Persistent Model

`GraphData` moves to schema version 3 and gains an optional `key_relations` array. A legacy graph
with no array loads as an empty relation graph.

Each namespace-scoped, undirected relation stores:

- canonical ordered endpoints `key_a` and `key_b`;
- `namespace`;
- `weight`, capped at 8;
- the persisted hysteresis state `active`;
- `evidence_count`;
- `last_evidence_id` for same-turn burst suppression;
- `updated_at`, the timestamp at which the stored weight was last materialized.

Malformed records, cross-namespace endpoints, dangling keys, and self-relations are dropped on
load and the graph is marked dirty for repair. Key merges rewrite relation endpoints. Colliding
relations merge conservatively with the greater decayed weight and evidence count rather than
summing them, so migration cannot manufacture a promotion.

## Learning and Forgetting

A successful read with `via_key_id` forms a star from that key to every other explicit key on the
selected memory. Auto-linked keys never create evidence. A read without `via_key_id` changes no
key relation.

The server passes an evidence ID derived from host agent, host session, and host turn. When host
identity is unavailable, it uses a five-minute time bucket. Repeating the same edge with the same
evidence ID neither increments evidence nor adds weight.

For every distinct confirmed event:

- selected relations first materialize wall-clock decay, then gain `+1` weight;
- other existing relations incident to `via_key_id` materialize wall-clock decay and one missed
  opportunity, multiplying weight by `0.5^(1/8)`;
- inactive relations promote at weight `>= 1.5` after at least two distinct evidence events;
- active relations demote below weight `1.0`;
- all relations apply lazy wall-clock decay with a three-day half-life.

Passive recall never reinforces or persists a relation. It computes effective decayed weight for
eligibility, so an expired active relation disappears from recall immediately without turning a
read operation into a graph write.

## Recall Integration

`searchKeys` first computes the existing direct candidates unchanged. Only those direct results
seed relation expansion. Each seed contributes at most three active neighbors in the requested
namespace, and expansion stops after one relation hop.

A related candidate receives:

- `match_type: "relation"`;
- the source key ID in `related_from`;
- its effective relation strength;
- score `source_score * 0.85 * min(1, effective_weight / 2.5)`.

This formula keeps every related candidate below its source, preserves literal entity priority,
and lets repeatedly confirmed relations compete with weak semantic/content matches. If a key is
already a direct candidate, the direct candidate wins and no duplicate is returned. Relation
candidates must still have active memories in the requested namespace.

A relation-surfaced candidate is not evidence. The relation changes only if the agent later makes
a successful explicit `read_memory` traversal. Thus graph output cannot reinforce itself.

## Components

A focused `KeyRelationGraph` module owns validation, persistence views, lazy decay, observation,
key rewrites, and neighbor ranking. `MemoryGraph` owns orchestration only:

- load/save the relation records with the existing atomic graph snapshot;
- call relation observation from `readMemory` under the existing mutation lock;
- merge one-hop related candidates into `searchKeys` under the existing read lock.

The MCP surface gains no new tool and no mode flag. Existing callers remain source-compatible;
`readMemory` receives one optional internal evidence-ID argument used by the server.

## Failure and Compatibility Rules

- Relation load repair never prevents memories and key-memory links from loading.
- Namespace filtering occurs before relation lookup and prevents cross-project leakage.
- Deferred `readMemory` persistence remains unchanged: relation mutations call `markDirty()` and
  are written by the existing serialized `flush()`/content-write path.
- Relation expansion performs no embedding call and is bounded by direct Top-K and three
  neighbors per seed.
- A relation record never keeps an otherwise dangling key alive.

## Verification Gate

Tests cover schema-v2 migration, schema-v3 round-trip, malformed/dangling repair, namespace
isolation, explicit-versus-auto key evidence, same-turn burst capping, two-event promotion,
hysteresis demotion, wall-clock decay, missed-opportunity decay, key-merge rewrites, deferred
flush, concurrent save integrity, one-hop-only expansion, scoring bounds, and graph-feedback
non-reinforcement.

The release gate is:

1. full test suite passes;
2. TypeScript build passes;
3. confirmed-relation fixture reaches associative Hit@5 at least 50%;
4. direct Hit@5 remains 100%;
5. not-found accuracy does not fall below the existing baseline;
6. real-trace replay still hides every one-shot edge and preserves all 10 serveable repeat uses.

Failure of any gate removes the feature rather than hiding it behind configuration.
