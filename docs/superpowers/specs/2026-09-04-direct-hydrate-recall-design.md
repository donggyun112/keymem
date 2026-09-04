# Default Direct-Hydrate Recall Design

## Decision

Normal `recall` is the core one-call retrieval path. It returns ranked keys and exactly one
passive memory: the highest-ranked memory under the highest-ranked key. No feature flag or opt-in
is involved.

The memory includes:

- `matched_key`: the incoming key edge used for selection;
- `connected_keys`: all key concepts and IDs attached to the memory, enabling another hop;
- `validity` and passive evidence metadata.

Passive hydration never changes access count, depth, link weights, aliases, or confirmation.
Another memory is never auto-injected. The caller may follow a `connected_keys[].key_id` through
`read_key`, then use `read_memory` for full inspection and explicit path reinforcement.

## Evidence

Across 30 real prompts, answer quality averaged 9.533/12 with no memory, 10.767 with the existing
manual traversal, 11.533 with direct Top-1, and 11.700 with direct Top-2. Top-1 beat no-memory by
2.0 (95% bootstrap CI +0.933..+3.067) and manual traversal by 0.767
(CI +0.233..+1.433). Top-2's additional +0.167 was not conclusive.

Therefore Top-1 is the default core policy; Top-2 and automatic multi-hop injection are not.
