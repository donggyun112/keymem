# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.27.2] - 2026-09-05

### Changed

- Keys attached by write-time auto-linking (embedding proximity, not named by the agent) are no
  longer offered in `connected_keys` of the passive Top-1 preview or of `inject` results. In the
  multi-hop experiments they were the main source of wasted `read_key` calls: a memory that
  mentions "keymem" was auto-linked to five generic keymem-* keys that all led back to itself.
  `read_memory` still lists every key and now flags these with `auto: true`. The flag is stored
  per link in graph.json (`auto: true`); links written by older versions load as explicit.

## [0.27.1] - 2026-09-05

### Fixed

- Claude Code plugin failed to load with "Duplicate hooks file detected": `hooks/hooks.json` is
  loaded automatically, and the manifest's `hooks` entry pointed at the same file. The entry is
  removed; `manifest.hooks` is only for additional hook files.

## [0.27.0] - 2026-09-05

### Added

- `recall()`'s passive Top-1 memory now scores each `connected_keys` entry with `relevance`, the
  cosine of that key concept to the query/context, sorted high→low. A transcript audit of ~80
  real sessions found zero second hops: agents take the first memory as the answer and stop.
  In a controlled test the same agents walked the hop 9/9 times once the question made the first
  memory look incomplete, so the missing piece is a visible cue, not capability. The score is
  that cue: it turns "should I hop?" into the same kind of decision as the first hop. Concept
  vectors only; no extra memory content enters context and no extra embedding call is made.
- `read_key` with a non-id argument (a concept name, a mistyped id) now explains what a
  `key_id` looks like and points to `recall(query)`; agents were observed passing concept text.

### Changed

- Tool descriptions, server instructions, and the skill now frame `recall` as answering a
  question: check whether the Top-1 memory answers it, otherwise take one more hop via the
  highest-relevance connected key, one call per hop, stop when complete.

### Fixed

- Duplicate detection (supersede) and contradiction detection are scoped to the memory's
  namespace. The same sentence saved under two namespaces used to make the second write
  silently supersede the first, and near-paraphrases across namespaces were linked as
  contradictions. Unscoped only when a caller supplies no namespace at all.

## [0.26.4] - 2026-09-05

### Fixed

- When `recall()` is given a `context` utterance, the content path now uses the sentence
  content gate instead of the lower short-keyword gate. The short gate is calibrated for
  keyword↔sentence cosines, so a sentence context could pull an unrelated key through it and
  auto-inject an irrelevant Top-1 memory. Applies to both the key search and the compatibility
  `recall_memories()` path.
- `dismiss()` now affects the passive Top-1: the cross-encoder score is squashed to (0,1) and
  weighted by the key→memory link, so a dismissed pairing can lose the preview slot to a sibling.

## [0.26.3] - 2026-09-05

### Fixed

- The default `recall()` path now honours `inject_max_chars` for the passive Top-1 preview; it was
  only applied on the `inject:true` path, so long memories were returned untruncated.

## [0.26.2] - 2026-09-05

### Fixed

- A model download whose body stalls (Hugging Face CDN, `UND_ERR_BODY_TIMEOUT`) no longer
  crashes the daemon with an unhandled stream `error` event; the failure is caught and recall
  falls back to fused ranking. The next attempt resumes the leftover `.tmp` with a `Range`
  request instead of restarting from zero.

## [0.26.1] - 2026-09-05

### Fixed

- Model auto-download is now single-flight per file. With reranking on by default, several
  concurrent `recall()` calls could each start downloading `bge-reranker-v2-m3`, truncate one
  another's `.tmp` stream, and rename a corrupt half-file into place; the daemon then thrashed
  on the download indefinitely. Concurrent callers now await the one in-flight download.

## [0.26.0] - 2026-09-05

### Added

- `recall()` now completes the first Key → Memory hop in one call: it returns the ranked key
  clusters plus one passive Top-1 memory under the strongest key, carrying `matched_key`,
  `validity`, and `connected_keys`. The new optional `context` argument (the raw user utterance)
  ranks memories within that key. The preview does not touch access/depth, links, aliases, or
  freshness; `read_key` → `read_memory` remains the explicit traversal path.
- Prompt-cache A/B benchmarks (`bench:prompt-cache`, `bench:prompt-cache-llm`) with results
  recorded in `BENCHMARKS.md`.

### Changed

- `recall()` key payloads are compacted to a fixed field set so the response stays stable and
  prompt-cache friendly across calls.
- Cross-encoder reranking is now a core default for both the passive Top-1 returned by `recall()`
  and compatibility `recall_memories()` results. Set `KEYMEM_RERANK=false` to disable it; model
  load failures continue to fall back to fused ranking.

## [0.25.1] - 2026-09-03

### Fixed

- Codex plugin and manual shim launch commands now request `keymem@latest` explicitly. This
  prevents npm 11 from treating a checkout whose local package is also named `keymem` as the
  requested package and then failing startup with `keymem-shim: command not found`.

## [0.25.0] - 2026-09-03

### Added

- Confirmation-aware freshness adds `transient` (7-day), `standard` (90-day), `stable`
  (365-day), and `permanent` decay profiles. Memory views now expose an additive `validity`
  payload (`fresh`, `aging`, or `stale`), and ranking softly discounts aging facts without
  deleting them.
- `confirm_memory(memory_id, evidence)` separates evidence-backed confirmation from retrieval.
  `read_memory` can still update access metadata and, when given `via_key_id`, the selected graph
  edge, but no longer refreshes freshness or depth. `remember`, `remember_batch`, and `correct`
  accept the additive `decay_profile` field; `correct` also accepts `ttl_seconds` while preserving
  omitted policies and retaining the immediate predecessor as one-step provenance.

### Changed

- Version-1 graphs migrate to schema version 2 by deriving `last_confirmed_at` from existing
  creation/access timestamps, seeding `confirmation_count` from access history with a minimum of
  one, and assigning the `standard` profile. The repaired graph is saved through the existing
  atomic persistence path.

### Fixed

- Expired memories no longer participate in duplicate, contradiction, or supersede candidate
  selection, so obsolete TTL records cannot block or redirect a new write.

## [0.24.0] - 2026-08-30

### Added

- BENCHMARKS.md §8 records an embedding-model comparison (bge-m3 vs EmbeddingGemma-300m vs
  granite-embedding-97m-r2) and why keymem stays on bge-m3: granite loses outright (8% vs 58%
  recall@1), Gemma ties every retrieval metric and buys no extra separability, and adopting
  either means replacing fastembed's tokenizer layer — `fastembed@2.1.0` pins
  `@anush008/tokenizers@^0.0.0`, which cannot parse a modern `tokenizer.json` at all. The
  comparison ran out of tree through the `__setTestEmbedder` seam; no dependency changed.

- BENCHMARKS.md §7 records an experiment that was **built, measured, and removed**: Hebbian
  key↔key associations learned from queries that reach two keys together. The edges formed and
  recall traversed them, but three independent measurements — the §6 fixture and a train/held-out
  split over the owner's real 530-memory store — moved not one metric in either direction, and
  the only fixture that showed a benefit was the one built to show it. Recorded rather than
  shipped default-off, because a flag ships the maintenance cost and the reader's question
  without answering it.

- **`dismiss(memory_id, key_id)` — the graph's first negative signal.** Every signal it carried
  was positive: a read deepens a memory and strengthens the edge it traversed, a weak match
  accrues heat toward becoming an alias. So a wrong surface cost nothing, every mis-hit silently
  got *stronger*, and the only defence against a bad edge was refusing to build it — which is why
  the phrase-bridge gate has to be as conservative as §6 shows. dismiss weakens the one
  key→memory edge that produced the false surface (3x what a read pays back) and spends the
  pending alias confirmation against the key instead of for it. It never touches the memory:
  "this key should not have pulled this up" is a claim about the edge, not the fact. `LINK_WEIGHT_MIN`
  floors the weight, so no amount of dismissing can sever an edge and orphan a memory. Wired into
  the MCP tool list, the server instructions, and the skill protocol — a feedback tool no agent
  calls is dead code.

- `bench/phrase-bridge.ts` + `bench/phrase-fixture.json` — gate ablation for phrase-key
  bridging (NO-BRIDGE vs the shipped cosine gate vs an experimental structural one).
  Documented as §6 of BENCHMARKS.md. The NO-BRIDGE condition is reconstructed inside the
  bench from the on-disk link set rather than switched off in production — bridging is core
  behavior, and a benchmark is not a reason to ship a knob. The structural gate lost on its own terms: no reach
  gain, worse ranking on the bridge queries, a degraded `direct` control, and it pulled
  off-topic token-sharing memories into the top 5. The cosine gate stays.

## [0.23.0] - 2026-08-30

### Fixed

- **Namespaces are case-folded on write, on query, and on load.** They were compared
  with exact string equality everywhere, so `Nexora` and `nexora` hard-partitioned the
  same project and a scoped recall silently returned half its memories (live store: 7
  vs 6 memories under the two spellings). One normalizer now runs on both sides, the
  loader repairs stored values, and the daemon's `namespaces.json` allowlist folds the
  same way so a hand-written `Nexora` mapping still matches.
- MCP handshake version is read from `package.json` at runtime. The hardcoded literal
  had gone stale twice (`0.14.7`, then `0.22.0` against a `0.22.2` package); hosts key
  upgrade detection off it.

### Added

- **Legacy phrase keys are bridged onto the atomic keys they contain.** `writeHints`
  warns about new 3+-token keys, but the ones already stored stayed orphaned: on the
  live store, 177 of 204 phrase keys were singletons, so their memory hung off a key
  nothing else ever reached — no hub, no associative traversal, reachable only by
  repeating the exact phrase. On load, every 1-2 token slice of a phrase key that
  ALREADY exists as a key is linked to that phrase key's memories (`git push 403` →
  `git push`, `git`). Requiring the atomic key to exist is the filter: no key is
  created, invented, or deleted, and the phrase key survives for literal recall.
  Each bridge is gated on the atomic key's own cosine to the memory against
  `contentRecallShort` — a shared token is not a shared topic, and ungated, half the
  bridges landed on generic hubs (`사용자`, `프로젝트`) that would dilute every later
  recall on them. Live store: 143 links added across 73 memories, bench unchanged.
  Idempotent.

## [0.22.2] - 2026-08-30

### Fixed

- Plugin MCP config ships as plugin-root `.mcp.json`. Claude Code only discovers a
  plugin's MCP servers from that file — a `mcpServers` field in `plugin.json` is
  ignored, whether it holds a path or an inline object (both gave "MCP servers (0)").
  Codex picks the same file up via default discovery, so one file replaces
  `mcp/keymem.json` plus the manifest field in both manifests. `.mcp.json` is no
  longer gitignored: it is a shipped plugin component, and it also gives anyone
  opening this repo keymem as a project-scoped server.

## [0.22.1] - 2026-08-13

### Added

- keymem ships as a plugin for Claude Code and Codex. One install wires up all three
  halves instead of three manual steps (`mcp add` + hook in settings + protocol in
  CLAUDE.md): `.claude-plugin/marketplace.json`, `.claude-plugin/plugin.json` (Claude)
  and `plugin.json` (Codex) over shared components, `hooks/hooks.json`
  (UserPromptSubmit, one hook file for both hosts), and `skills/keymem/SKILL.md` —
  the recall/remember protocol that until now only lived in the user's CLAUDE.md.
  `hooks/keymem-hook.mjs` is committed as a build artifact because plugin installs
  do not run a build.

### Fixed

- Close the three routes that could write a keyless memory.
- Build syncs the plugin manifest version from `package.json`. Both hosts key plugin
  upgrade detection off the manifest version, so a manifest left behind meant
  installed users never saw the release.

## [0.22.0] - 2026-07-29

### Fixed

- **correct()/supersede key-inheritance regression**: the drift-drop bar compared
  key-to-CONTENT cosine against keyRecall (0.62), but same-topic short keys measure
  0.477-0.643 against sentence content, so a typical no-keys correction dropped every
  inherited key and left the new version an unreachable keyless orphan (recall miss,
  explain:true misreporting empty_namespace). The bar now uses the content-calibrated
  gate (contentRecallShort), and a zero-key result keeps the old keys outright — a
  correction can never orphan a memory. Existing orphans repaired.

### Added

- Hook push cwd -> namespace scoping: `<data-dir>/namespaces.json` maps path prefixes
  to allowed namespace lists ("default" always allowed). Kills cross-project injection
  pollution that relevance floors cannot separate (measured: wrong 0.63-0.68 vs right
  0.61-0.70 fully overlap). Unmapped cwd keeps global behavior.
- Hook skips prompts shorter than KEYMEM_HOOK_MIN_CHARS (default 6) — acknowledgments
  are not recall cues and should not pay the ~0.5 s round trip.

## [0.21.1] - 2026-07-29

### Fixed

- Hook client: await the stdout write callback before process.exit — exiting first
  raced the pipe flush and silently dropped the injected context.
- `/inject` applies an absolute relevance floor (`KEYMEM_HOOK_MIN_REL`, default 0.6):
  once any candidate anchors, recall keeps the whole fused set, so ~0.47-relevance
  tail memories leaked into unprompted injection (measured: real hits 0.75+).

## [0.21.0] - 2026-07-29

### Added

- Daemon `/inject` endpoint: harness hooks POST the raw user utterance and get
  passively-relevant memories back in one localhost round trip (no MCP session).
  Precision comes from the dense anchor gate + no-bm25-only provenance filter;
  the keyword-oriented lexical-coverage filter is bypassed (Korean particles
  defeat its string-includes matching on natural sentences).
- `dist/hook.js`: Claude Code UserPromptSubmit hook client — the push half of
  keymem. Injects at most KEYMEM_HOOK_TOP_K (default 2) unconfirmed memories as
  additionalContext; never blocks (KEYMEM_HOOK_TIMEOUT_MS, default 800 ms) and
  never wakes the daemon.
- recall/searchKeys reuse the query embedding when context === query (halves
  embed calls on the hook path: ~815 ms → ~450 ms warm).

## [0.20.0] - 2026-07-29

### Added

- Binary vector sidecar (`vectors.bin` + `vectors.idx.json`): embeddings no longer
  live in graph.json as JSON floats (93 MB -> ~2 MB graph + 15 MB binary on the
  reference store). Legacy inline stores load unchanged and migrate on next save.
- Per-sentence vectors for multi-fact memories with max-sim content scoring in
  recall/searchKeys/nearestKeys. Measured motivation: sub-fact queries score
  +0.07~0.19 cosine higher against the best sentence than the whole-note centroid,
  which often sits below the content gate. `KEYMEM_SENTENCE_VECTORS=0` disables.
- `bench/backfill-sentence-vectors.ts` migration/backfill script.

## [0.19.1] - 2026-07-29

### Added

- `remember` hints flag phrase keys (3+ words) as `hints.phrase_keys`: measured on
  the live store, 3+-word keys are 91% singleton (unreachable, no hubs) vs 69% for
  single words, so keys must be atomic 1-2 word concepts. Guidance hardened to match.

## [0.19.0] - 2026-07-29

### Added

- `recall` returns `{status:"no_match", nearest_keys}` (closest ungated concepts)
  instead of a bare empty array, so a miss is a retry hint rather than a dead end.
- Optional `context` on `recall`: the raw utterance drives content matching while
  the keyword query keeps driving key matching (dual-path cues; sentence-shaped
  cues measure ~0.1-0.2 cosine higher against sentence content on bge-m3).
- `contentRecallShort` threshold (bgem3 0.46, `KEYMEM_CONTENT_RECALL_SHORT`):
  calibrated content gate for short keyword queries, which embed systematically
  lower against sentence content (the 0.55 gate cut 8/12 measured related pairs).
- `remember` response `hints`: near-neighbor existing concepts the new keys nearly
  duplicate, and a single-language key warning.
- `bench/real-eval.ts` (real-workload hit-rate eval) and
  `bench/quarantine-bench-data.ts` (moves HotpotQA/wiki pollution to a `bench`
  namespace).

## [0.18.0] - 2026-07-29

### Changed

- Make the agent guidance use namespace-scoped `recall` and complete
  `read_key(query) → read_memory(via_key_id)` for core facts; passive injection is
  no longer presented as an equivalent path.
- Enable conservative short-key semantic merging by default for calibrated
  `bge-m3`, while protecting sibling labels such as `Agent A` / `Agent B`.
- Add namespace-scoped `browse_keys`, optional explained recall statuses, and
  explicit score-kind metadata so empty results and incompatible score scales are
  distinguishable.

## [0.17.1] - 2026-07-26

### Fixed

- Preserve MCP sessions across transient standalone-SSE disconnects by giving the
  SDK reconnect loop a grace window before reaping abandoned sessions.
- Catch shim forwarding and close promise failures so a rejected HTTP send cannot
  terminate the stdio proxy through an unhandled rejection.

### Changed

- Strengthen the `remember`/`remember_batch` end-of-turn durability gate and add
  explicit persistent `CLAUDE.md` setup guidance for reliable Claude Code saving.
- Synchronize the MCP handshake version with the package release version.

## [0.17.0] - 2026-07-22

Run keymem as a single shared HTTP daemon behind a thin stdio shim, instead of one
full server process per host session. Previously every host session (Claude Code,
Codex, each terminal) spawned its own stdio server, so N sessions meant N processes
each loading the embedding model (~1.5GB apiece) and N writers racing on the same
`graph.json` under an in-process mutex that gave no cross-process protection. Now one
resident daemon holds the graph and model once; the shims are lightweight proxies.

### Added

- **`keymem-shim`** — a stdio<->HTTP bridge the host spawns in place of the server. It
  transparently proxies MCP messages to the shared daemon, attaching the host's
  session identity (`X-Keymem-Host-Agent` / `X-Keymem-Host-Session`) on every request
  so provenance stamping stays correct across many clients on one daemon. If the
  daemon can't be reached or started, the shim falls back to running the server
  in-process — identical to the previous behavior.
- **`keymem-daemon`** — a resident StreamableHTTP MCP server (loopback only, default
  port 8765 / `KEYMEM_DAEMON_PORT`). It loads the graph and embedding model once,
  serves every shim session from that shared state, and self-exits after an idle
  period (default 10 min / `KEYMEM_DAEMON_IDLE_MS`). Sessions whose connection drops
  abruptly are reaped so idle-exit still fires.

### Changed

- **Host-session provenance now travels over request headers, not ambient detection.**
  On the daemon path, `host_agent` / `host_session` / `host_turn` are derived from the
  per-request `X-Keymem-Host-*` headers rather than the previous env/mtime heuristic,
  so a memory saved through one session can never be stamped with another session's
  identity. The env-based path is preserved for the in-process fallback.
- **Single writer.** With one daemon owning `graph.json`, the existing `async-mutex`
  serialization is now actually effective (previously each of N processes had its own
  mutex over a shared file).

### Notes

- Backward compatible: the `keymem` bin still runs the in-process stdio server, so
  existing setups are unchanged. Opt into the shared daemon by pointing your MCP
  client's command at `keymem-shim` (i.e. `dist/shim.js`).

## [0.16.0] - 2026-07-08

Make keymem discoverable under on-demand (lazy) MCP tool loading, where a tool's own
description isn't seen until the tool is already loaded — so "recall first" guidance
and plain-language tool names now reach the model through the always-injected channels.

### Added

- **Server-level `instructions`** — the `initialize` response now carries a compact
  usage playbook (recall before the first reply and on topic shifts, navigate
  Key → Memory → Key, save durable facts with `remember`, `correct` on change, stay
  silent). Hosts inject this as always-on guidance, so the "recall first" behavior no
  longer depends on manually including the `memory_system_prompt` prompt or on the
  `recall` tool already being loaded. Full playbook still available via
  `memory_system_prompt`.

### Changed

- **Tool descriptions reworded for on-demand discovery** — first sentences now lead
  with plain, searchable vocabulary instead of internal jargon so tool-search ranking
  matches the terms a model actually queries:
  - `recall` — imperative "CALL THIS FIRST…" → "Search long-term memory for what is
    already known about the user, project, or topic…".
  - `read_key` — "Inspect one key cluster" → "List the memories stored under one key".
  - `read_memory` — "Read one full memory…" → "Read the full content of one stored
    memory…".
  - `related` — "Compatibility exploration from a known memory ID" → "Find other
    memories associated with a memory you already have".

## [0.15.0] - 2026-06-26

Trace a recalled memory back to the conversation it came from. `get_conversation`
used to read a log keymem never wrote (so it returned nothing); it now reads the
host coding agent's own transcripts and links saved memories to them.

### Added

- **`list_sessions(agent?, limit?)`** — discover recent conversation sessions
  recorded by host coding agents (Claude Code, Codex) on this machine, newest
  first, with `agent`, `session_id`, `cwd`, `modified`, and a first-message
  `preview`.
- **`get_conversation` reads native transcripts** — loads the host agent's own
  on-disk transcript (Claude Code `~/.claude/projects`, Codex `~/.codex/sessions`),
  normalized to `{turn, role, content, ts}` with reasoning/tool-call noise
  stripped. New `agent` parameter; `turn` fetches a focused ±2-turn window.
- **Memory → conversation link** — `remember` / `correct` / `remember_batch`
  stamp the active host session onto a memory's `source`
  (`host_session` / `host_agent` / `host_turn`). The session is identified
  deterministically from the env the host injects into the MCP server
  (`CLAUDE_CODE_SESSION_ID`, `CODEX_THREAD_ID`), with a most-recently-modified
  fallback for other hosts.
- **`trace` hint on `read_memory`** — when a memory carries a host link, the
  response includes a ready-to-run `get_conversation` call so the agent can drill
  to the verbatim exchange without remapping fields.

### Security

- Transcript tools (`get_conversation`, `list_sessions`) are gated to trusted
  local contexts. They are hidden from `tools/list`, refuse calls, and skip
  host-link stamping unless a recognized host injected its session env or the
  operator opts in with `KEYMEM_TRANSCRIPT_ACCESS=true` (`=false` force-disables).
  This keeps local transcripts from leaking when keymem is reached over a plain
  server or a non-owner agent. `session_id` is restricted to a UUID and resolved
  within whitelisted roots (with symlink checks) to prevent path traversal.

### Fixed

- Sync the MCP handshake version (`server.ts`) to `package.json`; it previously
  reported a stale `0.14.7`.
- Approve `onnxruntime-node` / `esbuild` build scripts for pnpm 10+.

### Chore

- Add `glama.json` (maintainers) for the Glama directory.
