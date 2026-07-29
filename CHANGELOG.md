# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
