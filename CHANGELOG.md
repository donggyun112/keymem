# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
