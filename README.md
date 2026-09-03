# keymem

[![npm version](https://img.shields.io/npm/v/keymem)](https://www.npmjs.com/package/keymem)
[![Node.js](https://img.shields.io/badge/Node.js-20%2B-green)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**The associative memory layer for LLM agents — recall by association, not just similarity.**

![keymem demo — asking about the party cake surfaces Mina's peanut allergy via a shared key](docs/keymem-demo.gif)

Most agent memory is a vector store. It surfaces what *sounds like* your query — and misses everything your query is *connected to*.

`keymem` stores memories in a **key graph** instead. A search for **"Newton"** can still reach **"strawberries"** — Newton → apple → fruit → strawberry. The path lives in the graph, not in embedding space. It runs locally as an **MCP server**, so any MCP-compatible agent gets human-like associative recall with no external database.

**Works with:** Claude Desktop · Claude Code · any MCP-compatible LLM agent

---

## Why associative memory?

Vector-store memory retrieves by embedding similarity. That works until the thing you need *isn't similar to the words you typed*:

```
Query: "Newton"
Similarity search finds: "Newton discovered gravity"  ✅
Similarity search misses: "user likes strawberries"   ❌
```

A person makes the leap anyway — Newton reminds them of the apple, apples are fruit, they like strawberries. `keymem` makes that same leap because the **path exists in the key graph**: `Newton → apple memory → fruit key → strawberry memory`. No embedding distance connects "Newton" and "strawberry"; a chain of shared keys does.

This is the core idea: memories are not islands ranked by distance. They are nodes in an **N:M key/value graph** that an agent can walk.

---

## How it works

```
Key Space (concepts)         Value Space (memories)
[apple]   ────────┬─────────→      ↑ same memory
[gravity] ────────┘
                  │
[apple]   ────────┼─────────→ "apples are red fruit"
[fruit]   ──────┬─┘
[red]     ──────┤
                │
[fruit]   ──────┼─────────→ "user likes strawberries"
[strawberry]────┘
```

Memories live in a **Value Space**, reached through a separate **Key Space** — one memory reachable via many keys, one key leading to many memories.

`recall("Newton")` returns matching key clusters such as `[Newton]` and `[apple]`, **not memory content**. The agent then navigates explicitly: `read_key(apple)` → select the Newton memory → `read_memory(...)` → discover its `[fruit]` key → `read_key(fruit)` → select the strawberry memory.

The default MCP flow is therefore **Key → Memory → Key**. Full memory content enters the model context only when the agent deliberately calls `read_memory()` — so broad concepts never flood the context window.

---

## Quick Start

> **keymem is an MCP server (a CLI), not a library.** Run it with `npx -y keymem` (recommended —
> always the latest) or install the command globally with `npm i -g keymem`. **Do not** add it to
> your app with `npm i keymem` as a dependency: it bundles `openai`, `zod`, and the MCP SDK, so
> inside an existing project it just duplicates those trees (and can clash with your app's `zod`/
> `openai` versions). The `npm i keymem` line npm shows on the package page is for libraries — it
> doesn't apply here.

```bash
# Optional global install (npx needs none). This puts a `keymem` command on PATH that
# MCP clients can spawn. Run bare, it starts a stdio MCP server and waits for a client —
# so point your MCP config at `keymem` (or just use `npx -y keymem` as shown below).
npm i -g keymem
```

### Claude Desktop

Add to `claude_desktop_config.json`:

**OpenAI embeddings:**
```json
{
  "mcpServers": {
    "keymem": {
      "command": "npx",
      "args": ["-y", "keymem"],
      "env": {
        "OPENAI_API_KEY": "your-openai-api-key"
      }
    }
  }
}
```

**Local embeddings (no API key required) — bge-m3 recommended:**
```json
{
  "mcpServers": {
    "keymem": {
      "command": "npx",
      "args": ["-y", "keymem"],
      "env": {
        "EMBEDDING_BACKEND": "local",
        "LOCAL_EMBEDDING_MODEL": "bge-m3"
      }
    }
  }
}
```

> `bge-m3` (multilingual, recommended) **auto-downloads ~570MB on first run**, then caches. Omit `LOCAL_EMBEDDING_MODEL` for the lighter default (`fast-multilingual-e5-large`). Add `"KEYMEM_RERANK": "true"` to enable cross-encoder reranking (downloads a second model on first use).

### Plugin (recommended — Claude Code & Codex)

The repo is also a plugin marketplace, so one install wires up everything: the MCP server
(daemon-backed shim), the `UserPromptSubmit` hook that passively surfaces related memories on every
prompt, and the `keymem` skill carrying the recall/remember protocol.

**Claude Code:**
```
/plugin marketplace add donggyun112/keymem
/plugin install keymem@keymem
```

**Codex CLI:**
```bash
codex plugin marketplace add donggyun112/keymem
codex plugin add keymem@keymem
```

Codex prompts once to trust the hook; approve it or the push path stays silent. The plugin defaults
to local `bge-m3` embeddings (auto-downloads ~570MB on first run, no API key). For OpenAI
embeddings, use the manual setup below instead — plugin MCP servers only see the env they declare.

### Claude Code (manual)

```bash
# OpenAI embeddings
claude mcp add keymem -e OPENAI_API_KEY=your-key -- npx -y keymem

# Local embeddings (no API key required) — bge-m3 recommended (auto-downloads ~570MB on first run)
claude mcp add keymem -e EMBEDDING_BACKEND=local -e LOCAL_EMBEDDING_MODEL=bge-m3 -- npx -y keymem
```

### Codex CLI (manual)

```bash
# OpenAI embeddings
codex mcp add keymem --env OPENAI_API_KEY=your-key -- npx -y -p keymem@latest keymem-shim

# Local embeddings (no API key required)
codex mcp add keymem --env EMBEDDING_BACKEND=local --env LOCAL_EMBEDDING_MODEL=bge-m3 -- npx -y -p keymem@latest keymem-shim
```

Use the `keymem-shim` entry point (not bare `keymem`): it runs the shared daemon the push-path hook
talks to. For the hook, add to `~/.codex/config.toml`:

```toml
[[hooks.UserPromptSubmit]]
[[hooks.UserPromptSubmit.hooks]]
type = "command"
command = "node /absolute/path/to/keymem/hooks/keymem-hook.mjs"
timeout = 5
```

Codex only forwards the env vars declared in its MCP entry, so pass every `KEYMEM_*` /
`SUPER_MEMORY_*` override with `--env`.

That's it — recall and remember work immediately. The agent calls `recall` before its first reply, navigates with `read_key`/`read_memory`, and saves with `remember`.

For reliable proactive saving in Claude Code, add the following to `~/.claude/CLAUDE.md` (MCP prompts are not automatically applied as persistent Claude Code instructions):

```markdown
## keymem

- Before the first reply and whenever the topic changes, call `recall` silently with short noun-keyword queries.
- Before ending every reply, check whether this turn revealed a durable fact: a name, preference, decision, correction, project fact, or goal.
- If it did, call `remember` or `remember_batch` silently in the same turn with 3-6 diverse keys. A durable fact left unsaved is a bug.
- Use `correct` when existing information changes. Save nothing only when the turn revealed nothing durable.
- Treat `read_memory` as retrieval, not confirmation. Use its `validity.status`: qualify `aging`, and never assert `stale` as current without checking an external source or asking the user.
- Call `confirm_memory` only after an explicit current user assertion, an authoritative current source, or direct observation — never merely because a read succeeded.
- Never mention memory lookup or saving to the user.
```

In Codex, put the same block in `~/.codex/AGENTS.md`. (The plugin install ships this as the `keymem`
skill instead, so you can skip it there.)

For other MCP clients, include the `memory_system_prompt` MCP prompt in the agent's persistent system instructions.

### Manual / Development

```bash
git clone https://github.com/donggyun112/keymem
cd keymem
pnpm install
```

Create `.env`:
```
OPENAI_API_KEY=your-openai-api-key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Or use local embeddings (no API key required):
```
EMBEDDING_BACKEND=local
LOCAL_EMBEDDING_MODEL=fast-multilingual-e5-large  # default; best fit for Korean/multilingual keys
```

```bash
pnpm dev
# or:
pnpm build
pnpm start
```

**Requirements:**
- Node.js 20+
- pnpm for local development
- OpenAI API key for OpenAI embeddings, or `fastembed` for local embeddings

---

## Features

- **N:M key/value graph** — memories and the concepts that index them are separate spaces, linked many-to-many. One memory is reachable through many keys; one key leads to many memories.
- **Agent-driven Key → Memory → Key navigation** — the agent walks the graph deliberately instead of collapsing it into one opaque similarity search.
- **Associative multi-hop recall** — reach memories no embedding distance would connect, by following chains of shared keys.
- **Confirmation-aware freshness** — reads learn useful paths without certifying old content; explicit evidence refreshes facts across four decay profiles.
- **One-step versioning** — a correction retains its immediate predecessor and records what it superseded.
- **Key types** — `concept` keys match by similarity; `name`/`proper_noun` keys match exactly, so "동건" never matches "뉴턴" just for being short.
- **Cross-lingual key merging (IDF)** — `파이썬` and `Python` collapse into one canonical cluster instead of fragmenting the key space.
- **Hebbian link learning** — the path an agent actually traverses gets reinforced ("fire together, wire together"), so useful associations become easier to reach.
- **Hybrid retrieval** (optional direct mode) — BM25 + dense + Reciprocal Rank Fusion, with depth/confirmation-freshness modulation and configurable multi-hop expansion.
- **Cross-encoder reranking** (opt-in) — `bge-reranker-v2-m3` re-scores candidates in direct mode.
- **Local-first** — all data in a local JSON graph; no external database. OpenAI or fully-local embeddings (auto-downloaded).

### Freshness and Depth

Every memory has a depth score `0.0 → 1.0`, but retrieval and confirmation are separate:

| Stage | Depth | Behavior |
| --- | --- | --- |
| Shallow | `< 0.3` | Little explicit confirmation; minimal ranking boost. |
| Medium | `0.3–0.7` | Repeatedly confirmed; moderate ranking boost. |
| Deep | `> 0.7` | Strongly confirmed; maximum ranking boost, but still correctable and age-sensitive. |

`read_memory()` increments access metadata and may reinforce the traversed key edge, but it does
**not** change depth, `last_confirmed_at`, `confirmation_count`, or freshness. Only
`confirm_memory(memory_id, evidence)` refreshes validity and increases depth `+0.05`; accepted
evidence is an explicit current user assertion, an authoritative current source, or direct
observation. A successful read alone is never confirmation.

Freshness decays from `last_confirmed_at` according to the memory's `decay_profile`:

| Profile | Default half-life | Intended use |
| --- | --- | --- |
| `transient` | 7 days | Fast-changing state such as temporary plans or availability |
| `standard` | 90 days | General facts; the default |
| `stable` | 365 days | Slowly changing facts |
| `permanent` | No decay | Deliberately timeless or immutable facts |

Every memory view includes a `validity` payload with `freshness`, `status` (`fresh`, `aging`, or
`stale`), `age_days`, `last_confirmed_at`, `confirmation_count`, `decay_profile`,
`verification_recommended`, and `verification_required`. `fresh` means freshness is at least
`0.5`; `aging` is at least `0.125` but below `0.5`; `stale` is below `0.125` and must not be
asserted as current without verification. Decay is soft: it lowers ranking through
`0.2 + 0.8 × freshness` and never deletes a memory. TTL remains the only automatic expiry mechanism.

For example, save a temporary plan with `remember(..., decay_profile:"transient")`; after the user
explicitly says it is still current, call `confirm_memory(memory_id, evidence:"user")`. Do not
confirm merely because `read_memory` returned it.

### Key Types

Not all keys should behave the same. Names shouldn't match semantically — "동건" shouldn't match "뉴턴" just because they're both short Korean words.

| Type | Matching | Use Case |
| --- | --- | --- |
| `concept` (default) | Embedding similarity ≥ threshold (0.28 OpenAI / 0.60 local) | Topics, categories, attributes |
| `name` | Exact match only | Person names |
| `proper_noun` | Exact match only | Brands, places |

Name/proper_noun keys also get an IDF penalty (`×0.5`) when they become hub keys connected to many memories, preventing them from polluting unrelated searches.

### Versioning (one-step predecessor)

```
"user lives in Seoul"   (depth: 0.4 → weakened to 0.12, preserved)
        ↑ superseded by
"user moved to Busan"   (depth: 0.0, new)
```

`keymem` retains the immediate predecessor instead of overwriting it. The superseded record is
excluded from active retrieval regardless of its depth, and the new version becomes current. A
later correction prunes the grandparent, so provenance is one step rather than a full-history
archive.

### Key Merging

```
Add key "파이썬"  → finds existing "Python" (similarity 0.87 > threshold 0.85)
                 → reuses existing key instead of creating duplicate
```

Prevents key space fragmentation. The same concept across languages or phrasing stays unified.

### Agent-driven Retrieval (default)

The default MCP API keeps Key Space and Value Space separate:

1. `recall(query)` searches canonical keys and aliases. It returns key IDs, concept labels, match scores, linked-memory counts, hub status, and specificity — never memory content.
2. `read_key(key_id)` returns ranked memory IDs and metadata, never content. Hub keys are paginated with `limit`/`offset` so broad concepts cannot flood context.
3. `read_memory(memory_id, via_key_id)` returns the full memory, its `validity`, and every connected key cluster. It updates access metadata and reinforces only the traversed `via_key_id` edge; it does not confirm or deepen the memory.
4. The agent follows any returned key with another `read_key()` call, producing an explicit **Key → Memory → Key** graph walk.

Semantically merged keys are preserved as aliases on one canonical key cluster (for example `Python` + `파이썬`). The recommended `bge-m3` profile enables conservative short-key merging by default; override or disable it with `KEYMEM_SHORT_KEY_MERGE`. A key linked to at least three active memories is surfaced as a hub with `is_hub`, `memory_count`, and `specificity` metadata rather than being hidden by IDF. Override the hub threshold with `KEYMEM_KEY_HUB_MIN_LINKS`.

### Direct Hybrid Retrieval (optional compatibility mode)

Set `KEYMEM_DIRECT_RECALL=true` to expose `recall_memories()`, a one-call memory retrieval path. Three signals run in parallel and are fused with **Reciprocal Rank Fusion** (`RRF_K = 60`):

- **BM25 (sparse):** lexical full-text search over memory content (MiniSearch, fuzzy + prefix). Catches exact terms, names, and rare tokens that embeddings blur.
- **Dense Path A (key matching):** query embedding → match keys → follow links → memories. Score = `keySim × IDF × linkWeight`, summed across all matching keys.
- **Dense Path B (content matching):** query embedding → directly compare against memory content embeddings. Finds memories even when they weren't tagged with the right keys.

Sparse and dense rank lists are merged by RRF, then modulated by depth and confirmation freshness before configurable multi-hop expansion (`hops=1–5`, default `2`). This compatibility tool is hidden by default so agents use explicit key navigation instead of collapsing the graph into one search call.

### Hebbian Link Learning

Reading a full memory is a **write**, not just a read. In the default flow, `recall()` and `read_key()` are read-only; `read_memory(memory_id, via_key_id)` reshapes the selected path:

- The traversed `via_key_id → memory_id` link is **reinforced** (`+0.1`, capped at `3.0`).
- A full read increments access metadata but changes neither depth nor `last_confirmed_at`, so it
  never refreshes freshness. Freshness continues to decay as elapsed time grows. Evidence-backed
  `confirm_memory()` increases depth and refreshes freshness by advancing `last_confirmed_at`.

Reinforcement is scoped to the key the agent actually traversed — not every key attached to the memory. This is the literal Hebbian rule ("fire together, wire together") and prevents unrelated associations from growing when the memory is reached through a different concept. Weights are clamped to `[0.1, 3.0]`.

Link weights feed back into `read_key()` ranking, so repeatedly selected paths become easier to reach. Optional `recall_memories()` retains the previous matched-link reinforcement and explored-link decay behavior.

---

## Architecture

```
│                      Key Space                          │
│   [name] [동건] [programming] [python] [fruit] [red]   │
│      ↓      ↓         ↓           ↓       ↓      ↓     │
│   [vec]  [exact]    [vec]       [vec]   [vec]  [vec]   │
                         │ N:M links
                         ↓
│                     Value Space                         │
│   "user's name is Donggeon"     depth: 0.85  (deep)    │
│   "user likes Python"           depth: 0.30  (medium)  │
│   "user likes strawberries"     depth: 0.05  (shallow) │
```

**Default MCP navigation:**

1. Embed the query and match canonical key concepts plus exact aliases.
2. Return key clusters with `memory_count`, `is_hub`, and `specificity`; do not return memory content.
3. Rank a selected key's memory handles by link weight, depth, and confirmation freshness in `read_key()`.
4. Return full content and adjacent key clusters from `read_memory()`; reinforce the traversed edge only when `via_key_id` is supplied.
5. Repeat `read_key(next_key_id)` to walk the graph deliberately.

**Optional `recall_memories()` algorithm (hybrid, configurable 1–5 hops; default 2):**

Three retrieval signals run in parallel, then get fused and expanded:

1. **BM25 (sparse):** lexical search over memory content (MiniSearch, fuzzy `0.2` + prefix). Keep top 50.
2. **Dense Path A (keys):** embed query → match keys (concept: cosine ≥ threshold; name/proper_noun: substring match → score `1.0`) → take top 10 keys → follow links. Score = `keySim × IDF × linkWeight`, summed across matching keys.
3. **Dense Path B (content):** compare query embedding directly against memory content embeddings (cosine ≥ threshold).
4. **RRF fusion:** merge the BM25 and dense rank lists via `score += 1 / (RRF_K + rank + 1)` (`RRF_K = 60`).
5. **Depth & freshness modulation:** `score × (0.9 + depth × 0.1) × (0.2 + 0.8 × freshness)`, using the selected 7/90/365-day half-life or no decay for `permanent`.
6. **Associative expansion (`hops`, default 2):** breadth-first from the directly-matched set — each round follows shared keys (`× HOP_DECAY(0.3) × IDF × linkWeight`) and explicit `related_to` links (bidirectional, `× HOP_DECAY`) to the next frontier. `hops=N` walks up to N steps, so a memory's `hop` is its shortest chain distance. Score decays by `HOP_DECAY` per hop.
7. **Hebbian update:** reinforce matched-key links of returned memories (`+0.1`), decay explored-but-unreturned links (`−0.005`).
8. Return ranked results with `hop` field (`1` = direct, `2+` = associative distance).

### Similarity thresholds (calibrated per embedding model)

Embedding backends have very different cosine distributions, so a single threshold set cannot serve all of them. The thresholds below are calibrated per model (`getThresholdProfile()` in `src/embedding.ts`):

| Threshold | OpenAI | Local BGE (en) | Local e5 (multilingual) |
| --- | --- | --- | --- |
| Key recall (query↔key cosine) | 0.28 | 0.60 | 0.85 |
| Content recall (query↔content cosine) | 0.28 | 0.50 | 0.80 |
| Key auto-link | 0.50 | 0.60 | 0.93 |
| Key merge | 0.85 | 0.85 | 0.97 |
| Memory dedup | 0.90 | 0.90 | 0.985 |

**Why e5 differs so much:** multilingual-e5 packs embeddings into a narrow high-cosine band (~0.86–0.99). Same-word query↔key pairs (asymmetric `query:`/`passage:` prefixes) still separate cleanly (~0.89 vs ≤0.82), but key↔key and content↔content do **not** — distinct facts like *"A uses Postgres"* and *"B uses Mongo"* sit at ~0.96, dangerously close to true paraphrases (~0.99). Hence e5's merge/dedup/auto-link thresholds are pushed high to avoid silently collapsing distinct memories.

**Drift escape hatch:** if you switch models or your data's character drifts, override any threshold without code changes:

```
KEYMEM_KEY_RECALL=0.82
KEYMEM_MEMORY_DEDUP=0.99
# also: _KEY_MERGE, _KEY_AUTOLINK, _CONTENT_RECALL  (values in [0,1])
```

**Score gate, distribution gate, and contradiction band** can also be tuned per deployment:

| Env var | Default (profile) | Description |
| --- | --- | --- |
| `KEYMEM_MIN_SCORE` | per-model (e.g. `0.55` for bge-m3) | Absolute cosine floor for optional `recall_memories()`. Set to `0` to disable. |
| `KEYMEM_GATE_Z` | `0` by default | Opt-in distribution gate for `recall_memories()` (robust-z, median/MAD). Values around 2–5 are typical; `0` disables it. |
| `KEYMEM_CONTRADICTION` | per-model (e.g. `0.80` for bge-m3) | Contradiction-band lower bound. Memory pairs whose cosine similarity falls in `[contradiction, memoryDedup)` are flagged as contradictions. `read_memory()`, `related()`, and optional `recall_memories()` expose conflicting IDs. |
| `KEYMEM_AUTOKEY` | `true` | Auto-key self-healing: learn missing search terms from real usage. Set `false` to disable. |
| `KEYMEM_AUTOKEY_PROMOTE_N` | `3` | Routing-confirmed selections of a `(key, query)` pair before the query is folded into the key space. |
| `KEYMEM_AUTOKEY_CONFIRM_FLOOR` | `0.45` | Lowest query↔key cosine eligible for routing-confirmation learning. Repeated selections through the same key can teach a below-gate query alias; this confirms routing only, never content freshness. Lower (e.g. `0.40`) to catch more borderline paraphrases; set `≥` the recall threshold to disable. |
| `KEYMEM_AUTOKEY_MAX_ALIASES` | `8` | Max learned aliases promoted per key. |
| `KEYMEM_AUTOKEY_PRUNE_AGE` | `2592000` | Seconds before a never-hit learned alias is pruned by `cleanup_expired` (30 days). |
| `KEYMEM_DECAY_TRANSIENT_DAYS` | `7` | Half-life in days for `transient` memories. Must be finite and greater than zero. |
| `KEYMEM_DECAY_STANDARD_DAYS` | `90` | Half-life in days for the default `standard` profile. Must be finite and greater than zero. |
| `KEYMEM_DECAY_STABLE_DAYS` | `365` | Half-life in days for `stable` memories. Must be finite and greater than zero. |

**Why e5 gates are opt-in:** multilingual-e5's narrow cosine band (~0.86–0.99) makes a static floor unreliable, while held-out tests showed distribution and key-proximity gates can also overfit. Both are disabled by default to avoid hiding real memories. Use bge-m3 for reliable not-found behavior, or calibrate e5 gates on your own corpus.

Distribution gate parameters:
- **`gateZ`** — set via `KEYMEM_GATE_Z` or the `min_z` parameter of optional `recall_memories()`.
- **`0` disables** the gate (default for bge-m3, bge, openai, minilm — where `min_score` already works).
- Both gates **compose (AND)**: a result must clear both `min_score` and `gateZ` to be returned.
- A **literal name/proper-noun key match** (e.g. querying a stored `name`-typed key exactly) is always a definite anchor and bypasses the distribution gate.
- **`GATE_MIN_POPULATION = 8`**: the gate is skipped when fewer than 8 memories exist (too few samples for a reliable distribution), so early-session recall is unaffected. The gate's background population is **namespace-filtered** and excludes superseded/expired memories, so recall scoped to a sparse namespace may fall below this threshold and skip the gate entirely.
- **Known e5 limitation:** the optional gate keys off `maxContentSim` (content cosine only). A relevant fuzzy-key hit with a flat content distribution may be rejected; literal key matches bypass the gate.

An uncalibrated `LOCAL_EMBEDDING_MODEL` falls back to the BGE profile **and logs a warning** so the miscalibration is never silent.

> **Multilingual note:** cross-lingual *content* matching has a same-language bias (a Korean query scores Korean memories higher regardless of meaning). The reliable cross-lingual path is the **key graph** — tag memories with keys in multiple languages (e.g. `["딸기", "strawberry"]`) so recall hits the key exactly instead of relying on biased content similarity.

---

## MCP Tools

The full trusted-local tool set contains 16 tools by default. Plain untrusted
servers hide the two transcript tools (`list_sessions`, `get_conversation`):

| Tool | Description |
| --- | --- |
| `recall(query, top_k?, namespace?, explain?)` | Search Key Space only. Returns canonical keys, aliases, scores, and hub metadata; never memory content. `explain:true` distinguishes `found`, `no_match`, and `empty_namespace`. |
| `browse_keys(namespace, hubs_only?, limit?, offset?)` | Browse a namespace's active key vocabulary, hubs first, when recall has no entry hit. |
| `read_key(key_id, query?, namespace?, limit?, offset?)` | List ranked memory IDs and metadata connected to one key. Pass the original query for relevance ordering; supports pagination for hubs. |
| `read_memory(memory_id, via_key_id?, namespace?)` | Read full memory content, connected keys, and `validity`. Updates access and, when `via_key_id` is supplied, that traversed edge; never confirms or deepens content. |
| `confirm_memory(memory_id, evidence, namespace?, source?)` | Refresh freshness and deepen a current memory after explicit user evidence, an authoritative source, or direct observation. A read alone is not evidence. |
| `remember(content, keys, key_types?, namespace?, ttl_seconds?, decay_profile?, related_to?)` | Save memory with key concepts, optional TTL, and a `transient`, `standard`, `stable`, or `permanent` decay profile. |
| `correct(memory_id, content, keys?, key_types?, ttl_seconds?, decay_profile?, related_to?)` | Versioned update. The immediate predecessor is preserved but inactive; omitted TTL/profile inherit from it. |
| `dismiss(memory_id, key_id, namespace?)` | Negative feedback: the fact is fine, this key should not have surfaced it. Weakens that one edge (floored, never severed) and cancels its pending alias learning |
| `related(memory_id)` | Find memories sharing keys (associative exploration) |
| `forget(memory_id)` | Permanently delete |
| `list_sessions(agent?, limit?)` | Discover recent host-agent conversation sessions (Claude Code, Codex) on this machine, newest first |
| `get_conversation(session_id, turn?, agent?)` | Load original conversation turns from the host agent's on-disk transcript (Claude Code / Codex), normalized to `{turn, role, content, ts}` |
| `list_memories(namespace?)` | List active memories with keys, depth, access count, and validity |
| `remember_batch(items)` | Save multiple memories; each item accepts `ttl_seconds` and `decay_profile` |
| `cleanup_expired()` | Delete memories whose TTL has expired |
| `memory_stats()` | Get current key/memory/link counts |

Scores carry `score_kind`. Key recall exposes cosine-like `key_relevance`; `read_key` exposes `content_relevance` plus a within-key rank score; injected/direct memories expose `relevance_score` separately from their small RRF `rank_score`. Compare thresholds only within the same score kind.

Set `KEYMEM_DIRECT_RECALL=true` to expose a seventeenth compatibility tool, `recall_memories(...)`, with BM25+dense+RRF multi-hop behavior.

A system prompt template is also available via the `memory_system_prompt` MCP prompt — include it to instruct the agent to recall silently, use diverse keys, and never mention the memory system to users.

---

## Local embedding models

**BGE-M3 (recommended, multilingual) — auto-downloaded:** set `LOCAL_EMBEDDING_MODEL=bge-m3` (aliases: `bgem3`, `baai/bge-m3`, `fast-bge-m3`). On first use the model is **fetched automatically if missing** — quantized ONNX (~570MB, from `onnx-community/bge-m3-ONNX`) plus the tokenizer/config (from `BAAI/bge-m3`) — and cached under `~/.keymem/models/bge-m3`. No manual download needed:

```
EMBEDDING_BACKEND=local
LOCAL_EMBEDDING_MODEL=bge-m3
# optional — point at an existing model dir to skip the download (backward compatible):
# LOCAL_EMBEDDING_MODEL_PATH=/absolute/path/to/model-dir   # dir with model.onnx + tokenizer files
# LOCAL_EMBEDDING_MODEL_FILE=model.onnx                    # optional; default is model.onnx
```

> First run downloads ~570MB once, then reuses the cache. If `LOCAL_EMBEDDING_MODEL_PATH` already holds the model it is used **as-is with no download** (a partial dir is self-healed — only missing files are fetched). Online-API backends (OpenAI) and fastembed built-ins are unaffected.

**Cross-encoder reranking (optional direct mode):** set `KEYMEM_DIRECT_RECALL=true` and `KEYMEM_RERANK=true` to re-score `recall_memories()` candidates with `bge-reranker-v2-m3`. The default key-navigation flow does not load the reranker. The model (~570MB, quantized) auto-downloads on first use and caches under `~/.keymem/models/reranker`.

```
KEYMEM_RERANK=true
KEYMEM_DIRECT_RECALL=true
# optional: KEYMEM_RERANK_MODEL_PATH=/dir   KEYMEM_RERANK_POOL=30  (candidates re-scored)
```

> Off by default. If the model cannot load, `recall_memories()` falls back to fused ranking. Query decomposition remains the caller's responsibility.

**Reranker not-found gate (`KEYMEM_RERANK_MIN_SCORE`):** in direct compatibility mode, reject the complete `recall_memories()` result when the top cross-encoder logit is below this floor.

```
KEYMEM_RERANK=true
KEYMEM_DIRECT_RECALL=true
KEYMEM_RERANK_MIN_SCORE=0   # reject when top rerank logit < 0 (bge-reranker-v2-m3 scale)
```

> ⚠️ **Caveats.** (1) Unset by default — no gate. (2) The logit scale is **model-dependent**; `0` (≈ sigmoid 0.5) suits `bge-reranker-v2-m3` (measured: same-language found ≈ +3.9, not-found ≈ −5 to −6) — recalibrate for other rerankers. (3) **Trusted for SAME-LANGUAGE only.** Cross-lingual relevance logits run low even when relevant (KR query ↔ EN memory ≈ −5.4), so the gate auto-**bypasses on a script mismatch** (KR↔Latin) to avoid false-rejecting cross-lingual hits — which means **cross-lingual content must be reachable via bilingual keys** (`["Jiwoo","지우"]`), and cross-lingual *not-found* precision is a known limitation. Leave this off if you can't tag bilingual keys.

> **Prefix behavior:** BGE-M3 does **not** use `passage:`/`query:` prefixes — embeddings are passed through as-is. All other local models (e5, BGE-en, MiniLM) continue to use prefixes unchanged.

> **Recommended for multilingual / cross-lingual use: `bge-m3`.** It separates unrelated queries more reliably and performs substantially better than e5 on the project's Korean↔English fixtures. In optional direct mode, bge-m3's absolute `min_score` gate reaches ≈96% on the gate fixture; e5 requires corpus-specific tuning.

If `OPENAI_API_KEY` is not set and `EMBEDDING_BACKEND` is unset, the server automatically uses the local `fastembed` backend.
For English-only use or lower local resource usage, set `LOCAL_EMBEDDING_MODEL=fast-bge-base-en-v1.5` or `fast-bge-small-en-v1.5`.

> **Switching backends is safe.** The graph records an embedding **fingerprint** (backend + model id) identifying the vector space its embeddings live in. On startup, if the current backend's fingerprint **or** dimension differs from what is stored, the graph **auto-migrates** — every key and memory is re-embedded with the new backend while content, links, depth, and access history are preserved (a `graph.json.bak.*` backup is written first). The fingerprint matters because two models can share a dimension yet produce incompatible vectors (e.g. `fast-multilingual-e5-large` and `bge-m3` are **both 1024-d**); a dimension check alone would miss that swap and silently corrupt every similarity. Disable with `KEYMEM_AUTO_MIGRATE=false`. Re-embedding via OpenAI incurs one-time API cost proportional to your memory count.
>
> **Migrating a pre-fingerprint (legacy) graph across same-dimension models.** A graph written before fingerprinting has no recorded vector space, so a same-dimension model swap off it cannot be detected automatically. Set `KEYMEM_FORCE_REEMBED=true` for **one** startup to re-embed unconditionally and stamp the fingerprint; remove it afterward (left on, it re-embeds on every start). This is exactly the one-shot needed when moving an existing e5 graph to bge-m3.

---

## Data Storage

All data is local. No external database required.

```
~/.keymem/
├── graph.json          # canonical keys, aliases, memories, weighted links
└── conversations/
    └── {session_id}.jsonl   # optional conversation log (only if a host integration writes one)
```

Set `KEYMEM_DATA_DIR` to use a different storage directory.

`get_conversation` / `list_sessions` read the **host coding agent's own transcripts** directly — keymem does not record conversations itself. Locations are auto-detected per OS and honour the agents' env overrides:

- **Claude Code** — `~/.claude/projects/**/{session_id}.jsonl` (`$CLAUDE_CONFIG_DIR`)
- **Codex** — `~/.codex/sessions/**/rollout-*-{session_id}.jsonl` (`$CODEX_HOME`)

`session_id` is restricted to a UUID and resolved within these roots (with symlink checks) to prevent path traversal.

**Access is gated.** Because transcripts are local, potentially sensitive history, `get_conversation` and `list_sessions` are **only exposed when keymem is trusted as the owner's personal local agent** — i.e. a recognized host injected its session env (`CLAUDE_CODE_SESSION_ID` / `CODEX_THREAD_ID`), or you explicitly opt in with `KEYMEM_TRANSCRIPT_ACCESS=true`. Otherwise (a plain server, a remote deployment, a non-owner/custom agent) the two tools are hidden from `tools/list`, calling them is refused, and memories are saved without a host link. Set `KEYMEM_TRANSCRIPT_ACCESS=false` to force-disable even under a host agent.

**Linking a memory to its source conversation.** When you save a memory, keymem stamps the active host session onto its `source` (`host_session` / `host_agent` / `host_turn`) so a recalled memory can drill back to the verbatim exchange via `get_conversation`. The active session is found two ways:

1. **Deterministic** — the host injects its session id into every MCP server it spawns, and keymem reads it directly: Claude Code → `CLAUDE_CODE_SESSION_ID`, Codex → `CODEX_THREAD_ID` (which equals the rollout file's session id). The link is exact, with no guessing.
2. **Heuristic fallback** — for hosts that don't expose a session id (e.g. Claude Desktop), keymem uses the most-recently-modified transcript (with a staleness guard). Reliable for a single active session; if ambiguous, use `list_sessions` to pick the right one.

---

## Benchmarks

On **HotpotQA** bridge questions (real external multi-hop data, gold labels, no LLM judge — and
keys generated *blind* by independent subagents that never saw the question or answer), the
key-graph retrieves **both** gold supporting paragraphs **63%** of the time vs **53%** for flat
semantic retrieval and **35%** for lexical (+10pp / +28pp) — the connected-but-dissimilar case
it's built for. (Honest: my own hand-derived keys inflated this to 78/60; on non-multi-hop
"comparison" questions the graph slightly *hurts*; it's retrieval-recall, not answer accuracy.)
The read path is also O(1) (`read_memory` p50 ~45ms → ~0.01ms @ 500 memories). Full methodology
and caveats: **[BENCHMARKS.md](BENCHMARKS.md)**.

---

## Limitations

- **Linear scan** — suitable for personal use (~10k memories). FAISS/ChromaDB integration planned for larger scale.
- **Agentic round trips** — the default `recall → read_key → read_memory` flow is more controllable and context-efficient, but needs more tool calls than one-shot retrieval.
- **Hub breadth** — broad keys can connect many memories. `read_key()` paginates hubs; the agent must choose whether to continue paging or follow a more specific adjacent key.
- **Agent quality matters** — key selection on `remember` affects retrieval quality. System prompt tuning is important.
- **Cross-lingual content bias** — with multilingual e5, raw content similarity favors same-language memories regardless of meaning. Tag memories with multilingual keys so the key graph (not biased content cosine) carries cross-lingual recall.
- **Threshold calibration** — thresholds are tuned per embedding model. A new/uncalibrated model falls back to the BGE profile (with a warning); recalibrate via the `KEYMEM_*` env overrides.

---

## Testing

```bash
pnpm test                        # unit tests (fast, no model download)
tsx test/scenarios.ts            # 21 end-to-end behavioral checks (local e5)
tsx test/robustness.ts           # threshold overrides + Hebbian pollution bounds
tsx test/migration.ts            # backend/dimension switch auto-migration (no brick)
tsx test/nhop.ts                 # N-hop chained traversal (recall hops parameter)
tsx test/depth-noise.ts          # deep-hop noise bounds + relative score floor
tsx test/live-multilingual.ts    # interactive multilingual recall demo

# Manual retriever-quality check (NOT part of pnpm test):
EMBEDDING_BACKEND=local npx tsx test/retriever-quality.live.ts
# For bge-m3: also set LOCAL_EMBEDDING_MODEL=bge-m3 LOCAL_EMBEDDING_MODEL_PATH=/abs/dir
```

`scenarios.ts` and `robustness.ts` exercise the real local embedding backend (direct/associative/cross-lingual recall, versioning, depth growth, dedup, TTL, Hebbian learning, namespace isolation). They double as a recalibration harness when tuning thresholds for a new model.

---

## Roadmap

- [ ] FAISS/ChromaDB for scale
- [ ] Coding agent profile (different key strategies for code context)
- [ ] Memory export/import
- [ ] Multi-user support

---

## Author

**donggyun112** — [github.com/donggyun112](https://github.com/donggyun112)

Repository: [donggyun112/keymem](https://github.com/donggyun112/keymem) · Issues & PRs welcome.

## License

MIT © [donggyun112](https://github.com/donggyun112)
