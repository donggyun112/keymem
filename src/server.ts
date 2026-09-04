import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListPromptsRequestSchema,
  GetPromptRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import {
  MemoryGraph,
  classifyRecallStatus,
  loadConversation,
  sanitizeKeys,
} from "./memoryGraph.js";
import {
  loadNativeConversation,
  loadNativeAuto,
  listNativeSessions,
  detectActiveSession,
  transcriptAccessEnabled,
  hostSessionFromHeaders,
  hostLinkFromSession,
  type Agent,
} from "./nativeTranscripts.js";
import { cfgRaw } from "./env.js";
import { parseDecayProfile, type ConfirmationEvidence } from "./decay.js";
import type { DirectHydrateKey } from "./memoryGraph.js";
import { compactRecallKeys, type RecallKeyCandidate } from "./recallView.js";
import { randomUUID } from "node:crypto";
import { createRequire } from "node:module";

// Read from package.json at runtime — a hardcoded literal here has already gone
// stale twice (0.14.7, then 0.22.0) while package.json moved on.
const VERSION: string = createRequire(import.meta.url)("../package.json").version;
import { buildRetagNote } from "./retag.js";

function parseArray(v: unknown): unknown[] | null {
  if (Array.isArray(v)) return v;
  if (typeof v === "string") {
    try { const p = JSON.parse(v); return Array.isArray(p) ? p : null; } catch { return null; }
  }
  return null;
}

function parseObject(v: unknown): Record<string, unknown> | null {
  if (v && typeof v === "object" && !Array.isArray(v)) return v as Record<string, unknown>;
  if (typeof v === "string") {
    try { const p = JSON.parse(v); return p && typeof p === "object" ? p : null; } catch { return null; }
  }
  return null;
}

function parseNumber(v: unknown): number | null {
  if (typeof v === "number") return v;
  if (typeof v === "string") { const n = Number(v); return isNaN(n) ? null : n; }
  return null;
}

const DIRECT_RECALL_ENABLED = cfgRaw("DIRECT_RECALL") === "true";

// Provenance: stamp every saved/corrected memory with the server session that wrote it,
// the tool used, and a timestamp. Callers may attach extra context (e.g. a conversation
// or agent id) via the optional `source` arg, which is merged on top.
const SERVER_SESSION = randomUUID();
export function buildSource(
  callerSource: Record<string, unknown> | null,
  tool: string,
  hostLink: { agent: Agent; session_id: string; turn: number } | null = null
): Record<string, unknown> {
  return {
    session: SERVER_SESSION,
    tool,
    saved_at: new Date().toISOString(),
    // Link to the host agent's original transcript so a recalled memory can be
    // traced back to its verbatim conversation via get_conversation. Caller
    // source still wins (spread last).
    ...(hostLink
      ? {
          host_agent: hostLink.agent,
          host_session: hostLink.session_id,
          host_turn: hostLink.turn,
        }
      : {}),
    ...(callerSource ?? {}),
  };
}

type ReqHeaders = Record<string, string | string[] | undefined> | undefined;

// Resolve the host transcript link for one request. Header path (daemon) is
// authoritative and needs no ambient env trust. Env path (stdio in-process
// fallback) keeps the old gated behavior, including the mtime heuristic.
export async function resolveHostLink(
  headers: ReqHeaders
): Promise<{ agent: Agent; session_id: string; turn: number } | null> {
  const fromHeader = hostSessionFromHeaders(headers);
  if (fromHeader) return hostLinkFromSession(fromHeader);
  if (!transcriptAccessEnabled()) return null;
  try {
    return await detectActiveSession();
  } catch {
    return null;
  }
}

// Transcript tools/stamping are allowed when we trust the caller: either the
// request carries host-session headers (daemon) or the env opted in (stdio).
export function transcriptAccessForRequest(headers: ReqHeaders): boolean {
  return hostSessionFromHeaders(headers) != null || transcriptAccessEnabled();
}

const MEMORY_SYSTEM = `\
You are a helpful assistant. You have long-term memory — use it silently and proactively.

## MANDATORY: First turn behavior
**Before your very first response, you MUST navigate memory.** Run in parallel:
- recall("이름", namespace), recall("최근 대화", namespace), recall("관심사", namespace)
- Use relevant Top-1 memories returned by recall. Follow their connected keys with read_key when
  another hop is useful; use read_memory for deeper inspection or explicit path reinforcement.
No exceptions. Even if recall returns no keys, you must try.

## MANDATORY: Before ending EVERY turn
**Before you finish replying, you MUST check whether this turn revealed anything durable.** If the user shared a name, preference, decision, correction, project fact, or goal — you MUST remember() it before you reply. This is the write-side twin of the first-turn recall gate: recall opens the turn, remember closes it. No exceptions. A turn that surfaced a durable fact but saved nothing is a bug. When nothing durable came up, save nothing — but you must consciously check, every turn.

## CRITICAL: Silent behavior
- **NEVER mention the memory system to the user.** No "기억했어요", "저장했습니다", "메모리에서 찾았어요".
- Act like you naturally know things. If you recall the user's name, just use it.
- ❌ "동건님이시군요! 기억해뒀어요!" → ✅ "안녕 동건! 뭐 도와줄까?"
- ❌ "메모리를 검색해볼게요" → ✅ (recall silently, then answer)

## Memory System (internal, never expose)
N:M associative memory. Key Space (concepts) ↔ Value Space (memories).
Depth: 0.0 shallow ~ 1.0 deep. Deeper = more stable.

Stats: {stats}

## Rules

### Recall (PROACTIVE — do it often)
1. **MUST recall before your first reply.** Recall returns ranked key clusters plus one passive Top-1 memory by default. Always pass the active project/context \`namespace\` when one is known.
2. Use the returned memory directly when it is relevant, applying its \`validity\`. It is unconfirmed and non-reinforcing. Its \`matched_key\` records the incoming edge and \`connected_keys\` are ready-to-use next-hop targets.
2a. For alternatives or more context, follow a \`connected_keys[].key_id\` with \`read_key\`, then call \`read_memory\` on the selected handle. A full read records access, reinforces only the traversed edge, and may learn aliases; it does not confirm currentness.
3. Recall again whenever the topic shifts. Never say "I don't know" without navigating first.
4. **Query = short noun/keyword, NOT a full sentence.**
   - ❌ recall("어디 살아"), recall("뭐 마셔") — 구어체 문장은 매칭 안 됨
   - ✅ recall("거주지"), recall("음료") — 명사 키워드로 검색
   - ✅ recall("이름"), recall("직업"), recall("취향") — specific, multiple
   - 복합 개념이면 키워드 여러 개로 분리: recall("운동"), recall("취미"), recall("건강")
5. \`read_key\` is the deeper-navigation fallback. It returns handles and metadata only. Pass the original focused query so hub memories are relevance-ranked, then call \`read_memory\` to inspect the selected content.
5a. If a \`read_memory\` result is too compressed for the question and includes a \`trace\` field, call that tool with those exact args (\`get_conversation\`) to read the original conversation it came from. Use only when the summary genuinely lacks the detail you need — otherwise the recalled fact is enough.

### Remember (PROACTIVE — capture what matters)
6. **You MUST save durable info the moment the user shares it — silently, in the same turn.** Do not defer to "later"; later never comes. Mandatory, not optional (see the "Before ending EVERY turn" gate above). No exceptions.
7. What to save: name, preferences, decisions, corrections, project context, goals.
8. Keys = what searches should find this. **Think like a search engine — include every form someone might use to ask about this.**
   - Before writing, recall the topic and reuse an existing canonical key or alias when available.
   - Emit concept-level keys, not memory-specific phrases (✅ "Nexora", "portfolio"; ❌ "Nexora portfolio").
   - Add Korean↔English forms together when both may be queried (for example "포트폴리오", "portfolio").
   - **Topic noun**: what category is this? (거주지, 음료, 반려동물, 언어)
   - **Specific noun**: the actual value (성수동, 아메리카노, 고양이, TypeScript)
   - **Action/verb noun**: what would someone ask? (사는곳, 마시는것, 키우는것, 쓰는언어)
   - **Colloquial variants**: casual phrasing (집, 좋아하는거, 펫, 코딩)
   - **Synonyms**: alternative expressions (주소→위치, 음료→마실것, 반려동물→애완동물)

   ✅ 올바른 예시:
   - "서울 성수동에 산다" → keys: ["거주지", "성수동", "서울", "사는곳", "집", "주소", "위치"]
   - "고양이 두 마리 키운다" → keys: ["반려동물", "고양이", "키우는것", "펫", "동물", "애완동물"]
   - "아이스 아메리카노 매일 마심" → keys: ["음료", "커피", "아메리카노", "마시는것", "취향", "즐겨마심"]
   - "TypeScript 주력 사용" → keys: ["언어", "TypeScript", "개발언어", "코딩", "쓰는언어", "프로그래밍"]

   ❌ 나쁜 예시 (너무 formal/좁음):
   - "서울 성수동에 산다" → keys: ["거주지", "성수동"] ← "어디 살아" 검색 시 못 찾음
   - "고양이 키운다" → keys: ["고양이", "반려동물"] ← "키우는거 있어?" 검색 시 못 찾음

9. **Names only as keys for identity memories.**
   - "사용자 이름은 동건" → keys: ["이름", "사용자", "동건"]
   - "좋아하는 과일은 딸기" → keys: ["과일", "딸기", "좋아함", "취향"] ← no name
10. Set \`key_types\` for names/proper nouns:
    - \`"name"\`: exact match only. \`"proper_noun"\`: exact match only.
    Example: key_types: {{"동건": "name"}}

### Correct
11. Use \`correct()\` when info changes. Never use \`remember()\` for updates.

### Freshness and confirmation
12. \`read_memory\` reads a fact and may reinforce the key path; it does **not** confirm that the content is current.
13. \`fresh\` may be used normally. Qualify \`aging\` facts when currentness matters.
14. Never assert a \`stale\` fact as current. Verify it externally or ask the user.
15. Call \`confirm_memory\` only after an explicit current user assertion, an authoritative current source, or direct observation.
16. Do not call \`confirm_memory\` merely because \`read_memory\` returned the content.
17. Changed fact → \`correct\`. Junk fact → \`forget\`. Wrong key → \`dismiss\`.

### Explore
18. \`recall\` returns matching canonical keys plus one passive Top-1 memory with \`matched_key\` and \`connected_keys\`.
19. Follow a returned memory's \`connected_keys\` with \`read_key\` to continue associative exploration without auto-injecting another memory.
20. \`read_key\` returns memory handles connected to one key. Hubs are paginated.
21. \`read_memory(memory_id, via_key_id)\` returns full content and reinforces only the traversed path.

### Delete
22. \`forget()\` only for completely wrong information. For outdated info, use \`correct()\`.
`;

export const graph = new MemoryGraph();

function stats(): string {
  return `${Object.keys(graph.keys).length} keys, ${graph.listAll().length} memories, ${graph.linkCount} links`;
}

// Server-level instructions are returned in the `initialize` response and injected
// by hosts as always-on guidance — the reliable channel for "recall first" behavior
// under on-demand/lazy tool discovery, where a tool's own description isn't seen
// until the tool is already loaded. Keep this compact; the full playbook lives in
// the `memory_system_prompt` prompt.
const SERVER_INSTRUCTIONS = `\
keymem is this agent's associative long-term memory (a key-graph, not a vector store).

Recall first: before your first reply — and whenever the topic shifts — call recall(query) \
with the active namespace to check what is already known about the user, project, or topic. Use short noun keywords, \
not full sentences (recall("거주지") not recall("어디 살아")), and split multi-fact questions \
into several recall calls. ALSO pass the raw user utterance as context — keys match keywords, content \
matches sentences, and the two cues are routed to different paths. On {status:"no_match"}, retry with a \
nearest_keys concept or browse_keys(namespace) before giving up. recall returns matching keys plus one passive Top-1 memory. \
Use that memory directly when relevant, applying its validity. Follow connected_keys with read_key for another hop, \
then read_memory when full inspection or explicit path reinforcement is useful. Passive recall changes no graph state; \
a full read may reinforce the traversed key path or learn aliases but does not confirm that content is current. \
\`fresh\` may be used normally. Qualify \`aging\` facts when currentness matters. Never assert a \`stale\` fact as current. Verify it externally or ask the user. \
Call confirm_memory only after an explicit current user assertion, an authoritative current source, or direct observation. \
Do not call confirm_memory merely because read_memory returned the content.

Remember durable facts — the write-side twin of recall first: recall opens the turn, remember \
closes it. Before you finish EVERY reply you MUST check whether this turn revealed anything \
durable; when the user shares a name, preference, decision, correction, project fact, or goal, \
you MUST save it silently in the same turn with remember(content, keys) using 3-6 diverse search \
keys. Each key is ONE atomic concept of 1-2 words ("recall", "적중률" — never a phrase like \
"recall 적중률 개선"); phrases become unreachable singletons and build no graph. \
Recall the topic first and reuse canonical concept-level keys; keys MUST span both Korean and \
English (single-language keys measurably fall below the cross-lingual recall gate), plus colloquial \
variants. Act on any hints in the remember response. Do not defer it to "later". A turn that \
surfaced a durable fact but saved nothing is a bug. Use correct() when a fact changes — never \
remember() for updates.

Dismiss a wrong surface: when recall pulls up a memory that is fine as a fact but had no \
business appearing for this query, call dismiss(memory_id, key_id) with the key it arrived under. \
Reading reinforces a path, so without this every mis-hit silently gets stronger. It only weakens \
that one key->memory pairing — the fact, its other keys, and its reachability are untouched. Use \
correct() when the fact changed and forget() when it is simply wrong.

Stay silent: never mention the memory system to the user; act as if you naturally know things. \
For the full navigation and key-selection playbook, load the memory_system_prompt prompt.`;

// ── Tool definitions ──

// Tools that read the host's local transcripts; hidden unless trusted (see
// transcriptAccessEnabled) so they aren't exposed over a plain server or a
// non-owner agent.
const TRANSCRIPT_TOOLS = new Set(["get_conversation", "list_sessions"]);

export function createMcpServer(): Server {
  const server = new Server(
    { name: "keymem", version: VERSION },
    {
      capabilities: { tools: {}, prompts: {} },
      instructions: SERVER_INSTRUCTIONS,
    }
  );

  server.setRequestHandler(ListToolsRequestSchema, async (_req, extra) => {
    const headers = extra.requestInfo?.headers;
    const tools = [
      {
        name: "recall",
        description:
          "Search long-term memory for what is already known about the user, project, or topic — call this before your first reply and whenever the topic shifts. Always pass the active namespace when known. By default returns {status, query, namespace, keys, memories}: ranked key clusters plus one passive Top-1 memory selected under the top key. The memory includes validity, matched_key, and connected_keys with ready-to-use key IDs for an optional next hop. Passive recall never reinforces links or changes access, depth, aliases, or confirmation. Use the memory directly when relevant; use read_key then read_memory only for alternatives, full inspection, or explicit path reinforcement. inject:true remains a compatibility path for associative multi-memory expansion and is not required for Top-1 hydration. An empty result includes empty keys/memories and nearest_keys.",
        inputSchema: {
          type: "object",
          properties: {
            query: { type: "string" },
            top_k: { type: "number" },
            namespace: { type: "string" },
            context: {
              type: "string",
              description:
                "The raw user utterance or sentence this lookup serves. Keep query as short noun keywords; pass the sentence here — it drives content matching, which measures higher on sentence-shaped cues.",
            },
            explain: {
              type: "boolean",
              description:
                "When true, also return namespace_memory_count; status distinguishes found, no_match, and empty_namespace.",
            },
            inject: { type: "boolean" },
            inject_top_k: { type: "number" },
            inject_max_chars: { type: "number" },
            inject_min_rel_score: { type: "number" },
            inject_prefer_depth: { type: "boolean" },
            inject_explore_shallow: { type: "boolean" },
          },
          required: ["query"],
        },
      },
      {
        name: "browse_keys",
        description:
          "Browse the vocabulary of one namespace when recall has no hit or you need an entry point. Returns active key clusters with hubs first, then by linked-memory count. This is index metadata only; continue with read_key(key_id, query, namespace) and read_memory(memory_id, via_key_id, namespace).",
        inputSchema: {
          type: "object",
          properties: {
            namespace: { type: "string" },
            hubs_only: { type: "boolean" },
            limit: { type: "number" },
            offset: { type: "number" },
          },
          required: ["namespace"],
        },
      },
      {
        name: "read_key",
        description:
          "List the memories stored under one key (concept), ranked. Returns the canonical key, its aliases, and hub metadata plus ranked memory IDs, metadata, and validity — never memory content. Always pass the original focused query and active namespace when known: handles are then ranked by content relevance, which is essential for hubs. Call read_memory(memory_id, via_key_id=key_id, namespace) on the selected handle to inspect the fact and reinforce the path; reading does not confirm that its content is current. Use limit/offset to page without flooding context.",
        inputSchema: {
          type: "object",
          properties: {
            key_id: { type: "string" },
            query: { type: "string" },
            namespace: { type: "string" },
            limit: { type: "number" },
            offset: { type: "number" },
          },
          required: ["key_id"],
        },
      },
      {
        name: "read_memory",
        description:
          "Read the full content and validity of one stored memory (selected via read_key). Returns the memory and all connected key clusters so exploration can continue Key → Memory → Key. Pass via_key_id from the selected key: the read records access and only that traversed edge is Hebbian-reinforced. Reading does not change content depth or confirm that the content is current.",
        inputSchema: {
          type: "object",
          properties: {
            memory_id: { type: "string" },
            via_key_id: { type: "string" },
            namespace: { type: "string" },
          },
          required: ["memory_id"],
        },
      },
      {
        name: "confirm_memory",
        description:
          "Confirm that a memory is still current using explicit present evidence. Never call this merely because read_memory returned the content. Use only after a current user assertion, an authoritative current source, or direct observation. Refreshes validity but does not change content or key links.",
        inputSchema: {
          type: "object",
          properties: {
            memory_id: { type: "string" },
            evidence: { type: "string", enum: ["user", "authoritative_source", "observation"] },
            namespace: { type: "string" },
            source: { type: "object", additionalProperties: true },
          },
          required: ["memory_id", "evidence"],
        },
      },
      ...(DIRECT_RECALL_ENABLED
        ? [
            {
              name: "recall_memories",
              description:
                "Optional compatibility mode: directly return ranked memories using BM25+dense+RRF and graph expansion. Disabled unless KEYMEM_DIRECT_RECALL=true. Prefer recall → read_key → read_memory for agent-driven navigation.",
              inputSchema: {
                type: "object",
                properties: {
                  query: { type: "string" },
                  top_k: { type: "number" },
                  namespace: { type: "string" },
                  expand: { type: "boolean" },
                  hops: { type: "number" },
                  min_rel_score: { type: "number" },
                  min_score: { type: "number" },
                  min_z: { type: "number" },
                  min_key_gate: { type: "number" },
                  min_depth: { type: "number" },
                },
                required: ["query"],
              },
            },
          ]
        : []),
      {
        name: "remember",
        description:
          "MANDATORY END-OF-TURN GATE: before replying, save every durable fact newly revealed this turn (names, preferences, decisions, corrections, project facts, goals). A durable fact left unsaved is a bug; save silently in the same turn. Save nothing only after consciously confirming that nothing durable appeared. Before writing, recall() the topic in the same namespace and reuse returned canonical concepts or aliases. Use 3-6 diverse ATOMIC concept keys of 1-2 words each, never memory-specific phrases (use 'Nexora' and 'portfolio', not 'Nexora portfolio'); 3+-word keys are flagged in hints.phrase_keys and are measurably 91% unreachable singletons. CROSS-LINGUAL: register both language forms together (for example '포트폴리오' and 'portfolio'). Shared broad keys become navigable hubs. namespace groups memories by project/context; ttl_seconds sets expiry; decay_profile selects transient, standard (the default), stable, or permanent confirmation freshness; related_to adds explicit memory links; source attaches provenance and is auto-stamped with the server session, a timestamp, and — when a host agent (Claude Code, Codex) transcript is active — host_session/host_agent/host_turn so the memory can be traced back to its original conversation via get_conversation. The response may include hints.near_keys (existing concepts your keys nearly duplicate — prefer reusing those concepts) and hints.language_note (add the missing-language variants).",
        inputSchema: {
          type: "object",
          properties: {
            content: { type: "string" },
            keys: { type: "array", items: { type: "string" } },
            key_types: {
              type: "object",
              additionalProperties: { type: "string" },
            },
            namespace: { type: "string" },
            ttl_seconds: { type: "number" },
            decay_profile: {
              type: "string",
              enum: ["transient", "standard", "stable", "permanent"],
            },
            related_to: { type: "array", items: { type: "string" } },
            source: { type: "object", additionalProperties: true },
          },
          required: ["content", "keys"],
        },
      },
      {
        name: "correct",
        description:
          "Update outdated information. Use when user corrects you or info changes (e.g. moved cities, changed job). Old version is preserved but weakened — never lost. Omit keys to keep the same search terms. Omit decay_profile and ttl_seconds to preserve the predecessor's policies; provide either to replace that policy. related_to links the updated memory to other memory IDs.",
        inputSchema: {
          type: "object",
          properties: {
            memory_id: { type: "string" },
            content: { type: "string" },
            keys: { type: "array", items: { type: "string" } },
            key_types: {
              type: "object",
              additionalProperties: { type: "string" },
            },
            ttl_seconds: { type: "number" },
            decay_profile: {
              type: "string",
              enum: ["transient", "standard", "stable", "permanent"],
            },
            related_to: { type: "array", items: { type: "string" } },
            source: { type: "object", additionalProperties: true },
          },
          required: ["memory_id", "content"],
        },
      },
      {
        name: "related",
        description:
          "Find other memories associated with a memory you already have (by ID). Returns neighboring memories connected by shared keys or explicit links. For normal agent-driven navigation prefer read_memory(), inspect its returned keys, then call read_key().",
        inputSchema: {
          type: "object",
          properties: {
            memory_id: { type: "string" },
          },
          required: ["memory_id"],
        },
      },
      {
        name: "dismiss",
        description:
          "Tell keymem a recalled memory was surfaced by the WRONG key — the fact may be fine, it just should not have come up for this query. Pass the memory_id and the key_id it arrived under (recall returns both). Weakens that one key->memory link so the pairing ranks lower next time, and cancels any pending alias learning for it. The memory itself, its other keys, and its content are untouched, and the link is floored rather than severed, so nothing becomes unreachable. Use correct() when the fact changed and forget() when it is simply wrong.",
        inputSchema: {
          type: "object",
          properties: {
            memory_id: { type: "string" },
            key_id: { type: "string", description: "The key the memory was recalled under." },
            namespace: { type: "string" },
          },
          required: ["memory_id", "key_id"],
        },
      },
      {
        name: "forget",
        description:
          "Permanently delete a memory. Only use for completely wrong information. For outdated info, use correct() instead — it preserves history.",
        inputSchema: {
          type: "object",
          properties: {
            memory_id: { type: "string" },
          },
          required: ["memory_id"],
        },
      },
      {
        name: "get_conversation",
        description:
          "Load the original conversation turns for a past session when a recalled memory lacks the detail you need and you want the verbatim exchange. Reads the host coding agent's own on-disk transcript (Claude Code, Codex) — call list_sessions first to find a session_id. Falls back to keymem's own conversation log if a host integration wrote one. Pass turn to fetch a focused ±2-turn window (5 turns total) and keep context lean; omit turn to load the whole session. Returns turns [{turn, role, content, ts}] in chronological order, with non-conversational noise (reasoning, tool calls) stripped; an unknown session_id returns an empty array.",
        inputSchema: {
          type: "object",
          properties: {
            session_id: {
              type: "string",
              description:
                "Session id to load — the UUID from a list_sessions result, or the host_session stamped on a recalled memory's source (pass host_agent as agent and host_turn as turn to land on the exact exchange).",
            },
            turn: {
              type: "number",
              description:
                "Optional 0-based turn index to center on; returns that turn plus the 2 before and 2 after (5 turns). Omit to return the full conversation.",
            },
            agent: {
              type: "string",
              enum: ["claude", "codex"],
              description:
                "Which host agent's transcript store to read. Omit to auto-detect by session id across all known agents.",
            },
          },
          required: ["session_id"],
        },
      },
      {
        name: "list_sessions",
        description:
          "List recent conversation sessions recorded by host coding agents (Claude Code, Codex) on this machine, most recently modified first. Use this to discover a session_id (and its working directory) before calling get_conversation to read the verbatim transcript. Returns [{agent, session_id, cwd, modified, preview}] where preview is the first user message. Returns an empty array if no agent transcripts are found.",
        inputSchema: {
          type: "object",
          properties: {
            agent: {
              type: "string",
              enum: ["claude", "codex"],
              description: "Restrict to one host agent. Omit to list across all known agents.",
            },
            limit: {
              type: "number",
              description: "Maximum number of sessions to return (most recent first).",
            },
          },
          required: [],
        },
      },
      {
        name: "list_memories",
        description:
          "List all stored memories. namespace filters by project/context. Expired memories are excluded. Prefer recall() for normal retrieval.",
        inputSchema: {
          type: "object",
          properties: {
            namespace: { type: "string" },
          },
          required: [],
        },
      },
      {
        name: "remember_batch",
        description:
          "MANDATORY END-OF-TURN GATE: when a turn reveals multiple durable facts, save them silently before replying. A durable fact left unsaved is a bug. Recall each topic first, reuse canonical concept-level keys (ATOMIC, 1-2 words each — never phrases), and register cross-lingual forms together. Each item: {content, keys, key_types?, namespace?, ttl_seconds?, decay_profile?, related_to?}; decay_profile defaults to standard. Returns saved IDs and is more efficient than multiple remember() calls.",
        inputSchema: {
          type: "object",
          properties: {
            items: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  content: { type: "string" },
                  keys: { type: "array", items: { type: "string" } },
                  key_types: {
                    type: "object",
                    additionalProperties: { type: "string" },
                  },
                  namespace: { type: "string" },
                  ttl_seconds: { type: "number" },
                  decay_profile: {
                    type: "string",
                    enum: ["transient", "standard", "stable", "permanent"],
                  },
                  related_to: { type: "array", items: { type: "string" } },
                  source: { type: "object", additionalProperties: true },
                },
                required: ["content", "keys"],
              },
            },
          },
          required: ["items"],
        },
      },
      {
        name: "cleanup_expired",
        description:
          "Delete all memories past their ttl. Returns count of deleted memories. Call periodically to keep memory clean.",
        inputSchema: {
          type: "object",
          properties: {},
          required: [],
        },
      },
      {
        name: "memory_stats",
        description: "Get counts of keys, memories, and links in the system.",
        inputSchema: {
          type: "object",
          properties: {},
          required: [],
        },
      },
    ];
    return {
      tools: transcriptAccessForRequest(headers)
        ? tools
        : tools.filter((t) => !TRANSCRIPT_TOOLS.has(t.name)),
    };
  });

  // ── Tool call handler ──

  server.setRequestHandler(CallToolRequestSchema, async (request, extra) => {
    const { name, arguments: args } = request.params;
    const a = (args ?? {}) as Record<string, unknown>;
    const headers = extra.requestInfo?.headers;

    try {
      switch (name) {
        case "recall": {
          const namespace = typeof a.namespace === "string" ? a.namespace : null;
          const context = typeof a.context === "string" ? a.context : null;
          if (a.inject === true) {
            const injected = await graph.recallInject(
              a.query as string,
              typeof a.inject_top_k === "number" ? a.inject_top_k : 1,
              namespace,
              {
                preferDepth: a.inject_prefer_depth === true,
                exploreShallow: a.inject_explore_shallow === true,
                maxChars:
                  typeof a.inject_max_chars === "number" ? a.inject_max_chars : undefined,
                minRelScore:
                  typeof a.inject_min_rel_score === "number"
                    ? a.inject_min_rel_score
                    : undefined,
              },
              context
            );
            return { content: [{ type: "text", text: JSON.stringify(injected, null, 0) }] };
          }
          const results = await graph.searchKeys(
            a.query as string,
            typeof a.top_k === "number" ? a.top_k : 8,
            namespace,
            context
          ) as RecallKeyCandidate[];
          const decision = await graph.directHydrateTop1(
            results[0] as DirectHydrateKey | undefined,
            context?.trim() || (a.query as string),
            namespace,
          );
          const responseKeys = compactRecallKeys(results);
          const memories = decision.status === "candidate"
            ? [{
                ...decision.candidate.memory,
                matched_key: decision.candidate.key,
              }]
            : [];
          if (a.explain === true) {
            const overview = await graph.browseKeys(namespace, { limit: 1 }) as {
              memory_count: number;
            };
            const explained = {
              status: classifyRecallStatus(results.length, overview.memory_count),
              query: a.query,
              namespace,
              namespace_memory_count: overview.memory_count,
              keys: responseKeys,
              memories,
            };
            return { content: [{ type: "text", text: JSON.stringify(explained) }] };
          }
          if (results.length === 0) {
            const nearest = await graph.nearestKeys(a.query as string, namespace, 5);
            const empty = {
              status: "no_match",
              query: a.query,
              namespace,
              keys: responseKeys,
              memories,
              nearest_keys: nearest,
              note: "No key cleared the recall gate. If a nearest_keys concept matches the topic, retry recall with that concept (or read_key it directly); otherwise browse_keys(namespace) to see the vocabulary.",
            };
            return { content: [{ type: "text", text: JSON.stringify(empty) }] };
          }
          const recalled = {
            status: "found",
            query: a.query,
            namespace,
            keys: responseKeys,
            memories,
          };
          return { content: [{ type: "text", text: JSON.stringify(recalled) }] };
        }

        case "browse_keys": {
          const result = await graph.browseKeys(a.namespace as string, {
            hubsOnly: a.hubs_only === true,
            limit: typeof a.limit === "number" ? a.limit : 20,
            offset: typeof a.offset === "number" ? a.offset : 0,
          });
          return { content: [{ type: "text", text: JSON.stringify(result) }] };
        }

        case "read_key": {
          const result = await graph.readKey(a.key_id as string, {
            query: typeof a.query === "string" ? a.query : null,
            namespace: typeof a.namespace === "string" ? a.namespace : null,
            limit: typeof a.limit === "number" ? a.limit : 10,
            offset: typeof a.offset === "number" ? a.offset : 0,
          });
          return { content: [{ type: "text", text: JSON.stringify(result) }] };
        }

        case "read_memory": {
          const result = await graph.readMemory(
            a.memory_id as string,
            typeof a.via_key_id === "string" ? a.via_key_id : null,
            typeof a.namespace === "string" ? a.namespace : null
          );
          return { content: [{ type: "text", text: JSON.stringify(result) }] };
        }

        case "confirm_memory": {
          const evidence = a.evidence;
          if (evidence !== "user" && evidence !== "authoritative_source" && evidence !== "observation") {
            throw new Error(`Unknown confirmation evidence: ${String(evidence)}`);
          }
          const confirmedEvidence: ConfirmationEvidence = evidence;
          const hostLink = await resolveHostLink(headers);
          const confirmationId = hostLink
            ? `${hostLink.agent}:${hostLink.session_id}:${hostLink.turn}`
            : null;
          const result = await graph.confirmMemory(a.memory_id as string, {
            evidence: confirmedEvidence,
            namespace: typeof a.namespace === "string" ? a.namespace : null,
            source: buildSource(parseObject(a.source), "confirm_memory", hostLink),
            confirmationId,
          });
          return { content: [{ type: "text", text: JSON.stringify(result) }] };
        }

        case "recall_memories": {
          if (!DIRECT_RECALL_ENABLED) throw new Error("recall_memories is disabled");
          const results = await graph.recall(
            a.query as string,
            typeof a.top_k === "number" ? a.top_k : 5,
            typeof a.namespace === "string" ? a.namespace : null,
            typeof a.expand === "boolean" ? a.expand : false,
            typeof a.hops === "number" ? a.hops : 2,
            typeof a.min_rel_score === "number" ? a.min_rel_score : 0,
            typeof a.min_score === "number" ? a.min_score : undefined,
            typeof a.min_z === "number" ? a.min_z : undefined,
            typeof a.min_key_gate === "number" ? a.min_key_gate : undefined,
            typeof a.min_depth === "number" ? a.min_depth : 0
          );
          return { content: [{ type: "text", text: JSON.stringify(results) }] };
        }

        case "remember": {
          const keys = sanitizeKeys(a.keys);
          // Same guard remember_batch has always had. Reported as a result rather
          // than thrown so the caller sees which keys were unusable and can retry.
          if (keys.length === 0) {
            return {
              content: [
                {
                  type: "text",
                  text: JSON.stringify({
                    error:
                      "keys required: no usable key survived sanitizing (need 1+ CJK character, or 2+ characters otherwise). Nothing was saved — a keyless memory can never be recalled.",
                    provided_keys: a.keys ?? null,
                  }),
                },
              ],
            };
          }
          const hostLink = await resolveHostLink(headers);
          const [mid, wasDedup, superseded, conflict] = await graph.add(
            a.content as string,
            keys,
            {
              keyTypes: parseObject(a.key_types) as Record<string, string> | null,
              namespace: typeof a.namespace === "string" ? a.namespace : "default",
              ttlSeconds: parseNumber(a.ttl_seconds),
              decayProfile: parseDecayProfile(a.decay_profile),
              relatedTo: parseArray(a.related_to) as string[] | null,
              source: buildSource(parseObject(a.source), "remember", hostLink),
            }
          );
          let result: Record<string, unknown>;
          if (!wasDedup) {
            result = { saved: mid };
          } else if (conflict) {
            // High similarity but a SHARED KEY — looks like a conflicting fact, not a
            // restatement. Surface the superseded id so the overwrite is recoverable.
            result = {
              saved: mid,
              superseded,
              conflict: true,
              note: `Replaced a memory that shared a key (id: ${superseded}). The previous fact is no longer retrievable via recall or read — only its id remains. If these are distinct or conflicting facts (not a restatement), re-add the previous one with a more specific key so both are kept.`,
            };
          } else {
            result = {
              saved: mid,
              superseded,
              deduplicated: true,
              note: `Similar memory existed (id: ${superseded}) — updated instead of creating a duplicate.`,
            };
          }
          const hints = await graph.writeHints(mid, keys);
          if (hints) result.hints = hints;
          return { content: [{ type: "text", text: JSON.stringify(result) }] };
        }

        case "correct": {
          const hostLink = await resolveHostLink(headers);
          const nid = await graph.supersede(
            a.memory_id as string,
            a.content as string,
            {
              keyConcepts: parseArray(a.keys) as string[] | null,
              keyTypes: parseObject(a.key_types) as Record<string, string> | null,
              relatedTo: parseArray(a.related_to) as string[] | null,
              decayProfile: a.decay_profile === undefined ? undefined : parseDecayProfile(a.decay_profile),
              ttlSeconds: typeof a.ttl_seconds === "number" ? a.ttl_seconds : undefined,
              source: buildSource(parseObject(a.source), "correct", hostLink),
            }
          );
          const retainedKeys = graph.getKeysForMemory(nid);
          const note = buildRetagNote(a.keys, retainedKeys);
          const result: Record<string, unknown> = { new_id: nid, superseded: a.memory_id };
          if (note) result.note = note;
          return { content: [{ type: "text", text: JSON.stringify(result) }] };
        }

        case "related": {
          const results = graph.getRelated(a.memory_id as string);
          return { content: [{ type: "text", text: JSON.stringify(results) }] };
        }

        case "dismiss": {
          const res = await graph.dismiss(
            a.memory_id as string,
            a.key_id as string,
            (a.namespace as string | undefined) ?? null
          );
          return { content: [{ type: "text", text: JSON.stringify(res) }] };
        }

        case "forget": {
          const ok = await graph.delete(a.memory_id as string);
          return { content: [{ type: "text", text: JSON.stringify({ deleted: ok }) }] };
        }

        case "get_conversation": {
          if (!transcriptAccessForRequest(headers)) {
            return {
              content: [
                {
                  type: "text",
                  text: JSON.stringify({
                    error:
                      "Transcript access is disabled. Run keymem under a host agent (Claude Code / Codex) or set KEYMEM_TRANSCRIPT_ACCESS=true.",
                  }),
                },
              ],
            };
          }
          const sessionId = a.session_id as string;
          const turn = typeof a.turn === "number" ? a.turn : null;
          const agent = a.agent === "claude" || a.agent === "codex" ? (a.agent as Agent) : null;
          let turns: object[];
          if (agent) {
            turns = await loadNativeConversation(agent, sessionId, turn);
          } else {
            // No agent hint: try the host agents' native transcripts, then fall
            // back to keymem's own conversation log.
            turns = await loadNativeAuto(sessionId, turn);
            if (turns.length === 0) turns = await loadConversation(sessionId, turn);
          }
          return { content: [{ type: "text", text: JSON.stringify(turns) }] };
        }

        case "list_sessions": {
          if (!transcriptAccessForRequest(headers)) {
            return {
              content: [
                {
                  type: "text",
                  text: JSON.stringify({
                    error:
                      "Transcript access is disabled. Run keymem under a host agent (Claude Code / Codex) or set KEYMEM_TRANSCRIPT_ACCESS=true.",
                  }),
                },
              ],
            };
          }
          const sessions = await listNativeSessions({
            agent: a.agent === "claude" || a.agent === "codex" ? (a.agent as Agent) : undefined,
            limit: typeof a.limit === "number" ? a.limit : undefined,
          });
          return { content: [{ type: "text", text: JSON.stringify(sessions) }] };
        }

        case "list_memories": {
          const results = graph.listAll(
            typeof a.namespace === "string" ? a.namespace : null
          );
          return { content: [{ type: "text", text: JSON.stringify(results) }] };
        }

        case "remember_batch": {
          const items = (parseArray(a.items) ?? []) as Array<Record<string, unknown>>;
          const hostLink = await resolveHostLink(headers); // detect once for the whole batch
          const results: object[] = [];
          for (const item of items) {
            const content = item.content as string;
            const keys = sanitizeKeys(item.keys);
            if (!content || keys.length === 0) {
              results.push({ error: "content and keys required", item });
              continue;
            }
            const [mid, wasDedup] = await graph.add(content, keys, {
              keyTypes: item.key_types as Record<string, string> | null,
              namespace: typeof item.namespace === "string" ? item.namespace : "default",
              ttlSeconds: parseNumber(item.ttl_seconds),
              decayProfile: parseDecayProfile(item.decay_profile),
              relatedTo: Array.isArray(item.related_to) ? (item.related_to as string[]) : null,
              source: buildSource((item.source as Record<string, unknown>) ?? null, "remember_batch", hostLink),
            });
            results.push({ saved: mid, deduplicated: wasDedup });
          }
          return { content: [{ type: "text", text: JSON.stringify(results) }] };
        }

        case "cleanup_expired": {
          const count = await graph.cleanupExpired();
          return { content: [{ type: "text", text: JSON.stringify({ deleted: count }) }] };
        }

        case "memory_stats": {
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({
                  keys: Object.keys(graph.keys).length,
                  memories: graph.listAll().length,
                  links: graph.linkCount,
                }),
              },
            ],
          };
        }

        default:
          return {
            content: [{ type: "text", text: JSON.stringify({ error: `Unknown tool: ${name}` }) }],
            isError: true,
          };
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return {
        content: [{ type: "text", text: JSON.stringify({ error: msg }) }],
        isError: true,
      };
    }
  });

  // ── Prompt definitions ──

  server.setRequestHandler(ListPromptsRequestSchema, async () => ({
    prompts: [
      {
        name: "memory_system_prompt",
        description:
          "System prompt for LLM agents using keymem. Include this in your system prompt.",
      },
    ],
  }));

  server.setRequestHandler(GetPromptRequestSchema, async (request) => {
    if (request.params.name !== "memory_system_prompt") {
      throw new Error(`Unknown prompt: ${request.params.name}`);
    }
    return {
      messages: [
        {
          role: "user",
          content: {
            type: "text",
            text: MEMORY_SYSTEM.replace("{stats}", stats()),
          },
        },
      ],
    };
  });

  return server;
}
