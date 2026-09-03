import type { ValidityView } from "./decay.js";

// Claude Code UserPromptSubmit hook client — the "push" half of keymem.
//
// recall() is pull: the agent must decide to ask. Human remembering is push: the
// context itself surfaces related memories. This hook inverts the trigger — the
// harness sends every user utterance here, the daemon's precision-gated inject
// path picks at most a couple of relevant memories, and they are added to the
// turn's context as an UNCONFIRMED hint the agent may inspect (read_memory) or ignore.
//
// Contract: NEVER block the prompt and NEVER wake the daemon.
//  - daemon down → ECONNREFUSED → exit 0 silently (the MCP shim owns daemon spawning)
//  - daemon cold / slow → abort at KEYMEM_HOOK_TIMEOUT_MS (default 800 ms) → exit 0
//  - any parse error → exit 0
// stdin:  Claude Code hook JSON ({ prompt, ... })
// stdout: { hookSpecificOutput: { hookEventName, additionalContext } } or nothing
const PORT = Number(process.env.KEYMEM_DAEMON_PORT ?? 8765);
const TIMEOUT_MS = Number(process.env.KEYMEM_HOOK_TIMEOUT_MS ?? 800);
const TOP_K = Number(process.env.KEYMEM_HOOK_TOP_K ?? 2);
const MAX_CHARS = 400; // per-memory preview budget in the injected context
// Acknowledgments ("ㄱㄱ", "ok") carry no recall cue — skip them and save the
// ~0.5 s round trip the hook adds to every prompt.
const MIN_CHARS = Number(process.env.KEYMEM_HOOK_MIN_CHARS ?? 6);

async function readStdin(): Promise<string> {
  let data = "";
  for await (const chunk of process.stdin) data += chunk;
  return data;
}

type InjectMemory = {
  id: string;
  content: string;
  relevance_score?: number;
  namespace?: string;
  validity?: ValidityView;
};

async function main(): Promise<void> {
  const input = JSON.parse(await readStdin()) as { prompt?: unknown; cwd?: unknown };
  const prompt = typeof input.prompt === "string" ? input.prompt.trim() : "";
  // Skip empties, acknowledgments, and slash commands — not recall cues.
  if (!prompt || prompt.length < MIN_CHARS || prompt.startsWith("/")) return;
  const cwd = typeof input.cwd === "string" ? input.cwd : undefined;

  const res = await fetch(`http://127.0.0.1:${PORT}/inject`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ prompt, top_k: TOP_K, cwd }),
    signal: AbortSignal.timeout(TIMEOUT_MS),
  });
  if (!res.ok) return;
  const { memories } = (await res.json()) as { memories?: InjectMemory[] };
  if (!memories || memories.length === 0) return;

  const lines = memories.map((m) => {
    const preview = m.content.length > MAX_CHARS ? `${m.content.slice(0, MAX_CHARS)}…` : m.content;
    const score = m.relevance_score !== undefined ? ` rel=${m.relevance_score}` : "";
    const validity = m.validity ? ` validity=${JSON.stringify(m.validity)}` : "";
    return `- [${m.id}${score}${m.namespace ? ` ns=${m.namespace}` : ""}${validity}] ${preview}`;
  });
  const additionalContext =
    `<keymem-surfaced>\n` +
    `Passively surfaced memories possibly related to this message (unconfirmed; ignore if irrelevant).\n` +
    `read_memory(id) retrieves full content and may reinforce its access/key path; it does not confirm that the content is current.\n` +
    `Use fresh memories normally and qualify aging memories when currentness matters. Never assert a stale memory as current; verify externally or ask the user.\n` +
    `${lines.join("\n")}\n</keymem-surfaced>`;

  const payload = JSON.stringify({
    hookSpecificOutput: { hookEventName: "UserPromptSubmit", additionalContext },
  });
  // Await the write callback: process.exit() below does NOT drain piped stdout, so
  // exiting before the callback races the flush and the hook output is silently lost.
  await new Promise<void>((resolve) => process.stdout.write(payload, () => resolve()));
}

main().catch(() => {}).finally(() => process.exit(0));
