// Claude Code UserPromptSubmit hook client — the "push" half of keymem.
//
// recall() is pull: the agent must decide to ask. Human remembering is push: the
// context itself surfaces related memories. This hook inverts the trigger — the
// harness sends every user utterance here, the daemon's precision-gated inject
// path picks at most a couple of relevant memories, and they are added to the
// turn's context as an UNCONFIRMED hint the agent may verify (read_memory) or ignore.
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
};

async function main(): Promise<void> {
  const input = JSON.parse(await readStdin()) as { prompt?: unknown };
  const prompt = typeof input.prompt === "string" ? input.prompt.trim() : "";
  // Skip empties and slash commands — they are harness directives, not recall cues.
  if (!prompt || prompt.startsWith("/")) return;

  const res = await fetch(`http://127.0.0.1:${PORT}/inject`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ prompt, top_k: TOP_K }),
    signal: AbortSignal.timeout(TIMEOUT_MS),
  });
  if (!res.ok) return;
  const { memories } = (await res.json()) as { memories?: InjectMemory[] };
  if (!memories || memories.length === 0) return;

  const lines = memories.map((m) => {
    const preview = m.content.length > MAX_CHARS ? `${m.content.slice(0, MAX_CHARS)}…` : m.content;
    const score = m.relevance_score !== undefined ? ` rel=${m.relevance_score}` : "";
    return `- [${m.id}${score}${m.namespace ? ` ns=${m.namespace}` : ""}] ${preview}`;
  });
  const additionalContext =
    `<keymem-surfaced>\n` +
    `Passively surfaced memories possibly related to this message (unconfirmed — ` +
    `verify with read_memory(id) before asserting; ignore if irrelevant):\n` +
    `${lines.join("\n")}\n</keymem-surfaced>`;

  process.stdout.write(
    JSON.stringify({
      hookSpecificOutput: { hookEventName: "UserPromptSubmit", additionalContext },
    })
  );
}

main().catch(() => {}).finally(() => process.exit(0));
