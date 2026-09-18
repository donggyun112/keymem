import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";

// KEYMEM_HOOK_INSTALLED (set by the plugin's .mcp.json, unset on a manual install) picks
// which "first turn behavior" text memory_system_prompt serves: a hook install already
// surfaces memories passively, so the blind 3-parallel-recall mandate would be redundant.
async function firstTurnBehaviorText(t: any, hookInstalled: boolean): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "keymem-hook-prompt-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  if (hookInstalled) process.env.KEYMEM_HOOK_INSTALLED = "true";
  else delete process.env.KEYMEM_HOOK_INSTALLED;
  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder(() => [1, 0]);
  t.after(() => embedding.__clearTestEmbedder());

  const { createMcpServer } = await import(`../src/server.ts?hook-installed-${hookInstalled}`);
  const server = createMcpServer();
  const client = new Client({ name: "hook-prompt-test", version: "0" });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  t.after(async () => { await client.close(); await server.close(); });
  await server.connect(serverTransport);
  await client.connect(clientTransport);

  const prompt = await client.getPrompt({ name: "memory_system_prompt" });
  return prompt.messages.map((m) => (m.content.type === "text" ? m.content.text : "")).join("\n");
}

test("no KEYMEM_HOOK_INSTALLED: memory_system_prompt keeps the mandatory blind-recall block", async (t) => {
  const text = await firstTurnBehaviorText(t, false);
  assert.match(text, /MANDATORY: First turn behavior/);
  assert.match(text, /Run in parallel/);
});

test("KEYMEM_HOOK_INSTALLED=true: memory_system_prompt drops the blind-recall mandate and defers to the hook", async (t) => {
  const text = await firstTurnBehaviorText(t, true);
  assert.doesNotMatch(text, /MANDATORY: First turn behavior/);
  assert.doesNotMatch(text, /Run in parallel/);
  assert.match(text, /UserPromptSubmit/);
  assert.match(text, /keymem-surfaced/);
});
