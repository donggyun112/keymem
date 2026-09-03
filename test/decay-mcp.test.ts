import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";

function textResult(result: any): string {
  const content = result.content?.find((item: any) => item.type === "text");
  assert.ok(content && typeof content.text === "string");
  return content.text;
}

test("MCP exposes confirmation and stale-memory guidance", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-decay-mcp-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder(() => [1, 0]);
  t.after(() => embedding.__clearTestEmbedder());

  const { createMcpServer } = await import("../src/server.ts?decay-mcp");
  const server = createMcpServer();
  const client = new Client({ name: "decay-test", version: "0" });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  t.after(async () => { await client.close(); await server.close(); });
  await server.connect(serverTransport);
  await client.connect(clientTransport);

  const listed = await client.listTools();
  const confirm = listed.tools.find((tool) => tool.name === "confirm_memory");
  assert.ok(confirm);
  assert.deepEqual(confirm.inputSchema.required, ["memory_id", "evidence"]);
  assert.match(confirm.description ?? "", /never.*merely.*read_memory/i);

  const remember = listed.tools.find((tool) => tool.name === "remember")!;
  assert.ok("decay_profile" in (remember.inputSchema.properties ?? {}));
  const rememberBatch = listed.tools.find((tool) => tool.name === "remember_batch")!;
  const batchItems = rememberBatch.inputSchema.properties?.items as any;
  assert.ok("decay_profile" in (batchItems?.items?.properties ?? {}));
  const correct = listed.tools.find((tool) => tool.name === "correct")!;
  assert.ok("ttl_seconds" in (correct.inputSchema.properties ?? {}));
  assert.ok("decay_profile" in (correct.inputSchema.properties ?? {}));

  const prompt = await client.getPrompt({ name: "memory_system_prompt" });
  const text = prompt.messages.map((m) => m.content.type === "text" ? m.content.text : "").join("\n");
  assert.match(text, /aging/);
  assert.match(text, /stale/);
  assert.match(text, /confirm_memory/);
  assert.match(text, /do not.*confirm.*merely.*read/i);
  const instructions = client.getInstructions() ?? "";
  assert.match(instructions, /fresh/);
  assert.match(instructions, /aging/);
  assert.match(instructions, /stale/);
  assert.match(instructions, /confirm_memory/);
  assert.match(instructions, /do not.*confirm.*merely.*read/i);

  const remembered = JSON.parse(textResult(await client.callTool({
    name: "remember",
    arguments: { content: "temporary fact", keys: ["temporary"], decay_profile: "transient" },
  })));
  const confirmed = JSON.parse(textResult(await client.callTool({
    name: "confirm_memory",
    arguments: { memory_id: remembered.saved, evidence: "user", source: { reason: "current assertion" } },
  })));
  assert.equal(confirmed.confirmed, true);
  assert.equal(confirmed.validity.status, "fresh");
  assert.equal(confirmed.validity.decay_profile, "transient");

  const invalidEvidence = await client.callTool({
    name: "confirm_memory",
    arguments: { memory_id: remembered.saved, evidence: "hearsay" },
  });
  assert.equal(invalidEvidence.isError, true);
  assert.match(textResult(invalidEvidence), /Unknown confirmation evidence/);

  const invalid = await client.callTool({
    name: "remember",
    arguments: { content: "bad profile", keys: ["profile"], decay_profile: "volatile" },
  });
  assert.equal(invalid.isError, true);
  assert.match(textResult(invalid), /Unknown decay profile/);
});
