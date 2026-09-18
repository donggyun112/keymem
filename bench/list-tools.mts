import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
const dir = await mkdtemp(join(tmpdir(), "km-list-tools-"));
process.env.KEYMEM_DATA_DIR = dir;
process.env.EMBEDDING_BACKEND = "local";
process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
const { createMcpServer } = await import("../src/server.ts");
const { Client } = await import("@modelcontextprotocol/sdk/client/index.js");
const { InMemoryTransport } = await import("@modelcontextprotocol/sdk/inMemory.js");
const server = createMcpServer();
const client = new Client({ name: "list-tools", version: "0" });
const [ct, st] = InMemoryTransport.createLinkedPair();
await server.connect(st);
await client.connect(ct);
const { tools } = await client.listTools();
console.log(`total tools: ${tools.length}\n`);
for (const t of tools) {
  const props = Object.keys(t.inputSchema?.properties ?? {});
  console.log(`${t.name} (${props.length} params): ${props.join(", ")}`);
}
await client.close();
await server.close();
await rm(dir, { recursive: true, force: true });
