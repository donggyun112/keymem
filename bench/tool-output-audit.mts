import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

process.env.EMBEDDING_BACKEND ??= "local";
process.env.LOCAL_EMBEDDING_MODEL ??= "bge-m3";
const dir = await mkdtemp(join(tmpdir(), "km-tool-audit-"));
process.env.KEYMEM_DATA_DIR = dir;

const { createMcpServer, graph } = await import("../src/server.ts");
const { Client } = await import("@modelcontextprotocol/sdk/client/index.js");
const { InMemoryTransport } = await import("@modelcontextprotocol/sdk/inMemory.js");
const server = createMcpServer();
const client = new Client({ name: "tool-audit", version: "0" });
const [ct, st] = InMemoryTransport.createLinkedPair();
await server.connect(st);
await client.connect(ct);

function text(result: any): string {
  return result.content?.find((c: any) => c.type === "text")?.text ?? "";
}
function dump(label: string, raw: string) {
  console.log(`\n=== ${label} ===`);
  console.log(JSON.stringify(JSON.parse(raw), null, 2));
}

const [coffeeId] = await graph.add("미나는 매일 오후 아이스 라떼를 마신다", ["커피", "음료"]);

dump("recall", text(await client.callTool({ name: "recall", arguments: { query: "커피", context: "미나가 마시는 커피 얘기해줘" } })));

const recallJson = JSON.parse(text(await client.callTool({ name: "recall", arguments: { query: "커피" } })));
const keyId = recallJson.keys[0].key_id;

dump("read_key", text(await client.callTool({ name: "read_key", arguments: { key_id: keyId, query: "커피" } })));

const readKeyJson = JSON.parse(text(await client.callTool({ name: "read_key", arguments: { key_id: keyId, query: "커피" } })));
const memoryId = readKeyJson.memories[0].memory_id;

dump("read_memory", text(await client.callTool({ name: "read_memory", arguments: { memory_id: memoryId, via_key_id: keyId } })));

dump("remember", text(await client.callTool({ name: "remember", arguments: { content: "미나는 산책을 좋아한다", keys: ["산책"] } })));

dump("confirm_memory", text(await client.callTool({ name: "confirm_memory", arguments: { memory_id: coffeeId, evidence: "user" } })));

dump("browse_keys", text(await client.callTool({ name: "browse_keys", arguments: { namespace: "default" } })));

await client.close();
await server.close();
await rm(dir, { recursive: true, force: true });
