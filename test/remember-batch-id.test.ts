// remember_batch stamps a shared source.batch_id across its items (when there's more than
// one) so recall() can surface them to each other as batch_sibling_ids. A single-item batch
// gets no batch_id — there's no sibling to speak of, so it behaves like a plain remember.
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

let n = 0;

test("remember_batch stamps every item with the same source.batch_id", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-batchid-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const embedding = await import("../src/embedding.ts");
  let i = 0;
  embedding.__setTestEmbedder(() => {
    const v = new Array(8).fill(0);
    v[i++ % 8] = 1;
    return v;
  });
  t.after(() => embedding.__clearTestEmbedder());

  const { createMcpServer, graph } = await import(`../src/server.ts?batchid=${n++}`);
  const server = createMcpServer();
  const client = new Client({ name: "batchid-test", version: "0" });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  t.after(async () => { await client.close(); await server.close(); });
  await server.connect(serverTransport);
  await client.connect(clientTransport);

  const result = JSON.parse(textResult(await client.callTool({
    name: "remember_batch",
    arguments: {
      items: [
        { content: "batch item one", keys: ["batchkeyone"] },
        { content: "batch item two", keys: ["batchkeytwo"] },
      ],
    },
  })));
  const [id1, id2] = result.map((r: { saved: string }) => r.saved);
  assert.ok(id1 && id2);

  const source1 = graph.memories[id1].source as Record<string, unknown> | null;
  const source2 = graph.memories[id2].source as Record<string, unknown> | null;
  assert.ok(source1?.batch_id, "first item must carry a batch_id");
  assert.equal(source1?.batch_id, source2?.batch_id, "both items must share the same batch_id");
});

test("a single-item remember_batch call gets no batch_id", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-batchid-solo-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder(() => [1, 0, 0]);
  t.after(() => embedding.__clearTestEmbedder());

  const { createMcpServer, graph } = await import(`../src/server.ts?batchid-solo=${n++}`);
  const server = createMcpServer();
  const client = new Client({ name: "batchid-solo-test", version: "0" });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  t.after(async () => { await client.close(); await server.close(); });
  await server.connect(serverTransport);
  await client.connect(clientTransport);

  const result = JSON.parse(textResult(await client.callTool({
    name: "remember_batch",
    arguments: { items: [{ content: "solo batch item", keys: ["solokey"] }] },
  })));
  const [{ saved: id }] = result;

  const source = graph.memories[id].source as Record<string, unknown> | null;
  assert.ok(!source?.batch_id, "a single-item batch must not get a batch_id");
});
