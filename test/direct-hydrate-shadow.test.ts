import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";

let moduleId = 0;

function vec(text: string): number[] {
  const normalized = text.toLowerCase();
  if (normalized.includes("coffee") || normalized.includes("drinks")) return [1, 0, 0];
  if (normalized.includes("tea")) return [0.8, 0.6, 0];
  return [0, 0, 1];
}

function textResult(result: any): string {
  const content = result.content?.find((item: any) => item.type === "text");
  assert.ok(content && typeof content.text === "string");
  return content.text;
}

test("directHydrateTop1 selects the top memory under the supplied key without reinforcement", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-direct-shadow-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vec(text));
  t.after(() => embedding.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?direct-shadow=${moduleId++}`);
  const graph = new MemoryGraph();
  await graph.load();
  const [teaId] = await graph.add("the user usually drinks tea", ["drinks"], {});
  const [coffeeId] = await graph.add("the user prefers coffee", ["drinks"], {});

  const keys = await graph.searchKeys("drinks", 8, null, "the user wants coffee");
  const topKey = keys[0] as { key_id: string; concept: string; score: number; match_type: string };
  const beforePage = await graph.readKey(topKey.key_id, {
    query: "the user wants coffee",
    limit: 2,
  }) as { memories: Array<{ memory_id: string; link_weight: number }> };
  const beforeWeights = Object.fromEntries(beforePage.memories.map((memory) => [memory.memory_id, memory.link_weight]));
  const beforeState = {
    coffeeAccess: graph.memories[coffeeId].access_count,
    coffeeDepth: graph.memories[coffeeId].depth,
    teaAccess: graph.memories[teaId].access_count,
    teaDepth: graph.memories[teaId].depth,
  };

  const decision = await graph.directHydrateTop1(topKey, "the user wants coffee", null);

  assert.equal(decision.status, "candidate");
  assert.equal(decision.candidate?.key.key_id, topKey.key_id);
  assert.equal(decision.candidate?.memory.id, coffeeId);
  assert.equal(decision.candidate?.memory.content, "the user prefers coffee");
  assert.equal(decision.candidate?.memory.evidence, "passive_preview");
  assert.equal(decision.candidate?.memory.reinforced, false);
  assert.deepEqual({
    coffeeAccess: graph.memories[coffeeId].access_count,
    coffeeDepth: graph.memories[coffeeId].depth,
    teaAccess: graph.memories[teaId].access_count,
    teaDepth: graph.memories[teaId].depth,
  }, beforeState);

  const afterPage = await graph.readKey(topKey.key_id, {
    query: "the user wants coffee",
    limit: 2,
  }) as { memories: Array<{ memory_id: string; link_weight: number }> };
  assert.deepEqual(
    Object.fromEntries(afterPage.memories.map((memory) => [memory.memory_id, memory.link_weight])),
    beforeWeights,
  );
});

test("shadow mode records candidate and no-key decisions without changing the recall response", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-direct-shadow-mcp-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.KEYMEM_DIRECT_HYDRATE_SHADOW = "true";
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  t.after(() => { delete process.env.KEYMEM_DIRECT_HYDRATE_SHADOW; });

  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vec(text));
  t.after(() => embedding.__clearTestEmbedder());

  const { createMcpServer, graph } = await import(`../src/server.ts?direct-shadow=${moduleId++}`);
  const [coffeeId] = await graph.add("the user prefers coffee", ["drinks"], {});
  const server = createMcpServer();
  const client = new Client({ name: "direct-shadow-test", version: "0" });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  t.after(async () => { await client.close(); await server.close(); });
  await server.connect(serverTransport);
  await client.connect(clientTransport);

  const before = {
    access: graph.memories[coffeeId].access_count,
    depth: graph.memories[coffeeId].depth,
  };
  const recallResult = JSON.parse(textResult(await client.callTool({
    name: "recall",
    arguments: { query: "drinks", context: "the user wants coffee" },
  })));
  assert.ok(Array.isArray(recallResult));
  assert.ok(recallResult.length > 0);
  assert.equal("memory" in recallResult[0], false);
  assert.equal("candidate" in recallResult[0], false);

  const logPath = join(dir, "direct-hydrate-shadow.jsonl");
  let events = (await readFile(logPath, "utf8")).trim().split("\n").map((line) => JSON.parse(line));
  assert.equal(events.length, 1);
  assert.equal(events[0].schema_version, 1);
  assert.equal(events[0].query, "drinks");
  assert.equal(events[0].context, "the user wants coffee");
  assert.equal(events[0].namespace, null);
  assert.equal(events[0].decision.status, "candidate");
  assert.equal(events[0].decision.candidate.memory.id, coffeeId);
  assert.equal(events[0].decision.candidate.memory.content, "the user prefers coffee");
  assert.equal(events[0].decision.candidate.memory.reinforced, false);
  assert.deepEqual({
    access: graph.memories[coffeeId].access_count,
    depth: graph.memories[coffeeId].depth,
  }, before);

  await client.callTool({
    name: "recall",
    arguments: { query: "astronomy", context: "tell me about distant galaxies" },
  });
  events = (await readFile(logPath, "utf8")).trim().split("\n").map((line) => JSON.parse(line));
  assert.equal(events.length, 2);
  assert.equal(events[1].query, "astronomy");
  assert.equal(events[1].decision.status, "no_key");
  assert.equal(events[1].decision.candidate, null);
});
