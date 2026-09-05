import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";

let moduleId = 0;

function vec(text: string): number[] {
  const normalized = text.toLowerCase();
  if (normalized.includes("coffee") || normalized.includes("drinks")) return [1, 0, 0];
  if (normalized.includes("profile")) return [0, 1, 0];
  if (normalized.includes("tea")) return [0.8, 0.6, 0];
  return [0, 0, 1];
}

function textResult(result: any): string {
  const content = result.content?.find((item: any) => item.type === "text");
  assert.ok(content && typeof content.text === "string");
  return content.text;
}

test("directHydrateTop1 selects the top memory under the supplied key without reinforcement", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-direct-recall-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vec(text));
  t.after(() => embedding.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?direct-recall=${moduleId++}`);
  const graph = new MemoryGraph();
  await graph.load();
  const [teaId] = await graph.add("the user usually drinks tea", ["drinks"], {});
  const [coffeeId] = await graph.add("the user prefers coffee", ["drinks", "profile"], {});

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
  assert.deepEqual(
    decision.candidate?.memory.connected_keys
      .map((key) => key.concept)
      .sort(),
    ["drinks", "profile"],
  );
  assert.ok(
    decision.candidate?.memory.connected_keys.every((key) => key.key_id.length > 0),
  );
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

test("directHydrateTop1 lets the cross-encoder choose from the top-key candidate pool", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-direct-rerank-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => {
    if (text.includes("QUERY") || text.includes("WRONG")) return [1, 0, 0];
    if (text.includes("RIGHT")) return [0, 1, 0];
    return [0, 0, 1];
  });
  t.after(() => embedding.__clearTestEmbedder());
  const reranker = await import("../src/reranker.ts");
  reranker.__setTestReranker((_query, texts) =>
    texts.map((text) => text.includes("RIGHT") ? 10 : 0),
  );
  t.after(() => reranker.__clearTestReranker());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?direct-rerank=${moduleId++}`);
  const graph = new MemoryGraph();
  await graph.load();
  await graph.add("WRONG candidate", ["topic"], {});
  const [rightId] = await graph.add("RIGHT candidate", ["topic"], {});
  const [topKey] = await graph.searchKeys("topic", 8, null, "QUERY");
  const page = await graph.readKey(topKey.key_id, { query: "QUERY", limit: 30 }) as {
    memories: Array<{ memory_id: string }>;
  };
  assert.equal(reranker.rerankEnabled(), true);
  assert.notEqual(page.memories[0].memory_id, rightId);

  const decision = await graph.directHydrateTop1(topKey, "QUERY", null);

  assert.equal(decision.status, "candidate");
  assert.equal(decision.candidate?.memory.id, rightId);
  assert.equal(decision.candidate?.memory.content, "RIGHT candidate");
});

test("recall returns the passive top-1 memory by default and no memory on a miss", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-direct-recall-mcp-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vec(text));
  t.after(() => embedding.__clearTestEmbedder());

  const { createMcpServer, graph } = await import(`../src/server.ts?direct-recall=${moduleId++}`);
  const [coffeeId] = await graph.add("the user prefers coffee", ["drinks", "profile"], {});
  const server = createMcpServer();
  const client = new Client({ name: "direct-recall-test", version: "0" });
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
  assert.equal(recallResult.status, "found");
  assert.equal(recallResult.query, "drinks");
  assert.equal(recallResult.namespace, null);
  assert.ok(Array.isArray(recallResult.keys));
  assert.ok(recallResult.keys.length > 0);
  assert.deepEqual(Object.keys(recallResult.keys[0]), [
    "key_id",
    "concept",
    "aliases",
    "key_type",
    "score",
    "match_type",
    "memory_count",
    "is_hub",
    "specificity",
  ]);
  assert.equal("score_kind" in recallResult.keys[0], false);
  assert.equal("cluster_size" in recallResult.keys[0], false);
  assert.equal("evidence" in recallResult.keys[0], false);
  assert.equal("suggested_tool" in recallResult.keys[0], false);
  assert.equal(recallResult.memories.length, 1);
  assert.equal(recallResult.memories[0].id, coffeeId);
  assert.equal(recallResult.memories[0].content, "the user prefers coffee");
  assert.equal(recallResult.memories[0].evidence, "passive_preview");
  assert.equal(recallResult.memories[0].reinforced, false);
  assert.equal(recallResult.memories[0].matched_key.key_id, recallResult.keys[0].key_id);
  assert.deepEqual(
    recallResult.memories[0].connected_keys.map((key: { concept: string }) => key.concept).sort(),
    ["drinks", "profile"],
  );
  assert.ok(
    recallResult.memories[0].connected_keys.every(
      (key: { key_id: string }) => key.key_id.length > 0,
    ),
  );
  assert.ok(recallResult.memories[0].validity);
  assert.deepEqual({
    access: graph.memories[coffeeId].access_count,
    depth: graph.memories[coffeeId].depth,
  }, before);

  const miss = JSON.parse(textResult(await client.callTool({
    name: "recall",
    arguments: { query: "astronomy", context: "tell me about distant galaxies" },
  })));
  assert.equal(miss.status, "no_match");
  assert.deepEqual(miss.keys, []);
  assert.deepEqual(miss.memories, []);

  // The test embedder is lexical, so add the long memory only after the miss check above.
  const [novelId] = await graph.add(`the user is writing a novel. ${"plot detail. ".repeat(40)}`, ["novel"], {});
  const clipped = JSON.parse(textResult(await client.callTool({
    name: "recall",
    arguments: { query: "novel", context: "what is the user writing", inject_max_chars: 256 },
  })));
  assert.equal(clipped.memories[0].id, novelId);
  assert.equal(clipped.memories[0].content_truncated, true);
  assert.ok(clipped.memories[0].content.length <= 256, "preview must respect inject_max_chars");
  assert.ok(clipped.memories[0].content.endsWith("[truncated; use read_memory for full content]"));
});
