import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

process.env.KEYMEM_DATA_DIR = await mkdtemp(join(tmpdir(), "keymem-e2e-"));
process.env.KEYMEM_TRANSCRIPT_ACCESS = "false"; // env 경로 차단, 헤더 경로만 검증

// 다른 통합 테스트(entry-literal-key.test.ts 등)와 동일한 관례: 결정론적 오프라인 임베더를
// 주입해 실제 OpenAI 호출(로컬 .env의 OPENAI_API_KEY 유효성)에 의존하지 않게 한다.
process.env.EMBEDDING_BACKEND = "local";
function vec(t: string): number[] {
  let h = 0;
  for (let i = 0; i < t.length; i++) h = (h * 31 + t.charCodeAt(i)) >>> 0;
  const v = new Array(64).fill(0);
  v[h % 64] = 1;
  return v;
}
const emb = await import("../src/embedding.ts");
emb.__setTestEmbedder((tx: string) => vec(tx));

const { startDaemon } = await import("../src/daemon.ts");

const SID_A = "aaaaaaaa-1602-4180-ac66-9f9acbd1f673";
const SID_B = "bbbbbbbb-1602-4180-ac66-9f9acbd1f673";

function client(port: number, sid: string) {
  const t = new StreamableHTTPClientTransport(new URL(`http://127.0.0.1:${port}/mcp`), {
    requestInit: { headers: { "X-Keymem-Host-Agent": "claude", "X-Keymem-Host-Session": sid } },
  });
  const c = new Client({ name: "test", version: "0" });
  return { c, connect: () => c.connect(t) };
}

type ToolTextResult = { content: Array<{ type: string; text: string }> };

function toolText(res: unknown): string {
  return (res as ToolTextResult).content[0].text;
}

// Navigates the same recall → read_key → read_memory path the memory_system_prompt
// documents, then returns the stored memory's `source` (which src/server.ts's
// buildSource() stamps with host_session from the request's X-Keymem-Host-Session
// header — see resolveHostLink/hostSessionFromHeaders). This proves the header→
// provenance path stamps the correct session per client on a *shared* daemon,
// not just that the fact is visible in the shared graph.
async function fetchStoredSource(
  c: Client,
  query: string
): Promise<{ host_session?: string } | undefined> {
  const recallRes = await c.callTool({ name: "recall", arguments: { query } });
  const keys = JSON.parse(toolText(recallRes)) as Array<{ key_id: string }>;
  assert.ok(keys.length > 0, `recall("${query}") returned at least one key`);
  const keyId = keys[0].key_id;

  const readKeyRes = await c.callTool({ name: "read_key", arguments: { key_id: keyId, query } });
  const keyPage = JSON.parse(toolText(readKeyRes)) as { memories: Array<{ memory_id: string }> };
  assert.ok(keyPage.memories.length > 0, `read_key(${keyId}) returned at least one memory`);
  const memoryId = keyPage.memories[0].memory_id;

  const readMemRes = await c.callTool({
    name: "read_memory",
    arguments: { memory_id: memoryId, via_key_id: keyId },
  });
  const memResult = JSON.parse(toolText(readMemRes)) as {
    memory: { source?: { host_session?: string } };
  };
  return memResult.memory.source;
}

test("two clients on one daemon persist to shared graph with correct provenance", async () => {
  // idleMs is intentionally LARGE (60s): src/daemon.ts's close() now sets a shutdown flag
  // before tearing down transports, so armIdle() no-ops during/after shutdown and no stray
  // timer is left pending — close() (and this test) resolve promptly regardless of idleMs.
  const d = await startDaemon({ port: 0, idleMs: 60_000 });
  try {
    const a = client(d.port, SID_A);
    const b = client(d.port, SID_B);
    await a.connect();
    await b.connect();

    await a.c.callTool({ name: "remember", arguments: { content: "fact from A", keys: ["alpha"] } });
    await b.c.callTool({ name: "remember", arguments: { content: "fact from B", keys: ["beta"] } });

    // 두 클라이언트가 같은 graph를 본다: A가 저장한 것을 B가 recall.
    const res = await b.c.callTool({ name: "recall", arguments: { query: "alpha" } });
    const text = toolText(res);
    assert.ok(text.length > 2, "recall returned keys from shared graph");

    // Per-client provenance: each client's own memory must be stamped with ITS OWN
    // host session (from its own request headers), not the other client's or the
    // shared daemon's ambient state. This is the header→provenance path that
    // resolveHostLink()/buildSource() in src/server.ts implement.
    const sourceA = await fetchStoredSource(a.c, "alpha");
    assert.equal(sourceA?.host_session, SID_A, "A's memory is stamped with A's host session");

    const sourceB = await fetchStoredSource(b.c, "beta");
    assert.equal(sourceB?.host_session, SID_B, "B's memory is stamped with B's host session");
  } finally {
    await d.close();
  }
});
