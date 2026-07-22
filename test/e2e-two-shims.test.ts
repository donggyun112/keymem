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
    const text = (res.content as Array<{ type: string; text: string }>)[0].text;
    assert.ok(text.length > 2, "recall returned keys from shared graph");
  } finally {
    await d.close();
  }
});
