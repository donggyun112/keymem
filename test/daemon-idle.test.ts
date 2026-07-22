import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

// graph는 KEYMEM_DATA_DIR로 격리 (env.ts가 존중)
process.env.KEYMEM_DATA_DIR = await mkdtemp(join(tmpdir(), "keymem-daemon-"));
process.env.KEYMEM_TRANSCRIPT_ACCESS = "false";

const { startDaemon } = await import("../src/daemon.ts");

test("health endpoint returns 200 after start", async () => {
  const d = await startDaemon({ port: 0, idleMs: 60_000 });
  try {
    const res = await fetch(`http://127.0.0.1:${d.port}/health`);
    assert.equal(res.status, 200);
    const body = await res.json();
    assert.equal(body.ok, true);
  } finally {
    await d.close();
  }
});

test("unknown path returns 404", async () => {
  const d = await startDaemon({ port: 0, idleMs: 60_000 });
  try {
    const res = await fetch(`http://127.0.0.1:${d.port}/nope`);
    assert.equal(res.status, 404);
  } finally {
    await d.close();
  }
});

test("stale/unknown mcp-session-id on a non-initialize request is rejected, not treated as a new session", async () => {
  const d = await startDaemon({ port: 0, idleMs: 60_000 });
  try {
    const res = await fetch(`http://127.0.0.1:${d.port}/mcp`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "mcp-session-id": "00000000-0000-0000-0000-000000000000",
      },
      body: JSON.stringify({ jsonrpc: "2.0", method: "tools/list", id: 1 }),
    });
    assert.equal(res.status, 400);

    // 데몬은 여전히 살아있고 상태(세션/유휴 타이머)가 오염되지 않아야 한다.
    const health = await fetch(`http://127.0.0.1:${d.port}/health`);
    assert.equal(health.status, 200);
  } finally {
    await d.close();
  }
});

test("close() is prompt after a real session lifecycle, even with a large idleMs", async () => {
  // idleMs is intentionally LARGE (60s): if close() left a fresh idle timer armed (the bug
  // this test guards against), the test process would hang for the full 60s waiting for that
  // stray setTimeout(() => process.exit(0), idleMs) to keep the event loop alive. A prompt
  // return here is only possible because armIdle() no-ops during/after shutdown.
  const d = await startDaemon({ port: 0, idleMs: 60_000 });
  try {
    const transport = new StreamableHTTPClientTransport(new URL(`http://127.0.0.1:${d.port}/mcp`));
    const client = new Client({ name: "test", version: "0" });
    // Connecting performs the initialize handshake, which creates a real server-side
    // transport and registers it in `transports`.
    await client.connect(transport);
    // Closing the client closes the transport, which fires the server transport's onclose
    // handler → armIdle(). Before the fix, this re-armed a fresh idle timer.
    await client.close();
  } finally {
    const start = Date.now();
    await d.close();
    const elapsed = Date.now() - start;
    // Generous upper bound well under idleMs (60_000) to prove no stray timer is keeping
    // close() (or the event loop) alive.
    assert.ok(elapsed < 5_000, `close() took ${elapsed}ms, expected it to resolve promptly`);
  }
});
