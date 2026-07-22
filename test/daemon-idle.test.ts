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

test("abrupt connection drop (no graceful DELETE) is reaped so idle-exit can proceed", async () => {
  // Simulates a shim being SIGKILL'd: the client transport's connection is aborted
  // directly, without ever sending an MCP DELETE (this mirrors what Client.close() /
  // StreamableHTTPClientTransport.close() actually do under the hood -- they only
  // abort the local AbortController; they do NOT send a session-termination DELETE).
  // Before the fix, the daemon's server-side transport.onclose only fires from an
  // explicit transport.close() call (DELETE handling or daemon shutdown) -- never from
  // the SDK's standalone-SSE ReadableStream `cancel()` callback that fires when the
  // GET stream's socket merely drops. So the session would live in `transports` forever.
  const d = await startDaemon({ port: 0, idleMs: 60_000 });
  try {
    const clientTransport = new StreamableHTTPClientTransport(new URL(`http://127.0.0.1:${d.port}/mcp`));
    const client = new Client({ name: "test", version: "0" });
    await client.connect(clientTransport);
    assert.equal(d.sessionCount(), 1, "session should be registered after connect");

    // The client opens its standalone GET SSE stream fire-and-forget, right after
    // connect() resolves (triggered by the notifications/initialized POST, without
    // being awaited by connect() itself -- see StreamableHTTPClientTransport.send()).
    // Give it a moment to actually reach the server before we simulate the drop,
    // otherwise we'd abort the fetch before it ever left the client and the daemon
    // would never see a GET /mcp request (and thus never attach the reap hook this
    // test is meant to exercise).
    await new Promise((r) => setTimeout(r, 300));

    // Abrupt drop: abort the transport directly (no DELETE sent).
    await clientTransport.close();

    // The reap happens asynchronously off the server's res 'close' event; poll briefly.
    const deadline = Date.now() + 2_000;
    while (d.sessionCount() > 0 && Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 20));
    }
    assert.equal(d.sessionCount(), 0, "phantom session should be reaped after the connection drops");
  } finally {
    await d.close();
  }
});

test("quiet-but-alive session (open GET SSE stream, no requests) is never reaped", async () => {
  // The live-proxy invariant: a shim that is alive but idle between tool calls keeps its
  // standalone GET SSE stream open. That connection must never be treated as dead just
  // because no /mcp requests have arrived recently -- only an actual socket drop may
  // reap a session. This guards against a naive "N minutes since last request" watchdog,
  // which would incorrectly kill this session.
  const d = await startDaemon({ port: 0, idleMs: 60_000 });
  const clientTransport = new StreamableHTTPClientTransport(new URL(`http://127.0.0.1:${d.port}/mcp`));
  const client = new Client({ name: "test", version: "0" });
  try {
    await client.connect(clientTransport);
    assert.equal(d.sessionCount(), 1, "session should be registered after connect");

    // Sit idle for a window comfortably longer than the reap check in the previous test,
    // without sending any further /mcp requests and without closing the connection.
    await new Promise((r) => setTimeout(r, 500));

    assert.equal(d.sessionCount(), 1, "quiet-but-alive session must not be reaped");
  } finally {
    await client.close();
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
