import assert from "node:assert/strict";
import test from "node:test";

const { hostHeaders, ensureDaemon } = await import("../src/shim.ts");

const UUID = "e7f5b1d2-1602-4180-ac66-9f9acbd1f673";

test("hostHeaders emits claude session header from env", () => {
  const h = hostHeaders({ CLAUDE_CODE_SESSION_ID: UUID } as NodeJS.ProcessEnv);
  assert.equal(h["X-Keymem-Host-Agent"], "claude");
  assert.equal(h["X-Keymem-Host-Session"], UUID);
});

test("hostHeaders emits codex session header from env", () => {
  const h = hostHeaders({ CODEX_THREAD_ID: UUID } as NodeJS.ProcessEnv);
  assert.equal(h["X-Keymem-Host-Agent"], "codex");
  assert.equal(h["X-Keymem-Host-Session"], UUID);
});

test("hostHeaders is empty when no session env", () => {
  assert.deepEqual(hostHeaders({} as NodeJS.ProcessEnv), {});
});

test("ensureDaemon returns false quickly when nothing to connect and no spawn", async () => {
  // Unused port, autostart off -> fallback signal
  const ok = await ensureDaemon("http://127.0.0.1:59999/mcp", { timeoutMs: 300, spawnDaemon: false });
  assert.equal(ok, false);
});
