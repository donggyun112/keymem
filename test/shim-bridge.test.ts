import assert from "node:assert/strict";
import test from "node:test";

const { hostHeaders, ensureDaemon, daemonEnv } = await import("../src/shim.ts");

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

test("daemonEnv strips CLAUDE_CODE_SESSION_ID and CODEX_THREAD_ID", () => {
  const env = {
    CLAUDE_CODE_SESSION_ID: UUID,
    CODEX_THREAD_ID: UUID,
    KEYMEM_TRANSCRIPT_ACCESS: "true",
    PATH: "/usr/bin",
  } as NodeJS.ProcessEnv;
  const result = daemonEnv(env);
  assert.equal("CLAUDE_CODE_SESSION_ID" in result, false);
  assert.equal("CODEX_THREAD_ID" in result, false);
  assert.equal(result.KEYMEM_TRANSCRIPT_ACCESS, "true");
  assert.equal(result.PATH, "/usr/bin");
});

test("daemonEnv does not mutate the input env object", () => {
  const env = {
    CLAUDE_CODE_SESSION_ID: UUID,
    CODEX_THREAD_ID: UUID,
  } as NodeJS.ProcessEnv;
  daemonEnv(env);
  assert.equal(env.CLAUDE_CODE_SESSION_ID, UUID);
  assert.equal(env.CODEX_THREAD_ID, UUID);
});
