// The remember/correct/remember_batch handlers stamp the active host-agent
// transcript link (host_session/host_agent/host_turn) onto a memory's source
// via buildSource, so a recalled memory can be traced back to its original
// conversation. Caller-provided source fields must still win.
import assert from "node:assert/strict";
import test from "node:test";

const { buildSource } = await import("../src/server.ts");

const HOST = { agent: "claude" as const, session_id: "e7f5b1d2-1602-4180-ac66-9f9acbd1f673", turn: 7 };

test("buildSource stamps the host transcript link when a session is active", () => {
  const source = buildSource(null, "remember", HOST);
  assert.equal(source.host_agent, "claude");
  assert.equal(source.host_session, "e7f5b1d2-1602-4180-ac66-9f9acbd1f673");
  assert.equal(source.host_turn, 7);
  assert.equal(source.tool, "remember");
});

test("buildSource omits host fields when no session is active", () => {
  const source = buildSource(null, "remember", null);
  assert.ok(!("host_session" in source));
  assert.ok(!("host_agent" in source));
});

test("caller-provided source overrides the auto-detected host link", () => {
  const source = buildSource(
    { host_session: "caller-supplied", conversation: "conv-9" },
    "remember",
    HOST
  );
  assert.equal(source.host_session, "caller-supplied"); // caller wins
  assert.equal(source.conversation, "conv-9");
  assert.equal(source.host_agent, "claude"); // un-overridden host field stays
});

const { resolveHostLink, transcriptAccessForRequest } = await import("../src/server.ts");

test("resolveHostLink builds link from headers without env", async () => {
  const link = await resolveHostLink({
    "x-keymem-host-agent": "claude",
    "x-keymem-host-session": "e7f5b1d2-1602-4180-ac66-9f9acbd1f673",
  });
  // 트랜스크립트 파일이 없으면 turn=0 로 폴백하되, 세션 정체는 헤더에서 확정된다.
  assert.equal(link?.agent, "claude");
  assert.equal(link?.session_id, "e7f5b1d2-1602-4180-ac66-9f9acbd1f673");
  assert.equal(typeof link?.turn, "number");
});

test("resolveHostLink returns null when no headers and env untrusted", async () => {
  const prev = process.env.KEYMEM_TRANSCRIPT_ACCESS;
  process.env.KEYMEM_TRANSCRIPT_ACCESS = "false";
  try {
    assert.equal(await resolveHostLink(undefined), null);
  } finally {
    if (prev === undefined) delete process.env.KEYMEM_TRANSCRIPT_ACCESS;
    else process.env.KEYMEM_TRANSCRIPT_ACCESS = prev;
  }
});

test("transcriptAccessForRequest is true when host headers present", () => {
  assert.equal(
    transcriptAccessForRequest({
      "x-keymem-host-agent": "codex",
      "x-keymem-host-session": "e7f5b1d2-1602-4180-ac66-9f9acbd1f673",
    }),
    true
  );
});
