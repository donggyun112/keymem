import assert from "node:assert/strict";
import test from "node:test";

const { hostSessionFromHeaders } = await import("../src/nativeTranscripts.ts");

const UUID = "e7f5b1d2-1602-4180-ac66-9f9acbd1f673";

test("parses agent + session from X-Keymem headers (lowercased by transport)", () => {
  const got = hostSessionFromHeaders({
    "x-keymem-host-agent": "claude",
    "x-keymem-host-session": UUID,
  });
  assert.deepEqual(got, { agent: "claude", session_id: UUID });
});

test("accepts codex agent", () => {
  const got = hostSessionFromHeaders({ "x-keymem-host-agent": "codex", "x-keymem-host-session": UUID });
  assert.equal(got?.agent, "codex");
});

test("returns null when headers absent", () => {
  assert.equal(hostSessionFromHeaders(undefined), null);
  assert.equal(hostSessionFromHeaders({}), null);
});

test("rejects unknown agent", () => {
  assert.equal(hostSessionFromHeaders({ "x-keymem-host-agent": "evil", "x-keymem-host-session": UUID }), null);
});

test("rejects non-UUID session", () => {
  assert.equal(hostSessionFromHeaders({ "x-keymem-host-agent": "claude", "x-keymem-host-session": "not-a-uuid" }), null);
});

test("takes first value when header is an array", () => {
  const got = hostSessionFromHeaders({ "x-keymem-host-agent": ["claude"], "x-keymem-host-session": [UUID] });
  assert.deepEqual(got, { agent: "claude", session_id: UUID });
});
