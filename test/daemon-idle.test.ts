import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

// graph는 KEYMEM_DATA_DIR로 격리 (env.ts가 존중)
process.env.KEYMEM_DATA_DIR = await mkdtemp(join(tmpdir(), "keymem-daemon-"));

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
