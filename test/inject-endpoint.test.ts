import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

process.env.KEYMEM_DATA_DIR = await mkdtemp(join(tmpdir(), "keymem-inject-"));
process.env.EMBEDDING_BACKEND = "local";

// Deterministic embedder: the fact aligns with the prompt; noise is orthogonal.
function vec(t: string): number[] {
  if (t.includes("아메리카노") || t.includes("음료")) return [1, 0, 0];
  return [0, 0, 1];
}
const emb = await import("../src/embedding.ts");
emb.__setTestEmbedder((tx: string) => vec(tx));

const { graph } = await import("../src/server.ts");
const { startDaemon } = await import("../src/daemon.ts");

test("POST /inject returns passively-relevant memories without an MCP session", async () => {
  const daemon = await startDaemon({ port: 0, idleMs: 60_000 });
  try {
    await graph.add("사용자는 아이스 아메리카노를 즐긴다", ["음료"], { namespace: "default" });
    await graph.add("전혀 무관한 잡음 데이터", ["잡음"], { namespace: "default" });

    const res = await fetch(`http://127.0.0.1:${daemon.port}/inject`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ prompt: "오늘도 음료는 아메리카노 마실까", top_k: 2 }),
    });
    assert.equal(res.status, 200);
    const body = (await res.json()) as { memories: Array<{ content: string }> };
    assert.ok(body.memories.length >= 1);
    assert.match(body.memories[0].content, /아메리카노/);
    assert.ok(!body.memories.some((m) => m.content.includes("잡음")));

    // malformed request → 400, never a crash
    const bad = await fetch(`http://127.0.0.1:${daemon.port}/inject`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({}),
    });
    assert.equal(bad.status, 400);
  } finally {
    await daemon.close();
  }
});
