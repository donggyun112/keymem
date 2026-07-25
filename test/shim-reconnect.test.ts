import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const dataDir = await mkdtemp(join(tmpdir(), "keymem-shim-reconnect-"));
process.env.KEYMEM_DATA_DIR = dataDir;
process.env.EMBEDDING_BACKEND = "local";
process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

const emb = await import("../src/embedding.ts");
emb.__setTestEmbedder(() => [1, 0, 0, 0]);
const { startDaemon } = await import("../src/daemon.ts");

type ToolTextResult = { content: Array<{ type: string; text: string }> };

function toolText(result: unknown): string {
  return (result as ToolTextResult).content[0].text;
}

async function within<T>(promise: Promise<T>, timeoutMs: number, diagnostics: () => string): Promise<T> {
  let timer: NodeJS.Timeout | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        timer = setTimeout(
          () => reject(new Error(`timed out after ${timeoutMs}ms\n${diagnostics()}`)),
          timeoutMs
        );
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

test("same stdio client reconnects after daemon restart and sees the same global graph", {
  timeout: 20_000,
}, async () => {
  let daemon = await startDaemon({ port: 0, idleMs: 60_000 });
  const port = daemon.port;
  const shim = new StdioClientTransport({
    command: join(process.cwd(), "node_modules", ".bin", "tsx"),
    args: ["src/shim.ts"],
    cwd: process.cwd(),
    stderr: "pipe",
    env: {
      KEYMEM_DAEMON_PORT: String(port),
      KEYMEM_DAEMON_AUTOSTART: "false",
      KEYMEM_DATA_DIR: dataDir,
      KEYMEM_TRANSCRIPT_ACCESS: "false",
      EMBEDDING_BACKEND: "local",
      LOCAL_EMBEDDING_MODEL: "bge-m3",
    },
  });
  let stderr = "";
  shim.stderr?.on("data", (chunk) => {
    stderr += String(chunk);
  });
  const client = new Client({ name: "shim-reconnect-test", version: "0" });

  try {
    await client.connect(shim);
    const shimPid = shim.pid;
    assert.ok(shimPid, "shim process started");

    await client.callTool({
      name: "remember",
      arguments: {
        content: "global memory survives daemon restart",
        keys: ["restart-global-memory"],
      },
    });

    await daemon.close();
    const pendingResult = client.callTool({ name: "list_memories", arguments: {} });
    await new Promise((resolve) => setTimeout(resolve, 150));
    daemon = await startDaemon({ port, idleMs: 60_000 });

    const result = await within(
      pendingResult,
      10_000,
      () => stderr
    );
    const memories = JSON.parse(toolText(result)) as Array<{ content: string }>;
    assert.ok(
      memories.some((memory) => memory.content === "global memory survives daemon restart"),
      "the reinitialized MCP session reads the same global graph"
    );
    assert.equal(shim.pid, shimPid, "stdio shim process stayed alive across daemon restart");
    assert.match(stderr, /\[shim reconnect\] restored/, "shim reported a successful reconnect");
  } finally {
    await client.close().catch(() => undefined);
    await daemon.close().catch(() => undefined);
    emb.__clearTestEmbedder();
    await rm(dataDir, { recursive: true, force: true });
  }
});
