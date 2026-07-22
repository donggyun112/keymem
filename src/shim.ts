#!/usr/bin/env node
import { spawn } from "node:child_process";
import { realpathSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { runInProcess } from "./index.js";

const PORT = Number(process.env.KEYMEM_DAEMON_PORT ?? 8765);
const MCP_URL = `http://127.0.0.1:${PORT}/mcp`;
const HEALTH_URL = `http://127.0.0.1:${PORT}/health`;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function hostHeaders(env: NodeJS.ProcessEnv = process.env): Record<string, string> {
  const claude = env.CLAUDE_CODE_SESSION_ID;
  if (claude && UUID.test(claude)) return { "X-Keymem-Host-Agent": "claude", "X-Keymem-Host-Session": claude };
  const codex = env.CODEX_THREAD_ID;
  if (codex && UUID.test(codex)) return { "X-Keymem-Host-Agent": "codex", "X-Keymem-Host-Session": codex };
  return {};
}

// Derives the /health URL for the same host:port as the given MCP url, so
// ensureDaemon() actually checks the target it was asked about rather than
// silently falling back to the module-level default port.
function healthUrlFor(url: string): string {
  try {
    const u = new URL(url);
    u.pathname = "/health";
    u.search = "";
    return u.toString();
  } catch {
    return HEALTH_URL;
  }
}

async function healthOk(url: string): Promise<boolean> {
  try {
    const res = await fetch(healthUrlFor(url), { signal: AbortSignal.timeout(500) });
    return res.ok;
  } catch {
    return false;
  }
}

// Poll health until it's up. If not up, autostart the daemon detached, then keep polling.
// In an autostart race (two shims at once), the loser dies with EADDRINUSE, but once the
// winner's daemon health is up we return true. Success criterion is "whoever wins, health 200".
export async function ensureDaemon(
  url: string,
  opts: { timeoutMs?: number; spawnDaemon?: boolean } = {}
): Promise<boolean> {
  const timeoutMs = opts.timeoutMs ?? 8000;
  const spawnDaemon = opts.spawnDaemon ?? true;
  if (await healthOk(url)) return true;

  if (spawnDaemon) {
    const here = dirname(fileURLToPath(import.meta.url));
    const daemonPath = join(here, "daemon.js");
    const child = spawn(process.execPath, [daemonPath], {
      detached: true,
      stdio: "ignore",
      env: process.env,
    });
    child.unref();
  }

  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await healthOk(url)) return true;
    await new Promise((r) => setTimeout(r, 150));
  }
  return false;
}

async function runProxy(): Promise<void> {
  const http = new StreamableHTTPClientTransport(new URL(MCP_URL), {
    requestInit: { headers: hostHeaders() },
  });
  const stdio = new StdioServerTransport();
  // Message-level transparent forwarding. No MCP semantics interpreted here.
  // Session id / SSE handling is owned by the HTTP transport.
  stdio.onmessage = (m) => {
    void http.send(m);
  };
  http.onmessage = (m) => {
    void stdio.send(m);
  };
  stdio.onclose = () => {
    void http.close();
  };
  http.onclose = () => {
    void stdio.close();
  };
  http.onerror = (e) => console.error("[shim http]", e);
  await http.start();
  await stdio.start();
}

async function main(): Promise<void> {
  const ok = await ensureDaemon(MCP_URL);
  if (ok) return runProxy();
  console.error("[shim] daemon unavailable; falling back to in-process server");
  return runInProcess();
}

// Resolve argv[1] to its realpath before comparing: when invoked via a
// symlinked bin (e.g. the package.json `keymem-shim` bin), process.argv[1]
// is the symlink path while import.meta.url is realpath-resolved, so a
// naive string compare is falsy and main() silently never runs.
function isCliEntry(): boolean {
  return process.argv[1] !== undefined && import.meta.url === pathToFileURL(realpathSync(process.argv[1])).href;
}

if (isCliEntry()) {
  main().catch((err) => {
    console.error("[shim fatal]", err);
    process.exit(1);
  });
}
