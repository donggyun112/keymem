#!/usr/bin/env node
import { spawn } from "node:child_process";
import { realpathSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import type { JSONRPCMessage } from "@modelcontextprotocol/sdk/types.js";
import { runInProcess } from "./index.js";

const PORT = Number(process.env.KEYMEM_DAEMON_PORT ?? 8765);
const MCP_URL = `http://127.0.0.1:${PORT}/mcp`;
const HEALTH_URL = `http://127.0.0.1:${PORT}/health`;
const DAEMON_AUTOSTART = process.env.KEYMEM_DAEMON_AUTOSTART !== "false";
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function hostHeaders(env: NodeJS.ProcessEnv = process.env): Record<string, string> {
  const claude = env.CLAUDE_CODE_SESSION_ID;
  if (claude && UUID.test(claude)) return { "X-Keymem-Host-Agent": "claude", "X-Keymem-Host-Session": claude };
  const codex = env.CODEX_THREAD_ID;
  if (codex && UUID.test(codex)) return { "X-Keymem-Host-Agent": "codex", "X-Keymem-Host-Session": codex };
  return {};
}

// The daemon must have NO ambient host-session identity: its only source of
// host identity is per-request X-Keymem-Host-* headers (see hostHeaders()).
// Strip the session-identifying vars before handing env to the spawned
// daemon so it doesn't inherit whichever shim happened to autostart it.
// KEYMEM_TRANSCRIPT_ACCESS (explicit opt-in/out) and everything else pass
// through untouched.
export function daemonEnv(env: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
  const copy = { ...env };
  delete copy.CLAUDE_CODE_SESSION_ID;
  delete copy.CODEX_THREAD_ID;
  return copy;
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
      env: daemonEnv(process.env),
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
  const stdio = new StdioServerTransport();
  let http: StreamableHTTPClientTransport | null = null;
  let httpGeneration = 0;
  let reconnecting: Promise<void> | null = null;
  let outbound = Promise.resolve();
  let initializeRequest: JSONRPCMessage | null = null;
  let initializedNotification: JSONRPCMessage | null = null;
  let replayResponseId: string | null = null;
  let replaySequence = 0;
  let closing = false;

  const methodOf = (message: JSONRPCMessage): string | null => {
    const method = (message as { method?: unknown }).method;
    return typeof method === "string" ? method : null;
  };

  const idOf = (message: JSONRPCMessage): string | number | null => {
    const id = (message as { id?: unknown }).id;
    return typeof id === "string" || typeof id === "number" ? id : null;
  };

  const shutdown = (error?: unknown) => {
    if (error) console.error("[shim fatal bridge]", error);
    if (closing) return;
    closing = true;
    httpGeneration += 1;
    const active = http;
    http = null;
    void Promise.allSettled([
      active?.close() ?? Promise.resolve(),
      stdio.close(),
    ]);
  };

  const connectHttp = async (reinitialize: boolean): Promise<void> => {
    const generation = ++httpGeneration;
    const next = new StreamableHTTPClientTransport(new URL(MCP_URL), {
      requestInit: { headers: hostHeaders() },
    });

    next.onmessage = (message) => {
      if (closing || generation !== httpGeneration) return;
      if (replayResponseId !== null && idOf(message) === replayResponseId) return;
      void stdio.send(message).catch((error) => shutdown(error));
    };
    next.onerror = (error) => {
      if (generation === httpGeneration && !closing) {
        console.error("[shim http]", error);
      }
    };
    next.onclose = () => {
      if (closing || generation !== httpGeneration) return;
      http = null;
      void reconnect("http transport closed");
    };

    try {
      await next.start();
      if (closing || generation !== httpGeneration) {
        next.onclose = undefined;
        await next.close();
        throw new Error("stale HTTP transport generation");
      }
      http = next;

      // A daemon restart destroys only the ephemeral MCP transport session. Replay the
      // original handshake with a private request id to create a fresh session against the
      // same global graph. Its response is swallowed because the stdio client initialized once.
      if (reinitialize && initializeRequest) {
        replayResponseId = `keymem-reconnect-${++replaySequence}`;
        const replay = {
          ...(initializeRequest as Record<string, unknown>),
          id: replayResponseId,
        } as JSONRPCMessage;
        await next.send(replay);
        replayResponseId = null;
        if (initializedNotification) await next.send(initializedNotification);
      }
    } catch (error) {
      replayResponseId = null;
      if (http === next) http = null;
      if (generation === httpGeneration) httpGeneration += 1;
      next.onclose = undefined;
      await next.close().catch(() => undefined);
      throw error;
    }
  };

  const reconnect = (reason: string): Promise<void> => {
    if (closing) return Promise.reject(new Error("shim is closing"));
    if (reconnecting) return reconnecting;

    const task = (async () => {
      console.error(`[shim reconnect] ${reason}`);
      const stale = http;
      http = null;
      httpGeneration += 1;
      if (stale) {
        stale.onclose = undefined;
        await stale.close().catch(() => undefined);
      }

      let delayMs = 100;
      while (!closing) {
        const available = await ensureDaemon(MCP_URL, {
          timeoutMs: 8_000,
          spawnDaemon: DAEMON_AUTOSTART,
        });
        if (available) {
          try {
            await connectHttp(initializeRequest !== null);
            console.error("[shim reconnect] restored");
            return;
          } catch (error) {
            console.error("[shim reconnect] handshake failed", error);
          }
        }
        await new Promise((resolve) => setTimeout(resolve, delayMs));
        delayMs = Math.min(delayMs * 2, 1_000);
      }
      throw new Error("shim closed while reconnecting");
    })();

    reconnecting = task;
    task.then(
      () => {
        if (reconnecting === task) reconnecting = null;
      },
      () => {
        if (reconnecting === task) reconnecting = null;
      }
    );
    return task;
  };

  const sendWithReconnect = async (message: JSONRPCMessage): Promise<void> => {
    while (!closing) {
      if (!http) await reconnect("no active HTTP transport");
      const active = http;
      if (!active) continue;
      try {
        await active.send(message);
        return;
      } catch (error) {
        if (closing) return;
        if (http === active) {
          await reconnect(`send failed: ${error instanceof Error ? error.message : String(error)}`);
        } else if (reconnecting) {
          await reconnecting;
        }
      }
    }
  };

  // Preserve ordering across reconnects. The handshake messages are retained only to recreate
  // an ephemeral MCP session; memories remain in the daemon's single shared graph.
  stdio.onmessage = (message) => {
    const method = methodOf(message);
    if (method === "initialize" && idOf(message) !== null) initializeRequest = message;
    if (method === "notifications/initialized") initializedNotification = message;
    outbound = outbound.then(() => sendWithReconnect(message));
    void outbound.catch((error) => {
      if (!closing) console.error("[shim outbound]", error);
    });
  };
  stdio.onclose = () => {
    if (closing) return;
    closing = true;
    httpGeneration += 1;
    const active = http;
    http = null;
    void active?.close().catch((error) => console.error("[shim http close]", error));
  };

  await connectHttp(false);
  await stdio.start();
}

async function main(): Promise<void> {
  const ok = await ensureDaemon(MCP_URL, { spawnDaemon: DAEMON_AUTOSTART });
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
