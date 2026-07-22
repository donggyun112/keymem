import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { randomUUID } from "node:crypto";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { graph, createMcpServer } from "./server.js";

const DEFAULT_PORT = Number(process.env.KEYMEM_DAEMON_PORT ?? 8765);
const DEFAULT_IDLE_MS = Number(process.env.KEYMEM_DAEMON_IDLE_MS ?? 10 * 60_000);

export async function startDaemon(
  opts: { port?: number; idleMs?: number } = {}
): Promise<{ port: number; close: () => Promise<void> }> {
  const idleMs = opts.idleMs ?? DEFAULT_IDLE_MS;
  await graph.load(); // 임베딩 모델은 첫 사용 시 lazy load

  // 세션별 transport. 각 shim = 1 MCP 세션 = 1 Server 인스턴스(graph는 공유).
  const transports = new Map<string, StreamableHTTPServerTransport>();
  let idleTimer: NodeJS.Timeout | null = null;

  const armIdle = () => {
    if (idleTimer) clearTimeout(idleTimer);
    if (transports.size === 0) {
      idleTimer = setTimeout(() => process.exit(0), idleMs);
    }
  };
  const cancelIdle = () => {
    if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
  };

  const readBody = (req: IncomingMessage): Promise<unknown> =>
    new Promise((resolve) => {
      let data = "";
      req.on("data", (c) => (data += c));
      req.on("end", () => {
        try { resolve(data ? JSON.parse(data) : undefined); } catch { resolve(undefined); }
      });
    });

  const httpServer = createServer(async (req: IncomingMessage, res: ServerResponse) => {
    const url = new URL(req.url ?? "/", "http://127.0.0.1");
    if (url.pathname === "/health") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ ok: true }));
      return;
    }
    if (url.pathname !== "/mcp") {
      res.writeHead(404).end();
      return;
    }

    const sid = req.headers["mcp-session-id"];
    const sessionId = Array.isArray(sid) ? sid[0] : sid;
    let transport = sessionId ? transports.get(sessionId) : undefined;

    if (!transport) {
      // 새 세션: initialize 요청에서 transport 생성.
      cancelIdle();
      transport = new StreamableHTTPServerTransport({
        sessionIdGenerator: () => randomUUID(),
        onsessioninitialized: (id) => { transports.set(id, transport!); },
      });
      transport.onclose = () => {
        if (transport!.sessionId) transports.delete(transport!.sessionId);
        armIdle();
      };
      const mcp = createMcpServer();
      await mcp.connect(transport);
    }

    const body = await readBody(req);
    await transport.handleRequest(req, res, body);
  });

  const port = await new Promise<number>((resolve) => {
    httpServer.listen(opts.port ?? DEFAULT_PORT, "127.0.0.1", () => {
      const addr = httpServer.address();
      resolve(typeof addr === "object" && addr ? addr.port : (opts.port ?? DEFAULT_PORT));
    });
  });
  armIdle(); // 접속 없이 시작하면 유휴 타이머 무장

  return {
    port,
    close: () =>
      new Promise<void>((resolve) => {
        cancelIdle();
        for (const t of transports.values()) void t.close();
        httpServer.close(() => resolve());
      }),
  };
}

// CLI 진입점
if (import.meta.url === `file://${process.argv[1]}`) {
  startDaemon().catch((err) => {
    console.error("[daemon fatal]", err);
    process.exit(1);
  });
}
