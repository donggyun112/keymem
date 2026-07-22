import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { randomUUID } from "node:crypto";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { isInitializeRequest } from "@modelcontextprotocol/sdk/types.js";
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

  const rejectBadRequest = (res: ServerResponse, message: string) => {
    res.writeHead(400, { "content-type": "application/json" }).end(
      JSON.stringify({ jsonrpc: "2.0", error: { code: -32000, message }, id: null })
    );
  };

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

    // 바디를 먼저 읽어 initialize 요청인지 판단한 뒤 라우팅한다 (SDK 예제와 동일한 패턴).
    // 빈 바디이거나 JSON 파싱에 실패하면 undefined가 되고, 아래 isInitializeRequest 게이트가
    // 자연스럽게 이를 거부한다 (별도의 에러 플래그가 필요 없음).
    const body = await readBody(req);

    const sid = req.headers["mcp-session-id"];
    const sessionId = Array.isArray(sid) ? sid[0] : sid;

    let transport: StreamableHTTPServerTransport | undefined;

    if (sessionId) {
      // 기존 세션: map에 있어야만 라우팅한다. 없으면(데몬 재시작 후 stale id 등) 새 세션을
      // 만들지 않고 즉시 거부한다 — idle 타이머/transports 상태를 절대 건드리지 않는다.
      transport = transports.get(sessionId);
      if (!transport) {
        rejectBadRequest(res, "Bad Request: unknown or expired session ID");
        return;
      }
    } else if (isInitializeRequest(body)) {
      // 새 세션은 오직 (세션 id 없음 + initialize 요청)일 때만 생성한다.
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
    } else {
      // 세션 id도 없고 initialize도 아님 → idle 상태를 건드리지 않고 거부.
      rejectBadRequest(res, "Bad Request: no valid session ID provided");
      return;
    }

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
    close: async () => {
      cancelIdle();
      await Promise.allSettled([...transports.values()].map((t) => t.close()));
      await new Promise<void>((resolve) => httpServer.close(() => resolve()));
    },
  };
}

// CLI 진입점
if (import.meta.url === `file://${process.argv[1]}`) {
  startDaemon().catch((err) => {
    console.error("[daemon fatal]", err);
    process.exit(1);
  });
}
