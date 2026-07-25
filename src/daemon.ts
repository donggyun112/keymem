import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { randomUUID } from "node:crypto";
import { realpathSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { isInitializeRequest } from "@modelcontextprotocol/sdk/types.js";
import { graph, createMcpServer } from "./server.js";

const DEFAULT_PORT = Number(process.env.KEYMEM_DAEMON_PORT ?? 8765);
const DEFAULT_IDLE_MS = Number(process.env.KEYMEM_DAEMON_IDLE_MS ?? 10 * 60_000);
const DEFAULT_SESSION_REAP_GRACE_MS = Number(process.env.KEYMEM_SESSION_REAP_GRACE_MS ?? 15_000);

export async function startDaemon(
  opts: { port?: number; idleMs?: number; sessionReapGraceMs?: number } = {}
): Promise<{
  port: number;
  close: () => Promise<void>;
  sessionCount: () => number;
  sseConnectionCount: () => number;
  disconnectSse: (sessionId?: string) => number;
}> {
  const idleMs = opts.idleMs ?? DEFAULT_IDLE_MS;
  const sessionReapGraceMs = opts.sessionReapGraceMs ?? DEFAULT_SESSION_REAP_GRACE_MS;
  await graph.load(); // 임베딩 모델은 첫 사용 시 lazy load

  // 세션별 transport. 각 shim = 1 MCP 세션 = 1 Server 인스턴스(graph는 공유).
  const transports = new Map<string, StreamableHTTPServerTransport>();
  const sseResponses = new Map<string, Set<ServerResponse>>();
  const reapTimers = new Map<string, NodeJS.Timeout>();
  let idleTimer: NodeJS.Timeout | null = null;
  let closing = false;

  const armIdle = () => {
    if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
    if (!closing && transports.size === 0) {
      idleTimer = setTimeout(() => process.exit(0), idleMs);
    }
  };
  const cancelIdle = () => {
    if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
  };
  const cancelSessionReap = (sessionId: string) => {
    const timer = reapTimers.get(sessionId);
    if (timer) clearTimeout(timer);
    reapTimers.delete(sessionId);
  };
  const scheduleSessionReap = (
    sessionId: string,
    transport: StreamableHTTPServerTransport
  ) => {
    cancelSessionReap(sessionId);
    if (closing) return;
    const timer = setTimeout(() => {
      reapTimers.delete(sessionId);
      if ((sseResponses.get(sessionId)?.size ?? 0) > 0) return;
      if (transports.get(sessionId) === transport) {
        transport.close().catch(() => {});
      }
    }, sessionReapGraceMs);
    reapTimers.set(sessionId, timer);
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
      cancelSessionReap(sessionId);
      if (req.method === "GET") {
        // GET /mcp는 이 세션의 standalone SSE 스트림이다 — shim이 살아있는 동안 계속 열려
        // 있는 연결로, "이 세션이 아직 쓰이고 있다"는 유일한 신호다. shim이 SIGKILL 등으로
        // 죽으면 이 소켓은 res.end() 없이 그냥 끊긴다. SDK(WebStandardStreamableHTTPServerTransport)
        // 는 이 경우 내부 스트림 매핑만 지우고 transport.onclose는 절대 호출하지 않는다
        // (node_modules/@modelcontextprotocol/sdk의 standalone SSE ReadableStream의 cancel()
        // 콜백을 직접 확인함) — 그래서 phantom 세션이 transports map에 영원히 남고 idle-exit이
        // 절대 발생하지 않는다.
        //
        // SDK 클라이언트는 일시적인 SSE 단절 뒤 같은 session id로 재접속한다. 따라서 close 즉시
        // transport를 삭제하면 정상 재접속도 stale-session 400으로 깨진다. 세션별 열린 SSE 수를
        // 추적하고 마지막 스트림이 비정상 종료된 뒤 grace window 동안 재접속이 없을 때만 reap한다.
        // res.writableEnded=true인 서버 주도 정상 종료는 reap을 예약하지 않는다.
        const staleTransport = transport;
        let responses = sseResponses.get(sessionId);
        if (!responses) {
          responses = new Set();
          sseResponses.set(sessionId, responses);
        }
        responses.add(res);
        res.on("close", () => {
          const active = sseResponses.get(sessionId);
          active?.delete(res);
          if (active?.size === 0) sseResponses.delete(sessionId);
          if (!res.writableEnded && (active?.size ?? 0) === 0) {
            scheduleSessionReap(sessionId, staleTransport);
          }
        });
      }
    } else if (isInitializeRequest(body)) {
      // 새 세션은 오직 (세션 id 없음 + initialize 요청)일 때만 생성한다.
      cancelIdle();
      transport = new StreamableHTTPServerTransport({
        sessionIdGenerator: () => randomUUID(),
        onsessioninitialized: (id) => { transports.set(id, transport!); },
      });
      transport.onclose = () => {
        if (transport!.sessionId) {
          cancelSessionReap(transport!.sessionId);
          sseResponses.delete(transport!.sessionId);
          transports.delete(transport!.sessionId);
        }
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

    // SSE가 끊긴 grace window 중에도 POST 요청은 도착할 수 있다. 그 요청은 shim이 아직
    // 살아있다는 신호이므로 기존 타이머를 연장하되, SSE가 끝내 복구되지 않는 경우 phantom
    // session이 영구 잔류하지 않도록 요청 완료 뒤 reap을 다시 예약한다.
    if (
      sessionId &&
      req.method !== "GET" &&
      transports.get(sessionId) === transport &&
      (sseResponses.get(sessionId)?.size ?? 0) === 0
    ) {
      scheduleSessionReap(sessionId, transport);
    }
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
    // 테스트 전용 읽기 전용 훅: 현재 열려 있는(reap되지 않은) 세션 수. 프로덕션 코드 경로에는
    // 영향 없음 — transports.size를 그대로 노출할 뿐이다.
    sessionCount: () => transports.size,
    sseConnectionCount: () =>
      [...sseResponses.values()].reduce((count, responses) => count + responses.size, 0),
    // 테스트에서 네트워크 단절을 재현하기 위한 훅. 실제 종료 로직과 동일한 res 'close' 경로를 탄다.
    disconnectSse: (sessionId?: string) => {
      const targets = sessionId
        ? [...(sseResponses.get(sessionId) ?? [])]
        : [...sseResponses.values()].flatMap((responses) => [...responses]);
      for (const response of targets) response.destroy();
      return targets.length;
    },
    close: async () => {
      closing = true;
      cancelIdle();
      for (const timer of reapTimers.values()) clearTimeout(timer);
      reapTimers.clear();
      await Promise.allSettled([...transports.values()].map((t) => t.close()));
      sseResponses.clear();
      // Belt-and-suspenders: onclose→armIdle() no-ops while closing===true, but cancel again
      // in case anything slipped in between.
      cancelIdle();
      await new Promise<void>((resolve) => httpServer.close(() => resolve()));
    },
  };
}

// CLI 진입점. Resolve argv[1] to its realpath before comparing: when invoked
// via a symlinked bin (e.g. the package.json `keymem-daemon` bin),
// process.argv[1] is the symlink path while import.meta.url is
// realpath-resolved, so a naive string compare is falsy and main() silently
// never runs.
function isCliEntry(): boolean {
  return process.argv[1] !== undefined && import.meta.url === pathToFileURL(realpathSync(process.argv[1])).href;
}

if (isCliEntry()) {
  startDaemon().catch((err) => {
    console.error("[daemon fatal]", err);
    process.exit(1);
  });
}
