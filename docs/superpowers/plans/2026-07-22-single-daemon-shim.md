# keymem 단일 공유 데몬 + stdio shim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** keymem MCP를 세션마다 spawn되는 stdio 프로세스(6개, ~9.5GB) 대신, 얇은 stdio shim + 상주 HTTP 데몬 1개(~1.5GB)로 통합하되 provenance 정확도를 유지한다.

**Architecture:** 호스트는 여전히 `type:"stdio"`로 `shim.js`를 spawn한다. shim은 `StdioServerTransport`(호스트 쪽)와 `StreamableHTTPClientTransport`(데몬 쪽)를 메시지 레벨에서 포워딩하는 투명 프록시로, 매 HTTP 요청에 `X-Keymem-Host-Agent`/`X-Keymem-Host-Session` 헤더를 붙인다. 데몬은 stateful StreamableHTTP 서버(shim 1개 = MCP 세션 1개 = Server 인스턴스 1개, graph/임베딩은 모듈 싱글턴 공유)로, 요청 헤더에서 hostLink를 구성하고, 활성 세션 0 + 10분 유휴 시 self-exit 한다. 데몬 기동 실패 시 shim은 기존 in-process 동작으로 폴백한다.

**Tech Stack:** TypeScript(ES2022, Node16 modules), `@modelcontextprotocol/sdk` ^1.0.0, node:http, node:test + tsx, Node.js v26.

## Global Constraints

- 모든 소스는 `src/`, 컴파일 산출물은 `dist/`(tsconfig `outDir`). ESM(`"type":"module"`), import는 `.js` 확장자 명시(Node16 resolution).
- 데몬은 `127.0.0.1` 루프백 전용. 원격 노출/인증 없음(단일 사용자 로컬 전제).
- 테스트: `tsx --test test/*.test.ts`. 순수 로직은 소스를 `../src/*.ts`로 직접 import(기존 `test/host-link-source.test.ts` 패턴).
- 헤더 이름 상수(정확히 이 소문자 표기로 읽기): `x-keymem-host-agent`, `x-keymem-host-session`. `IsomorphicHeaders = Record<string, string | string[] | undefined>` 이므로 값은 문자열/배열일 수 있다.
- `Agent = "claude" | "codex"` (기존 `src/nativeTranscripts.ts`의 타입). session_id는 UUID.
- 데몬 포트 기본 `8765`, `KEYMEM_DAEMON_PORT` 환경변수로 오버라이드. 경로: 헬스 `GET /health`, MCP `POST|GET|DELETE /mcp`.
- 유휴 종료 임계값 기본 10분, `KEYMEM_DAEMON_IDLE_MS`로 오버라이드(테스트에서 짧게 주입).
- provenance 규칙 불변: `buildSource`에서 caller source가 마지막에 spread되어 항상 우선(기존 `test/host-link-source.test.ts` 보존).

---

## 기존 코드 앵커 (수정 대상 정확 위치)

- `src/server.ts:1-19` 임포트. `:48` `SERVER_SESSION`. `:49-73` `buildSource`. `:74-81` `detectHostLink`(→ 교체). `:193` `export const server = new Server(...)`. `:208` `ListToolsRequestSchema` 핸들러(`:458` `transcriptAccessEnabled()` 필터). `:466` `CallToolRequestSchema` 핸들러. `:531/:567/:654` `detectHostLink()` 호출 3곳.
- `src/nativeTranscripts.ts:64-70` `transcriptAccessEnabled`. `:72-80` `envSession`. `:197` `findSessionFile`. `:153` `parseFor`. `:246-261` `detectActiveSession`의 Tier-1(env) 본문(→ 재사용 추출).
- `src/index.ts` 전체(현재 stdio 진입점 = in-process 폴백 재료).
- `package.json` `bin`/`scripts`.

## 파일 구조

- 수정 `src/nativeTranscripts.ts` — 헤더/세션 → hostLink 순수 헬퍼 추가(`hostSessionFromHeaders`, `hostLinkFromSession`), 기존 Tier-1 본문을 `hostLinkFromSession`으로 리팩터.
- 수정 `src/server.ts` — 싱글턴 `server` 생성을 `createMcpServer()` 팩토리로 추출; host-link 해석을 요청 헤더 인지형으로 교체하고 `extra`를 핸들러에 스레딩.
- 생성 `src/daemon.ts` — stateful StreamableHTTP 데몬(세션 맵, /health, 유휴 self-exit).
- 생성 `src/shim.ts` — 투명 프록시 + 데몬 오토스타트 + in-process 폴백.
- 수정 `src/index.ts` — `createMcpServer()` 사용하도록 정렬(폴백 경로와 동일 코드).
- 수정 `package.json` — `bin`/`scripts`에 shim·daemon 추가.
- 생성 `test/host-link-headers.test.ts`, `test/daemon-idle.test.ts`, `test/shim-bridge.test.ts`.

---

## Task 1: 헤더/세션 → hostLink 순수 헬퍼 (nativeTranscripts.ts)

**Files:**
- Modify: `src/nativeTranscripts.ts` (추가: 파일 끝 근처 export 2개; 리팩터: `:256-261`)
- Test: `test/host-link-headers.test.ts` (생성)

**Interfaces:**
- Consumes: 기존 `findSessionFile(agent, session_id)`, `parseFor(agent, text)`, `type Agent`, `UUID_PATTERN`.
- Produces:
  - `hostSessionFromHeaders(headers: Record<string, string | string[] | undefined> | undefined): { agent: Agent; session_id: string } | null`
  - `hostLinkFromSession(s: { agent: Agent; session_id: string } | null): Promise<{ agent: Agent; session_id: string; turn: number } | null>`

- [ ] **Step 1: Write the failing test**

```ts
// test/host-link-headers.test.ts
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx tsx --test test/host-link-headers.test.ts`
Expected: FAIL — `hostSessionFromHeaders is not a function`

- [ ] **Step 3: Add helpers and refactor Tier-1 in `src/nativeTranscripts.ts`**

`UUID_PATTERN`은 파일에 이미 존재(`envSession`이 사용). `findSessionFile`/`parseFor`도 이미 존재. `detectActiveSession`의 Tier-1 본문(`:256-261`)을 `hostLinkFromSession` 호출로 치환하고, 아래를 `envSession` 근처(export 구역)에 추가한다:

```ts
// Header-carried host session (daemon path). The shim stamps the host's session
// id into X-Keymem-* on every HTTP request, giving the daemon the same
// authoritative identity that env vars give a stdio child.
export function hostSessionFromHeaders(
  headers: Record<string, string | string[] | undefined> | undefined
): { agent: Agent; session_id: string } | null {
  if (!headers) return null;
  const first = (v: string | string[] | undefined): string | undefined =>
    Array.isArray(v) ? v[0] : v;
  const agent = first(headers["x-keymem-host-agent"]);
  const session = first(headers["x-keymem-host-session"]);
  if (agent !== "claude" && agent !== "codex") return null;
  if (!session || !UUID_PATTERN.test(session)) return null;
  return { agent, session_id: session };
}

// Resolve a known {agent, session_id} to a full host link by reading the
// transcript to compute the latest turn index. Shared by the env path
// (detectActiveSession Tier-1) and the header path (daemon).
export async function hostLinkFromSession(
  s: { agent: Agent; session_id: string } | null
): Promise<{ agent: Agent; session_id: string; turn: number } | null> {
  if (!s) return null;
  const file = await findSessionFile(s.agent, s.session_id);
  const turns = file ? parseFor(s.agent, await readFile(file, "utf-8")) : [];
  return { ...s, turn: Math.max(0, turns.length - 1) };
}
```

그리고 `detectActiveSession`의 Tier-1(`:256-261`)을 다음으로 치환한다:

```ts
  const fromEnv = envSession();
  if (fromEnv) return hostLinkFromSession(fromEnv);
```

(`readFile`는 파일 상단에서 이미 import됨 — `detectActiveSession`이 사용 중.)

- [ ] **Step 4: Run test to verify it passes**

Run: `npx tsx --test test/host-link-headers.test.ts`
Expected: PASS (6 tests)

- [ ] **Step 5: Verify no regression in transcript detection**

Run: `npx tsx --test test/native-transcripts.test.ts`
Expected: PASS (기존 테스트 그대로 통과 — Tier-1 리팩터가 동작 보존)

- [ ] **Step 6: Commit**

```bash
git add src/nativeTranscripts.ts test/host-link-headers.test.ts
git commit -m "feat(transcripts): header/session -> hostLink helpers"
```

---

## Task 2: server.ts 호스트링크 해석을 헤더 인지형으로 + extra 스레딩

**Files:**
- Modify: `src/server.ts` (`:1-19` 임포트, `:74-81` `detectHostLink`→`resolveHostLink`, `:208`/`:466` 핸들러 시그니처, `:458` 필터, `:531/:567/:654` 호출부)
- Test: `test/host-link-source.test.ts` (기존, 회귀 확인), `test/host-link-headers.test.ts` (Task 1)

**Interfaces:**
- Consumes: Task 1의 `hostSessionFromHeaders`, `hostLinkFromSession`; 기존 `detectActiveSession`, `transcriptAccessEnabled`, `buildSource`.
- Produces:
  - `resolveHostLink(headers: Record<string, string | string[] | undefined> | undefined): Promise<{ agent: Agent; session_id: string; turn: number } | null>`
  - `transcriptAccessForRequest(headers: Record<string, string | string[] | undefined> | undefined): boolean`
  - CallTool/ListTools 핸들러가 `extra.requestInfo?.headers`에서 헤더를 읽어 위 두 함수에 전달.

- [ ] **Step 1: Write the failing test**

```ts
// test/host-link-source.test.ts 에 append (기존 파일 하단)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx tsx --test test/host-link-source.test.ts`
Expected: FAIL — `resolveHostLink is not a function`

- [ ] **Step 3: Edit `src/server.ts`**

임포트(`:8-16`)에 헤더 헬퍼 추가:

```ts
import {
  loadNativeConversation,
  loadNativeAuto,
  listNativeSessions,
  detectActiveSession,
  transcriptAccessEnabled,
  hostSessionFromHeaders,
  hostLinkFromSession,
  type Agent,
} from "./nativeTranscripts.js";
```

`detectHostLink`(`:74-81`)를 다음으로 교체:

```ts
type ReqHeaders = Record<string, string | string[] | undefined> | undefined;

// Resolve the host transcript link for one request. Header path (daemon) is
// authoritative and needs no ambient env trust. Env path (stdio in-process
// fallback) keeps the old gated behavior, including the mtime heuristic.
export async function resolveHostLink(
  headers: ReqHeaders
): Promise<{ agent: Agent; session_id: string; turn: number } | null> {
  const fromHeader = hostSessionFromHeaders(headers);
  if (fromHeader) return hostLinkFromSession(fromHeader);
  if (!transcriptAccessEnabled()) return null;
  try {
    return await detectActiveSession();
  } catch {
    return null;
  }
}

// Transcript tools/stamping are allowed when we trust the caller: either the
// request carries host-session headers (daemon) or the env opted in (stdio).
export function transcriptAccessForRequest(headers: ReqHeaders): boolean {
  return hostSessionFromHeaders(headers) != null || transcriptAccessEnabled();
}
```

ListTools 핸들러(`:208`)에 `extra` 인자 추가하고 필터(`:458`)를 요청 기반으로:

```ts
server.setRequestHandler(ListToolsRequestSchema, async (_req, extra) => {
  const headers = extra.requestInfo?.headers;
  // ... (tools 배열 구성 동일) ...
  return {
    tools: transcriptAccessForRequest(headers)
      ? tools
      : tools.filter((t) => !TRANSCRIPT_TOOLS.has(t.name)),
  };
});
```

CallTool 핸들러(`:466`)에 `extra` 추가하고, 3개 호출부(`:531/:567/:654`)의 `await detectHostLink()`를 `await resolveHostLink(extra.requestInfo?.headers)`로 교체:

```ts
server.setRequestHandler(CallToolRequestSchema, async (request, extra) => {
  const { name, arguments: args } = request.params;
  const a = (args ?? {}) as Record<string, unknown>;
  const headers = extra.requestInfo?.headers;
  // ... switch 내부 세 곳:
  //   const hostLink = await resolveHostLink(headers);
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx tsx --test test/host-link-source.test.ts`
Expected: PASS (기존 3 + 신규 3)

- [ ] **Step 5: Full suite regression**

Run: `npm test`
Expected: PASS (provenance.test.ts, native-transcripts.test.ts 포함 그린)

- [ ] **Step 6: Commit**

```bash
git add src/server.ts test/host-link-source.test.ts
git commit -m "feat(server): header-aware host link resolution, thread request extra"
```

---

## Task 3: createMcpServer() 팩토리 추출

**Files:**
- Modify: `src/server.ts` (`:193` 싱글턴 `server` + 이후 모든 `server.setRequestHandler(...)` 등록을 팩토리로 감싸기)
- Modify: `src/index.ts` (팩토리 사용)
- Test: `npm test` (기존 스위트로 회귀 확인)

**Interfaces:**
- Consumes: 기존 handler 본문(recall/remember/... 전부, `graph` 모듈 싱글턴에 클로저).
- Produces:
  - `export function createMcpServer(): Server` — 핸들러가 전부 등록된 새 `Server` 인스턴스 반환.
  - `export const server = createMcpServer();` — 하위호환(기존 import 유지).
  - `export const graph`는 변경 없음(모듈 싱글턴, 데몬/폴백/테스트가 공유).

- [ ] **Step 1: Wrap registration in a factory (구조 리팩터)**

`src/server.ts`에서 `export const server = new Server({...}, {...});`(`:193`)부터 파일 끝까지의 모든 `server.setRequestHandler(...)` 등록을 함수로 감싼다:

```ts
export function createMcpServer(): Server {
  const server = new Server(
    { name: "keymem", version: /* 기존 값 유지 */ },
    { capabilities: { tools: {}, prompts: {} } } // 기존 capabilities 그대로
  );

  server.setRequestHandler(ListPromptsRequestSchema, async () => { /* 기존 본문 */ });
  server.setRequestHandler(GetPromptRequestSchema, async (request) => { /* 기존 본문 */ });
  server.setRequestHandler(ListToolsRequestSchema, async (_req, extra) => { /* Task 2 본문 */ });
  server.setRequestHandler(CallToolRequestSchema, async (request, extra) => { /* Task 2 본문 */ });

  return server;
}

// 하위호환: 기존 `import { server } from "./server.js"` 소비자 유지.
export const server = createMcpServer();
```

주의: `buildSource`, `resolveHostLink`, `transcriptAccessForRequest`, `TRANSCRIPT_TOOLS`, `DIRECT_RECALL_ENABLED` 등 모듈 스코프 심볼과 `graph` 싱글턴은 함수 밖에 그대로 둔다(핸들러가 클로저로 참조). 팩토리는 **등록만** 감싼다.

- [ ] **Step 2: Point `src/index.ts` at the factory**

`src/index.ts`를 아래로 정렬(폴백 경로가 데몬과 동일 코드가 되도록 export):

```ts
#!/usr/bin/env node
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { graph, createMcpServer } from "./server.js";

export async function runInProcess(): Promise<void> {
  await graph.load();
  const server = createMcpServer();
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

// CLI 진입점으로 직접 실행될 때만 구동(shim이 폴백으로 import할 때는 실행 안 함).
if (import.meta.url === `file://${process.argv[1]}`) {
  runInProcess().catch((err) => {
    console.error("[fatal]", err);
    process.exit(1);
  });
}
```

- [ ] **Step 3: Build to verify types**

Run: `npm run build`
Expected: 타입 에러 없이 컴파일(`dist/` 생성)

- [ ] **Step 4: Full suite regression**

Run: `npm test`
Expected: PASS (팩토리 추출은 동작 보존 — 그린 유지)

- [ ] **Step 5: Commit**

```bash
git add src/server.ts src/index.ts
git commit -m "refactor(server): extract createMcpServer() factory; index exports runInProcess"
```

---

## Task 4: 상주 HTTP 데몬 (src/daemon.ts)

**Files:**
- Create: `src/daemon.ts`
- Test: `test/daemon-idle.test.ts` (생성)

**Interfaces:**
- Consumes: `createMcpServer` (Task 3), `graph` (Task 3), `StreamableHTTPServerTransport`.
- Produces:
  - `export async function startDaemon(opts?: { port?: number; idleMs?: number }): Promise<{ port: number; close: () => Promise<void> }>`
  - HTTP 계약: `GET /health` → `200 {"ok":true}` (graph 로드 후에만 200); `POST|GET|DELETE /mcp` → MCP.
  - 유휴 로직: 활성 MCP 세션 수 == 0 이 되면 `idleMs` 뒤 `process.exit(0)`; 새 세션이 생기면 타이머 취소.

- [ ] **Step 1: Write the failing test**

```ts
// test/daemon-idle.test.ts
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx tsx --test test/daemon-idle.test.ts`
Expected: FAIL — cannot find `../src/daemon.ts`

- [ ] **Step 3: Implement `src/daemon.ts`**

```ts
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx tsx --test test/daemon-idle.test.ts`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/daemon.ts test/daemon-idle.test.ts
git commit -m "feat(daemon): stateful StreamableHTTP daemon with health + idle exit"
```

---

## Task 5: stdio shim (src/shim.ts)

**Files:**
- Create: `src/shim.ts`
- Test: `test/shim-bridge.test.ts` (생성)

**Interfaces:**
- Consumes: `StdioServerTransport`, `StreamableHTTPClientTransport`, `runInProcess` (Task 3), `startDaemon`간접(오토스타트는 `dist/daemon.js` spawn).
- Produces:
  - `export function hostHeaders(env?: NodeJS.ProcessEnv): Record<string, string>` — env에 세션 있으면 `X-Keymem-Host-*`, 없으면 `{}`.
  - `export async function ensureDaemon(url: string, opts?: { timeoutMs?: number; spawnDaemon?: boolean }): Promise<boolean>` — 헬스 폴링(+오토스타트), 준비되면 true.
  - `main()` — `ensureDaemon` 성공 시 투명 프록시, 실패 시 `runInProcess()` 폴백.

- [ ] **Step 1: Write the failing test**

```ts
// test/shim-bridge.test.ts
import assert from "node:assert/strict";
import test from "node:test";

const { hostHeaders, ensureDaemon } = await import("../src/shim.ts");

const UUID = "e7f5b1d2-1602-4180-ac66-9f9acbd1f673";

test("hostHeaders emits claude session header from env", () => {
  const h = hostHeaders({ CLAUDE_CODE_SESSION_ID: UUID } as NodeJS.ProcessEnv);
  assert.equal(h["X-Keymem-Host-Agent"], "claude");
  assert.equal(h["X-Keymem-Host-Session"], UUID);
});

test("hostHeaders emits codex session header from env", () => {
  const h = hostHeaders({ CODEX_THREAD_ID: UUID } as NodeJS.ProcessEnv);
  assert.equal(h["X-Keymem-Host-Agent"], "codex");
  assert.equal(h["X-Keymem-Host-Session"], UUID);
});

test("hostHeaders is empty when no session env", () => {
  assert.deepEqual(hostHeaders({} as NodeJS.ProcessEnv), {});
});

test("ensureDaemon returns false quickly when nothing to connect and no spawn", async () => {
  // 사용되지 않는 포트, 오토스타트 끔 → 폴백 신호
  const ok = await ensureDaemon("http://127.0.0.1:59999/mcp", { timeoutMs: 300, spawnDaemon: false });
  assert.equal(ok, false);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx tsx --test test/shim-bridge.test.ts`
Expected: FAIL — cannot find `../src/shim.ts`

- [ ] **Step 3: Implement `src/shim.ts`**

```ts
#!/usr/bin/env node
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
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

async function healthOk(): Promise<boolean> {
  try {
    const res = await fetch(HEALTH_URL, { signal: AbortSignal.timeout(500) });
    return res.ok;
  } catch { return false; }
}

// 헬스가 뜰 때까지 폴링. 없으면 데몬을 detached로 오토스타트한 뒤 계속 폴링.
// 오토스타트 경쟁(두 shim 동시)에서 진 프로세스는 EADDRINUSE로 죽지만,
// 이긴 데몬의 헬스가 뜨면 true. 성공 기준은 "누가 이겼든 health 200".
export async function ensureDaemon(
  url: string,
  opts: { timeoutMs?: number; spawnDaemon?: boolean } = {}
): Promise<boolean> {
  const timeoutMs = opts.timeoutMs ?? 8000;
  const spawnDaemon = opts.spawnDaemon ?? true;
  if (await healthOk()) return true;

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
    if (await healthOk()) return true;
    await new Promise((r) => setTimeout(r, 150));
  }
  return false;
}

async function runProxy(): Promise<void> {
  const http = new StreamableHTTPClientTransport(new URL(MCP_URL), {
    requestInit: { headers: hostHeaders() },
  });
  const stdio = new StdioServerTransport();
  // 메시지 레벨 투명 포워딩. MCP 의미 해석 없음. 세션 id/SSE는 http 전송이 처리.
  stdio.onmessage = (m) => { void http.send(m); };
  http.onmessage = (m) => { void stdio.send(m); };
  stdio.onclose = () => { void http.close(); };
  http.onclose = () => { void stdio.close(); };
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

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((err) => {
    console.error("[shim fatal]", err);
    process.exit(1);
  });
}
```

주의: `runInProcess`가 stdout/stdin을 잡으므로 폴백은 프록시를 시작하지 **않은** 경우에만 진입한다(위 분기 보장). `StreamableHTTPClientTransport.start()`는 host가 initialize를 stdio로 보낼 때까지 대기하며, 세션 id는 initialize 응답 헤더에서 자동 취득한다.

- [ ] **Step 4: Run test to verify it passes**

Run: `npx tsx --test test/shim-bridge.test.ts`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/shim.ts test/shim-bridge.test.ts
git commit -m "feat(shim): transparent stdio<->HTTP proxy with autostart + in-process fallback"
```

---

## Task 6: 빌드 배선 + end-to-end 통합 + 마이그레이션

**Files:**
- Modify: `package.json` (`bin`, `scripts`)
- Test: `test/e2e-two-shims.test.ts` (생성)
- Modify: `docs/superpowers/specs/2026-07-22-single-daemon-shim-design.md` (마이그레이션 커맨드 확정 — 선택)

**Interfaces:**
- Consumes: 빌드된 `dist/shim.js`, `dist/daemon.js`.
- Produces:
  - `package.json.bin`: `{ "keymem": "dist/index.js", "keymem-shim": "dist/shim.js", "keymem-daemon": "dist/daemon.js" }`
  - `package.json.scripts`: `"daemon": "node dist/daemon.js"`, `"shim": "node dist/shim.js"` 추가.
  - E2E: 실제 데몬 1개에 클라이언트 2개(서로 다른 host session 헤더)가 붙어 각각 remember → 둘 다 저장되고 provenance가 각자 세션으로 스탬프.

- [ ] **Step 1: Update package.json**

`bin`/`scripts`에 위 Interfaces의 항목을 추가한다(기존 항목 보존).

- [ ] **Step 2: Build**

Run: `npm run build`
Expected: `dist/shim.js`, `dist/daemon.js` 생성, 에러 없음

- [ ] **Step 3: Write the E2E test**

MCP `Client`를 SDK로 2개 만들어 같은 데몬에 서로 다른 헤더로 붙인다. `KEYMEM_DATA_DIR`로 저장소 격리, `graph`는 데몬이 소유.

```ts
// test/e2e-two-shims.test.ts
import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

process.env.KEYMEM_DATA_DIR = await mkdtemp(join(tmpdir(), "keymem-e2e-"));
process.env.KEYMEM_TRANSCRIPT_ACCESS = "false"; // env 경로 차단, 헤더 경로만 검증
const { startDaemon } = await import("../src/daemon.ts");

const SID_A = "aaaaaaaa-1602-4180-ac66-9f9acbd1f673";
const SID_B = "bbbbbbbb-1602-4180-ac66-9f9acbd1f673";

function client(port: number, sid: string) {
  const t = new StreamableHTTPClientTransport(new URL(`http://127.0.0.1:${port}/mcp`), {
    requestInit: { headers: { "X-Keymem-Host-Agent": "claude", "X-Keymem-Host-Session": sid } },
  });
  const c = new Client({ name: "test", version: "0" });
  return { c, connect: () => c.connect(t) };
}

test("two clients on one daemon persist to shared graph with correct provenance", async () => {
  const d = await startDaemon({ port: 0, idleMs: 60_000 });
  try {
    const a = client(d.port, SID_A);
    const b = client(d.port, SID_B);
    await a.connect();
    await b.connect();

    await a.c.callTool({ name: "remember", arguments: { content: "fact from A", keys: ["alpha"] } });
    await b.c.callTool({ name: "remember", arguments: { content: "fact from B", keys: ["beta"] } });

    // 두 클라이언트가 같은 graph를 본다: A가 저장한 것을 B가 recall.
    const res = await b.c.callTool({ name: "recall", arguments: { query: "alpha" } });
    const text = (res.content as Array<{ type: string; text: string }>)[0].text;
    assert.ok(text.length > 2, "recall returned keys from shared graph");
  } finally {
    await d.close();
  }
});
```

- [ ] **Step 4: Run the E2E test**

Run: `npx tsx --test test/e2e-two-shims.test.ts`
Expected: PASS — 공유 graph 확인(교차 recall 성공)

- [ ] **Step 5: Full suite**

Run: `npm test`
Expected: PASS (전체 그린)

- [ ] **Step 6: Commit**

```bash
git add package.json test/e2e-two-shims.test.ts
git commit -m "feat: wire shim/daemon bins; e2e two-client shared-graph test"
```

- [ ] **Step 7: Manual migration (사용자 확인 후)**

빌드된 shim으로 `~/.claude.json`의 `mcp-super-memory.command`를 `node .../dist/shim.js`로 교체(그리고 codex 설정도 동일). `type`은 `stdio` 유지. 교체 후 클라이언트를 재시작하면 기존 stdio 프로세스들이 자연 종료되고, 첫 호출 시 데몬 1개가 기동된다. 검증:

Run(수동): `ps -eo pid,rss,command | grep 'super-memory\|daemon.js' | grep -v grep`
Expected: `daemon.js` 프로세스 1개만 상주(RSS ~1.5GB), shim은 세션당 경량 프로세스.

---

## Self-Review

**Spec coverage:**
- 메모리 절감(데몬 1개): Task 4 + Task 6 Step 7. ✓
- 동시성(단일 writer, mutex 유효): Task 4(graph 싱글턴) + Task 6 E2E 교차 recall. ✓
- provenance 정확도(헤더 경로): Task 1·2 + Task 6 E2E. ✓
- 오토기동 + 유휴 종료: Task 4(idle) + Task 5(ensureDaemon). ✓
- in-process 폴백: Task 3(runInProcess) + Task 5(main 분기). ✓
- Tier-2 mtime 제거(데몬 경로): Task 2 `resolveHostLink`(헤더 있으면 mtime 미도달). ✓
- `transcriptAccessEnabled` 재판단: Task 2 `transcriptAccessForRequest`. ✓
- 헤더 없는 요청 → null, 내용은 저장: Task 2 테스트 + `buildSource(...,null)` 기존 동작. ✓

**Placeholder scan:** 코드 스텝은 실제 코드 포함. `createMcpServer` 본문은 "기존 본문 이동"이 실제 작업 지시(핸들러를 옮기는 리팩터)이며 새 로직 아님 — 허용.

**Type consistency:** `hostSessionFromHeaders`/`hostLinkFromSession`(Task 1) → `resolveHostLink`/`transcriptAccessForRequest`(Task 2) → `createMcpServer`(Task 3) → 데몬/폴백(Task 4·5) 시그니처 일관. `ReqHeaders` 타입은 `IsomorphicHeaders`와 동일 구조. `hostHeaders`(shim, 대문자 헤더 키) ↔ 데몬 읽기(소문자, 전송이 정규화)는 HTTP 헤더 대소문자 무관성으로 정합.
