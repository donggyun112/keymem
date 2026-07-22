# keymem 단일 공유 데몬 + stdio shim 설계

날짜: 2026-07-22
상태: 승인 대기

## 배경 / 문제

keymem(super-memory)은 현재 `type: "stdio"` MCP 서버로 등록되어 있고
(`src/index.ts`), 호스트 클라이언트(Claude Code, Codex)가 **세션마다 프로세스를
하나씩 spawn**한다. 그 결과:

- 실측 6개 프로세스가 동시 상주, 각 RSS 1.4~1.7GB (`EMBEDDING_BACKEND=local` +
  `bge-m3` 모델을 프로세스마다 별도 로드) → 합계 약 9.5GB.
- 6개 프로세스가 전부 같은 데이터 디렉터리(`~/.keymem`)의 `graph.json`에 쓴다
  (`src/memoryGraph.ts:19`). 그런데 동시성 제어는 `async-mutex` —
  **프로세스 내부 락**이라 프로세스 간에는 무효하다. atomic rename 덕에 파일
  깨짐은 드물지만 lost update는 가능하다.

롱텀 메모리는 세션별 격리가 불필요하다(설계 의도상 `graph.json`은 전역 공유).
따라서 프로세스를 하나로 합치면 메모리 낭비와 프로세스 간 동시 쓰기 위험이
동시에 해결된다.

## 목표 / 비목표

**목표**
- 상주 데몬 1개로 통합하여 메모리를 ~1.5GB 수준으로, 동시 writer를 1개로 축소.
- provenance(`host_agent`/`host_session`/`host_turn`) 정확도 유지 — 여러
  클라이언트가 한 데몬에 붙어도 각 메모리가 올바른 원본 세션으로 추적되어야 함.
- 평소 프로세스 0개, 쓸 때만 기동되고 유휴 시 자동 종료.

**비목표**
- 메모리 그래프의 세션 격리(불필요, 의도적으로 전역 공유).
- 원격/네트워크 노출. 데몬은 `127.0.0.1` 루프백 전용.
- 인증/멀티유저. 단일 사용자 로컬 전제.

## 아키텍처

구성 요소 3개.

### 1. `shim.js` — stdio↔HTTP 브릿지 (호스트가 spawn, 세션당 1개, ~50MB)

호스트는 여전히 `type: "stdio"`로 `node .../shim.js`를 spawn한다. shim은 MCP
의미를 해석하지 않는 **저수준 JSON-RPC passthrough**다.

- 부팅: 데몬 헬스체크(`GET /health`) → 없으면 detached로 `daemon.js`를
  기동하고, 헬스체크가 통과할 때까지 짧게 폴링(타임아웃 존재).
- 릴레이: stdin의 JSON-RPC 라인 → 데몬 `POST /mcp`. 데몬 응답(JSON 또는 SSE)을
  stdout으로 되돌린다.
- 정체 헤더: 매 요청에 자기 env 기반 헤더를 첨부한다.
  - `X-Keymem-Host-Agent`: `claude` | `codex`
  - `X-Keymem-Host-Session`: `CLAUDE_CODE_SESSION_ID` 또는 `CODEX_THREAD_ID`
  - env가 없으면(구형 클라이언트, Claude Desktop 등) 헤더를 **생략**한다.
- 폴백: 데몬 기동/연결이 끝내 실패하면 stderr에 로그하고, **in-process stdio
  모드(현재 동작)로 degrade**한다. 즉 최악의 경우에도 기능은 오늘과 동일하고
  프로세스 수만 안 줄어든다.

경계: shim은 "바이트를 나르고 헤더를 붙이는" 한 가지 일만 한다. 툴 목록/스키마가
바뀌어도 shim은 손댈 필요가 없다.

### 2. `daemon.js` — 상주 HTTP 서버 (전역 1개, ~1.5GB)

`StreamableHTTPServerTransport` 기반. 기존 `server.ts`의 `server`/`graph`를
그대로 재사용한다.

- 기동 시 `graph.load()` 1회 + 임베딩 모델 1회 로드 → 메모리·동시성 문제의
  해결 지점.
- `POST /mcp`를 SDK 트랜스포트로 처리. `GET /health`는 기동 완료 여부를 반환.
- **hostLink는 요청 헤더에서 구성**(아래 §핵심 변경점).
- 유휴 종료: 활성 연결 0 && 마지막 요청 후 10분 경과 → self-exit.

### 3. 클라이언트 설정

`~/.claude.json`(및 codex 설정)의 `mcp-super-memory`를 `command`만
`node .../shim.js`로 교체. `type`은 그대로 `stdio`.

## 데이터 흐름

```
Claude Code   ─stdio─▶ shim(A) ─┐
Codex         ─stdio─▶ shim(B) ─┼─ HTTP + X-Keymem-* 헤더 ─▶ daemon ─▶ graph.json
Claude(터미널) ─stdio─▶ shim(C) ─┘                                    (단일 writer,
                                                                     async-mutex 유효)
```

## 핵심 변경점 (server.ts / nativeTranscripts.ts)

현재 provenance 경로:

```
server.ts:531   const hostLink = await detectHostLink();
server.ts:540   source: buildSource(parseObject(a.source), "remember", hostLink)
server.ts:74    detectHostLink() → detectActiveSession()  // Tier1 env, Tier2 mtime 추측
```

데몬에서 문제가 되는 지점과 대응:

1. **Tier 1(env)이 데몬에는 없다.** 호스트가 데몬을 spawn하지 않으므로
   `CLAUDE_CODE_SESSION_ID` 등 env가 데몬 프로세스에 존재하지 않는다. 대신
   **요청 헤더**가 그 역할을 한다. 데몬은 요청 컨텍스트에서 `X-Keymem-Host-*`를
   읽어 `{ agent, session_id }`를 얻고, 그 session_id로 transcript를 읽어
   `turn`을 계산한다.
2. **Tier 2(mtime 추측)는 데몬 경로에서 제거**한다. 여러 클라이언트가 동시에
   도는 상황에서 "가장 최근 수정된 transcript"는 오답이다. 헤더가 없으면
   `hostLink = null`로 떨어뜨린다(메모리 내용 자체는 정상 저장, 역추적 링크만
   비게 됨).
3. **요청 헤더를 tool 핸들러까지 전달**해야 한다. SDK가 tool 핸들러에
   `extra.requestInfo?.headers`(v2: `ctx.http?.req?.headers.get()`)로 요청
   헤더를 노출하므로, `detectHostLink()`를 "요청 헤더에서 hostLink 구성"하는
   함수로 대체하고 핸들러에서 헤더를 넘긴다.
4. **`transcriptAccessEnabled()` 게이트 재판단.** 이 함수는
   `CLAUDE_CODE_SESSION_ID || CODEX_THREAD_ID` env 유무로 신뢰 여부를 판단한다
   (`nativeTranscripts.ts:64-68`). 데몬에는 env가 없어 항상 false가 되어 transcript
   접근/스탬핑이 꺼진다. 데몬 경로에서는 **요청 헤더 존재 여부로 재판단**하도록
   경로를 분리한다(헤더가 있으면 그 요청은 신뢰된 호스트에서 온 것).

`SERVER_SESSION`(server.ts:48) 상수는 유지 가능하다. provenance의 `session`
필드가 "데몬 인스턴스 id"로 의미가 바뀌지만 무해하며, 호스트 세션 추적은
`host_session` 필드가 담당한다.

## 에러 처리

| 상황 | 동작 |
|---|---|
| 데몬 부재 | shim이 detached 기동 후 헬스체크 폴링 |
| 데몬 기동/연결 실패 | shim이 in-process stdio 폴백(현재 동작). 메모리는 안 줆, 기능은 보장 |
| 유휴 종료와 신규 shim 접속 레이스 | shim은 연결 실패 시 1회 재기동/재시도 |
| 헤더 없는 요청(구형 클라이언트) | hostLink=null, 메모리 내용은 정상 저장 |

## 테스트

**shim 단위**
- env 있을 때 `X-Keymem-Host-*` 헤더 주입, env 없을 때 생략.
- 데몬 부재 시 기동 트리거 및 헬스체크 폴링.
- 기동 실패 시 in-process 폴백 진입.

**daemon 단위**
- 헤더 → hostLink 매핑, 헤더 없을 때 null.
- `transcriptAccessEnabled` 재판단이 헤더 기준으로 동작.
- 동시 2연결이 같은 graph에 안전하게 write(mutex 유효성).
- 유휴 10분 self-exit 타이머.

**통합**
- 실제 shim 2개(서로 다른 host session) 동시 remember → graph.json 무결성
  + 각 메모리의 `host_session`이 올바른 세션으로 스탬프.

## 마이그레이션

1. `shim.js`, `daemon.js` 추가, `index.ts`는 in-process 폴백 진입점으로 유지.
2. `buildSource`/hostLink 경로를 헤더 기반으로 전환.
3. 빌드 후 `.claude.json`의 `command`를 shim으로 교체.
4. 기존 stdio 프로세스들은 클라이언트 재시작 시 자연 종료.
