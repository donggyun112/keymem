// real-eval.ts — recall hit-rate eval over REAL stored facts.
// Run against a COPY of the live store, never the live one (readMemory mutates depth):
//   cp ~/.super-memory/graph.json <scratch>/graph.json
//   KEYMEM_DATA_DIR=<scratch> EMBEDDING_BACKEND=local LOCAL_EMBEDDING_MODEL=bge-m3 npx tsx bench/real-eval.ts
import { MemoryGraph } from "../src/memoryGraph.js";

type Case = { query: string; expect: string; ns?: string | null };
// expect = substring that must appear in the hit memory's content.
// Mix of keyword-style and sentence-style queries over facts known to exist in the
// owner's store; this measures the real workload (Korean personal/project facts),
// which the HotpotQA benches do not.
const CASES: Case[] = [
  { query: "커밋 서명 규칙", expect: "서명" },
  { query: "recall 적중률", expect: "적중률", ns: "keymem" },
  { query: "벤치 데이터 오염", expect: "HotpotQA", ns: "keymem" },
  { query: "임베딩 모델", expect: "bge-m3" },
  { query: "Nexora suspend", expect: "suspend", ns: "Nexora" },
  { query: "CodeCanvas MCP 전환", expect: "MCP" },
  { query: "arcmemory 커넥터 큐", expect: "커넥터", ns: "arcmemory" },
  { query: "사용자가 어떤 임베딩 백엔드를 쓰는지", expect: "bge-m3" },
  { query: "recall 적중률이 낮은 이유", expect: "오염", ns: "keymem" },
  { query: "Nexora 원자적 resume 커밋", expect: "fcad585", ns: "Nexora" },
  // Sub-fact queries against a multi-fact release-note memory: the whole-content
  // vector dilutes below the gate; only sentence-level max-sim can reach these.
  { query: "단일언어 경고", expect: "writeHints", ns: "keymem" },
  { query: "no_match 힌트", expect: "nearest_keys", ns: "keymem" },
];

async function main() {
  const g = new MemoryGraph();
  await g.load();
  let recallHits = 0;
  let keyHits = 0;
  for (const c of CASES) {
    const ns = c.ns ?? null;
    const mems = (await g.recall(
      c.query, 5, ns, false, 2, 0, undefined, undefined, undefined, 0, false
    )) as Array<{ content: string }>;
    const rHit = mems.some((m) => m.content.includes(c.expect));

    const keys = (await g.searchKeys(c.query, 8, ns)) as Array<{ key_id: string }>;
    let kHit = false;
    for (const k of keys.slice(0, 3)) {
      const page = (await g.readKey(k.key_id, { query: c.query, namespace: ns, limit: 5 })) as {
        memories: Array<{ memory_id: string }>;
      };
      for (const h of page.memories) {
        const full = (await g
          .readMemory(h.memory_id, null, ns)
          .catch(() => null)) as { memory?: { content?: string } } | null;
        if (full?.memory?.content?.includes(c.expect)) { kHit = true; break; }
      }
      if (kHit) break;
    }
    recallHits += rHit ? 1 : 0;
    keyHits += kHit ? 1 : 0;
    console.log(`${rHit ? "R✓" : "R✗"} ${kHit ? "K✓" : "K✗"}  ${c.query}`);
  }
  console.log(`\nrecall() hit@5: ${recallHits}/${CASES.length}   searchKeys→readKey hit: ${keyHits}/${CASES.length}`);
}

main().then(() => process.exit(0));
