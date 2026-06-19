// Quantifies the coining-dependence hypothesis AND whether key merge defends against it.
//
// Same facts are stored three ways:
//   specific — discriminative unique keys (good coining)
//   generic  — broad common-noun keys shared across facts (hub dilution; "LLM stores blind")
//   synonym  — each fact coins a *variant* term for its concept; the query uses a different
//              variant (tests whether semantic search + key MERGE bridge the gap)
//
// Retrieval is the agentic path: searchKeys -> readKey, building a path-ordered candidate
// list. Metric = navigation cost: rank of the target memory in that list (how many memories
// the agent must wade through). found@1/@5, MRR, avgCost, plus key count (merge shrinks it).
//
// The synonym arm runs with merge ON (default) and OFF (KEY_MERGE=1.0) to isolate merge.
//
//   npm run bench:keynav
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

process.env.EMBEDDING_BACKEND ??= "local";
process.env.LOCAL_EMBEDDING_MODEL ??= "bge-m3";

const FACTS = [
  { id: "drink", content: "미나는 매일 아이스 라떼를 마신다",            specific: ["아이스라떼", "음료"],     generic: ["취향", "일상"], synonym: ["마실것"] },
  { id: "food",  content: "미나가 좋아하는 음식은 떡볶이와 파스타다",     specific: ["떡볶이", "음식"],         generic: ["취향", "일상"], synonym: ["먹는것"] },
  { id: "music", content: "미나는 재즈 피아노를 듣는다",                 specific: ["재즈", "음악"],           generic: ["취향", "일상"], synonym: ["듣는것"] },
  { id: "pet",   content: "미나는 강아지 보리를 키운다",                 specific: ["강아지", "반려동물"],      generic: ["일상", "정보"], synonym: ["키우는것"] },
  { id: "job",   content: "미나는 핀테크 스타트업의 백엔드 엔지니어다",   specific: ["직업", "엔지니어"],        generic: ["정보", "일상"], synonym: ["하는일"] },
  { id: "live",  content: "미나는 서울 마포구에 산다",                   specific: ["거주지", "마포구"],        generic: ["정보", "일상"], synonym: ["사는곳"] },
  { id: "lang",  content: "미나는 Go와 Rust로 코딩한다",                specific: ["프로그래밍언어", "Go"],     generic: ["정보", "일상"], synonym: ["코딩"] },
  { id: "hobby", content: "미나는 주말에 클라이밍을 한다",               specific: ["취미", "클라이밍"],        generic: ["취향", "일상"], synonym: ["여가"] },
];
const QUERIES = [
  { q: "미나가 마시는 음료",                target: "drink" },
  { q: "미나가 좋아하는 음식",              target: "food" },
  { q: "미나가 듣는 음악",                  target: "music" },
  { q: "미나가 키우는 동물",                target: "pet" },
  { q: "미나의 직업",                       target: "job" },
  { q: "미나가 사는 곳",                    target: "live" },
  { q: "미나가 쓰는 프로그래밍 언어",        target: "lang" },
  { q: "미나의 취미",                       target: "hobby" },
];

const pct = (x: number) => `${(x * 100).toFixed(0)}%`;
const fmt = (x: number) => (x === Infinity ? "  ∞ " : x.toFixed(2));

function targetGid(g: any, factId: string): string {
  const content = FACTS.find((f) => f.id === factId)!.content;
  for (const [mid, m] of Object.entries(g.memories as Record<string, { content: string }>)) {
    if (m.content === content) return mid;
  }
  return "";
}

// Simulate searchKeys -> readKey, returning the rank of the target in the path-ordered list.
async function keynavRank(g: any, query: string, tg: string): Promise<number> {
  const keys = (await g.searchKeys(query, 5)) as Array<{ key_id: string }>;
  const seen: string[] = [];
  for (const k of keys) {
    const res = g.readKey(k.key_id, { limit: 10 }) as { memories: Array<{ memory_id: string }> };
    for (const m of res.memories) if (!seen.includes(m.memory_id)) seen.push(m.memory_id);
  }
  return seen.indexOf(tg) + 1; // 0 = not found
}

function score(ranks: number[]) {
  const found = ranks.filter((r) => r > 0);
  return {
    f1: ranks.filter((r) => r === 1).length / ranks.length,
    f5: ranks.filter((r) => r >= 1 && r <= 5).length / ranks.length,
    mrr: ranks.reduce((s, r) => s + (r > 0 ? 1 / r : 0), 0) / ranks.length,
    avgCost: found.length ? found.reduce((s, r) => s + r, 0) / found.length : Infinity,
    foundN: found.length,
  };
}

async function contentRank(g: any, query: string, tg: string): Promise<number> {
  const res = (await g.recall(query, 10)) as Array<{ id: string }>;
  return res.findIndex((m) => m.id === tg) + 1;
}

async function runArm(label: string, keyOf: (f: typeof FACTS[number]) => string[], tag: string) {
  const dir = await mkdtemp(join(tmpdir(), `sm-knv-${tag}-`));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  const mg = await import(`../src/memoryGraph.ts?knv=${tag}`);
  const g = new mg.MemoryGraph();
  await g.load();
  for (const f of FACTS) await g.add(f.content, keyOf(f), {});
  const knv: number[] = [];
  const cnt: number[] = [];
  for (const { q, target } of QUERIES) {
    const tg = targetGid(g, target);
    knv.push(await keynavRank(g, q, tg));
    cnt.push(await contentRank(g, q, tg));
  }
  const k = score(knv);
  const c = score(cnt);
  const keyCount = Object.keys(g.keys).length;
  await rm(dir, { recursive: true, force: true });
  return (
    `${label.padEnd(22)} keynav  found@1=${pct(k.f1)} found@5=${pct(k.f5)} MRR=${k.mrr.toFixed(2)} cost=${fmt(k.avgCost)} keys=${keyCount} (${k.foundN}/${QUERIES.length})\n` +
    `${" ".repeat(22)} content found@1=${pct(c.f1)} found@5=${pct(c.f5)} MRR=${c.mrr.toFixed(2)} cost=${fmt(c.avgCost)}          (${c.foundN}/${QUERIES.length})`
  );
}

const rows: string[] = [];
rows.push(await runArm("specific (good)", (f) => f.specific, "spec"));
rows.push(await runArm("generic (hubs)", (f) => f.generic, "gen"));
rows.push(await runArm("synonym, merge ON", (f) => f.synonym, "synon"));
process.env.SUPER_MEMORY_KEY_MERGE = "1.0"; // exact-match only -> merge effectively OFF
rows.push(await runArm("synonym, merge OFF", (f) => f.synonym, "synoff"));
delete process.env.SUPER_MEMORY_KEY_MERGE;

console.log(`\nkey-coining dependence + merge defense — model=${process.env.LOCAL_EMBEDDING_MODEL}`);
console.log("avgCost = mean rank of target in the searchKeys->readKey path (lower=better)");
console.log("─".repeat(92));
for (const r of rows) console.log(r);
console.log("─".repeat(92));
