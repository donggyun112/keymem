// Does _embedSentences (per-sentence embeddings, extra write-time model calls) actually
// rescue anything the whole-content centroid embedding misses, for a genuinely multi-sentence
// memory? npm run bench showed identical scores with it on/off -- but that fixture is all
// single-sentence facts, so it never exercises the mechanism at all.
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

process.env.EMBEDDING_BACKEND ??= "local";
process.env.LOCAL_EMBEDDING_MODEL ??= "bge-m3";

const multiSentence =
  "미나는 이번 주에 회사 프로젝트 마감 때문에 매일 야근을 했다. " +
  "그러다 스트레스가 쌓여서 주말에 한라산 등산을 다녀왔다. " +
  "산 정상에서 우연히 대학 동창을 만나 반갑게 인사를 나눴다.";
const query = "미나가 산에서 누구를 만났어";

async function run(sentenceVectorsEnabled: boolean) {
  const dir = await mkdtemp(join(tmpdir(), `km-sentvec-${sentenceVectorsEnabled}-`));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.KEYMEM_SENTENCE_VECTORS = sentenceVectorsEnabled ? "1" : "0";
  const { MemoryGraph } = await import(`../src/memoryGraph.ts?sentvec=${sentenceVectorsEnabled}`);
  const g: any = new MemoryGraph();
  await g.load();
  const addStart = performance.now();
  await g.add(multiSentence, ["등산", "동창"]);
  console.log(`sentenceVectors=${sentenceVectorsEnabled}: add() took ${(performance.now() - addStart).toFixed(1)}ms for a 3-sentence memory`);
  // A few short unrelated distractors so this isn't trivially the only memory.
  await g.add("미나는 아침에 커피를 마셨다", ["커피"]);
  await g.add("미나는 저녁에 산책을 했다", ["산책"]);

  const results = await g.recall(query, 3, null, false, 1);
  const top = results[0] as { content: string; relevance_score: number } | undefined;
  console.log(`sentenceVectors=${sentenceVectorsEnabled}: top match rank1 relevance=${top?.relevance_score}`);
  console.log(`  content: ${top?.content?.slice(0, 40)}...`);
  await rm(dir, { recursive: true, force: true });
}

await run(true);
await run(false);
