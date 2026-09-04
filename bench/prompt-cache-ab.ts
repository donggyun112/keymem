import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import {
  buildCacheEvalCase,
  compareCacheVariant,
  compactRecallKeys,
  type CacheComparisonReport,
  type CacheEvalCase,
  type RecallKeyCandidate,
} from "./prompt-cache-ab-lib.ts";
import {
  type PromptCacheLlmCase,
} from "./prompt-cache-llm-lib.ts";

type MemoryFixture = {
  id: string;
  content: string;
  keys: string[];
  key_types?: Record<string, string>;
};

type QueryFixture = {
  q: string;
  expect: string[];
  category: string;
};

export type RecallCorpus = {
  memories: MemoryFixture[];
  queries: QueryFixture[];
};

type GraphForCacheEval = {
  add(content: string, keys: string[], options?: { keyTypes?: Record<string, string> | null }): Promise<string[]>;
  browseKeys(namespace: string | null, options?: { limit?: number }): Promise<unknown>;
  searchKeys(query: string, topK?: number, namespace?: string | null): Promise<unknown[]>;
};

export interface RecallCorpusReport {
  cases: CacheEvalCase[];
  llm_cases: PromptCacheLlmCase[];
  top3: CacheComparisonReport;
  compact8: CacheComparisonReport;
}

const THRESHOLDS = {
  min_payload_reduction: 0.2,
  max_reachability_loss: 0,
  task_score_margin: 0,
};

function normalized(value: string): string {
  return value.trim().toLowerCase();
}

export async function evaluateRecallCorpus(
  graph: GraphForCacheEval,
  corpus: RecallCorpus,
): Promise<RecallCorpusReport> {
  for (const memory of corpus.memories) {
    await graph.add(memory.content, memory.keys, { keyTypes: memory.key_types ?? null });
  }

  const browsed = await graph.browseKeys(null, { limit: 100 }) as {
    keys: Array<{ key_id: string; concept: string; aliases?: string[]; learned_aliases?: string[] }>;
  };
  const keyIdByTerm = new Map<string, string>();
  for (const key of browsed.keys) {
    for (const term of [key.concept, ...(key.aliases ?? []), ...(key.learned_aliases ?? [])]) {
      keyIdByTerm.set(normalized(term), key.key_id);
    }
  }
  const memoryById = new Map(corpus.memories.map((memory) => [memory.id, memory]));
  const evidenceByKeyId = new Map<string, string[]>();
  for (const memory of corpus.memories) {
    for (const key of memory.keys) {
      const keyId = keyIdByTerm.get(normalized(key));
      if (!keyId) continue;
      const evidence = evidenceByKeyId.get(keyId) ?? [];
      if (!evidence.includes(memory.content)) evidence.push(memory.content);
      evidenceByKeyId.set(keyId, evidence);
    }
  }

  const cases: CacheEvalCase[] = [];
  const llmCases: PromptCacheLlmCase[] = [];
  for (let index = 0; index < corpus.queries.length; index++) {
    const query = corpus.queries[index];
    const expectedKeyIds = [...new Set(query.expect.flatMap((memoryId) => {
      const memory = memoryById.get(memoryId);
      if (!memory) throw new Error(`query ${index} references unknown memory ${memoryId}`);
      return memory.keys.map((key) => keyIdByTerm.get(normalized(key))).filter((id): id is string => Boolean(id));
    }))];
    if (query.expect.length > 0 && expectedKeyIds.length === 0) {
      throw new Error(`query ${index} has no resolvable expected keys`);
    }
    const ranked = await graph.searchKeys(query.q, 8, null) as RecallKeyCandidate[];
    cases.push(buildCacheEvalCase(`q${index}-${query.category}`, expectedKeyIds, ranked));
    const top8 = ranked.slice(0, 8);
    llmCases.push({
      id: `q${index}-${query.category}`,
      query: query.q,
      expected_memories: query.expect.map((memoryId) => memoryById.get(memoryId)!.content),
      variants: {
        top8,
        compact8: compactRecallKeys(top8),
      },
      evidence_by_key: Object.fromEntries(top8.map((key) => [
        key.key_id,
        evidenceByKeyId.get(key.key_id) ?? [],
      ])),
    });
  }

  return {
    cases,
    llm_cases: llmCases,
    top3: compareCacheVariant(cases, "top8", "top3", THRESHOLDS),
    compact8: compareCacheVariant(cases, "top8", "compact8", THRESHOLDS),
  };
}

async function main(): Promise<void> {
  const corpusPath = resolve(process.argv[2] ?? "bench/fixture.json");
  const corpus = JSON.parse(await readFile(corpusPath, "utf-8")) as RecallCorpus;
  const dataDir = await mkdtemp(join(tmpdir(), "keymem-prompt-cache-ab-"));
  process.env.KEYMEM_DATA_DIR = dataDir;
  process.env.EMBEDDING_BACKEND ??= "local";
  process.env.LOCAL_EMBEDDING_MODEL ??= "bge-m3";
  try {
    const { MemoryGraph } = await import(`../src/memoryGraph.ts?prompt-cache-ab=${Date.now()}`);
    const graph = new MemoryGraph();
    await graph.load();
    const report = await evaluateRecallCorpus(graph, corpus);
    console.log(JSON.stringify({
      corpus: corpusPath,
      queries: report.cases.length,
      top3: report.top3,
      compact8: report.compact8,
    }, null, 2));
  } finally {
    await rm(dataDir, { recursive: true, force: true });
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
