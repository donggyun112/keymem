// Tests whether HotpotQA comparison regressions are activation failures or
// downstream selection failures. GRAPH activation is held fixed; only the
// top-k evidence-selection policy changes.
//
//   pnpm exec tsx bench/evidence-projection.ts [N] [output.json]
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { projectComparisonEvidence } from "./evidence-projection-lib.ts";

process.env.EMBEDDING_BACKEND ??= "local";
process.env.LOCAL_EMBEDDING_MODEL ??= "bge-m3";
// Isolate associative activation from the optional cross-encoder.
process.env.KEYMEM_RERANK ??= "false";

type Row = {
  id: string;
  question: string;
  type: "bridge" | "comparison";
  support: string[];
  titles: string[];
  paras: string[];
};
type RecallHit = {
  id: string;
  score: number;
  relevance_score: number;
  hop: number;
  matched_via: string[];
};
type Evidence = RecallHit & { title: string };
type Score = { recall: number; both: boolean };
type Agg = { n: number; recall: number; both: number };

const all = JSON.parse(await readFile(resolve("bench/hotpot-slice.json"), "utf-8")) as Row[];
const requestedN = Number(process.argv[2]) || 120;
const rows = all.slice(0, requestedN);
const outputPath = resolve(process.argv[3] || "bench/evidence-projection-results.json");
const TOPK = 5;
const ACTIVATION_K = TOPK * 2;
const CONDITIONS = ["DIRECT", "GRAPH_GLOBAL", "TASK_PROJECTED"] as const;
const mk = (): Agg => ({ n: 0, recall: 0, both: 0 });
const aggregates: Record<string, Record<string, Agg>> = {};
for (const condition of CONDITIONS) {
  aggregates[condition] = { bridge: mk(), comparison: mk(), all: mk() };
}
const activation: Record<string, Agg> = { bridge: mk(), comparison: mk(), all: mk() };
const perQuery: Array<Record<string, unknown>> = [];
const diagnostics = {
  comparisonN: 0,
  projectionApplied: 0,
  directToGraphRegressions: 0,
  graphTop5Failures: 0,
  recoverableSelectionFailures: 0,
  activationFailures: 0,
  projectedRecoveries: 0,
  projectionRegressions: 0,
};

const score = (evidence: Evidence[], support: string[], limit = TOPK): Score => {
  const titles = evidence.slice(0, limit).map((item) => item.title);
  const found = support.filter((gold) => titles.includes(gold)).length;
  return { recall: found / support.length, both: found === support.length };
};
const add = (condition: string, category: string, value: Score) => {
  for (const key of [category, "all"]) {
    const agg = aggregates[condition][key];
    agg.n++;
    agg.recall += value.recall;
    agg.both += value.both ? 1 : 0;
  }
};
const addActivation = (category: string, value: Score) => {
  for (const key of [category, "all"]) {
    const agg = activation[key];
    agg.n++;
    agg.recall += value.recall;
    agg.both += value.both ? 1 : 0;
  }
};

const dir = await mkdtemp(join(tmpdir(), "km-evidence-projection-"));
let done = 0;
try {
  for (const row of rows) {
    process.env.KEYMEM_DATA_DIR = await mkdtemp(join(dir, "q-"));
    const mg = await import(`../src/memoryGraph.ts?projection=${done}`);
    const graph = new mg.MemoryGraph();
    await graph.load();

    const idToTitle: Record<string, string> = {};
    for (let i = 0; i < row.titles.length; i++) {
      const title = row.titles[i];
      const content = row.paras[i] ?? "";
      const lowerContent = content.toLowerCase();
      const keys = [
        title,
        ...row.titles.filter(
          (candidate) => candidate !== title && lowerContent.includes(candidate.toLowerCase())
        ),
      ];
      const [id] = await graph.add(content, keys, {});
      idToTitle[id] = title;
    }

    // Pure reads make the two conditions order-independent. GRAPH returns 2*topK
    // internally, which is the fixed activation pool used by both selectors.
    const recall = async (expand: boolean): Promise<Evidence[]> => {
      const hits = (await graph.recall(
        row.question,
        TOPK,
        null,
        expand,
        expand ? 2 : 1,
        0,
        0,
        0,
        0,
        0,
        false
      )) as RecallHit[];
      return hits
        .map((hit) => ({ ...hit, title: idToTitle[hit.id] }))
        .filter((hit) => Boolean(hit.title));
    };

    const direct = await recall(false);
    const graphPool = await recall(true);
    // Use HotpotQA's task label to isolate evidence-selection quality from a
    // separate intent-classification problem. Bridge rows are an unchanged
    // negative control; a production controller would have to infer this type.
    const projection =
      row.type === "comparison"
        ? projectComparisonEvidence(row.question, row.titles, graphPool, TOPK)
        : { applied: false, entities: [], selected: graphPool.slice(0, TOPK) };
    const directScore = score(direct, row.support);
    const graphScore = score(graphPool, row.support);
    const projectedScore = score(projection.selected, row.support);
    const activationScore = score(graphPool, row.support, ACTIVATION_K);

    add("DIRECT", row.type, directScore);
    add("GRAPH_GLOBAL", row.type, graphScore);
    add("TASK_PROJECTED", row.type, projectedScore);
    addActivation(row.type, activationScore);

    if (row.type === "comparison") {
      diagnostics.comparisonN++;
      if (projection.applied) diagnostics.projectionApplied++;
      if (directScore.both && !graphScore.both) diagnostics.directToGraphRegressions++;
      if (!graphScore.both) {
        diagnostics.graphTop5Failures++;
        if (activationScore.both) diagnostics.recoverableSelectionFailures++;
        else diagnostics.activationFailures++;
        if (projectedScore.both) diagnostics.projectedRecoveries++;
      }
      if (graphScore.both && !projectedScore.both) diagnostics.projectionRegressions++;
    }

    perQuery.push({
      id: row.id,
      type: row.type,
      question: row.question,
      support: row.support,
      entities: projection.entities,
      projectionApplied: projection.applied,
      direct: direct.slice(0, TOPK).map((x) => x.title),
      graphGlobal: graphPool.slice(0, TOPK).map((x) => x.title),
      graphActivationPool: graphPool.slice(0, ACTIVATION_K).map((x) => x.title),
      taskProjected: projection.selected.map((x) => x.title),
      scores: {
        direct: directScore,
        graphGlobal: graphScore,
        activation: activationScore,
        taskProjected: projectedScore,
      },
    });
    done++;
    if (done % 10 === 0) console.log(`processed ${done}/${rows.length}`);
  }
} finally {
  await rm(dir, { recursive: true, force: true });
}

const pct = (value: number) => `${(value * 100).toFixed(0)}%`;
console.log(`\nKeyMem task-conditioned evidence projection — model=bge-m3, N=${done}`);
console.log(`activation=GRAPH pool@${ACTIVATION_K}; selection=global vs entity-balanced top@${TOPK}`);
console.log("─".repeat(88));
console.log("category        n  metric             DIRECT  GRAPH_GLOBAL  TASK_PROJECTED  GRAPH_POOL");
for (const category of ["bridge", "comparison", "all"]) {
  const get = (condition: string) => aggregates[condition][category];
  const n = get("DIRECT").n;
  if (!n) continue;
  console.log(
    `${category.padEnd(15)} ${String(n).padStart(3)}  support-recall@5  ${CONDITIONS.map((c) => pct(get(c).recall / n).padStart(12)).join("  ")}  ${pct(activation[category].recall / n).padStart(10)}`
  );
  console.log(
    `${"".padEnd(19)}  both@5 / both@10  ${CONDITIONS.map((c) => pct(get(c).both / n).padStart(12)).join("  ")}  ${pct(activation[category].both / n).padStart(10)}`
  );
}
console.log("─".repeat(88));
console.log("comparison diagnostics:", diagnostics);

await writeFile(
  outputPath,
  JSON.stringify(
    {
      model: "bge-m3",
      reranker: false,
      reinforce: false,
      N: done,
      topK: TOPK,
      activationK: ACTIVATION_K,
      conditions: CONDITIONS,
      aggregates,
      activation,
      diagnostics,
      perQuery,
    },
    null,
    2
  )
);
console.log(`results -> ${outputPath}`);
