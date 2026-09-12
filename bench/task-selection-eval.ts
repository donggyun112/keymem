// Production-policy A/B: broad graph activation is identical in both arms; only the
// opt-in task-conditioned projection at the evidence-consumption boundary changes.
//
//   pnpm exec tsx bench/task-selection-eval.ts [N] [output.json] [start]
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { analyzeTaskEvidence } from "../src/evidenceSelection.ts";

process.env.EMBEDDING_BACKEND ??= "local";
process.env.LOCAL_EMBEDDING_MODEL ??= "bge-m3";
process.env.KEYMEM_RERANK ??= "false";
process.env.KEYMEM_TASK_EVIDENCE_SELECTION = "false";

type Row = {
  id: string;
  question: string;
  answer: string;
  type: "bridge" | "comparison";
  support: string[];
  titles: string[];
  paras: string[];
};
type Hit = { id: string };
type Metrics = { recall: number; both: number; ndcg: number; pairMrr: number; answerCoverage: number };
type Agg = Metrics & { n: number };
type Arm = "baseline" | "selected";
type Scenario = "recall5" | "inject5" | "inject2";
type Category = "bridge" | "comparison" | "all";
type Split = "dev" | "holdout" | "full";

const input = JSON.parse(await readFile(resolve("bench/hotpot-slice.json"), "utf8")) as Row[];
const start = Math.max(0, Math.min(input.length - 1, Number(process.argv[4]) || 0));
const requested = Math.max(1, Math.min(input.length - start, Number(process.argv[2]) || input.length));
const rows = input.slice(start, start + requested);
const outputPath = resolve(process.argv[3] || "bench/task-selection-eval-results.json");
const scenarios: Scenario[] = ["recall5", "inject5", "inject2"];
const arms: Arm[] = ["baseline", "selected"];
const categories: Category[] = ["bridge", "comparison", "all"];
const splits: Split[] = ["dev", "holdout", "full"];
const zero = (): Agg => ({ n: 0, recall: 0, both: 0, ndcg: 0, pairMrr: 0, answerCoverage: 0 });
const aggregates = Object.fromEntries(
  splits.map((split) => [
    split,
    Object.fromEntries(
      scenarios.map((scenario) => [
        scenario,
        Object.fromEntries(arms.map((arm) => [arm, Object.fromEntries(categories.map((c) => [c, zero()]))])),
      ])
    ),
  ])
) as Record<Split, Record<Scenario, Record<Arm, Record<Category, Agg>>>>;

const normalize = (value: string) => value.normalize("NFKC").toLocaleLowerCase().replace(/\s+/g, " ").trim();
const timed = async <T>(fn: () => Promise<T>): Promise<{ value: T; ms: number }> => {
  const started = performance.now();
  const value = await fn();
  return { value, ms: performance.now() - started };
};
const score = (titles: string[], row: Row, limit: number): Metrics => {
  const selected = titles.slice(0, limit);
  const support = [...new Set(row.support)];
  const ranks = support.map((title) => selected.indexOf(title) + 1).filter((rank) => rank > 0);
  const recall = support.length ? ranks.length / support.length : 0;
  const both = support.length > 0 && ranks.length === support.length ? 1 : 0;
  const dcg = selected.reduce(
    (sum, title, index) => sum + (support.includes(title) ? 1 / Math.log2(index + 2) : 0),
    0
  );
  const ideal = Array.from({ length: Math.min(limit, support.length) }, (_, index) => 1 / Math.log2(index + 2))
    .reduce((sum, value) => sum + value, 0);
  const pairMrr = both ? 1 / Math.max(...ranks) : 0;
  const answer = normalize(row.answer);
  const answerCoverage =
    answer && selected.some((title) => normalize(row.paras[row.titles.indexOf(title)] ?? "").includes(answer)) ? 1 : 0;
  return { recall, both, ndcg: ideal ? dcg / ideal : 0, pairMrr, answerCoverage };
};
const add = (split: Split, scenario: Scenario, arm: Arm, category: Category, value: Metrics) => {
  const agg = aggregates[split][scenario][arm][category];
  agg.n++;
  for (const metric of ["recall", "both", "ndcg", "pairMrr", "answerCoverage"] as const) {
    agg[metric] += value[metric];
  }
};
const rowSplits = (index: number): Split[] => (index < 120 ? ["dev", "full"] : ["holdout", "full"]);

const tempRoot = await mkdtemp(join(tmpdir(), "keymem-task-selection-"));
process.env.KEYMEM_DATA_DIR = tempRoot;
const { MemoryGraph } = await import("../src/memoryGraph.ts?task-selection-eval");
const perQuery: Array<Record<string, unknown>> = [];
try {
  for (let index = 0; index < rows.length; index++) {
    const row = rows[index];
    const absoluteIndex = start + index;
    const graph = new MemoryGraph();
    // The benchmark is read-only after construction; avoiding persistence makes the
    // 300 isolated per-question graphs fast without changing retrieval behavior.
    graph.save = async () => {};
    const idToTitle = new Map<string, string>();
    for (let i = 0; i < row.titles.length; i++) {
      const title = row.titles[i];
      const content = row.paras[i] ?? "";
      const lower = content.toLocaleLowerCase();
      const keys = [
        title,
        ...row.titles.filter((candidate) => candidate !== title && lower.includes(candidate.toLocaleLowerCase())),
      ];
      const [id] = await graph.add(content, keys, {
        keyTypes: Object.fromEntries(keys.map((candidate) => [candidate, "proper_noun"])),
      });
      idToTitle.set(id, title);
    }

    const recall = async (enabled: boolean) =>
      ((await graph.recall(row.question, 5, null, true, 2, 0, 0, 0, 0, 0, false, null, enabled)) as Hit[])
        .map((hit) => idToTitle.get(hit.id))
        .filter((title): title is string => Boolean(title));
    const inject = async (topK: number, enabled: boolean) =>
      ((await graph.recallInject(row.question, topK, null, { taskSelection: enabled })).memories as Hit[])
        .map((hit) => idToTitle.get(hit.id))
        .filter((title): title is string => Boolean(title));

    // A full-key analysis is a safe over-approximation of the production pool analysis.
    // If it is null, the selector cannot activate on any subset, so the selected arm is
    // provably byte-identical and we avoid three redundant embedding calls on bridge rows.
    const controllerEligible = Boolean(analyzeTaskEvidence(row.question, Object.values(graph.keys)));
    const recall5Baseline = await timed(() => recall(false));
    const recall5Selected = controllerEligible
      ? await timed(() => recall(true))
      : { value: [...recall5Baseline.value], ms: recall5Baseline.ms };
    const inject5Baseline = await timed(() => inject(5, false));
    const inject5Selected = controllerEligible
      ? await timed(() => inject(5, true))
      : { value: [...inject5Baseline.value], ms: inject5Baseline.ms };
    const inject2Baseline = controllerEligible
      ? await timed(() => inject(2, false))
      : { value: inject5Baseline.value.slice(0, 2), ms: inject5Baseline.ms };
    const inject2Selected = controllerEligible
      ? await timed(() => inject(2, true))
      : { value: inject5Selected.value.slice(0, 2), ms: inject5Selected.ms };
    const outputs: Record<Scenario, Record<Arm, string[]>> = {
      recall5: { baseline: recall5Baseline.value, selected: recall5Selected.value },
      inject5: { baseline: inject5Baseline.value, selected: inject5Selected.value },
      inject2: { baseline: inject2Baseline.value, selected: inject2Selected.value },
    };
    const latencyMs: Record<Scenario, Record<Arm, number>> = {
      recall5: { baseline: recall5Baseline.ms, selected: recall5Selected.ms },
      inject5: { baseline: inject5Baseline.ms, selected: inject5Selected.ms },
      inject2: { baseline: inject2Baseline.ms, selected: inject2Selected.ms },
    };
    const metrics = {} as Record<Scenario, Record<Arm, Metrics>>;
    for (const scenario of scenarios) {
      metrics[scenario] = {} as Record<Arm, Metrics>;
      const limit = scenario === "inject2" ? 2 : 5;
      for (const arm of arms) {
        const value = score(outputs[scenario][arm], row, limit);
        metrics[scenario][arm] = value;
        for (const split of rowSplits(absoluteIndex)) {
          add(split, scenario, arm, row.type, value);
          add(split, scenario, arm, "all", value);
        }
      }
    }
    perQuery.push({
      index: absoluteIndex,
      split: absoluteIndex < 120 ? "dev" : "holdout",
      id: row.id,
      type: row.type,
      question: row.question,
      answer: row.answer,
      support: row.support,
      controllerEligible,
      outputs,
      metrics,
      latencyMs,
    });
    if ((index + 1) % 10 === 0) console.log(`processed ${index + 1}/${rows.length}`);
  }
} finally {
  await rm(tempRoot, { recursive: true, force: true });
}

// Guardrail: ordinary associative/direct/not-found queries must be byte-identical because
// none are comparison-shaped. This catches accidental projection outside the intended task.
const fixture = JSON.parse(await readFile(resolve("bench/assoc-fixture.json"), "utf8")) as {
  memories: Array<{ id: string; content: string; keys: string[]; key_types?: Record<string, string> }>;
  queries: Array<{ q: string; category: string }>;
};
const fixtureGraph = new MemoryGraph();
fixtureGraph.save = async () => {};
for (const memory of fixture.memories) {
  await fixtureGraph.add(memory.content, memory.keys, { keyTypes: memory.key_types });
}
const fixtureChecks: Array<Record<string, unknown>> = [];
for (const query of fixture.queries) {
  const baseline = (await fixtureGraph.recall(query.q, 5, null, true, 2, 0, 0, 0, 0, 0, false, null, false) as Hit[]).map((x) => x.id);
  const selected = (await fixtureGraph.recall(query.q, 5, null, true, 2, 0, 0, 0, 0, 0, false, null, true) as Hit[]).map((x) => x.id);
  const injectBaseline = (await fixtureGraph.recallInject(query.q, 2, null, { taskSelection: false })).memories as Hit[];
  const injectSelected = (await fixtureGraph.recallInject(query.q, 2, null, { taskSelection: true })).memories as Hit[];
  fixtureChecks.push({
    query: query.q,
    category: query.category,
    recallIdentical: JSON.stringify(baseline) === JSON.stringify(selected),
    injectIdentical: JSON.stringify(injectBaseline.map((x) => x.id)) === JSON.stringify(injectSelected.map((x) => x.id)),
  });
}

const changedBridge = perQuery.filter((item) => {
  if (item.type !== "bridge") return false;
  const outputs = item.outputs as Record<Scenario, Record<Arm, string[]>>;
  return scenarios.some((scenario) => JSON.stringify(outputs[scenario].baseline) !== JSON.stringify(outputs[scenario].selected));
});
const paired = Object.fromEntries(
  scenarios.map((scenario) => [
    scenario,
    Object.fromEntries(
      categories.slice(0, 2).map((category) => {
        const rowsForCategory = perQuery.filter((item) => item.type === category);
        let wins = 0;
        let losses = 0;
        let orderChanged = 0;
        for (const item of rowsForCategory) {
          const output = item.outputs as Record<Scenario, Record<Arm, string[]>>;
          const metric = item.metrics as Record<Scenario, Record<Arm, Metrics>>;
          if (metric[scenario].selected.both > metric[scenario].baseline.both) wins++;
          if (metric[scenario].selected.both < metric[scenario].baseline.both) losses++;
          if (JSON.stringify(output[scenario].selected) !== JSON.stringify(output[scenario].baseline)) orderChanged++;
        }
        return [category, { wins, losses, orderChanged }];
      })
    ),
  ])
);

const result = {
  model: "bge-m3",
  reranker: false,
  N: rows.length,
  start,
  devN: rows.filter((_, index) => start + index < 120).length,
  holdoutN: rows.filter((_, index) => start + index >= 120).length,
  aggregates,
  paired,
  latency: Object.fromEntries(
    scenarios.map((scenario) => [
      scenario,
      Object.fromEntries(
        arms.map((arm) => [
          arm,
          Object.fromEntries(
            (["bridge", "comparison"] as const).map((category) => {
              const samples = perQuery
                .filter((item) => item.type === category)
                .map((item) => (item.latencyMs as Record<Scenario, Record<Arm, number>>)[scenario][arm])
                .sort((a, b) => a - b);
              const mean = samples.reduce((sum, value) => sum + value, 0) / Math.max(1, samples.length);
              const p95 = samples[Math.min(samples.length - 1, Math.floor(samples.length * 0.95))] ?? 0;
              return [category, { meanMs: mean, p95Ms: p95 }];
            })
          ),
        ])
      ),
    ])
  ),
  fixtureChecks,
  changedBridge,
  perQuery,
};
await writeFile(outputPath, JSON.stringify(result, null, 2));

const pct = (value: number) => `${(value * 100).toFixed(1)}%`;
for (const split of splits) {
  if (!aggregates[split].recall5.baseline.all.n) continue;
  console.log(`\n${split.toUpperCase()}`);
  for (const scenario of scenarios) {
    for (const category of ["bridge", "comparison"] as const) {
      const b = aggregates[split][scenario].baseline[category];
      const s = aggregates[split][scenario].selected[category];
      if (!b.n) continue;
      console.log(
        `${scenario.padEnd(8)} ${category.padEnd(10)} n=${b.n} both ${pct(b.both / b.n)}→${pct(s.both / s.n)} ` +
        `recall ${pct(b.recall / b.n)}→${pct(s.recall / s.n)} ndcg ${(b.ndcg / b.n).toFixed(3)}→${(s.ndcg / s.n).toFixed(3)} ` +
        `pairMRR ${(b.pairMrr / b.n).toFixed(3)}→${(s.pairMrr / s.n).toFixed(3)} answer ${pct(b.answerCoverage / b.n)}→${pct(s.answerCoverage / s.n)}`
      );
    }
  }
}
console.log("paired:", paired);
console.log(`bridge queries with any order change: ${changedBridge.length}`);
console.log(`fixture identical: ${fixtureChecks.filter((item) => item.recallIdentical && item.injectIdentical).length}/${fixtureChecks.length}`);
console.log(`results -> ${outputPath}`);
