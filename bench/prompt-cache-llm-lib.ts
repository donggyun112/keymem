import type { RecallKeyCandidate } from "./prompt-cache-ab-lib.ts";

export type PromptCacheVariant = "top8" | "compact8";

export interface PromptCacheLlmCase {
  id: string;
  query: string;
  expected_memories: string[];
  variants: Record<PromptCacheVariant, RecallKeyCandidate[]>;
  evidence_by_key: Record<string, string[]>;
}

export interface CodexUsage {
  input_tokens: number;
  cached_input_tokens: number;
  cache_write_input_tokens: number;
  output_tokens: number;
  reasoning_output_tokens: number;
}

export interface ParsedCodexRun<T> {
  thread_id: string;
  value: T;
  usage: CodexUsage;
}

export interface PairedJudgeScore {
  id: string;
  top8_score: number;
  compact8_score: number;
}

export interface LlmAnswer {
  id: string;
  selected_key_ids: string[];
  answer: string;
}

export interface BlindedJudgeScore {
  id: string;
  A_score: number;
  B_score: number;
}

export type BlindLabelMap = Record<string, Record<"A" | "B", PromptCacheVariant>>;

export interface PairedJudgeReport {
  pass: boolean;
  margin: number;
  mean_delta: number;
  ci95: [number, number];
  cases: number;
}

export function buildCodexExecArgs(model: string, schemaPath: string, cwd: string): string[] {
  return [
    "exec",
    "--ephemeral",
    "--json",
    "--color", "never",
    "--sandbox", "read-only",
    "--skip-git-repo-check",
    "--model", model,
    "--config", 'model_reasoning_effort="low"',
    "--output-schema", schemaPath,
    "--cd", cwd,
    "-",
  ];
}

const ANSWER_PROMPT_PREFIX = `You are evaluating whether a compressed memory-recall payload preserves answer quality.
Do not use tools or outside knowledge. Work only from the evaluation cases below.
For every case, choose up to three recall candidate key IDs that are useful for the user query.
Answer only from memory records attached to those selected key IDs. If the records do not support an answer, say that the information is unavailable.
Preserve the user's language. Keep each answer concise and factual. Do not mention this benchmark, candidate metadata, key IDs, or memory records in the answer.
Return one result for every case, in the same order, using the required JSON schema.`;

export function buildAnswerPrompt(
  cases: PromptCacheLlmCase[],
  variant: PromptCacheVariant,
): string {
  const payload = cases.map((item) => ({
    id: item.id,
    user_query: item.query,
    recall_candidates: item.variants[variant],
    memory_records_by_key: item.evidence_by_key,
  }));
  return `${ANSWER_PROMPT_PREFIX}\n<evaluation_cases>\n${JSON.stringify(payload)}\n</evaluation_cases>`;
}

const JUDGE_PROMPT_PREFIX = `You are a strict, blind evaluator of memory-grounded answers.
Do not use tools or outside knowledge. Score each answer independently against only the user query and reference facts.
Use this 12-point rubric: factual correctness 0-6, completeness for the query 0-3, grounding without unsupported claims 0-2, and correct abstention when reference facts are absent 0-1.
An answer may use different wording or language from the reference facts. Give no preference to answer A or B, length, style, or order.
Return one score pair for every case using the required JSON schema.`;

export function buildBlindedJudgePrompt(
  cases: PromptCacheLlmCase[],
  answers: Record<PromptCacheVariant, LlmAnswer[]>,
): { prompt: string; label_map: BlindLabelMap } {
  const answerMaps = {
    top8: new Map(answers.top8.map((answer) => [answer.id, answer])),
    compact8: new Map(answers.compact8.map((answer) => [answer.id, answer])),
  };
  const labelMap: BlindLabelMap = {};
  const payload = cases.map((item, index) => {
    const top8 = answerMaps.top8.get(item.id);
    const compact8 = answerMaps.compact8.get(item.id);
    if (!top8 || !compact8) throw new Error(`missing answer pair for ${item.id}`);
    const labels: Record<"A" | "B", PromptCacheVariant> = index % 2 === 0
      ? { A: "top8", B: "compact8" }
      : { A: "compact8", B: "top8" };
    labelMap[item.id] = labels;
    const byVariant = { top8, compact8 };
    return {
      id: item.id,
      user_query: item.query,
      reference_facts: item.expected_memories,
      answer_A: byVariant[labels.A].answer,
      answer_B: byVariant[labels.B].answer,
    };
  });
  return {
    prompt: `${JUDGE_PROMPT_PREFIX}\n<evaluation_cases>\n${JSON.stringify(payload)}\n</evaluation_cases>`,
    label_map: labelMap,
  };
}

export function unblindJudgeScores(
  scores: BlindedJudgeScore[],
  labelMap: BlindLabelMap,
): PairedJudgeScore[] {
  return scores.map((score) => {
    const labels = labelMap[score.id];
    if (!labels) throw new Error(`missing blind label map for ${score.id}`);
    const byLabel = { A: score.A_score, B: score.B_score };
    return {
      id: score.id,
      top8_score: labels.A === "top8" ? byLabel.A : byLabel.B,
      compact8_score: labels.A === "compact8" ? byLabel.A : byLabel.B,
    };
  });
}

function indexUniqueById<T extends { id: string }>(items: T[], label: string): Map<string, T> {
  const indexed = new Map<string, T>();
  for (const item of items) {
    if (!item.id || indexed.has(item.id)) throw new Error(`${label} has an invalid or duplicate id`);
    indexed.set(item.id, item);
  }
  return indexed;
}

export function validateLlmAnswers(
  cases: PromptCacheLlmCase[],
  variant: PromptCacheVariant,
  answers: LlmAnswer[],
): void {
  const indexed = indexUniqueById(answers, "answer output");
  if (indexed.size !== cases.length) throw new Error("answer output case count does not match input");
  for (const item of cases) {
    const answer = indexed.get(item.id);
    if (!answer) throw new Error(`missing answer for ${item.id}`);
    if (!Array.isArray(answer.selected_key_ids) || answer.selected_key_ids.length > 3) {
      throw new Error(`${item.id} has invalid selected keys`);
    }
    const allowed = new Set(item.variants[variant].map((key) => key.key_id));
    for (const keyId of answer.selected_key_ids) {
      if (typeof keyId !== "string" || !allowed.has(keyId)) {
        throw new Error(`${item.id} has unknown selected key ${String(keyId)}`);
      }
    }
    if (typeof answer.answer !== "string" || answer.answer.trim().length === 0) {
      throw new Error(`${item.id} has an empty answer`);
    }
  }
}

export function validateBlindedJudgeScores(
  cases: PromptCacheLlmCase[],
  scores: BlindedJudgeScore[],
): void {
  const indexed = indexUniqueById(scores, "judge output");
  for (const item of cases) {
    const score = indexed.get(item.id);
    if (!score) throw new Error(`missing judge score for ${item.id}`);
    for (const [label, value] of [["A", score.A_score], ["B", score.B_score]] as const) {
      if (!Number.isInteger(value) || value < 0 || value > 12) {
        throw new Error(`${item.id} has invalid ${label} score`);
      }
    }
  }
  if (indexed.size !== cases.length) throw new Error("judge output contains an unexpected case");
}

function nonNegativeInteger(value: unknown, field: string): number {
  if (!Number.isInteger(value) || (value as number) < 0) {
    throw new Error(`invalid Codex usage field: ${field}`);
  }
  return value as number;
}

export function parseCodexJsonl<T>(output: string): ParsedCodexRun<T> {
  let threadId: string | undefined;
  let value: T | undefined;
  let usage: CodexUsage | undefined;

  for (const line of output.split("\n").map((item) => item.trim()).filter(Boolean)) {
    const event = JSON.parse(line) as Record<string, unknown>;
    if (event.type === "thread.started" && typeof event.thread_id === "string") {
      threadId = event.thread_id;
    }
    if (event.type === "item.completed") {
      const item = event.item as { type?: string; text?: string } | undefined;
      if (item?.type === "agent_message" && typeof item.text === "string") {
        value = JSON.parse(item.text) as T;
      }
    }
    if (event.type === "turn.completed") {
      const raw = event.usage as Record<string, unknown> | undefined;
      if (!raw) throw new Error("Codex turn completed without usage");
      usage = {
        input_tokens: nonNegativeInteger(raw.input_tokens, "input_tokens"),
        cached_input_tokens: nonNegativeInteger(raw.cached_input_tokens, "cached_input_tokens"),
        cache_write_input_tokens: nonNegativeInteger(raw.cache_write_input_tokens ?? 0, "cache_write_input_tokens"),
        output_tokens: nonNegativeInteger(raw.output_tokens, "output_tokens"),
        reasoning_output_tokens: nonNegativeInteger(raw.reasoning_output_tokens ?? 0, "reasoning_output_tokens"),
      };
    }
  }

  if (!threadId) throw new Error("Codex JSONL is missing thread.started");
  if (value === undefined) throw new Error("Codex JSONL is missing a structured agent message");
  if (!usage) throw new Error("Codex JSONL is missing turn.completed usage");
  return { thread_id: threadId, value, usage };
}

function sumUsage(runs: CodexUsage[]) {
  if (runs.length === 0) throw new Error("at least one provider run is required");
  const totals = runs.reduce((sum, item) => ({
    input_tokens: sum.input_tokens + item.input_tokens,
    cached_input_tokens: sum.cached_input_tokens + item.cached_input_tokens,
    cache_write_input_tokens: sum.cache_write_input_tokens + item.cache_write_input_tokens,
    output_tokens: sum.output_tokens + item.output_tokens,
    reasoning_output_tokens: sum.reasoning_output_tokens + item.reasoning_output_tokens,
  }), {
    input_tokens: 0,
    cached_input_tokens: 0,
    cache_write_input_tokens: 0,
    output_tokens: 0,
    reasoning_output_tokens: 0,
  });
  return {
    ...totals,
    requests: runs.length,
    uncached_input_tokens: totals.input_tokens - totals.cached_input_tokens,
    cache_hit_rate: totals.input_tokens === 0 ? 0 : totals.cached_input_tokens / totals.input_tokens,
  };
}

function emptyUsageSummary() {
  return {
    input_tokens: 0,
    cached_input_tokens: 0,
    cache_write_input_tokens: 0,
    output_tokens: 0,
    reasoning_output_tokens: 0,
    requests: 0,
    uncached_input_tokens: 0,
    cache_hit_rate: 0,
  };
}

export function summarizeProviderUsage(runs: Record<PromptCacheVariant, CodexUsage[]>) {
  const top8 = sumUsage(runs.top8);
  const compact8 = sumUsage(runs.compact8);
  const pairedWarm = runs.top8.flatMap((top8Run, index) => {
    const compact8Run = runs.compact8[index];
    return compact8Run
      && top8Run.cached_input_tokens > 0
      && top8Run.cached_input_tokens === compact8Run.cached_input_tokens
      ? [{ top8: top8Run, compact8: compact8Run }]
      : [];
  });
  const pairedTop8 = pairedWarm.length > 0 ? sumUsage(pairedWarm.map((pair) => pair.top8)) : emptyUsageSummary();
  const pairedCompact8 = pairedWarm.length > 0 ? sumUsage(pairedWarm.map((pair) => pair.compact8)) : emptyUsageSummary();
  return {
    top8,
    compact8,
    cache_hit_rate_delta: compact8.cache_hit_rate - top8.cache_hit_rate,
    uncached_input_reduction: top8.uncached_input_tokens === 0
      ? 0
      : 1 - compact8.uncached_input_tokens / top8.uncached_input_tokens,
    paired_warm: {
      pairs: pairedWarm.length,
      top8: pairedTop8,
      compact8: pairedCompact8,
      cache_hit_rate_delta: pairedCompact8.cache_hit_rate - pairedTop8.cache_hit_rate,
      uncached_input_reduction: pairedTop8.uncached_input_tokens === 0
        ? 0
        : 1 - pairedCompact8.uncached_input_tokens / pairedTop8.uncached_input_tokens,
    },
  };
}

export interface ProviderGateOptions {
  min_input_reduction: number;
  max_cache_hit_loss: number;
}

export function assessProviderGate(
  usage: ReturnType<typeof summarizeProviderUsage>,
  options: ProviderGateOptions,
) {
  const measured = usage.paired_warm.pairs > 0 ? usage.paired_warm : usage;
  const inputReduction = measured.top8.input_tokens === 0
    ? 0
    : 1 - measured.compact8.input_tokens / measured.top8.input_tokens;
  const cacheHitRateDelta = measured.cache_hit_rate_delta;
  const failures: string[] = [];
  if (usage.paired_warm.pairs === 0) {
    failures.push("paired_warm_cache_unavailable");
  } else {
    if (inputReduction < options.min_input_reduction) failures.push("input_reduction");
    if (cacheHitRateDelta < -options.max_cache_hit_loss) failures.push("cache_hit_rate");
  }
  return {
    pass: failures.length === 0,
    failures,
    input_reduction: inputReduction,
    cache_hit_rate_delta: cacheHitRateDelta,
  };
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function pairedBootstrapCI95(deltas: number[], iterations = 10_000): [number, number] {
  if (deltas.length === 1 || deltas.every((delta) => delta === deltas[0])) {
    return [deltas[0], deltas[0]];
  }
  let state = 0x5eed1234;
  const next = (): number => {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    return state / 0x1_0000_0000;
  };
  const samples: number[] = [];
  for (let i = 0; i < iterations; i++) {
    let total = 0;
    for (let j = 0; j < deltas.length; j++) {
      total += deltas[Math.floor(next() * deltas.length)];
    }
    samples.push(total / deltas.length);
  }
  samples.sort((a, b) => a - b);
  const percentile = (p: number): number => samples[Math.floor((samples.length - 1) * p)];
  return [percentile(0.025), percentile(0.975)];
}

export function comparePairedJudgeScores(
  scores: PairedJudgeScore[],
  margin: number,
): PairedJudgeReport {
  if (scores.length === 0) throw new Error("at least one paired judge score is required");
  if (!Number.isFinite(margin) || margin < 0) throw new Error("judge margin must be non-negative");
  for (const score of scores) {
    for (const [name, value] of [["top8", score.top8_score], ["compact8", score.compact8_score]] as const) {
      if (!Number.isFinite(value) || value < 0 || value > 12) {
        throw new Error(`${score.id} has invalid ${name} score`);
      }
    }
  }
  const deltas = scores.map((score) => score.compact8_score - score.top8_score);
  const ci95 = pairedBootstrapCI95(deltas);
  return {
    pass: ci95[0] >= -margin,
    margin,
    mean_delta: mean(deltas),
    ci95,
    cases: scores.length,
  };
}
