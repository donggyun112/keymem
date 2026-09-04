import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, extname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { evaluateRecallCorpus, type RecallCorpus } from "./prompt-cache-ab.ts";
import {
  assessProviderGate,
  buildAnswerPrompt,
  buildBlindedJudgePrompt,
  buildCodexExecArgs,
  comparePairedJudgeScores,
  parseCodexJsonl,
  summarizeProviderUsage,
  unblindJudgeScores,
  validateBlindedJudgeScores,
  validateLlmAnswers,
  type BlindedJudgeScore,
  type CodexUsage,
  type LlmAnswer,
  type ParsedCodexRun,
  type PromptCacheLlmCase,
  type PromptCacheVariant,
} from "./prompt-cache-llm-lib.ts";

const ANSWER_SCHEMA = {
  type: "object",
  properties: {
    answers: {
      type: "array",
      items: {
        type: "object",
        properties: {
          id: { type: "string" },
          selected_key_ids: { type: "array", items: { type: "string" }, maxItems: 3 },
          answer: { type: "string" },
        },
        required: ["id", "selected_key_ids", "answer"],
        additionalProperties: false,
      },
    },
  },
  required: ["answers"],
  additionalProperties: false,
} as const;

const JUDGE_SCHEMA = {
  type: "object",
  properties: {
    scores: {
      type: "array",
      items: {
        type: "object",
        properties: {
          id: { type: "string" },
          A_score: { type: "integer", minimum: 0, maximum: 12 },
          B_score: { type: "integer", minimum: 0, maximum: 12 },
        },
        required: ["id", "A_score", "B_score"],
        additionalProperties: false,
      },
    },
  },
  required: ["scores"],
  additionalProperties: false,
} as const;

function chunks<T>(items: T[], size: number): T[][] {
  const result: T[][] = [];
  for (let index = 0; index < items.length; index += size) result.push(items.slice(index, index + size));
  return result;
}

async function runCodex<T>(
  prompt: string,
  schemaPath: string,
  model: string,
  cwd: string,
): Promise<ParsedCodexRun<T>> {
  const args = buildCodexExecArgs(model, schemaPath, cwd);

  return await new Promise((resolveRun, reject) => {
    const child = spawn("codex", args, { cwd, env: process.env, stdio: ["pipe", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
    }, 180_000);
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (data: string) => { stdout += data; });
    child.stderr.on("data", (data: string) => { stderr += data; });
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (timedOut) return reject(new Error("Codex evaluation timed out"));
      if (code !== 0) return reject(new Error(`Codex evaluation failed (${code}): ${stderr.trim()}`));
      try {
        resolveRun(parseCodexJsonl<T>(stdout));
      } catch (error) {
        reject(error);
      }
    });
    child.stdin.end(prompt);
  });
}

async function buildCases(corpusPath: string, dataDir: string): Promise<{
  cases: PromptCacheLlmCase[];
  retrieval: Awaited<ReturnType<typeof evaluateRecallCorpus>>["compact8"];
}> {
  const corpus = JSON.parse(await readFile(corpusPath, "utf8")) as RecallCorpus;
  process.env.KEYMEM_DATA_DIR = dataDir;
  process.env.EMBEDDING_BACKEND ??= "local";
  process.env.LOCAL_EMBEDDING_MODEL ??= "bge-m3";
  const { MemoryGraph } = await import(`../src/memoryGraph.ts?prompt-cache-llm=${Date.now()}`);
  const graph = new MemoryGraph();
  await graph.load();
  const report = await evaluateRecallCorpus(graph, corpus);
  return { cases: report.llm_cases, retrieval: report.compact8 };
}

async function main(): Promise<void> {
  const corpusPath = resolve(process.argv[2] ?? "bench/fixture.json");
  const stem = basename(corpusPath, extname(corpusPath));
  const outputPath = resolve(process.argv[3] ?? `bench/${stem}-prompt-cache-llm-results.json`);
  const model = process.env.KEYMEM_LLM_EVAL_MODEL ?? "gpt-5.6-sol";
  const batchSize = Number(process.env.KEYMEM_LLM_EVAL_BATCH ?? 4);
  if (!Number.isInteger(batchSize) || batchSize < 1 || batchSize > 10) {
    throw new Error("KEYMEM_LLM_EVAL_BATCH must be an integer from 1 to 10");
  }

  const tempRoot = await mkdtemp(join(tmpdir(), "keymem-prompt-cache-llm-"));
  const dataDir = join(tempRoot, "graph");
  const answerSchemaPath = join(tempRoot, "answer-schema.json");
  const judgeSchemaPath = join(tempRoot, "judge-schema.json");
  await writeFile(answerSchemaPath, JSON.stringify(ANSWER_SCHEMA));
  await writeFile(judgeSchemaPath, JSON.stringify(JUDGE_SCHEMA));

  try {
    const { cases, retrieval } = await buildCases(corpusPath, dataDir);
    const answerRuns: Record<PromptCacheVariant, CodexUsage[]> = { top8: [], compact8: [] };
    const answers: Record<PromptCacheVariant, LlmAnswer[]> = { top8: [], compact8: [] };

    for (const [batchIndex, batch] of chunks(cases, batchSize).entries()) {
      const order: PromptCacheVariant[] = batchIndex % 2 === 0
        ? ["top8", "compact8"]
        : ["compact8", "top8"];
      for (const variant of order) {
        const run = await runCodex<{ answers: LlmAnswer[] }>(
          buildAnswerPrompt(batch, variant),
          answerSchemaPath,
          model,
          tempRoot,
        );
        validateLlmAnswers(batch, variant, run.value.answers);
        answerRuns[variant].push(run.usage);
        answers[variant].push(...run.value.answers);
        console.error(`${stem} batch ${batchIndex + 1} ${variant}: cached ${run.usage.cached_input_tokens}/${run.usage.input_tokens}`);
      }
    }

    const blinded = buildBlindedJudgePrompt(cases, answers);
    const judgeRun = await runCodex<{ scores: BlindedJudgeScore[] }>(
      blinded.prompt,
      judgeSchemaPath,
      model,
      tempRoot,
    );
    validateBlindedJudgeScores(cases, judgeRun.value.scores);
    const pairedScores = unblindJudgeScores(judgeRun.value.scores, blinded.label_map);
    const quality = comparePairedJudgeScores(pairedScores, 0.25);
    const provider = summarizeProviderUsage(answerRuns);
    const providerGate = assessProviderGate(provider, {
      min_input_reduction: 0.02,
      max_cache_hit_loss: 0.01,
    });
    const result = {
      generated_at: new Date().toISOString(),
      corpus: corpusPath,
      model,
      cases: cases.length,
      batch_size: batchSize,
      retrieval,
      quality,
      provider,
      provider_runs: answerRuns,
      provider_gate: providerGate,
      judge_usage: judgeRun.usage,
      paired_scores: pairedScores,
      answers,
      runtime_rollout_allowed: retrieval.pass && quality.pass && providerGate.pass,
    };
    await writeFile(outputPath, `${JSON.stringify(result, null, 2)}\n`);
    console.log(JSON.stringify({ output: outputPath, ...result }, null, 2));
  } finally {
    await rm(tempRoot, { recursive: true, force: true });
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
