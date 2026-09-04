import assert from "node:assert/strict";
import test from "node:test";

import {
  buildAnswerPrompt,
  buildBlindedJudgePrompt,
  buildCodexExecArgs,
  assessProviderGate,
  comparePairedJudgeScores,
  parseCodexJsonl,
  summarizeProviderUsage,
  unblindJudgeScores,
  validateBlindedJudgeScores,
  validateLlmAnswers,
  type LlmAnswer,
  type PromptCacheLlmCase,
} from "../bench/prompt-cache-llm-lib.ts";

const llmCase: PromptCacheLlmCase = {
  id: "q0-direct",
  query: "Where does Mina live?",
  expected_memories: ["미나는 서울 마포구에 산다"],
  variants: {
    top8: [{
      key_id: "k-live",
      concept: "거주지",
      score: 0.9,
      match_type: "semantic",
      memory_count: 1,
      aliases: ["사는곳"],
      specificity: 1,
      suggested_tool: { name: "read_key" },
    }],
    compact8: [{
      key_id: "k-live",
      concept: "거주지",
      score: 0.9,
      match_type: "semantic",
      memory_count: 1,
    }],
  },
  evidence_by_key: {
    "k-live": ["미나는 서울 마포구에 산다"],
  },
};

test("answer prompts keep an identical stable prefix and vary only the candidate payload", () => {
  const top8 = buildAnswerPrompt([llmCase], "top8");
  const compact8 = buildAnswerPrompt([llmCase], "compact8");
  const marker = "\n<evaluation_cases>\n";

  assert.equal(top8.slice(0, top8.indexOf(marker)), compact8.slice(0, compact8.indexOf(marker)));
  assert.match(top8, /suggested_tool/);
  assert.doesNotMatch(compact8, /suggested_tool/);
  assert.match(compact8, /\"key_id\":\"k-live\"/);
});

test("Codex JSONL parsing returns the structured answer and provider usage", () => {
  const output = [
    JSON.stringify({ type: "thread.started", thread_id: "thread-1" }),
    JSON.stringify({
      type: "item.completed",
      item: {
        type: "agent_message",
        text: JSON.stringify({ answers: [{ id: "q0-direct", selected_key_ids: ["k-live"], answer: "마포구" }] }),
      },
    }),
    JSON.stringify({
      type: "turn.completed",
      usage: {
        input_tokens: 15_000,
        cached_input_tokens: 12_000,
        cache_write_input_tokens: 250,
        output_tokens: 80,
        reasoning_output_tokens: 20,
      },
    }),
  ].join("\n");

  const parsed = parseCodexJsonl<{ answers: unknown[] }>(output);

  assert.equal(parsed.thread_id, "thread-1");
  assert.equal(parsed.value.answers.length, 1);
  assert.deepEqual(parsed.usage, {
    input_tokens: 15_000,
    cached_input_tokens: 12_000,
    cache_write_input_tokens: 250,
    output_tokens: 80,
    reasoning_output_tokens: 20,
  });
});

test("provider summary uses weighted cache hit rate and uncached input totals", () => {
  const report = summarizeProviderUsage({
    top8: [
      { input_tokens: 10_000, cached_input_tokens: 8_000, cache_write_input_tokens: 0, output_tokens: 50, reasoning_output_tokens: 10 },
      { input_tokens: 20_000, cached_input_tokens: 17_000, cache_write_input_tokens: 0, output_tokens: 60, reasoning_output_tokens: 20 },
    ],
    compact8: [
      { input_tokens: 9_000, cached_input_tokens: 8_000, cache_write_input_tokens: 0, output_tokens: 45, reasoning_output_tokens: 10 },
      { input_tokens: 18_000, cached_input_tokens: 17_000, cache_write_input_tokens: 0, output_tokens: 55, reasoning_output_tokens: 20 },
    ],
  });

  assert.equal(report.top8.cache_hit_rate, 25_000 / 30_000);
  assert.equal(report.top8.uncached_input_tokens, 5_000);
  assert.equal(report.compact8.cache_hit_rate, 25_000 / 27_000);
  assert.equal(report.compact8.uncached_input_tokens, 2_000);
  assert.equal(report.uncached_input_reduction, 0.6);
});

test("provider summary isolates batch pairs with the same nonzero warm-cache prefix", () => {
  const report = summarizeProviderUsage({
    top8: [
      { input_tokens: 18_000, cached_input_tokens: 0, cache_write_input_tokens: 0, output_tokens: 1, reasoning_output_tokens: 0 },
      { input_tokens: 17_000, cached_input_tokens: 8_000, cache_write_input_tokens: 0, output_tokens: 1, reasoning_output_tokens: 0 },
      { input_tokens: 16_000, cached_input_tokens: 10_000, cache_write_input_tokens: 0, output_tokens: 1, reasoning_output_tokens: 0 },
    ],
    compact8: [
      { input_tokens: 16_000, cached_input_tokens: 0, cache_write_input_tokens: 0, output_tokens: 1, reasoning_output_tokens: 0 },
      { input_tokens: 15_000, cached_input_tokens: 0, cache_write_input_tokens: 0, output_tokens: 1, reasoning_output_tokens: 0 },
      { input_tokens: 14_000, cached_input_tokens: 10_000, cache_write_input_tokens: 0, output_tokens: 1, reasoning_output_tokens: 0 },
    ],
  });

  assert.equal(report.paired_warm.pairs, 1);
  assert.equal(report.paired_warm.top8.uncached_input_tokens, 6_000);
  assert.equal(report.paired_warm.compact8.uncached_input_tokens, 4_000);
  assert.ok(Math.abs(report.paired_warm.uncached_input_reduction - 1 / 3) < 1e-12);
});

test("paired judge gate rejects a confidence interval below the 0.25-point margin", () => {
  const report = comparePairedJudgeScores([
    { id: "q0", top8_score: 12, compact8_score: 12 },
    { id: "q1", top8_score: 12, compact8_score: 11 },
    { id: "q2", top8_score: 10, compact8_score: 10 },
    { id: "q3", top8_score: 11, compact8_score: 10 },
  ], 0.25);

  assert.equal(report.pass, false);
  assert.equal(report.mean_delta, -0.5);
  assert.ok(report.ci95[0] < -0.25);
});

test("paired judge gate passes identical answer quality", () => {
  const report = comparePairedJudgeScores([
    { id: "q0", top8_score: 12, compact8_score: 12 },
    { id: "q1", top8_score: 10, compact8_score: 10 },
  ], 0.25);

  assert.deepEqual(report, {
    pass: true,
    margin: 0.25,
    mean_delta: 0,
    ci95: [0, 0],
    cases: 2,
  });
});

test("judge prompt alternates blind labels and unblinds scores correctly", () => {
  const secondCase: PromptCacheLlmCase = { ...llmCase, id: "q1-direct", query: "미나는 어디 살아?" };
  const answers: Record<"top8" | "compact8", LlmAnswer[]> = {
    top8: [
      { id: "q0-direct", selected_key_ids: ["k-live"], answer: "Mapo-gu" },
      { id: "q1-direct", selected_key_ids: ["k-live"], answer: "서울 마포구" },
    ],
    compact8: [
      { id: "q0-direct", selected_key_ids: ["k-live"], answer: "Seoul" },
      { id: "q1-direct", selected_key_ids: ["k-live"], answer: "마포구" },
    ],
  };

  const blinded = buildBlindedJudgePrompt([llmCase, secondCase], answers);

  assert.doesNotMatch(blinded.prompt, /top8|compact8/);
  assert.deepEqual(blinded.label_map, {
    "q0-direct": { A: "top8", B: "compact8" },
    "q1-direct": { A: "compact8", B: "top8" },
  });
  assert.match(blinded.prompt, /미나는 서울 마포구에 산다/);
  assert.deepEqual(unblindJudgeScores([
    { id: "q0-direct", A_score: 12, B_score: 11 },
    { id: "q1-direct", A_score: 9, B_score: 10 },
  ], blinded.label_map), [
    { id: "q0-direct", top8_score: 12, compact8_score: 11 },
    { id: "q1-direct", top8_score: 10, compact8_score: 9 },
  ]);
});

test("answer validation rejects selected keys outside the supplied recall candidates", () => {
  assert.throws(() => validateLlmAnswers([llmCase], "compact8", [{
    id: "q0-direct",
    selected_key_ids: ["invented-key"],
    answer: "마포구",
  }]), /unknown selected key/);
});

test("judge validation requires exactly one bounded score pair per case", () => {
  assert.throws(() => validateBlindedJudgeScores([llmCase], [{
    id: "q0-direct",
    A_score: 13,
    B_score: 12,
  }]), /invalid A score/);
  assert.throws(() => validateBlindedJudgeScores([llmCase], []), /missing judge score/);
});

test("Codex execution preserves normal user config so provider prompt caching remains active", () => {
  const args = buildCodexExecArgs("gpt-5.6-sol", "/tmp/schema.json", "/tmp/run");

  assert.equal(args.includes("--ignore-user-config"), false);
  assert.equal(args.includes("--ignore-rules"), false);
  assert.deepEqual(args.slice(-3), ["--cd", "/tmp/run", "-"]);
});

test("provider gate rejects one-sided cache observations even when payload input shrinks", () => {
  const usage = summarizeProviderUsage({
    top8: [{ input_tokens: 18_000, cached_input_tokens: 10_000, cache_write_input_tokens: 0, output_tokens: 1, reasoning_output_tokens: 0 }],
    compact8: [{ input_tokens: 16_000, cached_input_tokens: 0, cache_write_input_tokens: 0, output_tokens: 1, reasoning_output_tokens: 0 }],
  });

  const gate = assessProviderGate(usage, { min_input_reduction: 0.02, max_cache_hit_loss: 0.01 });
  assert.equal(gate.pass, false);
  assert.deepEqual(gate.failures, ["paired_warm_cache_unavailable"]);
  assert.ok(Math.abs(gate.input_reduction - 2_000 / 18_000) < 1e-12);
  assert.equal(gate.cache_hit_rate_delta, -(10_000 / 18_000));
});

test("provider gate passes balanced warm-cache measurements with lower compact input", () => {
  const usage = summarizeProviderUsage({
    top8: [{ input_tokens: 18_000, cached_input_tokens: 10_000, cache_write_input_tokens: 0, output_tokens: 1, reasoning_output_tokens: 0 }],
    compact8: [{ input_tokens: 16_000, cached_input_tokens: 10_000, cache_write_input_tokens: 0, output_tokens: 1, reasoning_output_tokens: 0 }],
  });

  assert.equal(assessProviderGate(usage, { min_input_reduction: 0.02, max_cache_hit_loss: 0.01 }).pass, true);
});
