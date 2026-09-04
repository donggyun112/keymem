import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  buildCacheEvalCase,
  compactRecallKeys,
  compareCacheVariant,
  type CacheEvalCase,
  type RecallKeyCandidate,
} from "../bench/prompt-cache-ab-lib.ts";
import { evaluateRecallCorpus } from "../bench/prompt-cache-ab.ts";

let moduleId = 0;

const fullKeys: RecallKeyCandidate[] = [
  {
    key_id: "k1",
    concept: "alpha",
    aliases: ["a"],
    key_type: "concept",
    score: 0.9,
    score_kind: "key_relevance",
    match_type: "semantic",
    memory_count: 2,
    is_hub: false,
    specificity: 0.8,
    cluster_size: 1,
    evidence: "key_match",
    suggested_tool: { name: "read_key", arguments: { key_id: "k1" } },
  },
  {
    key_id: "k2",
    concept: "beta",
    aliases: [],
    key_type: "concept",
    score: 0.8,
    score_kind: "key_relevance",
    match_type: "literal",
    memory_count: 1,
    is_hub: false,
    specificity: 1,
    cluster_size: 1,
    evidence: "key_match",
    suggested_tool: { name: "read_key", arguments: { key_id: "k2" } },
  },
];

test("compact recall keys preserve candidate identity, order, and navigation metadata", () => {
  assert.deepEqual(compactRecallKeys(fullKeys), [
    {
      key_id: "k1", concept: "alpha", aliases: ["a"], key_type: "concept", score: 0.9,
      match_type: "semantic", memory_count: 2, is_hub: false, specificity: 0.8,
    },
    {
      key_id: "k2", concept: "beta", aliases: [], key_type: "concept", score: 0.8,
      match_type: "literal", memory_count: 1, is_hub: false, specificity: 1,
    },
  ]);
});

test("ranked recall case scores top-3 loss and compact-8 preservation", () => {
  const ranked = ["k1", "k2", "k3", "k4"].map((key_id, index) => ({
    ...fullKeys[0],
    key_id,
    concept: `concept-${index + 1}`,
    score: 1 - index * 0.1,
  }));

  const item = buildCacheEvalCase("rank-four", ["k4"], ranked);

  assert.deepEqual(item.variants.top8.key_ids, ["k1", "k2", "k3", "k4"]);
  assert.deepEqual(item.variants.top3.key_ids, ["k1", "k2", "k3"]);
  assert.deepEqual(item.variants.compact8.key_ids, ["k1", "k2", "k3", "k4"]);
  assert.equal(item.variants.top8.task_score, 1);
  assert.equal(item.variants.top3.task_score, 0);
  assert.equal(item.variants.compact8.task_score, 1);
  assert.ok(item.variants.compact8.payload_bytes < item.variants.top8.payload_bytes);
});

test("top-3 fails non-inferiority when the relevant key is ranked fourth", () => {
  const cases: CacheEvalCase[] = [{
    id: "rank-four",
    expected_key_ids: ["k4"],
    variants: {
      top8: { key_ids: ["k1", "k2", "k3", "k4"], payload_bytes: 1_000, task_score: 1 },
      top3: { key_ids: ["k1", "k2", "k3"], payload_bytes: 600, task_score: 0 },
    },
  }];

  const report = compareCacheVariant(cases, "top8", "top3", {
    min_payload_reduction: 0.2,
    max_reachability_loss: 0,
    task_score_margin: 0.25,
  });

  assert.equal(report.pass, false);
  assert.deepEqual(report.failures, ["reachability", "task_quality"]);
  assert.equal(report.reachability_delta, -1);
  assert.deepEqual(report.task_score_delta_ci95, [-1, -1]);
});

test("missing paired task scores cannot prove non-inferiority", () => {
  const cases: CacheEvalCase[] = [{
    id: "unscored",
    expected_key_ids: ["k1"],
    variants: {
      top8: { key_ids: ["k1"], payload_bytes: 1_000 },
      compact8: { key_ids: ["k1"], payload_bytes: 600 },
    },
  }];

  const report = compareCacheVariant(cases, "top8", "compact8", {
    min_payload_reduction: 0.2,
    max_reachability_loss: 0,
    task_score_margin: 0.25,
  });

  assert.equal(report.pass, false);
  assert.deepEqual(report.failures, ["task_quality_unmeasured"]);
  assert.equal(report.task_score_delta_ci95, null);
});

test("compact-8 passes when it preserves candidates and paired task quality", () => {
  const cases: CacheEvalCase[] = [
    {
      id: "one",
      expected_key_ids: ["k2"],
      variants: {
        top8: { key_ids: ["k1", "k2"], payload_bytes: 1_000, task_score: 1 },
        compact8: { key_ids: ["k1", "k2"], payload_bytes: 600, task_score: 1 },
      },
    },
    {
      id: "two",
      expected_key_ids: ["k1"],
      variants: {
        top8: { key_ids: ["k1", "k2"], payload_bytes: 1_000, task_score: 0.75 },
        compact8: { key_ids: ["k1", "k2"], payload_bytes: 500, task_score: 0.75 },
      },
    },
  ];

  const report = compareCacheVariant(cases, "top8", "compact8", {
    min_payload_reduction: 0.2,
    max_reachability_loss: 0,
    task_score_margin: 0.25,
  });

  assert.equal(report.pass, true);
  assert.deepEqual(report.failures, []);
  assert.equal(report.top1_identity_rate, 1);
  assert.equal(report.reachability_delta, 0);
  assert.ok(Math.abs(report.payload_reduction - 0.45) < 1e-12);
  assert.deepEqual(report.task_score_delta_ci95, [0, 0]);
});

test("corpus evaluation rejects top-3 when a gold key ranks fourth", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-cache-ab-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  process.env.KEYMEM_SHORT_KEY_MERGE = "0";
  t.after(() => delete process.env.KEYMEM_SHORT_KEY_MERGE);

  const embedding = await import("../src/embedding.ts");
  const rankedVector = (text: string): number[] => {
    const value = text.toLowerCase();
    if (value.includes("probe")) return [1, 0, 0, 0, 0];
    if (value.includes("one")) return [0.9, 0.436, 0, 0, 0];
    if (value.includes("two")) return [0.8, 0, 0.6, 0, 0];
    if (value.includes("three")) return [0.7, 0, 0, 0.714, 0];
    if (value.includes("target")) return [0.65, 0, 0, 0, 0.76];
    return [0, 0, 0, 0, 1];
  };
  embedding.__setTestEmbedder(rankedVector);
  t.after(() => embedding.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?cache-ab=${moduleId++}`);
  const graph = new MemoryGraph();
  await graph.load();
  const result = await evaluateRecallCorpus(graph, {
    memories: [
      { id: "one", content: "memory one", keys: ["key-one"] },
      { id: "two", content: "memory two", keys: ["key-two"] },
      { id: "three", content: "memory three", keys: ["key-three"] },
      { id: "target", content: "memory target", keys: ["key-target"] },
    ],
    queries: [{ q: "probe", expect: ["target"], category: "rank-four" }],
  });

  assert.equal(result.cases[0].variants.top8.task_score, 1);
  assert.equal(result.cases[0].variants.top3.task_score, 0);
  assert.equal(result.top3.pass, false);
  assert.deepEqual(result.top3.failures, ["reachability", "task_quality"]);
  assert.equal(result.compact8.pass, true);
  assert.deepEqual(result.llm_cases[0].expected_memories, ["memory target"]);
  assert.deepEqual(result.llm_cases[0].variants.top8.map((key) => key.concept), [
    "key-one",
    "key-two",
    "key-three",
    "key-target",
  ]);
  assert.deepEqual(result.llm_cases[0].variants.compact8.map(Object.keys), [
    ["key_id", "concept", "aliases", "key_type", "score", "match_type", "memory_count", "is_hub", "specificity"],
    ["key_id", "concept", "aliases", "key_type", "score", "match_type", "memory_count", "is_hub", "specificity"],
    ["key_id", "concept", "aliases", "key_type", "score", "match_type", "memory_count", "is_hub", "specificity"],
    ["key_id", "concept", "aliases", "key_type", "score", "match_type", "memory_count", "is_hub", "specificity"],
  ]);
  const targetKeyId = result.llm_cases[0].variants.top8[3].key_id;
  assert.deepEqual(result.llm_cases[0].evidence_by_key[targetKeyId], ["memory target"]);
});
