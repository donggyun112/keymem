export { compactRecallKeys } from "../src/recallView.ts";
export type { CompactRecallKey, RecallKeyCandidate } from "../src/recallView.ts";
import { compactRecallKeys } from "../src/recallView.ts";
import type { RecallKeyCandidate } from "../src/recallView.ts";

export interface CacheVariantObservation {
  key_ids: string[];
  payload_bytes: number;
  task_score?: number;
}

export interface CacheEvalCase {
  id: string;
  expected_key_ids: string[];
  variants: Record<string, CacheVariantObservation>;
}

function objectiveTaskScore(keyIds: string[], expectedKeyIds: string[]): number {
  if (expectedKeyIds.length === 0) return keyIds.length === 0 ? 1 : 0;
  return expectedKeyIds.some((keyId) => keyIds.includes(keyId)) ? 1 : 0;
}

export function buildCacheEvalCase(
  id: string,
  expectedKeyIds: string[],
  rankedKeys: RecallKeyCandidate[],
): CacheEvalCase {
  const top8 = rankedKeys.slice(0, 8);
  const top3 = top8.slice(0, 3);
  const compact8 = compactRecallKeys(top8);
  const observation = (keys: Array<{ key_id: string }>, payload: unknown): CacheVariantObservation => {
    const keyIds = keys.map((key) => key.key_id);
    return {
      key_ids: keyIds,
      payload_bytes: Buffer.byteLength(JSON.stringify(payload)),
      task_score: objectiveTaskScore(keyIds, expectedKeyIds),
    };
  };
  return {
    id,
    expected_key_ids: expectedKeyIds,
    variants: {
      top8: observation(top8, top8),
      top3: observation(top3, top3),
      compact8: observation(compact8, compact8),
    },
  };
}

export interface CacheComparisonOptions {
  min_payload_reduction: number;
  max_reachability_loss: number;
  task_score_margin: number;
}

export interface CacheComparisonReport {
  pass: boolean;
  failures: string[];
  top1_identity_rate: number;
  control_reachability: number;
  candidate_reachability: number;
  reachability_delta: number;
  payload_reduction: number;
  task_score_delta: number | null;
  task_score_delta_ci95: [number, number] | null;
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function percentile(sorted: number[], p: number): number {
  return sorted[Math.floor((sorted.length - 1) * p)];
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
  return [percentile(samples, 0.025), percentile(samples, 0.975)];
}

function reachable(observation: CacheVariantObservation, expected: string[]): boolean {
  return expected.some((keyId) => observation.key_ids.includes(keyId));
}

export function compareCacheVariant(
  cases: CacheEvalCase[],
  controlName: string,
  candidateName: string,
  options: CacheComparisonOptions,
): CacheComparisonReport {
  if (cases.length === 0) throw new Error("at least one evaluation case is required");

  const pairs = cases.map((item) => {
    const control = item.variants[controlName];
    const candidate = item.variants[candidateName];
    if (!control || !candidate) {
      throw new Error(`case ${item.id} is missing ${!control ? controlName : candidateName}`);
    }
    return { item, control, candidate };
  });

  const top1Identity = mean(pairs.map(({ control, candidate }) =>
    control.key_ids[0] === candidate.key_ids[0] ? 1 : 0));
  const answerable = pairs.filter(({ item }) => item.expected_key_ids.length > 0);
  const controlReachability = answerable.length === 0 ? 1 : mean(answerable.map(({ item, control }) =>
    reachable(control, item.expected_key_ids) ? 1 : 0));
  const candidateReachability = answerable.length === 0 ? 1 : mean(answerable.map(({ item, candidate }) =>
    reachable(candidate, item.expected_key_ids) ? 1 : 0));
  const reachabilityDelta = candidateReachability - controlReachability;

  const controlBytes = pairs.reduce((sum, { control }) => sum + control.payload_bytes, 0);
  const candidateBytes = pairs.reduce((sum, { candidate }) => sum + candidate.payload_bytes, 0);
  const payloadReduction = controlBytes === 0 ? 0 : 1 - candidateBytes / controlBytes;

  const scored = pairs.every(({ control, candidate }) =>
    typeof control.task_score === "number" && typeof candidate.task_score === "number");
  const deltas = scored
    ? pairs.map(({ control, candidate }) => candidate.task_score! - control.task_score!)
    : [];
  const taskScoreDelta = scored ? mean(deltas) : null;
  const taskScoreCI = scored ? pairedBootstrapCI95(deltas) : null;

  const failures: string[] = [];
  if (top1Identity < 1) failures.push("top1_identity");
  if (reachabilityDelta < -options.max_reachability_loss) failures.push("reachability");
  if (payloadReduction < options.min_payload_reduction) failures.push("payload_reduction");
  if (!scored) failures.push("task_quality_unmeasured");
  else if (taskScoreCI![0] < -options.task_score_margin) failures.push("task_quality");

  return {
    pass: failures.length === 0,
    failures,
    top1_identity_rate: top1Identity,
    control_reachability: controlReachability,
    candidate_reachability: candidateReachability,
    reachability_delta: reachabilityDelta,
    payload_reduction: payloadReduction,
    task_score_delta: taskScoreDelta,
    task_score_delta_ci95: taskScoreCI,
  };
}
