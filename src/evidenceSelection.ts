import type { Key } from "./types.js";

export type EvidenceSelectionCandidate<T = unknown> = {
  id: string;
  keys: Key[];
  utility: number;
  value: T;
};

export type EvidenceSelectionResult<T> = {
  applied: boolean;
  entities: string[];
  selected: EvidenceSelectionCandidate<T>[];
};

const EN_COMPARISON =
  /\b(?:vs\.?|versus|both|compare|compared|comparison|difference|differ|earlier|later|first)\b/iu;
const EN_OR_QUESTION = /\b(?:which|who)\b[^?]*\bor\b/iu;
const EN_SAME_PAIR = /\b(?:are|were|do|did)\b[^?]*\band\b[^?]*\bsame\b/iu;
const EN_THAN_COMPARISON = /\b(?:higher|lower|older|younger|more|less)\b[^?]*\bthan\b/iu;
const KO_COMPARISON =
  /(?:둘\s*다|(?:^|\s)중(?:\s|$)|보다(?:\s|$)|비교|차이|동일|같(?:은|다|나요|습니까)|먼저|나중|누가|어느|더\s|높(?:은|다|나요|습니까)|낮(?:은|다|나요|습니까)|크(?:다|거나|나요|습니까)|커(?:요|나요|습니까)?|작(?:은|다|나요|습니까))/u;
const DIRECT_SUFFICIENCY_MARGIN = 0.02;

type Mention = {
  key: Key;
  label: string;
  start: number;
  end: number;
  exact: boolean;
};

function normalize(value: string): string {
  return value
    .normalize("NFKC")
    .toLocaleLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function comparisonCue(query: string): boolean {
  return (
    EN_COMPARISON.test(query) ||
    EN_OR_QUESTION.test(query) ||
    EN_SAME_PAIR.test(query) ||
    EN_THAN_COMPARISON.test(query) ||
    KO_COMPARISON.test(query)
  );
}

function terminalAlias(value: string): string | null {
  const stripped = value.replace(/\s*\([^()]+\)\s*$/u, "").trim();
  return stripped && stripped !== value.trim() ? stripped : null;
}

function boundaryOK(text: string, start: number, end: number): boolean {
  const before = start === 0 ? "" : text[start - 1];
  const after = end >= text.length ? "" : text[end];
  const word = /[\p{L}\p{N}]/u;
  if (before && word.test(before)) return false;
  if (!after || !word.test(after)) return true;
  // Korean postpositions attach directly to entity names.
  return /^(?:과|와|은|는|이|가|을|를|의|보다|중)(?:\s|$)/u.test(text.slice(end));
}

function mentions(query: string, keys: Key[]): Mention[] {
  const normalizedQuery = normalize(query);
  const found: Mention[] = [];
  for (const key of keys) {
    if (key.key_type !== "name" && key.key_type !== "proper_noun") continue;
    const authored = [key.concept, ...(key.aliases ?? [])];
    const forms = new Map<string, boolean>();
    for (const raw of authored) {
      const exact = normalize(raw);
      if (exact) forms.set(exact, true);
      const shortened = terminalAlias(raw);
      if (shortened) {
        const value = normalize(shortened);
        if (value && !forms.has(value)) forms.set(value, false);
      }
    }
    for (const [form, exact] of forms) {
      let from = 0;
      while (from <= normalizedQuery.length - form.length) {
        const start = normalizedQuery.indexOf(form, from);
        if (start < 0) break;
        const end = start + form.length;
        if (boundaryOK(normalizedQuery, start, end)) {
          found.push({ key, label: key.concept, start, end, exact });
        }
        from = start + Math.max(1, form.length);
      }
    }
  }

  // Prefer authored exact names at an ambiguous span. Two shortened parenthetical
  // aliases at the same span are not enough evidence to infer an entity.
  const bySpan = new Map<string, Mention[]>();
  for (const item of found) {
    const span = `${item.start}:${item.end}`;
    const group = bySpan.get(span) ?? [];
    group.push(item);
    bySpan.set(span, group);
  }
  const resolved: Mention[] = [];
  for (const group of bySpan.values()) {
    const exact = group.filter((item) => item.exact);
    if (exact.length === 1) resolved.push(exact[0]);
    else if (exact.length > 1) resolved.push(...exact);
    else if (new Set(group.map((item) => item.key.id)).size === 1) resolved.push(group[0]);
  }

  // If "Apple" and "Apple Inc" overlap, only the more specific entity mention wins.
  return resolved
    .filter(
      (item) =>
        !resolved.some(
          (other) =>
            other.key.id !== item.key.id &&
            other.start <= item.start &&
            other.end >= item.end &&
            other.end - other.start > item.end - item.start
        )
    )
    .sort((a, b) => a.start - b.start || b.end - b.start - (a.end - a.start));
}

const TASK_FILLER = new Set([
  "which", "who", "what", "were", "was", "are", "is", "did", "do", "does",
  "and", "or", "both", "same", "than", "the", "a", "an", "of", "from", "between",
  "compare", "compared", "comparison", "versus", "vs", "more", "less", "earlier", "later",
  "first", "higher", "lower", "older", "younger", "어느", "누가", "중", "둘", "다", "보다",
  "비교", "차이", "동일", "같은", "같다", "같나요", "같습니까", "먼저", "나중", "더",
  "과", "와", "은", "는", "이", "가", "을", "를", "의",
]);

export function analyzeTaskEvidence(query: string, keys: Key[]): {
  entities: Key[];
  labels: string[];
  taskQuery: string;
} | null {
  if (!comparisonCue(query)) return null;
  const normalizedQuery = normalize(query);
  const entityMentions = mentions(query, keys);
  const unique = [...new Map(entityMentions.map((item) => [item.key.id, item])).values()].slice(0, 2);
  if (unique.length < 2) return null;
  const chars = [...normalizedQuery];
  for (const mention of unique) {
    for (let i = mention.start; i < mention.end; i++) chars[i] = " ";
  }
  const taskQuery = chars.join("").split(/\s+/u).filter((token) => token && !TASK_FILLER.has(token)).join(" ");
  return {
    entities: unique.map((item) => item.key),
    labels: unique.map((item) => item.label),
    taskQuery,
  };
}

/**
 * Reorders an already activated associative pool for comparison evidence use.
 * It never removes candidates from the pool and never changes association scores.
 */
export function projectTaskEvidence<T>(
  query: string,
  candidates: EvidenceSelectionCandidate<T>[],
  topK: number
): EvidenceSelectionResult<T> {
  if (topK < 2 || candidates.length <= topK || !comparisonCue(query)) {
    return { applied: false, entities: [], selected: candidates };
  }

  const uniqueKeys = new Map<string, Key>();
  for (const candidate of candidates) {
    for (const key of candidate.keys) uniqueKeys.set(key.id, key);
  }
  const analysis = analyzeTaskEvidence(query, [...uniqueKeys.values()]);
  if (!analysis) return { applied: false, entities: [], selected: candidates };

  // If the leading slots already form a one-to-one entity pair, direct evidence is
  // sufficient. Preserve the stronger associative order instead of second-guessing it.
  const front = candidates.slice(0, analysis.entities.length);
  const sufficient = (candidate: EvidenceSelectionCandidate<T>, entity: Key): boolean => {
    if (!candidate.keys.some((key) => key.id === entity.id)) return false;
    const best = Math.max(
      ...candidates
        .filter((item) => item.keys.some((key) => key.id === entity.id))
        .map((item) => item.utility)
    );
    return candidate.utility >= best - DIRECT_SUFFICIENCY_MARGIN;
  };
  if (
    analysis.entities.length === 2 &&
    front.length === 2 &&
    ((sufficient(front[0], analysis.entities[0]) && sufficient(front[1], analysis.entities[1])) ||
      (sufficient(front[0], analysis.entities[1]) && sufficient(front[1], analysis.entities[0])))
  ) {
    return { applied: false, entities: analysis.labels, selected: candidates };
  }

  const picked: EvidenceSelectionCandidate<T>[] = [];
  const pickedIds = new Set<string>();
  for (const entity of analysis.entities) {
    let best: EvidenceSelectionCandidate<T> | undefined;
    let bestRank = Number.POSITIVE_INFINITY;
    for (let rank = 0; rank < candidates.length; rank++) {
      const candidate = candidates[rank];
      if (pickedIds.has(candidate.id) || !candidate.keys.some((key) => key.id === entity.id)) continue;
      if (!best || candidate.utility > best.utility || (candidate.utility === best.utility && rank < bestRank)) {
        best = candidate;
        bestRank = rank;
      }
    }
    if (best) {
      picked.push(best);
      pickedIds.add(best.id);
    }
  }

  if (picked.length < 2) return { applied: false, entities: [], selected: candidates };
  return {
    applied: true,
    entities: analysis.labels,
    selected: [...picked, ...candidates.filter((candidate) => !pickedIds.has(candidate.id))],
  };
}
