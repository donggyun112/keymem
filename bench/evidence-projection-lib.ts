export interface RankedEvidence {
  id: string;
  title: string;
}

export interface ProjectionResult<T extends RankedEvidence> {
  applied: boolean;
  entities: string[];
  selected: T[];
}

function normalizeForMention(text: string): string {
  return text
    .normalize("NFKC")
    .toLocaleLowerCase("en")
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

/**
 * Benchmark-only entity extraction: HotpotQA paragraph titles stand in for
 * KeyMem entity keys. Gold support labels are deliberately not consulted.
 */
export function mentionedTitles(question: string, titles: string[]): string[] {
  const normalizedQuestion = ` ${normalizeForMention(question)} `;
  const unique = [...new Set(titles)];

  const matches = unique
    .map((title) => {
      const normalizedTitle = normalizeForMention(title);
      return {
        title,
        index: normalizedTitle ? normalizedQuestion.indexOf(` ${normalizedTitle} `) : -1,
        length: normalizedTitle.length,
      };
    })
    .filter((x) => x.index >= 0)
    .filter(
      (candidate, _index, all) =>
        !all.some(
          (other) =>
            other.length > candidate.length &&
            other.index <= candidate.index &&
            other.index + other.length >= candidate.index + candidate.length
        )
    );

  return matches
    .sort((a, b) => a.index - b.index || b.length - a.length)
    .map((x) => x.title);
}

/**
 * Project an already-activated set onto comparison entities, then fill the
 * remaining budget in the original activation order. This changes selection,
 * never graph traversal or activation scores.
 */
export function projectComparisonEvidence<T extends RankedEvidence>(
  question: string,
  corpusTitles: string[],
  activated: T[],
  topK: number
): ProjectionResult<T> {
  const entities = mentionedTitles(question, corpusTitles);
  if (entities.length < 2) {
    return { applied: false, entities, selected: activated.slice(0, topK) };
  }

  const selected: T[] = [];
  const selectedIds = new Set<string>();
  for (const entity of entities) {
    const hit = activated.find((candidate) => candidate.title === entity);
    if (!hit || selectedIds.has(hit.id)) continue;
    selected.push(hit);
    selectedIds.add(hit.id);
    if (selected.length === topK) break;
  }

  for (const candidate of activated) {
    if (selected.length === topK) break;
    if (selectedIds.has(candidate.id)) continue;
    selected.push(candidate);
    selectedIds.add(candidate.id);
  }

  return { applied: true, entities, selected };
}
