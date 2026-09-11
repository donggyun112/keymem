export interface RecallKeyCandidate {
  key_id: string;
  concept: string;
  score: number;
  match_type: string;
  memory_count: number;
  aliases?: string[];
  key_type?: string;
  is_hub?: boolean;
  specificity?: number;
  [key: string]: unknown;
}

export interface CompactRecallKey {
  key_id: string;
  concept: string;
  aliases: string[];
  key_type: string;
  score: number;
  match_type: string;
  memory_count: number;
  is_hub: boolean;
  specificity: number;
}

export function compactRecallKeys(keys: RecallKeyCandidate[]): CompactRecallKey[] {
  return keys.map(({
    key_id,
    concept,
    aliases,
    key_type,
    score,
    match_type,
    memory_count,
    is_hub,
    specificity,
  }) => ({
    key_id,
    concept,
    aliases: aliases ?? [],
    key_type: key_type ?? "concept",
    score,
    match_type,
    memory_count,
    is_hub: is_hub ?? false,
    specificity: specificity ?? 1,
  }));
}
