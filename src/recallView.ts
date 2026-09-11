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
  relation_strength?: number;
  relation_evidence?: number;
  relation_path?: { source_key_id: string; bridge_memory_id: string };
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
    relation_strength,
    relation_evidence,
    relation_path,
  }) => {
    const path =
      relation_path &&
      typeof relation_path === "object" &&
      typeof (relation_path as Record<string, unknown>).source_key_id === "string" &&
      typeof (relation_path as Record<string, unknown>).bridge_memory_id === "string"
        ? relation_path as CompactRecallKey["relation_path"]
        : undefined;
    return {
      key_id,
      concept,
      aliases: aliases ?? [],
      key_type: key_type ?? "concept",
      score,
      match_type,
      memory_count,
      is_hub: is_hub ?? false,
      specificity: specificity ?? 1,
      ...(typeof relation_strength === "number" ? { relation_strength } : {}),
      ...(typeof relation_evidence === "number" ? { relation_evidence } : {}),
      ...(path ? { relation_path: path } : {}),
    };
  });
}
