import { appendFile, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { cfgRaw, dataDir } from "./env.js";
import type { DirectHydrateTop1Decision } from "./memoryGraph.js";

export const DIRECT_HYDRATE_SHADOW_FILE = "direct-hydrate-shadow.jsonl";

export interface DirectHydrateShadowEvent {
  schema_version: 1;
  recorded_at: string;
  query: string;
  context: string | null;
  namespace: string | null;
  host: { agent: string; session_id: string; turn: number } | null;
  decision: DirectHydrateTop1Decision;
}

export function directHydrateShadowEnabled(): boolean {
  return cfgRaw("DIRECT_HYDRATE_SHADOW") === "true";
}

export async function recordDirectHydrateShadow(
  event: Omit<DirectHydrateShadowEvent, "schema_version" | "recorded_at">
): Promise<void> {
  const dir = dataDir();
  await mkdir(dir, { recursive: true });
  const record: DirectHydrateShadowEvent = {
    schema_version: 1,
    recorded_at: new Date().toISOString(),
    ...event,
  };
  await appendFile(
    join(dir, DIRECT_HYDRATE_SHADOW_FILE),
    `${JSON.stringify(record)}\n`,
    "utf8",
  );
}
