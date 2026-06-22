// Centralized config-env resolution with backward compatibility.
//
// The project was renamed `super-memory` → `keymem`. Config env vars now use the
// `KEYMEM_` prefix, but the legacy `SUPER_MEMORY_` prefix is still honored as a
// fallback so existing deployments, MCP configs, and on-disk data keep working.
// Only project-specific vars are remapped here — shared names like OPENAI_API_KEY,
// EMBEDDING_BACKEND, and LOCAL_EMBEDDING_MODEL are read directly elsewhere.
import { homedir } from "os";
import { join } from "path";
import { existsSync } from "fs";

const PRIMARY_PREFIX = "KEYMEM_";
const LEGACY_PREFIX = "SUPER_MEMORY_";

/** Resolve a config var by suffix (e.g. "KEY_MERGE"), preferring KEYMEM_, then SUPER_MEMORY_. */
export function cfgRaw(suffix: string): string | undefined {
  return process.env[PRIMARY_PREFIX + suffix] ?? process.env[LEGACY_PREFIX + suffix];
}

/** Primary env var name for a suffix — used in user-facing warnings. */
export function cfgName(suffix: string): string {
  return PRIMARY_PREFIX + suffix;
}

/**
 * Home-anchored base directory for cached artifacts (models). Independent of the
 * DATA_DIR override. Prefers ~/.keymem, but falls back to an existing legacy
 * ~/.super-memory so a prior install's cache/data is reused, not orphaned.
 */
export function homeBaseDir(): string {
  const primary = join(homedir(), ".keymem");
  const legacy = join(homedir(), ".super-memory");
  if (!existsSync(primary) && existsSync(legacy)) return legacy;
  return primary;
}

/** Graph/conversation storage directory. Honors KEYMEM_DATA_DIR / SUPER_MEMORY_DATA_DIR. */
export function dataDir(): string {
  return cfgRaw("DATA_DIR") ?? homeBaseDir();
}
