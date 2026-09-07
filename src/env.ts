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

/**
 * ONNX intra-op threads for an in-process model (reranker, bge-m3 embedder).
 *
 * These run on every conversational turn, so their CPU burst is something the user feels.
 * ONNX Runtime picks its own count when given none — about half the cores on an M4 Pro,
 * measured at 6.9. Capping is a deliberate trade, not a free win: over one 30-candidate
 * rerank pool on that machine, ORT's default is 503ms wall / 3464ms CPU / 6.9 cores, while
 * 4 threads is 670ms / 2678ms / 4.0 — a 42% lower peak and 23% less CPU for 33% more
 * latency. Never go above the performance-core count: at 14 (10P + 4E) it collapses to
 * 944ms / 12425ms / 13.2 cores, because every intra-op barrier waits on an E-core while
 * the fast cores spin.
 *
 * The default takes a quarter of the machine, capped at 6 — 1 thread on a 4-core laptop,
 * 4 on a 14-core M4 Pro, 6 on a 24-core desktop or larger. Callers pass
 * availableParallelism(), which honors cgroup/container limits, so a boxed-in daemon scales
 * down with its allowance. Model output is identical at any count.
 */
export function modelThreads(parallelism: number, override?: string): number {
  return Math.max(1, Number(override) || Math.min(6, Math.round(parallelism / 4)));
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
