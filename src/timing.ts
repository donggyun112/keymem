// KEYMEM_TIMING=1: per-stage latency lines on stderr. Profiling aid; no-op when unset.
const ON = Boolean(process.env.KEYMEM_TIMING);

export async function timed<T>(label: string, fn: () => Promise<T>): Promise<T> {
  if (!ON) return fn();
  const t = performance.now();
  try {
    return await fn();
  } finally {
    console.error(`[timing] ${label} ${(performance.now() - t).toFixed(0)}ms`);
  }
}
