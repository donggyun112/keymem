// Perf check: _bridgePhraseKeys() now runs eagerly inside add()/supersede() whenever the
// just-written key set includes a phrase (3+ tokens). It's O(total keys) per call (rebuilds a
// byConcept map over every key + alias). Does that matter at a realistic key-count scale?
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

process.env.EMBEDDING_BACKEND ??= "local";
process.env.LOCAL_EMBEDDING_MODEL ??= "bge-m3";

const dir = await mkdtemp(join(tmpdir(), "km-phrase-perf-"));
process.env.KEYMEM_DATA_DIR = dir;
const { MemoryGraph } = await import("../src/memoryGraph.ts");
const g: any = new MemoryGraph();
await g.load();

const N_KEYS = 2000; // realistic-to-large personal store key count
console.log(`seeding ${N_KEYS} atomic keys...`);
const seedStart = performance.now();
for (let i = 0; i < N_KEYS; i++) {
  await g.add(`filler fact number ${i} about topic ${i}`, [`topic${i}`]);
}
console.log(`seeded in ${(performance.now() - seedStart).toFixed(0)}ms\n`);

const N_TRIALS = 20;

async function timeAdds(label: string, makeKeys: (i: number) => string[]) {
  const start = performance.now();
  for (let i = 0; i < N_TRIALS; i++) {
    await g.add(`trial content ${label} ${i}`, makeKeys(i));
  }
  const total = performance.now() - start;
  console.log(`${label}: ${(total / N_TRIALS).toFixed(2)}ms/add avg over ${N_TRIALS} (total ${total.toFixed(0)}ms)`);
}

await timeAdds("atomic-key (no phrase, no bridge call)", (i) => [`atomic${i}`]);
await timeAdds("phrase-key (triggers _bridgePhraseKeys)", (i) => [`some phrase key number ${i}`]);

await rm(dir, { recursive: true, force: true });
