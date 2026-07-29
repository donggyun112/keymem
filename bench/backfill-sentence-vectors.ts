// backfill-sentence-vectors.ts — one-off migration: compute sentence vectors for
// existing multi-fact memories (namespace !== "bench" — no point embedding wiki junk)
// and save, which also migrates all vectors to the binary sidecar.
// Usage: KEYMEM_DATA_DIR=<data-dir> EMBEDDING_BACKEND=local LOCAL_EMBEDDING_MODEL=bge-m3 \
//        npx tsx bench/backfill-sentence-vectors.ts        (daemon MUST be stopped first)
import { MemoryGraph, splitSentences } from "../src/memoryGraph.js";
import { embedTextAsync } from "../src/embedding.js";

async function main() {
  const g = new MemoryGraph() as any;
  await g.load();
  let backfilled = 0;
  let skipped = 0;
  for (const [mid, mem] of Object.entries<any>(g.memories)) {
    if (mem.namespace === "bench") { skipped++; continue; }
    if (g._sentVecs[mid]?.length > 0) continue;
    const sentences = splitSentences(mem.content ?? "");
    if (sentences.length === 0) continue;
    const vecs: number[][] = [];
    for (const s of sentences) vecs.push(await embedTextAsync(s));
    g._sentVecs[mid] = vecs;
    backfilled++;
    if (backfilled % 25 === 0) console.log(`  ...${backfilled} memories backfilled`);
  }
  await g.save();
  console.log(
    `backfilled ${backfilled} memories with sentence vectors ` +
      `(bench namespace skipped: ${skipped}); store migrated to vector sidecar`
  );
  process.exit(0);
}
main();
