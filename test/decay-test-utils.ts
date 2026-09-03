import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

let graphImport = 0;

export async function freshDecayGraph(
  t: { after(fn: () => void | Promise<void>): void },
  now: () => number,
  vector: (text: string) => number[] = () => [1, 0]
) {
  const dir = await mkdtemp(join(tmpdir(), "keymem-decay-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vector(text));
  t.after(() => embedding.__clearTestEmbedder());
  const module = await import(`../src/memoryGraph.ts?decay-test=${graphImport++}`);
  const graph = new module.MemoryGraph({ now });
  await graph.load();
  return { graph, dir };
}
