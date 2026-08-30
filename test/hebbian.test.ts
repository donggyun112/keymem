import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

// The point of an association is to carry what PAST queries revealed into a FUTURE query
// that only reaches one side. So the training query must match both keys and the test
// query only one — if the test query matched both, both clusters would already be hop-1
// hits and the edge would prove nothing.
//
//   cos(배포, 릴리스)      = 0.50 — below KEY_RECALL (0.62): the test query "배포" reaches
//                                  배포 only. Also below KEY_MERGE, so they stay two keys.
//   cos("배포 릴리스", 배포) = 0.80, cos(…, 릴리스) = 0.92 — the training query reaches both.
//   cos(A, 릴리스)         = 0.45 — below KEY_AUTO_LINK (0.62), so the hop-1 memory never
//                                  picks up 릴리스 as a second key; without this, ordinary
//                                  2-hop traversal would reach B on its own.
//   cos(B, "배포")         = 0    — the content path cannot see B either.
const VECTORS: Record<string, number[]> = {
  배포: [1, 0, 0, 0],
  릴리스: [0.5, 0.866, 0, 0],
  "배포 릴리스": [0.8, 0.6, 0, 0],
  "태그를 먼저 올린다": [0.9, 0, 0.436, 0],
  "레지스트리 access는 restricted": [0, 0.9, 0.436, 0],
};

function vec(text: string): number[] {
  return VECTORS[text.toLowerCase()] ?? [0, 0, 0, 1];
}

async function build(t: { after: (fn: () => unknown) => void }, hebbian: boolean) {
  const dir = await mkdtemp(join(tmpdir(), "keymem-hebb-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.KEYMEM_SHORT_KEY_MERGE = "0";
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  if (hebbian) process.env.KEYMEM_HEBBIAN = "true";
  else delete process.env.KEYMEM_HEBBIAN;
  t.after(() => {
    delete process.env.KEYMEM_SHORT_KEY_MERGE;
    delete process.env.KEYMEM_HEBBIAN;
  });

  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vec(text));
  t.after(() => embedding.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?hebb=${n++}`);
  const graph = new MemoryGraph();
  await graph.load();
  const [near] = await graph.add("태그를 먼저 올린다", ["배포"]);
  const [far] = await graph.add("레지스트리 access는 restricted", ["릴리스"]);

  const reach = async (): Promise<string[]> =>
    ((await graph.recall("배포", 10, null, true, 2)) as Array<{ id: string }>).map((m) => m.id);

  // One round of realistic use: a query that reaches both keys, then a confirmed read.
  const train = async () => {
    await graph.searchKeys("배포 릴리스", 8, null);
    const kid = graph
      .getKeyRefsForMemory(near)
      .find((k: { concept: string }) => k.concept === "배포")!.key_id;
    await graph.readMemory(near, kid);
  };

  return { graph, near, far, reach, train };
}

test("keys asked about together become a traversable edge after confirmed reads", async (t) => {
  const { near, far, reach, train } = await build(t, true);

  assert.ok(
    !(await reach()).includes(far),
    "premise: the clusters share no memory and the test query reaches only one of them"
  );

  await train();
  assert.ok(
    !(await reach()).includes(far),
    "one co-match is a coincidence — the edge must not open below HEBBIAN_PROMOTE_N"
  );

  await train();
  await train();
  const ranked = await reach();
  assert.ok(ranked.includes(far), "after 3 confirmed co-matches recall should cross the clusters");
  assert.ok(
    ranked.indexOf(near) < ranked.indexOf(far),
    `an association is weaker evidence than a shared memory and must not outrank it: ${ranked.join(", ")}`
  );
});
