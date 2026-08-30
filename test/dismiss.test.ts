import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

const VECTORS: Record<string, number[]> = {
  배포: [1, 0, 0, 0],
  릴리스: [0.9, 0.44, 0, 0],
  "릴리스는 태그를 먼저 올린다": [0.7, 0.71, 0, 0],
  // Starts CLOSER to the 배포 key than its sibling (0.9 vs 0.7) so the rank flip below
  // is caused by the dismissal and not by the ordering it already had. Orthogonal
  // second component keeps the two contents from being near-duplicates (dedup would
  // supersede one of them).
  "사내 레지스트리는 access를 restricted로 둔다": [0.9, 0, 0.436, 0],
};

function vec(text: string): number[] {
  return VECTORS[text.toLowerCase()] ?? [0, 0, 0, 1];
}

test("dismiss weakens only the link it names, and can never orphan the memory", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-dismiss-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.KEYMEM_SHORT_KEY_MERGE = "0";
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  t.after(() => delete process.env.KEYMEM_SHORT_KEY_MERGE);

  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vec(text));
  t.after(() => embedding.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?dismiss=${n++}`);
  const graph = new MemoryGraph();
  await graph.load();
  const [mid] = await graph.add("릴리스는 태그를 먼저 올린다", ["배포", "릴리스"]);

  const keyRefs = graph.getKeyRefsForMemory(mid) as Array<{ concept: string; key_id: string }>;
  const wrong = keyRefs.find((k) => k.concept === "배포")!;
  const other = keyRefs.find((k) => k.concept === "릴리스")!;

  const first = (await graph.dismiss(mid, wrong.key_id)) as {
    link_weight: number;
    previous_link_weight: number;
    floored: boolean;
  };
  assert.ok(first.link_weight < first.previous_link_weight, "dismiss must weaken the link");

  // The other key is untouched — dismissal is a claim about one pairing, not the fact.
  const afterOne = graph.getKeyRefsForMemory(mid).map((k: { concept: string }) => k.concept).sort();
  assert.deepEqual(afterOne, ["릴리스", "배포"].sort());

  // Dismissing repeatedly floors the weight but never severs the link: the memory
  // stays reachable, which is the invariant that makes negative feedback safe.
  for (let i = 0; i < 20; i++) await graph.dismiss(mid, wrong.key_id);
  const floored = (await graph.dismiss(mid, wrong.key_id)) as { link_weight: number; floored: boolean };
  assert.equal(floored.floored, true);
  assert.ok(floored.link_weight > 0, "weight is floored, not zeroed");
  assert.ok(
    graph.getKeysForMemory(mid).includes("배포"),
    "a floored link is still a link — the memory must not be orphaned"
  );

  // A read pays part of it back, so a mistaken dismissal is recoverable.
  await graph.readMemory(mid, wrong.key_id);
  const recovered = (await graph.dismiss(mid, wrong.key_id)) as { previous_link_weight: number };
  assert.ok(recovered.previous_link_weight > floored.link_weight, "reads reinforce back up");

  // Dismissing a key that never linked this memory is an error, not a silent no-op.
  await assert.rejects(() => graph.dismiss(mid, "no-such-key"), /not linked/);
  await assert.rejects(() => graph.dismiss("no-such-memory", other.key_id), /not found/);
});

// The weight dismiss moves multiplies the key-path score in recall (keySim * idf * lw),
// so the signal has to show up as a rank change, not just a stored number.
test("a dismissed memory ranks below its sibling under the same key", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-dismiss-rank-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.KEYMEM_SHORT_KEY_MERGE = "0";
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  t.after(() => delete process.env.KEYMEM_SHORT_KEY_MERGE);

  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vec(text));
  t.after(() => embedding.__clearTestEmbedder());

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?dismissrank=${n++}`);
  const graph = new MemoryGraph();
  await graph.load();
  const [keep] = await graph.add("릴리스는 태그를 먼저 올린다", ["배포"]);
  const [drop] = await graph.add("사내 레지스트리는 access를 restricted로 둔다", ["배포"]);

  const rank = async (): Promise<string[]> =>
    ((await graph.recall("배포", 5, null, false, 1)) as Array<{ id: string }>).map((m) => m.id);

  const before = await rank();
  assert.ok(before.includes(keep) && before.includes(drop), "both start reachable");
  assert.ok(
    before.indexOf(drop) < before.indexOf(keep),
    `the memory to dismiss must start AHEAD, or the flip proves nothing: ${before.join(", ")}`
  );

  const kid = graph
    .getKeyRefsForMemory(drop)
    .find((k: { concept: string }) => k.concept === "배포")!.key_id;
  for (let i = 0; i < 5; i++) await graph.dismiss(drop, kid);

  const after = await rank();
  assert.ok(
    after.indexOf(drop) > after.indexOf(keep),
    `dismissed memory should rank last, got ${after.join(", ")}`
  );
  assert.ok(after.includes(drop), "still reachable — dismiss demotes, it does not delete");
});
