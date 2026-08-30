import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;

function vec(text: string): number[] {
  const terms = ["one", "two", "key"];
  const out = new Array(terms.length + 1).fill(0);
  const i = terms.indexOf(text.toLowerCase());
  out[i >= 0 ? i : terms.length] = 1;
  return out;
}

// "Nexora" and "nexora" used to hard-partition the same project: namespaces are
// compared with exact equality, so a scoped recall silently missed half its memories.
test("namespaces are case-folded on write, on query, and repaired on load", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-ns-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.KEYMEM_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";

  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vec(text));
  t.after(() => embedding.__clearTestEmbedder());

  const { MemoryGraph, normalizeNamespace } =
    await import(`../src/memoryGraph.ts?ns=${n++}`);

  assert.equal(normalizeNamespace("  Nexora "), "nexora");
  assert.equal(normalizeNamespace(""), null);
  assert.equal(normalizeNamespace(null), null);

  const graph = new MemoryGraph();
  await graph.load();
  await graph.add("one", ["key"], { namespace: "Nexora" });
  await graph.add("two", ["key"], { namespace: "nexora" });

  // Both writes land in one namespace, and either casing queries it.
  assert.equal(graph.listAll("nexora").length, 2);
  assert.equal(graph.listAll("NEXORA").length, 2);
  await graph.flush();

  // A graph written before normalization still carries the mixed-case value; the
  // loader repairs it instead of leaving a partitioned namespace on disk.
  const path = join(dir, "graph.json");
  const raw = JSON.parse(await readFile(path, "utf8"));
  const mid = Object.keys(raw.memories)[0];
  raw.memories[mid].namespace = "NeXoRa";
  await writeFile(path, JSON.stringify(raw));

  const { MemoryGraph: Reloaded } = await import(`../src/memoryGraph.ts?ns=${n++}`);
  const reloaded = new Reloaded();
  await reloaded.load();
  assert.equal(reloaded.listAll("nexora").length, 2);
  await reloaded.flush();
  const after = JSON.parse(await readFile(path, "utf8"));
  assert.equal(after.memories[mid].namespace, "nexora");
});
