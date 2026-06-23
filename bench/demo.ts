// Terminal demo of keymem's associative recall, for an asciinema cast → GIF.
// Shows: two facts saved; a later question that never mentions peanuts or Mina still surfaces
// the allergy via the shared 'peanuts' key — a hop a vector store can't make.
/* eslint-disable no-console */
console.error = () => {}; // hide keymem's internal [graph]/[keymem] logs
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

process.env.EMBEDDING_BACKEND = "local";
process.env.LOCAL_EMBEDDING_MODEL = "fast-bge-small-en-v1.5";
const dir = await mkdtemp(join(tmpdir(), "keymem-demo-"));
process.env.KEYMEM_DATA_DIR = dir;
const { MemoryGraph } = await import("../src/memoryGraph.ts");

const C = { b: "\x1b[1m", d: "\x1b[2m", g: "\x1b[32m", c: "\x1b[36m", y: "\x1b[33m", r: "\x1b[31m", x: "\x1b[0m" };
const sleep = (ms: number) => new Promise((res) => setTimeout(res, ms));
const out = async (s = "", d = 550) => { process.stdout.write(s + "\n"); await sleep(d); };
const type = async (s: string, d = 700) => { process.stdout.write(s); await sleep(d); process.stdout.write("\n"); };

const graph = new MemoryGraph();
await graph.load();

await out(`${C.b}${C.c}keymem${C.x}  ${C.d}— memory that recalls by association, not just similarity${C.x}`);
await out();
await out(`${C.d}# the agent quietly saves two facts from different conversations${C.x}`);
const facts = [
  { content: "Mina is allergic to peanuts", keys: ["Mina", "allergy", "peanuts"] },
  { content: "The party cake recipe uses peanut butter frosting", keys: ["party cake", "recipe", "peanut butter", "peanuts"] },
];
const known: Record<string, string> = {};
for (const f of facts) {
  const [id] = await graph.add(f.content, f.keys, {});
  known[id] = f.content;
  await out(`  ${C.g}remember${C.x}  ${f.content}`);
  await out(`           ${C.d}keys: ${f.keys.join(" · ")}${C.x}`, 350);
}
await out();
await out(`${C.d}# later — a question that never says "peanuts", "allergy", or "Mina"${C.x}`);
await type(`  ${C.y}recall${C.x}  "is the party cake ok to serve to the kids?"`, 1100);
const res = (await graph.recallInject("is the party cake ok to serve to the kids", 5, null, {})) as {
  memories: Array<{ id: string; content?: string }>;
};
await out();
await out(`  ${C.b}↳ memories surfaced in one call:${C.x}`);
for (const m of res.memories) {
  const txt = known[m.id] ?? m.content ?? m.id;
  const flag = /allerg/i.test(txt) ? `   ${C.r}← connected-but-dissimilar, via the shared 'peanuts' key${C.x}` : "";
  await out(`     • ${txt}${flag}`, 700);
}
await out();
await out(`${C.d}A vector store ranks by similarity to "party cake" and never reaches${C.x}`, 350);
await out(`${C.d}"Mina is allergic to peanuts." keymem walks the shared key and catches it.${C.x}`, 1200);
await out();
await out(`  ${C.c}npx -y keymem${C.x}    ${C.d}·  github.com/donggyun112/keymem  ·  MIT${C.x}`, 1500);

await rm(dir, { recursive: true, force: true });
