// quarantine-bench-data.ts — move HotpotQA/wiki benchmark pollution out of `default`.
// Matched memories (default namespace, empty source, no Hangul in content) are moved to
// the `bench` namespace. Reversible: full backup + moved-id log are written next to the store.
// Usage: npx tsx bench/quarantine-bench-data.ts <data-dir>   (daemon MUST be stopped first)
import { readFileSync, writeFileSync, copyFileSync, renameSync } from "node:fs";
import { join } from "node:path";

const dir = process.argv[2];
if (!dir) {
  console.error("usage: quarantine-bench-data.ts <data-dir>");
  process.exit(1);
}
const file = join(dir, "graph.json");
const hasHangul = (s: string) => /[가-힣]/.test(s);

copyFileSync(file, `${file}.pre-quarantine.bak`);
const g = JSON.parse(readFileSync(file, "utf-8"));
const moved: string[] = [];
for (const [id, m] of Object.entries<any>(g.memories)) {
  const src = m.source ?? {};
  const emptySource = !src || Object.keys(src).length === 0;
  if (m.namespace === "default" && emptySource && !hasHangul(m.content ?? "")) {
    m.namespace = "bench";
    moved.push(id);
  }
}
writeFileSync(join(dir, "bench-moved-ids.json"), JSON.stringify(moved, null, 2));
const tmp = `${file}.quarantine.tmp`;
writeFileSync(tmp, JSON.stringify(g));
renameSync(tmp, file);
console.log(`moved ${moved.length} memories default→bench; backup at graph.json.pre-quarantine.bak`);
