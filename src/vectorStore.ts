// Binary vector sidecar: all embeddings live in vectors.bin (Float32Array concat)
// indexed by vectors.idx.json, keeping graph.json small (floats as JSON text made a
// 1.3k-memory store ~93 MB; binary is ~13 MB). Entry key convention:
//   "m:<mid>"  one whole-memory vector      (n = dim)
//   "k:<kid>"  one key-concept vector       (n = dim)
//   "s:<mid>"  sentence pack for a memory   (n = count * dim, concatenated)
// Writes are atomic (tmp + rename), bin before idx, so a torn write leaves at worst
// an idx pointing at a fully-written bin from the same save.
import { readFile, writeFile, rename } from "node:fs/promises";
import { join } from "node:path";

export type VectorIndex = {
  schema: 1;
  dim: number;
  entries: Record<string, { off: number; n: number }>;
};

const BIN = "vectors.bin";
const IDX = "vectors.idx.json";

export async function writeVectors(dir: string, vecs: Map<string, number[][]>): Promise<void> {
  let dim = 0;
  for (const list of vecs.values()) {
    if (list.length > 0 && list[0].length > 0) { dim = list[0].length; break; }
  }
  const entries: VectorIndex["entries"] = {};
  let total = 0;
  for (const [id, list] of vecs) {
    const n = list.reduce((acc, v) => acc + v.length, 0);
    if (n === 0) continue;
    entries[id] = { off: total, n };
    total += n;
  }
  const flat = new Float32Array(total);
  let cursor = 0;
  for (const [id, list] of vecs) {
    if (!(id in entries)) continue;
    for (const v of list) { flat.set(v, cursor); cursor += v.length; }
  }
  const idx: VectorIndex = { schema: 1, dim, entries };
  const binTmp = join(dir, `${BIN}.${process.pid}.tmp`);
  const idxTmp = join(dir, `${IDX}.${process.pid}.tmp`);
  await writeFile(binTmp, Buffer.from(flat.buffer, flat.byteOffset, flat.byteLength));
  await rename(binTmp, join(dir, BIN));
  await writeFile(idxTmp, JSON.stringify(idx), "utf-8");
  await rename(idxTmp, join(dir, IDX));
}

export async function readVectors(dir: string): Promise<Map<string, number[][]> | null> {
  let idx: VectorIndex;
  let buf: Buffer;
  try {
    idx = JSON.parse(await readFile(join(dir, IDX), "utf-8")) as VectorIndex;
    buf = await readFile(join(dir, BIN));
  } catch {
    return null; // no/unreadable sidecar → caller falls back to inline / re-embed
  }
  if (idx.schema !== 1 || !Number.isFinite(idx.dim) || idx.dim <= 0) return null;
  const totalFloats = Math.floor(buf.byteLength / 4);
  const flat = new Float32Array(buf.buffer, buf.byteOffset, totalFloats);
  const out = new Map<string, number[][]>();
  for (const [id, { off, n }] of Object.entries(idx.entries)) {
    if (off < 0 || n <= 0 || off + n > totalFloats || n % idx.dim !== 0) continue;
    const list: number[][] = [];
    for (let i = 0; i < n; i += idx.dim) {
      list.push(Array.from(flat.subarray(off + i, off + i + idx.dim)));
    }
    out.set(id, list);
  }
  return out;
}
