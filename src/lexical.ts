import { readFileSync, existsSync, statSync, renameSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";
import { KiwiBuilder, type Kiwi } from "kiwi-nlp";
import { homeBaseDir } from "./env.js";
import { httpDownload } from "./modelDownload.js";

const MODEL_FILES = ["combiningRule.txt", "cong.mdl", "default.dict", "dialect.dict", "extract.mdl", "multi.dict", "nounchr.mdl", "sj.morph", "typo.dict"];
let kiwi: Kiwi | undefined;
let initializing: Promise<void> | undefined;

/** Load the pinned Kiwi model once before MiniSearch invokes its synchronous tokenizer. */
export async function ensureLexicalTokenizer(): Promise<void> {
  if (kiwi) return;
  if (!initializing) {
    initializing = (async () => {
      const dir = join(homeBaseDir(), "models", "kiwi", "0.24.0");
      for (const name of MODEL_FILES) {
        const dest = join(dir, name);
        if (existsSync(dest) && statSync(dest).size > 0) continue;
        const host = /\.(mdl|morph)$/.test(name) ? "media.githubusercontent.com/media" : "raw.githubusercontent.com";
        // A separate destination prevents concurrent processes from sharing a partial download.
        const temporary = `${dest}.${process.pid}`;
        try {
          await httpDownload(`https://${host}/bab2min/Kiwi/v0.24.0/models/cong/base/${name}`, temporary);
          renameSync(temporary, dest);
        } finally {
          rmSync(temporary, { force: true });
          rmSync(`${temporary}.tmp`, { force: true });
        }
      }
      const require = createRequire(import.meta.url);
      const builder = await KiwiBuilder.create(join(dirname(require.resolve("kiwi-nlp")), "kiwi-wasm.wasm"));
      kiwi = await builder.build({
        modelType: "cong",
        modelFiles: Object.fromEntries(MODEL_FILES.map(name => [name, readFileSync(join(dir, name))])),
      });
    })().finally(() => { initializing = undefined; });
  }
  await initializing;
}

export function tokenizeLexical(text: string): string[] {
  if (!kiwi) throw new Error("Kiwi tokenizer is not initialized; await ensureLexicalTokenizer() first");
  return kiwi.tokenize(text.normalize("NFKC").toLowerCase())
    .filter(token => /^(?:[NVM]|SL|SN|SH|XR)/.test(token.tag))
    .map(token => token.str);
}
