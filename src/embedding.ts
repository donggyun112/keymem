import { config } from "dotenv";
config();

import OpenAI from "openai";
import { existsSync, statSync } from "node:fs";
import { join } from "node:path";
import { KNOWN_MODELS, defaultModelDir, ensureModelFiles, type Fetcher } from "./modelDownload.js";
import { availableParallelism } from "node:os";
import { cfgRaw, cfgName, homeBaseDir, modelThreads } from "./env.js";
import { timed } from "./timing.js";

export const OPENAI_API_KEY = process.env.OPENAI_API_KEY ?? "";
export const OPENAI_EMBEDDING_MODEL =
  process.env.OPENAI_EMBEDDING_MODEL ?? "text-embedding-3-small";

const _DEFAULT_BACKEND = OPENAI_API_KEY ? "openai" : "local";
export const EMBEDDING_BACKEND =
  process.env.EMBEDDING_BACKEND ?? _DEFAULT_BACKEND;
export const LOCAL_EMBEDDING_MODEL =
  process.env.LOCAL_EMBEDDING_MODEL ?? "fast-multilingual-e5-large";
// One deterministic home for ALL downloaded models. Defaults to <keymem home>/models
// (i.e. ~/.keymem/models) — the same parent bge-m3 uses — so e5 and bge-m3 land together instead of
// in a relative ./local_cache that scatters per launch dir. Override (e.g. Docker → /data/models).
export const LOCAL_EMBEDDING_CACHE_DIR =
  process.env.LOCAL_EMBEDDING_CACHE_DIR ?? join(homeBaseDir(), "models");

const EMBED_RETRIES = 3;
type EmbeddingInputType = "passage" | "query";

const LOCAL_MODEL_ALIASES: Record<string, string> = {
  "fast-multilingual-e5-large": "MLE5Large",
  "multilingual-e5-large": "MLE5Large",
  mle5large: "MLE5Large",
  "fast-bge-base-en-v1.5": "BGEBaseENV15",
  "bge-base-en-v1.5": "BGEBaseENV15",
  bgebaseenv15: "BGEBaseENV15",
  "fast-bge-small-en-v1.5": "BGESmallENV15",
  "bge-small-en-v1.5": "BGESmallENV15",
  bgesmallenv15: "BGESmallENV15",
  "fast-all-minilm-l6-v2": "AllMiniLML6V2",
  "all-minilm-l6-v2": "AllMiniLML6V2",
  allminilml6v2: "AllMiniLML6V2",
  "bge-m3": "CUSTOM",
  bgem3: "CUSTOM",
  "baai/bge-m3": "CUSTOM",
  "fast-bge-m3": "CUSTOM",
};

let _openaiClient: OpenAI | null = null;

function getOpenAIClient(): OpenAI {
  if (!_openaiClient) {
    _openaiClient = new OpenAI({ apiKey: OPENAI_API_KEY });
  }
  return _openaiClient;
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function normalizeModelName(modelName: string): string {
  return modelName
    .trim()
    .replace(/^EmbeddingModel\./i, "")
    .toLowerCase()
    .replace(/[\s_]/g, "-");
}

function resolveLocalModel(
  modelName: string,
  embeddingModel: Record<string, string>
): string {
  const normalized = normalizeModelName(modelName);
  const aliasKey = LOCAL_MODEL_ALIASES[normalized];
  if (aliasKey && aliasKey in embeddingModel) return embeddingModel[aliasKey];

  for (const [key, value] of Object.entries(embeddingModel)) {
    if (key.toLowerCase() === modelName.trim().toLowerCase()) return value;
    if (value.toLowerCase() === normalized) return value;
  }

  const supported = Object.values(embeddingModel)
    .filter((value) => value !== embeddingModel.CUSTOM)
    .join(", ");
  throw new Error(
    `Unsupported LOCAL_EMBEDDING_MODEL "${modelName}". Supported values: ${supported}`
  );
}

// ── Key threshold profiles ──
// Cosine distributions differ by model; e5 needs higher key thresholds to avoid
// merging unrelated concepts in its narrow high-cosine band.
export interface ThresholdProfile {
  keyMerge: number;
  keyAutoLink: number;
  keyRecall: number;
  // Short concepts merge only at a conservative synonym threshold; 0 disables it.
  shortKeyMerge: number;
}

export const THRESHOLD_PROFILES: Record<string, ThresholdProfile> = {
  openai: { keyMerge: 0.85, keyAutoLink: 0.5, keyRecall: 0.28, shortKeyMerge: 0 },
  bge: { keyMerge: 0.85, keyAutoLink: 0.6, keyRecall: 0.6, shortKeyMerge: 0 },
  e5: { keyMerge: 0.97, keyAutoLink: 0.93, keyRecall: 0.85, shortKeyMerge: 0 },
  minilm: { keyMerge: 0.85, keyAutoLink: 0.6, keyRecall: 0.5, shortKeyMerge: 0 },
  bgem3: { keyMerge: 0.86, keyAutoLink: 0.62, keyRecall: 0.62, shortKeyMerge: 0.9 },
};

export function familyForModel(
  modelName: string
): "e5" | "bge" | "minilm" | "bgem3" | "unknown" {
  const normalized = normalizeModelName(modelName);
  if (["bge-m3", "bgem3", "baai/bge-m3", "fast-bge-m3", "bge_m3"].includes(normalized)) {
    return "bgem3";
  }
  const alias = LOCAL_MODEL_ALIASES[normalized];
  if (alias === "MLE5Large") return "e5";
  if (alias === "AllMiniLML6V2") return "minilm";
  if (alias === "BGEBaseENV15" || alias === "BGESmallENV15") return "bge";
  return "unknown";
}

export function usesE5Prefix(family: string): boolean {
  return family === "e5";
}

// Identifies the embedding vector space the graph was built in. Two models of the
// same dimension (e5-large ↔ bge-m3, both 1024-d) produce INCOMPATIBLE vectors, so
// a dimension check alone cannot detect a backend swap. The graph stores this
// fingerprint; on load a mismatch triggers a re-embed. Read live from env so a
// per-process backend change is reflected even though the module-level consts were
// snapshotted at import. OpenAI variants differ by model name; local by model id.
export function embeddingFingerprint(): string {
  const backend = process.env.EMBEDDING_BACKEND ?? _DEFAULT_BACKEND;
  if (backend !== "local") {
    return `openai:${process.env.OPENAI_EMBEDDING_MODEL ?? "text-embedding-3-small"}`;
  }
  // bge-m3 now embeds through our own session, without fastembed's 512-padding. Same model,
  // but the vector moves (cosine ~0.98 against the padded one), so it is a distinct space and
  // must be named as one — otherwise a graph built on fastembed's padded vectors would be
  // silently compared against unpadded ones. The suffix is what triggers the one-time
  // re-embed on first load after upgrading.
  const suffix = localModelFamily() === "bgem3" ? "+nopad" : "";
  return `local:${normalizeModelName(process.env.LOCAL_EMBEDDING_MODEL ?? "fast-multilingual-e5-large")}${suffix}`;
}

export function customModelConfig(): { dir: string; file: string } {
  const dir = process.env.LOCAL_EMBEDDING_MODEL_PATH ?? "";
  if (!dir.trim()) {
    throw new Error(
      `LOCAL_EMBEDDING_MODEL="${LOCAL_EMBEDDING_MODEL}" resolves to a CUSTOM model, ` +
        `so LOCAL_EMBEDDING_MODEL_PATH (absolute dir containing model.onnx + tokenizer ` +
        `files) is required. Optionally set LOCAL_EMBEDDING_MODEL_FILE (default model.onnx).`
    );
  }
  return { dir, file: process.env.LOCAL_EMBEDDING_MODEL_FILE ?? "model.onnx" };
}

function modelFilePresent(dir: string, file: string): boolean {
  try {
    return existsSync(join(dir, file)) && statSync(join(dir, file)).size > 0;
  } catch {
    return false;
  }
}

// Resolve the local CUSTOM model directory, auto-downloading a known model (bge-m3) when
// its files are missing. Backward compatible: an existing LOCAL_EMBEDDING_MODEL_PATH with
// the model present is used unchanged with NO download; an unknown custom model with no
// usable path still throws the original guidance. Only the local backend's CUSTOM path
// calls this — OpenAI and fastembed built-ins never do.
export async function ensureCustomEmbeddingModel(fetcher?: Fetcher): Promise<{ dir: string; file: string }> {
  const fileEnv = process.env.LOCAL_EMBEDDING_MODEL_FILE;
  const file = fileEnv ?? "model.onnx";
  const envPath = (process.env.LOCAL_EMBEDDING_MODEL_PATH ?? "").trim();

  // 1. A custom model FILENAME means the user manages this dir themselves → trust it
  //    entirely with no download (preserves the original behavior for bespoke setups).
  if (fileEnv && envPath && modelFilePresent(envPath, file)) return { dir: envPath, file };

  // 2. A known model (bge-m3) → ensure all files, downloading ONLY the missing ones. A
  //    complete dir downloads nothing (backward compatible); a partial/empty one heals.
  if (localModelFamily() === "bgem3") {
    const dir = envPath || defaultModelDir("bge-m3");
    await ensureModelFiles(KNOWN_MODELS["bge-m3"], dir, fetcher);
    return { dir, file: "model.onnx" };
  }

  // 3. Unknown custom model → original path-or-throw guidance.
  return customModelConfig();
}

let _warnedUncalibrated = false;
function localModelFamily(): "e5" | "bge" | "minilm" | "bgem3" {
  const fam = familyForModel(LOCAL_EMBEDDING_MODEL);
  if (fam !== "unknown") return fam;
  // Unknown model: thresholds are NOT calibrated for it. Make the miscalibration
  // loud and point at the env override escape hatch (see original warning).
  if (!_warnedUncalibrated) {
    _warnedUncalibrated = true;
    console.error(
      `[keymem] WARNING: no calibrated threshold profile for ` +
        `LOCAL_EMBEDDING_MODEL="${LOCAL_EMBEDDING_MODEL}". Falling back to BGE ` +
        `thresholds — the graph may mis-cluster. Override per-threshold with ` +
        `KEYMEM_KEY_MERGE / _MEMORY_DEDUP / _KEY_AUTOLINK / _KEY_RECALL / _CONTENT_RECALL / _MIN_SCORE / _CONTRADICTION.`
    );
  }
  return "bge";
}

function envThreshold(suffix: string): number | undefined {
  const raw = cfgRaw(suffix);
  if (raw === undefined || raw.trim() === "") return undefined;
  const n = Number(raw);
  if (!Number.isFinite(n) || n < 0 || n > 1) {
    console.error(`[keymem] WARNING: ignoring ${cfgName(suffix)}="${raw}" (must be a number in [0,1]).`);
    return undefined;
  }
  return n;
}

// Per-threshold env overrides are an escape hatch for when the model or the
// character of stored data drifts away from what the built-in profiles assume.
export function getThresholdProfile(): ThresholdProfile {
  const base =
    EMBEDDING_BACKEND !== "local"
      ? THRESHOLD_PROFILES.openai
      : THRESHOLD_PROFILES[localModelFamily()];
  return {
    keyMerge: envThreshold("KEY_MERGE") ?? base.keyMerge,
    keyAutoLink: envThreshold("KEY_AUTOLINK") ?? base.keyAutoLink,
    keyRecall: envThreshold("KEY_RECALL") ?? base.keyRecall,
    shortKeyMerge: envThreshold("SHORT_KEY_MERGE") ?? base.shortKeyMerge,
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _localModel: any = null;

let _testEmbedder:
  | ((text: string, inputType: EmbeddingInputType) => number[])
  | null = null;

// Test-only seam. Lets tests drive embedding-dependent code paths with crafted
// vectors so cosine similarities are deterministic and no ONNX model is loaded.
// Never set this in production code.
export function __setTestEmbedder(
  fn: (text: string, inputType: EmbeddingInputType) => number[]
): void {
  _testEmbedder = fn;
}

export function __clearTestEmbedder(): void {
  _testEmbedder = null;
}

async function getLocalModel() {
  if (!_localModel) {
    try {
      const { FlagEmbedding, EmbeddingModel } = await import("fastembed");
      const model = resolveLocalModel(LOCAL_EMBEDDING_MODEL, EmbeddingModel);
      const initOpts: Record<string, unknown> = {
        model: model as never,
        cacheDir: LOCAL_EMBEDDING_CACHE_DIR,
      };
      if (model === EmbeddingModel.CUSTOM) {
        const { dir, file } = await ensureCustomEmbeddingModel();
        initOpts.modelAbsoluteDirPath = dir;
        initOpts.modelName = file;
      }
      _localModel = await FlagEmbedding.init(initOpts as never);
    } catch (err) {
      throw new Error(
        "Failed to initialize local fastembed model.\n" +
          "Install with: npm install fastembed\n" +
          "Or set OPENAI_API_KEY to use OpenAI embeddings.\n" +
          `Cause: ${errorMessage(err)}`
      );
    }
  }
  return _localModel;
}

// ── bge-m3: our own ONNX session, without fastembed's fixed-512 padding ───────
// fastembed pads EVERY input to maxLength (fastembed.js:220), then CLS-pools the first token
// and L2-normalizes (fastembed.js:255) — so a 4-token query costs exactly as much as a full
// page: 199ms / 1.4 core-seconds / a 9.6-core peak per embed, on every single recall.
// Tokenizing to the actual length instead measures 15ms / 0.06 core-seconds: 23x less CPU.
//
// The pooling below is byte-faithful to fastembed — replicating it *with* the padding scores
// cosine 1.000000 against fastembed's own output. Dropping the padding does move the vector
// (cosine 0.980–0.988 against the padded one), so this is a different embedding space and
// embeddingFingerprint() names it, which makes an existing graph re-embed itself on load.
// Retrieval is unaffected: bench 92%/97%/0.93, the ablation grid, and real-eval on a copy of
// a 3018-vector live store all scored identically to the padded path, case for case.
let _bgeSession: { session: any; tokenizer: any; ort: any } | null = null;

async function getBgeM3Session() {
  if (_bgeSession) return _bgeSession;
  const { createRequire } = await import("node:module");
  const { readFileSync } = await import("node:fs");
  const { dir, file } = await ensureCustomEmbeddingModel();
  // onnxruntime-node and the tokenizer are fastembed's own native deps, resolved through it:
  // if these cannot load, fastembed could not have run either, so there is nothing to fall
  // back to. Failing loudly beats silently embedding into the padded space, which would be
  // compared against nopad vectors and quietly degrade every score.
  const feReq = createRequire(createRequire(import.meta.url).resolve("fastembed"));
  const ort = feReq("onnxruntime-node");
  const tk = feReq("@anush008/tokenizers");
  const tokenizer = tk.Tokenizer.fromFile(join(dir, "tokenizer.json"));
  tokenizer.setTruncation(512); // no setPadding — that is the whole point
  // fastembed registers these explicitly; replicating it is part of what makes the padded
  // variant match at cosine 1.000000, so keep it.
  const map = JSON.parse(readFileSync(join(dir, "special_tokens_map.json"), "utf-8"));
  for (const token of Object.values(map)) {
    if (typeof token === "string") tokenizer.addSpecialTokens([token]);
  }
  const session = await ort.InferenceSession.create(join(dir, file), {
    intraOpNumThreads: modelThreads(availableParallelism(), cfgRaw("EMBED_THREADS")),
  });
  _bgeSession = { session, tokenizer, ort };
  return _bgeSession;
}

/**
 * CLS pooling + L2 normalization over a [1, tokens, dim] last_hidden_state.
 * The CLS vector is the FIRST token, so it is the leading `dim` floats — get this slice
 * wrong and every vector is quietly wrong in a way no type or test signature would catch.
 */
export function clsPool(hidden: ArrayLike<number>, dim: number): number[] {
  const cls = Array.prototype.slice.call(hidden, 0, dim) as number[];
  const norm = Math.sqrt(cls.reduce((acc, v) => acc + v * v, 0));
  return cls.map((v) => v / Math.max(norm, 1e-12));
}

async function embedBgeM3(text: string): Promise<number[]> {
  const { session, tokenizer, ort } = await getBgeM3Session();
  const enc = await tokenizer.encode(text);
  const ids = enc.getIds() as number[];
  const n = ids.length;
  const big = (a: number[]) => BigInt64Array.from(a.map((x) => BigInt(x)));
  const out = await session.run({
    input_ids: new ort.Tensor("int64", big(ids), [1, n]),
    attention_mask: new ort.Tensor("int64", big(enc.getAttentionMask() as number[]), [1, n]),
    token_type_ids: new ort.Tensor("int64", big(enc.getTypeIds() as number[]), [1, n]),
  });
  const hidden = out.last_hidden_state;
  return clsPool(hidden.data as Float32Array, hidden.dims[2] as number);
}

async function embedLocal(
  text: string,
  inputType: EmbeddingInputType
): Promise<number[]> {
  // bge-m3 owns its session (no prefix, no padding); every other family stays on fastembed.
  if (localModelFamily() === "bgem3") return embedBgeM3(text);
  const model = await getLocalModel();
  if (inputType === "query" && typeof model.queryEmbed === "function") {
    return Array.from(await model.queryEmbed(text)) as number[];
  }
  const gen =
    typeof model.passageEmbed === "function"
      ? model.passageEmbed([text], 256)
      : model.embed([text]);
  for await (const batch of gen) {
    return Array.from(batch[0]) as number[];
  }
  throw new Error("fastembed returned no embeddings");
}

// Short concept keys (e.g. "Agent A" / "Agent B") embed almost identically and
// would over-merge under semantic matching, conflating distinct entities. Treat
// them like proper nouns: merge only on exact string match. Tunable.
export const SHORT_CONCEPT_MAX_TOKENS = 2;
export const SHORT_CONCEPT_MAX_CHARS = 15;

// Contradiction band: similar enough to be about the same subject, but below the
// dedup threshold so it is a distinct (possibly conflicting) fact, not a paraphrase.
export function inContradictionBand(sim: number, floor: number, dedup: number): boolean {
  return sim >= floor && sim < dedup;
}

export function isShortConcept(concept: string): boolean {
  const trimmed = concept.trim();
  const tokens = trimmed.split(/\s+/).filter(Boolean);
  return tokens.length <= SHORT_CONCEPT_MAX_TOKENS || trimmed.length <= SHORT_CONCEPT_MAX_CHARS;
}

export function embedTextAsync(
  text: string,
  inputType: EmbeddingInputType = "passage"
): Promise<number[]> {
  return timed(`embed:${inputType}(${text.length}ch)`, () => embedTextImpl(text, inputType));
}

async function embedTextImpl(text: string, inputType: EmbeddingInputType): Promise<number[]> {
  if (_testEmbedder) return _testEmbedder(text, inputType);
  if (EMBEDDING_BACKEND === "local") {
    return embedLocal(text, inputType);
  }

  const client = getOpenAIClient();
  for (let attempt = 0; attempt < EMBED_RETRIES; attempt++) {
    try {
      const resp = await client.embeddings.create({
        model: OPENAI_EMBEDDING_MODEL,
        input: text,
      });
      return resp.data[0].embedding;
    } catch (err) {
      if (attempt === EMBED_RETRIES - 1) throw err;
      await new Promise((resolve) =>
        setTimeout(resolve, Math.pow(2, attempt) * 1000)
      );
    }
  }
  throw new Error("unreachable");
}
