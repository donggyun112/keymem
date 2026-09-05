// Auto-download for the local CUSTOM embedding model (bge-m3), with strict backward
// compatibility: a complete model dir downloads NOTHING; a partial/empty one fetches only
// the missing files (self-healing). The fetcher is injected so these run offline. Online-API
// backends and fastembed built-ins never call ensureCustomEmbeddingModel.
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

let n = 0;
const ALL = ["model.onnx", "tokenizer.json", "tokenizer_config.json", "config.json", "special_tokens_map.json"];

test("backward compat: a COMPLETE model dir downloads nothing", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-mdl-full-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  for (const f of ALL) await writeFile(join(dir, f), "stub");
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  process.env.LOCAL_EMBEDDING_MODEL_PATH = dir;
  delete process.env.LOCAL_EMBEDDING_MODEL_FILE;

  const emb = await import(`../src/embedding.ts?dl=${n++}`);
  let calls = 0;
  const r = await emb.ensureCustomEmbeddingModel(async () => { calls++; });
  assert.equal(r.dir, dir);
  assert.equal(calls, 0, "a complete dir must not download anything");
});

test("auto-download: an EMPTY dir fetches model + tokenizer + config", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-mdl-empty-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  process.env.LOCAL_EMBEDDING_MODEL_PATH = dir;
  delete process.env.LOCAL_EMBEDDING_MODEL_FILE;

  const emb = await import(`../src/embedding.ts?dl=${n++}`);
  const fetched: string[] = [];
  await emb.ensureCustomEmbeddingModel(async (_u: string, dest: string) => { fetched.push(dest); writeFileSync(dest, "x"); });
  assert.ok(fetched.some((d) => d.endsWith("model.onnx")), "model.onnx fetched");
  assert.ok(fetched.some((d) => d.endsWith("tokenizer.json")), "tokenizer fetched");
  assert.ok(fetched.length >= 3, `expected >=3 files, got ${fetched.length}`);
});

test("self-heal: a PARTIAL dir (model only) fetches just the missing tokenizer/config", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-mdl-partial-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  await writeFile(join(dir, "model.onnx"), "stub"); // model present, tokenizer/config missing
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  process.env.LOCAL_EMBEDDING_MODEL_PATH = dir;
  delete process.env.LOCAL_EMBEDDING_MODEL_FILE;

  const emb = await import(`../src/embedding.ts?dl=${n++}`);
  const fetched: string[] = [];
  await emb.ensureCustomEmbeddingModel(async (_u: string, dest: string) => { fetched.push(dest); writeFileSync(dest, "x"); });
  assert.ok(!fetched.some((d) => d.endsWith("model.onnx")), "model.onnx already present → not re-downloaded");
  assert.ok(fetched.some((d) => d.endsWith("tokenizer.json")), "missing tokenizer must be fetched");
});

test("single-flight: concurrent callers share one download per file", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "sm-mdl-race-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const { ensureModelFiles, KNOWN_MODELS } = await import(`../src/modelDownload.ts?dl=${n++}`);
  const fetched: string[] = [];
  const fetcher = async (_u: string, dest: string) => {
    fetched.push(dest);
    await new Promise((r) => setTimeout(r, 20)); // keep the download in flight
    writeFileSync(dest, "x");
  };
  await Promise.all([1, 2, 3, 4, 5].map(() => ensureModelFiles(KNOWN_MODELS.reranker, dir, fetcher)));
  assert.equal(fetched.length, KNOWN_MODELS.reranker.files.length, `each file fetched once, got ${fetched.length}`);
});

test("httpDownload: a stalled body rejects (no crash) and the next call resumes via Range", async (t) => {
  const { createServer } = await import("node:http");
  const { readFileSync } = await import("node:fs");
  const dir = await mkdtemp(join(tmpdir(), "sm-mdl-resume-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const body = Buffer.from("0123456789");
  let calls = 0;
  const srv = createServer((req, res) => {
    calls++;
    if (calls === 1) {
      res.writeHead(200, { "content-length": body.length });
      res.write(body.subarray(0, 4));
      setTimeout(() => res.destroy(), 20); // stall, then drop the connection
      return;
    }
    const from = Number(/bytes=(\d+)-/.exec(String(req.headers.range))?.[1] ?? 0);
    res.writeHead(206);
    res.end(body.subarray(from));
  });
  await new Promise<void>((r) => srv.listen(0, r));
  t.after(() => srv.close());
  const url = `http://127.0.0.1:${(srv.address() as { port: number }).port}/model.onnx`;
  const { httpDownload } = await import(`../src/modelDownload.ts?dl=${n++}`);
  const dest = join(dir, "model.onnx");
  await assert.rejects(httpDownload(url, dest), "first attempt must reject, not throw unhandled");
  await httpDownload(url, dest);
  assert.equal(readFileSync(dest, "utf8"), "0123456789");
  assert.equal(calls, 2);
});
