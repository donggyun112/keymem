# keymem — associative key-graph memory MCP server (stdio).
# Pinned to amd64: fastembed's @anush008/tokenizers ships native binaries only for linux-x64-gnu,
# darwin-universal, and win32-x64-msvc — there is NO linux-arm64-gnu, so an arm64 image cannot embed
# at runtime. amd64 runs natively on x64 hosts/CI (incl. Glama) and under emulation on Apple Silicon.
FROM --platform=linux/amd64 node:22-slim

WORKDIR /app

# Copy sources BEFORE install so the `prepare` (tsc) build hook can run, AND native postinstall
# scripts are executed. Do NOT pass --ignore-scripts: fastembed's @anush008/tokenizers places a
# platform-specific native binary in a postinstall; skipping it makes the tokenizer module go
# missing and every embed (recall/remember) fail at runtime, even though introspection still works.
COPY package.json tsconfig.json ./
COPY src ./src
RUN npm install && npm prune --omit=dev

COPY README.md LICENSE ./

# Pin all writable state to one volume so it is deterministic and survives restarts:
#   /data               — graph.json + conversation store (KEYMEM_DATA_DIR)
#   /data/models        — fastembed cache for built-in models, e.g. e5 (LOCAL_EMBEDDING_CACHE_DIR)
#   /data/models/bge-m3 — bge-m3 ONNX files when LOCAL_EMBEDDING_MODEL=bge-m3 (LOCAL_EMBEDDING_MODEL_PATH)
# Without these the model lands in a relative ./local_cache and re-downloads (~500MB+) every run.
ENV NODE_ENV=production \
    KEYMEM_DATA_DIR=/data \
    LOCAL_EMBEDDING_CACHE_DIR=/data/models \
    LOCAL_EMBEDDING_MODEL_PATH=/data/models/bge-m3
RUN mkdir -p /data/models
VOLUME ["/data"]

# MCP server speaks JSON-RPC over stdio; responds to initialize/tools-list introspection instantly.
# The embedding model downloads lazily to /data/models on first recall/remember, not at startup.
ENTRYPOINT ["node", "dist/index.js"]
