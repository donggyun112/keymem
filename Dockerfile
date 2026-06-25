# keymem — associative key-graph memory MCP server (stdio).
# Single stage keeps native deps (fastembed/onnxruntime) simple; dev deps are pruned after build.
FROM node:22-slim

WORKDIR /app

# Install deps without running the `prepare` (tsc) hook — src isn't present yet at install time.
COPY package.json ./
RUN npm install --ignore-scripts

# Build the TypeScript server, then drop dev-only deps (typescript/tsx) from the image.
COPY tsconfig.json ./
COPY src ./src
RUN npm run build && npm prune --omit=dev

COPY README.md LICENSE ./

ENV NODE_ENV=production
# MCP server speaks JSON-RPC over stdio; this responds to initialize/tools-list introspection.
# Embedding model (bge-m3) is loaded lazily on first recall/remember, not at startup.
ENTRYPOINT ["node", "dist/index.js"]
