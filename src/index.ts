#!/usr/bin/env node
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { realpathSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { graph, createMcpServer } from "./server.js";

export async function runInProcess(): Promise<void> {
  await graph.load();
  const server = createMcpServer();
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

// Only run when this module is the CLI entry point directly (so the shim's
// fallback import doesn't auto-run the server). Resolve argv[1] to its
// realpath before comparing: when invoked via a symlinked bin (e.g. the
// package.json `keymem-*` bins), process.argv[1] is the symlink path while
// import.meta.url is realpath-resolved, so a naive string compare is falsy
// and main() silently never runs.
function isCliEntry(): boolean {
  return process.argv[1] !== undefined && import.meta.url === pathToFileURL(realpathSync(process.argv[1])).href;
}

if (isCliEntry()) {
  runInProcess().catch((err) => {
    console.error("[fatal]", err);
    process.exit(1);
  });
}
