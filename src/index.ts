#!/usr/bin/env node
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { graph, createMcpServer } from "./server.js";

export async function runInProcess(): Promise<void> {
  await graph.load();
  const server = createMcpServer();
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

// Only run when this module is the CLI entry point directly (so the shim's
// fallback import doesn't auto-run the server).
if (import.meta.url === `file://${process.argv[1]}`) {
  runInProcess().catch((err) => {
    console.error("[fatal]", err);
    process.exit(1);
  });
}
