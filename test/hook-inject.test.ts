import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { createServer } from "node:http";
import test from "node:test";
import { fileURLToPath } from "node:url";

test("hook renders fresh and stale injected validity with currentness guidance", async (t) => {
  const daemon = createServer(async (req, res) => {
    for await (const _chunk of req) { /* consume request */ }
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({
      memories: [
        {
          id: "fresh-memory",
          content: "The user prefers tea.",
          relevance_score: 0.95,
          namespace: "default",
          validity: {
            freshness: 1,
            status: "fresh",
            age_days: 0,
            last_confirmed_at: 2,
            confirmation_count: 2,
            decay_profile: "standard",
            verification_recommended: false,
            verification_required: false,
          },
        },
        {
          id: "stale-memory",
          content: "The user lives in Busan.",
          relevance_score: 0.9,
          namespace: "default",
          validity: {
            freshness: 0.1,
            status: "stale",
            age_days: 300,
            last_confirmed_at: 1,
            confirmation_count: 1,
            decay_profile: "standard",
            verification_recommended: true,
            verification_required: true,
          },
        },
      ],
    }));
  });
  daemon.listen(0, "127.0.0.1");
  await once(daemon, "listening");
  t.after(() => new Promise<void>((resolve, reject) => {
    daemon.close((error) => error ? reject(error) : resolve());
  }));

  const address = daemon.address();
  assert.ok(address && typeof address === "object");
  const child = spawn(
    fileURLToPath(new URL("../node_modules/.bin/tsx", import.meta.url)),
    [fileURLToPath(new URL("../src/hook.ts", import.meta.url))],
    {
      env: {
        ...process.env,
        KEYMEM_DAEMON_PORT: String(address.port),
        KEYMEM_HOOK_TIMEOUT_MS: "2000",
      },
      stdio: ["pipe", "pipe", "pipe"],
    }
  );
  let stdout = "";
  let stderr = "";
  child.stdout.setEncoding("utf-8").on("data", (chunk) => { stdout += chunk; });
  child.stderr.setEncoding("utf-8").on("data", (chunk) => { stderr += chunk; });
  child.stdin.end(JSON.stringify({ prompt: "Where does the user live?" }));
  const [code] = await once(child, "close");
  assert.equal(code, 0, stderr);

  const payload = JSON.parse(stdout);
  const context = payload.hookSpecificOutput.additionalContext as string;
  assert.match(
    context,
    /- \[fresh-memory[^\n]*validity=\{[^}\n]*"status":"fresh"[^}\n]*"verification_recommended":false[^}\n]*"verification_required":false\}/
  );
  assert.match(context, /use fresh memories normally/i);
  assert.match(context, /validity=\{[^}]*"status":"stale"/);
  assert.match(
    context,
    /- \[stale-memory[^\n]*validity=\{[^}\n]*"status":"stale"[^}\n]*"verification_recommended":true/
  );
  assert.match(context, /"verification_required":true/);
  assert.match(context, /read_memory.*does not confirm.*current/is);
  assert.match(context, /stale.*(?:verify externally|ask the user)/is);
  assert.doesNotMatch(context, /verify with read_memory.*before asserting/i);
});
