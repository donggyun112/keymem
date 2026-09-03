import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import test from "node:test";
import { freshDecayGraph } from "./decay-test-utils.js";

test("read is validity-neutral and confirmation refreshes exactly once", async (t) => {
  let now = 1_800_000_000;
  const { graph, dir } = await freshDecayGraph(t, () => now);
  const [mid] = await graph.add("사용자는 서울에 산다", ["거주지"]);
  const initial = { ...graph.memories[mid] };
  now += 365 * 24 * 3600;

  const read = (await graph.readMemory(mid)) as any;
  assert.equal(graph.memories[mid].access_count, initial.access_count + 1);
  assert.equal(graph.memories[mid].depth, initial.depth);
  assert.equal(graph.memories[mid].last_confirmed_at, initial.last_confirmed_at);
  assert.equal(graph.memories[mid].confirmation_count, initial.confirmation_count);
  assert.equal(read.memory.validity.status, "stale");

  await assert.rejects(
    () => graph.confirmMemory(mid, { evidence: "rumor" as any }),
    /Unknown confirmation evidence/
  );
  assert.equal(graph.memories[mid].confirmation_count, initial.confirmation_count);

  const source = { reason: "user said this is still current" };
  const first = (await graph.confirmMemory(mid, {
    evidence: "user",
    source,
    confirmationId: "codex:session-1:turn-7",
  })) as any;
  assert.equal(graph.memories[mid].last_confirmed_at, now);
  assert.equal(graph.memories[mid].confirmation_count, initial.confirmation_count + 1);
  assert.equal(graph.memories[mid].depth, initial.depth + 0.05);
  assert.equal(graph.memories[mid].last_confirmation_evidence, "user");
  assert.deepEqual(graph.memories[mid].last_confirmation_source, source);
  assert.equal(graph.memories[mid].last_confirmation_id, "codex:session-1:turn-7");
  assert.equal(first.validity.status, "fresh");

  const persisted = JSON.parse(await readFile(join(dir, "graph.json"), "utf8")).memories[mid];
  assert.equal(persisted.last_confirmation_evidence, "user");
  assert.deepEqual(persisted.last_confirmation_source, source);
  assert.equal(persisted.last_confirmation_id, "codex:session-1:turn-7");

  now += 100;
  const duplicate = (await graph.confirmMemory(mid, {
    evidence: "user",
    confirmationId: "codex:session-1:turn-7",
  })) as any;
  assert.equal(graph.memories[mid].last_confirmed_at, now - 100);
  assert.equal(graph.memories[mid].confirmation_count, initial.confirmation_count + 1);
  assert.equal(duplicate.confirmed, false);
  assert.equal(duplicate.duplicate, true);
});

test("confirmation hides missing, expired, superseded, and cross-namespace ids", async (t) => {
  let now = 1_800_000_000;
  const vectors: Record<string, number[]> = {
    expiring: [1, 0, 0, 0],
    old: [0, 1, 0, 0],
    replacement: [0, 0, 1, 0],
    private: [0, 0, 0, 1],
    expiring_key: [1, 1, 0, 0],
    old_key: [1, 0, 1, 0],
    private_key: [1, 0, 0, 1],
  };
  const { graph } = await freshDecayGraph(
    t,
    () => now,
    (text) => vectors[text] ?? [0.5, 0.5, 0.5, 0.5]
  );
  const [expiredId] = await graph.add("expiring", ["expiring_key"], { ttlSeconds: 1 });
  const [oldId] = await graph.add("old", ["old_key"]);
  const [privateId] = await graph.add("private", ["private_key"], { namespace: "private" });
  const replacementId = await graph.supersede(oldId, "replacement");
  now += 2;

  await assert.rejects(() => graph.confirmMemory("missing", { evidence: "user" }), /not found/);
  await assert.rejects(() => graph.confirmMemory(expiredId, { evidence: "user" }), /not found/);
  await assert.rejects(() => graph.confirmMemory(oldId, { evidence: "user" }), /not found/);
  await assert.rejects(
    () => graph.confirmMemory(privateId, { evidence: "user", namespace: "default" }),
    /not found/
  );
  assert.equal(graph.memories[replacementId].confirmation_count, 1);
});

test("an empty confirmation id is still idempotent", async (t) => {
  let now = 1_800_000_000;
  const { graph } = await freshDecayGraph(t, () => now);
  const [mid] = await graph.add("사용자는 서울에 산다", ["거주지"]);
  const initialCount = graph.memories[mid].confirmation_count;

  await graph.confirmMemory(mid, { evidence: "user", confirmationId: "" });
  const confirmedAt = graph.memories[mid].last_confirmed_at;
  now += 100;
  const duplicate = (await graph.confirmMemory(mid, {
    evidence: "observation",
    confirmationId: "",
  })) as any;

  assert.equal(duplicate.confirmed, false);
  assert.equal(duplicate.duplicate, true);
  assert.equal(graph.memories[mid].last_confirmed_at, confirmedAt);
  assert.equal(graph.memories[mid].confirmation_count, initialCount + 1);
  assert.equal(graph.memories[mid].last_confirmation_evidence, "user");
});

test("confirmation depth is capped at one from near or at the maximum", async (t) => {
  const { graph } = await freshDecayGraph(t, () => 1_800_000_000);
  const [mid] = await graph.add("사용자는 서울에 산다", ["거주지"]);

  graph.memories[mid].depth = 0.98;
  await graph.confirmMemory(mid, { evidence: "observation" });
  assert.equal(graph.memories[mid].depth, 1);

  await graph.confirmMemory(mid, { evidence: "authoritative_source" });
  assert.equal(graph.memories[mid].depth, 1);
});
