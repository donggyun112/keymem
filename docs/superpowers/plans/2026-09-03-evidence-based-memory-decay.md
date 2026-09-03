# Evidence-Based Memory Decay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build evidence-based soft forgetting that marks stale memories explicitly, separates access from confirmation, ranks freshness consistently, and prevents expired records from poisoning new writes.

**Architecture:** Put deterministic half-life and validity calculations in a new pure `src/decay.ts` module. Keep persistence and state transitions in `MemoryGraph`, expose the confirmation operation and behavior contract from `server.ts`, and drive every change with a fake-clock RED→GREEN test. Existing memories stay retrievable unless TTL-expired or superseded.

**Tech Stack:** TypeScript ESM, Node.js 20+, Node test runner via `tsx --test`, `@modelcontextprotocol/sdk` in-memory transport, JSON graph persistence.

**Spec:** `docs/superpowers/specs/2026-09-03-evidence-based-memory-decay-design.md`

## Global Constraints

- Age or freshness never automatically deletes or excludes a memory; explicit TTL is the only automatic hard expiration.
- Existing MCP tool names and arguments remain valid; new response fields and `confirm_memory` are additive.
- `read_memory` remains allowed to reinforce the traversed key association, but it must not change content depth, confirmation count, or confirmation time.
- `confirm_memory` is the only read-side operation that refreshes content validity.
- Candidate inclusion stays relevance-driven; the freshness ranking factor has a `0.2` floor.
- Link-weight time decay, event sourcing, and server-side truth judging are out of scope.
- `KEYMEM_` configuration names are primary and `SUPER_MEMORY_` aliases continue through `cfgRaw`.
- No new runtime dependency is introduced.
- Every production behavior is written test-first and observed failing for the intended reason.

---

### Task 1: Pure half-life and validity policy

**Files:**
- Create: `src/decay.ts`
- Create: `test/decay.test.ts`

**Interfaces:**
- Produces: `DecayProfile`, `ConfirmationEvidence`, `FreshnessStatus`, `DecayConfig`, `ValidityView`.
- Produces: `loadDecayConfig(read?, warn?)`, `parseDecayProfile(value)`, `computeFreshness(lastConfirmedAt, profile, now, config?)`, `freshnessRankFactor(freshness)`, and `buildValidityView(memory, now, config?)`.
- Consumes: `cfgRaw` from `src/env.ts`; no graph types or state.

- [ ] **Step 1: Write failing formula, boundary, configuration, and clock-skew tests**

Create `test/decay.test.ts` with deterministic assertions:

```ts
import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_DECAY_CONFIG,
  buildValidityView,
  computeFreshness,
  freshnessRankFactor,
  loadDecayConfig,
  parseDecayProfile,
} from "../src/decay.js";

const DAY = 24 * 3600;
const confirmedAt = 1_800_000_000;

test("freshness follows exact half-life boundaries", () => {
  assert.equal(computeFreshness(confirmedAt, "standard", confirmedAt), 1);
  assert.equal(computeFreshness(confirmedAt, "standard", confirmedAt + 90 * DAY), 0.5);
  assert.equal(computeFreshness(confirmedAt, "standard", confirmedAt + 270 * DAY), 0.125);
});

test("validity status makes aging and stale memories actionable", () => {
  const base = { last_confirmed_at: confirmedAt, confirmation_count: 2, decay_profile: "standard" as const };
  assert.equal(buildValidityView(base, confirmedAt + 90 * DAY).status, "fresh");
  assert.equal(buildValidityView(base, confirmedAt + 270 * DAY).status, "aging");
  const stale = buildValidityView(base, confirmedAt + 270 * DAY + 1);
  assert.equal(stale.status, "stale");
  assert.equal(stale.verification_recommended, true);
  assert.equal(stale.verification_required, true);
});

test("future confirmation timestamps clamp to zero age", () => {
  const view = buildValidityView(
    { last_confirmed_at: confirmedAt + DAY, confirmation_count: 1, decay_profile: "standard" },
    confirmedAt
  );
  assert.equal(view.age_days, 0);
  assert.equal(view.freshness, 1);
});

test("permanent memories do not decay", () => {
  assert.equal(computeFreshness(confirmedAt, "permanent", confirmedAt + 100 * 365 * DAY), 1);
});

test("freshness rank factor is bounded between 0.2 and 1", () => {
  assert.equal(freshnessRankFactor(1), 1);
  assert.equal(freshnessRankFactor(0.5), 0.6);
  assert.equal(freshnessRankFactor(0), 0.2);
});

test("decay profiles reject unknown tool input", () => {
  assert.equal(parseDecayProfile(undefined), "standard");
  assert.equal(parseDecayProfile("stable"), "stable");
  assert.throws(() => parseDecayProfile("forever"), /Unknown decay profile/);
});

test("invalid configured half-lives fall back once per field", () => {
  const warnings: string[] = [];
  const values: Record<string, string> = {
    DECAY_TRANSIENT_DAYS: "0",
    DECAY_STANDARD_DAYS: "not-a-number",
    DECAY_STABLE_DAYS: "730",
  };
  const config = loadDecayConfig((key) => values[key], (message) => warnings.push(message));
  assert.equal(config.halfLivesSeconds.transient, DEFAULT_DECAY_CONFIG.halfLivesSeconds.transient);
  assert.equal(config.halfLivesSeconds.standard, DEFAULT_DECAY_CONFIG.halfLivesSeconds.standard);
  assert.equal(config.halfLivesSeconds.stable, 730 * DAY);
  assert.equal(warnings.length, 2);
});
```

- [ ] **Step 2: Run the test and verify RED**

Run: `pnpm exec tsx --test test/decay.test.ts`

Expected: FAIL because `src/decay.ts` does not exist.

- [ ] **Step 3: Implement the pure decay module**

Create `src/decay.ts` with this public shape and formulas:

```ts
import { cfgName, cfgRaw } from "./env.js";

export type DecayProfile = "transient" | "standard" | "stable" | "permanent";
export type ConfirmationEvidence = "user" | "authoritative_source" | "observation";
export type FreshnessStatus = "fresh" | "aging" | "stale";

export interface DecayConfig {
  halfLivesSeconds: Record<Exclude<DecayProfile, "permanent">, number>;
}

export interface ValidityView {
  freshness: number;
  status: FreshnessStatus;
  age_days: number;
  last_confirmed_at: number;
  confirmation_count: number;
  decay_profile: DecayProfile;
  verification_recommended: boolean;
  verification_required: boolean;
}

const DAY = 24 * 3600;
export const DEFAULT_DECAY_CONFIG: DecayConfig = {
  halfLivesSeconds: { transient: 7 * DAY, standard: 90 * DAY, stable: 365 * DAY },
};

export function loadDecayConfig(
  read: (suffix: string) => string | undefined = cfgRaw,
  warn: (message: string) => void = console.error
): DecayConfig {
  const load = (suffix: string, fallback: number): number => {
    const raw = read(suffix);
    if (raw === undefined) return fallback;
    const days = Number(raw);
    if (Number.isFinite(days) && days > 0) return days * DAY;
    warn(`[decay] invalid ${cfgName(suffix)}=${JSON.stringify(raw)}; using ${fallback / DAY} days`);
    return fallback;
  };
  return {
    halfLivesSeconds: {
      transient: load("DECAY_TRANSIENT_DAYS", DEFAULT_DECAY_CONFIG.halfLivesSeconds.transient),
      standard: load("DECAY_STANDARD_DAYS", DEFAULT_DECAY_CONFIG.halfLivesSeconds.standard),
      stable: load("DECAY_STABLE_DAYS", DEFAULT_DECAY_CONFIG.halfLivesSeconds.stable),
    },
  };
}

export const DECAY_CONFIG = loadDecayConfig();

export function parseDecayProfile(value: unknown): DecayProfile {
  if (value === undefined || value === null) return "standard";
  if (value === "transient" || value === "standard" || value === "stable" || value === "permanent") return value;
  throw new Error(`Unknown decay profile: ${String(value)}`);
}

export function computeFreshness(
  lastConfirmedAt: number,
  profile: DecayProfile,
  now: number,
  config: DecayConfig = DECAY_CONFIG
): number {
  if (profile === "permanent") return 1;
  const age = Math.max(0, now - lastConfirmedAt);
  return 2 ** -(age / config.halfLivesSeconds[profile]);
}

export function freshnessRankFactor(freshness: number): number {
  return 0.2 + 0.8 * Math.max(0, Math.min(1, freshness));
}

export function buildValidityView(
  memory: { last_confirmed_at: number; confirmation_count: number; decay_profile: DecayProfile },
  now: number,
  config: DecayConfig = DECAY_CONFIG
): ValidityView {
  const age = Math.max(0, now - memory.last_confirmed_at);
  const raw = computeFreshness(memory.last_confirmed_at, memory.decay_profile, now, config);
  const status: FreshnessStatus = raw >= 0.5 ? "fresh" : raw >= 0.125 ? "aging" : "stale";
  return {
    freshness: Math.round(raw * 1000) / 1000,
    status,
    age_days: Math.round((age / DAY) * 1000) / 1000,
    last_confirmed_at: memory.last_confirmed_at,
    confirmation_count: memory.confirmation_count,
    decay_profile: memory.decay_profile,
    verification_recommended: status !== "fresh",
    verification_required: status === "stale",
  };
}
```

- [ ] **Step 4: Run the focused test and build**

Run: `pnpm exec tsx --test test/decay.test.ts`

Expected: PASS, 7 tests and 0 failures.

Run: `pnpm run build`

Expected: PASS with TypeScript exit code 0.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/decay.ts test/decay.test.ts
git commit -m "feat(decay): define freshness policy"
```

---

### Task 2: Persist confirmation state and inject a deterministic clock

**Files:**
- Modify: `src/types.ts:28-50`
- Modify: `src/memoryGraph.ts:346-389, 821-984, 1104-1160, 1220-1286`
- Modify: `test/memoryGraph.test.ts:19-34`
- Create: `test/decay-test-utils.ts`
- Create: `test/decay-migration.test.ts`

**Interfaces:**
- Consumes: decay types and `parseDecayProfile` from Task 1.
- Produces: `MemoryGraphOptions { now?: () => number }`; `new MemoryGraph(options?)`.
- Produces: required persisted confirmation fields on every in-memory `Memory` and `meta.schemaVersion = 2`.

- [ ] **Step 1: Write a failing legacy migration and fake-clock test**

Create `test/decay-test-utils.ts` first so every later graph hypothesis uses the same real graph setup:

```ts
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

let graphImport = 0;

export async function freshDecayGraph(
  t: { after(fn: () => void | Promise<void>): void },
  now: () => number,
  vector: (text: string) => number[] = () => [1, 0]
) {
  const dir = await mkdtemp(join(tmpdir(), "keymem-decay-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder((text: string) => vector(text));
  t.after(() => embedding.__clearTestEmbedder());
  const module = await import(`../src/memoryGraph.ts?decay-test=${graphImport++}`);
  const graph = new module.MemoryGraph({ now });
  await graph.load();
  return { graph, dir };
}
```

Then create `test/decay-migration.test.ts` with the legacy fixture and fake-clock assertions:

```ts
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { freshDecayGraph } from "./decay-test-utils.js";

let n = 0;

test("v1 memories migrate confirmation state and persist schema version 2", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-decay-migration-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder(() => [1, 0]);
  t.after(() => embedding.__clearTestEmbedder());

  await writeFile(join(dir, "graph.json"), JSON.stringify({
    keys: {},
    links: [],
    memories: {
      m1: {
        id: "m1", content: "legacy", embedding: [1, 0], created_at: 100,
        source: null, supersedes: null, depth: 0.4, access_count: 4,
        last_accessed: 250, namespace: "default", ttl: null, links: [], contradicts: [],
      },
    },
    meta: {},
  }), "utf-8");

  const { MemoryGraph } = await import(`../src/memoryGraph.ts?decay-migration=${n++}`);
  const graph = new MemoryGraph({ now: () => 500 });
  await graph.load();
  assert.deepEqual(
    {
      last_confirmed_at: graph.memories.m1.last_confirmed_at,
      confirmation_count: graph.memories.m1.confirmation_count,
      decay_profile: graph.memories.m1.decay_profile,
      last_confirmation_evidence: graph.memories.m1.last_confirmation_evidence,
      last_confirmation_source: graph.memories.m1.last_confirmation_source,
      last_confirmation_id: graph.memories.m1.last_confirmation_id,
    },
    {
      last_confirmed_at: 250,
      confirmation_count: 4,
      decay_profile: "standard",
      last_confirmation_evidence: null,
      last_confirmation_source: null,
      last_confirmation_id: null,
    }
  );
  await graph.flush();
  const saved = JSON.parse(await readFile(join(dir, "graph.json"), "utf-8"));
  assert.equal(saved.meta.schemaVersion, 2);
  assert.equal(saved.memories.m1.last_confirmed_at, 250);
});

test("new memory timestamps come from the injected epoch-seconds clock", async (t) => {
  const { graph } = await freshDecayGraph(t, () => 1_800_000_000);
  const [id] = await graph.add("clock fact", ["clock"]);
  assert.equal(graph.memories[id].created_at, 1_800_000_000);
  assert.equal(graph.memories[id].last_confirmed_at, 1_800_000_000);
  assert.equal(graph.memories[id].last_accessed, 1_800_000_000);
});
```

- [ ] **Step 2: Run the migration test and verify RED**

Run: `pnpm exec tsx --test test/decay-migration.test.ts`

Expected: FAIL because the constructor does not accept options and migrated memories lack confirmation fields/schema version.

- [ ] **Step 3: Add persisted types, clock, migration, and initialization**

In `src/types.ts`, import decay types and append these exact properties to `Memory`:

```ts
import type { ConfirmationEvidence, DecayProfile } from "./decay.js";

export interface Memory {
  last_confirmed_at: number;
  confirmation_count: number;
  decay_profile: DecayProfile;
  last_confirmation_evidence: ConfirmationEvidence | null;
  last_confirmation_source: Record<string, unknown> | null;
  last_confirmation_id: string | null;
}

// Replace GraphData.meta's current type with:
meta?: { embeddingFingerprint?: string; schemaVersion?: number };
```

In `MemoryGraph`, add:

```ts
export interface MemoryGraphOptions { now?: () => number; }

private readonly _now: () => number;

constructor(options: MemoryGraphOptions = {}) {
  this._now = options.now ?? (() => Date.now() / 1000);
  this._bm25 = new MiniSearch({
    fields: ["content"], storeFields: [], idField: "id",
    tokenize: (text: string) => text.toLowerCase().split(/[\s\p{P}]+/u).filter((t) => t.length >= 1),
    processTerm: (term: string) => (term.length < 1 ? false : term.toLowerCase()),
  });
}
```

Replace lifecycle/ranking `Date.now() / 1000` calls with `this._now()`. Do not alter transcript timestamps outside `MemoryGraph`.

Add `decayProfile?: DecayProfile` to `add` options. During load, validate/derive confirmation fields
and call `markDirty()` if any are repaired. Use this normalization for each loaded memory (with
`repaired = true` whenever an assignment is made), so corrupt v2 fields recover by the same policy as
missing v1 fields:

```ts
const legacyConfirmedAt = Math.max(
  Number.isFinite(mem.created_at) ? mem.created_at : 0,
  Number.isFinite(mem.last_accessed) ? mem.last_accessed : 0
);
if (!Number.isFinite(mem.last_confirmed_at)) {
  mem.last_confirmed_at = legacyConfirmedAt;
  repaired = true;
}
if (!Number.isFinite(mem.confirmation_count) || mem.confirmation_count < 1) {
  mem.confirmation_count = Math.max(1, Number.isFinite(mem.access_count) ? mem.access_count : 1);
  repaired = true;
}
try {
  mem.decay_profile = parseDecayProfile(mem.decay_profile);
} catch {
  mem.decay_profile = "standard";
  repaired = true;
}
if (!("last_confirmation_evidence" in mem)) { mem.last_confirmation_evidence = null; repaired = true; }
if (!("last_confirmation_source" in mem)) { mem.last_confirmation_source = null; repaired = true; }
if (!("last_confirmation_id" in mem)) { mem.last_confirmation_id = null; repaired = true; }
```

New memories and superseded successors receive:

```ts
last_confirmed_at: now,
confirmation_count: 1,
decay_profile: parseDecayProfile(options.decayProfile),
last_confirmation_evidence: "user",
last_confirmation_source: options.source ?? null,
last_confirmation_id: null,
```

Write `meta: { embeddingFingerprint: fingerprint, schemaVersion: 2 }` in `save()`.

Update the typed `memory()` fixture in `test/memoryGraph.test.ts` with safe v2 defaults.

- [ ] **Step 4: Run focused migration tests and the build**

Run: `pnpm exec tsx --test test/decay-migration.test.ts test/memoryGraph.test.ts test/model-migration.test.ts test/vector-sidecar.test.ts`

Expected: PASS, 0 failures.

Run: `pnpm run build`

Expected: PASS with no TypeScript errors.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/types.ts src/memoryGraph.ts test/memoryGraph.test.ts test/decay-test-utils.ts test/decay-migration.test.ts
git commit -m "feat(decay): persist confirmation state"
```

---

### Task 3: Keep expired records out of deduplication, contradiction, and correction

**Files:**
- Modify: `src/memoryGraph.ts:740-777, 1220-1286`
- Modify: `test/dedup-supersede-surface.test.ts`
- Create: `test/decay-expiration.test.ts`

**Interfaces:**
- Consumes: fake-clock constructor and confirmation-aware `add`/`supersede` from Task 2.
- Produces: `supersede(..., { ttlSeconds?, decayProfile? })` with explicit TTL reset support.
- Preserves: cleanup remains physical maintenance, not a correctness prerequisite.

- [ ] **Step 1: Write failing expired-dedup and expired-correction tests**

Add an injectable clock parameter to the existing `freshGraph` helper in
`test/dedup-supersede-surface.test.ts`, then add:

```ts
test("expired duplicate cannot supersede a newly remembered fact", async (t) => {
  let now = 1_800_000_000;
  const g = await freshGraph(t, () => now);
  const [expiredId] = await g.add("회의는 월요일이다", ["회의"], { ttlSeconds: 1 });
  now += 2;
  const [newId, deduped, superseded] = await g.add(
    "회의는 월요일이다",
    ["회의"],
    { ttlSeconds: 60 }
  );
  assert.notEqual(newId, expiredId);
  assert.equal(deduped, false);
  assert.equal(superseded, null);
  assert.ok(g.listAll().some((m: any) => m.id === newId));
});
```

Create `test/decay-expiration.test.ts` with a deterministic graph fixture and:

```ts
import assert from "node:assert/strict";
import test from "node:test";
import { freshDecayGraph } from "./decay-test-utils.js";

test("supersede rejects an expired source and explicit TTL starts at correction time", async (t) => {
  let now = 1_800_000_000;
  const { graph } = await freshDecayGraph(t, () => now);
  const [expiredId] = await graph.add("temporary", ["temporary"], { ttlSeconds: 1 });
  now += 2;
  await assert.rejects(() => graph.supersede(expiredId, "updated"), /not found/);

  const [liveId] = await graph.add("live", ["live"], { ttlSeconds: 100 });
  now += 10;
  const correctedId = await graph.supersede(liveId, "live updated", { ttlSeconds: 200 });
  assert.equal(graph.memories[correctedId].ttl, now + 200);
});

test("expired memory cannot become a live contradiction", async (t) => {
  let now = 1_800_000_000;
  const vectors: Record<string, number[]> = {
    "expired proposition": [1, 0],
    "current proposition": [0.85, Math.sqrt(1 - 0.85 ** 2)],
    topic: [0, 1],
  };
  const { graph } = await freshDecayGraph(t, () => now, (text) => vectors[text]);
  const [expiredId] = await graph.add("expired proposition", ["topic"], { ttlSeconds: 1 });
  now += 2;
  const [currentId] = await graph.add("current proposition", ["topic"]);
  assert.deepEqual(graph.memories[currentId].contradicts, []);
  assert.ok(!graph.memories[expiredId].contradicts.includes(currentId));
});
```

- [ ] **Step 2: Run both files and verify RED**

Run: `pnpm exec tsx --test test/dedup-supersede-surface.test.ts test/decay-expiration.test.ts`

Expected: FAIL because expired memories still participate in `_findDuplicate()` and `supersede()` has neither expiry rejection nor relative TTL override.

- [ ] **Step 3: Implement the minimal expiry fix**

Change both candidate loops:

```ts
const activeMems = Object.entries(this.memories).filter(
  ([mid, mem]) => !(mid in this._supersededBy) && !this._isExpired(mem)
);
```

In `_findContradiction`, continue when `this._isExpired(mem)`.

At the start of the locked `supersede` transition, treat an expired resolved head as not found. Add
`ttlSeconds?: number | null` and `decayProfile?: DecayProfile` to supersede options, then set:

```ts
ttl: options.ttlSeconds !== undefined
  ? (options.ttlSeconds === null ? null : now + options.ttlSeconds)
  : old.ttl,
decay_profile: options.decayProfile ?? old.decay_profile,
```

In `add()`'s duplicate branch, forward both write policies into the successor so a re-remembered
duplicate cannot silently inherit a different expiry/profile:

```ts
const newId = await this.supersede(dupId, content, {
  keyConcepts,
  keyTypes: options.keyTypes ?? undefined,
  source: options.source,
  namespace: options.namespace,
  relatedTo: options.relatedTo,
  ttlSeconds: options.ttlSeconds,
  decayProfile: options.decayProfile,
});
```

The server wiring for these arguments is deferred to Task 6; this task proves graph behavior.

- [ ] **Step 4: Run focused expiry/dedup tests**

Run: `pnpm exec tsx --test test/dedup-supersede-surface.test.ts test/decay-expiration.test.ts test/correct-key-inheritance.test.ts test/correct-pollution.test.ts`

Expected: PASS, 0 failures.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/memoryGraph.ts test/dedup-supersede-surface.test.ts test/decay-expiration.test.ts
git commit -m "fix(decay): ignore expired write candidates"
```

---

### Task 4: Separate memory access from explicit confirmation

**Files:**
- Modify: `src/memoryGraph.ts:1879-1963`
- Modify: `test/agentic-navigation.test.ts:77-87`
- Modify: `test/autokey-integration.test.ts:136-147`
- Create: `test/confirm-memory.test.ts`

**Interfaces:**
- Produces: `MemoryGraph.confirmMemory(memoryId, options): Promise<object>`.
- `options`: `{ namespace?: string | null; evidence: ConfirmationEvidence; source?: Record<string, unknown> | null; confirmationId?: string | null }`.
- Consumes: `buildValidityView` and the graph clock.

- [ ] **Step 1: Change access expectations and write failing confirmation tests**

In `test/agentic-navigation.test.ts`, change the post-read depth assertion to `0` while keeping
`access_count === 1` and traversed `link_weight === 1.1`.

In `test/autokey-integration.test.ts`, rename the existing read-depth test and assert that two reads
increment access count twice but leave depth and confirmation count unchanged.

Create `test/confirm-memory.test.ts` with a fake-clock graph and these behaviors:

```ts
import assert from "node:assert/strict";
import test from "node:test";
import { freshDecayGraph } from "./decay-test-utils.js";

test("read is validity-neutral and confirmation refreshes exactly once", async (t) => {
  let now = 1_800_000_000;
  const { graph } = await freshDecayGraph(t, () => now);
  const [mid] = await graph.add("사용자는 서울에 산다", ["거주지"]);
  const initial = { ...graph.memories[mid] };
  now += 365 * 24 * 3600;

  await graph.readMemory(mid);
  assert.equal(graph.memories[mid].access_count, initial.access_count + 1);
  assert.equal(graph.memories[mid].depth, initial.depth);
  assert.equal(graph.memories[mid].last_confirmed_at, initial.last_confirmed_at);
  assert.equal(graph.memories[mid].confirmation_count, initial.confirmation_count);

  await assert.rejects(
    () => graph.confirmMemory(mid, { evidence: "rumor" as any }),
    /Unknown confirmation evidence/
  );
  assert.equal(graph.memories[mid].confirmation_count, initial.confirmation_count);

  const first = await graph.confirmMemory(mid, {
    evidence: "user",
    source: { reason: "user said this is still current" },
    confirmationId: "codex:session-1:turn-7",
  }) as any;
  assert.equal(graph.memories[mid].last_confirmed_at, now);
  assert.equal(graph.memories[mid].confirmation_count, initial.confirmation_count + 1);
  assert.equal(graph.memories[mid].depth, initial.depth + 0.05);
  assert.equal(first.validity.status, "fresh");

  now += 100;
  await graph.confirmMemory(mid, {
    evidence: "user",
    confirmationId: "codex:session-1:turn-7",
  });
  assert.equal(graph.memories[mid].last_confirmed_at, now - 100);
  assert.equal(graph.memories[mid].confirmation_count, initial.confirmation_count + 1);
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
  const { graph } = await freshDecayGraph(t, () => now, (text) => vectors[text] ?? [0.5, 0.5, 0.5, 0.5]);
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
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pnpm exec tsx --test test/agentic-navigation.test.ts test/autokey-integration.test.ts test/confirm-memory.test.ts`

Expected: FAIL because reads still deepen memory and `confirmMemory` does not exist.

- [ ] **Step 3: Implement validity-neutral reads and confirmation**

Remove the depth increment from `readMemory`, but retain access timestamp/count, traversed link
reinforcement, alias learning, dirty marking, and deferred persistence. Include
`validity: buildValidityView(mem, this._now())` in the returned memory payload.

Add:

```ts
async confirmMemory(
  memoryId: string,
  options: {
    namespace?: string | null;
    evidence: ConfirmationEvidence;
    source?: Record<string, unknown> | null;
    confirmationId?: string | null;
  }
): Promise<object> {
  return this._lock.runExclusive(async () => {
    if (
      options.evidence !== "user" &&
      options.evidence !== "authoritative_source" &&
      options.evidence !== "observation"
    ) {
      throw new Error(`Unknown confirmation evidence: ${String(options.evidence)}`);
    }
    const mem = this.memories[memoryId];
    const ns = normalizeNamespace(options.namespace);
    if (!mem || this._isExpired(mem) || memoryId in this._supersededBy || (ns && mem.namespace !== ns)) {
      throw new Error(`Memory ${memoryId} not found`);
    }
    if (options.confirmationId && mem.last_confirmation_id === options.confirmationId) {
      return { memory_id: memoryId, confirmed: false, duplicate: true, validity: buildValidityView(mem, this._now()) };
    }
    mem.last_confirmed_at = this._now();
    mem.confirmation_count += 1;
    mem.depth = Math.min(mem.depth + DEPTH_INCREMENT, DEPTH_MAX);
    mem.last_confirmation_evidence = options.evidence;
    mem.last_confirmation_source = options.source ?? null;
    mem.last_confirmation_id = options.confirmationId ?? null;
    await this.save();
    return { memory_id: memoryId, confirmed: true, duplicate: false, validity: buildValidityView(mem, this._now()) };
  });
}
```

- [ ] **Step 4: Run access/confirmation tests**

Run: `pnpm exec tsx --test test/agentic-navigation.test.ts test/autokey-integration.test.ts test/confirm-memory.test.ts test/read-defer.test.ts test/dismiss.test.ts`

Expected: PASS, including unchanged link reinforcement and alias-learning behavior.

- [ ] **Step 5: Commit Task 4**

```bash
git add src/memoryGraph.ts test/agentic-navigation.test.ts test/autokey-integration.test.ts test/confirm-memory.test.ts
git commit -m "feat(decay): separate reads from confirmation"
```

---

### Task 5: Apply freshness and expose validity across every retrieval path

**Files:**
- Modify: `src/memoryGraph.ts:579-584, 1382-1530, 1680-1732, 1971-2032, 2037-2465, 2470-2564, 2588-2608`
- Create: `test/decay-ranking.test.ts`
- Verify: `test/recall-inject.test.ts`

**Interfaces:**
- Consumes: `buildValidityView`, `computeFreshness`, and `freshnessRankFactor` from Task 1.
- Produces: private `_validity(mem)` and `_freshnessFactor(mem)` helpers in `MemoryGraph`.
- Produces: additive `validity` objects in all memory-bearing outputs.

- [ ] **Step 1: Write failing same-evidence ranking tests for all paths**

Create `test/decay-ranking.test.ts`. These fixtures use real writes and graph links; the only direct
mutation is the controlled confirmation timestamp, which isolates freshness as the sole variable:

```ts
import assert from "node:assert/strict";
import test, { type TestContext } from "node:test";
import { freshDecayGraph } from "./decay-test-utils.js";

const DAY = 24 * 3600;

async function rankedFixture(t: TestContext) {
  let now = 1_800_000_000;
  const vectors: Record<string, number[]> = {
    "same relevance": [1, 0, 0, 0],
    "stale fact": [0.8, 0.6, 0, 0],
    "fresh fact": [0.8, -0.6, 0, 0],
    shared: [0, 0, 1, 0],
  };
  const { graph } = await freshDecayGraph(t, () => now, (text) => vectors[text] ?? [0, 0, 0, 1]);
  const [staleId] = await graph.add("stale fact", ["shared"]);
  const [freshId] = await graph.add("fresh fact", ["shared"]);
  graph.memories[staleId].last_confirmed_at = now - 4 * 90 * DAY;
  return { graph, ids: { staleId, freshId } };
}

async function keyEntryFixture(t: TestContext) {
  let now = 1_800_000_000;
  const vectors: Record<string, number[]> = {
    "query cue": [1, 0, 0, 0],
    "stale entry": [0.8, 0.6, 0, 0],
    "fresh entry": [0.8, -0.6, 0, 0],
    "stale-key": [0, 0, 1, 0],
    "fresh-key": [0, 0, 0, 1],
  };
  const { graph } = await freshDecayGraph(t, () => now, (text) => vectors[text] ?? [0.5, 0.5, 0.5, 0.5]);
  const [staleId] = await graph.add("stale entry", ["stale-key"]);
  await graph.add("fresh entry", ["fresh-key"]);
  graph.memories[staleId].last_confirmed_at = now - 4 * 90 * DAY;
  const staleKeyId = Object.values(graph.keys).find((key) => key.concept === "stale-key")!.id;
  const freshKeyId = Object.values(graph.keys).find((key) => key.concept === "fresh-key")!.id;
  return { graph, ids: { staleKeyId, freshKeyId } };
}

async function hopFixture(t: TestContext) {
  let now = 1_800_000_000;
  const vectors: Record<string, number[]> = {
    anchor: [1, 0, 0, 0, 0],
    "stale hop": [0, 1, 0, 0, 0],
    "fresh hop": [0, 0, 1, 0, 0],
    "bridge-stale": [0, 0, 0, 1, 0],
    "bridge-fresh": [0, 0, 0, 0, 1],
  };
  const { graph } = await freshDecayGraph(t, () => now, (text) => vectors[text] ?? [0.4, 0.4, 0.4, 0.4, 0.4]);
  const [source] = await graph.add("anchor", ["bridge-stale", "bridge-fresh"]);
  const [stale] = await graph.add("stale hop", ["bridge-stale"]);
  const [fresh] = await graph.add("fresh hop", ["bridge-fresh"]);
  graph.memories[stale].last_confirmed_at = now - 4 * 90 * DAY;
  return { graph, ids: { source, stale, fresh } };
}

test("readKey ranks fresh before stale when other evidence is equal", async (t) => {
  const { graph, ids } = await rankedFixture(t);
  const key = Object.values(graph.keys).find((candidate) => candidate.concept === "shared")!;
  const page = await graph.readKey(key.id, { query: "same relevance" }) as any;
  assert.deepEqual(page.memories.map((memory: any) => memory.memory_id), [ids.freshId, ids.staleId]);
  assert.deepEqual(page.memories.map((memory: any) => memory.validity.status), ["fresh", "stale"]);
});

test("searchKeys discounts stale member content but keeps literal key reachability", async (t) => {
  const { graph, ids } = await keyEntryFixture(t);
  const result = await graph.searchKeys("query cue", 10) as any[];
  assert.equal(result[0].key_id, ids.freshKeyId);
  const literal = await graph.searchKeys("stale-key", 10) as any[];
  assert.ok(literal.some((key) => key.key_id === ids.staleKeyId));
});

test("direct recall ranks fresh before stale when semantic evidence is equal", async (t) => {
  const { graph, ids } = await rankedFixture(t);
  const result = await graph.recall("same relevance", 10, null, false, 1, 0, 0, 0, 0, 0, false) as any[];
  assert.ok(result.findIndex((memory) => memory.id === ids.freshId) < result.findIndex((memory) => memory.id === ids.staleId));
});

test("second-hop expansion applies target freshness", async (t) => {
  const { graph, ids } = await hopFixture(t);
  const result = await graph.recall("anchor", 10, null, false, 2, 0, 0, 0, 0, 0, false) as any[];
  assert.ok(result.findIndex((memory) => memory.id === ids.fresh) < result.findIndex((memory) => memory.id === ids.stale));
});

test("related ranks fresh before stale when graph evidence is equal", async (t) => {
  const { graph, ids } = await hopFixture(t);
  const result = graph.getRelated(ids.source) as any[];
  assert.deepEqual(result.slice(0, 2).map((memory) => memory.id), [ids.fresh, ids.stale]);
  assert.ok(result.every((memory) => "validity" in memory));
});

test("list, read, recall, and injection expose validity", async (t) => {
  const { graph, ids } = await rankedFixture(t);
  assert.ok((graph.listAll() as any[]).every((memory) => "validity" in memory));
  assert.ok("validity" in ((await graph.readMemory(ids.freshId)) as any).memory);
  assert.ok(((await graph.recall("same relevance", 5, null, false, 1, 0, 0, 0, 0, 0, false)) as any[]).every((memory) => "validity" in memory));
  assert.ok(((await graph.recallInject("same relevance", 1)) as any).memories.every((memory: any) => "validity" in memory));
});
```

- [ ] **Step 2: Run the ranking suite and verify RED**

Run: `pnpm exec tsx --test test/decay-ranking.test.ts test/recall-inject.test.ts`

Expected: FAIL because current ranking uses creation time inconsistently and outputs lack validity.

- [ ] **Step 3: Centralize graph freshness helpers and update each path**

Delete the obsolete `MemoryGraph.TIME_HALF_LIFE` constant and replace `_timeFactor` with:

```ts
private _validity(mem: Memory): ValidityView {
  return buildValidityView(mem, this._now());
}

private _freshnessFactor(mem: Memory): number {
  return freshnessRankFactor(computeFreshness(mem.last_confirmed_at, mem.decay_profile, this._now()));
}
```

Apply `_freshnessFactor(mem)` exactly as specified:

```ts
// searchKeys member content contribution
const s = (memSim.get(mid) ?? 0) * this._freshnessFactor(this.memories[mid]);

// readKey
const score = rel * linkWeight * (0.9 + mem.depth * 0.1) * this._freshnessFactor(mem);

// direct initial candidates
memScores[mid] *= (0.9 + mem.depth * 0.1) * this._freshnessFactor(mem);

// shared-key hop contribution
memScores[otherMid] = (memScores[otherMid] ?? 0)
  + baseScore * MemoryGraph.HOP_DECAY * idf * lw * this._freshnessFactor(this.memories[otherMid]);

// explicit-link hop contribution
memScores[linkedId] = (memScores[linkedId] ?? 0)
  + baseScore * MemoryGraph.HOP_DECAY * this._freshnessFactor(this.memories[linkedId]);

// related, after evidence aggregation and before sorting
for (const entry of Object.values(related)) {
  entry._score *= this._freshnessFactor(this.memories[entry.id]);
}
```

Add `validity: this._validity(mem)` to the memory entry created by each of these exact output sites:
`readKey().memories[]`, `readMemory().memory`, each final `recall()` result, each `getRelated()` entry,
and each `listAll()` entry. `recallInject()` inherits the field from its selected `recall()` result.
Extend `getRelated()`'s local entry type with `validity: ValidityView` and initialize it in all three
entry constructors (shared key, forward explicit link, reverse explicit link). Change `readKey()`'s
human-readable `scoring.score` labels from `time_factor` to `freshness_factor`. Preserve `read_key`'s
no-content invariant and preserve `recallInject(... reinforce=false)`.

- [ ] **Step 4: Run focused retrieval regression tests**

Run: `pnpm exec tsx --test test/decay-ranking.test.ts test/agentic-navigation.test.ts test/recall-inject.test.ts test/related-hub.test.ts test/searchkeys-ranking.test.ts test/nhop.ts`

Expected: PASS, 0 failures.

- [ ] **Step 5: Commit Task 5**

```bash
git add src/memoryGraph.ts test/decay-ranking.test.ts
git commit -m "feat(decay): rank and expose freshness consistently"
```

---

### Task 6: Expose `confirm_memory` and the stale-memory protocol over MCP

**Files:**
- Modify: `src/server.ts:108-220, 251-550, 580-750`
- Create: `test/decay-mcp.test.ts`

**Interfaces:**
- Consumes: `parseDecayProfile`, `ConfirmationEvidence`, `MemoryGraph.confirmMemory`.
- Produces: additive `confirm_memory` MCP tool.
- Extends: `remember`, `remember_batch`, and `correct` inputs with `decay_profile`; extends `correct` with `ttl_seconds`.
- Produces: updated initialization instructions and `memory_system_prompt` behavior contract.

- [ ] **Step 1: Write a failing in-memory MCP contract test**

Create `test/decay-mcp.test.ts` using the installed SDK transport:

```ts
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";

function textResult(result: any): string {
  const content = result.content?.find((item: any) => item.type === "text");
  assert.ok(content && typeof content.text === "string");
  return content.text;
}

test("MCP exposes confirmation and stale-memory guidance", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "keymem-decay-mcp-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  process.env.SUPER_MEMORY_DATA_DIR = dir;
  process.env.EMBEDDING_BACKEND = "local";
  process.env.LOCAL_EMBEDDING_MODEL = "bge-m3";
  const embedding = await import("../src/embedding.ts");
  embedding.__setTestEmbedder(() => [1, 0]);
  t.after(() => embedding.__clearTestEmbedder());

  const { createMcpServer } = await import("../src/server.ts?decay-mcp");
  const server = createMcpServer();
  const client = new Client({ name: "decay-test", version: "0" });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  t.after(async () => { await client.close(); await server.close(); });
  await server.connect(serverTransport);
  await client.connect(clientTransport);

  const listed = await client.listTools();
  const confirm = listed.tools.find((tool) => tool.name === "confirm_memory");
  assert.ok(confirm);
  assert.deepEqual(confirm.inputSchema.required, ["memory_id", "evidence"]);
  assert.match(confirm.description ?? "", /never.*merely.*read_memory/i);

  const remember = listed.tools.find((tool) => tool.name === "remember")!;
  assert.ok("decay_profile" in (remember.inputSchema.properties ?? {}));
  const correct = listed.tools.find((tool) => tool.name === "correct")!;
  assert.ok("ttl_seconds" in (correct.inputSchema.properties ?? {}));

  const prompt = await client.getPrompt({ name: "memory_system_prompt" });
  const text = prompt.messages.map((m) => m.content.type === "text" ? m.content.text : "").join("\n");
  assert.match(text, /aging/);
  assert.match(text, /stale/);
  assert.match(text, /confirm_memory/);
  assert.match(text, /do not.*confirm.*merely.*read/i);

  const remembered = JSON.parse(textResult(await client.callTool({
    name: "remember",
    arguments: { content: "temporary fact", keys: ["temporary"], decay_profile: "transient" },
  })));
  const confirmed = JSON.parse(textResult(await client.callTool({
    name: "confirm_memory",
    arguments: { memory_id: remembered.saved, evidence: "user", source: { reason: "current assertion" } },
  })));
  assert.equal(confirmed.confirmed, true);
  assert.equal(confirmed.validity.status, "fresh");
  assert.equal(confirmed.validity.decay_profile, "transient");

  const invalid = await client.callTool({
    name: "remember",
    arguments: { content: "bad profile", keys: ["profile"], decay_profile: "volatile" },
  });
  assert.equal(invalid.isError, true);
  assert.match(textResult(invalid), /Unknown decay profile/);
});
```

- [ ] **Step 2: Run the MCP contract test and verify RED**

Run: `pnpm exec tsx --test test/decay-mcp.test.ts`

Expected: FAIL because `confirm_memory`, decay arguments, and stale-memory instructions do not exist.

- [ ] **Step 3: Add schemas, handlers, provenance identity, and instructions**

Add `decay_profile` enums to `remember`, each `remember_batch.items` entry, and `correct`; add
`ttl_seconds` to `correct`. Parse and forward them at the three handler sites exactly as follows:

```ts
// remember options
decayProfile: parseDecayProfile(a.decay_profile),

// each remember_batch item
decayProfile: parseDecayProfile(item.decay_profile),

// correct/supersede options: omitted values preserve the predecessor policy
decayProfile: a.decay_profile === undefined ? undefined : parseDecayProfile(a.decay_profile),
ttlSeconds: typeof a.ttl_seconds === "number" ? a.ttl_seconds : undefined,
```

Add the tool definition:

```ts
{
  name: "confirm_memory",
  description:
    "Confirm that a memory is still current using explicit present evidence. Never call this merely because read_memory returned the content. Use only after a current user assertion, an authoritative current source, or direct observation. Refreshes validity but does not change content or key links.",
  inputSchema: {
    type: "object",
    properties: {
      memory_id: { type: "string" },
      evidence: { type: "string", enum: ["user", "authoritative_source", "observation"] },
      namespace: { type: "string" },
      source: { type: "object", additionalProperties: true },
    },
    required: ["memory_id", "evidence"],
  },
}
```

Add handler logic:

```ts
case "confirm_memory": {
  const evidence = a.evidence;
  if (evidence !== "user" && evidence !== "authoritative_source" && evidence !== "observation") {
    throw new Error(`Unknown confirmation evidence: ${String(evidence)}`);
  }
  const hostLink = await resolveHostLink(headers);
  const confirmationId = hostLink
    ? `${hostLink.agent}:${hostLink.session_id}:${hostLink.turn}`
    : null;
  const result = await graph.confirmMemory(a.memory_id as string, {
    evidence,
    namespace: typeof a.namespace === "string" ? a.namespace : null,
    source: buildSource(parseObject(a.source), "confirm_memory", hostLink),
    confirmationId,
  });
  return { content: [{ type: "text", text: JSON.stringify(result) }] };
}
```

Update both prompt strings so they no longer say a full read confirms or deepens content. State the
fresh/aging/stale rules verbatim and prohibit read-triggered confirmation.

- [ ] **Step 4: Run MCP and relevant server integration tests**

Run: `pnpm exec tsx --test test/decay-mcp.test.ts test/e2e-two-shims.test.ts test/shim-reconnect.test.ts test/daemon-idle.test.ts`

Expected: PASS, 0 failures.

Run: `pnpm run build`

Expected: PASS with TypeScript exit code 0.

- [ ] **Step 5: Commit Task 6**

```bash
git add src/server.ts test/decay-mcp.test.ts
git commit -m "feat(decay): expose explicit memory confirmation"
```

---

### Task 7: Update the public protocol and run complete verification

**Files:**
- Modify: `skills/keymem/SKILL.md`
- Modify: `README.md:234-244, MCP Tools section`
- Modify: `CHANGELOG.md:7`
- Create: `test/decay-docs.test.ts`
- Verify: all changed source and tests from Tasks 1-6

**Interfaces:**
- Consumes: final MCP behavior from Tasks 1-6.
- Produces: one consistent agent-facing protocol in the plugin skill, server prompt, README, and changelog.

- [ ] **Step 1: Write a failing repository protocol test**

Create `test/decay-docs.test.ts` that reads `skills/keymem/SKILL.md` and `README.md` and asserts:

```ts
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("public protocol distinguishes read from confirmation", async () => {
  const skill = await readFile(new URL("../skills/keymem/SKILL.md", import.meta.url), "utf-8");
  const readme = await readFile(new URL("../README.md", import.meta.url), "utf-8");
  for (const text of [skill, readme]) {
    assert.match(text, /confirm_memory/);
    assert.match(text, /aging/);
    assert.match(text, /stale/);
    assert.match(text, /read_memory.*(?:does not|doesn't).*confirm/is);
  }
});
```

- [ ] **Step 2: Run the documentation contract test and verify RED**

Run: `pnpm exec tsx --test test/decay-docs.test.ts`

Expected: FAIL because the existing docs still say `read_memory` confirms/deepens facts.

- [ ] **Step 3: Update protocol documentation and changelog**

In `skills/keymem/SKILL.md`, add a `Freshness and confirmation` section with these exact rules:

```md
## Freshness and confirmation

- `read_memory` reads a fact and may reinforce the key path; it does **not** confirm that the content is current.
- `fresh` may be used normally. Qualify `aging` facts when currentness matters.
- Never assert a `stale` fact as current. Verify it externally or ask the user.
- Call `confirm_memory` only after an explicit current user assertion, an authoritative current source, or direct observation.
- Never call `confirm_memory` merely because `read_memory` returned the content.
- Changed fact → `correct`. Junk fact → `forget`. Wrong key → `dismiss`.
```

Update README's depth section, tool table, configuration table, and examples to document the four
profiles, real half-lives, validity payload, and `confirm_memory`. Remove the false statement that a
read confirms facts or that superseded deep memories remain current.

Add an `[Unreleased]` section at the top of `CHANGELOG.md` describing confirmation-aware freshness,
the additive tool/schema fields, expired-dedup correction, and migration behavior.

- [ ] **Step 4: Run the documentation test and complete verification**

Run: `pnpm exec tsx --test test/decay-docs.test.ts`

Expected: PASS.

Run: `pnpm test`

Expected: all repository tests pass with 0 failures.

Run: `pnpm run build`

Expected: TypeScript build exits 0 and the hook/plugin version sync completes without errors.

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 5: Verify every design hypothesis against named tests**

Record the mapping in the implementation handoff:

```text
H1 -> decay-expiration + dedup-supersede tests
H2 -> confirm-memory + decay-ranking tests
H3 -> decay pure formula/status tests
H4 -> decay-ranking path tests
H5 -> agentic-navigation + confirm-memory idempotency tests
H6 -> decay-migration tests
H7 -> decay-mcp + decay-docs contract tests
```

- [ ] **Step 6: Commit Task 7**

```bash
git add skills/keymem/SKILL.md README.md CHANGELOG.md test/decay-docs.test.ts
git commit -m "docs(decay): publish freshness protocol"
```

---

## Plan Self-Review

- **Spec coverage:** H1–H7 each map to a named RED→GREEN test in Task 7's handoff matrix.
- **Executable fixtures:** every new test helper is defined or imported; no mocked retrieval method,
  wall-clock sleep, placeholder, or unstated external service is required.
- **Type consistency:** decay types originate in `src/decay.ts`; `src/types.ts`, `MemoryGraph`, and MCP
  handlers consume that single vocabulary without a parallel string union.
- **Compatibility:** old tool calls remain valid, v1 graph data migrates through the atomic save path,
  literal navigation remains age-independent, and TTL remains the only automatic exclusion rule.
- **Commit isolation:** each task stages only files it creates or modifies; regression-only files are
  executed but not staged.
