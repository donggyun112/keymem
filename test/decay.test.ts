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
  assert.ok(Math.abs(freshnessRankFactor(0.5) - 0.6) < Number.EPSILON);
  assert.equal(freshnessRankFactor(0), 0.2);
});

test("freshness rank factor preserves the contractual formula", () => {
  const freshness = 0.1234;
  assert.ok(Math.abs(freshnessRankFactor(freshness) - (0.2 + 0.8 * freshness)) < Number.EPSILON);
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
    DECAY_STABLE_DAYS: "1e308",
  };
  const config = loadDecayConfig((key) => values[key], (message) => warnings.push(message));
  assert.equal(config.halfLivesSeconds.transient, DEFAULT_DECAY_CONFIG.halfLivesSeconds.transient);
  assert.equal(config.halfLivesSeconds.standard, DEFAULT_DECAY_CONFIG.halfLivesSeconds.standard);
  assert.equal(config.halfLivesSeconds.stable, DEFAULT_DECAY_CONFIG.halfLivesSeconds.stable);
  assert.equal(warnings.length, 3);
});
