import assert from "node:assert/strict";
import test from "node:test";

import {
  mentionedTitles,
  projectComparisonEvidence,
  type RankedEvidence,
} from "../bench/evidence-projection-lib.ts";

const evidence = (...titles: string[]): RankedEvidence[] =>
  titles.map((title, index) => ({ id: String(index), title }));

test("mentionedTitles finds complete title mentions in question order", () => {
  assert.deepEqual(
    mentionedTitles(
      "Were Scott Derrickson and Ed Wood of the same nationality?",
      ["Ed Wood (film)", "Ed Wood", "Scott Derrickson", "Wood"]
    ),
    ["Scott Derrickson", "Ed Wood"]
  );
});

test("comparison projection reserves one slot per named entity", () => {
  const activated = evidence("Apple iPhone", "Apple", "Microsoft Windows", "Microsoft", "Other", "Tail");
  const result = projectComparisonEvidence(
    "Which was founded first, Apple or Microsoft?",
    ["Apple", "Microsoft", "Other"],
    activated,
    5
  );

  assert.equal(result.applied, true);
  assert.deepEqual(result.entities, ["Apple", "Microsoft"]);
  assert.deepEqual(result.selected.map((x) => x.title), [
    "Apple",
    "Microsoft",
    "Apple iPhone",
    "Microsoft Windows",
    "Other",
  ]);
});

test("projection cannot invent an entity missing from the activation pool", () => {
  const result = projectComparisonEvidence(
    "Which was founded first, Apple or Microsoft?",
    ["Apple", "Microsoft"],
    evidence("Apple", "Other"),
    2
  );

  assert.equal(result.applied, true);
  assert.deepEqual(result.selected.map((x) => x.title), ["Apple", "Other"]);
});

test("single-entity bridge query preserves global activation order", () => {
  const activated = evidence("Bridge seed", "Associated answer", "Other");
  const result = projectComparisonEvidence(
    "What did Bridge seed lead to?",
    ["Bridge seed", "Associated answer"],
    activated,
    2
  );

  assert.equal(result.applied, false);
  assert.deepEqual(result.selected, activated.slice(0, 2));
});
