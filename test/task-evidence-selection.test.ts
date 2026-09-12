import assert from "node:assert/strict";
import test from "node:test";

import { analyzeTaskEvidence, projectTaskEvidence } from "../src/evidenceSelection.js";
import type { Key } from "../src/types.js";

const key = (
  id: string,
  concept: string,
  key_type: Key["key_type"] = "proper_noun",
  aliases: string[] = []
): Key => ({ id, concept, aliases, key_type, embedding: [1, 0] });

const apple = key("apple", "Apple");
const microsoft = key("microsoft", "Microsoft");
const company = key("company", "company", "concept");

test("comparison projection selects useful evidence for both named entities", () => {
  const candidates = [
    { id: "iphone", keys: [apple], utility: 0.4, value: "iphone" },
    { id: "jobs", keys: [apple], utility: 0.5, value: "jobs" },
    { id: "apple-founded", keys: [apple], utility: 0.9, value: "apple-founded" },
    { id: "microsoft-founded", keys: [microsoft], utility: 0.85, value: "microsoft-founded" },
    { id: "generic", keys: [company], utility: 0.95, value: "generic" },
  ];

  const result = projectTaskEvidence(
    "Which was founded first, Apple or Microsoft?",
    candidates,
    2
  );

  assert.equal(result.applied, true);
  assert.deepEqual(result.entities, ["Apple", "Microsoft"]);
  assert.deepEqual(result.selected.slice(0, 2).map((item) => item.id), [
    "apple-founded",
    "microsoft-founded",
  ]);
  assert.deepEqual(new Set(result.selected.map((item) => item.id)), new Set(candidates.map((item) => item.id)));
});

test("task analysis removes entity identity and comparison scaffolding", () => {
  const analysis = analyzeTaskEvidence(
    "Were Scott Derrickson and Ed Wood of the same nationality?",
    [key("scott", "Scott Derrickson"), key("ed", "Ed Wood")]
  );
  assert.deepEqual(analysis?.labels, ["Scott Derrickson", "Ed Wood"]);
  assert.equal(analysis?.taskQuery, "nationality");
});

test("Korean postpositions still identify both comparison entities", () => {
  const result = projectTaskEvidence(
    "Apple과 Microsoft 중 어느 회사가 먼저 설립됐나?",
    [
      { id: "a-noise", keys: [apple], utility: 0.2, value: null },
      { id: "a-answer", keys: [apple], utility: 0.8, value: null },
      { id: "b-noise", keys: [microsoft], utility: 0.2, value: null },
      { id: "b-answer", keys: [microsoft], utility: 0.7, value: null },
    ],
    2
  );
  assert.equal(result.applied, true);
  assert.deepEqual(result.selected.slice(0, 2).map((item) => item.id), ["a-answer", "b-answer"]);
});

test("a Korean word containing 중 does not become a comparison cue", () => {
  const candidates = [
    { id: "a", keys: [apple], utility: 0.8, value: null },
    { id: "b", keys: [microsoft], utility: 0.7, value: null },
    { id: "c", keys: [apple], utility: 0.6, value: null },
  ];
  assert.equal(
    projectTaskEvidence("중국에서 Apple과 Microsoft가 행사를 열었다", candidates, 2).applied,
    false
  );
});

test("quoted more and same-actor bridge wording do not trigger comparison projection", () => {
  const max = key("max", "Max Mutchnick");
  const david = key("david", "David Kohan");
  const polar = key("polar", "The Polar Bears");
  const armie = key("armie", "Armie Hammer");
  const candidates = [
    { id: "a", keys: [max], utility: 0.8, value: null },
    { id: "b", keys: [david], utility: 0.7, value: null },
    { id: "c", keys: [max], utility: 0.6, value: null },
  ];
  assert.equal(
    projectTaskEvidence(
      'What station broadcast "Marry Me a Little More", created by Max Mutchnick and David Kohan?',
      candidates,
      2
    ).applied,
    false
  );
  assert.equal(
    projectTaskEvidence(
      "The Polar Bears features Armie Hammer, the same voice actor for which Cars 3 character?",
      [
        { id: "p", keys: [polar], utility: 0.8, value: null },
        { id: "a", keys: [armie], utility: 0.7, value: null },
        { id: "x", keys: [polar], utility: 0.6, value: null },
      ],
      2
    ).applied,
    false
  );
});

test("already balanced leading entity evidence is preserved", () => {
  const candidates = [
    { id: "a", keys: [apple], utility: 0.89, value: null },
    { id: "b", keys: [microsoft], utility: 0.79, value: null },
    { id: "a-task", keys: [apple], utility: 0.9, value: null },
    { id: "b-task", keys: [microsoft], utility: 0.8, value: null },
  ];
  const result = projectTaskEvidence("Which is older, Apple or Microsoft?", candidates, 2);
  assert.equal(result.applied, false);
  assert.deepEqual(result.selected, candidates);
});

test("balanced entity names are insufficient when later evidence has much higher task utility", () => {
  const candidates = [
    { id: "a-noise", keys: [apple], utility: 0.2, value: null },
    { id: "b-noise", keys: [microsoft], utility: 0.3, value: null },
    { id: "a-task", keys: [apple], utility: 0.9, value: null },
    { id: "b-task", keys: [microsoft], utility: 0.8, value: null },
  ];
  const result = projectTaskEvidence("Which is older, Apple or Microsoft?", candidates, 2);
  assert.equal(result.applied, true);
  assert.deepEqual(result.selected.slice(0, 2).map((item) => item.id), ["a-task", "b-task"]);
});

test("bridge-shaped query and concept-only keys keep associative order unchanged", () => {
  const candidates = [
    { id: "a", keys: [apple], utility: 0.5, value: null },
    { id: "b", keys: [microsoft], utility: 0.4, value: null },
    { id: "c", keys: [company], utility: 0.3, value: null },
  ];
  assert.deepEqual(projectTaskEvidence("Where is Apple headquartered?", candidates, 2), {
    applied: false,
    entities: [],
    selected: candidates,
  });
  assert.deepEqual(
    projectTaskEvidence(
      "Which company was founded first?",
      candidates.map((candidate) => ({ ...candidate, keys: [company] })),
      2
    ),
    { applied: false, entities: [], selected: candidates.map((candidate) => ({ ...candidate, keys: [company] })) }
  );
});

test("ambiguous shortened parenthetical aliases do not trigger projection", () => {
  const adventure = key("adventure", "The Hard Easy (Adventure Time)");
  const film = key("film", "The Hard Easy (film)");
  const candidates = [
    { id: "a", keys: [adventure], utility: 0.8, value: null },
    { id: "b", keys: [film], utility: 0.7, value: null },
    { id: "c", keys: [adventure], utility: 0.6, value: null },
  ];
  const result = projectTaskEvidence("What came before The Hard Easy?", candidates, 2);
  assert.equal(result.applied, false);
  assert.deepEqual(result.selected, candidates);
});
