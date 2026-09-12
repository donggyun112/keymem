# Task-Conditioned Associative Recall: Hypothesis and Experiment

**Date:** 2026-09-13
**Status:** Adopted as the default evidence-consumption policy
**Reframes:** H1 in `2026-09-04-recall-hit-rate-v2-design.md`

## Claim

KeyMem should distinguish two different rankings:

\[
A(m \mid q) = \operatorname{Semantic}(m,q) + \operatorname{Associative}(m)
\]

measures how naturally a memory activates, while

\[
U(m \mid q,t) = \operatorname{TaskRelevance}(m,q,t)
  + \operatorname{EvidenceCoverage}(m,t)
  + \operatorname{Contrastiveness}(m,t)
\]

measures how useful that activated memory is as evidence for task \(t\).

Broad activation, including activation through highly connected keys, is a feature of associative
memory. The comparison regression appears when a single global top-k policy is asked to serve both
association and inference. The problem is not over-association but under-selection.

## Hypotheses

### H1: Comparison failures are mostly selection failures

If GRAPH misses one of two comparison supports at top-5, both supports should usually still be
present in a wider, unchanged activation pool. A selector that covers both named entities should
recover them without changing graph traversal or association scores.

**Pass:** GRAPH pool@10 contains both supports in most global top-5 failures, and task projection
improves paired both@5 with no material regressions.

**Disproof:** the missing support is absent from the activation pool, or projection cannot improve
selection without dropping already-correct pairs.

### H2: Bridge benefit belongs to activation and must be preserved

Bridge questions should retain the GRAPH advantage because the answer entity is not named directly.
Task projection must leave bridge selection unchanged unless a future controller can demonstrate
that it has complete evidence.

**Pass:** bridge `TASK_PROJECTED` is byte-for-byte the same ranked top-5 as `GRAPH_GLOBAL` in this
controlled experiment.

### H3: Entity resolution, not hub suppression, is the next bottleneck

When an entity is named with an alias such as `Hayden` but stored as `Hayden (musician)`, a literal
projection cannot reserve its evidence slot even when the memory activated correctly. The next
experiment should improve query-to-key entity resolution on a separate held-out set; it should not
tune aliases on the 24 cases already inspected here.

## Experimental Design

Dataset: the existing first 120 HotpotQA distractor cases (96 bridge, 24 comparison), ten paragraph
memories per question, bge-m3, heuristic title/mentioned-title keys.

Conditions:

1. `DIRECT`: existing one-hop semantic recall, top-5.
2. `GRAPH_GLOBAL`: existing two-hop recall and global top-5.
3. `GRAPH_POOL`: the unchanged GRAPH activation pool, top-10 reachability only.
4. `TASK_PROJECTED`: on gold-labeled comparison rows only, reserve one slot for each paragraph title
   literally named in the question, then fill the five-slot budget in original activation order.
   Bridge rows return the unchanged `GRAPH_GLOBAL` top-5.

Controls:

- no gold support labels are available to the selector;
- the HotpotQA question-type label is used to isolate selection from intent classification;
- graph traversal, hop decay, IDF/link weights, and hub behavior are unchanged;
- `reinforce=false` prevents the DIRECT condition from mutating the graph before GRAPH runs;
- `KEYMEM_RERANK=false` isolates the key-graph and deterministic selector;
- every question uses a fresh graph containing only its ten distractor paragraphs.

Run:

```bash
pnpm run bench:evidence-projection
```

## Results

| question type | metric | DIRECT | GRAPH_GLOBAL | TASK_PROJECTED | GRAPH_POOL |
|---|---|---:|---:|---:|---:|
| bridge (96) | support-recall@5 | 77% | 86% | 86% | 99%@10 |
| | both@5 | 57% | 76% | 76% | 99%@10 |
| comparison (24) | support-recall@5 | 82% | 76% | **98%** | 100%@10 |
| | both@5 | 63% | 50% | **96%** | 100%@10 |

Comparison diagnostics:

- all 12 `GRAPH_GLOBAL` both@5 failures contained both supports in `GRAPH_POOL`;
- projection recovered 11/12 failures;
- versus `GRAPH_GLOBAL`: 11 paired improvements, 0 regressions (two-sided exact McNemar
  \(p=0.00098\));
- versus `DIRECT`: 8 paired improvements, 0 regressions (\(p=0.0078\));
- the only projected failure was the unresolved `Hayden` → `Hayden (musician)` alias.

The current-revision pure-read numbers differ slightly from the older checked-in HotpotQA table,
but reproduce its qualitative result: DIRECT beats global GRAPH selection on comparison, while
GRAPH strongly improves bridge retrieval.

## Interpretation

H1 and H2 are supported as a mechanism-level result. The graph activated all comparison evidence;
the fixed global top-k discarded half of the complete pairs. Reprojecting that same activation set
raised both@5 from 50% to 96% without changing any association.

This does **not** yet validate a production controller. It is an exploratory upper-bound because it
uses the dataset's task label and title metadata, the comparison sample is only 24 questions, and
the hypothesis was motivated by this benchmark family. It measures evidence retrieval, not final
answer accuracy.

## Adoption Gate

Do not implement generic hub penalties, graph pruning, or `comparison -> no expansion` from this
result. A production candidate should instead expose a broad activation pool plus a separate,
task-conditioned evidence selector. It should ship only after a frozen controller clears:

1. an untouched comparison holdout without gold task labels or corpus-title lookup;
2. the blind-key HotpotQA bridge control;
3. conversational comparison cases with aliases and uneven evidence coverage;
4. end-answer accuracy in addition to support both@k;
5. latency and false-injection guardrails from the recall-quality design.

The appropriate implementation name is `Task-Conditioned Associative Recall` or `Adaptive Evidence
Selection`, not `Adaptive Graph Expansion`.

## Production Adoption Experiment — Accepted after re-review

The initial production trial was rejected because its gate required byte-identical bridge ordering.
That was the wrong failure criterion: one changed row was labelled `bridge` but asked the explicit
comparison "Who is older, Annie Morton or Terry Richardson?", and the changed injection recovered
the missing second entity. Order change is therefore diagnostic information, not a quality metric.

The final controller activates only with a comparison cue and two literal `name`/`proper_noun` keys.
It removes those entity strings from the query to form a task residual (`nationality`, `founded`,
`members`, ...), scores only the activated pool against that residual, and projects one useful memory
per entity. A leading entity pair is considered sufficient when both utilities are within 0.02 of
their entity's best activated evidence. Graph membership, association scores, and the associative
winner used for reinforcement remain unchanged. Inject's generic structured/lexical filters retain
the selected per-entity evidence instead of requiring both entities to occur in one memory.

The final A/B uses all 300 Hotpot rows, with the first 120 as development and the remaining 180 as a
separate holdout. The controller receives no dataset task label or gold support. Reranking and
reinforcement are disabled in both arms; title and mentioned-title keys are typed `proper_noun` in
both arms. Results and per-query outputs are in `bench/task-selection-eval-results.json`.

The holdout was inspected while fixing losses in an earlier candidate, so the final table is a
regression validation, not an unbiased estimate of generalization. A different corpus and end-answer
evaluation remain follow-up work; the adoption claim here is limited to retrieval quality and the
measured KeyMem paths.

| split | surface | comparison both baseline → selected | paired wins/losses | bridge quality |
|---|---|---:|---:|---|
| dev (24 comparison) | graph recall @5 | 54.2% → **66.7%** | 3 / 0 | identical |
| dev | inject @5 | 16.7% → **29.2%** | 3 / 0 | +1 win, 0 loss |
| dev | inject @2 | 12.5% → **20.8%** | 2 / 0 | +1 win, 0 loss |
| holdout (28 comparison) | graph recall @5 | 82.1% → **89.3%** | 2 / 0 | identical |
| holdout | inject @5 | 28.6% → **50.0%** | 6 / 0 | identical |
| holdout | inject @2 | 21.4% → **39.3%** | 5 / 0 | identical |

Across all 52 comparison questions, both@5 is 69.2% → 78.8% for graph recall and 23.1% →
40.4% for inject; daemon-shaped inject@2 is 17.3% → 30.8%. There are no paired losses on any
surface. Of 248 bridge rows, three change order: one improves inject both@k and two leave every gold
metric unchanged. All 14 direct/assoc2/not-found fixture queries remain byte-identical.

Warm local-model timing adds about 2.3 ms mean to eligible comparison recall (17.5 → 19.7 ms) and
5.4 ms to inject (27.0 → 32.4 ms). Ineligible bridge queries follow the original path. The feature is
default-on and can be disabled with `KEYMEM_TASK_EVIDENCE_SELECTION=false`.
