# keymem — Benchmarks

What this measures and, honestly, what it doesn't. The goal is to **prove how much keymem's
key-graph actually buys you** — isolated causally, not asserted by metaphor — and to mark the
limits plainly.

> TL;DR: On **HotpotQA bridge questions** (real external multi-hop data, gold labels, no LLM judge,
> **and keys generated blind by independent subagents**), graph traversal retrieves **both** gold
> supporting paragraphs **63%** of the time vs **53%** for flat semantic and **35%** for lexical
> (+10pp / +28pp). The read path was also made **O(1) instead of O(graph)** (read_memory p50 ~45ms
> → ~0.01ms @ 500 memories). Honest costs: the gain is specific to the *multi-hop* case (it
> slightly *hurts* "comparison" questions), my own hand-derived keys *inflated* it (78% vs 60% —
> trust the blind-key 63/53), it's retrieval-recall not end-task accuracy, and none of this is a
> head-to-head SOTA claim vs mem0/Zep. Details below.

---

## Why an ablation, not a leaderboard score

The standard agent-memory benchmarks — [LoCoMo](https://github.com/snap-research/locomo)
(1,982 questions over long conversations) and [LongMemEval](https://github.com/xiaowu0162/LongMemEval)
(~115k-token histories) — score a full pipeline with an **LLM-as-judge**, and the published
vendor numbers are openly disputed (Zep vs Mem0 contest each other's methodology; see
[Zep's critique](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)).
Running those credibly needs the competing systems installed, the full datasets, and thousands
of judge calls — none of which a single-author project can do cleanly or cheaply, and a number
produced that way would be exactly the kind of disputed score the field is tired of.

So this benchmark answers a narrower, **causally clean** question instead:

> Holding the engine, the data, and the embeddings fixed, **how much does the key-graph
> traversal itself add** over flat 1-hop semantic retrieval?

That isolates *our* contribution rather than comparing incomparable stacks. It is a smaller
claim, but an honest one.

---

## 1. Associative-recall ablation

**Design.** Same data, same real embeddings (`bge-m3`), three retrievers — two flat baselines
(one external) and keymem's graph:

| Condition | how | meaning |
|---|---|---|
| **BM25** | standalone MiniSearch over content | classic flat *lexical* store (no embeddings, no graph) |
| **DIRECT** | `recall(expand=false, hops=1)` | flat *semantic* reach — keymem 1-hop, no expansion |
| **GRAPH** | `recall(expand=true, hops=2)` | keymem's multi-hop key-graph traversal |

BM25 and DIRECT are the flat baselines; GRAPH adds graph traversal. The delta isolates what the
key-graph buys over flat lexical / flat semantic retrieval.

**Dataset** (`bench/assoc-fixture.json`): a 14-memory bilingual persona graph. Each `assoc2`
query's answer is a **far memory reachable only via a shared key two hops away** (e.g. *"미나가
키우는 강아지"* → the dog's *allergy* fact, reachable only through the shared key `보리`). The
query has low direct similarity to that target — so flat retrieval should miss it and graph
traversal should reach it. `direct` queries are 1-hop controls; `notfound` must return nothing.

**Metrics.** `reach@10` (target anywhere in the top 10 — does the system find it *at all*),
`hit@5` (target in top 5 — ranking-sensitive), `MRR`. Run: `tsx bench/ablation.ts`.

### Results (`bge-m3`, n per category)

| category | metric | BM25 | DIRECT | GRAPH |
|---|---|---:|---:|---:|
| **assoc2** (6) | **reach@10** | 33% | 50% | **83%** |
| | hit@5 | 17% | 33% | 33% |
| | MRR | 0.19 | 0.13 | 0.23 |
| direct (5) | reach@10 | 80% | 100% | 100% |
| | hit@5 | 80% | 80% | 80% |
| | MRR | 0.48 | 0.82 | **0.69** |
| notfound (3) | not-found acc | 1/3 | 1/3 | 1/3 |

### What this proves (and doesn't)

- ✅ **The key-graph reaches connected-but-dissimilar memories that *both* flat baselines cannot.**
  On `assoc2`, BM25 reaches 2/6 targets, DIRECT (flat semantic) 3/6, GRAPH 5/6 (the 6th lands at
  rank 11, just outside the window). +33pp over flat-dense, +50pp over flat-lexical — *measured*,
  not asserted. Per-query, the graph pulls in the dog-allergy / climbing-injury / caffeine-sleep
  facts via shared keys that neither lexical nor 1-hop semantic similarity ever surfaces. (On the
  `direct` control, DIRECT/GRAPH both reach 100% vs BM25's 80% — embeddings already beat lexical
  on plain queries; the graph's distinct contribution is the associative reach.)
- ⚠️ **The gain is in *reachability*, not top rank.** `HOP_DECAY` scores 2-hop hits low, so they
  arrive at ranks 9–11 — `hit@5` shows **no** gain. The value is real for an agent that
  navigates/pages (the intended `recall → read_key → read_memory` flow), much weaker if you only
  ever read top-5.
- ⚠️ **Honest costs.** Graph expansion slightly *hurt* direct-query ranking (MRR 0.82 → 0.69) by
  mixing associative neighbours into clean results. And **not-found precision is poor (1/3)** —
  2 of 3 distractors returned something — at this small scale the absolute-score gate is too
  loose. Both are the same under DIRECT, so they're engine/gate issues, not graph-specific, but
  they're real.

---

## 2. External validation: HotpotQA multi-hop retrieval

§1 uses a dataset I built, so here's the same question on data I didn't. [HotpotQA](https://hotpotqa.github.io/)
(distractor) ships, per question, **10 paragraphs (2 gold "supporting" + 8 distractors) plus gold
supporting-fact labels** — so we measure support *retrieval* with **no LLM judge**. Mapping to
keymem: each paragraph becomes a memory keyed by its own title + any other paragraph title it
mentions in-text, so a **bridge** entity (what links the question's paragraph to the answer's)
becomes a shared key. For **bridge** questions the answer paragraph is connected-but-dissimilar to
the query (the query never names it) — exactly keymem's case. **comparison** questions name both
entities up front (no bridge to traverse) — a built-in negative control. Each question is scored
in isolation over only its 10 paragraphs. Run: `tsx bench/hotpot.ts`.

### Results (`bge-m3`, N=120 = 96 bridge + 24 comparison, top-5 of 10)

| question type | metric | BM25 | DIRECT | GRAPH |
|---|---|---:|---:|---:|
| **bridge** (96) | support-recall@5 | 70% | 78% | **88%** |
| | **both@5** (got both golds) | 49% | 60% | **78%** |
| comparison (24) | support-recall@5 | 57% | 81% | 77% |
| | both@5 | 25% | 63% | **54%** |
| all (120) | both@5 | 44% | 61% | **73%** |

- ✅ **On bridge (multi-hop, connected-but-dissimilar — keymem's case) the graph clearly wins**:
  both gold paragraphs retrieved **78%** of the time vs **60%** (flat semantic) and **49%**
  (lexical) — +18pp / +29pp, on real external data with gold labels, n=96. This is a stronger
  result than §1 (here the bridge-reached support lands inside top-5 because the pool is only 10).
- ⚠️ **On comparison questions the graph slightly *hurts*** (both@5 54% vs DIRECT's 63%) — expected:
  both entities are already in the query, so there's no bridge to traverse and expansion just adds
  noise. An honest negative that confirms the gain is *specifically* the multi-hop case, not free.
- This is **retrieval-recall of the gold paragraphs, not end-task answer accuracy** — getting both
  supports is necessary, not sufficient, for a correct answer (no LLM judge here).

**Caveats.** The table above derives keys *myself* (title + mentioned-titles), which mirrors the
gold bridge structure — so it risks measuring a graph I keyed to match the answer. Both
conditions share those keys (the ablation is internally fair), but the absolute gain could be
inflated. The next check removes exactly that doubt. (Also: one dataset, one embedder, document
multi-hop not conversational memory; retrieval-recall not end-task accuracy.)

### Validity check: blind agent-generated keys

To kill the "you keyed it to match the answer" objection, the keys were regenerated by
**independent subagents that saw only each paragraph's text** — no question, no gold supports,
no other paragraphs' role — and tagged each for findability (realistic keymem write-time keying).
Then the same bridge questions were re-run on those blind keys (`bench/hotpot-agentkeys.ts`,
N=40 bridge, all 400 paragraphs keyed by agents, 0 fallbacks):

| metric | BM25 | DIRECT | GRAPH | vs heuristic keys |
|---|---:|---:|---:|---|
| support-recall@5 | 61% | 72% | **79%** | (heuristic: 70 / 78 / 88) |
| **both@5** | 35% | 53% | **63%** | (heuristic: 49 / 60 / 78) |

- ✅ **The gain survives blind keying**: GRAPH both@5 **63% vs DIRECT 53% (+10pp), vs BM25 35%
  (+28pp)**. With keys an independent agent produced without ever seeing the task, graph traversal
  still retrieves the connected support more often. The "I keyed it to match the gold" objection is
  answered.
- ⚠️ **…but smaller — my heuristic keys *were* optimistic** (78/60 → 63/53). Honest: the §2 table
  over-states the effect; the blind-key numbers are the ones to trust. The real, defensible claim
  is **+10pp both@5 over flat semantic, +28pp over lexical, with realistic keys.**

## 3. Read-path latency (the v0.12.1 fix)

`read_memory` rewrote the entire `graph.json` on every call (it bumps depth/access), making each
read **O(graph size)**. Reads are the frequent path (every `recall → read_key → read_memory`);
deferring that persistence to `flush()` makes reads O(1). Measured with a synthetic 1024-dim
embedder to isolate graph-op cost from embedding inference (`bench/perf.ts`):

| memories | read_memory p50 — before | after (v0.12.1) |
|---:|---:|---:|
| 500 | 44.8 ms | **0.01 ms** |
| 1,500 | 132.8 ms | flat |
| 3,000 | 262.5 ms | flat |

Before the fix, read latency grew linearly with the store (the full-file rewrite). After, a read
is a RAM mutation + dirty flag. `searchKeys` was already cheap (1–8 ms at 0.5–3k keys); the write
path (`add`) is deliberately left eager — writes are rare, so its O(n) per-save save cost is not
worth trading durability for.

---

## 4. Honest scope & the trajectory caveat

**Scope.** Small synthetic persona graph (14 memories), one embedder, one author's fixtures.
This is **not** LoCoMo/LongMemEval scale, and it is **not** a head-to-head vs mem0/Zep — those
remain future work (they need the competing systems + an LLM judge). What's proven here is the
*marginal contribution of keymem's own graph*, on a probe built specifically to stress the
connected-but-dissimilar case.

**The trajectory caveat.** keymem (like all clever memory layers) bets that *structure beats raw
model reasoning over flat content*. That bet weakens as agentic search improves: an
[Amazon Science AAAI-2026 result](https://www.amazon.science/) reports agentic keyword search at
~94.5% of RAG faithfulness with **no** vector store, and Karpathy has noted that at personal
scale a full RAG stack often adds more latency/noise than it removes
([context](https://venturebeat.com/data/context-architecture-is-replacing-rag-as-agentic-ai-pushes-enterprise-retrieval-to-its-limits)).
The 2026 consensus is **hybrid** (small index + lots of tools), not pure-vector or pure-agentic.

So keymem's durable value is **not** "smarter retrieval than the model" — the model keeps getting
smarter. It is:
1. **Reach** — surfacing connected-but-dissimilar memories an agent wouldn't think to query for
   (the +33pp above), and
2. **Amortization + legibility** — the association is computed once into an explicit, auditable
   edge, instead of re-derived by an LLM hop every query, and you can *read why* two things are
   linked (a key path) rather than trust an opaque cosine.

Whether that earns its complexity over "a strong model + grep + re-query" is, ultimately, an
empirical question per use case. This doc is the start of measuring it honestly, not the last
word.

---

## 5. Experimental: inject mode (one-shot associated-memory recall)

The default flow makes an agent walk `recall → read_key → read_memory` (controllable, but ~7
calls per associative leap). An opt-in `recallInject` returns navigation keys **plus** the top-N
expanded memories in one call. Is that worth it, and at what noise cost? Sweep on the blind-agent
-keyed bridge set (`bench/inject-sweep.ts`):

| top-N | both@N: DIRECT → GRAPH | avg noise slots (GRAPH) |
|---:|---|---:|
| 2 | 15% → **25%** | 0.6 / 2 |
| 3 | 35% → **43%** | 1.3 / 3 |
| 5 | 53% → **63%** | 3.0 / 5 |
| 8 | 70% → **85%** | 5.7 / 8 |
| 10 | 100% → 100% | 7.5 / 10 |

- ✅ **GRAPH beats DIRECT at every N** (until saturation) — so inject adds value *beyond just taking
  a bigger k*; at a fixed budget, graph traversal lands both supports more often (+8 to +15pp).
- ⚠️ **The N=10 tie is a HotpotQA artifact** (only 10 candidates → top-10 = everything). In a real
  store top-N ≠ the whole set, so the graph edge would persist past N=10.
- ⚠️ **Noise scales hard**: both@5 = 63% costs **3 of 5** injected slots being non-support; both@8
  = 85% costs ~5.7/8. So inject trades context noise for recall — the right N depends on how much
  noise the consuming model tolerates (stronger models → push N higher). This is exactly the curve
  a *depth-weighted* injection would target: fill those noise slots with confirmed (deep) memories
  rather than arbitrary neighbours.

**Now wired** (v0.13.0): the `recall` MCP tool takes `inject:true` (→ `{keys, memories}` in one
call), plus `inject_prefer_depth` (surface confirmed/deep memories first) and
`inject_explore_shallow` (reserve one slot for a weak/recent memory — an ε-exploration so shallow
memories can resurface and be reinforced). The selection policy is unit-tested (`selectInject`,
`test/inject-select.test.ts`). **Honest limit:** depth-weighting and exploration only matter once
memories sit at *different* depths over real use — a one-shot benchmark has all memories at equal
(near-zero) depth, so their *value* is longitudinal and **not** demonstrated here; only the policy
logic is verified. Defaults are off; the deliberate-navigation flow is unchanged.

---

## 6. Phrase-key bridging: choosing the gate

**Why.** §1 found the graph's gain is in *reach*, not rank — 2-hop hits land at ranks 9-11 and
`hit@5` shows no gain. Bridging a legacy phrase key onto the atomic keys it contains turns a
2-hop path into a 1-hop link, so the hypothesis was that it moves those hits *into* the top 5.
The design question was which gate decides that a bridge is warranted.

**Design** (`bench/phrase-fixture.json`, 60 memories / 14 queries): six memories filed under a
phrase key only — the pathology measured on the owner's live store, where 177 of 204 3+-token
keys were singletons — in a store that already has an atomic hub for the same concept. `bridge`
queries ask with the atomic concept. `direct` queries are controls. Two memories carry a phrase
key that shares a *token* with a hub but is off-topic for it (`예산 리뷰 회의 결과` vs the review
hub); their appearance in a topical query's top 5 is the precision cost. 44 of the 60 memories
are distractors so the top-10 window is selective — at 16 memories every condition scored 100%
and the fixture had no resolution at all.

| Condition | gate |
|---|---|
| **NO-BRIDGE** | `KEYMEM_PHRASE_BRIDGE=false` — phrase keys stay orphaned |
| **COSINE** | ships: bridge only if the atomic key's own cosine to the memory clears `contentRecallShort` (0.46) |
| **STRUCTURAL** | experimental: bridge if the phrase key is a singleton *and* the atomic key is a hub, at a fixed low weight (0.3) |

### Results (`bge-m3`)

| category | metric | NO-BRIDGE | COSINE | STRUCTURAL |
|---|---|---:|---:|---:|
| **bridge** (6) | reach@10 | 100% | 100% | 100% |
| | **hit@5** | 83% | **100%** | 100% |
| | MRR | 0.33 | **0.38** | 0.36 |
| direct (4) | hit@5 | 100% | 100% | 100% |
| | MRR | 0.88 | 0.88 | **0.75** |
| notfound (2) | not-found acc | 2/2 | 2/2 | 2/2 |
| — | off-topic in top-5 | 0 | **0** | **2** |

### What this shows (and doesn't)

- ✅ **The cosine gate ships; the structural one loses on its own terms.** STRUCTURAL was meant to
  buy reach that the cosine gate refuses. It bought none (reach@10 identical at 100%), ranked
  worse on the bridge queries it was built for (MRR 0.36 vs 0.38), degraded the `direct` control
  (0.88 → 0.75), and pulled both off-topic memories into a topical top 5. A shared token really
  is not a shared topic, and the store's own calibrated bar is the better judge.
- ⚠️ **The gain is small and n is small.** Six bridge queries; `hit@5` 83% → 100% is one query
  moving, and per-query ranks improved in only 2 of 6 (3→2 and 6→4), stayed put in 4. Read this
  as "no regression anywhere, a nudge where it fires", not as a headline number.
- ⚠️ **The gate confines bridging to memories that were already reachable.** Measured
  key↔content cosines for the six orphans: 0.488 / 0.480 / 0.474 / 0.469 (bridged) vs 0.390 /
  0.293 (refused). Every firing sits in a narrow band just above the 0.46 bar — so bridging is a
  *rank-level* fix, not the reach-level fix §1 asks for. Whatever closes that gap will not be
  another link-time heuristic on the same signal.

---

## 7. Hebbian key associations — built, measured, removed

Recorded because the negative result is the useful part: it costs a day to rebuild this
and rediscover the same thing.

**The idea.** keymem's graph is bipartite — keys connect only through the memories they
share, so two key clusters with no memory in common are unreachable from each other at
*any* hop count, however related they are. A user's queries carry knowledge the data does
not: asking about `배포` and `릴리스` in the same breath, repeatedly, is evidence they belong
together even when no single memory carries both. So: keys that co-match a query and then get
a confirmed read accrue an association, and after three confirmations it becomes a traversable
edge, scored below `HOP_DECAY` so it can never outrank a real shared memory.

It worked, in the sense that the edges formed and recall traversed them — a unit test on a
purpose-built fixture confirmed it. **It was removed anyway, because no non-circular
measurement could show it helping.**

| instrument | OFF | ON | edges formed |
|---|---|---|---|
| §6 fixture (60 memories), after a recall→read training phase | bridge/direct/precision/notfound all at ceiling | **identical, every cell** | 30 traversable |
| Owner's real store (530 memories), real-eval queries, train/held-out split | train 5/6, held-out 6/6 | **train 5/6, held-out 6/6** | 36 traversable |

### Why it was removed rather than shipped behind a flag

- **Three independent measurements, no movement in either direction.** Not better, not worse.
  For a mechanism whose entire job is adding edges to a graph, "no effect" is not a neutral
  result — it means the edges are not load-bearing.
- **The only place the benefit appeared was a fixture built to show it.** The synthetic
  training trace hand-writes "the user asked about A and B together" and then checks that A
  reaches B — that re-proves the implementation, not the idea. The held-out split on the real
  store was the honest instrument, and it moved nothing. (Its held-out half already sat at 6/6
  before training, so strictly it could only have detected harm — which is itself a finding:
  on this store there is no reachability gap for associations to fill.)
- **Edges formed promiscuously.** Pairing every key a query matched accrued 96 traversable
  edges from 12 queries in 3 rounds; capping the co-match set to a query's top 3 keys cut it to
  30 with identical metrics. A mechanism that needs a cap to avoid connecting everything to
  everything, and that shows no benefit once capped, is carrying risk for nothing.
- **A default-off flag is not a resolution.** It ships the maintenance cost and the reader's
  question ("should I turn this on?") without ever answering it.

**What would change the verdict:** a store with a genuine reachability gap — disjoint key
clusters where the answer is unreachable at any hop — plus a real usage trace (host transcripts,
not a synthetic one) showing those clusters get queried together. Neither exists here yet.

---

## 8. Embedding model comparison — why it is still bge-m3

Asked in 2026-08 whether a newer, smaller multilingual model should replace bge-m3 (568M,
1024-dim). Answer: no, and the reason is worth writing down because the obvious evidence
(MTEB rank, parameter count) points the other way.

**First, a constraint that blocks evaluation itself.** `fastembed@2.1.0` — the current
release — depends on `@anush008/tokenizers@^0.0.0`, which cannot parse a modern
`tokenizer.json`: merges serialized as `["Ġ","Ġ"]` pairs rather than `"Ġ Ġ"` strings, plus
`ignore_merges`, both introduced in newer `tokenizers`. Every model tried here fails to load
through keymem's CUSTOM path with `data did not match any variant of untagged enum
ModelWrapper`. bge-m3 works only because BAAI's tokenizer is the older Unigram serialization
(the note in `modelDownload.ts` had already found the edge of this). Rewriting the merges by
hand makes the model load, but changes tokenization — numbers measured that way are measuring
the workaround, not the model. The comparison below therefore ran **out of tree**, embedding
with `@huggingface/transformers` and injecting the vectors through the `__setTestEmbedder`
seam, so no production code or dependency changed.

**Harness validity.** bge-m3 through that path (CLS pooling, q8) reproduces its native
fastembed scorecard — 58% / 75% / 0.67 vs 58% / 72% / 0.65, zero embedder cache misses — so
the seam is measuring what the real pipeline measures. Pooling is per-model on purpose:
scoring bge-m3 with mean pooling instead of CLS drops a pair's related/unrelated gap from
0.245 to 0.132. Getting that detail wrong is how model comparisons produce confident nonsense.

### Results (`bench/fixture.json`, 12 answerable + 4 not-found; bge-m3 thresholds throughout)

| model | params / dim | recall@1 | recall@5 | MRR | not-found | dup↔indep margin |
|---|---|---:|---:|---:|---:|---:|
| **bge-m3** (current) | 568M / 1024 | **58%** | **75%** | **0.67** | 0/4 | −0.093 |
| EmbeddingGemma-300m | 300M / 768 | 58% | 75% | 0.67 | 1/4 | −0.095 |
| granite-embedding-97m-r2 | 97M / 384 | 8% | 33% | 0.19 | 0/4 | — |

### Verdict

- **granite-97m-r2 loses outright** on this workload — 8% recall@1, and 0% on both
  cross-lingual categories — despite a strong sub-100M retrieval claim. Its ONNX is 93MB
  quantized and it handles 32K context; none of that helps here. MTEB rank did not predict it.
- **EmbeddingGemma-300m ties exactly** on every retrieval metric at 300M/768-dim. Its one
  edge is not-found accuracy (1/4 vs 0/4) — a single query out of four — and its cosine bands
  sit higher but with the *same* shape: the duplicate↔independent margin stays negative and
  nearly identical (−0.095 vs −0.093), so it buys no separability, which is the property
  bge-m3 was chosen for in the first place.
- **A tie on 12 queries is not a reason to migrate.** Adopting either model means replacing
  fastembed's tokenizer layer and taking ownership of per-model pooling and prompt
  conventions. Nothing measured here pays for that. Revisit when there is an eval large
  enough for a tie to mean something, or a model that wins outright rather than matching.

---

## 9. Prompt-cache payload ablation — keep eight, make them compact

**Question.** Can default recall reduce ranked keys from eight to three without degrading the
memory available to the consuming LLM? Payload savings alone are not evidence: a relevant key at
rank 4–8 may be the only route to the correct memory.

**Method.** `bench/prompt-cache-ab.ts` loads a labeled corpus into a real `MemoryGraph`, obtains one
top-8 key ranking per query, and evaluates three paired views of that exact ranking:

| variant | candidates | representation |
|---|---:|---|
| top8 (control) | 8 | current full key objects |
| top3 | 3 | current full key objects |
| compact8 | 8 | identity/ranking fields plus `aliases`, `key_type`, `is_hub`, `specificity` |

Task score is objective expected-key reachability on a 0–1 scale, not an LLM judge. The gate
requires identical Top-1 identity, zero reachability loss, paired task-score CI lower bound ≥ 0,
and at least 20% payload reduction. Missing paired task scores can never pass. The answer-quality
gate runs blinded `gpt-5.6-sol` pairs on a 12-point rubric with a 0.25-point non-inferiority margin.
Provider metrics come directly from Codex JSONL usage. Because cache routing can be intermittent,
the rollout gate uses only batch pairs where both variants report the same nonzero cached prefix.

### Results (`bge-m3`, 30 labeled queries)

| corpus | top8 reachable | top3 reachable | top3 payload | compact8 reachable | compact8 payload |
|---|---:|---:|---:|---:|---:|
| retrieval fixture (16 queries / 12 answerable) | 12/12 | 11/12 | −58.5% | 12/12 | −37.2% |
| associative fixture (14 queries / 11 answerable) | 8/11 | 4/11 | −62.5% | 8/11 | −37.3% |
| **combined** | **20/23** | **15/23** | — | **20/23** | — |

The first five-field compact profile failed one repeated LLM run: the “미나 취향” answer selected
an extra broad key, giving mean delta −0.125 and CI95 [−0.375, 0]. The final navigation-compact
profile therefore retains `aliases`, `key_type`, `is_hub`, and `specificity`, while removing
`score_kind`, `cluster_size`, `evidence`, and `suggested_tool`.

| final navigation-compact gate | retrieval fixture | associative fixture | combined |
|---|---:|---:|---:|
| LLM-judged cases | 16 | 14 | 30 |
| score delta / CI95 | 0 / [0, 0] | 0 / [0, 0] | all paired deltas 0 |
| paired-warm batches | 2 | 1 | 3 |
| cache hit rate, top8 → compact8 | 72.31% → 74.99% | 69.92% → 73.08% | 71.50% → 74.34% |
| uncached input reduction | 12.88% | 14.38% | 13.42% |

**Verdict.** **Reject top-3.** It loses five relevant-key routes, a combined −21.7 percentage
points versus top-8. **Use navigation-compact top-8.** It preserves every candidate and all 30
paired answer scores, reduces key-list payload about 37%, and improves matched warm-cache hit rate
by 2.84 percentage points. The runtime keeps `top_k=8` and compacts only the serialized `keys`
view. Machine-readable results are in `bench/prompt-cache-ab-results.json`,
`bench/fixture-prompt-cache-llm-results.json`, and
`bench/assoc-fixture-prompt-cache-llm-results.json`.

---

## Reproduce

```bash
tsx bench/ablation.ts     # §1 associative-recall ablation (real bge-m3, ~570MB first run)
tsx bench/perf.ts         # §3 latency vs store size (synthetic embedder)
tsx bench/run.ts          # the existing search-quality regression fixture

# §2 external HotpotQA — first fetch a slice (no full download), then run:
curl -s "https://datasets-server.huggingface.co/rows?dataset=hotpotqa/hotpot_qa&config=distractor&split=validation&offset=0&length=100" \
  | python3 -c "import sys,json;rows=[r['row'] for r in json.load(sys.stdin)['rows']];print(json.dumps([{'id':r['id'],'question':r['question'],'answer':r['answer'],'type':r['type'],'support':r['supporting_facts']['title'],'titles':r['context']['title'],'paras':[' '.join(s) for s in r['context']['sentences']]} for r in rows]))" \
  > bench/hotpot-slice.json
tsx bench/hotpot.ts 100
tsx bench/hotpot-agentkeys.ts bench/hotpot-agentkeys.json   # §2 validity check w/ blind agent keys
tsx bench/inject-sweep.ts                                   # §5 inject top-N value/noise sweep
tsx bench/phrase-bridge.ts                                  # §6 phrase-key bridging gate ablation
npm run bench:prompt-cache -- bench/fixture.json            # §9 top3 vs compact8
npm run bench:prompt-cache -- bench/assoc-fixture.json      # §9 associative controls
npm run bench:prompt-cache-llm -- bench/fixture.json        # §9 LLM + provider gate
npm run bench:prompt-cache-llm -- bench/assoc-fixture.json  # §9 associative LLM + provider gate
```

Sources: [LoCoMo](https://github.com/snap-research/locomo) · [LongMemEval](https://github.com/xiaowu0162/LongMemEval) · [Zep vs Mem0 methodology dispute](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/) · [Mem0 paper](https://arxiv.org/pdf/2504.19413) · [Agentic search replacing RAG (VentureBeat, 2026)](https://venturebeat.com/data/context-architecture-is-replacing-rag-as-agentic-ai-pushes-enterprise-retrieval-to-its-limits)
