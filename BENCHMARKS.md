# keymem — Benchmarks

What this measures and, honestly, what it doesn't. The goal is to **prove how much keymem's
key-graph actually buys you** — isolated causally, not asserted by metaphor — and to mark the
limits plainly.

> TL;DR: On connected-but-dissimilar recall, the key-graph reaches **+33pp** more targets than
> flat 1-hop retrieval can reach at all (reach@10 50% → 83%). The read path was also made
> **O(1) instead of O(graph)** (read_memory p50 ~45ms → ~0.01ms @ 500 memories). But the
> associative hits land **low-ranked** (not top-5), not-found precision is weak at small scale,
> and none of this is a head-to-head SOTA claim vs mem0/Zep. Details below.

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

**Design.** Same engine, same data, same real embeddings (`bge-m3`), two retrieval conditions
that differ *only* in graph expansion:

| Condition | call | meaning |
|---|---|---|
| **DIRECT** | `recall(expand=false, hops=1)` | 1-hop semantic match — approximates a flat semantic store's reach |
| **GRAPH** | `recall(expand=true, hops=2)` | keymem's multi-hop key-graph traversal |

**Dataset** (`bench/assoc-fixture.json`): a 14-memory bilingual persona graph. Each `assoc2`
query's answer is a **far memory reachable only via a shared key two hops away** (e.g. *"미나가
키우는 강아지"* → the dog's *allergy* fact, reachable only through the shared key `보리`). The
query has low direct similarity to that target — so flat retrieval should miss it and graph
traversal should reach it. `direct` queries are 1-hop controls; `notfound` must return nothing.

**Metrics.** `reach@10` (target anywhere in the top 10 — does the system find it *at all*),
`hit@5` (target in top 5 — ranking-sensitive), `MRR`. Run: `tsx bench/ablation.ts`.

### Results (`bge-m3`, n per category)

| category | metric | DIRECT | GRAPH | Δ |
|---|---|---:|---:|---:|
| **assoc2** (6) | **reach@10** | 50% | **83%** | **+33pp** |
| | hit@5 | 33% | 33% | 0pp |
| | MRR | 0.13 | 0.23 | +0.10 |
| direct (5) | reach@10 | 100% | 100% | 0pp |
| | hit@5 | 80% | 80% | 0pp |
| | MRR | 0.82 | 0.69 | **−0.13** |
| notfound (3) | not-found acc | 1/3 | 1/3 | — |

### What this proves (and doesn't)

- ✅ **The key-graph reaches connected-but-dissimilar memories that flat retrieval cannot.**
  On `assoc2`, DIRECT reaches 3/6 targets; GRAPH reaches 5/6 (the 6th lands at rank 11, just
  outside the window). That +33pp reach is the thesis — *measured*, not asserted. Per-query,
  the graph pulls in the dog-allergy / climbing-injury / caffeine-sleep facts via shared keys
  that 1-hop similarity never surfaces.
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

## 2. Read-path latency (the v0.12.1 fix)

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

## 3. Honest scope & the trajectory caveat

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

## Reproduce

```bash
tsx bench/ablation.ts     # associative-recall ablation (real bge-m3, ~570MB first run)
tsx bench/perf.ts         # latency vs store size (synthetic embedder)
tsx bench/run.ts          # the existing search-quality regression fixture
```

Sources: [LoCoMo](https://github.com/snap-research/locomo) · [LongMemEval](https://github.com/xiaowu0162/LongMemEval) · [Zep vs Mem0 methodology dispute](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/) · [Mem0 paper](https://arxiv.org/pdf/2504.19413) · [Agentic search replacing RAG (VentureBeat, 2026)](https://venturebeat.com/data/context-architecture-is-replacing-rag-as-agentic-ai-pushes-enterprise-retrieval-to-its-limits)
