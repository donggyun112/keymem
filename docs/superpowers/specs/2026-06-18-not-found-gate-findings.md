# Not-Found Gate — Empirical Findings (improve→test loop)

**Date:** 2026-06-18
**Method:** eval harness over a real 19-memory / 78-key snapshot, two query sets
(a tuning set used to pick thresholds, and a **held-out** set of harder phrasings +
adversarial near-topic absent facts). Metrics: FOUND recall (real query → ≥1 hit) and
NOT-FOUND detection (unrelated query → `[]`). Hiding a real memory (false negative) is
treated as a worse failure than returning noise (leak).

## What was tried, and what each scored

| Config (e5) | tuning set | held-out | note |
|---|---|---|---|
| absolute gate `minScore=0.8` (0.7.0) | recall 100% / NF 0% | — | never hides, never gates (inert on e5) |
| dist gate `gateZ=2.5` (shipped 0.8.0) | acc 63%, **hides 50% of real matches** | — | harmful default |
| key-proximity `keyGate=0.88` | **100%** | **63%** | overfit — gap was an artifact of the tuning set |
| e5 gate **off** (chosen default) | recall 100% / NF 0% | recall 100% / NF 0% | safe: never hides |

| Config (bge-m3, re-embedded same corpus) | tuning set | held-out |
|---|---|---|
| absolute gate `minScore=0.55` | **96%** (NF 92%) | 67% (recall 11/12) |

## Why no similarity gate separates found/not-found on this corpus

Measured feature distributions (real corpus) overlap badly between found and not-found:

- **content robust-z:** found ranged 1.5–5.8, not-found 0.8–4.7 (`전화번호`, a not-found
  query, scored z=4.70 — higher than most real matches).
- **best concept-key cosine (e5):** clean on the tuning set (found ≥0.883, not-found
  ≤0.875) but collapsed held-out (`파이썬 쓰니` found 0.849 < `회사 이름` not-found 0.887).
- **top content cosine (bge-m3):** `동균 나이` (not-found) 0.645 > `운동` (found) 0.416.

**Root cause:** in a personal-memory store, person-attribute queries that share the
subject ("동균 나이", "동균 형제자매", "전화번호") are semantically near the real facts
about that subject. No similarity threshold distinguishes "attribute X is stored" from
"attribute X is absent but the subject is known" — that requires reasoning (NLI/LLM),
which is an explicit non-goal of the heuristic design.

## Decisions

1. **e5 not-found gate is OFF by default** (`gateZ=0`, `keyGate=0`). e5 reverts to 0.7.0
   behavior: returns noise for unrelated queries but never hides a real match. The earlier
   shipped default (`gateZ=2.5`) hid 50% of real matches and was a regression.
2. **bge-m3 is the recommended path** for not-found detection: its absolute gate
   (`minScore=0.55`) scores ~96% on realistic queries (validated by re-embedding the real
   corpus), clearly better than e5. No `minScore` retune helps (distributions overlap);
   left at 0.55.
3. **The gate machinery (`gateZ`, `keyGate`) is retained as env-opt-in**
   (`SUPER_MEMORY_GATE_Z`, `SUPER_MEMORY_KEY_GATE`, recall `min_z` / `min_key_gate`) for
   users who accept the precision/recall tradeoff, and it remains useful where a model
   separates cleanly.
4. **In-result noise trimming stays `min_rel_score`** (verified effective on e5).

## Residual limitation (documented, not fixed)

Adversarial person-attribute "not found" queries leak on every similarity-only config
(e5 and bge-m3). Reliably answering "this specific attribute is not stored" needs
NLI/LLM judgement — out of scope for the heuristic retriever. Operators who need it
should layer an LLM check above recall, or rely on `min_rel_score` + the agent ignoring
clearly off-topic results.
