# Neural memory experiment

A graph-free test of whether a small model can learn **what to write into its own changing
parameters** from delayed recall utility.

This is intentionally not integrated with keymem's key/value graph or an LLM. Phase 0 isolates the
write-selection mechanism:

```text
observation (key, value, noisy evidence)
        │
        ├─ learned gate + surprise
        ▼
matrix-valued fast weights W(t)     ← the memory
        │
future query ───────────────────────► recalled value
```

The synthetic task supplies observations, many of which are distractors. Only a subset is queried
later. The model never receives a direct `write` label; future recall loss trains the write gate.
Because the fast-weight dimension is deliberately smaller than the number of possible bindings,
writing every distractor creates interference.

## Run

The subproject uses its own environment and does not change the root TypeScript package.

```bash
cd experiments/neural-memory
uv sync --python 3.13
uv run pytest -q
uv run python train.py
uv run python train_competition.py
```

For a quick smoke run:

```bash
uv run python train.py --device cpu --steps 20 --eval-batches 2 --batch-size 8
```

The command trains two otherwise identical models:

- `learned_gate`: decides its update strength from the current input and surprise.
- `always_write`: writes every observation with strength 1.

It prints JSON containing held-out recall accuracy and the learned gate separation. Chance accuracy
is included explicitly. Do not interpret a single seed as evidence; the decision run should use at
least three seeds and report mean and standard deviation.

See [RESEARCH.md](RESEARCH.md) for the prior-work mapping and staged decision gates, and
[RESULTS.md](RESULTS.md) for the first three-seed run and its interpretation boundary.

## Phase 1: competitive allocation

`train_competition.py` removes the write-evidence shortcut and asks a separate question: can
delayed recall loss teach a small router to allocate colliding associations across independent
fast-weight blocks? The same local key deliberately maps to different values in four contexts.
No target block or allocation label is supplied to the learned variant.

The comparison fixes both the per-write update budget and total fast-state capacity at 576 scalars:

- `single`: one 24 x 24 matrix.
- `uniform`: four 12 x 12 matrices, every write spread equally.
- `competitive`: four 12 x 12 matrices, allocation learned only through recall loss.
- `oracle`: four 12 x 12 matrices, one block assigned to each context; a routing reference.

Besides recall accuracy, the run reports wrong-context intrusion rate and whether contexts settle
into a one-to-one block assignment. The context identifier is still an explicit synthetic feature;
this tests allocation dynamics, not discovery of latent semantic context from language.

On an M4 Pro, the default 200-step run takes roughly 25 seconds with MPS. See the phase-1 section
of [RESULTS.md](RESULTS.md) for the three-seed result.

## Phase 2: frozen text cues plus joint write/allocation

Phase 2 removes categorical context from the learned path. A frozen BGE-small sentence encoder
turns held-out natural-language paraphrases into features; the memory layer sees those features,
not context IDs or write labels. Its outer-loop recall loss jointly trains:

```text
stable-vs-temporary cue embedding ──► scalar write gate
context paraphrase embedding ───────► competitive block allocation
                                      │
key/value observation ───────────────► per-episode fast weights
```

Prepare the frozen features once, then train only the memory layer:

```bash
uv sync --group text
uv run python prepare_text_embeddings.py
uv run python analyze_text_embeddings.py
uv run python train_semantic.py --summary
```

Training sentences and evaluation paraphrases are disjoint. The synthetic episodes put a
temporary, conflicting observation after every stable binding, while reusing the same local keys
across four semantic contexts. The key/value targets are randomized each episode, so slow weights
cannot memorize them. Categorical context IDs and stable/distractor flags exist only inside the
generator for metrics and the oracle control.

## Phase 3: episode-local online allocation

Phase 3 uses eight global semantic domains but samples a different subset of four in every
episode. There are only four blocks, so a fixed global domain-to-block map cannot avoid collisions.
`OnlinePrototypeMemory` instead stores episode-local context prototypes alongside its fast weights:

```text
new context ──► first empty block + new fast prototype
seen context ─► nearest occupied prototype + existing block
```

The BGE feature is compressed through a fixed random projection to 16 dimensions. Four 11 x 11
association matrices use 484 scalars and routing metadata uses 68, for 552 total runtime scalars;
the static baseline gets four 12 x 12 matrices, or 576 scalars.

```bash
uv run python prepare_online_embeddings.py
uv run python analyze_online_embeddings.py
uv run python train_online.py --summary
```

The allocation prior (novel context claims an empty block) is architectural. Delayed recall loss
learns its novelty threshold but is not yet learning the entire allocation algorithm. Context
features are held constant within an episode in this phase so the experiment isolates online block
assignment from semantic drift.

## Phase 4: utility-aware eviction

Phase 4 overfills memory with six active contexts and four blocks. Only four contexts are queried
later, and usefulness is randomized independently of arrival order. A frozen natural-language cue
describes each context as durable or temporary. The learned model receives the cue embedding but
not the retention label.

`PriorityEvictionMemory` gives each occupied block a learned scalar priority. When a novel context
arrives to a full memory, it either replaces the lowest-priority block or rejects the incoming
write. The discrete forward action has a straight-through comparison so delayed recall loss can
train the priority scorer.

```bash
uv run python train_eviction.py --summary
```

The model uses 484 association scalars plus 72 prototype/count/priority scalars, for 556 total
runtime state. The cue vocabulary is still hand-authored, context is stable within each episode,
and the eviction prior itself is architectural rather than discovered from scratch.

## Phase 5: delayed utility and eligibility traces

Phase 5 withholds every durability cue during observation. Six contexts arrive in a fully
interleaved stream and are first written to six short-lived trace slots. Only after the observation
stream ends do natural-language outcome cues arrive. The memory then scores the existing traces,
keeps the top four, and zeros the other association matrices before recall.

```bash
uv run python train_delayed.py --summary
```

Each trace uses a 9 x 9 association matrix, a 16-dimensional context prototype, a count, and a
priority: 594 transient runtime scalars in total. The forward top-k consolidation is discrete; a
straight-through mask carries future recall gradients into the linear priority scorer.

This is an explicit short-term trace, not a biological timing model. It has enough slots for all
six candidate contexts, does not decay with wall-clock time, and receives feedback paired with the
relevant context. Later work must learn that credit assignment from implicit outcomes.

## Phase 6: global feedback over real conversation traces

Phase 6 removes the explicit context pointer and replaces synthetic observations with the user's
local Claude prompt history. The loader sessionizes `~/.claude/history.jsonl` by project and idle
gap, strips tool and internal metadata, and weak-labels cases where a later user turn uniquely
reuses specific anchors from one of eight older turns. Raw text is never written to the repository
or embedding artifact.

```bash
uv run python analyze_claude_logs.py --summary
HF_HUB_OFFLINE=1 uv run python prepare_claude_log_embeddings.py
uv run python train_global_feedback.py --steps 1000 --seed 7 --summary
```

`GlobalFeedbackSelector` observes eight frozen 384-dimensional BGE features and stores only eight
learned 64-dimensional traces. A later global feedback embedding is projected into the same memory
space and scores every trace; there is no trace ID, target label, or context pointer in the forward
path. Its top-k mask is the consolidation action.

The current weak label is deliberately only a bootstrap target. It is constructed from lexical
anchor recurrence, and full-dimensional frozen cosine remains a stronger baseline. The next label
must come from outcomes independent of semantic recurrence, such as later corrections, accepted
actions, or task success, before this can support a claim that the model learns what is worth
storing rather than which old utterance resembles the present one.

## Phase 7: natural delayed strengthen/revise outcomes

Phase 7 extracts explicit acceptance and correction events from the same local history. Short
feedback such as `ㅇㅇ`, `좋아`, and `ㄱㄱ` is weak-labeled as accept/strengthen; high-precision
correction forms such as `ㄴㄴ`, `아니`, `잘못`, and `되돌려` are revise/weaken. These rules are the
teacher only. The learned gate receives frozen signed character n-gram features and never receives
the action label.

```bash
uv run python analyze_claude_outcomes.py --summary
uv run python prepare_claude_outcome_features.py
uv run python train_outcome_gate.py --steps 1000 --seed 7 --summary
```

`DelayedOutcomeGate.observe()` first writes a 64-dimensional provisional request trace. Its later
`act()` call compares three conditions: request-only, feedback-only, and request-plus-feedback.
`apply_outcome()` converts the action logits to a differentiable strength update. Accepted traces
remain strong; correction traces are weakened and marked for revision.

The immediate-previous-interaction association is still an architectural assumption, and regex
weak labels are not ground truth. Phase 6 and 7 together now implement the two halves needed for a
real memory update—select a responsible old trace and choose strengthen versus revise—but they have
not yet been trained jointly from downstream task reward.
