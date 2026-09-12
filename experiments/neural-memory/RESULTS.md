# Phase 0 results

Date: 2026-09-12

Command, repeated with seeds 7, 17, and 29:

```bash
uv run python train.py --steps 600 --eval-batches 50 --batch-size 64 --seed <seed>
```

Hardware selected by PyTorch: Apple M4 Pro via MPS.

| Seed | Learned-gate accuracy | Always-write accuracy | Accuracy lift | Useful gate | Distractor gate | Gate separation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.9255 | 0.6507 | +0.2748 | 0.6238 | 0.3920 | 0.2318 |
| 17 | 0.9235 | 0.6504 | +0.2731 | 0.6177 | 0.3929 | 0.2248 |
| 29 | 0.9252 | 0.6493 | +0.2760 | 0.6242 | 0.3947 | 0.2295 |
| **Mean** | **0.9248** | **0.6501** | **+0.2746** | **0.6219** | **0.3932** | **0.2287** |

Chance recall accuracy is 0.0078 (1/128). All three seeds clear the phase-0 mechanism gate:
the learned updater both beats the always-write control and assigns larger updates to observations
that will be useful later.

## Interpretation boundary

This is positive evidence for the narrow mechanism, not yet for human-like or semantic memory.
The present-time evidence feature is intentionally correlated with future usefulness, although its
distributions overlap. The experiment shows that delayed recall loss can train a differentiable
write gate to exploit such evidence and reduce interference. It does not show that a language model
can discover the relevant evidence from raw conversation.

The next experiment should remove the scalar evidence shortcut. It should derive write strength
from frozen text embeddings containing novelty, repetition, correction, and goal-relevance cues,
while retaining the same always-write control.

# Phase 1: competitive allocation

Date: 2026-09-12

Command, run with seeds 7, 17, and 29:

```bash
uv run python train_competition.py \
  --steps 200 --eval-batches 50 --batch-size 64 --seed <seed> --summary
```

All variants had the same 576-scalar fast-state capacity and unit total update budget per write.
The task used four contexts and eight shared keys per context. For every shared key, each context
assigned a different target value, creating deliberate overwrite pressure.

| Variant | Recall accuracy | Wrong-context intrusion | One-to-one assignment |
| --- | ---: | ---: | ---: |
| Single matrix | 0.2347 +/- 0.0012 | 0.7042 +/- 0.0036 | N/A |
| Uniform four blocks | 0.1956 +/- 0.0004 | 0.5868 +/- 0.0013 | 0.2500 |
| **Learned competition** | **0.9243 +/- 0.0004** | **0.0038 +/- 0.0002** | **0.9982 +/- 0.0006** |
| Oracle routing | 0.9190 +/- 0.0005 | 0.0039 +/- 0.0002 | 1.0000 |

Values are mean +/- population standard deviation over three seeds. Chance recall accuracy was
0.0156. Each seed took about 25 seconds on an Apple M4 Pro using MPS.

## What this establishes

With no block labels, delayed recall loss reliably taught the router to send four conflicting
contexts to four different fast-weight blocks. The allocation was almost exactly one-to-one and
removed nearly all wrong-context intrusions. Episode values were randomized, so the slow weights
could learn the allocation policy but could not memorize the recalled associations themselves.

This is a positive mechanism result, not a semantic-memory result. Context was supplied as an
explicit categorical input at both write and recall time. The experiment therefore does not yet
show that a model can infer latent context from raw text, decide whether an observation deserves
storage, allocate more contexts than available blocks, or forget over long horizons. The learned
variant slightly exceeding oracle accuracy is not evidence against the reference: models were
trained independently, and learned routing can retain tiny soft cross-block mixtures.

The next falsifiable step is to replace the context ID with frozen text embeddings and include
novelty, repetition, and correction episodes. A learned encoder/router must then recover useful
latent contexts without receiving context or write labels.

# Phase 2: semantic joint write and allocation

Date: 2026-09-12

Frozen encoder: `BAAI/bge-small-en-v1.5` (384 dimensions). The encoder was used only once to
precompute features and was never updated by the memory experiment. Training and evaluation used
disjoint natural-language paraphrases.

Final command, run with seeds 7, 17, and 29:

```bash
uv run python train_semantic.py \
  --embeddings artifacts/bge_small_en.pt \
  --steps 600 --eval-batches 50 --batch-size 64 --seed <seed> \
  --variants single_gate competitive_always joint oracle --summary
```

| Variant | Recall | Context intrusion | Distractor overwrite | Useful / distractor gate | Assignment |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gate only, single matrix | 0.2497 +/- 0.0000 | 0.7490 +/- 0.0001 | 0.0001 +/- 0.0000 | 0.122 / 0.010 | N/A |
| Allocation only, always write | 0.0341 +/- 0.0018 | 0.0652 +/- 0.0028 | 0.4289 +/- 0.0100 | 1.000 / 1.000 | 0.4765 +/- 0.0163 |
| **Joint learned memory** | **0.9757 +/- 0.0006** | **0.0010 +/- 0.0000** | **0.0030 +/- 0.0002** | **0.658 / 0.105** | **0.9988 +/- 0.0003** |
| Oracle write and routing | 0.9823 +/- 0.0003 | 0.0009 +/- 0.0000 | 0.0003 +/- 0.0000 | 1.000 / 0.000 | 1.0000 |

Values are mean +/- population standard deviation over three seeds. Chance recall was 0.0156.
All variants again had 576 fast-state scalars. The joint model reached 99.3% of oracle recall while
receiving neither write labels nor context IDs in its learned path.

## Ablation reading

The gate-only model correctly weakened temporary observations but suffered the expected 3/4
wrong-context intrusion because one matrix could not distinguish four colliding contexts. The
allocation-only model could not recover the stable value because the later temporary value was
always written. Only the joint model learned both necessary operations.

An earlier MiniLM run exposed a useful failure mode. It embedded the held-out sentence "one-off
occurrence rather than my usual choice" closer to durable preferences, apparently underweighting
the negation. The joint model then plateaued at 0.6528 recall with a 0.3210 overwrite rate. BGE-small
separated all held-out cue paraphrases and restored oracle-level recall. This isolates frozen input
representation as a real bottleneck rather than hiding it inside the memory update rule.

## Interpretation boundary and next test

This establishes a stronger synthetic phenomenon: frozen natural-language features can drive a
separate model-state memory to decide both update strength and memory region using only delayed
recall loss. It still uses four globally recurring semantic domains, short episodes, fixed
key/value vocabularies, and hand-authored durability contrasts. It does not yet demonstrate
open-world context creation, capacity eviction, correction chains, or integration with a frozen
generative LLM.

The next experiment should randomize which semantic domains co-occur per episode and provide more
domains than blocks. That forces online allocation, reuse, and eviction instead of allowing the
slow router to learn a permanent domain-to-block map.

# Phase 3: episode-local online allocation

Date: 2026-09-12

Eight global semantic domains were represented by frozen BGE-small embeddings. Each episode drew
a new random subset of four domains for four available blocks, randomized the order of first
arrival, and used held-out context phrasings at evaluation. The same local keys had conflicting
values across all four active contexts.

Final command, run with seeds 7, 17, and 29:

```bash
uv run python train_online.py \
  --steps 300 --eval-batches 50 --batch-size 64 --seed <seed> --summary
```

| Variant | Total runtime state | Recall | Context intrusion | Unique blocks used | Collision-free episodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static semantic router | 576 | 0.6174 +/- 0.0186 | 0.2668 +/- 0.0106 | 0.7643 +/- 0.0139 | 0.1907 +/- 0.0239 |
| **Online prototype router** | **552** | **0.8809 +/- 0.0004** | **0.0054 +/- 0.0001** | **1.0000** | **1.0000** |
| Episode-local oracle | 484 | 0.8809 +/- 0.0004 | 0.0054 +/- 0.0001 | 1.0000 | 1.0000 |

Values are mean +/- population standard deviation over three seeds. The online router exactly
matched the oracle at every seed even though the order and identity of the four active domains
changed per episode. Its learned novelty threshold moved from 0.9000 to 0.9627 +/- 0.0027.

The capacity comparison includes the online model's routing metadata. It used four 11 x 11
association matrices (484 scalars), four 16-dimensional projected prototypes, and four counts
(68 routing scalars), totaling 552. The static baseline had four 12 x 12 matrices (576) and no
episode-local routing state.

## Interpretation boundary

This rules out the simplest permanent domain-to-block shortcut. It shows that a memory layer can
create an episode-specific permutation, reuse it consistently, and recover the right region at
query time without using the domain ID. However, the architectural prior still explicitly says
that a novel context should claim an empty block; delayed recall learns the novelty threshold, not
the whole allocation algorithm. Context features are intentionally constant within each episode,
and active contexts never exceed block count, so semantic drift and eviction remain untested.

The next experiment must allow five or six simultaneously active contexts with four blocks and
vary future query utility. Success then requires retaining useful contexts while evicting or
weakening lower-impact ones rather than merely filling empty slots.

# Phase 4: utility-aware eviction

Date: 2026-09-12

Each episode contained six semantic contexts competing for four blocks. Four randomly selected
contexts were queried later; usefulness was independent of arrival order. Training and evaluation
used disjoint frozen BGE cue phrasings. The learned model never consumed the retention mask; that
mask was reserved for query construction, metrics, and the oracle branch.

Final command, run with seeds 7, 17, and 29:

```bash
uv run python train_eviction.py \
  --steps 600 --eval-batches 50 --batch-size 64 --seed <seed> --summary
```

| Variant | Runtime state | Recall | Useful-context coverage | Useful blocks at end | Useful / temporary priority |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recency/least-used eviction | 552 | 0.5921 +/- 0.0015 | 0.6682 +/- 0.0015 | 0.6682 +/- 0.0015 | N/A |
| **Learned priority eviction** | **556** | **0.8926 +/- 0.0007** | **1.0000** | **1.0000** | **0.342 / 0.188** |
| Oracle priority eviction | 556 | 0.8927 +/- 0.0009 | 1.0000 | 1.0000 | 1.000 / 0.000 |

Values are mean +/- population standard deviation over three seeds. The learned model accepted
some temporary contexts while empty capacity remained, but later replaced them: temporary-context
acceptance was 0.7103 while final useful-block purity was 1.0000. This distinction matters because
successful forgetting is a state transition, not simply refusing every low-value input.

The priority scorer learned only a relative ordering. Its absolute outputs remained moderate, but
useful contexts ranked above temporary ones reliably enough to match oracle retention and recall.
Association state used four 11 x 11 matrices (484 scalars); four compressed prototypes, counts, and
priorities added 72, for 556 total runtime scalars.

## Interpretation boundary

This is positive evidence for joint write, online allocation, and eviction in a separate
model-state memory. It is still a controlled curriculum: usefulness is encoded by hand-authored
durable-versus-temporary sentences, every context arrives as one contiguous chunk, context
features do not drift within an episode, and only the priority comparison is learned. Real
conversation may provide ambiguous or delayed evidence, revisit evicted contexts, and mix
corrections with preference changes.

The next stress test should remove chunked arrivals, paraphrase the same context over time, and
delay utility evidence until after the first write. That requires an eligibility trace or a
retroactive consolidation update instead of a priority score available at initial allocation.

# Phase 5: delayed utility and eligibility traces

Date: 2026-09-12

All six context streams were fully interleaved and carried no usefulness cue during observation.
Six model-state trace slots temporarily held their associations. After observation ended, one
frozen natural-language outcome cue arrived for each context; only then did the memory consolidate
four traces and suppress two. Learned mode did not consume the retention mask.

Final command, run with seeds 7, 17, and 29:

```bash
uv run python train_delayed.py \
  --steps 600 --eval-batches 50 --batch-size 64 --seed <seed> --summary
```

| Consolidation | Transient state | Recall | Useful-trace coverage | Useful / temporary priority | Active traces after feedback |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recency top-4 | 594 | 0.5784 +/- 0.0018 | 0.6656 +/- 0.0005 | 0.501 / 0.498 | 4.0 |
| **Learned delayed top-4** | **594** | **0.8630 +/- 0.0014** | **1.0000** | **0.971 / 0.909** | **4.0** |
| Oracle delayed top-4 | 594 | 0.8626 +/- 0.0017 | 1.0000 | 1.000 / 0.000 | 4.0 |

Values are mean +/- population standard deviation over three seeds. Learned consolidation matched
the oracle at all seeds. Its priority outputs were not calibrated probabilities: both classes had
high absolute scores, but the useful traces consistently ranked above the temporary traces. The
competitive top-k ordering, not a hand-set threshold, produced perfect retention selection.

The transient state comprises six 9 x 9 association matrices plus six 16-dimensional compressed
context prototypes, counts, and priorities: `6 * (81 + 16 + 2) = 594` scalars. After consolidation,
only four association matrices remain nonzero, although all six prototypes remain available to
route a query to either a retained matrix or a suppressed trace.

## Interpretation boundary

This shows retroactive consolidation in model state: associations can be written provisionally
before their utility is known, then strengthened or forgotten after delayed semantic evidence.
It does not yet solve unconstrained temporal credit assignment. The feedback event explicitly
contains the corresponding context representation, the trace has room for every candidate, and
there is no time decay. A real conversation may reveal utility indirectly through task success,
corrections, or later behavior without restating the original context.

The next experiment should remove the explicit context pointer from feedback. A global outcome
must then update the responsible recent traces through learned eligibility weights, which is the
first condition that genuinely tests temporal credit assignment rather than delayed ranking.

# Phase 6: global feedback on local Claude conversation logs

Date: 2026-09-12

The user authorized local use of the full `~/.claude` conversation history. No raw prompt text is
copied into this repository or persisted in the generated tensor artifact. The global history was
sessionized per project with a six-hour idle boundary. After filtering short/internal turns it
contained 75 projects, 495 conversations, and 16,622 user turns.

A weak delayed-revisit target was emitted when a later user turn uniquely reused specific lexical
anchors from one of eight candidates ending at least two turns earlier. This produced 1,349
examples. A stable project hash reserved entire projects for evaluation: 775 train and 574 eval.
Median target delay was five turns and the 90th percentile was eight turns.

Commands:

```bash
uv run python analyze_claude_logs.py --summary
HF_HUB_OFFLINE=1 uv run python prepare_claude_log_embeddings.py
uv run python train_global_feedback.py --steps 1000 --seed <seed> --summary
```

| Selector | Stored trace width | Top-1 | Top-2 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Random candidate | N/A | 0.1376 | 0.2485 | 0.3471 |
| Recency | N/A | 0.2178 | 0.3885 | 0.4299 |
| Untrained random projection | 64 | 0.7863 +/- 0.0109 | 0.8728 +/- 0.0079 | 0.8616 +/- 0.0073 |
| **Learned shared projection** | **64** | **0.8002 +/- 0.0036** | **0.8722 +/- 0.0030** | **0.8676 +/- 0.0022** |
| Full frozen BGE cosine | 384 | 0.8206 | 0.8990 | 0.8851 |

Values are means and population standard deviations over seeds 7, 17, and 29. The learned model
has 24,577 trainable parameters. It compresses each candidate from 384 to 64 scalars, so eight
retained traces require 512 rather than 3,072 scalars. Training improved top-1 by 1.39 percentage
points over the untrained compressed projection, and the learned trace remained 2.03 points below
full-dimensional cosine.

## Interpretation boundary

This is the first experiment in the series where a single later event must choose among multiple
old traces without receiving the responsible context or trace ID. Its large advantage over recency
shows that relevant past state can be activated reliably from natural, delayed conversation data,
and most of the frozen semantic geometry survives sixfold trace compression.

It is not yet evidence that the layer discovers what is intrinsically worth remembering. The weak
teacher defines relevance using lexical recurrence, full frozen cosine still wins, and training
adds only a modest gain over a random shared projection. The next experiment should label memory
utility using downstream events that are not semantic restatements: corrections, accepted versus
reverted actions, or later task success. That would separate delayed causal credit from ordinary
similarity retrieval.

# Phase 7: natural delayed outcome gating

Date: 2026-09-12

The full global history was re-sessionized while retaining short acknowledgements and corrections.
Across 557 conversations and 23,240 usable turns, high-precision weak rules found 2,940 explicit
outcomes: 1,677 accept/strengthen and 1,263 correct/revise. Stable project-level splitting yielded
1,721 train and 1,219 held-out-project evaluation examples. No raw text is stored in the feature
artifact; a frozen 512-dimensional signed character n-gram hash supports both Korean shorthand and
English feedback without downloading another encoder.

Commands, repeated with seeds 7, 17, and 29:

```bash
uv run python analyze_claude_outcomes.py --summary
uv run python prepare_claude_outcome_features.py
uv run python train_outcome_gate.py --steps 1000 --seed <seed> --summary
```

| Available evidence at decision time | Accuracy | Balanced accuracy | Accept recall | Correction recall |
| --- | ---: | ---: | ---: | ---: |
| Majority accept | 0.5874 | 0.5000 | 1.0000 | 0.0000 |
| Request at initial write only | 0.5562 +/- 0.0158 | 0.5429 +/- 0.0073 | 0.6192 | 0.4665 |
| Later feedback only | 0.9773 +/- 0.0004 | 0.9790 +/- 0.0003 | 0.9693 | 0.9887 |
| **Request trace + later feedback** | **0.9776 +/- 0.0010** | **0.9796 +/- 0.0009** | **0.9679** | **0.9914** |

The joint gate assigned mean retention 0.9652 to accepted traces and 0.0110 to correction traces.
The full training module has 66,310 parameters; its persistent per-interaction state is one
64-dimensional request trace plus one strength scalar. `apply_outcome()` uses the learned retention
as an actual model-state update rather than returning a diagnostic class alone.

## Interpretation boundary

This experiment answers the original storage bottleneck more directly: these conversations do not
contain enough evidence at initial observation to predict future utility reliably. Request-only
balanced accuracy was just 54.3%. Once delayed user feedback arrived, the same small gate made a
stable strengthen/revise decision at 98.0%. The useful architecture is therefore provisional write
followed by retroactive consolidation, not a supposedly omniscient write gate at input time.

The result is still imitation learning from explicit surface forms. The weak-label rules define the
classes, feedback usually belongs to the immediately preceding interaction, and a character hash
can expose those forms easily. A stronger next test must use implicit outcomes—whether a code change
survived, was reverted, passed tests, or was reused later—and jointly train trace selection plus the
strengthen/revise/forget action. That is the point where delayed utility becomes causal rather than
linguistic.

# Phase 8: implicit tool outcomes and file survival

Date: 2026-09-13

## Test and build outcomes

The project-session logs contained 917 high-confidence verification outcomes across 11 projects
and 33 sessions: 496 passes and 421 failures. Of these, 525 followed one or more Edit, Write,
apply-patch, or formatter mutations. Stable project-level splitting produced 835 train and 82 eval
events. Raw commands, code, and tool output were used only to build ignored 512-dimensional hashed
feature artifacts and were not persisted in git.

Five-seed results on all held-out-project events:

| Evidence | Accuracy | Balanced accuracy | Pass / fail strength |
| --- | ---: | ---: | ---: |
| Majority pass | 0.6707 | 0.5000 | 0.731 / 0.731 |
| Action and command before result | 0.6268 | 0.6370 | varies by seed |
| Test result only | 0.9829 | 0.9741 | 0.956 / 0.068 |
| **Action trace + test result** | **0.9927** | **0.9889** | **0.994 / 0.034** |

On the mutation-only subset (471 train / 54 eval), feedback-only balanced accuracy was 0.8761 and
joint accuracy was 0.8708. The action context did not add reliable causal attribution once the
dataset was restricted to real mutations. Test output is therefore a strong update signal, but this
experiment alone cannot identify which one of several changes caused the result.

## File-change survival

Claude file history contained 430 multi-version file chains from 35 sessions and 1,679 versions.
There were only 11 exact `A -> B -> A` restorations. Line-survival scoring expanded this to 672
high-confidence next-version outcomes: 637 retained and 35 reverted, with revert labels spread over
13 sessions. A stratified whole-session split yielded 566 train examples (23 reverted) and 106 eval
examples (12 reverted).

Commands, repeated with seeds 7, 17, 29, 41, and 53:

```bash
uv run python analyze_file_survival.py --summary
uv run python prepare_file_survival_features.py
uv run python train_survival_gate.py --steps 2000 --seed <seed> --summary
```

| Decision path | Balanced accuracy | Revert recall | Revert precision | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Change at observation only | 0.4798 | 0.0000 | 0.0000 | 0.520 |
| Future file delta only | 0.5567 | 0.1667 | 0.2857 | 0.561 |
| Generic concatenation gate | 0.5716 | 0.1667 | about 0.48 | 0.659 |
| Raw 512d cosine threshold | 0.6711 | 0.4167 | 0.4167 | 0.8954 |
| **Learned 64d relational gate** | **0.8340 +/- 0.0021** | **1.0000** | **0.2778** | **0.9021 +/- 0.0012** |
| Relational gate, train-F1 threshold | 0.6090 | 0.2500 | 0.5000 | 0.9021 |

The relational gate has a fixed 512-to-64 projection and only six trainable decision parameters.
Its persistent state is a 64-dimensional change trace plus one strength scalar. Freezing the
projection removed seed collapse. The large gap between request-only, outcome-only, and relational
comparison is positive evidence that the delayed state relation—not a surface outcome token—is the
useful signal.

## Interpretation boundary

This is the strongest phenomenon result so far: an old model-state trace is necessary to interpret
a later external event, and the learned policy ranks all 12 held-out reverts above most retained
changes. It is nevertheless not ready to delete memories. At the high-recall threshold it flags too
many retained changes, while a precision-oriented threshold misses most reverts. Only 35 negative
examples and 13 negative-bearing sessions exist, and next-version line survival is a proxy rather
than end-task utility.

The next investment should collect richer causal episodes prospectively: record each candidate
memory/action ID, test outcome, subsequent fix, final accepted diff, and later reuse. Training can
then assign a global outcome across multiple traces and distinguish revise from permanent forget
without relying on file adjacency heuristics.
