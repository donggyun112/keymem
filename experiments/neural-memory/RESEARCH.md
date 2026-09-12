# Research reuse map

This experiment deliberately combines established mechanisms instead of proposing a new
architecture before the basic phenomenon is measured.

| Prior work | Mechanism reused here | Deferred from phase 0 |
| --- | --- | --- |
| [Learning to (Learn at Test Time)](https://arxiv.org/abs/2407.04620) | A model-valued hidden state updated online; slow parameters learn the update behavior | Full TTT language-model stack and optimized mini-batch scan |
| [Titans](https://arxiv.org/abs/2501.00663) | Prediction residual as surprise; explicit retention/decay | Momentum, deep memory MLP, and integration into a transformer backbone |
| [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174) | Residual delta-rule update, which can replace an existing key/value association | Kernelized multi-head fast weights |
| [MIRAS](https://arxiv.org/abs/2504.13173) | Separate the choices of memory architecture, attentional objective, retention, and optimizer | Alternative objectives and retention rules |
| [Fast Weight Memory](https://arxiv.org/abs/2011.07831) | Evaluate compositional associative behavior as a learned recurrent state | Its LSTM controller and task-specific reasoning stack |
| [Synaptic plasticity-dependent competition rule influences memory formation](https://doi.org/10.1038/s41467-021-24269-4) | Treat memory allocation as competition among plastic parameter groups, not only a global write/no-write decision | Any claim that mouse fear-memory allocation is a general model of human memory |

The official [TTT PyTorch implementation](https://github.com/test-time-training/ttt-lm-pytorch)
is a useful equation and tensor-shape reference, but its authors describe it as a slow tutorial
implementation and do not recommend it for training. Copying the full language-model stack would
therefore increase the first experiment's cost without testing the write-selection hypothesis.

## Frozen design choices for phase 0

- **Memory representation:** one matrix-valued fast-weight model state per episode.
- **Write rule:** normalized residual delta update.
- **Attentional objective:** later key-to-value recall cross-entropy.
- **Retention:** fixed multiplicative decay.
- **Write selection:** learned scalar gate conditioned on the observation, noisy present-time
  evidence, and prediction surprise.
- **Outer supervision:** future recall only. `retained` is used for episode construction and
  evaluation, never as a direct write-gate label.
- **No graph or external store:** the changing parameter tensor is the memory.

These choices isolate one question: can delayed future utility teach a model when to change its own
fast weights? They do not yet establish semantic text memory.

## Decision gates

1. **Mechanism gate:** learned write gating must beat an identically trained always-write model on
   held-out random bindings, while assigning higher mean gates to useful events.
2. **Dynamics gate:** add correction, delayed decay, and multi-timescale consolidation only after
   gate 1 passes.
3. **Language gate:** replace symbolic IDs with frozen text encoder representations and add a small
   decoder only after the state-update mechanism passes.
4. **Agent gate:** connect the memory output to a frozen LLM and measure spontaneous recall without
   memory instructions only after the standalone model passes.

Moving in this order prevents a capable language decoder from hiding a failed memory mechanism.

## Biological allocation hypothesis for phase 1

Jeong et al. manipulated potentiation and depression at auditory inputs to the lateral amygdala in
mice. Potentiated inputs were preferentially recruited into the fear-memory engram; post-training
depression could reverse that allocation. The total recall-active population remained approximately
constant, which the authors interpret as evidence for competitive allocation under limited
plasticity rather than independent binary selection by every neuron.

This suggests a specific computational ablation, not a biological equivalence claim:

```text
single global write gate              competitive block allocation
g = sigmoid(controller(x))     vs.    a = sparse competition(score_1 ... score_n)
W <- W + g * delta                    W_i <- W_i + a_i * delta_i
```

The competition should have a fixed plasticity budget (`sum(a) = 1` or top-k allocation). A recent
event can still be reassigned during a short eligibility window: strengthening one block competes
with or depresses neighboring blocks. The testable question is whether this improves capacity,
correction, and memory linking over the scalar-gate phase-0 model at an equal parameter and update
budget.

Do not use this study as evidence that the same rule governs all human memory. It examined a
specific fear-conditioning circuit in mice with artificial optogenetic LTP/LTD.
