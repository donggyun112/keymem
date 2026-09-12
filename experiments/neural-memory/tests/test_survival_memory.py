import torch

from neural_memory.survival_memory import ChangeSurvivalGate


def test_relational_gate_compares_trace_without_target_pointer() -> None:
    model = ChangeSurvivalGate(feature_dim=32, memory_dim=8)
    changes = torch.randn(6, 32)
    outcomes = torch.randn(6, 32)

    state = model.observe(changes)
    logits = model.act(state, outcomes)
    updated = model.apply_outcome(state, logits)

    assert state.trace.shape == (6, 8)
    assert model.projection.weight.requires_grad is False
    assert logits.shape == (6, 2)
    assert updated.strength.shape == (6,)
    assert torch.all(updated.strength >= 0.0)
    assert torch.all(updated.strength <= 1.0)
