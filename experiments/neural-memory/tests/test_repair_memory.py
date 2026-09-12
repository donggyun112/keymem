from __future__ import annotations

import pytest
import torch

from neural_memory.repair_memory import DecoupledMultiTraceMemory, MultiTraceActionMemory


def test_joint_credit_outputs_one_action_per_trace() -> None:
    model = MultiTraceActionMemory(feature_dim=12, memory_dim=5)
    candidates = torch.randn(3, 7, 12)
    outcome = torch.randn(3, 12)
    state = model.observe(candidates)
    logits = model.act(state, outcome)
    assert state.traces.shape == (3, 7, 5)
    assert logits.shape == (3, 7, 3)
    assert model.strengths(logits).shape == (3, 7)


def test_candidate_mode_does_not_depend_on_outcome() -> None:
    torch.manual_seed(3)
    model = MultiTraceActionMemory(feature_dim=8, memory_dim=4)
    candidates = torch.randn(2, 3, 8)
    first = model(candidates, torch.randn(2, 8), mode="candidate")
    second = model(candidates, torch.randn(2, 8), mode="candidate")
    assert torch.equal(first, second)


def test_invalid_shapes_and_modes_are_rejected() -> None:
    model = MultiTraceActionMemory(feature_dim=8, memory_dim=4)
    with pytest.raises(ValueError):
        model.observe(torch.randn(2, 8))
    state = model.observe(torch.randn(2, 3, 8))
    with pytest.raises(ValueError):
        model.act(state, torch.randn(2, 3, 8))
    with pytest.raises(ValueError):
        model.act(state, torch.randn(2, 8), mode="invalid")  # type: ignore[arg-type]


def test_decoupled_memory_composes_valid_three_class_logits() -> None:
    model = DecoupledMultiTraceMemory(feature_dim=8, memory_dim=4)
    selection, action = model(torch.randn(2, 3, 8), torch.randn(2, 8))
    logits = model.class_logits(selection, action)
    assert selection.shape == (2, 3)
    assert action.shape == (2, 3, 2)
    assert logits.shape == (2, 3, 3)
    assert torch.allclose(logits.exp().sum(dim=-1), torch.ones(2, 3))
