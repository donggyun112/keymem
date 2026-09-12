from pathlib import Path

import pytest
import torch

from neural_memory.outcome_memory import DelayedOutcomeGate, load_outcome_banks


def test_gate_outputs_actions_without_label_input() -> None:
    model = DelayedOutcomeGate(feature_dim=24, memory_dim=8)
    requests = torch.randn(5, 24)
    feedback = torch.randn(5, 24)
    state = model.observe(requests)

    for mode in ("request", "feedback", "joint"):
        logits = model.act(state, feedback, mode=mode)
        updated = model.apply_outcome(state, logits)
        assert logits.shape == (5, 2)
        assert model.retention(logits).shape == (5,)
        assert updated.strength.shape == (5,)
        assert torch.all(updated.strength >= 0.0)
        assert torch.all(updated.strength <= 1.0)


def test_outcome_artifact_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "outcomes.pt"
    payload: dict[str, torch.Tensor] = {}
    for split, size in (("train", 5), ("eval", 3)):
        payload[f"{split}_requests"] = torch.randn(size, 16)
        payload[f"{split}_feedback"] = torch.randn(size, 16)
        payload[f"{split}_targets"] = torch.randint(2, (size,))
    torch.save(payload, path)

    banks = load_outcome_banks(path)

    assert banks.feature_dim == 16
    assert len(banks.eval_targets) == 3


def test_outcome_artifact_rejects_missing_data(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({}, path)
    with pytest.raises(ValueError, match="missing required tensors"):
        load_outcome_banks(path)
