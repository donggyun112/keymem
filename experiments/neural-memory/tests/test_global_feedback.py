from pathlib import Path

import pytest
import torch

from neural_memory.global_feedback import (
    GlobalFeedbackSelector,
    load_global_feedback_banks,
)


def test_selector_has_no_target_or_pointer_input() -> None:
    model = GlobalFeedbackSelector(text_dim=12, memory_dim=5)
    candidates = torch.randn(4, 8, 12)
    feedback = torch.randn(4, 12)

    state = model.observe(candidates)
    logits = model.feedback(state, feedback)
    mask = model.consolidate(logits, keep=2)

    assert state.traces.shape == (4, 8, 5)
    assert logits.shape == (4, 8)
    assert torch.equal(mask.sum(dim=-1), torch.full((4,), 2.0))


def test_artifact_loader_rejects_missing_tensors(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"train_candidates": torch.randn(2, 3, 4)}, path)
    with pytest.raises(ValueError, match="missing required tensors"):
        load_global_feedback_banks(path)


def test_artifact_loader_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "banks.pt"
    payload: dict[str, torch.Tensor] = {}
    for split, size in (("train", 5), ("eval", 2)):
        payload[f"{split}_candidates"] = torch.randn(size, 8, 12)
        payload[f"{split}_feedback"] = torch.randn(size, 12)
        payload[f"{split}_targets"] = torch.randint(8, (size,))
        payload[f"{split}_delays"] = torch.randint(2, 10, (size,))
    torch.save(payload, path)

    banks = load_global_feedback_banks(path)

    assert banks.text_dim == 12
    assert banks.candidate_count == 8
    assert len(banks.eval_targets) == 2
