from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GlobalFeedbackBanks:
    train_candidates: Tensor
    train_feedback: Tensor
    train_targets: Tensor
    train_delays: Tensor
    eval_candidates: Tensor
    eval_feedback: Tensor
    eval_targets: Tensor
    eval_delays: Tensor

    @property
    def text_dim(self) -> int:
        return self.train_candidates.shape[-1]

    @property
    def candidate_count(self) -> int:
        return self.train_candidates.shape[1]

    def split(self, name: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if name == "train":
            return (
                self.train_candidates,
                self.train_feedback,
                self.train_targets,
                self.train_delays,
            )
        if name == "eval":
            return (
                self.eval_candidates,
                self.eval_feedback,
                self.eval_targets,
                self.eval_delays,
            )
        raise ValueError(f"unknown split: {name}")


def load_global_feedback_banks(path: str | Path) -> GlobalFeedbackBanks:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    names = tuple(GlobalFeedbackBanks.__dataclass_fields__)
    if any(name not in payload for name in names):
        raise ValueError("global-feedback artifact is missing required tensors")
    banks = GlobalFeedbackBanks(**{name: payload[name] for name in names})
    for split in ("train", "eval"):
        candidates, feedback, targets, delays = banks.split(split)
        if candidates.ndim != 3 or feedback.ndim != 2:
            raise ValueError("features must be [examples, candidates, dim] and [examples, dim]")
        if targets.ndim != 1 or delays.ndim != 1:
            raise ValueError("targets and delays must be vectors")
        if not (len(candidates) == len(feedback) == len(targets) == len(delays)):
            raise ValueError("split tensors must contain the same number of examples")
    return banks


@dataclass(frozen=True)
class GlobalTraceState:
    traces: Tensor


class GlobalFeedbackSelector(nn.Module):
    """Select an earlier trace from a later global outcome, with no trace pointer."""

    def __init__(self, text_dim: int, memory_dim: int = 64) -> None:
        super().__init__()
        if text_dim < 1 or memory_dim < 1:
            raise ValueError("dimensions must be positive")
        self.text_dim = text_dim
        self.memory_dim = memory_dim
        # Observation and outcome are the same language modality. Sharing this
        # projection preserves the frozen encoder's geometry while still forcing
        # every retained trace through a compact learned memory representation.
        self.projection = nn.Linear(text_dim, memory_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(2.0))

    def observe(self, candidate_features: Tensor) -> GlobalTraceState:
        if candidate_features.ndim != 3:
            raise ValueError("candidate features must have shape [batch, traces, text_dim]")
        traces = F.normalize(self.projection(candidate_features), dim=-1)
        return GlobalTraceState(traces)

    def feedback(self, state: GlobalTraceState, feedback_features: Tensor) -> Tensor:
        if feedback_features.ndim != 2:
            raise ValueError("feedback features must have shape [batch, text_dim]")
        feedback = F.normalize(self.projection(feedback_features), dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return torch.einsum("bnd,bd->bn", state.traces, feedback) * scale

    def forward(self, candidate_features: Tensor, feedback_features: Tensor) -> Tensor:
        return self.feedback(self.observe(candidate_features), feedback_features)

    @staticmethod
    def consolidate(logits: Tensor, *, keep: int) -> Tensor:
        if not 1 <= keep <= logits.shape[-1]:
            raise ValueError("keep must be within trace count")
        indices = logits.topk(keep, dim=-1).indices
        mask = torch.zeros_like(logits)
        mask.scatter_(1, indices, 1.0)
        return mask
