from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


GateMode = Literal["request", "feedback", "joint"]


@dataclass(frozen=True)
class OutcomeBanks:
    train_requests: Tensor
    train_feedback: Tensor
    train_targets: Tensor
    eval_requests: Tensor
    eval_feedback: Tensor
    eval_targets: Tensor

    @property
    def feature_dim(self) -> int:
        return self.train_requests.shape[-1]

    def split(self, name: str) -> tuple[Tensor, Tensor, Tensor]:
        if name == "train":
            return self.train_requests, self.train_feedback, self.train_targets
        if name == "eval":
            return self.eval_requests, self.eval_feedback, self.eval_targets
        raise ValueError(f"unknown split: {name}")


def load_outcome_banks(path: str | Path) -> OutcomeBanks:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    names = tuple(OutcomeBanks.__dataclass_fields__)
    if any(name not in payload for name in names):
        raise ValueError("outcome artifact is missing required tensors")
    banks = OutcomeBanks(**{name: payload[name] for name in names})
    for split in ("train", "eval"):
        requests, feedback, targets = banks.split(split)
        if requests.ndim != 2 or feedback.ndim != 2 or targets.ndim != 1:
            raise ValueError("outcome tensors must be [examples, features] plus target vector")
        if not (len(requests) == len(feedback) == len(targets)):
            raise ValueError("outcome split tensors must have equal lengths")
    return banks


@dataclass(frozen=True)
class OutcomeTraceState:
    trace: Tensor
    strength: Tensor


class DelayedOutcomeGate(nn.Module):
    """Turn a provisional request trace into strengthen/revise after later feedback."""

    def __init__(self, feature_dim: int, memory_dim: int = 64) -> None:
        super().__init__()
        self.request_projection = nn.Linear(feature_dim, memory_dim)
        self.feedback_projection = nn.Linear(feature_dim, memory_dim)
        self.request_head = nn.Linear(memory_dim, 2)
        self.feedback_head = nn.Linear(memory_dim, 2)
        self.joint_head = nn.Linear(memory_dim * 3, 2)
        nn.init.zeros_(self.joint_head.weight)
        nn.init.zeros_(self.joint_head.bias)

    def observe(self, request_features: Tensor) -> OutcomeTraceState:
        trace = torch.tanh(self.request_projection(request_features))
        return OutcomeTraceState(trace, torch.ones(trace.shape[0], device=trace.device))

    def act(
        self,
        state: OutcomeTraceState,
        feedback_features: Tensor,
        *,
        mode: GateMode,
    ) -> Tensor:
        feedback = torch.tanh(self.feedback_projection(feedback_features))
        if mode == "request":
            return self.request_head(state.trace)
        if mode == "feedback":
            return self.feedback_head(feedback)
        if mode == "joint":
            combined = torch.cat((state.trace, feedback, state.trace * feedback), dim=-1)
            return self.feedback_head(feedback) + self.joint_head(combined)
        raise ValueError(f"unknown gate mode: {mode}")

    def forward(
        self,
        request_features: Tensor,
        feedback_features: Tensor,
        *,
        mode: GateMode,
    ) -> Tensor:
        return self.act(self.observe(request_features), feedback_features, mode=mode)

    @staticmethod
    def retention(logits: Tensor) -> Tensor:
        """Probability of action 0 (accept/strengthen); correction becomes weakening."""

        return F.softmax(logits, dim=-1)[:, 0]

    def apply_outcome(self, state: OutcomeTraceState, logits: Tensor) -> OutcomeTraceState:
        """Strengthen accepted traces and weaken traces that require revision."""

        strength = state.strength * self.retention(logits)
        return OutcomeTraceState(state.trace, strength)
