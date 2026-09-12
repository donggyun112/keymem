from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


CreditMode = Literal["candidate", "outcome", "joint"]


@dataclass(frozen=True)
class RepairBanks:
    train_candidates: Tensor
    train_outcomes: Tensor
    train_targets: Tensor
    train_masks: Tensor
    eval_candidates: Tensor
    eval_outcomes: Tensor
    eval_targets: Tensor
    eval_masks: Tensor

    @property
    def feature_dim(self) -> int:
        return self.train_candidates.shape[-1]

    def split(self, name: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if name == "train":
            return (
                self.train_candidates,
                self.train_outcomes,
                self.train_targets,
                self.train_masks,
            )
        if name == "eval":
            return (
                self.eval_candidates,
                self.eval_outcomes,
                self.eval_targets,
                self.eval_masks,
            )
        raise ValueError(f"unknown split: {name}")


def load_repair_banks(path: str | Path) -> RepairBanks:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    names = tuple(RepairBanks.__dataclass_fields__)
    if any(name not in payload for name in names):
        raise ValueError("repair artifact is missing required tensors")
    banks = RepairBanks(**{name: payload[name] for name in names})
    for split in ("train", "eval"):
        candidates, outcomes, targets, masks = banks.split(split)
        if candidates.ndim != 3 or outcomes.ndim != 2:
            raise ValueError("features must be [episodes, candidates, dim] and [episodes, dim]")
        if targets.shape != masks.shape or targets.shape != candidates.shape[:2]:
            raise ValueError("targets and masks must match candidate slots")
        if len(candidates) != len(outcomes):
            raise ValueError("split tensors must have equal episode counts")
    return banks


@dataclass(frozen=True)
class RepairTraceState:
    traces: Tensor


class MultiTraceActionMemory(nn.Module):
    """Keep candidate actions as compact traces, then update each from one global outcome."""

    def __init__(self, feature_dim: int, memory_dim: int = 64) -> None:
        super().__init__()
        if feature_dim < 1 or memory_dim < 1:
            raise ValueError("dimensions must be positive")
        self.feature_dim = feature_dim
        self.memory_dim = memory_dim
        self.candidate_projection = nn.Linear(feature_dim, memory_dim)
        self.outcome_projection = nn.Linear(feature_dim, memory_dim)
        self.candidate_head = nn.Linear(memory_dim, 3)
        self.outcome_head = nn.Linear(memory_dim, 3)
        self.relation_head = nn.Sequential(
            nn.Linear(memory_dim * 4, memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, 3),
        )

    def observe(self, candidate_features: Tensor) -> RepairTraceState:
        if candidate_features.ndim != 3:
            raise ValueError("candidates must have shape [batch, traces, feature_dim]")
        return RepairTraceState(torch.tanh(self.candidate_projection(candidate_features)))

    def act(
        self,
        state: RepairTraceState,
        outcome_features: Tensor,
        *,
        mode: CreditMode = "joint",
    ) -> Tensor:
        if outcome_features.ndim != 2:
            raise ValueError("outcomes must have shape [batch, feature_dim]")
        outcome = torch.tanh(self.outcome_projection(outcome_features))
        expanded = outcome[:, None, :].expand_as(state.traces)
        if mode == "candidate":
            return self.candidate_head(state.traces)
        if mode == "outcome":
            return self.outcome_head(expanded)
        if mode != "joint":
            raise ValueError(f"unknown mode: {mode}")
        relation = torch.cat(
            (
                state.traces,
                expanded,
                state.traces * expanded,
                (state.traces - expanded).abs(),
            ),
            dim=-1,
        )
        return (
            self.candidate_head(state.traces)
            + self.outcome_head(expanded)
            + self.relation_head(relation)
        )

    def forward(
        self,
        candidate_features: Tensor,
        outcome_features: Tensor,
        *,
        mode: CreditMode = "joint",
    ) -> Tensor:
        return self.act(self.observe(candidate_features), outcome_features, mode=mode)

    @staticmethod
    def strengths(logits: Tensor) -> Tensor:
        probabilities = F.softmax(logits, dim=-1)
        return probabilities[..., 1] - probabilities[..., 2]


class DecoupledMultiTraceMemory(nn.Module):
    """Decide trace activation separately from strengthen-versus-revise."""

    def __init__(self, feature_dim: int, memory_dim: int = 64) -> None:
        super().__init__()
        if feature_dim < 1 or memory_dim < 1:
            raise ValueError("dimensions must be positive")
        self.feature_dim = feature_dim
        self.memory_dim = memory_dim
        self.candidate_projection = nn.Linear(feature_dim, memory_dim)
        self.outcome_projection = nn.Linear(feature_dim, memory_dim)
        self.candidate_selector = nn.Linear(memory_dim, 1)
        self.outcome_selector = nn.Linear(memory_dim, 1)
        self.candidate_action = nn.Linear(memory_dim, 2)
        self.outcome_action = nn.Linear(memory_dim, 2)
        self.relation = nn.Sequential(
            nn.Linear(memory_dim * 4, memory_dim),
            nn.GELU(),
        )
        self.relation_selector = nn.Linear(memory_dim, 1)
        self.relation_action = nn.Linear(memory_dim, 2)

    def observe(self, candidate_features: Tensor) -> RepairTraceState:
        if candidate_features.ndim != 3:
            raise ValueError("candidates must have shape [batch, traces, feature_dim]")
        return RepairTraceState(torch.tanh(self.candidate_projection(candidate_features)))

    def act(
        self,
        state: RepairTraceState,
        outcome_features: Tensor,
        *,
        mode: CreditMode = "joint",
    ) -> tuple[Tensor, Tensor]:
        if outcome_features.ndim != 2:
            raise ValueError("outcomes must have shape [batch, feature_dim]")
        outcome = torch.tanh(self.outcome_projection(outcome_features))
        expanded = outcome[:, None, :].expand_as(state.traces)
        if mode == "candidate":
            return (
                self.candidate_selector(state.traces).squeeze(-1),
                self.candidate_action(state.traces),
            )
        if mode == "outcome":
            return (
                self.outcome_selector(expanded).squeeze(-1),
                self.outcome_action(expanded),
            )
        if mode != "joint":
            raise ValueError(f"unknown mode: {mode}")
        relation = self.relation(
            torch.cat(
                (
                    state.traces,
                    expanded,
                    state.traces * expanded,
                    (state.traces - expanded).abs(),
                ),
                dim=-1,
            )
        )
        selection = (
            self.candidate_selector(state.traces)
            + self.outcome_selector(expanded)
            + self.relation_selector(relation)
        ).squeeze(-1)
        action = (
            self.candidate_action(state.traces)
            + self.outcome_action(expanded)
            + self.relation_action(relation)
        )
        return selection, action

    def forward(
        self,
        candidate_features: Tensor,
        outcome_features: Tensor,
        *,
        mode: CreditMode = "joint",
    ) -> tuple[Tensor, Tensor]:
        return self.act(self.observe(candidate_features), outcome_features, mode=mode)

    @staticmethod
    def class_logits(selection: Tensor, action: Tensor) -> Tensor:
        log_selected = F.logsigmoid(selection)
        log_ignored = F.logsigmoid(-selection)
        selected_actions = log_selected[..., None] + F.log_softmax(action, dim=-1)
        return torch.cat((log_ignored[..., None], selected_actions), dim=-1)
