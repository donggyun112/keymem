from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SurvivalTraceState:
    trace: Tensor
    strength: Tensor


class ChangeSurvivalGate(nn.Module):
    """Compare a compact code-change trace with a later file delta."""

    def __init__(self, feature_dim: int, memory_dim: int = 64) -> None:
        super().__init__()
        if feature_dim < 1 or memory_dim < 1:
            raise ValueError("dimensions must be positive")
        self.projection = nn.Linear(feature_dim, memory_dim, bias=False)
        generator = torch.Generator().manual_seed(0)
        with torch.no_grad():
            self.projection.weight.copy_(
                torch.randn(memory_dim, feature_dim, generator=generator)
                / math.sqrt(memory_dim)
            )
        self.projection.weight.requires_grad_(False)
        self.relation_head = nn.Linear(2, 2)

    def observe(self, change_features: Tensor) -> SurvivalTraceState:
        trace = F.normalize(self.projection(change_features), dim=-1)
        strength = torch.ones(trace.shape[0], device=trace.device)
        return SurvivalTraceState(trace, strength)

    def act(self, state: SurvivalTraceState, outcome_features: Tensor) -> Tensor:
        outcome = F.normalize(self.projection(outcome_features), dim=-1)
        cosine = (state.trace * outcome).sum(dim=-1)
        mean_distance = (state.trace - outcome).abs().mean(dim=-1)
        relations = torch.stack((cosine, mean_distance), dim=-1)
        return self.relation_head(relations)

    def forward(self, change_features: Tensor, outcome_features: Tensor) -> Tensor:
        return self.act(self.observe(change_features), outcome_features)

    def apply_outcome(
        self, state: SurvivalTraceState, logits: Tensor
    ) -> SurvivalTraceState:
        strength = state.strength * F.softmax(logits, dim=-1)[:, 0]
        return SurvivalTraceState(state.trace, strength)
