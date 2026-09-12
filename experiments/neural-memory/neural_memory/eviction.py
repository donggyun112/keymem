from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .online import OnlinePrototypeMemory


PriorityMode = Literal["learned", "oracle"]


@dataclass(frozen=True)
class PriorityMemoryState:
    weights: Tensor
    prototypes: Tensor
    counts: Tensor
    priorities: Tensor


@dataclass(frozen=True)
class PriorityTelemetry:
    route: Tensor
    accepted: Tensor
    matched_existing: Tensor
    incoming_priority: Tensor


class PriorityEvictionMemory(OnlinePrototypeMemory):
    """Online memory that learns which context deserves a limited block."""

    def __init__(self, *args: object, eviction_sharpness: float = 20.0, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        if eviction_sharpness <= 0.0:
            raise ValueError("eviction_sharpness must be positive")
        self.eviction_sharpness = eviction_sharpness
        self.priority_scorer = nn.Linear(self.text_dim, 1)

    @property
    def routing_state_size(self) -> int:
        return self.num_blocks * (self.prototype_dim + 2)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> PriorityMemoryState:
        base = super().initial_state(
            batch_size, device=device, dtype=dtype
        )
        return PriorityMemoryState(
            weights=base.weights,
            prototypes=base.prototypes,
            counts=base.counts,
            priorities=torch.zeros_like(base.counts),
        )

    def write(
        self,
        state: PriorityMemoryState,
        context_features: Tensor,
        cue_features: Tensor,
        key_ids: Tensor,
        value_ids: Tensor,
        *,
        priority_mode: PriorityMode,
        should_retain: Tensor | None = None,
    ) -> tuple[PriorityMemoryState, PriorityTelemetry]:
        context = self._project_context(context_features)
        prototypes = F.normalize(state.prototypes, dim=-1)
        similarities = torch.einsum("bnh,bh->bn", prototypes, context)
        occupied = state.counts > 0
        masked_similarity = similarities.masked_fill(~occupied, -1e4)
        best_score, best_route = masked_similarity.max(dim=-1)
        has_occupied = occupied.any(dim=-1)
        matched = has_occupied & (best_score >= self.threshold)

        has_empty = (~occupied).any(dim=-1)
        empty_route = (~occupied).float().argmax(dim=-1)
        masked_priority = state.priorities.masked_fill(~occupied, float("inf"))
        lowest_priority, lowest_route = masked_priority.min(dim=-1)
        candidate_route = torch.where(has_empty, empty_route, lowest_route)
        candidate_priority = torch.where(
            has_empty, torch.zeros_like(lowest_priority), lowest_priority
        )

        if priority_mode == "learned":
            incoming_priority = torch.sigmoid(
                self.priority_scorer(cue_features)
            ).squeeze(-1)
        elif priority_mode == "oracle":
            if should_retain is None:
                raise ValueError("oracle priority requires should_retain")
            incoming_priority = should_retain.to(dtype=context.dtype)
        else:
            raise ValueError(f"unknown priority mode: {priority_mode}")

        novel_accept = has_empty | (incoming_priority > candidate_priority)
        accepted = matched | novel_accept
        route = torch.where(matched, best_route, candidate_route)
        hard_route = F.one_hot(route, num_classes=self.num_blocks).to(context.dtype)

        comparison_probability = torch.sigmoid(
            (incoming_priority - candidate_priority) * self.eviction_sharpness
        )
        novel_probability = torch.where(
            has_empty, torch.ones_like(comparison_probability), comparison_probability
        )
        accept_probability = torch.where(
            matched, torch.ones_like(novel_probability), novel_probability
        )
        hard_accept = accepted.to(context.dtype)
        accept_strength = hard_accept + accept_probability - accept_probability.detach()

        accepted_novel = accepted & ~matched
        reset = accepted_novel[:, None] & hard_route.bool()
        base_weights = state.weights * (~reset)[:, :, None, None]
        key = F.normalize(self.key_embedding(key_ids), dim=-1)
        value = F.normalize(self.value_embedding(value_ids), dim=-1)
        predicted = torch.einsum("bnij,bj->bni", base_weights, key)
        residual = value[:, None, :] - predicted
        delta = residual.unsqueeze(-1) * key[:, None, None, :]
        denominator = key.square().sum(dim=-1).clamp_min(1e-6)
        delta = delta / denominator[:, None, None, None]
        allocation = hard_route * accept_strength[:, None]
        next_weights = self.retention * base_weights
        next_weights = next_weights + allocation[:, :, None, None] * delta

        update_route = hard_route * hard_accept[:, None]
        selected_count = (state.counts * hard_route).sum(dim=-1)
        selected_count = torch.where(
            accepted_novel, torch.zeros_like(selected_count), selected_count
        )
        selected_prototype = (state.prototypes * hard_route[:, :, None]).sum(dim=1)
        selected_prototype = torch.where(
            accepted_novel[:, None],
            torch.zeros_like(selected_prototype),
            selected_prototype,
        )
        next_selected_count = selected_count + 1.0
        next_selected_prototype = (
            selected_prototype * selected_count[:, None] + context
        ) / next_selected_count[:, None]
        next_selected_prototype = F.normalize(next_selected_prototype, dim=-1)

        next_counts = state.counts * (1.0 - update_route)
        next_counts = next_counts + update_route * next_selected_count[:, None]
        next_prototypes = state.prototypes * (1.0 - update_route[:, :, None])
        next_prototypes = (
            next_prototypes
            + update_route[:, :, None] * next_selected_prototype[:, None, :]
        )
        next_priorities = state.priorities * (1.0 - update_route)
        next_priorities = next_priorities + update_route * incoming_priority[:, None]

        return PriorityMemoryState(
            next_weights, next_prototypes, next_counts, next_priorities
        ), PriorityTelemetry(
            route=route,
            accepted=accepted,
            matched_existing=matched,
            incoming_priority=incoming_priority,
        )

    def read(
        self,
        state: PriorityMemoryState,
        context_features: Tensor,
        key_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        context = self._project_context(context_features)
        prototypes = F.normalize(state.prototypes, dim=-1)
        similarities = torch.einsum("bnh,bh->bn", prototypes, context)
        similarities = similarities.masked_fill(state.counts <= 0, -1e4)
        route = similarities.argmax(dim=-1)
        allocation = F.one_hot(route, num_classes=self.num_blocks).to(context.dtype)
        key = F.normalize(self.key_embedding(key_ids), dim=-1)
        block_reads = torch.einsum("bnij,bj->bni", state.weights, key)
        recalled = (allocation[:, :, None] * block_reads).sum(dim=1)
        value_bank = F.normalize(self.value_embedding.weight, dim=-1)
        logits = recalled @ value_bank.T * self.logit_scale.exp().clamp(max=100.0)
        return logits, route
