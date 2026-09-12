from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .online import OnlineMemoryState, OnlinePrototypeMemory, OnlineRoutingTelemetry


ConsolidationMode = Literal["recency", "learned", "oracle"]


@dataclass(frozen=True)
class EligibilityState:
    weights: Tensor
    prototypes: Tensor
    counts: Tensor
    priorities: Tensor
    consolidated_mask: Tensor


class EligibilityTraceMemory(OnlinePrototypeMemory):
    """Short-lived association traces consolidated after delayed feedback."""

    def __init__(
        self,
        *args: object,
        consolidation_sharpness: float = 20.0,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        if consolidation_sharpness <= 0.0:
            raise ValueError("consolidation_sharpness must be positive")
        self.consolidation_sharpness = consolidation_sharpness
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
    ) -> EligibilityState:
        base = super().initial_state(batch_size, device=device, dtype=dtype)
        return EligibilityState(
            weights=base.weights,
            prototypes=base.prototypes,
            counts=base.counts,
            priorities=torch.zeros_like(base.counts),
            consolidated_mask=torch.ones_like(base.counts),
        )

    def observe(
        self,
        state: EligibilityState,
        context_features: Tensor,
        key_ids: Tensor,
        value_ids: Tensor,
    ) -> tuple[EligibilityState, OnlineRoutingTelemetry]:
        base = OnlineMemoryState(state.weights, state.prototypes, state.counts)
        next_base, telemetry = super().write(
            base, context_features, key_ids, value_ids
        )
        return EligibilityState(
            weights=next_base.weights,
            prototypes=next_base.prototypes,
            counts=next_base.counts,
            priorities=state.priorities,
            consolidated_mask=state.consolidated_mask,
        ), telemetry

    def feedback(
        self,
        state: EligibilityState,
        context_features: Tensor,
        cue_features: Tensor,
        *,
        mode: ConsolidationMode,
        should_retain: Tensor | None = None,
    ) -> tuple[EligibilityState, Tensor, Tensor]:
        context = self._project_context(context_features)
        prototypes = F.normalize(state.prototypes, dim=-1)
        similarities = torch.einsum("bnh,bh->bn", prototypes, context)
        similarities = similarities.masked_fill(state.counts <= 0, -1e4)
        route = similarities.argmax(dim=-1)

        if mode == "learned":
            priority = torch.sigmoid(self.priority_scorer(cue_features)).squeeze(-1)
        elif mode == "oracle":
            if should_retain is None:
                raise ValueError("oracle feedback requires should_retain")
            priority = should_retain.to(dtype=context.dtype)
        elif mode == "recency":
            priority = route.to(dtype=context.dtype) / max(self.num_blocks - 1, 1)
        else:
            raise ValueError(f"unknown consolidation mode: {mode}")

        route_mask = F.one_hot(route, num_classes=self.num_blocks).to(context.dtype)
        priorities = state.priorities * (1.0 - route_mask)
        priorities = priorities + route_mask * priority[:, None]
        return EligibilityState(
            state.weights,
            state.prototypes,
            state.counts,
            priorities,
            state.consolidated_mask,
        ), route, priority

    def consolidate(
        self,
        state: EligibilityState,
        *,
        keep_slots: int,
    ) -> EligibilityState:
        if not 1 <= keep_slots <= self.num_blocks:
            raise ValueError("keep_slots must be within trace capacity")
        top = state.priorities.topk(keep_slots, dim=-1)
        hard_mask = torch.zeros_like(state.priorities)
        hard_mask.scatter_(1, top.indices, 1.0)
        boundary = top.values[:, -1:].detach()
        soft_mask = torch.sigmoid(
            (state.priorities - boundary) * self.consolidation_sharpness
        )
        mask = hard_mask + soft_mask - soft_mask.detach()
        return EligibilityState(
            weights=state.weights * mask[:, :, None, None],
            prototypes=state.prototypes,
            counts=state.counts,
            priorities=state.priorities,
            consolidated_mask=hard_mask,
        )

    def read(
        self,
        state: EligibilityState,
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
