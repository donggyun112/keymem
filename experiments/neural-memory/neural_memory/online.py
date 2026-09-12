from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class OnlineMemoryState:
    weights: Tensor
    prototypes: Tensor
    counts: Tensor


@dataclass(frozen=True)
class OnlineRoutingTelemetry:
    allocation: Tensor
    route: Tensor
    matched_existing: Tensor
    threshold: Tensor


class OnlinePrototypeMemory(nn.Module):
    """Fast-weight memory that allocates context prototypes at runtime.

    Block identity is episode-local. A novel context claims an empty block;
    later observations recover it by similarity to the fast prototype state.
    The straight-through allocation lets delayed recall loss tune the novelty
    threshold without supplying context IDs to the model.
    """

    def __init__(
        self,
        *,
        text_dim: int,
        num_keys: int,
        num_values: int,
        num_blocks: int,
        d_model: int,
        prototype_dim: int | None = None,
        projection_seed: int = 0,
        initial_threshold: float = 0.9,
        match_temperature: float = 0.05,
        match_sharpness: float = 20.0,
        retention: float = 1.0,
    ) -> None:
        super().__init__()
        if min(text_dim, num_keys, num_values, num_blocks, d_model) < 1:
            raise ValueError("model dimensions must be positive")
        if not 0.0 < initial_threshold < 1.0:
            raise ValueError("initial_threshold must be in (0, 1)")
        if match_temperature <= 0.0 or match_sharpness <= 0.0:
            raise ValueError("matching scale parameters must be positive")
        if not 0.0 <= retention <= 1.0:
            raise ValueError("retention must be in [0, 1]")

        resolved_prototype_dim = prototype_dim or text_dim
        if resolved_prototype_dim < 1:
            raise ValueError("prototype_dim must be positive")
        self.text_dim = text_dim
        self.prototype_dim = resolved_prototype_dim
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.match_temperature = match_temperature
        self.match_sharpness = match_sharpness
        self.retention = retention

        self.key_embedding = nn.Embedding(num_keys, d_model)
        self.value_embedding = nn.Embedding(num_values, d_model)
        if resolved_prototype_dim == text_dim:
            projection = torch.eye(text_dim)
        else:
            generator = torch.Generator().manual_seed(projection_seed)
            projection = torch.randn(
                text_dim, resolved_prototype_dim, generator=generator
            )
            projection = F.normalize(projection, dim=0)
        self.register_buffer("context_projection", projection)
        threshold_logit = math.log(initial_threshold / (1.0 - initial_threshold))
        self.threshold_logit = nn.Parameter(torch.tensor(threshold_logit))
        self.logit_scale = nn.Parameter(torch.tensor(2.0))

    @property
    def fast_capacity(self) -> int:
        return self.num_blocks * self.d_model * self.d_model

    @property
    def routing_state_size(self) -> int:
        return self.num_blocks * (self.prototype_dim + 1)

    @property
    def threshold(self) -> Tensor:
        return torch.sigmoid(self.threshold_logit)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> OnlineMemoryState:
        parameter = self.key_embedding.weight
        resolved_device = device or parameter.device
        resolved_dtype = dtype or parameter.dtype
        return OnlineMemoryState(
            weights=torch.zeros(
                batch_size,
                self.num_blocks,
                self.d_model,
                self.d_model,
                device=resolved_device,
                dtype=resolved_dtype,
            ),
            prototypes=torch.zeros(
                batch_size,
                self.num_blocks,
                self.prototype_dim,
                device=resolved_device,
                dtype=resolved_dtype,
            ),
            counts=torch.zeros(
                batch_size,
                self.num_blocks,
                device=resolved_device,
                dtype=resolved_dtype,
            ),
        )

    def _allocate(
        self,
        state: OnlineMemoryState,
        context_features: Tensor,
    ) -> OnlineRoutingTelemetry:
        context = self._project_context(context_features)
        prototypes = F.normalize(state.prototypes, dim=-1)
        similarities = torch.einsum("bnh,bh->bn", prototypes, context)
        occupied = state.counts > 0
        masked = similarities.masked_fill(~occupied, -1e4)
        best_score, best_route = masked.max(dim=-1)
        has_occupied = occupied.any(dim=-1)
        has_empty = (~occupied).any(dim=-1)
        empty_route = (~occupied).float().argmax(dim=-1)
        least_used_route = state.counts.argmin(dim=-1)
        novel_route = torch.where(has_empty, empty_route, least_used_route)

        threshold = self.threshold
        matched = has_occupied & (best_score >= threshold)
        route = torch.where(matched, best_route, novel_route)
        hard = F.one_hot(route, num_classes=self.num_blocks).to(context.dtype)

        existing_soft = F.softmax(masked / self.match_temperature, dim=-1)
        match_probability = torch.sigmoid(
            (best_score - threshold) * self.match_sharpness
        )
        match_probability = match_probability * has_occupied.to(context.dtype)
        novel_hard = F.one_hot(novel_route, num_classes=self.num_blocks).to(context.dtype)
        soft = (
            match_probability[:, None] * existing_soft
            + (1.0 - match_probability[:, None]) * novel_hard
        )
        allocation = hard + soft - soft.detach()
        return OnlineRoutingTelemetry(
            allocation=allocation,
            route=route,
            matched_existing=matched,
            threshold=threshold,
        )

    def _project_context(self, context_features: Tensor) -> Tensor:
        if context_features.ndim != 2 or context_features.shape[-1] != self.text_dim:
            raise ValueError("context_features must have shape [batch, text_dim]")
        projected = context_features @ self.context_projection
        return F.normalize(projected, dim=-1)

    def write(
        self,
        state: OnlineMemoryState,
        context_features: Tensor,
        key_ids: Tensor,
        value_ids: Tensor,
    ) -> tuple[OnlineMemoryState, OnlineRoutingTelemetry]:
        telemetry = self._allocate(state, context_features)
        hard = F.one_hot(telemetry.route, num_classes=self.num_blocks).to(
            state.weights.dtype
        )
        novel = ~telemetry.matched_existing

        # Reusing a non-empty block for a novel context is an eviction. Clear its
        # association matrix and prototype before writing the replacement.
        reset = novel[:, None] * hard.bool()
        base_weights = state.weights * (~reset)[:, :, None, None]
        key = F.normalize(self.key_embedding(key_ids), dim=-1)
        value = F.normalize(self.value_embedding(value_ids), dim=-1)
        predicted = torch.einsum("bnij,bj->bni", base_weights, key)
        residual = value[:, None, :] - predicted
        delta = residual.unsqueeze(-1) * key[:, None, None, :]
        denominator = key.square().sum(dim=-1).clamp_min(1e-6)
        delta = delta / denominator[:, None, None, None]
        next_weights = self.retention * base_weights
        next_weights = next_weights + telemetry.allocation[:, :, None, None] * delta

        selected_count = (state.counts * hard).sum(dim=-1)
        selected_count = torch.where(novel, torch.zeros_like(selected_count), selected_count)
        selected_prototype = (state.prototypes * hard[:, :, None]).sum(dim=1)
        selected_prototype = torch.where(
            novel[:, None], torch.zeros_like(selected_prototype), selected_prototype
        )
        next_selected_count = selected_count + 1.0
        context = self._project_context(context_features)
        next_selected_prototype = (
            selected_prototype * selected_count[:, None] + context
        ) / next_selected_count[:, None]
        next_selected_prototype = F.normalize(next_selected_prototype, dim=-1)
        next_counts = state.counts * (1.0 - hard) + hard * next_selected_count[:, None]
        next_prototypes = state.prototypes * (1.0 - hard[:, :, None])
        next_prototypes = next_prototypes + hard[:, :, None] * next_selected_prototype[:, None, :]

        return OnlineMemoryState(next_weights, next_prototypes, next_counts), telemetry

    def read(
        self,
        state: OnlineMemoryState,
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
