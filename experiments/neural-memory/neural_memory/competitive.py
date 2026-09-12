from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


AllocationMode = Literal["learned", "uniform", "oracle"]


@dataclass(frozen=True)
class CompetitiveMemoryState:
    """Per-episode block fast weights; these tensors are the stored memory."""

    weights: Tensor

    def detached(self) -> "CompetitiveMemoryState":
        return CompetitiveMemoryState(self.weights.detach())


@dataclass(frozen=True)
class AllocationTelemetry:
    allocation: Tensor


class CompetitiveFastWeightMemory(nn.Module):
    """Fast-weight memory with a learned, budget-constrained block allocator.

    The router's slow weights learn *how* to allocate. The block matrices are
    recreated per episode and store the actual episode-specific associations.
    Every allocation vector sums to one, so all variants receive the same total
    fast-weight update budget at each write.
    """

    def __init__(
        self,
        *,
        num_contexts: int,
        num_keys: int,
        num_values: int,
        num_blocks: int,
        d_model: int,
        router_dim: int = 16,
        temperature: float = 0.5,
        retention: float = 1.0,
    ) -> None:
        super().__init__()
        if min(num_contexts, num_keys, num_values, num_blocks, d_model) < 1:
            raise ValueError("model dimensions must be positive")
        if router_dim < 2:
            raise ValueError("router_dim must be at least 2")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= retention <= 1.0:
            raise ValueError("retention must be in [0, 1]")

        self.num_contexts = num_contexts
        self.num_keys = num_keys
        self.num_values = num_values
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.temperature = temperature
        self.retention = retention

        self.key_embedding = nn.Embedding(num_keys, d_model)
        self.value_embedding = nn.Embedding(num_values, d_model)
        self.context_embedding = nn.Embedding(num_contexts, router_dim)
        self.router = nn.Sequential(
            nn.Linear(router_dim, router_dim),
            nn.SiLU(),
            nn.Linear(router_dim, num_blocks),
        )
        self.logit_scale = nn.Parameter(torch.tensor(2.0))

    @property
    def fast_capacity(self) -> int:
        return self.num_blocks * self.d_model * self.d_model

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> CompetitiveMemoryState:
        parameter = self.key_embedding.weight
        return CompetitiveMemoryState(
            torch.zeros(
                batch_size,
                self.num_blocks,
                self.d_model,
                self.d_model,
                device=device or parameter.device,
                dtype=dtype or parameter.dtype,
            )
        )

    def allocation(self, context_ids: Tensor, mode: AllocationMode) -> Tensor:
        if context_ids.ndim != 1:
            raise ValueError("context_ids must have shape [batch]")
        if mode == "learned":
            logits = self.router(self.context_embedding(context_ids))
            return F.softmax(logits / self.temperature, dim=-1)
        if mode == "uniform":
            return torch.full(
                (context_ids.shape[0], self.num_blocks),
                1.0 / self.num_blocks,
                device=context_ids.device,
                dtype=self.key_embedding.weight.dtype,
            )
        if mode == "oracle":
            if self.num_blocks < self.num_contexts:
                raise ValueError("oracle allocation requires at least one block per context")
            return F.one_hot(context_ids, num_classes=self.num_blocks).to(
                dtype=self.key_embedding.weight.dtype
            )
        raise ValueError(f"unknown allocation mode: {mode}")

    def write(
        self,
        state: CompetitiveMemoryState,
        context_ids: Tensor,
        key_ids: Tensor,
        value_ids: Tensor,
        *,
        mode: AllocationMode,
    ) -> tuple[CompetitiveMemoryState, AllocationTelemetry]:
        if state.weights.ndim != 4:
            raise ValueError("state.weights must have shape [batch, blocks, d_model, d_model]")
        if not (context_ids.ndim == key_ids.ndim == value_ids.ndim == 1):
            raise ValueError("write inputs must be rank-1 tensors")
        if not (context_ids.shape == key_ids.shape == value_ids.shape):
            raise ValueError("write inputs must have the same batch shape")
        if state.weights.shape[0] != key_ids.shape[0]:
            raise ValueError("memory state and write inputs must have the same batch size")

        allocation = self.allocation(context_ids, mode)
        key = F.normalize(self.key_embedding(key_ids), dim=-1)
        value = F.normalize(self.value_embedding(value_ids), dim=-1)
        predicted = torch.einsum("bnij,bj->bni", state.weights, key)
        residual = value[:, None, :] - predicted
        denominator = key.square().sum(dim=-1).clamp_min(1e-6)
        delta = residual.unsqueeze(-1) * key[:, None, None, :]
        delta = delta / denominator[:, None, None, None]
        next_weights = self.retention * state.weights
        next_weights = next_weights + allocation[:, :, None, None] * delta
        return CompetitiveMemoryState(next_weights), AllocationTelemetry(allocation)

    def read(
        self,
        state: CompetitiveMemoryState,
        context_ids: Tensor,
        key_ids: Tensor,
        *,
        mode: AllocationMode,
    ) -> Tensor:
        if context_ids.ndim != 1 or key_ids.ndim != 1:
            raise ValueError("read inputs must have shape [batch]")
        if context_ids.shape != key_ids.shape or state.weights.shape[0] != key_ids.shape[0]:
            raise ValueError("memory state and read inputs must have the same batch size")

        allocation = self.allocation(context_ids, mode)
        key = F.normalize(self.key_embedding(key_ids), dim=-1)
        block_reads = torch.einsum("bnij,bj->bni", state.weights, key)
        recalled = (allocation.unsqueeze(-1) * block_reads).sum(dim=1)
        value_bank = F.normalize(self.value_embedding.weight, dim=-1)
        return recalled @ value_bank.T * self.logit_scale.exp().clamp(max=100.0)
