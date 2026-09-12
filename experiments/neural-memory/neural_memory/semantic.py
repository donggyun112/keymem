from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .competitive import CompetitiveMemoryState


SemanticAllocationMode = Literal["learned", "uniform", "oracle"]
GateMode = Literal["learned", "always", "oracle"]


@dataclass(frozen=True)
class SemanticTelemetry:
    allocation: Tensor
    gate: Tensor
    surprise: Tensor


class SemanticFastWeightMemory(nn.Module):
    """Joint semantic write gate and competitive fast-weight allocator."""

    def __init__(
        self,
        *,
        text_dim: int,
        num_contexts: int,
        num_keys: int,
        num_values: int,
        num_blocks: int,
        d_model: int,
        router_dim: int = 32,
        gate_mlp: bool = False,
        temperature: float = 0.5,
        retention: float = 1.0,
    ) -> None:
        super().__init__()
        if min(text_dim, num_contexts, num_keys, num_values, num_blocks, d_model) < 1:
            raise ValueError("model dimensions must be positive")
        if router_dim < 2:
            raise ValueError("router_dim must be at least 2")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= retention <= 1.0:
            raise ValueError("retention must be in [0, 1]")

        self.text_dim = text_dim
        self.num_contexts = num_contexts
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.temperature = temperature
        self.retention = retention

        self.key_embedding = nn.Embedding(num_keys, d_model)
        self.value_embedding = nn.Embedding(num_values, d_model)
        self.context_router = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, router_dim),
            nn.SiLU(),
            nn.Linear(router_dim, num_blocks),
        )
        if gate_mlp:
            self.write_gate = nn.Sequential(
                nn.LayerNorm(text_dim + 1),
                nn.Linear(text_dim + 1, router_dim),
                nn.SiLU(),
                nn.Linear(router_dim, 1),
            )
        else:
            # A linear probe preserves more of the frozen encoder's geometry and
            # has far fewer ways to overfit the small set of training phrasings.
            self.write_gate = nn.Linear(text_dim + 1, 1)
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

    def allocation(
        self,
        context_features: Tensor,
        *,
        mode: SemanticAllocationMode,
        context_ids: Tensor | None = None,
    ) -> Tensor:
        if context_features.ndim != 2 or context_features.shape[-1] != self.text_dim:
            raise ValueError("context_features must have shape [batch, text_dim]")
        if mode == "learned":
            logits = self.context_router(context_features)
            return F.softmax(logits / self.temperature, dim=-1)
        if mode == "uniform":
            return torch.full(
                (context_features.shape[0], self.num_blocks),
                1.0 / self.num_blocks,
                device=context_features.device,
                dtype=context_features.dtype,
            )
        if mode == "oracle":
            if context_ids is None:
                raise ValueError("oracle allocation requires context_ids")
            if self.num_blocks < self.num_contexts:
                raise ValueError("oracle allocation requires one block per context")
            return F.one_hot(context_ids, num_classes=self.num_blocks).to(
                dtype=context_features.dtype
            )
        raise ValueError(f"unknown allocation mode: {mode}")

    def write(
        self,
        state: CompetitiveMemoryState,
        context_features: Tensor,
        cue_features: Tensor,
        key_ids: Tensor,
        value_ids: Tensor,
        *,
        allocation_mode: SemanticAllocationMode,
        gate_mode: GateMode,
        context_ids: Tensor | None = None,
        should_write: Tensor | None = None,
    ) -> tuple[CompetitiveMemoryState, SemanticTelemetry]:
        if cue_features.ndim != 2 or cue_features.shape[-1] != self.text_dim:
            raise ValueError("cue_features must have shape [batch, text_dim]")
        if key_ids.ndim != 1 or value_ids.ndim != 1:
            raise ValueError("key_ids and value_ids must have shape [batch]")

        allocation = self.allocation(
            context_features, mode=allocation_mode, context_ids=context_ids
        )
        key = F.normalize(self.key_embedding(key_ids), dim=-1)
        value = F.normalize(self.value_embedding(value_ids), dim=-1)
        predicted = torch.einsum("bnij,bj->bni", state.weights, key)
        residual = value[:, None, :] - predicted
        surprise = residual.square().mean(dim=(1, 2))

        if gate_mode == "learned":
            gate_input = torch.cat((cue_features, surprise[:, None]), dim=-1)
            gate = torch.sigmoid(self.write_gate(gate_input)).squeeze(-1)
        elif gate_mode == "always":
            gate = torch.ones_like(surprise)
        elif gate_mode == "oracle":
            if should_write is None:
                raise ValueError("oracle gate requires should_write")
            gate = should_write.to(dtype=surprise.dtype)
        else:
            raise ValueError(f"unknown gate mode: {gate_mode}")

        denominator = key.square().sum(dim=-1).clamp_min(1e-6)
        delta = residual.unsqueeze(-1) * key[:, None, None, :]
        delta = delta / denominator[:, None, None, None]
        strength = gate[:, None] * allocation
        next_weights = self.retention * state.weights
        next_weights = next_weights + strength[:, :, None, None] * delta
        return CompetitiveMemoryState(next_weights), SemanticTelemetry(
            allocation=allocation,
            gate=gate,
            surprise=surprise,
        )

    def read(
        self,
        state: CompetitiveMemoryState,
        context_features: Tensor,
        key_ids: Tensor,
        *,
        allocation_mode: SemanticAllocationMode,
        context_ids: Tensor | None = None,
    ) -> Tensor:
        allocation = self.allocation(
            context_features, mode=allocation_mode, context_ids=context_ids
        )
        key = F.normalize(self.key_embedding(key_ids), dim=-1)
        block_reads = torch.einsum("bnij,bj->bni", state.weights, key)
        recalled = (allocation.unsqueeze(-1) * block_reads).sum(dim=1)
        value_bank = F.normalize(self.value_embedding.weight, dim=-1)
        return recalled @ value_bank.T * self.logit_scale.exp().clamp(max=100.0)
