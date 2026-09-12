from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MemoryState:
    """Per-episode fast weights. This tensor is the memory, not a graph index."""

    weights: Tensor

    def detached(self) -> "MemoryState":
        return MemoryState(self.weights.detach())


@dataclass(frozen=True)
class WriteTelemetry:
    gate: Tensor
    surprise: Tensor


class FastWeightMemory(nn.Module):
    """A minimal learned fast-weight associative memory.

    Slow parameters learn representations and the write rule. ``MemoryState.weights``
    changes for every observation and carries the episode-specific information.
    The residual outer-product update is the delta rule used by Fast Weight
    Programmers; surprise and retention follow the design vocabulary of Titans.
    """

    def __init__(
        self,
        num_keys: int,
        num_values: int,
        d_model: int = 32,
        retention: float = 0.995,
    ) -> None:
        super().__init__()
        if num_keys < 2 or num_values < 2:
            raise ValueError("num_keys and num_values must both be at least 2")
        if d_model < 2:
            raise ValueError("d_model must be at least 2")
        if not 0.0 <= retention <= 1.0:
            raise ValueError("retention must be in [0, 1]")

        self.num_keys = num_keys
        self.num_values = num_values
        self.d_model = d_model
        self.retention = retention

        self.key_embedding = nn.Embedding(num_keys, d_model)
        self.value_embedding = nn.Embedding(num_values, d_model)
        self.write_gate = nn.Sequential(
            nn.Linear(2 * d_model + 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        self.logit_scale = nn.Parameter(torch.tensor(2.0))

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> MemoryState:
        parameter = self.key_embedding.weight
        return MemoryState(
            torch.zeros(
                batch_size,
                self.d_model,
                self.d_model,
                device=device or parameter.device,
                dtype=dtype or parameter.dtype,
            )
        )

    def write(
        self,
        state: MemoryState,
        key_ids: Tensor,
        value_ids: Tensor,
        evidence: Tensor,
        *,
        force_gate: float | Tensor | None = None,
    ) -> tuple[MemoryState, WriteTelemetry]:
        """Update the model-state memory from one observation.

        ``evidence`` is a noisy contextual feature, not a write label. During
        training the gate receives supervision only through later recall loss.
        """

        if state.weights.ndim != 3:
            raise ValueError("state.weights must have shape [batch, d_model, d_model]")
        if key_ids.ndim != 1 or value_ids.ndim != 1 or evidence.ndim != 1:
            raise ValueError("key_ids, value_ids, and evidence must be rank-1 tensors")
        if not (key_ids.shape == value_ids.shape == evidence.shape):
            raise ValueError("write inputs must have the same batch shape")
        if state.weights.shape[0] != key_ids.shape[0]:
            raise ValueError("memory state and write inputs must have the same batch size")

        key = F.normalize(self.key_embedding(key_ids), dim=-1)
        value = F.normalize(self.value_embedding(value_ids), dim=-1)
        predicted = torch.bmm(state.weights, key.unsqueeze(-1)).squeeze(-1)
        residual = value - predicted
        surprise = residual.square().mean(dim=-1)

        gate_features = torch.cat(
            (key, value, evidence.unsqueeze(-1), surprise.unsqueeze(-1)), dim=-1
        )
        learned_gate = torch.sigmoid(self.write_gate(gate_features)).squeeze(-1)
        if force_gate is None:
            gate = learned_gate
        elif isinstance(force_gate, Tensor):
            gate = force_gate.to(device=learned_gate.device, dtype=learned_gate.dtype)
            gate = torch.broadcast_to(gate, learned_gate.shape)
        else:
            gate = torch.full_like(learned_gate, float(force_gate))

        # Delta rule: write only the residual, so a repeated key updates rather
        # than blindly accumulating a second copy of the old association.
        denominator = key.square().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        delta = torch.bmm(residual.unsqueeze(-1), key.unsqueeze(1))
        delta = delta / denominator.unsqueeze(-1)
        next_weights = self.retention * state.weights + gate[:, None, None] * delta
        return MemoryState(next_weights), WriteTelemetry(gate=gate, surprise=surprise)

    def read(self, state: MemoryState, key_ids: Tensor) -> Tensor:
        """Return value logits produced by a forward pass through fast weights."""

        if key_ids.ndim != 1:
            raise ValueError("key_ids must have shape [batch]")
        if state.weights.shape[0] != key_ids.shape[0]:
            raise ValueError("memory state and read keys must have the same batch size")
        query = F.normalize(self.key_embedding(key_ids), dim=-1)
        recalled = torch.bmm(state.weights, query.unsqueeze(-1)).squeeze(-1)
        value_bank = F.normalize(self.value_embedding.weight, dim=-1)
        return recalled @ value_bank.T * self.logit_scale.exp().clamp(max=100.0)

