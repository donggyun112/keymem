from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CompetitionBatch:
    """Context-collision episodes for testing competitive memory allocation."""

    event_contexts: Tensor
    event_keys: Tensor
    event_values: Tensor
    query_contexts: Tensor
    query_keys: Tensor
    query_slots: Tensor
    targets: Tensor
    all_values: Tensor

    def to(self, device: torch.device | str) -> "CompetitionBatch":
        return CompetitionBatch(
            event_contexts=self.event_contexts.to(device),
            event_keys=self.event_keys.to(device),
            event_values=self.event_values.to(device),
            query_contexts=self.query_contexts.to(device),
            query_keys=self.query_keys.to(device),
            query_slots=self.query_slots.to(device),
            targets=self.targets.to(device),
            all_values=self.all_values.to(device),
        )


def generate_competition_batch(
    *,
    batch_size: int,
    num_contexts: int,
    num_local_keys: int,
    num_values: int,
    events_per_context: int,
    generator: torch.Generator | None = None,
) -> CompetitionBatch:
    """Generate conflicting bindings that require context-dependent allocation.

    Each episode samples one shared set of local keys. Every context observes all
    of those keys, but the same key maps to a different value in every context.
    Thus a single undivided fast-weight matrix is forced to overwrite colliding
    bindings, while independently allocated blocks can retain them.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_contexts < 2:
        raise ValueError("num_contexts must be at least 2")
    if not 1 <= events_per_context <= num_local_keys:
        raise ValueError("require 1 <= events_per_context <= num_local_keys")
    if num_values < num_contexts:
        raise ValueError("num_values must be at least num_contexts")

    shared_keys = torch.rand(
        batch_size, num_local_keys, generator=generator
    ).argsort(dim=1)[:, :events_per_context]

    # For each shared key slot, use distinct targets across contexts so that a
    # wrong-context recall can be identified unambiguously as an intrusion.
    value_order = torch.rand(
        batch_size, events_per_context, num_values, generator=generator
    ).argsort(dim=-1)[..., :num_contexts]
    all_values = value_order.permute(0, 2, 1).contiguous()

    contexts = torch.arange(num_contexts).view(1, num_contexts, 1)
    contexts = contexts.expand(batch_size, -1, events_per_context)
    keys = shared_keys[:, None, :].expand(-1, num_contexts, -1)
    slots = torch.arange(events_per_context).view(1, 1, events_per_context)
    slots = slots.expand(batch_size, num_contexts, -1)

    query_contexts = contexts.reshape(batch_size, -1)
    query_keys = keys.reshape(batch_size, -1)
    query_slots = slots.reshape(batch_size, -1)
    targets = all_values.reshape(batch_size, -1)

    # Interleave writes differently in every episode while keeping queries in a
    # stable order for simple metric computation.
    event_count = num_contexts * events_per_context
    permutation = torch.rand(batch_size, event_count, generator=generator).argsort(dim=1)
    event_contexts = query_contexts.gather(1, permutation)
    event_keys = query_keys.gather(1, permutation)
    event_values = targets.gather(1, permutation)

    return CompetitionBatch(
        event_contexts=event_contexts,
        event_keys=event_keys,
        event_values=event_values,
        query_contexts=query_contexts,
        query_keys=query_keys,
        query_slots=query_slots,
        targets=targets,
        all_values=all_values,
    )
