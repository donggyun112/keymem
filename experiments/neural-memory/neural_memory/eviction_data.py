from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .online_data import OnlineSplit, OnlineTextBanks
from .semantic_data import FrozenTextBanks


@dataclass(frozen=True)
class EvictionEpisodeBatch:
    event_context_features: Tensor
    event_cue_features: Tensor
    event_local_ids: Tensor
    event_keys: Tensor
    event_values: Tensor
    event_should_retain: Tensor
    query_context_features: Tensor
    query_local_ids: Tensor
    query_keys: Tensor
    query_slots: Tensor
    targets: Tensor
    retained_contexts: Tensor

    def to(self, device: torch.device | str) -> "EvictionEpisodeBatch":
        return EvictionEpisodeBatch(
            **{name: value.to(device) for name, value in self.__dict__.items()}
        )


def generate_eviction_episode_batch(
    *,
    context_banks: OnlineTextBanks,
    cue_banks: FrozenTextBanks,
    split: OnlineSplit,
    batch_size: int,
    active_contexts: int,
    retained_contexts: int,
    num_local_keys: int,
    num_values: int,
    events_per_context: int,
    generator: torch.Generator | None = None,
) -> EvictionEpisodeBatch:
    """Generate capacity-overflow episodes with latent future utility cues."""

    context_bank = context_banks.contexts(split)
    cue_bank = cue_banks.cues(split)
    num_domains = context_bank.shape[0]
    if not 1 <= retained_contexts < active_contexts <= num_domains:
        raise ValueError("require retained_contexts < active_contexts <= num_domains")
    if not 1 <= events_per_context <= num_local_keys:
        raise ValueError("require 1 <= events_per_context <= num_local_keys")
    if num_values < active_contexts:
        raise ValueError("num_values must be at least active_contexts")

    active_domains = torch.rand(
        batch_size, num_domains, generator=generator
    ).argsort(dim=1)[:, :active_contexts]
    context_variants = torch.randint(
        context_bank.shape[1],
        (batch_size, active_contexts),
        generator=generator,
    )
    active_features = context_bank[active_domains, context_variants]

    retained = torch.zeros(batch_size, active_contexts, dtype=torch.bool)
    retained_order = torch.rand(
        batch_size, active_contexts, generator=generator
    ).argsort(dim=1)
    retained.scatter_(1, retained_order[:, :retained_contexts], True)

    cue_classes = (~retained).long()
    cue_variants = torch.randint(
        cue_bank.shape[1], cue_classes.shape, generator=generator
    )
    active_cues = cue_bank[cue_classes, cue_variants]

    shared_keys = torch.rand(
        batch_size, num_local_keys, generator=generator
    ).argsort(dim=1)[:, :events_per_context]
    value_order = torch.rand(
        batch_size, events_per_context, num_values, generator=generator
    ).argsort(dim=-1)[..., :active_contexts]
    all_values = value_order.permute(0, 2, 1).contiguous()

    # Contexts arrive as short chunks. Whether a useful context arrives early or
    # late is randomized, so recency alone cannot solve the task.
    arrival_order = torch.rand(
        batch_size, active_contexts, generator=generator
    ).argsort(dim=1)
    ordered_features = active_features.gather(
        1, arrival_order[:, :, None].expand(-1, -1, context_bank.shape[-1])
    )
    ordered_cues = active_cues.gather(
        1, arrival_order[:, :, None].expand(-1, -1, cue_bank.shape[-1])
    )
    ordered_values = all_values.gather(
        1, arrival_order[:, :, None].expand(-1, -1, events_per_context)
    )
    ordered_retain = retained.gather(1, arrival_order)

    event_context_features = ordered_features[:, :, None, :].expand(
        -1, -1, events_per_context, -1
    ).flatten(1, 2)
    event_cue_features = ordered_cues[:, :, None, :].expand(
        -1, -1, events_per_context, -1
    ).flatten(1, 2)
    event_local_ids = arrival_order[:, :, None].expand(
        -1, -1, events_per_context
    ).flatten(1)
    event_keys = shared_keys[:, None, :].expand(
        -1, active_contexts, -1
    ).reshape(batch_size, -1)
    event_values = ordered_values.flatten(1)
    event_should_retain = ordered_retain[:, :, None].expand(
        -1, -1, events_per_context
    ).flatten(1)

    retained_local_ids = retained.nonzero(as_tuple=False)[:, 1].reshape(
        batch_size, retained_contexts
    )
    query_context_features = active_features.gather(
        1,
        retained_local_ids[:, :, None].expand(-1, -1, context_bank.shape[-1]),
    )
    query_context_features = query_context_features[:, :, None, :].expand(
        -1, -1, events_per_context, -1
    ).flatten(1, 2)
    query_local_ids = retained_local_ids[:, :, None].expand(
        -1, -1, events_per_context
    ).flatten(1)
    query_keys = shared_keys[:, None, :].expand(
        -1, retained_contexts, -1
    ).reshape(batch_size, -1)
    query_slots = torch.arange(events_per_context).view(1, 1, events_per_context)
    query_slots = query_slots.expand(batch_size, retained_contexts, -1).flatten(1)
    targets = all_values.gather(
        1,
        retained_local_ids[:, :, None].expand(-1, -1, events_per_context),
    ).flatten(1)

    return EvictionEpisodeBatch(
        event_context_features=event_context_features,
        event_cue_features=event_cue_features,
        event_local_ids=event_local_ids,
        event_keys=event_keys,
        event_values=event_values,
        event_should_retain=event_should_retain,
        query_context_features=query_context_features,
        query_local_ids=query_local_ids,
        query_keys=query_keys,
        query_slots=query_slots,
        targets=targets,
        retained_contexts=retained,
    )
