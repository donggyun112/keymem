from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .online_data import OnlineSplit, OnlineTextBanks
from .semantic_data import FrozenTextBanks


@dataclass(frozen=True)
class DelayedUtilityBatch:
    observation_context_features: Tensor
    observation_local_ids: Tensor
    observation_keys: Tensor
    observation_values: Tensor
    feedback_context_features: Tensor
    feedback_cue_features: Tensor
    feedback_local_ids: Tensor
    feedback_should_retain: Tensor
    query_context_features: Tensor
    query_local_ids: Tensor
    query_keys: Tensor
    targets: Tensor
    retained_contexts: Tensor

    def to(self, device: torch.device | str) -> "DelayedUtilityBatch":
        return DelayedUtilityBatch(
            **{name: value.to(device) for name, value in self.__dict__.items()}
        )


def generate_delayed_utility_batch(
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
) -> DelayedUtilityBatch:
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
        context_bank.shape[1], (batch_size, active_contexts), generator=generator
    )
    active_features = context_bank[active_domains, context_variants]

    retained = torch.zeros(batch_size, active_contexts, dtype=torch.bool)
    utility_order = torch.rand(
        batch_size, active_contexts, generator=generator
    ).argsort(dim=1)
    retained.scatter_(1, utility_order[:, :retained_contexts], True)
    cue_classes = (~retained).long()
    cue_variants = torch.randint(
        cue_bank.shape[1], cue_classes.shape, generator=generator
    )
    active_cues = cue_bank[cue_classes, cue_variants]

    shared_keys = torch.rand(
        batch_size, num_local_keys, generator=generator
    ).argsort(dim=1)[:, :events_per_context]
    values = torch.rand(
        batch_size, events_per_context, num_values, generator=generator
    ).argsort(dim=-1)[..., :active_contexts]
    all_values = values.permute(0, 2, 1).contiguous()

    local_ids = torch.arange(active_contexts).view(1, active_contexts, 1)
    local_ids = local_ids.expand(batch_size, -1, events_per_context)
    context_features = active_features[:, :, None, :].expand(
        -1, -1, events_per_context, -1
    )
    keys = shared_keys[:, None, :].expand(-1, active_contexts, -1)
    event_count = active_contexts * events_per_context
    event_order = torch.rand(batch_size, event_count, generator=generator).argsort(dim=1)

    flat_contexts = context_features.flatten(1, 2)
    observation_context_features = flat_contexts.gather(
        1, event_order[:, :, None].expand(-1, -1, context_bank.shape[-1])
    )
    observation_local_ids = local_ids.flatten(1).gather(1, event_order)
    observation_keys = keys.reshape(batch_size, -1).gather(1, event_order)
    observation_values = all_values.flatten(1).gather(1, event_order)

    feedback_order = torch.rand(
        batch_size, active_contexts, generator=generator
    ).argsort(dim=1)
    feedback_context_features = active_features.gather(
        1, feedback_order[:, :, None].expand(-1, -1, context_bank.shape[-1])
    )
    feedback_cue_features = active_cues.gather(
        1, feedback_order[:, :, None].expand(-1, -1, cue_bank.shape[-1])
    )
    feedback_should_retain = retained.gather(1, feedback_order)

    retained_local_ids = retained.nonzero(as_tuple=False)[:, 1].reshape(
        batch_size, retained_contexts
    )
    query_context_features = active_features.gather(
        1, retained_local_ids[:, :, None].expand(-1, -1, context_bank.shape[-1])
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
    targets = all_values.gather(
        1, retained_local_ids[:, :, None].expand(-1, -1, events_per_context)
    ).flatten(1)

    return DelayedUtilityBatch(
        observation_context_features=observation_context_features,
        observation_local_ids=observation_local_ids,
        observation_keys=observation_keys,
        observation_values=observation_values,
        feedback_context_features=feedback_context_features,
        feedback_cue_features=feedback_cue_features,
        feedback_local_ids=feedback_order,
        feedback_should_retain=feedback_should_retain,
        query_context_features=query_context_features,
        query_local_ids=query_local_ids,
        query_keys=query_keys,
        targets=targets,
        retained_contexts=retained,
    )
