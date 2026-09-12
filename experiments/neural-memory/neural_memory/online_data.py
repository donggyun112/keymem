from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor


OnlineSplit = Literal["train", "eval"]


@dataclass(frozen=True)
class OnlineTextBanks:
    train_contexts: Tensor
    eval_contexts: Tensor

    @property
    def text_dim(self) -> int:
        return self.train_contexts.shape[-1]

    @property
    def num_domains(self) -> int:
        return self.train_contexts.shape[0]

    def contexts(self, split: OnlineSplit) -> Tensor:
        return self.train_contexts if split == "train" else self.eval_contexts


def load_online_text_banks(path: str | Path) -> OnlineTextBanks:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = ("train_contexts", "eval_contexts")
    if any(name not in payload for name in required):
        raise ValueError("online embedding artifact is missing context banks")
    banks = OnlineTextBanks(**{name: payload[name].float() for name in required})
    if banks.train_contexts.ndim != 3 or banks.eval_contexts.ndim != 3:
        raise ValueError("context banks must have shape [domains, variants, text_dim]")
    if banks.train_contexts.shape[0] != banks.eval_contexts.shape[0]:
        raise ValueError("train and eval banks must contain the same domains")
    if banks.train_contexts.shape[-1] != banks.eval_contexts.shape[-1]:
        raise ValueError("train and eval embedding dimensions must match")
    return banks


@dataclass(frozen=True)
class OnlineEpisodeBatch:
    event_context_features: Tensor
    event_domain_ids: Tensor
    event_local_ids: Tensor
    event_keys: Tensor
    event_values: Tensor
    query_context_features: Tensor
    query_domain_ids: Tensor
    query_local_ids: Tensor
    query_keys: Tensor
    query_slots: Tensor
    targets: Tensor
    all_values: Tensor
    active_domain_ids: Tensor

    def to(self, device: torch.device | str) -> "OnlineEpisodeBatch":
        return OnlineEpisodeBatch(
            **{name: value.to(device) for name, value in self.__dict__.items()}
        )


def generate_online_episode_batch(
    *,
    banks: OnlineTextBanks,
    split: OnlineSplit,
    batch_size: int,
    active_contexts: int,
    num_local_keys: int,
    num_values: int,
    events_per_context: int,
    generator: torch.Generator | None = None,
) -> OnlineEpisodeBatch:
    """Sample a new subset of semantic domains and assignments per episode."""

    context_bank = banks.contexts(split)
    num_domains = context_bank.shape[0]
    if not 2 <= active_contexts <= num_domains:
        raise ValueError("require 2 <= active_contexts <= num_domains")
    if not 1 <= events_per_context <= num_local_keys:
        raise ValueError("require 1 <= events_per_context <= num_local_keys")
    if num_values < active_contexts:
        raise ValueError("num_values must be at least active_contexts")

    active_domain_ids = torch.rand(
        batch_size, num_domains, generator=generator
    ).argsort(dim=1)[:, :active_contexts]
    shared_keys = torch.rand(
        batch_size, num_local_keys, generator=generator
    ).argsort(dim=1)[:, :events_per_context]

    value_order = torch.rand(
        batch_size, events_per_context, num_values, generator=generator
    ).argsort(dim=-1)[..., :active_contexts]
    all_values = value_order.permute(0, 2, 1).contiguous()

    domain_ids = active_domain_ids[:, :, None].expand(-1, -1, events_per_context)
    local_ids = torch.arange(active_contexts).view(1, active_contexts, 1)
    local_ids = local_ids.expand(batch_size, -1, events_per_context)
    keys = shared_keys[:, None, :].expand(-1, active_contexts, -1)
    slots = torch.arange(events_per_context).view(1, 1, events_per_context)
    slots = slots.expand(batch_size, active_contexts, -1)

    active_variants = torch.randint(
        context_bank.shape[1],
        (batch_size, active_contexts),
        generator=generator,
    )
    active_context_features = context_bank[active_domain_ids, active_variants]
    context_features = active_context_features[:, :, None, :].expand(
        -1, -1, events_per_context, -1
    )

    query_domain_ids = domain_ids.reshape(batch_size, -1)
    query_local_ids = local_ids.reshape(batch_size, -1)
    query_keys = keys.reshape(batch_size, -1)
    query_slots = slots.reshape(batch_size, -1)
    targets = all_values.reshape(batch_size, -1)
    query_context_features = context_features.flatten(1, 2)

    event_count = active_contexts * events_per_context
    order = torch.rand(batch_size, event_count, generator=generator).argsort(dim=1)
    event_domain_ids = query_domain_ids.gather(1, order)
    event_local_ids = query_local_ids.gather(1, order)
    event_keys = query_keys.gather(1, order)
    event_values = targets.gather(1, order)
    event_context_features = query_context_features.gather(
        1, order[:, :, None].expand(-1, -1, context_bank.shape[-1])
    )

    return OnlineEpisodeBatch(
        event_context_features=event_context_features,
        event_domain_ids=event_domain_ids,
        event_local_ids=event_local_ids,
        event_keys=event_keys,
        event_values=event_values,
        query_context_features=query_context_features,
        query_domain_ids=query_domain_ids,
        query_local_ids=query_local_ids,
        query_keys=query_keys,
        query_slots=query_slots,
        targets=targets,
        all_values=all_values,
        active_domain_ids=active_domain_ids,
    )
