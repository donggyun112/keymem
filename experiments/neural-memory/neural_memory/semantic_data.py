from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor


Split = Literal["train", "eval"]


@dataclass(frozen=True)
class FrozenTextBanks:
    train_contexts: Tensor
    eval_contexts: Tensor
    train_cues: Tensor
    eval_cues: Tensor

    @property
    def text_dim(self) -> int:
        return self.train_contexts.shape[-1]

    def contexts(self, split: Split) -> Tensor:
        return self.train_contexts if split == "train" else self.eval_contexts

    def cues(self, split: Split) -> Tensor:
        return self.train_cues if split == "train" else self.eval_cues


def load_text_banks(path: str | Path) -> FrozenTextBanks:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = ("train_contexts", "eval_contexts", "train_cues", "eval_cues")
    if any(name not in payload for name in required):
        raise ValueError("embedding artifact is missing required tensor banks")
    banks = FrozenTextBanks(**{name: payload[name].float() for name in required})
    if banks.train_contexts.ndim != 3 or banks.eval_contexts.ndim != 3:
        raise ValueError("context banks must have shape [contexts, variants, text_dim]")
    if banks.train_cues.ndim != 3 or banks.eval_cues.ndim != 3:
        raise ValueError("cue banks must have shape [classes, variants, text_dim]")
    if banks.train_cues.shape[0] != 2 or banks.eval_cues.shape[0] != 2:
        raise ValueError("cue bank class 0 must mean retain and class 1 distractor")
    return banks


@dataclass(frozen=True)
class SemanticEpisodeBatch:
    event_context_features: Tensor
    event_cue_features: Tensor
    event_context_ids: Tensor
    event_keys: Tensor
    event_values: Tensor
    event_should_write: Tensor
    query_context_features: Tensor
    query_context_ids: Tensor
    query_keys: Tensor
    query_slots: Tensor
    targets: Tensor
    all_values: Tensor
    distractor_values: Tensor

    def to(self, device: torch.device | str) -> "SemanticEpisodeBatch":
        return SemanticEpisodeBatch(
            **{
                name: value.to(device)
                for name, value in self.__dict__.items()
            }
        )


def _sample_features(
    bank: Tensor,
    class_ids: Tensor,
    *,
    generator: torch.Generator | None,
) -> Tensor:
    variant_ids = torch.randint(
        bank.shape[1], class_ids.shape, generator=generator
    )
    return bank[class_ids, variant_ids]


def generate_semantic_episode_batch(
    *,
    banks: FrozenTextBanks,
    split: Split,
    batch_size: int,
    num_local_keys: int,
    num_values: int,
    events_per_context: int,
    generator: torch.Generator | None = None,
) -> SemanticEpisodeBatch:
    """Generate joint write-selection and semantic-allocation episodes.

    Every useful binding is followed by a temporary conflicting observation for
    the same context and key. Recall asks for the stable value. Context and
    durability are exposed only through frozen sentence embeddings; categorical
    IDs and write labels are retained solely for oracle controls and metrics.
    """

    context_bank = banks.contexts(split)
    cue_bank = banks.cues(split)
    num_contexts = context_bank.shape[0]
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 1 <= events_per_context <= num_local_keys:
        raise ValueError("require 1 <= events_per_context <= num_local_keys")
    if num_values < 2 * num_contexts:
        raise ValueError("num_values must be at least twice num_contexts")

    shared_keys = torch.rand(
        batch_size, num_local_keys, generator=generator
    ).argsort(dim=1)[:, :events_per_context]
    value_order = torch.rand(
        batch_size, events_per_context, num_values, generator=generator
    ).argsort(dim=-1)[..., : 2 * num_contexts]
    value_order = value_order.permute(0, 2, 1).contiguous()
    all_values = value_order[:, :num_contexts]
    distractor_values = value_order[:, num_contexts:]

    contexts = torch.arange(num_contexts).view(1, num_contexts, 1)
    contexts = contexts.expand(batch_size, -1, events_per_context)
    keys = shared_keys[:, None, :].expand(-1, num_contexts, -1)
    slots = torch.arange(events_per_context).view(1, 1, events_per_context)
    slots = slots.expand(batch_size, num_contexts, -1)

    pair_contexts = contexts.reshape(batch_size, -1)
    pair_keys = keys.reshape(batch_size, -1)
    pair_slots = slots.reshape(batch_size, -1)
    pair_targets = all_values.reshape(batch_size, -1)
    pair_distractors = distractor_values.reshape(batch_size, -1)
    pair_count = pair_contexts.shape[1]
    pair_order = torch.rand(batch_size, pair_count, generator=generator).argsort(dim=1)

    pair_contexts = pair_contexts.gather(1, pair_order)
    pair_keys = pair_keys.gather(1, pair_order)
    pair_targets = pair_targets.gather(1, pair_order)
    pair_distractors = pair_distractors.gather(1, pair_order)
    event_context_ids = torch.stack((pair_contexts, pair_contexts), dim=-1).flatten(1)
    event_keys = torch.stack((pair_keys, pair_keys), dim=-1).flatten(1)
    event_values = torch.stack((pair_targets, pair_distractors), dim=-1).flatten(1)
    event_should_write = torch.tensor((True, False)).view(1, 1, 2)
    event_should_write = event_should_write.expand(batch_size, pair_count, -1).flatten(1)

    event_context_features = _sample_features(
        context_bank, event_context_ids, generator=generator
    )
    cue_class = (~event_should_write).long()
    event_cue_features = _sample_features(cue_bank, cue_class, generator=generator)

    query_context_ids = contexts.reshape(batch_size, -1)
    query_keys = keys.reshape(batch_size, -1)
    query_slots = slots.reshape(batch_size, -1)
    targets = all_values.reshape(batch_size, -1)
    query_context_features = _sample_features(
        context_bank, query_context_ids, generator=generator
    )

    return SemanticEpisodeBatch(
        event_context_features=event_context_features,
        event_cue_features=event_cue_features,
        event_context_ids=event_context_ids,
        event_keys=event_keys,
        event_values=event_values,
        event_should_write=event_should_write,
        query_context_features=query_context_features,
        query_context_ids=query_context_ids,
        query_keys=query_keys,
        query_slots=query_slots,
        targets=targets,
        all_values=all_values,
        distractor_values=distractor_values,
    )
