from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class EpisodeBatch:
    event_keys: Tensor
    event_values: Tensor
    evidence: Tensor
    retained: Tensor
    query_keys: Tensor
    targets: Tensor

    def to(self, device: torch.device | str) -> "EpisodeBatch":
        return EpisodeBatch(
            event_keys=self.event_keys.to(device),
            event_values=self.event_values.to(device),
            evidence=self.evidence.to(device),
            retained=self.retained.to(device),
            query_keys=self.query_keys.to(device),
            targets=self.targets.to(device),
        )


def generate_episode_batch(
    *,
    batch_size: int,
    num_keys: int,
    num_values: int,
    num_events: int,
    num_queries: int,
    generator: torch.Generator | None = None,
) -> EpisodeBatch:
    """Generate delayed-recall episodes with noisy future-utility evidence.

    Every event is a key/value observation. Only ``num_queries`` observations
    will be queried later. Their evidence distribution overlaps with distractors,
    so no deterministic threshold reveals the target set. The write gate is
    trained only from the delayed query loss.
    """

    if not 1 <= num_queries < num_events <= num_keys:
        raise ValueError("require 1 <= num_queries < num_events <= num_keys")

    key_noise = torch.rand(batch_size, num_keys, generator=generator)
    event_keys = key_noise.argsort(dim=1)[:, :num_events]
    event_values = torch.randint(
        num_values, (batch_size, num_events), generator=generator
    )

    retained = torch.zeros(batch_size, num_events, dtype=torch.bool)
    for row in range(batch_size):
        chosen = torch.randperm(num_events, generator=generator)[:num_queries]
        retained[row, chosen] = True

    # Useful events tend to look salient, but the distributions overlap. This is
    # the only present-time clue; the actual supervision arrives at future recall.
    evidence = torch.randn(batch_size, num_events, generator=generator) * 0.22
    evidence = evidence + torch.where(retained, 0.68, 0.32)
    evidence = evidence.clamp(0.0, 1.0)

    query_keys = event_keys[retained].reshape(batch_size, num_queries)
    targets = event_values[retained].reshape(batch_size, num_queries)
    return EpisodeBatch(
        event_keys=event_keys,
        event_values=event_values,
        evidence=evidence,
        retained=retained,
        query_keys=query_keys,
        targets=targets,
    )

