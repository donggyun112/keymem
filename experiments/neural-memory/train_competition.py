from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from neural_memory import (
    AllocationMode,
    CompetitionBatch,
    CompetitiveFastWeightMemory,
    generate_competition_batch,
)
from train import choose_device


@dataclass(frozen=True)
class CompetitionMetrics:
    loss: float
    accuracy: float
    intrusion_rate: float
    assignment_score: float
    allocation_entropy: float
    max_allocation: float


@dataclass(frozen=True)
class Variant:
    name: str
    mode: AllocationMode
    num_blocks: int
    d_model: int


def forward_episode(
    model: CompetitiveFastWeightMemory,
    batch: CompetitionBatch,
    *,
    mode: AllocationMode,
) -> Tensor:
    state = model.initial_state(batch.event_keys.shape[0])
    for step in range(batch.event_keys.shape[1]):
        state, _ = model.write(
            state,
            batch.event_contexts[:, step],
            batch.event_keys[:, step],
            batch.event_values[:, step],
            mode=mode,
        )

    logits = [
        model.read(
            state,
            batch.query_contexts[:, step],
            batch.query_keys[:, step],
            mode=mode,
        )
        for step in range(batch.query_keys.shape[1])
    ]
    return torch.stack(logits, dim=1)


def allocation_diagnostics(
    model: CompetitiveFastWeightMemory,
    mode: AllocationMode,
) -> tuple[float, float, float]:
    device = model.key_embedding.weight.device
    contexts = torch.arange(model.num_contexts, device=device)
    allocation = model.allocation(contexts, mode)
    max_allocation = float(allocation.max(dim=-1).values.mean().detach().cpu())

    if model.num_blocks == 1:
        return 0.0, 0.0, max_allocation

    entropy = -(allocation * allocation.clamp_min(1e-9).log()).sum(dim=-1)
    entropy = entropy / math.log(model.num_blocks)

    if model.num_blocks < model.num_contexts:
        assignment_score = 0.0
    else:
        scores = []
        for assignment in itertools.permutations(
            range(model.num_blocks), model.num_contexts
        ):
            block_ids = torch.tensor(assignment, device=device)
            scores.append(allocation[contexts, block_ids].mean())
        assignment_score = float(torch.stack(scores).max().detach().cpu())

    return (
        assignment_score,
        float(entropy.mean().detach().cpu()),
        max_allocation,
    )


def measure(
    model: CompetitiveFastWeightMemory,
    logits: Tensor,
    batch: CompetitionBatch,
    *,
    mode: AllocationMode,
) -> CompetitionMetrics:
    loss = F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten())
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == batch.targets).float().mean()

    candidate_index = batch.query_slots[:, None, :].expand(
        -1, batch.all_values.shape[1], -1
    )
    same_key_values = batch.all_values.gather(2, candidate_index)
    other_context = (
        torch.arange(batch.all_values.shape[1], device=predictions.device)[None, :, None]
        != batch.query_contexts[:, None, :]
    )
    intrusions = ((predictions[:, None, :] == same_key_values) & other_context).any(dim=1)
    intrusion_rate = intrusions.float().mean()

    assignment_score, entropy, max_allocation = allocation_diagnostics(model, mode)
    return CompetitionMetrics(
        loss=float(loss.detach().cpu()),
        accuracy=float(accuracy.detach().cpu()),
        intrusion_rate=float(intrusion_rate.detach().cpu()),
        assignment_score=assignment_score,
        allocation_entropy=entropy,
        max_allocation=max_allocation,
    )


def make_batch(
    args: argparse.Namespace,
    generator: torch.Generator,
    device: torch.device,
) -> CompetitionBatch:
    return generate_competition_batch(
        batch_size=args.batch_size,
        num_contexts=args.num_contexts,
        num_local_keys=args.num_local_keys,
        num_values=args.num_values,
        events_per_context=args.events_per_context,
        generator=generator,
    ).to(device)


@torch.no_grad()
def evaluate(
    model: CompetitiveFastWeightMemory,
    args: argparse.Namespace,
    device: torch.device,
    *,
    mode: AllocationMode,
    seed: int,
) -> CompetitionMetrics:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    totals = torch.zeros(3)
    for _ in range(args.eval_batches):
        batch = make_batch(args, generator, device)
        logits = forward_episode(model, batch, mode=mode)
        current = measure(model, logits, batch, mode=mode)
        totals += torch.tensor([current.loss, current.accuracy, current.intrusion_rate])
    totals /= args.eval_batches
    assignment, entropy, max_allocation = allocation_diagnostics(model, mode)
    return CompetitionMetrics(
        loss=float(totals[0]),
        accuracy=float(totals[1]),
        intrusion_rate=float(totals[2]),
        assignment_score=assignment,
        allocation_entropy=entropy,
        max_allocation=max_allocation,
    )


def train_variant(
    args: argparse.Namespace,
    device: torch.device,
    variant: Variant,
    *,
    seed: int,
) -> tuple[CompetitiveFastWeightMemory, CompetitionMetrics]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = CompetitiveFastWeightMemory(
        num_contexts=args.num_contexts,
        num_keys=args.num_local_keys,
        num_values=args.num_values,
        num_blocks=variant.num_blocks,
        d_model=variant.d_model,
        router_dim=args.router_dim,
        temperature=args.temperature,
        retention=args.retention,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(seed + 1)

    model.train()
    for _ in range(args.steps):
        batch = make_batch(args, generator, device)
        logits = forward_episode(model, batch, mode=variant.mode)
        loss = F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    metrics = evaluate(
        model,
        args,
        device,
        mode=variant.mode,
        seed=seed + 10_000,
    )
    return model, metrics


def variants(args: argparse.Namespace) -> list[Variant]:
    fast_capacity = args.num_blocks * args.block_dim * args.block_dim
    single_dim = math.isqrt(fast_capacity)
    if single_dim * single_dim != fast_capacity:
        raise ValueError(
            "num_blocks * block_dim^2 must be a square to capacity-match single memory"
        )
    if args.num_blocks < args.num_contexts:
        raise ValueError("num_blocks must be at least num_contexts for the oracle control")
    return [
        Variant("single", "uniform", 1, single_dim),
        Variant("uniform", "uniform", args.num_blocks, args.block_dim),
        Variant("competitive", "learned", args.num_blocks, args.block_dim),
        Variant("oracle", "oracle", args.num_blocks, args.block_dim),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test learned competitive fast-weight allocation")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-contexts", type=int, default=4)
    parser.add_argument("--num-local-keys", type=int, default=32)
    parser.add_argument("--num-values", type=int, default=64)
    parser.add_argument("--events-per-context", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--block-dim", type=int, default=12)
    parser.add_argument("--router-dim", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--retention", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print a compact metric table instead of the full JSON payload",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    result: dict[str, object] = {
        "device": str(device),
        "config": vars(args),
        "chance_accuracy": 1.0 / args.num_values,
        "fast_capacity": args.num_blocks * args.block_dim * args.block_dim,
    }
    for variant in variants(args):
        model, metrics = train_variant(args, device, variant, seed=args.seed)
        result[variant.name] = {
            **asdict(metrics),
            "fast_capacity": model.fast_capacity,
        }
    if args.summary:
        compact = {
            name: {
                key: result[name][key]  # type: ignore[index]
                for key in ("accuracy", "intrusion_rate", "assignment_score")
            }
            for name in ("single", "uniform", "competitive", "oracle")
        }
        print(json.dumps(compact, separators=(",", ":"), sort_keys=True))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
