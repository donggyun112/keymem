from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from neural_memory import (
    OnlineEpisodeBatch,
    OnlinePrototypeMemory,
    OnlineTextBanks,
    SemanticFastWeightMemory,
    generate_online_episode_batch,
    load_online_text_banks,
)
from train import choose_device


VariantKind = Literal["static", "online", "oracle"]
MemoryModel = SemanticFastWeightMemory | OnlinePrototypeMemory


@dataclass(frozen=True)
class Variant:
    name: str
    kind: VariantKind
    num_blocks: int
    d_model: int


@dataclass(frozen=True)
class OnlineMetrics:
    loss: float
    accuracy: float
    context_intrusion_rate: float
    unique_block_ratio: float
    collision_free_episodes: float
    routing_consistency: float
    novelty_threshold: float


def forward_episode(
    model: MemoryModel,
    batch: OnlineEpisodeBatch,
    variant: Variant,
) -> tuple[Tensor, Tensor]:
    state = model.initial_state(batch.event_keys.shape[0])
    if isinstance(model, OnlinePrototypeMemory):
        for step in range(batch.event_keys.shape[1]):
            state, _ = model.write(
                state,
                batch.event_context_features[:, step],
                batch.event_keys[:, step],
                batch.event_values[:, step],
            )
        outputs = [
            model.read(
                state,
                batch.query_context_features[:, step],
                batch.query_keys[:, step],
            )
            for step in range(batch.query_keys.shape[1])
        ]
        logits, routes = zip(*outputs, strict=True)
        return torch.stack(logits, dim=1), torch.stack(routes, dim=1)

    allocation_mode = "learned" if variant.kind == "static" else "oracle"
    cue = torch.zeros(
        batch.event_keys.shape[0],
        model.text_dim,
        device=batch.event_keys.device,
    )
    for step in range(batch.event_keys.shape[1]):
        state, _ = model.write(
            state,
            batch.event_context_features[:, step],
            cue,
            batch.event_keys[:, step],
            batch.event_values[:, step],
            allocation_mode=allocation_mode,
            gate_mode="always",
            context_ids=batch.event_local_ids[:, step],
        )

    logits: list[Tensor] = []
    routes: list[Tensor] = []
    for step in range(batch.query_keys.shape[1]):
        features = batch.query_context_features[:, step]
        context_ids = batch.query_local_ids[:, step]
        logits.append(
            model.read(
                state,
                features,
                batch.query_keys[:, step],
                allocation_mode=allocation_mode,
                context_ids=context_ids,
            )
        )
        allocation = model.allocation(
            features,
            mode=allocation_mode,
            context_ids=context_ids,
        )
        routes.append(allocation.argmax(dim=-1))
    return torch.stack(logits, dim=1), torch.stack(routes, dim=1)


def measure(
    model: MemoryModel,
    logits: Tensor,
    routes: Tensor,
    batch: OnlineEpisodeBatch,
    *,
    active_contexts: int,
    events_per_context: int,
    num_blocks: int,
) -> OnlineMetrics:
    loss = F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten())
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == batch.targets).float().mean()

    candidate_index = batch.query_slots[:, None, :].expand(
        -1, active_contexts, -1
    )
    same_key_values = batch.all_values.gather(2, candidate_index)
    other_context = (
        torch.arange(active_contexts, device=predictions.device)[None, :, None]
        != batch.query_local_ids[:, None, :]
    )
    context_intrusions = (
        (predictions[:, None, :] == same_key_values) & other_context
    ).any(dim=1)

    context_routes = routes.reshape(
        routes.shape[0], active_contexts, events_per_context
    )
    primary_routes = context_routes[:, :, 0]
    routing_consistency = (
        context_routes == primary_routes[:, :, None]
    ).float().mean()
    used = F.one_hot(primary_routes, num_classes=num_blocks).bool().any(dim=1)
    unique_counts = used.sum(dim=-1)

    threshold = (
        float(model.threshold.detach().cpu())
        if isinstance(model, OnlinePrototypeMemory)
        else 0.0
    )
    return OnlineMetrics(
        loss=float(loss.detach().cpu()),
        accuracy=float(accuracy.detach().cpu()),
        context_intrusion_rate=float(context_intrusions.float().mean().detach().cpu()),
        unique_block_ratio=float((unique_counts / active_contexts).mean().detach().cpu()),
        collision_free_episodes=float(
            (unique_counts == active_contexts).float().mean().detach().cpu()
        ),
        routing_consistency=float(routing_consistency.detach().cpu()),
        novelty_threshold=threshold,
    )


def make_batch(
    args: argparse.Namespace,
    banks: OnlineTextBanks,
    split: str,
    generator: torch.Generator,
    device: torch.device,
) -> OnlineEpisodeBatch:
    return generate_online_episode_batch(
        banks=banks,
        split=split,  # type: ignore[arg-type]
        batch_size=args.batch_size,
        active_contexts=args.active_contexts,
        num_local_keys=args.num_local_keys,
        num_values=args.num_values,
        events_per_context=args.events_per_context,
        generator=generator,
    ).to(device)


@torch.no_grad()
def evaluate(
    model: MemoryModel,
    banks: OnlineTextBanks,
    args: argparse.Namespace,
    device: torch.device,
    variant: Variant,
    *,
    seed: int,
) -> OnlineMetrics:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    totals = torch.zeros(6)
    for _ in range(args.eval_batches):
        batch = make_batch(args, banks, "eval", generator, device)
        logits, routes = forward_episode(model, batch, variant)
        current = measure(
            model,
            logits,
            routes,
            batch,
            active_contexts=args.active_contexts,
            events_per_context=args.events_per_context,
            num_blocks=variant.num_blocks,
        )
        totals += torch.tensor(
            [
                current.loss,
                current.accuracy,
                current.context_intrusion_rate,
                current.unique_block_ratio,
                current.collision_free_episodes,
                current.routing_consistency,
            ]
        )
    totals /= args.eval_batches
    threshold = (
        float(model.threshold.detach().cpu())
        if isinstance(model, OnlinePrototypeMemory)
        else 0.0
    )
    return OnlineMetrics(
        *[float(value) for value in totals],
        novelty_threshold=threshold,
    )


def build_model(
    args: argparse.Namespace,
    banks: OnlineTextBanks,
    variant: Variant,
) -> MemoryModel:
    if variant.kind == "online":
        return OnlinePrototypeMemory(
            text_dim=banks.text_dim,
            num_keys=args.num_local_keys,
            num_values=args.num_values,
            num_blocks=variant.num_blocks,
            d_model=variant.d_model,
            prototype_dim=args.prototype_dim,
            initial_threshold=args.initial_threshold,
            retention=args.retention,
        )
    return SemanticFastWeightMemory(
        text_dim=banks.text_dim,
        num_contexts=args.active_contexts,
        num_keys=args.num_local_keys,
        num_values=args.num_values,
        num_blocks=variant.num_blocks,
        d_model=variant.d_model,
        router_dim=args.router_dim,
        temperature=args.temperature,
        retention=args.retention,
    )


def train_variant(
    args: argparse.Namespace,
    banks: OnlineTextBanks,
    device: torch.device,
    variant: Variant,
    *,
    seed: int,
) -> tuple[MemoryModel, OnlineMetrics]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = build_model(args, banks, variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(seed + 1)

    model.train()
    for _ in range(args.steps):
        batch = make_batch(args, banks, "train", generator, device)
        logits, _ = forward_episode(model, batch, variant)
        loss = F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return model, evaluate(
        model, banks, args, device, variant, seed=seed + 10_000
    )


def experiment_variants(args: argparse.Namespace) -> list[Variant]:
    if args.active_contexts > args.num_blocks:
        raise ValueError("this experiment requires one available block per active context")
    return [
        Variant("static", "static", args.num_blocks, args.block_dim),
        Variant("online", "online", args.num_blocks, args.online_block_dim),
        Variant("oracle", "oracle", args.num_blocks, args.online_block_dim),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test episode-local online block allocation")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("artifacts/bge_online_domains.pt"),
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--active-contexts", type=int, default=4)
    parser.add_argument("--num-local-keys", type=int, default=32)
    parser.add_argument("--num-values", type=int, default=64)
    parser.add_argument("--events-per-context", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--block-dim", type=int, default=12)
    parser.add_argument("--online-block-dim", type=int, default=11)
    parser.add_argument("--prototype-dim", type=int, default=16)
    parser.add_argument("--router-dim", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--initial-threshold", type=float, default=0.9)
    parser.add_argument("--retention", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("static", "online", "oracle"),
    )
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.embeddings.exists():
        raise SystemExit(
            f"missing {args.embeddings}; run prepare_online_embeddings.py first"
        )
    banks = load_online_text_banks(args.embeddings)
    device = choose_device(args.device)
    selected = [
        variant
        for variant in experiment_variants(args)
        if args.variants is None or variant.name in args.variants
    ]
    result: dict[str, object] = {
        "device": str(device),
        "num_domains": banks.num_domains,
        "config": {**vars(args), "embeddings": str(args.embeddings)},
        "chance_accuracy": 1.0 / args.num_values,
        "fast_capacity": args.num_blocks * args.block_dim * args.block_dim,
    }
    for variant in selected:
        model, metrics = train_variant(args, banks, device, variant, seed=args.seed)
        result[variant.name] = {
            **asdict(metrics),
            "fast_capacity": model.fast_capacity,
            "routing_state_size": (
                model.routing_state_size
                if isinstance(model, OnlinePrototypeMemory)
                else 0
            ),
            "total_runtime_state": model.fast_capacity
            + (
                model.routing_state_size
                if isinstance(model, OnlinePrototypeMemory)
                else 0
            ),
        }

    if args.summary:
        fields = (
            "accuracy",
            "context_intrusion_rate",
            "unique_block_ratio",
            "collision_free_episodes",
            "routing_consistency",
            "novelty_threshold",
        )
        print("fields=" + ",".join(fields))
        rows = []
        for variant in selected:
            values = [
                float(result[variant.name][field])  # type: ignore[index]
                for field in fields
            ]
            rows.append(
                f"{variant.name}=" + ",".join(f"{value:.6f}" for value in values)
            )
        print(";".join(rows))
    else:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
