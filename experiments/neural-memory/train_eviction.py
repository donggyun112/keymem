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
    EvictionEpisodeBatch,
    FrozenTextBanks,
    OnlinePrototypeMemory,
    OnlineTextBanks,
    PriorityEvictionMemory,
    generate_eviction_episode_batch,
    load_online_text_banks,
    load_text_banks,
)
from train import choose_device


VariantKind = Literal["fifo", "learned", "oracle"]
EvictionModel = OnlinePrototypeMemory | PriorityEvictionMemory


@dataclass(frozen=True)
class Variant:
    name: str
    kind: VariantKind


@dataclass(frozen=True)
class EvictionMetrics:
    loss: float
    accuracy: float
    useful_coverage: float
    stored_useful_fraction: float
    useful_accept_rate: float
    distractor_accept_rate: float
    useful_priority: float
    distractor_priority: float


def forward_episode(
    model: EvictionModel,
    batch: EvictionEpisodeBatch,
    variant: Variant,
    *,
    num_blocks: int,
    events_per_context: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    state = model.initial_state(batch.event_keys.shape[0])
    block_labels = torch.full(
        (batch.event_keys.shape[0], num_blocks),
        -1,
        dtype=torch.long,
        device=batch.event_keys.device,
    )
    accepted_events: list[Tensor] = []
    priority_events: list[Tensor] = []
    for step in range(batch.event_keys.shape[1]):
        if isinstance(model, PriorityEvictionMemory):
            priority_mode = "learned" if variant.kind == "learned" else "oracle"
            state, telemetry = model.write(
                state,
                batch.event_context_features[:, step],
                batch.event_cue_features[:, step],
                batch.event_keys[:, step],
                batch.event_values[:, step],
                priority_mode=priority_mode,
                should_retain=batch.event_should_retain[:, step],
            )
            accepted = telemetry.accepted
            matched = telemetry.matched_existing
            priority = telemetry.incoming_priority
            route = telemetry.route
        else:
            state, telemetry = model.write(
                state,
                batch.event_context_features[:, step],
                batch.event_keys[:, step],
                batch.event_values[:, step],
            )
            accepted = torch.ones_like(telemetry.matched_existing)
            matched = telemetry.matched_existing
            priority = torch.zeros_like(accepted, dtype=state.weights.dtype)
            route = telemetry.route

        new_assignment = accepted & ~matched
        route_mask = F.one_hot(route, num_classes=num_blocks).bool()
        update_mask = new_assignment[:, None] & route_mask
        labels = batch.event_local_ids[:, step, None].expand(-1, num_blocks)
        block_labels = torch.where(update_mask, labels, block_labels)
        accepted_events.append(accepted)
        priority_events.append(priority)

    logits = [
        model.read(
            state,
            batch.query_context_features[:, step],
            batch.query_keys[:, step],
        )[0]
        for step in range(batch.query_keys.shape[1])
    ]
    return (
        torch.stack(logits, dim=1),
        torch.stack(accepted_events, dim=1),
        torch.stack(priority_events, dim=1),
        block_labels,
    )


def measure(
    logits: Tensor,
    accepted: Tensor,
    priorities: Tensor,
    block_labels: Tensor,
    batch: EvictionEpisodeBatch,
    *,
    retained_contexts: int,
    events_per_context: int,
) -> EvictionMetrics:
    loss = F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten())
    accuracy = (logits.argmax(dim=-1) == batch.targets).float().mean()

    valid_labels = block_labels >= 0
    safe_labels = block_labels.clamp_min(0)
    block_is_useful = batch.retained_contexts.gather(1, safe_labels) & valid_labels
    stored_useful_fraction = block_is_useful.float().sum(dim=-1) / valid_labels.float().sum(
        dim=-1
    ).clamp_min(1.0)

    useful_ids = batch.retained_contexts.nonzero(as_tuple=False)[:, 1].reshape(
        batch.retained_contexts.shape[0], retained_contexts
    )
    coverage = (block_labels[:, :, None] == useful_ids[:, None, :]).any(dim=1)

    first_events = torch.arange(
        0, accepted.shape[1], events_per_context, device=accepted.device
    )
    first_accept = accepted[:, first_events]
    first_priority = priorities[:, first_events]
    first_useful = batch.event_should_retain[:, first_events]
    return EvictionMetrics(
        loss=float(loss.detach().cpu()),
        accuracy=float(accuracy.detach().cpu()),
        useful_coverage=float(coverage.float().mean().detach().cpu()),
        stored_useful_fraction=float(stored_useful_fraction.mean().detach().cpu()),
        useful_accept_rate=float(first_accept[first_useful].float().mean().detach().cpu()),
        distractor_accept_rate=float(
            first_accept[~first_useful].float().mean().detach().cpu()
        ),
        useful_priority=float(first_priority[first_useful].mean().detach().cpu()),
        distractor_priority=float(first_priority[~first_useful].mean().detach().cpu()),
    )


def make_batch(
    args: argparse.Namespace,
    context_banks: OnlineTextBanks,
    cue_banks: FrozenTextBanks,
    split: str,
    generator: torch.Generator,
    device: torch.device,
) -> EvictionEpisodeBatch:
    return generate_eviction_episode_batch(
        context_banks=context_banks,
        cue_banks=cue_banks,
        split=split,  # type: ignore[arg-type]
        batch_size=args.batch_size,
        active_contexts=args.active_contexts,
        retained_contexts=args.retained_contexts,
        num_local_keys=args.num_local_keys,
        num_values=args.num_values,
        events_per_context=args.events_per_context,
        generator=generator,
    ).to(device)


@torch.no_grad()
def evaluate(
    model: EvictionModel,
    context_banks: OnlineTextBanks,
    cue_banks: FrozenTextBanks,
    args: argparse.Namespace,
    device: torch.device,
    variant: Variant,
    *,
    seed: int,
) -> EvictionMetrics:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    totals = torch.zeros(8)
    for _ in range(args.eval_batches):
        batch = make_batch(args, context_banks, cue_banks, "eval", generator, device)
        outputs = forward_episode(
            model,
            batch,
            variant,
            num_blocks=args.num_blocks,
            events_per_context=args.events_per_context,
        )
        current = measure(
            *outputs,
            batch,
            retained_contexts=args.retained_contexts,
            events_per_context=args.events_per_context,
        )
        totals += torch.tensor(list(asdict(current).values()))
    totals /= args.eval_batches
    return EvictionMetrics(*[float(value) for value in totals])


def build_model(
    args: argparse.Namespace,
    text_dim: int,
    variant: Variant,
) -> EvictionModel:
    common = dict(
        text_dim=text_dim,
        num_keys=args.num_local_keys,
        num_values=args.num_values,
        num_blocks=args.num_blocks,
        d_model=args.block_dim,
        prototype_dim=args.prototype_dim,
        initial_threshold=args.initial_threshold,
        retention=args.retention,
    )
    if variant.kind == "fifo":
        return OnlinePrototypeMemory(**common)
    return PriorityEvictionMemory(**common)


def train_variant(
    args: argparse.Namespace,
    context_banks: OnlineTextBanks,
    cue_banks: FrozenTextBanks,
    device: torch.device,
    variant: Variant,
    *,
    seed: int,
) -> tuple[EvictionModel, EvictionMetrics]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = build_model(args, context_banks.text_dim, variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(seed + 1)

    model.train()
    for _ in range(args.steps):
        batch = make_batch(args, context_banks, cue_banks, "train", generator, device)
        logits, _, _, _ = forward_episode(
            model,
            batch,
            variant,
            num_blocks=args.num_blocks,
            events_per_context=args.events_per_context,
        )
        loss = F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    metrics = evaluate(
        model,
        context_banks,
        cue_banks,
        args,
        device,
        variant,
        seed=seed + 10_000,
    )
    return model, metrics


def variants(args: argparse.Namespace) -> list[Variant]:
    if args.retained_contexts > args.num_blocks:
        raise ValueError("retained_contexts cannot exceed available blocks")
    return [
        Variant("fifo", "fifo"),
        Variant("learned", "learned"),
        Variant("oracle", "oracle"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train utility-aware online eviction")
    parser.add_argument(
        "--context-embeddings",
        type=Path,
        default=Path("artifacts/bge_online_domains.pt"),
    )
    parser.add_argument(
        "--cue-embeddings",
        type=Path,
        default=Path("artifacts/bge_small_en.pt"),
    )
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--active-contexts", type=int, default=6)
    parser.add_argument("--retained-contexts", type=int, default=4)
    parser.add_argument("--num-local-keys", type=int, default=32)
    parser.add_argument("--num-values", type=int, default=64)
    parser.add_argument("--events-per-context", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--block-dim", type=int, default=11)
    parser.add_argument("--prototype-dim", type=int, default=16)
    parser.add_argument("--initial-threshold", type=float, default=0.9)
    parser.add_argument("--retention", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("fifo", "learned", "oracle"),
    )
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.context_embeddings, args.cue_embeddings):
        if not path.exists():
            raise SystemExit(f"missing embedding artifact: {path}")
    context_banks = load_online_text_banks(args.context_embeddings)
    cue_banks = load_text_banks(args.cue_embeddings)
    device = choose_device(args.device)
    selected = [
        variant
        for variant in variants(args)
        if args.variants is None or variant.name in args.variants
    ]
    result: dict[str, object] = {
        "device": str(device),
        "config": {
            **vars(args),
            "context_embeddings": str(args.context_embeddings),
            "cue_embeddings": str(args.cue_embeddings),
        },
        "chance_accuracy": 1.0 / args.num_values,
    }
    for variant in selected:
        model, metrics = train_variant(
            args, context_banks, cue_banks, device, variant, seed=args.seed
        )
        result[variant.name] = {
            **asdict(metrics),
            "association_state": model.fast_capacity,
            "routing_state": model.routing_state_size,
            "total_runtime_state": model.fast_capacity + model.routing_state_size,
        }

    if args.summary:
        fields = (
            "accuracy",
            "useful_coverage",
            "stored_useful_fraction",
            "useful_accept_rate",
            "distractor_accept_rate",
            "useful_priority",
            "distractor_priority",
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
