from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from neural_memory import EpisodeBatch, FastWeightMemory, generate_episode_batch


@dataclass(frozen=True)
class Metrics:
    loss: float
    accuracy: float
    retained_gate: float
    distractor_gate: float


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def forward_episode(
    model: FastWeightMemory,
    batch: EpisodeBatch,
    *,
    force_gate: float | None,
) -> tuple[Tensor, Tensor]:
    state = model.initial_state(batch.event_keys.shape[0])
    gates: list[Tensor] = []
    for step in range(batch.event_keys.shape[1]):
        state, telemetry = model.write(
            state,
            batch.event_keys[:, step],
            batch.event_values[:, step],
            batch.evidence[:, step],
            force_gate=force_gate,
        )
        gates.append(telemetry.gate)

    logits = [model.read(state, batch.query_keys[:, step]) for step in range(batch.query_keys.shape[1])]
    return torch.stack(logits, dim=1), torch.stack(gates, dim=1)


def measure(logits: Tensor, gates: Tensor, batch: EpisodeBatch) -> Metrics:
    loss = F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten())
    accuracy = (logits.argmax(dim=-1) == batch.targets).float().mean()
    retained_gate = gates[batch.retained].mean()
    distractor_gate = gates[~batch.retained].mean()
    return Metrics(
        loss=float(loss.detach().cpu()),
        accuracy=float(accuracy.detach().cpu()),
        retained_gate=float(retained_gate.detach().cpu()),
        distractor_gate=float(distractor_gate.detach().cpu()),
    )


@torch.no_grad()
def evaluate(
    model: FastWeightMemory,
    args: argparse.Namespace,
    device: torch.device,
    *,
    force_gate: float | None,
    seed: int,
) -> Metrics:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    totals = torch.zeros(4)
    for _ in range(args.eval_batches):
        batch = generate_episode_batch(
            batch_size=args.batch_size,
            num_keys=args.num_keys,
            num_values=args.num_values,
            num_events=args.events,
            num_queries=args.queries,
            generator=generator,
        ).to(device)
        logits, gates = forward_episode(model, batch, force_gate=force_gate)
        current = measure(logits, gates, batch)
        totals += torch.tensor(
            [current.loss, current.accuracy, current.retained_gate, current.distractor_gate]
        )
    totals /= args.eval_batches
    return Metrics(*[float(value) for value in totals])


def train_variant(
    args: argparse.Namespace,
    device: torch.device,
    *,
    force_gate: float | None,
    seed: int,
) -> tuple[FastWeightMemory, Metrics]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = FastWeightMemory(
        num_keys=args.num_keys,
        num_values=args.num_values,
        d_model=args.d_model,
        retention=args.retention,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(seed + 1)

    model.train()
    for _ in range(args.steps):
        batch = generate_episode_batch(
            batch_size=args.batch_size,
            num_keys=args.num_keys,
            num_values=args.num_values,
            num_events=args.events,
            num_queries=args.queries,
            generator=generator,
        ).to(device)
        logits, _ = forward_episode(model, batch, force_gate=force_gate)
        loss = F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    metrics = evaluate(
        model,
        args,
        device,
        force_gate=force_gate,
        seed=seed + 10_000,
    )
    return model, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a graph-free fast-weight memory")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-keys", type=int, default=128)
    parser.add_argument("--num-values", type=int, default=128)
    parser.add_argument("--events", type=int, default=32)
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=24)
    parser.add_argument("--retention", type=float, default=0.995)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    _, learned = train_variant(args, device, force_gate=None, seed=args.seed)
    _, always_write = train_variant(args, device, force_gate=1.0, seed=args.seed)
    result = {
        "device": str(device),
        "config": vars(args),
        "chance_accuracy": 1.0 / args.num_values,
        "learned_gate": asdict(learned),
        "always_write": asdict(always_write),
        "accuracy_lift": learned.accuracy - always_write.accuracy,
        "gate_separation": learned.retained_gate - learned.distractor_gate,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

