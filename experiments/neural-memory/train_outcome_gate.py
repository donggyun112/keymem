from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from neural_memory.outcome_memory import (
    DelayedOutcomeGate,
    GateMode,
    OutcomeBanks,
    load_outcome_banks,
)


@dataclass(frozen=True)
class OutcomeMetrics:
    loss: float
    accuracy: float
    balanced_accuracy: float
    accept_recall: float
    correction_recall: float
    accept_retention: float
    correction_retention: float


@torch.no_grad()
def measure(logits: Tensor, targets: Tensor) -> OutcomeMetrics:
    predictions = logits.argmax(dim=-1)
    recalls = []
    for label in (0, 1):
        mask = targets == label
        recalls.append(float((predictions[mask] == label).float().mean().cpu()))
    retention = DelayedOutcomeGate.retention(logits)
    accept = targets == 0
    correction = targets == 1
    return OutcomeMetrics(
        loss=float(F.cross_entropy(logits, targets).cpu()),
        accuracy=float((predictions == targets).float().mean().cpu()),
        balanced_accuracy=sum(recalls) / 2,
        accept_recall=recalls[0],
        correction_recall=recalls[1],
        accept_retention=float(retention[accept].mean().cpu()),
        correction_retention=float(retention[correction].mean().cpu()),
    )


def train_variant(
    banks: OutcomeBanks,
    *,
    mode: GateMode,
    steps: int,
    batch_size: int,
    memory_dim: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> DelayedOutcomeGate:
    torch.manual_seed(seed)
    model = DelayedOutcomeGate(banks.feature_dim, memory_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    requests, feedback, targets = (tensor.to(device) for tensor in banks.split("train"))
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    model.train()
    for _ in range(steps):
        indices = torch.randint(
            len(targets), (batch_size,), generator=generator, device=device
        )
        logits = model(requests[indices], feedback[indices], mode=mode)
        loss = F.cross_entropy(logits, targets[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


@torch.no_grad()
def evaluate(
    banks: OutcomeBanks,
    model: DelayedOutcomeGate,
    *,
    mode: GateMode,
    device: torch.device,
) -> OutcomeMetrics:
    requests, feedback, targets = (tensor.to(device) for tensor in banks.split("eval"))
    model.eval()
    return measure(model(requests, feedback, mode=mode), targets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train delayed strengthen/revise outcome gates")
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/claude_outcomes.pt")
    )
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--memory-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    banks = load_outcome_banks(args.features)
    device = torch.device(args.device)
    results: dict[str, OutcomeMetrics] = {}
    parameters = 0
    for mode in ("request", "feedback", "joint"):
        model = train_variant(
            banks,
            mode=mode,  # type: ignore[arg-type]
            steps=args.steps,
            batch_size=args.batch_size,
            memory_dim=args.memory_dim,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=device,
        )
        results[mode] = evaluate(
            banks, model, mode=mode, device=device  # type: ignore[arg-type]
        )
        parameters = sum(parameter.numel() for parameter in model.parameters())
    majority = torch.zeros_like(banks.eval_targets)
    majority_logits = F.one_hot(majority, num_classes=2).float()
    results["majority"] = measure(majority_logits, banks.eval_targets)

    payload = {
        "config": vars(args),
        "train_examples": len(banks.train_targets),
        "eval_examples": len(banks.eval_targets),
        "parameters": parameters,
        "metrics": {name: asdict(value) for name, value in results.items()},
    }
    if args.summary:
        fields = (
            "accuracy",
            "balanced_accuracy",
            "accept_recall",
            "correction_recall",
            "accept_retention",
            "correction_retention",
        )
        print("fields=" + ",".join(fields))
        print(
            ";".join(
                name + "=" + ",".join(f"{getattr(value, field):.6f}" for field in fields)
                for name, value in results.items()
            )
        )
    else:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
