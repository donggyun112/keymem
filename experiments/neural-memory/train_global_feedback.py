from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from neural_memory.global_feedback import (
    GlobalFeedbackBanks,
    GlobalFeedbackSelector,
    load_global_feedback_banks,
)


@dataclass(frozen=True)
class SelectionMetrics:
    loss: float
    top1: float
    top2: float
    mean_reciprocal_rank: float


def metrics(logits: Tensor, targets: Tensor) -> SelectionMetrics:
    loss = F.cross_entropy(logits, targets)
    order = logits.argsort(dim=-1, descending=True)
    matches = order == targets[:, None]
    ranks = matches.float().argmax(dim=-1) + 1
    return SelectionMetrics(
        loss=float(loss.detach().cpu()),
        top1=float(matches[:, :1].any(dim=-1).float().mean().cpu()),
        top2=float(matches[:, :2].any(dim=-1).float().mean().cpu()),
        mean_reciprocal_rank=float((1.0 / ranks.float()).mean().cpu()),
    )


def baseline_logits(candidates: Tensor, feedback: Tensor, mode: str) -> Tensor:
    if mode == "cosine":
        return torch.einsum(
            "bnd,bd->bn", F.normalize(candidates, dim=-1), F.normalize(feedback, dim=-1)
        )
    if mode == "recency":
        return torch.arange(candidates.shape[1], device=candidates.device)[None].expand(
            len(candidates), -1
        ).float()
    if mode == "random":
        return torch.rand(candidates.shape[:2], device=candidates.device)
    raise ValueError(f"unknown baseline: {mode}")


def train(
    banks: GlobalFeedbackBanks,
    *,
    steps: int,
    batch_size: int,
    memory_dim: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> GlobalFeedbackSelector:
    torch.manual_seed(seed)
    model = GlobalFeedbackSelector(banks.text_dim, memory_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    candidates, feedback, targets, _ = banks.split("train")
    candidates, feedback, targets = (
        candidates.to(device),
        feedback.to(device),
        targets.to(device),
    )
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    model.train()
    for _ in range(steps):
        indices = torch.randint(
            len(targets), (batch_size,), generator=generator, device=device
        )
        logits = model(candidates[indices], feedback[indices])
        loss = F.cross_entropy(logits, targets[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


@torch.no_grad()
def evaluate(
    banks: GlobalFeedbackBanks,
    model: GlobalFeedbackSelector,
    *,
    device: torch.device,
) -> dict[str, SelectionMetrics]:
    candidates, feedback, targets, _ = (
        tensor.to(device) for tensor in banks.split("eval")
    )
    model.eval()
    result = {"learned": metrics(model(candidates, feedback), targets)}
    for mode in ("cosine", "recency", "random"):
        result[mode] = metrics(baseline_logits(candidates, feedback, mode), targets)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train global delayed-feedback trace selector")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("artifacts/claude_global_feedback.pt"),
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    banks = load_global_feedback_banks(args.embeddings)
    device = torch.device(args.device)
    model = train(
        banks,
        steps=args.steps,
        batch_size=args.batch_size,
        memory_dim=args.memory_dim,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=device,
    )
    result = evaluate(banks, model, device=device)
    payload = {
        "config": vars(args),
        "train_examples": len(banks.train_targets),
        "eval_examples": len(banks.eval_targets),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "metrics": {name: asdict(value) for name, value in result.items()},
    }
    if args.summary:
        fields = ("top1", "top2", "mean_reciprocal_rank")
        print("fields=" + ",".join(fields))
        print(
            ";".join(
                name + "=" + ",".join(f"{getattr(value, field):.6f}" for field in fields)
                for name, value in result.items()
            )
        )
    else:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
