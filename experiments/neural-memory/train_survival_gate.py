from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from neural_memory.survival_memory import ChangeSurvivalGate


@dataclass(frozen=True)
class SurvivalMetrics:
    accuracy: float
    balanced_accuracy: float
    retained_recall: float
    reverted_recall: float
    reverted_precision: float
    roc_auc: float
    retained_strength: float
    reverted_strength: float


@torch.no_grad()
def measure(logits: Tensor, targets: Tensor) -> SurvivalMetrics:
    predictions = logits.argmax(dim=-1)
    recalls = []
    for label in (0, 1):
        mask = targets == label
        recalls.append(float((predictions[mask] == label).float().mean().cpu()))
    strength = DelayedOutcomeGate.retention(logits)
    retained = targets == 0
    reverted = targets == 1
    predicted_reverted = predictions == 1
    true_reverted = (predicted_reverted & reverted).sum()
    precision = true_reverted / predicted_reverted.sum().clamp(min=1)
    scores = logits[:, 1] - logits[:, 0]
    reverted_scores = scores[reverted]
    retained_scores = scores[retained]
    comparisons = reverted_scores[:, None] - retained_scores[None, :]
    auc = (comparisons.gt(0).float() + 0.5 * comparisons.eq(0).float()).mean()
    return SurvivalMetrics(
        accuracy=float((predictions == targets).float().mean().cpu()),
        balanced_accuracy=sum(recalls) / 2,
        retained_recall=recalls[0],
        reverted_recall=recalls[1],
        reverted_precision=float(precision.cpu()),
        roc_auc=float(auc.cpu()),
        retained_strength=float(strength[retained].mean().cpu()),
        reverted_strength=float(strength[reverted].mean().cpu()),
    )


def balanced_indices(
    targets: Tensor, batch_size: int, *, generator: torch.Generator
) -> Tensor:
    retained = torch.where(targets == 0)[0]
    reverted = torch.where(targets == 1)[0]
    half = batch_size // 2
    left = retained[
        torch.randint(len(retained), (half,), generator=generator, device=targets.device)
    ]
    right = reverted[
        torch.randint(
            len(reverted),
            (batch_size - half,),
            generator=generator,
            device=targets.device,
        )
    ]
    return torch.cat((left, right))


def similarity_baseline(
    train_requests: Tensor,
    train_feedback: Tensor,
    train_targets: Tensor,
    eval_requests: Tensor,
    eval_feedback: Tensor,
) -> Tensor:
    train_scores = F.cosine_similarity(train_requests, train_feedback)
    eval_scores = F.cosine_similarity(eval_requests, eval_feedback)
    candidates = torch.unique(train_scores).sort().values
    best_balanced = -1.0
    best_threshold = 0.0
    best_direction = 1.0
    for direction in (1.0, -1.0):
        for threshold in candidates:
            predictions = (direction * (train_scores - threshold) > 0).long()
            retained = predictions[train_targets == 0] == 0
            reverted = predictions[train_targets == 1] == 1
            balanced = float((retained.float().mean() + reverted.float().mean()) / 2)
            if balanced > best_balanced:
                best_balanced = balanced
                best_threshold = float(threshold)
                best_direction = direction
    negative_logit = best_direction * (eval_scores - best_threshold) * 10.0
    return torch.stack((-negative_logit, negative_logit), dim=-1)


def calibrate_revert_f1(train_logits: Tensor, train_targets: Tensor) -> float:
    scores = train_logits[:, 1] - train_logits[:, 0]
    best_f1 = -1.0
    best_threshold = 0.0
    for threshold in torch.unique(scores).sort().values:
        predicted = scores > threshold
        actual = train_targets == 1
        true_positive = (predicted & actual).sum().float()
        precision = true_positive / predicted.sum().clamp(min=1)
        recall = true_positive / actual.sum().clamp(min=1)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
        if float(f1) > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold


def shift_threshold(logits: Tensor, threshold: float) -> Tensor:
    shifted = logits.clone()
    shifted[:, 1] -= threshold
    return shifted


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2e-3)
    requests, feedback, targets = (tensor.to(device) for tensor in banks.split("train"))
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    model.train()
    for _ in range(steps):
        indices = balanced_indices(targets, batch_size, generator=generator)
        logits = model(requests[indices], feedback[indices], mode=mode)
        loss = F.cross_entropy(logits, targets[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


def train_relational(
    banks: OutcomeBanks,
    *,
    steps: int,
    batch_size: int,
    memory_dim: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> ChangeSurvivalGate:
    torch.manual_seed(seed)
    model = ChangeSurvivalGate(banks.feature_dim, memory_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2e-3)
    requests, feedback, targets = (tensor.to(device) for tensor in banks.split("train"))
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    model.train()
    for _ in range(steps):
        indices = balanced_indices(targets, batch_size, generator=generator)
        logits = model(requests[indices], feedback[indices])
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
) -> SurvivalMetrics:
    requests, feedback, targets = (tensor.to(device) for tensor in banks.split("eval"))
    model.eval()
    return measure(model(requests, feedback, mode=mode), targets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train delayed file-survival memory gate")
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/file_survival.pt")
    )
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--memory-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    banks = load_outcome_banks(args.features)
    device = torch.device(args.device)
    results: dict[str, SurvivalMetrics] = {}
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
    relational = train_relational(
        banks,
        steps=args.steps,
        batch_size=args.batch_size,
        memory_dim=args.memory_dim,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=device,
    )
    eval_requests, eval_feedback, eval_targets = (
        tensor.to(device) for tensor in banks.split("eval")
    )
    relational.eval()
    with torch.no_grad():
        train_requests_device, train_feedback_device, train_targets_device = (
            tensor.to(device) for tensor in banks.split("train")
        )
        train_relational_logits = relational(
            train_requests_device, train_feedback_device
        )
        eval_relational_logits = relational(eval_requests, eval_feedback)
        results["relational"] = measure(eval_relational_logits, eval_targets)
        threshold = calibrate_revert_f1(
            train_relational_logits, train_targets_device
        )
        results["relational_f1"] = measure(
            shift_threshold(eval_relational_logits, threshold), eval_targets
        )
    train_requests, train_feedback, train_targets = banks.split("train")
    eval_requests, eval_feedback, eval_targets = banks.split("eval")
    results["similarity"] = measure(
        similarity_baseline(
            train_requests,
            train_feedback,
            train_targets,
            eval_requests,
            eval_feedback,
        ),
        eval_targets,
    )
    majority = torch.zeros_like(banks.eval_targets)
    results["majority"] = measure(F.one_hot(majority, num_classes=2).float(), banks.eval_targets)
    fields = (
        "accuracy",
        "balanced_accuracy",
        "retained_recall",
        "reverted_recall",
        "reverted_precision",
        "roc_auc",
        "retained_strength",
        "reverted_strength",
    )
    if args.summary:
        print("fields=" + ",".join(fields))
        print(
            ";".join(
                name + "=" + ",".join(f"{getattr(value, field):.6f}" for field in fields)
                for name, value in results.items()
            )
        )
    else:
        print(f"train={len(banks.train_targets)},eval={len(banks.eval_targets)}")
        for name, value in results.items():
            print(name, value)


if __name__ == "__main__":
    main()
