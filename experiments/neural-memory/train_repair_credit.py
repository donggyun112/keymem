from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from neural_memory.repair_memory import (
    CreditMode,
    DecoupledMultiTraceMemory,
    MultiTraceActionMemory,
    RepairBanks,
    load_repair_banks,
)


@dataclass(frozen=True)
class CreditMetrics:
    accuracy: float
    balanced_accuracy: float
    ignore_recall: float
    strengthen_recall: float
    revise_recall: float
    selection_precision: float
    selection_recall: float
    selected_action_accuracy: float
    episode_exact: float


@torch.no_grad()
def measure(logits: Tensor, targets: Tensor, masks: Tensor) -> CreditMetrics:
    predictions = logits.argmax(dim=-1)
    valid_predictions = predictions[masks]
    valid_targets = targets[masks]
    recalls: list[float] = []
    for label in range(3):
        label_mask = valid_targets == label
        recalls.append(float((valid_predictions[label_mask] == label).float().mean().cpu()))
    predicted_selected = valid_predictions != 0
    target_selected = valid_targets != 0
    true_selected = predicted_selected & target_selected
    precision = true_selected.sum() / predicted_selected.sum().clamp_min(1)
    recall = true_selected.sum() / target_selected.sum().clamp_min(1)
    selected_action = (
        (valid_predictions[target_selected] == valid_targets[target_selected]).float().mean()
    )
    correct_slots = (predictions == targets) | ~masks
    return CreditMetrics(
        accuracy=float((valid_predictions == valid_targets).float().mean().cpu()),
        balanced_accuracy=sum(recalls) / 3,
        ignore_recall=recalls[0],
        strengthen_recall=recalls[1],
        revise_recall=recalls[2],
        selection_precision=float(precision.cpu()),
        selection_recall=float(recall.cpu()),
        selected_action_accuracy=float(selected_action.cpu()),
        episode_exact=float(correct_slots.all(dim=-1).float().mean().cpu()),
    )


def train_variant(
    banks: RepairBanks,
    *,
    mode: CreditMode,
    steps: int,
    batch_size: int,
    memory_dim: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> MultiTraceActionMemory:
    torch.manual_seed(seed)
    model = MultiTraceActionMemory(banks.feature_dim, memory_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    candidates, outcomes, targets, masks = (
        tensor.to(device) for tensor in banks.split("train")
    )
    counts = torch.bincount(targets[masks], minlength=3).float()
    weights = counts.sum() / counts.clamp_min(1)
    weights /= weights.mean()
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    model.train()
    for _ in range(steps):
        indices = torch.randint(
            len(targets), (batch_size,), generator=generator, device=device
        )
        logits = model(candidates[indices], outcomes[indices], mode=mode)
        loss = F.cross_entropy(
            logits.flatten(0, 1),
            targets[indices].flatten(),
            weight=weights,
            ignore_index=-100,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


def train_decoupled_variant(
    banks: RepairBanks,
    *,
    mode: CreditMode,
    steps: int,
    batch_size: int,
    memory_dim: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> DecoupledMultiTraceMemory:
    torch.manual_seed(seed)
    model = DecoupledMultiTraceMemory(banks.feature_dim, memory_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    candidates, outcomes, targets, masks = (
        tensor.to(device) for tensor in banks.split("train")
    )
    selected = targets[masks] != 0
    selection_weight = (~selected).sum() / selected.sum().clamp_min(1)
    action_counts = torch.bincount(targets[masks][selected] - 1, minlength=2).float()
    action_weights = action_counts.sum() / action_counts.clamp_min(1)
    action_weights /= action_weights.mean()
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    model.train()
    for _ in range(steps):
        indices = torch.randint(
            len(targets), (batch_size,), generator=generator, device=device
        )
        batch_targets = targets[indices]
        batch_masks = masks[indices]
        selection_logits, action_logits = model(
            candidates[indices], outcomes[indices], mode=mode
        )
        valid_selection = selection_logits[batch_masks]
        selection_targets = (batch_targets[batch_masks] != 0).float()
        selection_loss = F.binary_cross_entropy_with_logits(
            valid_selection,
            selection_targets,
            pos_weight=selection_weight,
        )
        action_mask = batch_masks & (batch_targets != 0)
        action_loss = F.cross_entropy(
            action_logits[action_mask], batch_targets[action_mask] - 1, weight=action_weights
        )
        loss = selection_loss + action_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


@torch.no_grad()
def evaluate(
    banks: RepairBanks,
    model: MultiTraceActionMemory,
    *,
    mode: CreditMode,
    device: torch.device,
    shuffle_outcomes: bool = False,
) -> CreditMetrics:
    candidates, outcomes, targets, masks = (
        tensor.to(device) for tensor in banks.split("eval")
    )
    if shuffle_outcomes and len(outcomes) > 1:
        outcomes = outcomes.roll(1, dims=0)
    model.eval()
    return measure(model(candidates, outcomes, mode=mode), targets, masks)


@torch.no_grad()
def evaluate_decoupled(
    banks: RepairBanks,
    model: DecoupledMultiTraceMemory,
    *,
    mode: CreditMode,
    device: torch.device,
    shuffle_outcomes: bool = False,
) -> CreditMetrics:
    candidates, outcomes, targets, masks = (
        tensor.to(device) for tensor in banks.split("eval")
    )
    if shuffle_outcomes and len(outcomes) > 1:
        outcomes = outcomes.roll(1, dims=0)
    model.eval()
    selection, action = model(candidates, outcomes, mode=mode)
    return measure(model.class_logits(selection, action), targets, masks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-trace delayed action credit")
    parser.add_argument("--features", type=Path, default=Path("artifacts/repair_chains.pt"))
    parser.add_argument("--steps", type=int, default=1_500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    banks = load_repair_banks(args.features)
    device = torch.device(args.device)
    results: dict[str, CreditMetrics] = {}
    models: dict[str, MultiTraceActionMemory] = {}
    for mode in ("candidate", "outcome", "joint"):
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
        models[mode] = model
        results[mode] = evaluate(
            banks, model, mode=mode, device=device  # type: ignore[arg-type]
        )
    results["joint_shuffled_outcome"] = evaluate(
        banks,
        models["joint"],
        mode="joint",
        device=device,
        shuffle_outcomes=True,
    )
    targets = banks.eval_targets
    masks = banks.eval_masks
    majority = torch.full((*targets.shape, 3), -1.0)
    majority[..., 2] = 1.0
    results["majority_revise"] = measure(majority, targets, masks)
    payload = {
        "config": vars(args),
        "train_episodes": len(banks.train_targets),
        "eval_episodes": len(banks.eval_targets),
        "train_labels": int(banks.train_masks.sum()),
        "eval_labels": int(banks.eval_masks.sum()),
        "parameters": sum(parameter.numel() for parameter in models["joint"].parameters()),
        "metrics": {name: asdict(value) for name, value in results.items()},
    }
    if args.summary:
        fields = (
            "accuracy",
            "balanced_accuracy",
            "selection_precision",
            "selection_recall",
            "selected_action_accuracy",
            "episode_exact",
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
