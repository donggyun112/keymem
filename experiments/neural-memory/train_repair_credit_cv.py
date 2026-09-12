from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from neural_memory.repair_memory import DecoupledMultiTraceMemory, RepairBanks
from train_repair_credit import measure, train_decoupled_variant, train_variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leave-one-project-out repair credit evaluation")
    parser.add_argument("--features", type=Path, default=Path("artifacts/repair_chains.pt"))
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    parser.add_argument("--architecture", default="coupled", choices=("coupled", "decoupled"))
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    required = (
        "all_candidates",
        "all_outcomes",
        "all_targets",
        "all_masks",
        "all_project_ids",
    )
    if any(name not in payload for name in required):
        raise ValueError(
            "artifact lacks cross-validation tensors; rerun prepare_repair_features.py"
        )
    candidates = payload["all_candidates"]
    outcomes = payload["all_outcomes"]
    targets = payload["all_targets"]
    masks = payload["all_masks"]
    project_ids = payload["all_project_ids"]
    device = torch.device(args.device)
    predictions: dict[str, list[torch.Tensor]] = {
        name: []
        for name in ("candidate", "outcome", "joint", "joint_shuffled_outcome")
    }
    heldout_targets: list[torch.Tensor] = []
    heldout_masks: list[torch.Tensor] = []

    projects = project_ids.unique(sorted=True)
    for fold, project_id in enumerate(projects):
        eval_mask = project_ids == project_id
        train_mask = ~eval_mask
        banks = RepairBanks(
            train_candidates=candidates[train_mask],
            train_outcomes=outcomes[train_mask],
            train_targets=targets[train_mask],
            train_masks=masks[train_mask],
            eval_candidates=candidates[eval_mask],
            eval_outcomes=outcomes[eval_mask],
            eval_targets=targets[eval_mask],
            eval_masks=masks[eval_mask],
        )
        fold_models = {}
        for mode in ("candidate", "outcome", "joint"):
            trainer = train_decoupled_variant if args.architecture == "decoupled" else train_variant
            model = trainer(
                banks,
                mode=mode,  # type: ignore[arg-type]
                steps=args.steps,
                batch_size=args.batch_size,
                memory_dim=args.memory_dim,
                learning_rate=args.learning_rate,
                seed=args.seed + fold * 1009,
                device=device,
            )
            fold_models[mode] = model
            with torch.no_grad():
                raw_logits = model(
                    banks.eval_candidates.to(device),
                    banks.eval_outcomes.to(device),
                    mode=mode,  # type: ignore[arg-type]
                )
                logits = (
                    model.class_logits(*raw_logits)
                    if isinstance(model, DecoupledMultiTraceMemory)
                    else raw_logits
                )
            predictions[mode].append(logits.cpu())
        shuffled = banks.eval_outcomes.roll(1, dims=0)
        with torch.no_grad():
            raw_shuffled = fold_models["joint"](
                banks.eval_candidates.to(device), shuffled.to(device), mode="joint"
            )
            shuffled_logits = (
                fold_models["joint"].class_logits(*raw_shuffled)
                if isinstance(fold_models["joint"], DecoupledMultiTraceMemory)
                else raw_shuffled
            )
        predictions["joint_shuffled_outcome"].append(shuffled_logits.cpu())
        heldout_targets.append(banks.eval_targets)
        heldout_masks.append(banks.eval_masks)

    combined_targets = torch.cat(heldout_targets)
    combined_masks = torch.cat(heldout_masks)
    results = {
        name: measure(torch.cat(logits), combined_targets, combined_masks)
        for name, logits in predictions.items()
    }
    majority = torch.full((*combined_targets.shape, 3), -1.0)
    majority[..., 2] = 1.0
    results["majority_revise"] = measure(majority, combined_targets, combined_masks)
    output = {
        "config": vars(args),
        "projects": len(projects),
        "episodes": len(combined_targets),
        "labels": int(combined_masks.sum()),
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
        print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
