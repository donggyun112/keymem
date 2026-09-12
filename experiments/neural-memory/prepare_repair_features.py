from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import torch

from neural_memory.claude_logs import stable_eval_split
from neural_memory.outcome_labels import hashed_text_features
from neural_memory.repair_chains import iter_repair_episodes


LABELS = {"ignore": 0, "strengthen": 1, "revise": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare repair-chain trace features")
    parser.add_argument("--root", type=Path, default=Path.home() / ".claude" / "projects")
    parser.add_argument("--output", type=Path, default=Path("artifacts/repair_chains.pt"))
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--max-per-action", type=int, default=4)
    parser.add_argument("--distractors", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = list(
        iter_repair_episodes(
            args.root,
            max_per_action=args.max_per_action,
            distractors=args.distractors,
        )
    )
    if not episodes:
        raise RuntimeError("no repair episodes found")
    max_candidates = max(len(episode.candidates) for episode in episodes)
    rows: dict[str, dict[str, list[torch.Tensor]]] = {
        split: {"candidates": [], "outcomes": [], "targets": [], "masks": []}
        for split in ("train", "eval")
    }
    all_rows: dict[str, list[torch.Tensor]] = {
        "candidates": [],
        "outcomes": [],
        "targets": [],
        "masks": [],
    }
    project_ids: list[int] = []
    cache: dict[str, torch.Tensor] = {}

    def encode(text: str) -> torch.Tensor:
        feature = cache.get(text)
        if feature is None:
            feature = hashed_text_features(text, dimension=args.feature_dim)
            cache[text] = feature
        return feature

    for episode in episodes:
        split = "eval" if stable_eval_split(episode.project) else "train"
        seed_text = f"{episode.project}:{episode.session_id}:{episode.family}"
        seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "big")
        candidates = list(episode.candidates)
        random.Random(seed).shuffle(candidates)
        feature_row = torch.zeros(max_candidates, args.feature_dim)
        target_row = torch.full((max_candidates,), -100, dtype=torch.long)
        mask_row = torch.zeros(max_candidates, dtype=torch.bool)
        for index, candidate in enumerate(candidates):
            feature_row[index] = encode(candidate.context)
            target_row[index] = LABELS[candidate.action]
            mask_row[index] = True
        rows[split]["candidates"].append(feature_row)
        rows[split]["outcomes"].append(encode(episode.outcome_context))
        rows[split]["targets"].append(target_row)
        rows[split]["masks"].append(mask_row)
        all_rows["candidates"].append(feature_row)
        all_rows["outcomes"].append(encode(episode.outcome_context))
        all_rows["targets"].append(target_row)
        all_rows["masks"].append(mask_row)
        project_ids.append(
            int.from_bytes(hashlib.sha256(episode.project.encode()).digest()[:8], "big")
            % (2**63 - 1)
        )

    output: dict[str, torch.Tensor] = {}
    for split in ("train", "eval"):
        if not rows[split]["candidates"]:
            raise RuntimeError(f"no {split} repair episodes found")
        for field, values in rows[split].items():
            output[f"{split}_{field}"] = torch.stack(values)
    for field, values in all_rows.items():
        output[f"all_{field}"] = torch.stack(values)
    output["all_project_ids"] = torch.tensor(project_ids, dtype=torch.long)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        f"saved train={len(output['train_targets'])},eval={len(output['eval_targets'])},"
        f"candidates={max_candidates},dim={args.feature_dim},raw_text=false to {args.output}"
    )


if __name__ == "__main__":
    main()
