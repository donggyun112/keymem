from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neural_memory.file_survival import (
    find_change_survival,
    iter_version_chains,
    stratified_eval_sessions,
)
from neural_memory.outcome_labels import hashed_text_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare delayed file-change survival features")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.home() / ".claude" / "file-history",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/file_survival.pt")
    )
    parser.add_argument("--feature-dim", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = [
        event
        for chain in iter_version_chains(args.root)
        for event in find_change_survival(chain)
    ]
    eval_sessions = stratified_eval_sessions(events)
    rows: dict[str, dict[str, list[torch.Tensor | int]]] = {
        "train": {"requests": [], "feedback": [], "targets": []},
        "eval": {"requests": [], "feedback": [], "targets": []},
    }
    cache: dict[str, torch.Tensor] = {}

    def encode(text: str) -> torch.Tensor:
        feature = cache.get(text)
        if feature is None:
            feature = hashed_text_features(text, dimension=args.feature_dim)
            cache[text] = feature
        return feature

    for event in events:
        split = "eval" if event.session_id in eval_sessions else "train"
        rows[split]["requests"].append(encode(event.change_text))
        rows[split]["feedback"].append(encode(event.outcome_text))
        rows[split]["targets"].append(0 if event.retained else 1)

    output: dict[str, torch.Tensor] = {}
    for split in ("train", "eval"):
        output[f"{split}_requests"] = torch.stack(rows[split]["requests"])  # type: ignore[arg-type]
        output[f"{split}_feedback"] = torch.stack(rows[split]["feedback"])  # type: ignore[arg-type]
        output[f"{split}_targets"] = torch.tensor(rows[split]["targets"], dtype=torch.long)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        f"saved train={len(output['train_targets'])},eval={len(output['eval_targets'])},"
        f"train_reverted={int(output['train_targets'].sum())},"
        f"eval_reverted={int(output['eval_targets'].sum())},dim={args.feature_dim},"
        f"raw_text=false to {args.output}"
    )


if __name__ == "__main__":
    main()
