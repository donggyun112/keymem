from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neural_memory.claude_logs import stable_eval_split
from neural_memory.outcome_labels import hashed_text_features
from neural_memory.tool_outcomes import iter_verification_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare implicit test/build outcome features")
    parser.add_argument(
        "--root", type=Path, default=Path.home() / ".claude" / "projects"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/tool_outcomes.pt")
    )
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--recent-mutations", type=int, default=4)
    parser.add_argument("--require-mutations", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    events = iter_verification_events(
        args.root, recent_mutations=args.recent_mutations
    )
    for event in events:
        if args.require_mutations and event.mutation_count == 0:
            continue
        split = "eval" if stable_eval_split(event.project) else "train"
        rows[split]["requests"].append(encode(event.action_context))
        rows[split]["feedback"].append(encode(event.result))
        rows[split]["targets"].append(0 if event.passed else 1)

    output: dict[str, torch.Tensor] = {}
    for split in ("train", "eval"):
        if not rows[split]["requests"]:
            raise RuntimeError(f"no {split} verification events found")
        output[f"{split}_requests"] = torch.stack(rows[split]["requests"])  # type: ignore[arg-type]
        output[f"{split}_feedback"] = torch.stack(rows[split]["feedback"])  # type: ignore[arg-type]
        output[f"{split}_targets"] = torch.tensor(rows[split]["targets"], dtype=torch.long)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        f"saved train={len(output['train_targets'])},eval={len(output['eval_targets'])},"
        f"dim={args.feature_dim},require_mutations={str(args.require_mutations).lower()},"
        f"raw_text=false to {args.output}"
    )


if __name__ == "__main__":
    main()
