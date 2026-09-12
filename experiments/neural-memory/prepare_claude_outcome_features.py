from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neural_memory.claude_logs import iter_history_conversations, stable_eval_split
from neural_memory.outcome_labels import hashed_text_features, iter_outcome_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare privacy-local delayed outcome features")
    parser.add_argument(
        "--history",
        type=Path,
        default=Path.home() / ".claude" / "history.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/claude_outcomes.pt"),
    )
    parser.add_argument("--gap-hours", type=int, default=6)
    parser.add_argument("--feature-dim", type=int, default=512)
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

    conversations = iter_history_conversations(
        args.history,
        gap_seconds=args.gap_hours * 60 * 60,
        min_chars=1,
    )
    for conversation in conversations:
        split = "eval" if stable_eval_split(conversation.project) else "train"
        for event in iter_outcome_events(conversation):
            rows[split]["requests"].append(encode(conversation.turns[event.request_index]))
            rows[split]["feedback"].append(encode(conversation.turns[event.feedback_index]))
            rows[split]["targets"].append(0 if event.label == "accept" else 1)

    output: dict[str, torch.Tensor] = {}
    for split in ("train", "eval"):
        if not rows[split]["requests"]:
            raise RuntimeError(f"no {split} outcome events found")
        output[f"{split}_requests"] = torch.stack(rows[split]["requests"])  # type: ignore[arg-type]
        output[f"{split}_feedback"] = torch.stack(rows[split]["feedback"])  # type: ignore[arg-type]
        output[f"{split}_targets"] = torch.tensor(rows[split]["targets"], dtype=torch.long)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        f"saved train={len(output['train_targets'])},eval={len(output['eval_targets'])},"
        f"dim={args.feature_dim},raw_text=false to {args.output}"
    )


if __name__ == "__main__":
    main()
