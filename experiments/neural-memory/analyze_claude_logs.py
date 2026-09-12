from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from neural_memory.claude_logs import (
    iter_conversations,
    iter_history_conversations,
    iter_revisit_examples,
    stable_eval_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Claude logs for delayed-recall examples")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.home() / ".claude" / "projects",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path.home() / ".claude" / "history.jsonl",
    )
    parser.add_argument("--source", choices=("history", "projects"), default="history")
    parser.add_argument("--gap-hours", type=int, default=6)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--min-delay", type=int, default=2)
    parser.add_argument("--min-overlap", type=float, default=2.0)
    parser.add_argument("--min-margin", type=float, default=0.5)
    parser.add_argument("--include-subagents", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args()


def quantiles(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p90": 0.0, "max": 0.0}
    ordered = sorted(float(value) for value in values)
    return {
        "p50": statistics.median(ordered),
        "p90": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
        "max": ordered[-1],
    }


def main() -> None:
    args = parse_args()
    if args.source == "history":
        conversations = list(
            iter_history_conversations(
                args.history,
                gap_seconds=args.gap_hours * 60 * 60,
            )
        )
        source_path = args.history
    else:
        conversations = list(
            iter_conversations(args.root, include_subagents=args.include_subagents)
        )
        source_path = args.root
    examples = [
        example
        for conversation in conversations
        for example in iter_revisit_examples(
            conversation,
            candidates=args.candidates,
            min_delay=args.min_delay,
            min_overlap=args.min_overlap,
            min_margin=args.min_margin,
        )
    ]
    train_examples = [e for e in examples if not stable_eval_split(e.project)]
    eval_examples = [e for e in examples if stable_eval_split(e.project)]
    project_counts = Counter(conversation.project for conversation in conversations)
    payload = {
        "source": args.source,
        "source_path": str(source_path),
        "raw_text_persisted": False,
        "conversations": len(conversations),
        "projects": len(project_counts),
        "turns": sum(len(conversation.turns) for conversation in conversations),
        "turns_per_conversation": quantiles(
            [len(conversation.turns) for conversation in conversations]
        ),
        "weak_revisit_examples": len(examples),
        "train_examples": len(train_examples),
        "held_out_project_examples": len(eval_examples),
        "delay": quantiles([example.delay for example in examples]),
        "overlap": quantiles([example.overlap_score for example in examples]),
        "margin": quantiles([example.margin for example in examples]),
        "config": {
            "candidates": args.candidates,
            "min_delay": args.min_delay,
            "min_overlap": args.min_overlap,
            "min_margin": args.min_margin,
            "include_subagents": args.include_subagents,
            "gap_hours": args.gap_hours,
        },
    }
    if args.summary:
        print(
            "conversations={conversations},projects={projects},turns={turns},"
            "examples={weak_revisit_examples},train={train_examples},eval={held_out_project_examples},"
            "delay_p50={delay[p50]:.1f},delay_p90={delay[p90]:.1f},"
            "overlap_p50={overlap[p50]:.1f},margin_p50={margin[p50]:.1f}".format(
                **payload
            )
        )
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
