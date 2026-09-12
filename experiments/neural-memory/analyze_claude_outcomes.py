from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from neural_memory.claude_logs import iter_history_conversations, stable_eval_split
from neural_memory.outcome_labels import iter_outcome_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit explicit delayed outcomes in Claude history")
    parser.add_argument(
        "--history",
        type=Path,
        default=Path.home() / ".claude" / "history.jsonl",
    )
    parser.add_argument("--gap-hours", type=int, default=6)
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conversations = list(
        iter_history_conversations(
            args.history,
            gap_seconds=args.gap_hours * 60 * 60,
            min_chars=1,
        )
    )
    events = [
        event
        for conversation in conversations
        for event in iter_outcome_events(conversation)
    ]
    train = [event for event in events if not stable_eval_split(event.project)]
    evaluation = [event for event in events if stable_eval_split(event.project)]
    counts = Counter(event.label for event in events)
    train_counts = Counter(event.label for event in train)
    eval_counts = Counter(event.label for event in evaluation)
    payload = {
        "raw_text_persisted": False,
        "conversations": len(conversations),
        "turns": sum(len(conversation.turns) for conversation in conversations),
        "events": len(events),
        "labels": dict(counts),
        "train_events": len(train),
        "train_labels": dict(train_counts),
        "eval_events": len(evaluation),
        "eval_labels": dict(eval_counts),
    }
    if args.summary:
        print(
            f"conversations={payload['conversations']},turns={payload['turns']},"
            f"events={payload['events']},accept={counts['accept']},correct={counts['correct']},"
            f"train={len(train)},train_accept={train_counts['accept']},"
            f"train_correct={train_counts['correct']},eval={len(evaluation)},"
            f"eval_accept={eval_counts['accept']},eval_correct={eval_counts['correct']}"
        )
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
