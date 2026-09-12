from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
DEFAULT_EVENT_PATH = ROOT / "local-data" / "prospective-events.jsonl"
sys.path.insert(0, str(ROOT / "neural_memory"))
from prospective import iter_credit_examples, load_events  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize privacy-preserving prospective outcomes"
    )
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENT_PATH)
    parser.add_argument("--examples", type=Path, help="optional derived JSONL output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = load_events(args.events)
    examples = list(iter_credit_examples(events))
    if args.examples:
        args.examples.parent.mkdir(parents=True, exist_ok=True)
        with args.examples.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example.__dict__, sort_keys=True) + "\n")
    summary = {
        "events": len(events),
        "sessions": len({event["session_id"] for event in events}),
        "event_kinds": Counter(str(event.get("kind")) for event in events),
        "credit_examples": len(examples),
        "labels": Counter(example.label for example in examples),
        "sources": Counter(example.source for example in examples),
        "multi_action_examples": sum(len(example.action_ids) > 1 for example in examples),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
