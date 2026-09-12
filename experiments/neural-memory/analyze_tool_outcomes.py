from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from neural_memory.claude_logs import stable_eval_split
from neural_memory.tool_outcomes import iter_verification_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit implicit verification outcomes")
    parser.add_argument(
        "--root", type=Path, default=Path.home() / ".claude" / "projects"
    )
    parser.add_argument("--recent-mutations", type=int, default=4)
    parser.add_argument("--include-subagents", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = list(
        iter_verification_events(
            args.root,
            include_subagents=args.include_subagents,
            recent_mutations=args.recent_mutations,
        )
    )
    train = [event for event in events if not stable_eval_split(event.project)]
    evaluation = [event for event in events if stable_eval_split(event.project)]
    labels = Counter("pass" if event.passed else "fail" for event in events)
    train_labels = Counter("pass" if event.passed else "fail" for event in train)
    eval_labels = Counter("pass" if event.passed else "fail" for event in evaluation)
    changed_train = [event for event in train if event.mutation_count > 0]
    changed_eval = [event for event in evaluation if event.mutation_count > 0]
    changed_train_labels = Counter("pass" if event.passed else "fail" for event in changed_train)
    changed_eval_labels = Counter("pass" if event.passed else "fail" for event in changed_eval)
    payload = {
        "raw_text_persisted": False,
        "events": len(events),
        "projects": len({event.project for event in events}),
        "sessions": len({event.session_id for event in events}),
        "labels": dict(labels),
        "with_mutations": sum(event.mutation_count > 0 for event in events),
        "train_events": len(train),
        "train_labels": dict(train_labels),
        "eval_events": len(evaluation),
        "eval_labels": dict(eval_labels),
        "changed_train_events": len(changed_train),
        "changed_train_labels": dict(changed_train_labels),
        "changed_eval_events": len(changed_eval),
        "changed_eval_labels": dict(changed_eval_labels),
    }
    if args.summary:
        print(
            f"events={len(events)},projects={payload['projects']},sessions={payload['sessions']},"
            f"pass={labels['pass']},fail={labels['fail']},"
            f"with_mutations={payload['with_mutations']},train={len(train)},"
            f"train_pass={train_labels['pass']},train_fail={train_labels['fail']},"
            f"eval={len(evaluation)},eval_pass={eval_labels['pass']},eval_fail={eval_labels['fail']},"
            f"changed_train={len(changed_train)},changed_train_pass={changed_train_labels['pass']},"
            f"changed_train_fail={changed_train_labels['fail']},changed_eval={len(changed_eval)},"
            f"changed_eval_pass={changed_eval_labels['pass']},changed_eval_fail={changed_eval_labels['fail']}"
        )
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
