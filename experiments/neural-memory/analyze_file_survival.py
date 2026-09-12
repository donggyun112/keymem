from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from neural_memory.file_survival import (
    find_change_survival,
    find_reverts,
    iter_version_chains,
    stratified_eval_sessions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit exact file-history revert outcomes")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.home() / ".claude" / "file-history",
    )
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chains = list(iter_version_chains(args.root))
    reverts = [event for chain in chains for event in find_reverts(chain)]
    survival = [event for chain in chains for event in find_change_survival(chain)]
    eval_sessions = stratified_eval_sessions(survival)
    train_survival = [event for event in survival if event.session_id not in eval_sessions]
    eval_survival = [event for event in survival if event.session_id in eval_sessions]
    version_counts = [len(chain.versions) for chain in chains]
    distances = [event.version_distance for event in reverts]
    no_op_versions = sum(
        left.digest == right.digest
        for chain in chains
        for left, right in zip(chain.versions, chain.versions[1:])
    )
    payload = {
        "raw_text_persisted": False,
        "chains": len(chains),
        "sessions": len({chain.session_id for chain in chains}),
        "versions": sum(version_counts),
        "multi_version_chains": sum(count >= 3 for count in version_counts),
        "no_op_versions": no_op_versions,
        "exact_reverts": len(reverts),
        "revert_sessions": len({event.session_id for event in reverts}),
        "median_revert_distance": statistics.median(distances) if distances else 0,
        "max_revert_distance": max(distances, default=0),
        "survival_examples": len(survival),
        "retained": sum(event.retained for event in survival),
        "reverted": sum(not event.retained for event in survival),
        "revert_label_sessions": len(
            {event.session_id for event in survival if not event.retained}
        ),
        "train_survival": len(train_survival),
        "train_retained": sum(event.retained for event in train_survival),
        "train_reverted": sum(not event.retained for event in train_survival),
        "eval_survival": len(eval_survival),
        "eval_retained": sum(event.retained for event in eval_survival),
        "eval_reverted": sum(not event.retained for event in eval_survival),
    }
    if args.summary:
        print(
            f"chains={payload['chains']},sessions={payload['sessions']},"
            f"versions={payload['versions']},multi_version={payload['multi_version_chains']},"
            f"no_op={payload['no_op_versions']},exact_reverts={payload['exact_reverts']},"
            f"revert_sessions={payload['revert_sessions']},"
            f"distance_p50={payload['median_revert_distance']},"
            f"distance_max={payload['max_revert_distance']},"
            f"survival_examples={payload['survival_examples']},"
            f"retained={payload['retained']},reverted={payload['reverted']},"
            f"revert_label_sessions={payload['revert_label_sessions']},"
            f"train={payload['train_survival']},train_retained={payload['train_retained']},"
            f"train_reverted={payload['train_reverted']},eval={payload['eval_survival']},"
            f"eval_retained={payload['eval_retained']},eval_reverted={payload['eval_reverted']}"
        )
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
