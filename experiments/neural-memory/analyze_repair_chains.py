from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from neural_memory.claude_logs import stable_eval_split
from neural_memory.codex_repair_chains import iter_codex_repair_episodes
from neural_memory.repair_chains import iter_repair_episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze retrospective verification repair chains")
    parser.add_argument("--root", type=Path, default=Path.home() / ".claude" / "projects")
    parser.add_argument(
        "--codex-root", type=Path, default=Path.home() / ".codex" / "sessions"
    )
    parser.add_argument("--max-per-action", type=int, default=4)
    parser.add_argument("--distractors", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    claude_episodes = list(
        iter_repair_episodes(
            args.root,
            max_per_action=args.max_per_action,
            distractors=args.distractors,
        )
    )
    codex_episodes = list(
        iter_codex_repair_episodes(
            args.codex_root,
            max_per_action=args.max_per_action,
            distractors=args.distractors,
        )
    )
    episodes = [*claude_episodes, *codex_episodes]
    labels = Counter(
        candidate.action for episode in episodes for candidate in episode.candidates
    )
    split_labels = {
        split: Counter(
            candidate.action
            for episode in episodes
            if stable_eval_split(episode.project) == (split == "eval")
            for candidate in episode.candidates
        )
        for split in ("train", "eval")
    }
    payload = {
        "episodes": len(episodes),
        "sources": {"claude": len(claude_episodes), "codex": len(codex_episodes)},
        "projects": len({episode.project for episode in episodes}),
        "sessions": len({episode.session_id for episode in episodes}),
        "train_episodes": sum(not stable_eval_split(episode.project) for episode in episodes),
        "eval_episodes": sum(stable_eval_split(episode.project) for episode in episodes),
        "families": Counter(episode.family for episode in episodes),
        "labels": labels,
        "split_labels": split_labels,
        "multi_action_episodes": sum(len(episode.candidates) > 2 for episode in episodes),
        "mean_candidates": statistics.fmean(len(episode.candidates) for episode in episodes)
        if episodes
        else 0.0,
        "mean_failed_attempts": statistics.fmean(episode.failed_attempts for episode in episodes)
        if episodes
        else 0.0,
        "raw_text_persisted": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
