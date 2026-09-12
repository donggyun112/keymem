from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neural_memory.claude_logs import (
    iter_conversations,
    iter_history_conversations,
    iter_revisit_examples,
    stable_eval_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare privacy-local Claude log traces")
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
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/claude_global_feedback.pt"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--min-delay", type=int, default=2)
    parser.add_argument("--min-overlap", type=float, default=2.0)
    parser.add_argument("--min-margin", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    from sentence_transformers import SentenceTransformer

    args = parse_args()
    if args.source == "history":
        conversations = list(
            iter_history_conversations(
                args.history,
                gap_seconds=args.gap_hours * 60 * 60,
            )
        )
    else:
        conversations = list(iter_conversations(args.root))
    texts = [turn for conversation in conversations for turn in conversation.turns]
    offsets: list[int] = []
    offset = 0
    for conversation in conversations:
        offsets.append(offset)
        offset += len(conversation.turns)

    encoder = SentenceTransformer(args.model, device=args.device)
    embeddings = encoder.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).cpu()

    rows: dict[str, dict[str, list[torch.Tensor | int]]] = {
        "train": {"candidates": [], "feedback": [], "targets": [], "delays": []},
        "eval": {"candidates": [], "feedback": [], "targets": [], "delays": []},
    }
    for conversation, base in zip(conversations, offsets, strict=True):
        split = "eval" if stable_eval_split(conversation.project) else "train"
        for example in iter_revisit_examples(
            conversation,
            candidates=args.candidates,
            min_delay=args.min_delay,
            min_overlap=args.min_overlap,
            min_margin=args.min_margin,
        ):
            rows[split]["candidates"].append(
                embeddings[[base + index for index in example.candidate_indices]]
            )
            rows[split]["feedback"].append(embeddings[base + example.query_index])
            rows[split]["targets"].append(example.target_offset)
            rows[split]["delays"].append(example.delay)

    output: dict[str, torch.Tensor] = {}
    for split in ("train", "eval"):
        if not rows[split]["candidates"]:
            raise RuntimeError(f"no {split} examples passed weak-label filters")
        output[f"{split}_candidates"] = torch.stack(rows[split]["candidates"])  # type: ignore[arg-type]
        output[f"{split}_feedback"] = torch.stack(rows[split]["feedback"])  # type: ignore[arg-type]
        output[f"{split}_targets"] = torch.tensor(rows[split]["targets"], dtype=torch.long)
        output[f"{split}_delays"] = torch.tensor(rows[split]["delays"], dtype=torch.long)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        f"saved train={len(output['train_targets'])},eval={len(output['eval_targets'])},"
        f"candidates={args.candidates},dim={embeddings.shape[-1]},raw_text=false to {args.output}"
    )


if __name__ == "__main__":
    main()
