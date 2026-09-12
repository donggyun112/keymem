from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from neural_memory import load_text_banks
from neural_memory.text_corpus import CUE_TEXTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check frozen embedding separability")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("artifacts/bge_small_en.pt"),
    )
    return parser.parse_args()


def centroid_accuracy(train: torch.Tensor, evaluation: torch.Tensor) -> tuple[float, float]:
    centroids = F.normalize(train.mean(dim=1), dim=-1)
    queries = F.normalize(evaluation.flatten(0, 1), dim=-1)
    labels = torch.arange(evaluation.shape[0]).repeat_interleave(evaluation.shape[1])
    similarities = queries @ centroids.T
    accuracy = (similarities.argmax(dim=-1) == labels).float().mean()
    correct = similarities.gather(1, labels[:, None]).squeeze(1)
    wrong = similarities.masked_fill(
        F.one_hot(labels, num_classes=centroids.shape[0]).bool(), -1.0
    ).max(dim=-1).values
    return float(accuracy), float((correct - wrong).mean())


def main() -> None:
    banks = load_text_banks(parse_args().embeddings)
    context_accuracy, context_margin = centroid_accuracy(
        banks.train_contexts, banks.eval_contexts
    )
    cue_accuracy, cue_margin = centroid_accuracy(banks.train_cues, banks.eval_cues)
    print(
        "context_centroid_accuracy="
        f"{context_accuracy:.6f},context_margin={context_margin:.6f},"
        f"cue_centroid_accuracy={cue_accuracy:.6f},cue_margin={cue_margin:.6f}"
    )
    cue_centroids = F.normalize(banks.train_cues.mean(dim=1), dim=-1)
    for cue_class, variants in enumerate(CUE_TEXTS["eval"]):
        similarities = F.normalize(banks.eval_cues[cue_class], dim=-1) @ cue_centroids.T
        for text, scores in zip(variants, similarities, strict=True):
            margin = float(scores[cue_class] - scores[1 - cue_class])
            print(f"cue_class={cue_class},margin={margin:.6f},text={text}")


if __name__ == "__main__":
    main()
