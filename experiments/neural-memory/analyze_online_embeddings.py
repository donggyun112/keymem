from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from neural_memory import load_online_text_banks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure online context similarity margins")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("artifacts/bge_online_domains.pt"),
    )
    return parser.parse_args()


def main() -> None:
    banks = load_online_text_banks(parse_args().embeddings)
    train = F.normalize(banks.train_contexts, dim=-1)
    evaluation = F.normalize(banks.eval_contexts, dim=-1)
    similarities = torch.einsum("ivh,jwh->ivjw", train, evaluation)
    domain_count = train.shape[0]
    same_mask = torch.eye(domain_count, dtype=torch.bool)[:, None, :, None]
    same = similarities.expand(-1, -1, -1, evaluation.shape[1])[same_mask.expand_as(similarities)]
    different = similarities[~same_mask.expand_as(similarities)]
    print(
        f"same_mean={same.mean():.6f},same_min={same.min():.6f},"
        f"different_mean={different.mean():.6f},different_max={different.max():.6f}"
    )


if __name__ == "__main__":
    main()
