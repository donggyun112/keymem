from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neural_memory.online_corpus import ONLINE_CONTEXT_TEXTS, online_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare frozen online-domain embeddings")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/bge_online_domains.pt"),
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    from sentence_transformers import SentenceTransformer

    args = parse_args()
    texts, groups = online_texts()
    encoder = SentenceTransformer(args.model, device=args.device)
    embeddings = encoder.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).cpu()

    output: dict[str, torch.Tensor] = {"version": torch.tensor(1)}
    for split in ("train", "eval"):
        domain_count = len(ONLINE_CONTEXT_TEXTS[split])
        variant_count = len(ONLINE_CONTEXT_TEXTS[split][0])
        output[f"{split}_contexts"] = embeddings[list(groups[split])].reshape(
            domain_count, variant_count, -1
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        f"saved {len(texts)} frozen {embeddings.shape[-1]}d embeddings to {args.output}"
    )


if __name__ == "__main__":
    main()
