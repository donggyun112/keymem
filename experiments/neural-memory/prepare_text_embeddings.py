from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neural_memory.text_corpus import CONTEXT_TEXTS, CUE_TEXTS, all_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare frozen sentence embeddings")
    parser.add_argument(
        "--model",
        default="BAAI/bge-small-en-v1.5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/bge_small_en.pt"),
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    from sentence_transformers import SentenceTransformer

    args = parse_args()
    texts, groups = all_texts()
    encoder = SentenceTransformer(args.model, device=args.device)
    embeddings = encoder.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).cpu()

    output: dict[str, torch.Tensor] = {"version": torch.tensor(1)}
    for split in ("train", "eval"):
        context_indices = groups[(split, "context")]
        context_count = len(CONTEXT_TEXTS[split])
        context_variants = len(CONTEXT_TEXTS[split][0])
        output[f"{split}_contexts"] = embeddings[list(context_indices)].reshape(
            context_count, context_variants, -1
        )

        cue_indices = groups[(split, "cue")]
        cue_classes = len(CUE_TEXTS[split])
        cue_variants = len(CUE_TEXTS[split][0])
        output[f"{split}_cues"] = embeddings[list(cue_indices)].reshape(
            cue_classes, cue_variants, -1
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        f"saved {len(texts)} frozen {embeddings.shape[-1]}d embeddings to {args.output}"
    )


if __name__ == "__main__":
    main()
