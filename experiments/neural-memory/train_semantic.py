from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from neural_memory import (
    FrozenTextBanks,
    GateMode,
    SemanticAllocationMode,
    SemanticEpisodeBatch,
    SemanticFastWeightMemory,
    generate_semantic_episode_batch,
    load_text_banks,
)
from train import choose_device


@dataclass(frozen=True)
class SemanticMetrics:
    loss: float
    accuracy: float
    context_intrusion_rate: float
    distractor_overwrite_rate: float
    useful_gate: float
    distractor_gate: float
    assignment_score: float
    allocation_entropy: float


@dataclass(frozen=True)
class Variant:
    name: str
    allocation_mode: SemanticAllocationMode
    gate_mode: GateMode
    num_blocks: int
    d_model: int


def forward_episode(
    model: SemanticFastWeightMemory,
    batch: SemanticEpisodeBatch,
    variant: Variant,
) -> tuple[Tensor, Tensor]:
    state = model.initial_state(batch.event_keys.shape[0])
    gates: list[Tensor] = []
    for step in range(batch.event_keys.shape[1]):
        state, telemetry = model.write(
            state,
            batch.event_context_features[:, step],
            batch.event_cue_features[:, step],
            batch.event_keys[:, step],
            batch.event_values[:, step],
            allocation_mode=variant.allocation_mode,
            gate_mode=variant.gate_mode,
            context_ids=batch.event_context_ids[:, step],
            should_write=batch.event_should_write[:, step],
        )
        gates.append(telemetry.gate)

    logits = [
        model.read(
            state,
            batch.query_context_features[:, step],
            batch.query_keys[:, step],
            allocation_mode=variant.allocation_mode,
            context_ids=batch.query_context_ids[:, step],
        )
        for step in range(batch.query_keys.shape[1])
    ]
    return torch.stack(logits, dim=1), torch.stack(gates, dim=1)


def allocation_diagnostics(
    model: SemanticFastWeightMemory,
    banks: FrozenTextBanks,
    variant: Variant,
) -> tuple[float, float]:
    context_bank = banks.eval_contexts.to(model.key_embedding.weight.device)
    contexts, variants, text_dim = context_bank.shape
    features = context_bank.reshape(contexts * variants, text_dim)
    context_ids = torch.arange(contexts, device=features.device).repeat_interleave(variants)
    allocation = model.allocation(
        features,
        mode=variant.allocation_mode,
        context_ids=context_ids,
    ).reshape(contexts, variants, model.num_blocks).mean(dim=1)

    if model.num_blocks == 1:
        return 0.0, 0.0
    entropy = -(allocation * allocation.clamp_min(1e-9).log()).sum(dim=-1)
    entropy = float((entropy / math.log(model.num_blocks)).mean().detach().cpu())

    scores = []
    for assignment in itertools.permutations(range(model.num_blocks), contexts):
        block_ids = torch.tensor(assignment, device=features.device)
        scores.append(allocation[torch.arange(contexts, device=features.device), block_ids].mean())
    assignment_score = float(torch.stack(scores).max().detach().cpu())
    return assignment_score, entropy


def measure(
    model: SemanticFastWeightMemory,
    banks: FrozenTextBanks,
    logits: Tensor,
    gates: Tensor,
    batch: SemanticEpisodeBatch,
    variant: Variant,
) -> SemanticMetrics:
    loss = F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten())
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == batch.targets).float().mean()

    candidate_index = batch.query_slots[:, None, :].expand(
        -1, batch.all_values.shape[1], -1
    )
    same_key_values = batch.all_values.gather(2, candidate_index)
    other_context = (
        torch.arange(batch.all_values.shape[1], device=predictions.device)[None, :, None]
        != batch.query_context_ids[:, None, :]
    )
    context_intrusions = (
        (predictions[:, None, :] == same_key_values) & other_context
    ).any(dim=1)

    batch_ids = torch.arange(predictions.shape[0], device=predictions.device)[:, None]
    same_context_distractor = batch.distractor_values[
        batch_ids,
        batch.query_context_ids,
        batch.query_slots,
    ]
    distractor_overwrites = predictions == same_context_distractor
    assignment, entropy = allocation_diagnostics(model, banks, variant)
    return SemanticMetrics(
        loss=float(loss.detach().cpu()),
        accuracy=float(accuracy.detach().cpu()),
        context_intrusion_rate=float(context_intrusions.float().mean().detach().cpu()),
        distractor_overwrite_rate=float(
            distractor_overwrites.float().mean().detach().cpu()
        ),
        useful_gate=float(gates[batch.event_should_write].mean().detach().cpu()),
        distractor_gate=float(gates[~batch.event_should_write].mean().detach().cpu()),
        assignment_score=assignment,
        allocation_entropy=entropy,
    )


def make_batch(
    args: argparse.Namespace,
    banks: FrozenTextBanks,
    split: str,
    generator: torch.Generator,
    device: torch.device,
) -> SemanticEpisodeBatch:
    return generate_semantic_episode_batch(
        banks=banks,
        split=split,  # type: ignore[arg-type]
        batch_size=args.batch_size,
        num_local_keys=args.num_local_keys,
        num_values=args.num_values,
        events_per_context=args.events_per_context,
        generator=generator,
    ).to(device)


@torch.no_grad()
def evaluate(
    model: SemanticFastWeightMemory,
    banks: FrozenTextBanks,
    args: argparse.Namespace,
    device: torch.device,
    variant: Variant,
    *,
    seed: int,
) -> SemanticMetrics:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    totals = torch.zeros(6)
    for _ in range(args.eval_batches):
        batch = make_batch(args, banks, "eval", generator, device)
        logits, gates = forward_episode(model, batch, variant)
        current = measure(model, banks, logits, gates, batch, variant)
        totals += torch.tensor(
            [
                current.loss,
                current.accuracy,
                current.context_intrusion_rate,
                current.distractor_overwrite_rate,
                current.useful_gate,
                current.distractor_gate,
            ]
        )
    totals /= args.eval_batches
    assignment, entropy = allocation_diagnostics(model, banks, variant)
    return SemanticMetrics(
        *[float(value) for value in totals],
        assignment_score=assignment,
        allocation_entropy=entropy,
    )


def train_variant(
    args: argparse.Namespace,
    banks: FrozenTextBanks,
    device: torch.device,
    variant: Variant,
    *,
    seed: int,
) -> tuple[SemanticFastWeightMemory, SemanticMetrics]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = SemanticFastWeightMemory(
        text_dim=banks.text_dim,
        num_contexts=banks.train_contexts.shape[0],
        num_keys=args.num_local_keys,
        num_values=args.num_values,
        num_blocks=variant.num_blocks,
        d_model=variant.d_model,
        router_dim=args.router_dim,
        gate_mlp=args.gate_mlp,
        temperature=args.temperature,
        retention=args.retention,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(seed + 1)

    model.train()
    for _ in range(args.steps):
        batch = make_batch(args, banks, "train", generator, device)
        logits, _ = forward_episode(model, batch, variant)
        loss = F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    metrics = evaluate(model, banks, args, device, variant, seed=seed + 10_000)
    return model, metrics


def experiment_variants(args: argparse.Namespace) -> list[Variant]:
    fast_capacity = args.num_blocks * args.block_dim * args.block_dim
    single_dim = math.isqrt(fast_capacity)
    if single_dim * single_dim != fast_capacity:
        raise ValueError("blocked fast capacity must be a square")
    return [
        Variant("single_gate", "uniform", "learned", 1, single_dim),
        Variant("uniform_gate", "uniform", "learned", args.num_blocks, args.block_dim),
        Variant(
            "competitive_always",
            "learned",
            "always",
            args.num_blocks,
            args.block_dim,
        ),
        Variant("joint", "learned", "learned", args.num_blocks, args.block_dim),
        Variant("oracle", "oracle", "oracle", args.num_blocks, args.block_dim),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train semantic joint-selection memory")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("artifacts/bge_small_en.pt"),
    )
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-local-keys", type=int, default=32)
    parser.add_argument("--num-values", type=int, default=64)
    parser.add_argument("--events-per-context", type=int, default=6)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--block-dim", type=int, default=12)
    parser.add_argument("--router-dim", type=int, default=32)
    parser.add_argument(
        "--gate-mlp",
        action="store_true",
        help="use the earlier nonlinear gate instead of the default linear probe",
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--retention", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=(
            "single_gate",
            "uniform_gate",
            "competitive_always",
            "joint",
            "oracle",
        ),
        help="run only the selected variants (default: all)",
    )
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.embeddings.exists():
        raise SystemExit(
            f"missing {args.embeddings}; run prepare_text_embeddings.py first"
        )
    banks = load_text_banks(args.embeddings)
    device = choose_device(args.device)
    result: dict[str, object] = {
        "device": str(device),
        "config": {**vars(args), "embeddings": str(args.embeddings)},
        "chance_accuracy": 1.0 / args.num_values,
        "fast_capacity": args.num_blocks * args.block_dim * args.block_dim,
    }
    selected = [
        variant
        for variant in experiment_variants(args)
        if args.variants is None or variant.name in args.variants
    ]
    for variant in selected:
        model, metrics = train_variant(args, banks, device, variant, seed=args.seed)
        result[variant.name] = {
            **asdict(metrics),
            "fast_capacity": model.fast_capacity,
        }

    if args.summary:
        fields = (
            "accuracy",
            "context_intrusion_rate",
            "distractor_overwrite_rate",
            "useful_gate",
            "distractor_gate",
            "assignment_score",
        )
        names = tuple(variant.name for variant in selected)
        rows = []
        for name in names:
            values = [
                float(result[name][field])  # type: ignore[index]
                for field in fields
            ]
            rows.append(f"{name}=" + ",".join(f"{value:.6f}" for value in values))
        print("fields=" + ",".join(fields))
        print(";".join(rows))
    else:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
