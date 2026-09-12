from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from neural_memory import (
    ConsolidationMode,
    DelayedUtilityBatch,
    EligibilityTraceMemory,
    FrozenTextBanks,
    OnlineTextBanks,
    generate_delayed_utility_batch,
    load_online_text_banks,
    load_text_banks,
)
from train import choose_device


@dataclass(frozen=True)
class DelayedMetrics:
    loss: float
    accuracy: float
    useful_coverage: float
    useful_priority: float
    distractor_priority: float
    active_slots: float


def forward_episode(
    model: EligibilityTraceMemory,
    batch: DelayedUtilityBatch,
    *,
    mode: ConsolidationMode,
    keep_slots: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    state = model.initial_state(batch.observation_keys.shape[0])
    trace_labels = torch.full(
        (batch.observation_keys.shape[0], model.num_blocks),
        -1,
        dtype=torch.long,
        device=batch.observation_keys.device,
    )
    for step in range(batch.observation_keys.shape[1]):
        state, telemetry = model.observe(
            state,
            batch.observation_context_features[:, step],
            batch.observation_keys[:, step],
            batch.observation_values[:, step],
        )
        new_assignment = ~telemetry.matched_existing
        route_mask = F.one_hot(telemetry.route, num_classes=model.num_blocks).bool()
        labels = batch.observation_local_ids[:, step, None].expand(
            -1, model.num_blocks
        )
        trace_labels = torch.where(
            new_assignment[:, None] & route_mask, labels, trace_labels
        )

    priorities: list[Tensor] = []
    feedback_labels: list[Tensor] = []
    for step in range(batch.feedback_local_ids.shape[1]):
        state, _, priority = model.feedback(
            state,
            batch.feedback_context_features[:, step],
            batch.feedback_cue_features[:, step],
            mode=mode,
            should_retain=batch.feedback_should_retain[:, step],
        )
        priorities.append(priority)
        feedback_labels.append(batch.feedback_should_retain[:, step])
    state = model.consolidate(state, keep_slots=keep_slots)

    logits = [
        model.read(
            state,
            batch.query_context_features[:, step],
            batch.query_keys[:, step],
        )[0]
        for step in range(batch.query_keys.shape[1])
    ]
    priority_tensor = torch.stack(priorities, dim=1)
    feedback_tensor = torch.stack(feedback_labels, dim=1)
    return torch.stack(logits, dim=1), priority_tensor, feedback_tensor, trace_labels


def measure(
    logits: Tensor,
    priorities: Tensor,
    feedback_labels: Tensor,
    trace_labels: Tensor,
    consolidated_mask: Tensor,
    batch: DelayedUtilityBatch,
    *,
    retained_contexts: int,
) -> DelayedMetrics:
    loss = F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten())
    accuracy = (logits.argmax(dim=-1) == batch.targets).float().mean()
    useful_ids = batch.retained_contexts.nonzero(as_tuple=False)[:, 1].reshape(
        batch.retained_contexts.shape[0], retained_contexts
    )
    selected_labels = torch.where(
        consolidated_mask.bool(), trace_labels, torch.full_like(trace_labels, -1)
    )
    coverage = (selected_labels[:, :, None] == useful_ids[:, None, :]).any(dim=1)
    return DelayedMetrics(
        loss=float(loss.detach().cpu()),
        accuracy=float(accuracy.detach().cpu()),
        useful_coverage=float(coverage.float().mean().detach().cpu()),
        useful_priority=float(priorities[feedback_labels].mean().detach().cpu()),
        distractor_priority=float(priorities[~feedback_labels].mean().detach().cpu()),
        active_slots=float(consolidated_mask.sum(dim=-1).float().mean().detach().cpu()),
    )


def make_batch(
    args: argparse.Namespace,
    context_banks: OnlineTextBanks,
    cue_banks: FrozenTextBanks,
    split: str,
    generator: torch.Generator,
    device: torch.device,
) -> DelayedUtilityBatch:
    return generate_delayed_utility_batch(
        context_banks=context_banks,
        cue_banks=cue_banks,
        split=split,  # type: ignore[arg-type]
        batch_size=args.batch_size,
        active_contexts=args.active_contexts,
        retained_contexts=args.retained_contexts,
        num_local_keys=args.num_local_keys,
        num_values=args.num_values,
        events_per_context=args.events_per_context,
        generator=generator,
    ).to(device)


@torch.no_grad()
def evaluate(
    model: EligibilityTraceMemory,
    context_banks: OnlineTextBanks,
    cue_banks: FrozenTextBanks,
    args: argparse.Namespace,
    device: torch.device,
    *,
    mode: ConsolidationMode,
    seed: int,
) -> DelayedMetrics:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    totals = torch.zeros(6)
    for _ in range(args.eval_batches):
        batch = make_batch(args, context_banks, cue_banks, "eval", generator, device)
        state = model.initial_state(batch.observation_keys.shape[0])
        # Use the shared forward path, then reconstruct its deterministic final
        # mask from the same episode for metric bookkeeping.
        logits, priorities, feedback_labels, trace_labels = forward_episode(
            model, batch, mode=mode, keep_slots=args.retained_contexts
        )
        priority_by_slot = torch.zeros(
            batch.observation_keys.shape[0], args.trace_slots, device=device
        )
        for step in range(batch.feedback_local_ids.shape[1]):
            local_id = batch.feedback_local_ids[:, step]
            trace_slot = (
                trace_labels == local_id[:, None]
            ).float().argmax(dim=-1)
            priority_by_slot.scatter_(1, trace_slot[:, None], priorities[:, step : step + 1])
        mask = torch.zeros_like(priority_by_slot)
        mask.scatter_(1, priority_by_slot.topk(args.retained_contexts, dim=-1).indices, 1.0)
        current = measure(
            logits,
            priorities,
            feedback_labels,
            trace_labels,
            mask,
            batch,
            retained_contexts=args.retained_contexts,
        )
        totals += torch.tensor(list(asdict(current).values()))
    totals /= args.eval_batches
    return DelayedMetrics(*[float(value) for value in totals])


def build_model(args: argparse.Namespace, text_dim: int) -> EligibilityTraceMemory:
    return EligibilityTraceMemory(
        text_dim=text_dim,
        num_keys=args.num_local_keys,
        num_values=args.num_values,
        num_blocks=args.trace_slots,
        d_model=args.block_dim,
        prototype_dim=args.prototype_dim,
        initial_threshold=args.initial_threshold,
        retention=args.retention,
    )


def train_variant(
    args: argparse.Namespace,
    context_banks: OnlineTextBanks,
    cue_banks: FrozenTextBanks,
    device: torch.device,
    *,
    mode: ConsolidationMode,
    seed: int,
) -> tuple[EligibilityTraceMemory, DelayedMetrics]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = build_model(args, context_banks.text_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(seed + 1)

    model.train()
    for _ in range(args.steps):
        batch = make_batch(args, context_banks, cue_banks, "train", generator, device)
        logits, _, _, _ = forward_episode(
            model, batch, mode=mode, keep_slots=args.retained_contexts
        )
        loss = F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    metrics = evaluate(
        model,
        context_banks,
        cue_banks,
        args,
        device,
        mode=mode,
        seed=seed + 10_000,
    )
    return model, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train delayed-utility consolidation")
    parser.add_argument(
        "--context-embeddings",
        type=Path,
        default=Path("artifacts/bge_online_domains.pt"),
    )
    parser.add_argument(
        "--cue-embeddings",
        type=Path,
        default=Path("artifacts/bge_small_en.pt"),
    )
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--active-contexts", type=int, default=6)
    parser.add_argument("--retained-contexts", type=int, default=4)
    parser.add_argument("--trace-slots", type=int, default=6)
    parser.add_argument("--num-local-keys", type=int, default=32)
    parser.add_argument("--num-values", type=int, default=64)
    parser.add_argument("--events-per-context", type=int, default=6)
    parser.add_argument("--block-dim", type=int, default=9)
    parser.add_argument("--prototype-dim", type=int, default=16)
    parser.add_argument("--initial-threshold", type=float, default=0.9)
    parser.add_argument("--retention", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("recency", "learned", "oracle"),
    )
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trace_slots < args.active_contexts:
        raise SystemExit("trace_slots must cover all active contexts in this experiment")
    context_banks = load_online_text_banks(args.context_embeddings)
    cue_banks = load_text_banks(args.cue_embeddings)
    device = choose_device(args.device)
    modes: tuple[ConsolidationMode, ...] = ("recency", "learned", "oracle")
    selected = [mode for mode in modes if args.variants is None or mode in args.variants]
    result: dict[str, object] = {
        "device": str(device),
        "config": {**vars(args)},
        "trace_runtime_state": args.trace_slots
        * (args.block_dim * args.block_dim + args.prototype_dim + 2),
    }
    for mode in selected:
        model, metrics = train_variant(
            args, context_banks, cue_banks, device, mode=mode, seed=args.seed
        )
        result[mode] = asdict(metrics)

    if args.summary:
        fields = (
            "accuracy",
            "useful_coverage",
            "useful_priority",
            "distractor_priority",
            "active_slots",
        )
        print("fields=" + ",".join(fields))
        rows = []
        for mode in selected:
            values = [float(result[mode][field]) for field in fields]  # type: ignore[index]
            rows.append(f"{mode}=" + ",".join(f"{value:.6f}" for value in values))
        print(";".join(rows))
    else:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
