from pathlib import Path

import torch

from neural_memory import (
    FrozenTextBanks,
    SemanticFastWeightMemory,
    generate_semantic_episode_batch,
    load_text_banks,
)
from neural_memory.text_corpus import CUE_TEXTS, CONTEXT_TEXTS


def make_banks() -> FrozenTextBanks:
    return FrozenTextBanks(
        train_contexts=torch.randn(2, 3, 6),
        eval_contexts=torch.randn(2, 2, 6),
        train_cues=torch.randn(2, 4, 6),
        eval_cues=torch.randn(2, 3, 6),
    )


def test_train_and_eval_texts_do_not_overlap() -> None:
    train = {text for group in CONTEXT_TEXTS["train"] for text in group}
    train |= {text for group in CUE_TEXTS["train"] for text in group}
    evaluation = {text for group in CONTEXT_TEXTS["eval"] for text in group}
    evaluation |= {text for group in CUE_TEXTS["eval"] for text in group}
    assert train.isdisjoint(evaluation)


def test_semantic_episode_places_distractor_after_each_stable_value() -> None:
    batch = generate_semantic_episode_batch(
        banks=make_banks(),
        split="train",
        batch_size=3,
        num_local_keys=10,
        num_values=12,
        events_per_context=4,
        generator=torch.Generator().manual_seed(9),
    )
    assert torch.all(batch.event_should_write[:, 0::2])
    assert not torch.any(batch.event_should_write[:, 1::2])
    assert torch.equal(batch.event_context_ids[:, 0::2], batch.event_context_ids[:, 1::2])
    assert torch.equal(batch.event_keys[:, 0::2], batch.event_keys[:, 1::2])
    assert torch.all(batch.event_values[:, 0::2] != batch.event_values[:, 1::2])


def test_learned_semantic_route_does_not_read_oracle_ids() -> None:
    model = SemanticFastWeightMemory(
        text_dim=6,
        num_contexts=2,
        num_keys=4,
        num_values=4,
        num_blocks=2,
        d_model=3,
    )
    features = torch.randn(5, 6)
    first = model.allocation(
        features,
        mode="learned",
        context_ids=torch.zeros(5, dtype=torch.long),
    )
    second = model.allocation(
        features,
        mode="learned",
        context_ids=torch.ones(5, dtype=torch.long),
    )
    assert torch.equal(first, second)
    assert torch.allclose(first.sum(dim=-1), torch.ones(5))


def test_oracle_joint_controls_ignore_conflicting_distractors() -> None:
    model = SemanticFastWeightMemory(
        text_dim=4,
        num_contexts=2,
        num_keys=2,
        num_values=2,
        num_blocks=2,
        d_model=2,
        retention=1.0,
    )
    with torch.no_grad():
        model.key_embedding.weight.copy_(torch.eye(2))
        model.value_embedding.weight.copy_(torch.eye(2))

    state = model.initial_state(1)
    features = torch.randn(1, 4)
    cue = torch.randn(1, 4)
    for context, stable, distractor in ((0, 0, 1), (1, 1, 0)):
        state, stable_telemetry = model.write(
            state,
            features,
            cue,
            torch.tensor([0]),
            torch.tensor([stable]),
            allocation_mode="oracle",
            gate_mode="oracle",
            context_ids=torch.tensor([context]),
            should_write=torch.tensor([True]),
        )
        state, distractor_telemetry = model.write(
            state,
            features,
            cue,
            torch.tensor([0]),
            torch.tensor([distractor]),
            allocation_mode="oracle",
            gate_mode="oracle",
            context_ids=torch.tensor([context]),
            should_write=torch.tensor([False]),
        )
        assert stable_telemetry.gate.item() == 1.0
        assert distractor_telemetry.gate.item() == 0.0

    for context, target in ((0, 0), (1, 1)):
        logits = model.read(
            state,
            features,
            torch.tensor([0]),
            allocation_mode="oracle",
            context_ids=torch.tensor([context]),
        )
        assert logits.argmax(dim=-1).item() == target


def test_text_banks_round_trip(tmp_path: Path) -> None:
    banks = make_banks()
    path = tmp_path / "embeddings.pt"
    torch.save(
        {
            "train_contexts": banks.train_contexts,
            "eval_contexts": banks.eval_contexts,
            "train_cues": banks.train_cues,
            "eval_cues": banks.eval_cues,
        },
        path,
    )
    loaded = load_text_banks(path)
    assert loaded.text_dim == 6
    assert torch.equal(loaded.eval_cues, banks.eval_cues)
