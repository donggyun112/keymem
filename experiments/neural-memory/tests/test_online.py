from pathlib import Path

import torch

from neural_memory import (
    OnlinePrototypeMemory,
    OnlineTextBanks,
    generate_online_episode_batch,
    load_online_text_banks,
)


def make_banks() -> OnlineTextBanks:
    return OnlineTextBanks(
        train_contexts=torch.randn(6, 3, 10),
        eval_contexts=torch.randn(6, 2, 10),
    )


def test_online_episode_changes_domain_subset_but_keeps_episode_context_stable() -> None:
    batch = generate_online_episode_batch(
        banks=make_banks(),
        split="train",
        batch_size=5,
        active_contexts=3,
        num_local_keys=10,
        num_values=12,
        events_per_context=4,
        generator=torch.Generator().manual_seed(11),
    )
    assert batch.active_domain_ids.unique(dim=0).shape[0] > 1
    query_features = batch.query_context_features.reshape(5, 3, 4, 10)
    assert torch.equal(
        query_features,
        query_features[:, :, :1, :].expand_as(query_features),
    )


def test_online_prototypes_claim_empty_blocks_and_recover_them() -> None:
    model = OnlinePrototypeMemory(
        text_dim=2,
        num_keys=2,
        num_values=2,
        num_blocks=2,
        d_model=2,
        initial_threshold=0.9,
    )
    with torch.no_grad():
        model.key_embedding.weight.copy_(torch.eye(2))
        model.value_embedding.weight.copy_(torch.eye(2))

    first_context = torch.tensor([[1.0, 0.0]])
    second_context = torch.tensor([[0.0, 1.0]])
    state = model.initial_state(1)
    state, first = model.write(
        state, first_context, torch.tensor([0]), torch.tensor([0])
    )
    state, second = model.write(
        state, second_context, torch.tensor([0]), torch.tensor([1])
    )
    state, repeated = model.write(
        state, first_context, torch.tensor([1]), torch.tensor([0])
    )
    assert first.route.item() == 0
    assert second.route.item() == 1
    assert repeated.route.item() == 0
    assert repeated.matched_existing.item()
    assert state.counts.tolist() == [[2.0, 1.0]]

    first_logits, first_route = model.read(state, first_context, torch.tensor([0]))
    second_logits, second_route = model.read(state, second_context, torch.tensor([0]))
    assert first_route.item() == 0
    assert second_route.item() == 1
    assert first_logits.argmax(dim=-1).item() == 0
    assert second_logits.argmax(dim=-1).item() == 1


def test_online_text_banks_round_trip(tmp_path: Path) -> None:
    banks = make_banks()
    path = tmp_path / "online.pt"
    torch.save(
        {
            "train_contexts": banks.train_contexts,
            "eval_contexts": banks.eval_contexts,
        },
        path,
    )
    loaded = load_online_text_banks(path)
    assert loaded.num_domains == 6
    assert loaded.text_dim == 10
    assert torch.equal(loaded.eval_contexts, banks.eval_contexts)
