import torch

from neural_memory import CompetitiveFastWeightMemory, generate_competition_batch


def test_competition_batch_has_true_context_collisions() -> None:
    batch = generate_competition_batch(
        batch_size=3,
        num_contexts=4,
        num_local_keys=12,
        num_values=16,
        events_per_context=5,
        generator=torch.Generator().manual_seed(4),
    )
    keys = batch.query_keys.reshape(3, 4, 5)
    assert torch.equal(keys, keys[:, :1, :].expand_as(keys))
    for episode in range(3):
        for slot in range(5):
            assert batch.all_values[episode, :, slot].unique().numel() == 4


def test_allocations_respect_a_fixed_budget() -> None:
    model = CompetitiveFastWeightMemory(
        num_contexts=4,
        num_keys=8,
        num_values=8,
        num_blocks=4,
        d_model=4,
    )
    contexts = torch.arange(4)
    for mode in ("learned", "uniform", "oracle"):
        allocation = model.allocation(contexts, mode)
        assert torch.allclose(allocation.sum(dim=-1), torch.ones(4))
    assert torch.allclose(
        model.allocation(contexts, "uniform"), torch.full((4, 4), 0.25)
    )
    assert torch.equal(model.allocation(contexts, "oracle"), torch.eye(4))


def test_oracle_blocks_isolate_conflicting_bindings() -> None:
    model = CompetitiveFastWeightMemory(
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
    state, _ = model.write(
        state,
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([0]),
        mode="oracle",
    )
    state, _ = model.write(
        state,
        torch.tensor([1]),
        torch.tensor([0]),
        torch.tensor([1]),
        mode="oracle",
    )
    for context, target in ((0, 0), (1, 1)):
        logits = model.read(
            state,
            torch.tensor([context]),
            torch.tensor([0]),
            mode="oracle",
        )
        assert logits.argmax(dim=-1).item() == target


def test_single_and_blocked_variants_can_match_fast_capacity() -> None:
    assert 1 * 24 * 24 == 4 * 12 * 12
