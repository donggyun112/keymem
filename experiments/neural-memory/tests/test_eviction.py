import torch

from neural_memory import (
    FrozenTextBanks,
    OnlineTextBanks,
    PriorityEvictionMemory,
    generate_eviction_episode_batch,
)


def test_eviction_batch_queries_only_retained_contexts() -> None:
    context_banks = OnlineTextBanks(
        train_contexts=torch.randn(8, 3, 10),
        eval_contexts=torch.randn(8, 2, 10),
    )
    cue_banks = FrozenTextBanks(
        train_contexts=torch.randn(4, 3, 10),
        eval_contexts=torch.randn(4, 2, 10),
        train_cues=torch.randn(2, 4, 10),
        eval_cues=torch.randn(2, 3, 10),
    )
    batch = generate_eviction_episode_batch(
        context_banks=context_banks,
        cue_banks=cue_banks,
        split="train",
        batch_size=4,
        active_contexts=6,
        retained_contexts=4,
        num_local_keys=12,
        num_values=16,
        events_per_context=3,
        generator=torch.Generator().manual_seed(12),
    )
    assert batch.retained_contexts.sum(dim=1).tolist() == [4, 4, 4, 4]
    assert batch.query_keys.shape == (4, 12)
    queried = batch.retained_contexts.gather(1, batch.query_local_ids)
    assert torch.all(queried)


def test_oracle_priority_evicts_an_unretained_context() -> None:
    model = PriorityEvictionMemory(
        text_dim=3,
        num_keys=2,
        num_values=3,
        num_blocks=2,
        d_model=3,
        initial_threshold=0.9,
    )
    with torch.no_grad():
        model.key_embedding.weight.copy_(torch.eye(2, 3))
        model.value_embedding.weight.copy_(torch.eye(3))

    contexts = torch.eye(3)
    cue = torch.zeros(1, 3)
    state = model.initial_state(1)
    for context, target, retain in ((0, 0, False), (1, 1, True), (2, 2, True)):
        state, telemetry = model.write(
            state,
            contexts[context : context + 1],
            cue,
            torch.tensor([0]),
            torch.tensor([target]),
            priority_mode="oracle",
            should_retain=torch.tensor([retain]),
        )
        assert telemetry.accepted.item()

    assert state.priorities.sort(dim=-1).values.tolist() == [[1.0, 1.0]]
    for context, target in ((1, 1), (2, 2)):
        logits, _ = model.read(
            state, contexts[context : context + 1], torch.tensor([0])
        )
        assert logits.argmax(dim=-1).item() == target


def test_oracle_priority_rejects_low_value_when_full() -> None:
    model = PriorityEvictionMemory(
        text_dim=2,
        num_keys=2,
        num_values=2,
        num_blocks=1,
        d_model=2,
    )
    state = model.initial_state(1)
    state, _ = model.write(
        state,
        torch.tensor([[1.0, 0.0]]),
        torch.zeros(1, 2),
        torch.tensor([0]),
        torch.tensor([0]),
        priority_mode="oracle",
        should_retain=torch.tensor([True]),
    )
    before = state.weights.clone()
    state, telemetry = model.write(
        state,
        torch.tensor([[0.0, 1.0]]),
        torch.zeros(1, 2),
        torch.tensor([0]),
        torch.tensor([1]),
        priority_mode="oracle",
        should_retain=torch.tensor([False]),
    )
    assert not telemetry.accepted.item()
    assert torch.equal(state.weights, before)


def test_learned_priority_does_not_read_oracle_retention_label() -> None:
    model = PriorityEvictionMemory(
        text_dim=3,
        num_keys=2,
        num_values=2,
        num_blocks=2,
        d_model=2,
    )
    state = model.initial_state(1)
    context = torch.tensor([[1.0, 0.0, 0.0]])
    cue = torch.tensor([[0.0, 1.0, 0.0]])
    _, first = model.write(
        state,
        context,
        cue,
        torch.tensor([0]),
        torch.tensor([0]),
        priority_mode="learned",
        should_retain=torch.tensor([True]),
    )
    _, second = model.write(
        state,
        context,
        cue,
        torch.tensor([0]),
        torch.tensor([0]),
        priority_mode="learned",
        should_retain=torch.tensor([False]),
    )
    assert torch.equal(first.incoming_priority, second.incoming_priority)
    assert torch.equal(first.accepted, second.accepted)
