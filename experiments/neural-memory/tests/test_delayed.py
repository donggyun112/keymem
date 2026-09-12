import torch

from neural_memory import (
    EligibilityTraceMemory,
    FrozenTextBanks,
    OnlineTextBanks,
    generate_delayed_utility_batch,
)


def test_delayed_batch_hides_utility_until_feedback() -> None:
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
    batch = generate_delayed_utility_batch(
        context_banks=context_banks,
        cue_banks=cue_banks,
        split="train",
        batch_size=4,
        active_contexts=6,
        retained_contexts=4,
        num_local_keys=12,
        num_values=16,
        events_per_context=3,
        generator=torch.Generator().manual_seed(13),
    )
    assert batch.retained_contexts.sum(dim=1).tolist() == [4, 4, 4, 4]
    assert batch.observation_context_features.shape == (4, 18, 10)
    assert batch.feedback_cue_features.shape == (4, 6, 10)
    assert batch.query_keys.shape == (4, 12)
    assert not hasattr(batch, "observation_should_retain")


def test_oracle_feedback_consolidates_retained_traces() -> None:
    model = EligibilityTraceMemory(
        text_dim=3,
        num_keys=2,
        num_values=3,
        num_blocks=3,
        d_model=3,
        initial_threshold=0.9,
    )
    with torch.no_grad():
        model.key_embedding.weight.copy_(torch.eye(2, 3))
        model.value_embedding.weight.copy_(torch.eye(3))

    contexts = torch.eye(3)
    state = model.initial_state(1)
    for context in range(3):
        state, _ = model.observe(
            state,
            contexts[context : context + 1],
            torch.tensor([0]),
            torch.tensor([context]),
        )
    for context, retain in ((0, True), (1, False), (2, True)):
        state, _, _ = model.feedback(
            state,
            contexts[context : context + 1],
            torch.zeros(1, 3),
            mode="oracle",
            should_retain=torch.tensor([retain]),
        )
    state = model.consolidate(state, keep_slots=2)
    assert state.consolidated_mask.sum().item() == 2
    assert state.consolidated_mask.tolist() == [[1.0, 0.0, 1.0]]
    for context in (0, 2):
        logits, _ = model.read(
            state, contexts[context : context + 1], torch.tensor([0])
        )
        assert logits.argmax(dim=-1).item() == context


def test_learned_feedback_does_not_read_oracle_label() -> None:
    model = EligibilityTraceMemory(
        text_dim=2,
        num_keys=2,
        num_values=2,
        num_blocks=1,
        d_model=2,
    )
    state = model.initial_state(1)
    state, _ = model.observe(
        state,
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([0]),
        torch.tensor([0]),
    )
    cue = torch.tensor([[0.0, 1.0]])
    _, first_route, first_priority = model.feedback(
        state,
        torch.tensor([[1.0, 0.0]]),
        cue,
        mode="learned",
        should_retain=torch.tensor([True]),
    )
    _, second_route, second_priority = model.feedback(
        state,
        torch.tensor([[1.0, 0.0]]),
        cue,
        mode="learned",
        should_retain=torch.tensor([False]),
    )
    assert torch.equal(first_route, second_route)
    assert torch.equal(first_priority, second_priority)
