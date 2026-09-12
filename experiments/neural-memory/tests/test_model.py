import torch

from neural_memory import FastWeightMemory, generate_episode_batch


def test_delta_rule_overwrites_an_existing_binding() -> None:
    model = FastWeightMemory(num_keys=2, num_values=2, d_model=2, retention=1.0)
    with torch.no_grad():
        model.key_embedding.weight.copy_(torch.eye(2))
        model.value_embedding.weight.copy_(torch.eye(2))

    state = model.initial_state(1)
    state, _ = model.write(
        state,
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([1.0]),
        force_gate=1.0,
    )
    assert model.read(state, torch.tensor([0])).argmax(dim=-1).item() == 0

    state, _ = model.write(
        state,
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([1.0]),
        force_gate=1.0,
    )
    assert model.read(state, torch.tensor([0])).argmax(dim=-1).item() == 1


def test_episode_queries_only_retained_observations() -> None:
    batch = generate_episode_batch(
        batch_size=4,
        num_keys=20,
        num_values=20,
        num_events=10,
        num_queries=3,
        generator=torch.Generator().manual_seed(3),
    )
    assert batch.retained.sum(dim=1).tolist() == [3, 3, 3, 3]
    expected_keys = batch.event_keys[batch.retained].reshape(4, 3)
    expected_values = batch.event_values[batch.retained].reshape(4, 3)
    assert torch.equal(batch.query_keys, expected_keys)
    assert torch.equal(batch.targets, expected_values)


def test_learned_gate_is_bounded() -> None:
    model = FastWeightMemory(num_keys=8, num_values=8, d_model=4)
    state = model.initial_state(3)
    _, telemetry = model.write(
        state,
        torch.tensor([0, 1, 2]),
        torch.tensor([3, 4, 5]),
        torch.tensor([0.1, 0.5, 0.9]),
    )
    assert telemetry.gate.shape == (3,)
    assert torch.all((telemetry.gate >= 0.0) & (telemetry.gate <= 1.0))

