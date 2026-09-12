from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from neural_memory.prospective import (
    ProspectiveLogger,
    classify_tool,
    decode_sketch,
    iter_credit_examples,
    load_events,
    signed_ngram_sketch,
)


def _payload(event: str, tool_id: str = "tool-1", **extra: object) -> dict[str, object]:
    return {
        "hook_event_name": event,
        "session_id": "private-session",
        "cwd": str(extra.pop("cwd")),
        "tool_use_id": tool_id,
        **extra,
    }


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


def test_sketch_is_fixed_length_and_does_not_contain_text() -> None:
    encoded = signed_ngram_sketch("TOP_SECRET user text", dimension=32)
    assert "TOP_SECRET" not in encoded
    assert len(decode_sketch(encoded)) == 32
    assert sum(value * value for value in decode_sketch(encoded)) == pytest.approx(1.0)


def test_tool_classification() -> None:
    assert classify_tool("Write", {"file_path": "x.py"}) == "mutation"
    assert classify_tool("Bash", {"command": "uv run pytest -q"}) == "verification"
    assert classify_tool("Read", {"file_path": "x.py"}) == "other"


def test_logger_links_mutation_verification_and_feedback_without_raw_text(repo: Path) -> None:
    path = repo / "events.jsonl"
    logger = ProspectiveLogger(path, salt=b"x" * 32)
    secret_path = repo / "private-name.py"

    write_input = {"file_path": str(secret_path), "content": "TOP_SECRET"}
    logger.handle(
        _payload("PreToolUse", cwd=repo, tool_name="Write", tool_input=write_input)
    )
    secret_path.write_text("TOP_SECRET")
    logger.handle(
        _payload(
            "PostToolUse",
            cwd=repo,
            tool_name="Write",
            tool_input=write_input,
            tool_response={"ok": True},
        )
    )
    logger.handle(_payload("Stop", cwd=repo, last_assistant_message="I wrote TOP_SECRET"))
    logger.handle(_payload("UserPromptSubmit", cwd=repo, prompt="ㅇㅇ TOP_SECRET"))

    logger.handle(
        _payload(
            "PreToolUse",
            "tool-2",
            cwd=repo,
            tool_name="Bash",
            tool_input={"command": "pytest -q"},
        )
    )
    logger.handle(
        _payload(
            "PostToolUse",
            "tool-2",
            cwd=repo,
            tool_name="Bash",
            tool_input={"command": "pytest -q"},
            tool_response="3 passed",
        )
    )

    serialized = path.read_text()
    assert "TOP_SECRET" not in serialized
    assert "private-name.py" not in serialized
    events = load_events(path)
    feedback = next(event for event in events if event["kind"] == "user_feedback")
    verification = next(
        event
        for event in events
        if event.get("category") == "verification" and event["kind"] == "action_result"
    )
    assert feedback["feedback"] == "strengthen"
    assert feedback["credit_action_ids"]
    assert feedback["candidate_action_ids"] == feedback["credit_action_ids"]
    assert verification["verification_passed"] is True
    assert verification["credit_action_ids"]
    assert {example.label for example in iter_credit_examples(events)} == {"strengthen"}


def test_exact_workspace_revert_emits_forget(repo: Path) -> None:
    path = repo / "events.jsonl"
    logger = ProspectiveLogger(path, salt=b"y" * 32)
    target = repo / "model.py"

    write_input = {"file_path": str(target), "content": "bad"}
    logger.handle(
        _payload("PreToolUse", cwd=repo, tool_name="Write", tool_input=write_input)
    )
    target.write_text("bad")
    logger.handle(
        _payload(
            "PostToolUse",
            cwd=repo,
            tool_name="Write",
            tool_input=write_input,
            tool_response={"ok": True},
        )
    )
    verify_input = {"command": "pytest"}
    logger.handle(
        _payload(
            "PreToolUse", "verify", cwd=repo, tool_name="Bash", tool_input=verify_input
        )
    )
    logger.handle(
        _payload(
            "PostToolUseFailure",
            "verify",
            cwd=repo,
            tool_name="Bash",
            tool_input=verify_input,
            error="1 failed",
        )
    )
    revert_input = {"file_path": str(target), "old_string": "bad", "new_string": ""}
    logger.handle(
        _payload(
            "PreToolUse", "revert", cwd=repo, tool_name="Edit", tool_input=revert_input
        )
    )
    target.unlink()
    logger.handle(
        _payload(
            "PostToolUse",
            "revert",
            cwd=repo,
            tool_name="Edit",
            tool_input=revert_input,
            tool_response={"ok": True},
        )
    )

    events = load_events(path)
    repair = next(
        event
        for event in events
        if event.get("action_id") and event.get("parent_action_ids")
    )
    assert repair["parent_action_ids"]
    examples = list(iter_credit_examples(events))
    assert [example.label for example in examples] == ["revise", "forget"]
    assert examples[-1].source == "exact_revert"


def test_replay_preserves_multi_action_credit() -> None:
    feature = signed_ngram_sketch("candidate", dimension=8)
    events = [
        {
            "schema": 1,
            "kind": "candidate",
            "session_id": "s",
            "action_id": action,
            "feature": feature,
        }
        for action in ("a", "b")
    ]
    events.append(
        {
            "schema": 1,
            "kind": "action_result",
            "session_id": "s",
            "action_id": "verify",
            "category": "verification",
            "verification_passed": True,
            "credit_action_ids": ["a", "b"],
            "outcome_feature": feature,
        }
    )
    example = next(iter(iter_credit_examples(events)))
    assert example.action_ids == ("a", "b")
    assert example.label == "strengthen"


def test_invalid_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{}\nnot json\n{"schema": 1, "session_id": "s"}\n')
    assert load_events(path) == [{"schema": 1, "session_id": "s"}]
