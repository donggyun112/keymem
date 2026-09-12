from __future__ import annotations

import json
from pathlib import Path

from neural_memory.repair_chains import load_repair_episodes, verification_family


def _message(*blocks: dict[str, object]) -> str:
    return json.dumps({"sessionId": "s", "message": {"content": list(blocks)}})


def _use(tool_id: str, name: str, tool_input: dict[str, object]) -> dict[str, object]:
    return {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}


def _result(tool_id: str, text: str) -> dict[str, object]:
    return {"type": "tool_result", "tool_use_id": tool_id, "content": text}


def test_verification_family_normalizes_variants() -> None:
    assert verification_family("uv run pytest -q") == "pytest"
    assert verification_family("pnpm run test") == "javascript-test"
    assert verification_family("ruff check .") == "lint"
    assert verification_family("echo hello") is None


def test_failure_repair_pass_yields_three_actions(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    transcript = root / "project-a" / "session.jsonl"
    transcript.parent.mkdir(parents=True)
    lines = [
        _message(_use("old", "Write", {"file_path": "old.py", "content": "old"})),
        _message(_use("bad", "Edit", {"file_path": "x.py", "new_string": "bad"})),
        _message(_use("test-1", "Bash", {"command": "uv run pytest -q"})),
        _message(_result("test-1", "1 failed")),
        _message(_use("fix", "Edit", {"file_path": "x.py", "new_string": "fixed"})),
        _message(_use("test-2", "Bash", {"command": "pytest tests/test_x.py"})),
        _message(_result("test-2", "3 passed")),
    ]
    transcript.write_text("\n".join(lines))

    episodes = load_repair_episodes(
        transcript, root=root, max_per_action=1, distractors=1
    )
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.family == "pytest"
    assert [candidate.action for candidate in episode.candidates] == [
        "ignore",
        "revise",
        "strengthen",
    ]
    assert episode.failed_attempts == 1


def test_repeated_failure_marks_failed_repairs_as_revise(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    transcript = root / "project-a" / "session.jsonl"
    transcript.parent.mkdir(parents=True)
    lines = []
    for tool_id, text in (("bad", "first"), ("still-bad", "second"), ("good", "third")):
        lines.append(_message(_use(tool_id, "Edit", {"file_path": "x.py", "new_string": text})))
        test_id = f"test-{tool_id}"
        lines.append(_message(_use(test_id, "Bash", {"command": "pytest"})))
        lines.append(_message(_result(test_id, "2 passed" if tool_id == "good" else "1 failed")))
    transcript.write_text("\n".join(lines))

    episode = load_repair_episodes(transcript, root=root)[0]
    assert [candidate.action for candidate in episode.candidates] == [
        "revise",
        "revise",
        "strengthen",
    ]
    assert episode.failed_attempts == 2


def test_unresolved_failure_emits_no_episode(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    transcript = root / "p" / "session.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        "\n".join(
            [
                _message(_use("bad", "Write", {"file_path": "x", "content": "bad"})),
                _message(_use("test", "Bash", {"command": "pytest"})),
                _message(_result("test", "1 failed")),
            ]
        )
    )
    assert load_repair_episodes(transcript, root=root) == []
