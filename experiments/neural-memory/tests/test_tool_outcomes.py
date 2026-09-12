import json
from pathlib import Path

import pytest

from neural_memory.tool_outcomes import (
    classify_verification,
    load_verification_events,
)


@pytest.mark.parametrize(
    ("command", "result", "expected"),
    [
        ("uv run pytest -q", "42 passed in 1.2s", True),
        ("pnpm test", "Tests: 18 passed", True),
        ("cargo test", "2 failed, 9 passed", False),
        ("npm run build", "BUILD FAILED with an error", False),
        ("git status", "clean", None),
        ("pytest", "no recognizable summary", None),
    ],
)
def test_classify_verification(command: str, result: str, expected: bool | None) -> None:
    assert classify_verification(command, result) is expected


def test_load_verification_event_pairs_mutation_and_result(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    rows = [
        {
            "sessionId": "s1",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "edit-1",
                        "name": "Edit",
                        "input": {"file_path": "src/app.py", "new_string": "fixed"},
                    },
                    {
                        "type": "tool_use",
                        "id": "test-1",
                        "name": "Bash",
                        "input": {"command": "uv run pytest -q"},
                    },
                ]
            },
        },
        {
            "sessionId": "s1",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "test-1",
                        "content": [{"type": "text", "text": "7 passed in 0.2s"}],
                    }
                ]
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))

    events = load_verification_events(path, root=tmp_path)

    assert len(events) == 1
    assert events[0].passed is True
    assert events[0].mutation_count == 1
    assert "src/app.py" in events[0].action_context
