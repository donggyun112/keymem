from __future__ import annotations

import json
from pathlib import Path

from neural_memory.codex_repair_chains import load_codex_repair_episodes


def _record(item_type: str, payload: dict[str, object]) -> str:
    return json.dumps({"type": item_type, "payload": payload})


def _call(call_id: str, name: str, arguments: object) -> str:
    return _record(
        "response_item",
        {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": json.dumps(arguments),
        },
    )


def _output(call_id: str, output: str) -> str:
    return _record(
        "response_item",
        {"type": "function_call_output", "call_id": call_id, "output": output},
    )


def test_codex_function_calls_form_repair_episode(tmp_path: Path) -> None:
    transcript = tmp_path / "rollout.jsonl"
    records = [
        _record(
            "session_meta",
            {"type": "session_meta", "session_id": "s", "cwd": str(tmp_path)},
        ),
        _call("bad", "apply_patch", "bad patch"),
        _output("bad", "done"),
        _call("test-1", "exec_command", {"cmd": "uv run pytest -q"}),
        _output("test-1", "1 failed"),
        _call("fix", "ctx_edit", {"path": "x.py", "content": "fix"}),
        _output("fix", "done"),
        _call("test-2", "ctx_shell", {"command": "pytest tests/test_x.py"}),
        _output("test-2", "4 passed"),
    ]
    transcript.write_text("\n".join(records))

    episode = load_codex_repair_episodes(transcript)[0]
    assert episode.project == str(tmp_path.resolve())
    assert episode.session_id == "s"
    assert [candidate.action for candidate in episode.candidates] == [
        "revise",
        "strengthen",
    ]


def test_codex_custom_tool_calls_are_supported(tmp_path: Path) -> None:
    transcript = tmp_path / "rollout.jsonl"
    records = [
        _record(
            "response_item",
            {
                "type": "custom_tool_call",
                "call_id": "bad",
                "name": "apply_patch",
                "input": "*** patch",
            },
        ),
        _record(
            "response_item",
            {
                "type": "custom_tool_call_output",
                "call_id": "bad",
                "output": "done",
            },
        ),
        _record(
            "response_item",
            {
                "type": "custom_tool_call",
                "call_id": "test-1",
                "name": "exec",
                "input": 'tools.exec_command({cmd:"pytest"})',
            },
        ),
        _record(
            "response_item",
            {
                "type": "custom_tool_call_output",
                "call_id": "test-1",
                "output": "1 failed",
            },
        ),
        _record(
            "response_item",
            {
                "type": "custom_tool_call",
                "call_id": "fix",
                "name": "exec",
                "input": 'tools.apply_patch("fix")',
            },
        ),
        _record(
            "response_item",
            {
                "type": "custom_tool_call_output",
                "call_id": "fix",
                "output": "done",
            },
        ),
        _record(
            "response_item",
            {
                "type": "custom_tool_call",
                "call_id": "test-2",
                "name": "exec",
                "input": 'tools.exec_command({cmd:"pytest -q"})',
            },
        ),
        _record(
            "response_item",
            {
                "type": "custom_tool_call_output",
                "call_id": "test-2",
                "output": "5 passed",
            },
        ),
    ]
    transcript.write_text("\n".join(records))
    assert len(load_codex_repair_episodes(transcript)) == 1
