from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


_VERIFY_COMMAND = re.compile(
    r"(?:\bpytest\b|\bpy\.test\b|\bnpm\s+(?:run\s+)?test\b|"
    r"\bpnpm\s+(?:run\s+)?test\b|\byarn\s+(?:run\s+)?test\b|"
    r"\bbun\s+test\b|\bcargo\s+test\b|\bgo\s+test\b|"
    r"\b(?:vitest|jest|rspec)\b|\bruff\s+check\b|\bmypy\b|"
    r"\btsc\b|\bnpm\s+run\s+build\b|\bpnpm\s+(?:run\s+)?build\b|"
    r"\bgradle\w*\s+test\b|\bmvn\w*\s+test\b)",
    re.IGNORECASE,
)
_MUTATION_COMMAND = re.compile(
    r"(?:\bapply_patch\b|\bsed\s+-i\b|\bperl\s+-pi\b|"
    r"\bruff\s+.*--fix\b|\bprettier\s+.*--write\b)",
    re.IGNORECASE,
)
_FAILURE = re.compile(
    r"(?:\b\d+\s+failed\b|={2,}\s*failures?\s*={2,}|\bbuild failed\b|"
    r"\btests? failed\b|\bcommand failed\b|\bexit code\s*[1-9]\d*\b|"
    r"\bnpm err!\b|\bfound\s+[1-9]\d*\s+errors?\b|"
    r"\b(?:error|failed):\s)",
    re.IGNORECASE,
)
_SUCCESS = re.compile(
    r"(?:\b\d+\s+passed\b|\bbuild success(?:ful)?\b|\ball tests passed\b|"
    r"\btests?:\s*\d+\s+passed\b|\bfound 0 errors?\b|"
    r"\bsuccessfully compiled\b|\bno issues found\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VerificationEvent:
    project: str
    session_id: str
    action_context: str
    result: str
    passed: bool
    mutation_count: int


@dataclass(frozen=True)
class _ToolCall:
    name: str
    input_text: str
    command: str
    verifies: bool


def classify_verification(command: str, result: str) -> bool | None:
    if not _VERIFY_COMMAND.search(command):
        return None
    if _FAILURE.search(result):
        return False
    if _SUCCESS.search(result):
        return True
    return None


def _tool_result_text(block: dict[str, object]) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        text = item.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def _input_text(tool_input: object) -> str:
    if not isinstance(tool_input, dict):
        return ""
    return json.dumps(tool_input, ensure_ascii=False, sort_keys=True)[:8_000]


def _command(tool_input: object) -> str:
    if not isinstance(tool_input, dict):
        return ""
    for key in ("command", "cmd"):
        value = tool_input.get(key)
        if isinstance(value, str):
            return value
    return ""


def _is_mutation(name: str, command: str) -> bool:
    if name in {"Edit", "Write", "MultiEdit", "mcp__lean-ctx__ctx_patch"}:
        return True
    return bool(_MUTATION_COMMAND.search(command))


def load_verification_events(
    path: Path,
    *,
    root: Path,
    recent_mutations: int = 4,
) -> list[VerificationEvent]:
    try:
        project = path.relative_to(root).parts[0]
    except ValueError:
        project = path.parent.name
    session_id = path.stem
    pending: dict[str, _ToolCall] = {}
    mutations: deque[str] = deque(maxlen=recent_mutations)
    events: list[VerificationEvent] = []

    try:
        lines = path.open(encoding="utf-8")
    except OSError:
        return events
    with lines:
        for line in lines:
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            raw_session_id = item.get("sessionId") or item.get("session_id")
            if isinstance(raw_session_id, str):
                session_id = raw_session_id
            message = item.get("message")
            if not isinstance(message, dict) or not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    tool_id = block.get("id")
                    name = block.get("name")
                    if not isinstance(tool_id, str) or not isinstance(name, str):
                        continue
                    tool_input = block.get("input")
                    command = _command(tool_input)
                    verifies = bool(_VERIFY_COMMAND.search(command))
                    pending[tool_id] = _ToolCall(
                        name, _input_text(tool_input), command, verifies
                    )
                    if _is_mutation(name, command) and not verifies:
                        mutations.append(f"{name} {pending[tool_id].input_text}")
                elif block.get("type") == "tool_result":
                    tool_id = block.get("tool_use_id")
                    if not isinstance(tool_id, str):
                        continue
                    call = pending.pop(tool_id, None)
                    if call is None or not call.verifies:
                        continue
                    result = _tool_result_text(block)[:8_000]
                    passed = classify_verification(call.command, result)
                    if passed is None:
                        continue
                    action_parts = list(mutations)
                    action_parts.append(f"verify {call.input_text}")
                    events.append(
                        VerificationEvent(
                            project=project,
                            session_id=session_id,
                            action_context="\n".join(action_parts),
                            result=result,
                            passed=passed,
                            mutation_count=len(mutations),
                        )
                    )
                    mutations.clear()
    return events


def iter_verification_events(
    root: str | Path,
    *,
    include_subagents: bool = False,
    recent_mutations: int = 4,
) -> Iterator[VerificationEvent]:
    root = Path(root).expanduser()
    for path in sorted(root.rglob("*.jsonl")):
        if not include_subagents and "subagents" in path.parts:
            continue
        yield from load_verification_events(
            path, root=root, recent_mutations=recent_mutations
        )
