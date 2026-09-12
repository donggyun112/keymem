from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

from .tool_outcomes import (
    _command,
    _input_text,
    _is_mutation,
    _tool_result_text,
    classify_verification,
)


RepairAction = Literal["ignore", "strengthen", "revise"]

_FAMILIES = (
    ("pytest", re.compile(r"\b(?:pytest|py\.test)\b", re.IGNORECASE)),
    (
        "javascript-test",
        re.compile(
            r"\b(?:npm|pnpm|yarn)\s+(?:run\s+)?test\b|\b(?:vitest|jest)\b",
            re.IGNORECASE,
        ),
    ),
    ("build", re.compile(r"\b(?:npm|pnpm|yarn)\s+(?:run\s+)?build\b|\btsc\b", re.IGNORECASE)),
    (
        "lint",
        re.compile(
            r"\bruff\s+check\b|\b(?:npm|pnpm|yarn)\s+(?:run\s+)?lint\b",
            re.IGNORECASE,
        ),
    ),
    ("types", re.compile(r"\bmypy\b|\btsc\b", re.IGNORECASE)),
    ("cargo", re.compile(r"\bcargo\s+(?:test|check)\b", re.IGNORECASE)),
    ("go", re.compile(r"\bgo\s+test\b", re.IGNORECASE)),
    ("ruby", re.compile(r"\brspec\b", re.IGNORECASE)),
    ("jvm", re.compile(r"\b(?:gradle\w*|mvn\w*)\s+test\b", re.IGNORECASE)),
)


@dataclass(frozen=True)
class RepairCandidate:
    action_id: str
    context: str
    action: RepairAction
    order: int


@dataclass(frozen=True)
class RepairEpisode:
    project: str
    session_id: str
    family: str
    candidates: tuple[RepairCandidate, ...]
    outcome_context: str
    failed_attempts: int


@dataclass(frozen=True)
class _Mutation:
    action_id: str
    context: str
    order: int


@dataclass
class _FailureChain:
    failure_result: str
    bad_actions: list[_Mutation]
    last_failure_order: int
    failed_attempts: int = 1


@dataclass(frozen=True)
class _ToolCall:
    name: str
    input_text: str
    command: str
    family: str | None
    order: int


def verification_family(command: str) -> str | None:
    for name, pattern in _FAMILIES:
        if pattern.search(command):
            return name
    return None


def _deduplicate(actions: list[_Mutation]) -> list[_Mutation]:
    seen: set[str] = set()
    result: list[_Mutation] = []
    for action in actions:
        if action.action_id in seen:
            continue
        seen.add(action.action_id)
        result.append(action)
    return result


def _episode(
    *,
    project: str,
    session_id: str,
    family: str,
    chain: _FailureChain,
    repair_actions: list[_Mutation],
    older_actions: list[_Mutation],
    pass_result: str,
    max_per_action: int,
    distractors: int,
) -> RepairEpisode | None:
    bad = _deduplicate(chain.bad_actions)[-max_per_action:]
    repaired = _deduplicate(repair_actions)[-max_per_action:]
    used = {action.action_id for action in [*bad, *repaired]}
    ignored = [action for action in reversed(older_actions) if action.action_id not in used]
    ignored = list(reversed(_deduplicate(ignored)[:distractors]))
    if not bad or not repaired:
        return None
    candidates = [
        *(RepairCandidate(item.action_id, item.context, "ignore", item.order) for item in ignored),
        *(RepairCandidate(item.action_id, item.context, "revise", item.order) for item in bad),
        *(
            RepairCandidate(item.action_id, item.context, "strengthen", item.order)
            for item in repaired
        ),
    ]
    candidates.sort(key=lambda item: item.order)
    outcome = f"failed verification\n{chain.failure_result}\npassed verification\n{pass_result}"
    return RepairEpisode(
        project=project,
        session_id=session_id,
        family=family,
        candidates=tuple(candidates),
        outcome_context=outcome,
        failed_attempts=chain.failed_attempts,
    )


def load_repair_episodes(
    path: Path,
    *,
    root: Path,
    max_per_action: int = 4,
    distractors: int = 2,
) -> list[RepairEpisode]:
    if max_per_action < 1 or distractors < 0:
        raise ValueError(
            "candidate limits must be non-negative and max_per_action positive"
        )
    try:
        project = path.relative_to(root).parts[0]
    except ValueError:
        project = path.parent.name
    session_id = path.stem
    pending: dict[str, _ToolCall] = {}
    mutations: deque[_Mutation] = deque(maxlen=64)
    chains: dict[str, _FailureChain] = {}
    episodes: list[RepairEpisode] = []
    order = 0

    try:
        lines = path.open(encoding="utf-8")
    except OSError:
        return episodes
    with lines:
        for line in lines:
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            raw_session = item.get("sessionId") or item.get("session_id")
            if isinstance(raw_session, str):
                session_id = raw_session
            message = item.get("message")
            if not isinstance(message, dict) or not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "tool_use":
                    tool_id = block.get("id")
                    name = block.get("name")
                    if not isinstance(tool_id, str) or not isinstance(name, str):
                        continue
                    order += 1
                    tool_input = block.get("input")
                    command = _command(tool_input)
                    family = verification_family(command)
                    pending[tool_id] = _ToolCall(
                        name=name,
                        input_text=_input_text(tool_input),
                        command=command,
                        family=family,
                        order=order,
                    )
                    if _is_mutation(name, command) and family is None:
                        digest = hashlib.sha256(tool_id.encode()).hexdigest()[:16]
                        mutations.append(
                            _Mutation(digest, f"{name} {_input_text(tool_input)}", order)
                        )
                elif block_type == "tool_result":
                    tool_id = block.get("tool_use_id")
                    if not isinstance(tool_id, str):
                        continue
                    call = pending.pop(tool_id, None)
                    if call is None or call.family is None:
                        continue
                    result = _tool_result_text(block)[:8_000]
                    passed = classify_verification(call.command, result)
                    if passed is None:
                        continue
                    family = call.family
                    current = [
                        action for action in mutations if action.order < call.order
                    ]
                    chain = chains.get(family)
                    if not passed:
                        lower_bound = chain.last_failure_order if chain else -1
                        recent = [action for action in current if action.order > lower_bound]
                        if chain is None:
                            chains[family] = _FailureChain(
                                result, recent[-max_per_action:], call.order
                            )
                        else:
                            chain.bad_actions.extend(recent[-max_per_action:])
                            chain.failure_result = result
                            chain.last_failure_order = call.order
                            chain.failed_attempts += 1
                        continue
                    if chain is None:
                        continue
                    repair_actions = [
                        action for action in current if action.order > chain.last_failure_order
                    ]
                    older = [
                        action
                        for action in current
                        if action.order < min(item.order for item in chain.bad_actions)
                    ] if chain.bad_actions else []
                    episode = _episode(
                        project=project,
                        session_id=session_id,
                        family=family,
                        chain=chain,
                        repair_actions=repair_actions,
                        older_actions=older,
                        pass_result=result,
                        max_per_action=max_per_action,
                        distractors=distractors,
                    )
                    if episode is not None:
                        episodes.append(episode)
                    del chains[family]
    return episodes


def iter_repair_episodes(
    root: str | Path,
    *,
    include_subagents: bool = False,
    max_per_action: int = 4,
    distractors: int = 2,
) -> Iterator[RepairEpisode]:
    root = Path(root).expanduser()
    for path in sorted(root.rglob("*.jsonl")):
        if not include_subagents and "subagents" in path.parts:
            continue
        yield from load_repair_episodes(
            path,
            root=root,
            max_per_action=max_per_action,
            distractors=distractors,
        )
