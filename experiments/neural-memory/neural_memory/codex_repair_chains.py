from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .repair_chains import (
    RepairEpisode,
    _FailureChain,
    _Mutation,
    _episode,
    verification_family,
)
from .tool_outcomes import _is_mutation, classify_verification


_MUTATION_TOOLS = {
    "apply_patch",
    "ctx_edit",
    "ctx_patch",
    "edit",
    "write",
    "write_file",
}


@dataclass(frozen=True)
class _CodexCall:
    name: str
    input_text: str
    family: str | None
    order: int


def _payload_call(payload: dict[str, object]) -> tuple[str, str, str] | None:
    kind = payload.get("type")
    if kind not in {"function_call", "custom_tool_call"}:
        return None
    name = payload.get("name")
    call_id = payload.get("call_id") or payload.get("id")
    raw_input = payload.get("arguments") if kind == "function_call" else payload.get("input")
    if not isinstance(name, str) or not isinstance(call_id, str):
        return None
    if isinstance(raw_input, str):
        input_text = raw_input
    else:
        input_text = json.dumps(raw_input, ensure_ascii=False, sort_keys=True, default=str)
    return call_id, name, input_text[:32_000]


def _payload_result(payload: dict[str, object]) -> tuple[str, str] | None:
    if payload.get("type") not in {"function_call_output", "custom_tool_call_output"}:
        return None
    call_id = payload.get("call_id")
    if not isinstance(call_id, str):
        return None
    output = payload.get("output")
    if isinstance(output, str):
        return call_id, output[:32_000]
    return call_id, json.dumps(output, ensure_ascii=False, sort_keys=True, default=str)[:32_000]


def _is_codex_mutation(name: str, input_text: str) -> bool:
    lowered = name.lower()
    return (
        lowered in _MUTATION_TOOLS
        or any(word in lowered for word in ("write", "edit", "patch"))
        or _is_mutation(name, input_text)
    )


def load_codex_repair_episodes(
    path: Path,
    *,
    max_per_action: int = 4,
    distractors: int = 2,
) -> list[RepairEpisode]:
    if max_per_action < 1 or distractors < 0:
        raise ValueError(
            "candidate limits must be non-negative and max_per_action positive"
        )
    project = path.parent.name
    session_id = path.stem
    pending: dict[str, _CodexCall] = {}
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
            item_type = item.get("type")
            payload = item.get("payload")
            if not isinstance(payload, dict):
                continue
            if item_type == "session_meta":
                raw_session = payload.get("session_id") or payload.get("id")
                raw_project = payload.get("cwd")
                if isinstance(raw_session, str):
                    session_id = raw_session
                if isinstance(raw_project, str):
                    project = str(Path(raw_project).expanduser().resolve())
                continue
            if item_type == "turn_context" and isinstance(payload.get("cwd"), str):
                project = str(Path(str(payload["cwd"])).expanduser().resolve())
                continue
            if item_type != "response_item":
                continue
            call = _payload_call(payload)
            if call is not None:
                call_id, name, input_text = call
                order += 1
                family = verification_family(input_text)
                pending[call_id] = _CodexCall(name, input_text, family, order)
                if family is None and _is_codex_mutation(name, input_text):
                    digest = hashlib.sha256(call_id.encode()).hexdigest()[:16]
                    mutations.append(_Mutation(digest, f"{name} {input_text}", order))
                continue
            result = _payload_result(payload)
            if result is None:
                continue
            call_id, output = result
            pending_call = pending.pop(call_id, None)
            if pending_call is None or pending_call.family is None:
                continue
            passed = classify_verification(pending_call.input_text, output)
            if passed is None:
                continue
            family = pending_call.family
            current = [
                action for action in mutations if action.order < pending_call.order
            ]
            chain = chains.get(family)
            if not passed:
                lower_bound = chain.last_failure_order if chain else -1
                recent = [action for action in current if action.order > lower_bound]
                if chain is None:
                    chains[family] = _FailureChain(
                        output, recent[-max_per_action:], pending_call.order
                    )
                else:
                    chain.bad_actions.extend(recent[-max_per_action:])
                    chain.failure_result = output
                    chain.last_failure_order = pending_call.order
                    chain.failed_attempts += 1
                continue
            if chain is None:
                continue
            repair_actions = [
                action for action in current if action.order > chain.last_failure_order
            ]
            older = (
                [
                    action
                    for action in current
                    if action.order < min(item.order for item in chain.bad_actions)
                ]
                if chain.bad_actions
                else []
            )
            episode = _episode(
                project=project,
                session_id=session_id,
                family=family,
                chain=chain,
                repair_actions=repair_actions,
                older_actions=older,
                pass_result=output,
                max_per_action=max_per_action,
                distractors=distractors,
            )
            if episode is not None:
                episodes.append(episode)
            del chains[family]
    return episodes


def iter_codex_repair_episodes(
    root: str | Path,
    *,
    max_per_action: int = 4,
    distractors: int = 2,
) -> Iterator[RepairEpisode]:
    root = Path(root).expanduser()
    for path in sorted(root.rglob("*.jsonl")):
        yield from load_codex_repair_episodes(
            path,
            max_per_action=max_per_action,
            distractors=distractors,
        )
