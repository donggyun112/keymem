from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import subprocess
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping

SCHEMA_VERSION = 1
FEATURE_DIMENSION = 256
MAX_CHANGED_FILES = 256
MAX_HASH_BYTES = 16 * 1024 * 1024

_VERIFY_COMMAND = re.compile(
    r"(?:\bpytest\b|\bpy\.test\b|\b(?:npm|pnpm|yarn)\s+(?:run\s+)?(?:test|build|lint)\b|"
    r"\bbun\s+test\b|\bcargo\s+(?:test|check)\b|\bgo\s+test\b|"
    r"\b(?:vitest|jest|rspec|mypy|tsc)\b|\bruff\s+check\b|\bgradle\w*\s+test\b|"
    r"\bmvn\w*\s+test\b)",
    re.IGNORECASE,
)
_MUTATION_COMMAND = re.compile(
    r"(?:\bapply_patch\b|\bsed\s+-i\b|\bperl\s+-pi\b|"
    r"\b(?:ruff|prettier)\b[^\n]*(?:--fix|--write)\b)",
    re.IGNORECASE,
)
_FAILURE = re.compile(
    r"(?:\b\d+\s+failed\b|={2,}\s*failures?\s*={2,}|\bbuild failed\b|"
    r"\btests? failed\b|\bcommand failed\b|\bexit code\s*[1-9]\d*\b|"
    r"\bnpm err!\b|\bfound\s+[1-9]\d*\s+errors?\b|\b(?:error|failed):\s)",
    re.IGNORECASE,
)
_SUCCESS = re.compile(
    r"(?:\b\d+\s+passed\b|\bbuild success(?:ful)?\b|\ball tests passed\b|"
    r"\btests?:\s*\d+\s+passed\b|\bfound 0 errors?\b|"
    r"\bsuccessfully compiled\b|\bno issues found\b)",
    re.IGNORECASE,
)
_CORRECTION = re.compile(
    r"^(?:ㄴㄴ+|아니(?:야|지|요)?\b|그게\s*아니|아님\b|no[,!.\s]|wrong\b|"
    r"actually[,!.\s]|revert\b|undo\b)|"
    r"(?:잘못(?:됐|된|했|알)|틀렸|안\s*돼|안돼|하지\s*마|"
    r"말고\b|되돌려|취소해)",
    re.IGNORECASE,
)
_ACCEPTANCE = re.compile(
    r"^(?:ㅇㅇ+|응+|어+\b|그래+\b|맞아\b|좋아\b|좋네\b|굳+\b|굿+\b|"
    r"오케이\b|계속\b|ㄱㄱ+|yes\b|yep\b|ok(?:ay)?\b|good\b|great\b|"
    r"perfect\b|works?\b|continue\b)",
    re.IGNORECASE,
)
_MUTATION_TOOLS = {
    "Edit",
    "Write",
    "MultiEdit",
    "NotebookEdit",
    "mcp__lean_ctx__ctx_patch",
    "mcp__lean_ctx__ctx_edit",
}
_PATH_KEYS = ("file_path", "path", "notebook_path")

ActionLabel = Literal["strengthen", "revise", "forget"]


@dataclass(frozen=True)
class CreditExample:
    session_id: str
    outcome_id: str
    action_ids: tuple[str, ...]
    action_features: tuple[str, ...]
    outcome_feature: str
    label: ActionLabel
    source: str


def _digest(value: str | bytes, salt: bytes = b"") -> str:
    raw = value.encode() if isinstance(value, str) else value
    return hashlib.blake2b(raw, key=salt, digest_size=16).hexdigest()


def signed_ngram_sketch(
    text: str, *, dimension: int = FEATURE_DIMENSION, salt: bytes = b""
) -> str:
    """Return a normalized int8 feature vector without retaining source text."""

    if dimension < 1:
        raise ValueError("dimension must be positive")
    normalized = " ".join(text.lower().split())
    values = [0] * dimension
    wrapped = f"^{normalized}$"
    for width in (1, 2, 3, 4):
        for start in range(max(0, len(wrapped) - width + 1)):
            token = wrapped[start : start + width].encode()
            hashed = int.from_bytes(
                hashlib.blake2b(token, key=salt, digest_size=8).digest(), "big"
            )
            values[hashed % dimension] += 1 if hashed & 1 else -1
    norm = math.sqrt(sum(value * value for value in values))
    quantized = (
        bytes(
            (max(-127, min(127, round(value * 127 / norm))) % 256)
            for value in values
        )
        if norm
        else bytes(dimension)
    )
    return base64.b64encode(quantized).decode("ascii")


def decode_sketch(encoded: str) -> tuple[float, ...]:
    raw = base64.b64decode(encoded, validate=True)
    values = tuple((byte if byte < 128 else byte - 256) / 127.0 for byte in raw)
    norm = math.sqrt(sum(value * value for value in values))
    return tuple(value / norm for value in values) if norm else values


def _command(tool_input: object) -> str:
    if not isinstance(tool_input, Mapping):
        return ""
    for key in ("command", "cmd"):
        value = tool_input.get(key)
        if isinstance(value, str):
            return value
    return ""


def _classify_outcome(text: str) -> Literal["accept", "correct"] | None:
    normalized = " ".join(text.strip().split()).lower()
    if _CORRECTION.search(normalized):
        return "correct"
    if _ACCEPTANCE.search(normalized):
        return "accept"
    return None


def _classify_verification(command: str, result: str) -> bool | None:
    if not _VERIFY_COMMAND.search(command):
        return None
    if _FAILURE.search(result):
        return False
    if _SUCCESS.search(result):
        return True
    return None


def classify_tool(
    tool_name: str, tool_input: object
) -> Literal["mutation", "verification", "other"]:
    command = _command(tool_input)
    if _VERIFY_COMMAND.search(command):
        return "verification"
    lowered = tool_name.lower()
    if tool_name in _MUTATION_TOOLS or any(
        word in lowered for word in ("write", "edit", "patch", "update")
    ):
        return "mutation"
    if _MUTATION_COMMAND.search(command):
        return "mutation"
    return "other"


def _feature_text(tool_name: str, tool_input: object) -> str:
    if not isinstance(tool_input, Mapping):
        return tool_name
    # Values are used transiently to make the sketch; none are copied to the event.
    serialized = json.dumps(
        tool_input, ensure_ascii=False, sort_keys=True, default=str
    )[:32_000]
    return f"{tool_name} {serialized}"


def _target_paths(tool_input: object, cwd: Path) -> list[Path]:
    if not isinstance(tool_input, Mapping):
        return []
    paths: list[Path] = []
    for key in _PATH_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str):
            path = Path(value).expanduser()
            paths.append(path if path.is_absolute() else cwd / path)
    return paths


def _git_root(cwd: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=1,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    root = Path(result.stdout.strip()).resolve()
    return root if root.is_dir() else None


def _changed_paths(root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=root,
            capture_output=True,
            timeout=2,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    paths: list[Path] = []
    fields = result.stdout.split(b"\0")
    index = 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if len(field) < 4:
            continue
        status = field[:2]
        raw_path = field[3:]
        if status[:1] in {b"R", b"C"} and index < len(fields):
            raw_path = fields[index]
            index += 1
        try:
            relative = os.fsdecode(raw_path)
        except UnicodeDecodeError:
            continue
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        paths.append(candidate)
    return paths[:MAX_CHANGED_FILES]


def _file_record(path: Path, *, root: Path, salt: bytes) -> dict[str, object]:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        relative = path.name
    record: dict[str, object] = {"path_id": _digest(relative, salt)}
    try:
        stat = path.stat()
    except OSError:
        return {**record, "exists": False}
    record.update({"exists": True, "size": stat.st_size})
    if path.is_file() and stat.st_size <= MAX_HASH_BYTES:
        try:
            record["content_digest"] = _digest(path.read_bytes(), salt)
        except OSError:
            pass
    return record


def workspace_snapshot(
    cwd: Path,
    *,
    salt: bytes,
    targets: Iterable[Path] = (),
    exclude_roots: Iterable[Path] = (),
) -> dict[str, object]:
    root = _git_root(cwd) or cwd.resolve()
    paths = {path.resolve() for path in targets}
    paths.update(_changed_paths(root))
    excluded = tuple(path.resolve() for path in exclude_roots)
    paths = {
        path
        for path in paths
        if not any(
            path == excluded_path or excluded_path in path.parents
            for excluded_path in excluded
        )
    }
    files = [_file_record(path, root=root, salt=salt) for path in sorted(paths, key=str)]
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "root_id": _digest(str(root), salt),
        "digest": _digest(canonical),
        "files": files,
        "truncated": len(paths) >= MAX_CHANGED_FILES,
    }


def load_or_create_salt(state_dir: Path) -> bytes:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / ".salt"
    try:
        value = path.read_bytes()
    except OSError:
        value = os.urandom(32)
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return path.read_bytes()
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
    return value


def _safe_response_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)[:32_000]


class ProspectiveLogger:
    """Append-only observer for delayed action credit; never writes raw hook payloads."""

    def __init__(self, event_path: Path, *, salt: bytes | None = None) -> None:
        self.event_path = event_path
        self.state_dir = event_path.parent / ".prospective-state"
        self.salt = salt or load_or_create_salt(self.state_dir)

    def handle(self, payload: Mapping[str, object]) -> dict[str, object] | None:
        event_name = payload.get("hook_event_name")
        session = payload.get("session_id")
        cwd_value = payload.get("cwd")
        if not isinstance(event_name, str) or not isinstance(session, str):
            return None
        if event_name not in {
            "PreToolUse",
            "PostToolUse",
            "PostToolUseFailure",
            "UserPromptSubmit",
            "Stop",
            "SessionEnd",
        }:
            return None
        cwd = Path(cwd_value) if isinstance(cwd_value, str) else Path.cwd()
        session_id = _digest(session, self.salt)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        lock_descriptor = os.open(
            self.state_dir / f"{session_id}.lock", os.O_WRONLY | os.O_CREAT, 0o600
        )
        try:
            if hasattr(os, "lockf"):
                os.lockf(lock_descriptor, os.F_LOCK, 0)
            state = self._load_state(session_id)
            if event_name == "PreToolUse":
                event = self._pre_tool(payload, cwd, session_id, state)
            elif event_name in {"PostToolUse", "PostToolUseFailure"}:
                event = self._post_tool(payload, cwd, session_id, state)
            elif event_name == "UserPromptSubmit":
                event = self._prompt(payload, cwd, session_id, state)
            else:
                event = self._boundary(payload, cwd, session_id, state)
            self._append(event)
            self._save_state(session_id, state)
        finally:
            os.close(lock_descriptor)
        return event

    def _base(self, *, kind: str, session_id: str, cwd: Path) -> dict[str, object]:
        return {
            "schema": SCHEMA_VERSION,
            "time_ns": time.time_ns(),
            "kind": kind,
            "session_id": session_id,
            "cwd_id": _digest(str(cwd.resolve()), self.salt),
        }

    def _snapshot(self, cwd: Path, tool_input: object = None) -> dict[str, object]:
        return workspace_snapshot(
            cwd,
            salt=self.salt,
            targets=_target_paths(tool_input, cwd),
            exclude_roots=(self.event_path, self.state_dir),
        )

    def _action_id(self, payload: Mapping[str, object]) -> str:
        raw = payload.get("tool_use_id")
        return _digest(raw if isinstance(raw, str) else f"missing:{time.time_ns()}", self.salt)

    def _pre_tool(
        self, payload: Mapping[str, object], cwd: Path, session_id: str, state: dict[str, object]
    ) -> dict[str, object]:
        name = payload.get("tool_name") if isinstance(payload.get("tool_name"), str) else "unknown"
        tool_input = payload.get("tool_input")
        action_id = self._action_id(payload)
        category = classify_tool(name, tool_input)
        snapshot = self._snapshot(cwd, tool_input)
        event = {
            **self._base(kind="candidate", session_id=session_id, cwd=cwd),
            "action_id": action_id,
            "tool_id": _digest(name, self.salt),
            "category": category,
            "feature": signed_ngram_sketch(_feature_text(name, tool_input), salt=self.salt),
            "workspace_before": snapshot,
        }
        pending = state.setdefault("pending", {})
        assert isinstance(pending, dict)
        pending[action_id] = {
            "category": category,
            "workspace_before": snapshot,
            "feature": event["feature"],
        }
        return event

    def _post_tool(
        self, payload: Mapping[str, object], cwd: Path, session_id: str, state: dict[str, object]
    ) -> dict[str, object]:
        name = payload.get("tool_name") if isinstance(payload.get("tool_name"), str) else "unknown"
        tool_input = payload.get("tool_input")
        action_id = self._action_id(payload)
        pending = state.setdefault("pending", {})
        assert isinstance(pending, dict)
        prior = pending.pop(action_id, {})
        category = prior.get("category") if isinstance(prior, dict) else None
        if category not in {"mutation", "verification", "other"}:
            category = classify_tool(name, tool_input)
        after = self._snapshot(cwd, tool_input)
        before = prior.get("workspace_before") if isinstance(prior, dict) else None
        changed = isinstance(before, dict) and before.get("digest") != after.get("digest")
        success = payload.get("hook_event_name") == "PostToolUse"
        if category == "other" and success and changed:
            category = "mutation"
        response_text = _safe_response_text(payload.get("tool_response") or payload.get("error"))
        event: dict[str, object] = {
            **self._base(kind="action_result", session_id=session_id, cwd=cwd),
            "action_id": action_id,
            "category": category,
            "tool_succeeded": success,
            "workspace_after": after,
            "workspace_changed": changed,
            "outcome_feature": signed_ngram_sketch(response_text, salt=self.salt),
        }
        recent = deque(state.get("recent_mutations", []), maxlen=8)
        if category == "mutation" and success and changed:
            event["parent_action_ids"] = list(state.get("failed_credit", []))
            recent.append(action_id)
            state["recent_mutations"] = list(recent)
            memory_actions = deque(state.get("memory_actions", []), maxlen=32)
            memory_actions.append(action_id)
            state["memory_actions"] = list(memory_actions)
            state["turn_mutations"] = [*state.get("turn_mutations", []), action_id]
            state["failed_credit"] = []
        elif category == "verification":
            verification = _classify_verification(_command(tool_input), response_text)
            if verification is None and not success:
                verification = False
            credit = list(recent)
            event.update({"verification_passed": verification, "credit_action_ids": credit})
            if verification is False:
                state["failed_credit"] = credit
            elif verification is True:
                state["failed_credit"] = []
                state["recent_mutations"] = []
        return event

    def _prompt(
        self, payload: Mapping[str, object], cwd: Path, session_id: str, state: dict[str, object]
    ) -> dict[str, object]:
        prompt = payload.get("prompt") if isinstance(payload.get("prompt"), str) else ""
        label = _classify_outcome(prompt)
        event = {
            **self._base(kind="user_feedback", session_id=session_id, cwd=cwd),
            "feature": signed_ngram_sketch(prompt, salt=self.salt),
            "feedback": (
                "strengthen"
                if label == "accept"
                else "revise"
                if label == "correct"
                else None
            ),
            "credit_action_ids": list(state.get("last_turn_mutations", [])),
            "candidate_action_ids": list(state.get("memory_actions", [])),
        }
        return event

    def _boundary(
        self, payload: Mapping[str, object], cwd: Path, session_id: str, state: dict[str, object]
    ) -> dict[str, object]:
        kind = str(payload.get("hook_event_name")).lower()
        event = {
            **self._base(kind=kind, session_id=session_id, cwd=cwd),
            "workspace": self._snapshot(cwd),
            "turn_action_ids": list(state.get("turn_mutations", [])),
        }
        message = payload.get("last_assistant_message")
        if isinstance(message, str):
            event["assistant_feature"] = signed_ngram_sketch(message, salt=self.salt)
        if kind == "stop":
            state["last_turn_mutations"] = list(state.get("turn_mutations", []))
            state["turn_mutations"] = []
        return event

    def _state_path(self, session_id: str) -> Path:
        return self.state_dir / f"{session_id}.json"

    def _load_state(self, session_id: str) -> dict[str, object]:
        try:
            value = json.loads(self._state_path(session_id).read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _save_state(self, session_id: str, state: dict[str, object]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        target = self._state_path(session_id)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, separators=(",", ":")))
        os.chmod(temporary, 0o600)
        temporary.replace(target)

    def _append(self, event: Mapping[str, object]) -> None:
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode()
        descriptor = os.open(self.event_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            if hasattr(os, "lockf"):
                os.lockf(descriptor, os.F_LOCK, 0)
            os.write(descriptor, encoded)
        finally:
            os.close(descriptor)


def iter_credit_examples(events: Iterable[Mapping[str, object]]) -> Iterable[CreditExample]:
    candidates: dict[str, Mapping[str, object]] = {}
    results: dict[str, Mapping[str, object]] = {}
    by_session: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for event in events:
        session_id = event.get("session_id")
        if isinstance(session_id, str):
            by_session[session_id].append(event)
        action_id = event.get("action_id")
        if not isinstance(action_id, str):
            continue
        if event.get("kind") == "candidate":
            candidates[action_id] = event
        elif event.get("kind") == "action_result":
            results[action_id] = event

    emitted: set[tuple[str, str, str]] = set()
    for session_id, session_events in by_session.items():
        for index, event in enumerate(session_events):
            source = ""
            label: ActionLabel | None = None
            ids: list[str] = []
            if event.get("kind") == "action_result" and event.get("category") == "verification":
                passed = event.get("verification_passed")
                if isinstance(passed, bool):
                    label = "strengthen" if passed else "revise"
                    source = "verification"
                    ids = [
                        value
                        for value in event.get("credit_action_ids", [])
                        if isinstance(value, str)
                    ]
            elif event.get("kind") == "user_feedback" and event.get("feedback") in {
                "strengthen",
                "revise",
            }:
                label = event["feedback"]  # type: ignore[assignment]
                source = "explicit_feedback"
                ids = [
                    value
                    for value in event.get("credit_action_ids", [])
                    if isinstance(value, str)
                ]
            if label is None or not ids:
                continue
            available = [action_id for action_id in ids if action_id in candidates]
            if not available:
                continue
            outcome_id = _digest(json.dumps(event, sort_keys=True, default=str))
            key = (session_id, outcome_id, label)
            if key in emitted:
                continue
            emitted.add(key)
            yield CreditExample(
                session_id=session_id,
                outcome_id=outcome_id,
                action_ids=tuple(available),
                action_features=tuple(
                    str(candidates[action_id].get("feature", ""))
                    for action_id in available
                ),
                outcome_feature=str(event.get("outcome_feature") or event.get("feature") or ""),
                label=label,
                source=source,
            )

        # Exact state restoration is a later, action-specific forget signal rather
        # than a relabeling of the earlier global test outcome.
        for action_id, candidate in candidates.items():
            if candidate.get("session_id") != session_id:
                continue
            result = results.get(action_id)
            before = candidate.get("workspace_before")
            if not isinstance(result, Mapping) or not isinstance(before, Mapping):
                continue
            try:
                result_index = session_events.index(result)
            except ValueError:
                continue
            before_digest = before.get("digest")
            for later in session_events[result_index + 1 :]:
                later_workspace = later.get("workspace_after") or later.get("workspace")
                if (
                    not isinstance(later_workspace, Mapping)
                    or later_workspace.get("digest") != before_digest
                ):
                    continue
                outcome_id = _digest(json.dumps(later, sort_keys=True, default=str))
                key = (session_id, outcome_id, "forget")
                if key not in emitted:
                    emitted.add(key)
                    yield CreditExample(
                        session_id=session_id,
                        outcome_id=outcome_id,
                        action_ids=(action_id,),
                        action_features=(str(candidate.get("feature", "")),),
                        outcome_feature=str(
                            later.get("outcome_feature")
                            or later.get("assistant_feature")
                            or ""
                        ),
                        label="forget",
                        source="exact_revert",
                    )
                break


def load_events(path: Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    try:
        lines = path.open(encoding="utf-8")
    except OSError:
        return events
    with lines:
        for line in lines:
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(event, dict) and event.get("schema") == SCHEMA_VERSION:
                events.append(event)
    return events
