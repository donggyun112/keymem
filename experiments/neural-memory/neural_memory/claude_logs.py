from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


_HIDDEN_BLOCK = re.compile(
    r"<(?:system-reminder|local-command-caveat|command-name|command-message)\b[^>]*>"
    r".*?</(?:system-reminder|local-command-caveat|command-name|command-message)>",
    re.DOTALL | re.IGNORECASE,
)
_TOKEN = re.compile(r"[A-Za-z0-9_./:@-]{3,}|[가-힣]{2,}")
_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "before",
    "could",
    "from",
    "have",
    "into",
    "just",
    "make",
    "more",
    "should",
    "that",
    "then",
    "there",
    "these",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "같아",
    "그거",
    "그냥",
    "근데",
    "다시",
    "대한",
    "되고",
    "되면",
    "만들",
    "뭔가",
    "여기",
    "이거",
    "있는",
    "있어",
    "지금",
    "하면",
    "해서",
}


@dataclass(frozen=True)
class Conversation:
    project: str
    session_id: str
    source: Path
    turns: tuple[str, ...]


@dataclass(frozen=True)
class RevisitExample:
    project: str
    session_id: str
    query_index: int
    candidate_indices: tuple[int, ...]
    target_offset: int
    overlap_score: float
    margin: float

    @property
    def delay(self) -> int:
        return self.query_index - self.candidate_indices[self.target_offset]


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def clean_user_text(content: object, *, max_chars: int = 4_000) -> str:
    text = _HIDDEN_BLOCK.sub(" ", _content_text(content))
    text = " ".join(text.split()).strip()
    return text[:max_chars]


def load_conversation(
    path: Path,
    *,
    root: Path,
    min_chars: int = 12,
    max_chars: int = 4_000,
) -> Conversation | None:
    turns: list[str] = []
    session_id = path.stem
    try:
        project = path.relative_to(root).parts[0]
    except ValueError:
        project = path.parent.name

    try:
        lines = path.open(encoding="utf-8")
    except OSError:
        return None
    with lines:
        for line in lines:
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if item.get("type") != "user" or item.get("isSidechain") is True:
                continue
            message = item.get("message")
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            text = clean_user_text(message.get("content"), max_chars=max_chars)
            if len(text) < min_chars or (turns and text == turns[-1]):
                continue
            turns.append(text)
            raw_session_id = item.get("sessionId") or item.get("session_id")
            if isinstance(raw_session_id, str):
                session_id = raw_session_id
    if len(turns) < 3:
        return None
    return Conversation(project, session_id, path, tuple(turns))


def iter_conversations(
    root: str | Path,
    *,
    include_subagents: bool = False,
    min_chars: int = 12,
    max_chars: int = 4_000,
) -> Iterator[Conversation]:
    root = Path(root).expanduser()
    for path in sorted(root.rglob("*.jsonl")):
        if not include_subagents and "subagents" in path.parts:
            continue
        conversation = load_conversation(
            path,
            root=root,
            min_chars=min_chars,
            max_chars=max_chars,
        )
        if conversation is not None:
            yield conversation


def iter_history_conversations(
    path: str | Path,
    *,
    gap_seconds: int = 6 * 60 * 60,
    min_chars: int = 12,
    max_chars: int = 4_000,
) -> Iterator[Conversation]:
    """Sessionize Claude's global prompt history by project and idle gap."""

    if gap_seconds < 1:
        raise ValueError("gap_seconds must be positive")
    path = Path(path).expanduser()
    active: dict[str, tuple[int, int, list[str]]] = {}
    completed: list[tuple[int, Conversation]] = []

    def finish(project: str, start: int, turns: list[str]) -> None:
        if len(turns) < 3:
            return
        digest = hashlib.sha256(f"{project}:{start}".encode()).hexdigest()[:16]
        completed.append(
            (start, Conversation(project, f"history-{digest}", path, tuple(turns)))
        )

    with path.open(encoding="utf-8") as lines:
        for line in lines:
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            project = item.get("project")
            timestamp = item.get("timestamp")
            if not isinstance(project, str) or not isinstance(timestamp, int):
                continue
            text = clean_user_text(item.get("display"), max_chars=max_chars)
            if len(text) < min_chars:
                continue
            previous = active.get(project)
            if previous is None:
                active[project] = (timestamp, timestamp, [text])
                continue
            start, last, turns = previous
            if timestamp - last > gap_seconds * 1_000:
                finish(project, start, turns)
                active[project] = (timestamp, timestamp, [text])
            else:
                if text != turns[-1]:
                    turns.append(text)
                active[project] = (start, timestamp, turns)

    for project, (start, _, turns) in active.items():
        finish(project, start, turns)
    for _, conversation in sorted(completed, key=lambda item: item[0]):
        yield conversation


def content_tokens(text: str) -> frozenset[str]:
    return frozenset(
        token
        for raw in _TOKEN.findall(text.lower())
        if (token := raw.strip("./-_:"))
        and token not in _STOPWORDS
        and not token.isdigit()
    )


def _token_weight(token: str) -> float:
    if any(mark in token for mark in ("/", ".", "_", "-", "@")):
        return 2.0
    if len(token) >= 9:
        return 1.5
    return 1.0


def overlap_score(left: Iterable[str], right: Iterable[str]) -> float:
    return sum(_token_weight(token) for token in set(left).intersection(right))


def iter_revisit_examples(
    conversation: Conversation,
    *,
    candidates: int = 8,
    min_delay: int = 2,
    min_overlap: float = 2.0,
    min_margin: float = 0.5,
) -> Iterator[RevisitExample]:
    """Weak-label delayed recalls using later reuse of specific lexical anchors.

    The query itself supplies no pointer to an earlier turn. A label is emitted only
    when one eligible history turn has materially more token overlap than its peers.
    The label is used as a noisy teacher signal, never as a model input.
    """

    if candidates < 2:
        raise ValueError("candidates must be at least two")
    if min_delay < 1:
        raise ValueError("min_delay must be positive")
    tokens = [content_tokens(turn) for turn in conversation.turns]
    for query_index in range(candidates + min_delay - 1, len(tokens)):
        end = query_index - min_delay + 1
        start = end - candidates
        candidate_indices = tuple(range(start, end))
        scores = [
            overlap_score(tokens[index], tokens[query_index])
            for index in candidate_indices
        ]
        order = sorted(range(candidates), key=scores.__getitem__, reverse=True)
        best, second = order[:2]
        margin = scores[best] - scores[second]
        if scores[best] < min_overlap or margin < min_margin:
            continue
        yield RevisitExample(
            project=conversation.project,
            session_id=conversation.session_id,
            query_index=query_index,
            candidate_indices=candidate_indices,
            target_offset=best,
            overlap_score=scores[best],
            margin=margin,
        )


def stable_eval_split(project: str, *, buckets: int = 5) -> bool:
    digest = hashlib.sha256(project.encode()).digest()
    return int.from_bytes(digest[:4], "big") % buckets == 0
