from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


_VERSIONED = re.compile(r"^(?P<identity>.+)@v(?P<version>\d+)$")


@dataclass(frozen=True)
class FileVersion:
    version: int
    digest: str
    size: int
    path: Path


@dataclass(frozen=True)
class VersionChain:
    session_id: str
    file_id: str
    versions: tuple[FileVersion, ...]


@dataclass(frozen=True)
class RevertEvent:
    session_id: str
    file_id: str
    changed_version: int
    restored_version: int
    version_distance: int


@dataclass(frozen=True)
class ChangeSurvivalEvent:
    session_id: str
    file_id: str
    changed_version: int
    observed_version: int
    change_text: str
    outcome_text: str
    retention_score: float
    retained: bool


def iter_version_chains(root: str | Path) -> Iterator[VersionChain]:
    root = Path(root).expanduser()
    grouped: dict[tuple[str, str], list[FileVersion]] = {}
    for path in root.rglob("*@v*"):
        if not path.is_file():
            continue
        match = _VERSIONED.match(path.name)
        if match is None:
            continue
        content = path.read_bytes()
        key = (path.parent.name, match.group("identity"))
        grouped.setdefault(key, []).append(
            FileVersion(
                version=int(match.group("version")),
                digest=hashlib.sha256(content).hexdigest(),
                size=len(content),
                path=path,
            )
        )
    for (session_id, file_id), versions in sorted(grouped.items()):
        ordered = tuple(sorted(versions, key=lambda item: item.version))
        if len(ordered) >= 2:
            yield VersionChain(session_id, file_id, ordered)


def find_reverts(chain: VersionChain) -> tuple[RevertEvent, ...]:
    """Find high-precision A -> B -> A content restoration events."""

    events: list[RevertEvent] = []
    versions = chain.versions
    for changed_index in range(1, len(versions)):
        previous = versions[changed_index - 1]
        changed = versions[changed_index]
        if changed.digest == previous.digest:
            continue
        for restored_index in range(changed_index + 1, len(versions)):
            restored = versions[restored_index]
            if restored.digest != previous.digest:
                continue
            events.append(
                RevertEvent(
                    session_id=chain.session_id,
                    file_id=chain.file_id,
                    changed_version=changed.version,
                    restored_version=restored.version,
                    version_distance=restored.version - changed.version,
                )
            )
            break
    return tuple(events)


def _lines(version: FileVersion, *, max_bytes: int = 1_000_000) -> set[str] | None:
    content = version.path.read_bytes()
    if len(content) > max_bytes or b"\x00" in content:
        return None
    text = content.decode("utf-8", errors="ignore")
    return {line.strip() for line in text.splitlines() if line.strip()}


def _delta_text(before: set[str], after: set[str]) -> str:
    added = sorted(after - before)
    removed = sorted(before - after)
    return "\n".join(
        [*(f"ADD {line}" for line in added), *(f"REMOVE {line}" for line in removed)]
    )[:8_000]


def find_change_survival(
    chain: VersionChain,
    *,
    retain_threshold: float = 0.8,
    revert_threshold: float = 0.2,
) -> tuple[ChangeSurvivalEvent, ...]:
    """Label whether an A->B line delta survives into the next version C."""

    if not 0.0 <= revert_threshold < retain_threshold <= 1.0:
        raise ValueError("require 0 <= revert threshold < retain threshold <= 1")
    events: list[ChangeSurvivalEvent] = []
    for index in range(1, len(chain.versions) - 1):
        before, changed, observed = chain.versions[index - 1 : index + 2]
        before_lines = _lines(before)
        changed_lines = _lines(changed)
        observed_lines = _lines(observed)
        if before_lines is None or changed_lines is None or observed_lines is None:
            continue
        added = changed_lines - before_lines
        removed = before_lines - changed_lines
        changed_count = len(added) + len(removed)
        if changed_count == 0:
            continue
        survived = len(added.intersection(observed_lines))
        survived += len(removed - observed_lines)
        score = survived / changed_count
        if score >= retain_threshold:
            retained = True
        elif score <= revert_threshold:
            retained = False
        else:
            continue
        events.append(
            ChangeSurvivalEvent(
                session_id=chain.session_id,
                file_id=chain.file_id,
                changed_version=changed.version,
                observed_version=observed.version,
                change_text=_delta_text(before_lines, changed_lines),
                outcome_text=_delta_text(changed_lines, observed_lines),
                retention_score=score,
                retained=retained,
            )
        )
    return tuple(events)


def stratified_eval_sessions(
    events: list[ChangeSurvivalEvent], *, fraction: float = 0.2
) -> frozenset[str]:
    """Hold out whole sessions while preserving rare revert-bearing groups."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between zero and one")
    sessions = {event.session_id for event in events}
    negative_sessions = {
        event.session_id for event in events if not event.retained
    }
    groups = (negative_sessions, sessions - negative_sessions)
    selected: set[str] = set()
    for group in groups:
        ordered = sorted(
            group,
            key=lambda session: hashlib.sha256(session.encode()).digest(),
        )
        count = max(1, round(len(ordered) * fraction)) if ordered else 0
        selected.update(ordered[:count])
    return frozenset(selected)
