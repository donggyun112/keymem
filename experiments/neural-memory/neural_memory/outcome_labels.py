from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass
from typing import Iterator, Literal

import torch
from torch import Tensor

from .claude_logs import Conversation


OutcomeLabel = Literal["accept", "correct"]

_CORRECTION = re.compile(
    r"^(?:ㄴㄴ+|아니(?:야|지|요)?\b|그게\s*아니|아님\b|no[,!.\s]|wrong\b|"
    r"actually[,!.\s]|revert\b|undo\b)|"
    r"(?:잘못(?:됐|된|했|알)|틀렸|안\s*돼|안돼|하지\s*마|말고\b|되돌려|취소해)",
    re.IGNORECASE,
)
_ACCEPTANCE = re.compile(
    r"^(?:ㅇㅇ+|응+|어+\b|그래+\b|맞아\b|좋아\b|좋네\b|굳+\b|굿+\b|"
    r"오케이\b|계속\b|ㄱㄱ+|yes\b|yep\b|ok(?:ay)?\b|good\b|great\b|"
    r"perfect\b|works?\b|continue\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OutcomeEvent:
    project: str
    session_id: str
    request_index: int
    feedback_index: int
    label: OutcomeLabel


def classify_outcome(text: str) -> OutcomeLabel | None:
    normalized = " ".join(text.strip().split()).lower()
    if not normalized:
        return None
    if _CORRECTION.search(normalized):
        return "correct"
    if _ACCEPTANCE.search(normalized):
        return "accept"
    return None


def iter_outcome_events(conversation: Conversation) -> Iterator[OutcomeEvent]:
    """Pair explicit feedback with the immediately preceding interaction request."""

    for feedback_index in range(1, len(conversation.turns)):
        label = classify_outcome(conversation.turns[feedback_index])
        if label is None:
            continue
        yield OutcomeEvent(
            project=conversation.project,
            session_id=conversation.session_id,
            request_index=feedback_index - 1,
            feedback_index=feedback_index,
            label=label,
        )


def hashed_text_features(text: str, *, dimension: int = 512) -> Tensor:
    """Frozen, language-agnostic signed character n-gram features."""

    if dimension < 1:
        raise ValueError("dimension must be positive")
    normalized = " ".join(text.lower().split())
    features = torch.zeros(dimension)
    wrapped = f"^{normalized}$"
    for width in (1, 2, 3, 4):
        for start in range(max(0, len(wrapped) - width + 1)):
            ngram = wrapped[start : start + width].encode()
            digest = hashlib.blake2b(ngram, digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % dimension
            features[index] += 1.0 if value & 1 else -1.0
    norm = features.norm()
    return features / norm if norm > 0 else features
