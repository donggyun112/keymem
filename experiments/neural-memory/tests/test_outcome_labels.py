from pathlib import Path

import pytest

from neural_memory.claude_logs import Conversation
from neural_memory.outcome_labels import classify_outcome, iter_outcome_events


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ㅇㅇ ㄱㄱ", "accept"),
        ("좋아, 계속 실험해봐", "accept"),
        ("Perfect, this works", "accept"),
        ("ㄴㄴ 그게 아니야", "correct"),
        ("아니, 저장 모델 자체를 말한 거야", "correct"),
        ("이 부분은 잘못됐으니 되돌려", "correct"),
        ("새로운 기능을 구현해줘", None),
    ],
)
def test_classify_outcome(text: str, expected: str | None) -> None:
    assert classify_outcome(text) == expected


def test_outcome_event_points_to_previous_interaction() -> None:
    conversation = Conversation(
        project="project-a",
        session_id="session-1",
        source=Path("history.jsonl"),
        turns=("implement alpha", "ㅇㅇ 계속해", "implement beta", "ㄴㄴ 그건 틀렸어"),
    )

    events = list(iter_outcome_events(conversation))

    assert [(event.request_index, event.feedback_index, event.label) for event in events] == [
        (0, 1, "accept"),
        (2, 3, "correct"),
    ]
