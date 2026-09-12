import json
from pathlib import Path

from neural_memory.claude_logs import (
    clean_user_text,
    iter_conversations,
    iter_history_conversations,
    iter_revisit_examples,
    stable_eval_split,
)


def _row(content: object, *, sidechain: bool = False) -> str:
    return json.dumps(
        {
            "type": "user",
            "isSidechain": sidechain,
            "sessionId": "session-1",
            "message": {"role": "user", "content": content},
        }
    )


def test_clean_user_text_drops_internal_blocks_and_tool_results() -> None:
    content = [
        {"type": "tool_result", "content": "secret diagnostic"},
        {"type": "text", "text": "keep this request"},
    ]
    assert clean_user_text(content) == "keep this request"
    tagged = "<system-reminder>private metadata</system-reminder> actual request"
    assert clean_user_text(tagged) == "actual request"


def test_loader_keeps_only_root_user_language(tmp_path: Path) -> None:
    project = tmp_path / "project-a"
    project.mkdir()
    path = project / "session.jsonl"
    rows = [
        _row("Please keep the alpha_config decision for later."),
        json.dumps({"type": "assistant", "message": {"role": "assistant"}}),
        _row("Ignore this sidechain event completely.", sidechain=True),
        _row("We are discussing an unrelated deployment detail."),
        _row("Use alpha_config again for the release workflow."),
    ]
    path.write_text("\n".join(rows))

    conversations = list(iter_conversations(tmp_path))

    assert len(conversations) == 1
    assert conversations[0].turns == (
        "Please keep the alpha_config decision for later.",
        "We are discussing an unrelated deployment detail.",
        "Use alpha_config again for the release workflow.",
    )


def test_revisit_label_uses_delayed_unique_anchor() -> None:
    from neural_memory.claude_logs import Conversation

    conversation = Conversation(
        project="project-a",
        session_id="session-1",
        source=Path("session.jsonl"),
        turns=(
            "Remember alpha_config for the release pipeline.",
            "Discuss database migration planning.",
            "Review the user interface layout.",
            "Check unrelated unit testing details.",
            "The release pipeline should use alpha_config now.",
        ),
    )
    examples = list(
        iter_revisit_examples(
            conversation,
            candidates=3,
            min_delay=2,
            min_overlap=2.0,
            min_margin=0.5,
        )
    )

    assert len(examples) == 1
    assert examples[0].candidate_indices == (0, 1, 2)
    assert examples[0].target_offset == 0
    assert examples[0].delay == 4


def test_project_split_is_stable() -> None:
    assert stable_eval_split("project-a") == stable_eval_split("project-a")


def test_history_loader_splits_on_project_idle_gap(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    rows = [
        {"display": "first alpha_config decision", "project": "/alpha", "timestamp": 0},
        {"display": "second deployment detail", "project": "/alpha", "timestamp": 1_000},
        {"display": "third alpha_config reuse", "project": "/alpha", "timestamp": 2_000},
        {"display": "new session request one", "project": "/alpha", "timestamp": 20_000},
        {"display": "new session request two", "project": "/alpha", "timestamp": 21_000},
        {"display": "new session request three", "project": "/alpha", "timestamp": 22_000},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))

    conversations = list(iter_history_conversations(path, gap_seconds=10))

    assert [len(conversation.turns) for conversation in conversations] == [3, 3]
    assert conversations[0].session_id != conversations[1].session_id
