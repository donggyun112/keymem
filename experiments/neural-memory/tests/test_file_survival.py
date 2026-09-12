from pathlib import Path

from neural_memory.file_survival import (
    find_change_survival,
    find_reverts,
    iter_version_chains,
    stratified_eval_sessions,
)


def test_detects_exact_a_b_a_revert(tmp_path: Path) -> None:
    session = tmp_path / "session-1"
    session.mkdir()
    (session / "file-id@v1").write_text("original")
    (session / "file-id@v2").write_text("changed")
    (session / "file-id@v3").write_text("original")

    chains = list(iter_version_chains(tmp_path))
    reverts = find_reverts(chains[0])

    assert len(reverts) == 1
    assert reverts[0].changed_version == 2
    assert reverts[0].restored_version == 3
    assert reverts[0].version_distance == 1


def test_ignores_monotonic_and_duplicate_versions(tmp_path: Path) -> None:
    session = tmp_path / "session-1"
    session.mkdir()
    (session / "file-id@v1").write_text("one")
    (session / "file-id@v2").write_text("two")
    (session / "file-id@v3").write_text("two")
    (session / "file-id@v4").write_text("three")

    chain = next(iter_version_chains(tmp_path))

    assert find_reverts(chain) == ()


def test_labels_line_change_that_survives_next_version(tmp_path: Path) -> None:
    session = tmp_path / "session-1"
    session.mkdir()
    (session / "file-id@v1").write_text("base\nold")
    (session / "file-id@v2").write_text("base\nnew")
    (session / "file-id@v3").write_text("base\nnew\nmore")

    events = find_change_survival(next(iter_version_chains(tmp_path)))

    assert len(events) == 1
    assert events[0].retained is True
    assert events[0].retention_score == 1.0
    assert "ADD new" in events[0].change_text


def test_labels_line_change_that_is_reverted_next_version(tmp_path: Path) -> None:
    session = tmp_path / "session-1"
    session.mkdir()
    (session / "file-id@v1").write_text("base\nold")
    (session / "file-id@v2").write_text("base\nnew")
    (session / "file-id@v3").write_text("base\nold")

    events = find_change_survival(next(iter_version_chains(tmp_path)))

    assert len(events) == 1
    assert events[0].retained is False
    assert events[0].retention_score == 0.0
    assert "REMOVE new" in events[0].outcome_text


def test_stratified_split_holds_out_whole_negative_session(tmp_path: Path) -> None:
    sessions = []
    for index, reverted in enumerate((True, True, False, False)):
        session = tmp_path / f"session-{index}"
        session.mkdir()
        (session / "file@v1").write_text("base\nold")
        (session / "file@v2").write_text("base\nnew")
        final = "base\nold" if reverted else "base\nnew\nmore"
        (session / "file@v3").write_text(final)
        sessions.extend(find_change_survival(next(iter_version_chains(session))))

    selected = stratified_eval_sessions(sessions, fraction=0.5)

    assert len(selected) == 2
    assert any(
        event.session_id in selected and not event.retained for event in sessions
    )
    assert any(event.session_id in selected and event.retained for event in sessions)
