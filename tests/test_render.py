from __future__ import annotations

from pathlib import Path

from consolidate_agent.render import read_cursor, write_cursor, write_pitfalls_markdown
from consolidate_agent.types import CursorState, PitfallCategory, PitfallEvidence, PitfallRecord, PitfallScope, utc_now


def test_write_markdown_groups_records(tmp_path: Path) -> None:
    path = tmp_path / "pitfalls.md"
    record = PitfallRecord(
        id="pitfall_1",
        title="Default interpreter assumption breaks validation",
        category=PitfallCategory.TOOLING_ENVIRONMENT,
        trigger="The workflow assumes python is available.",
        failure_mode="Validation fails too early.",
        impact="Noise.",
        preventive_rule="Use .venv/bin/python.",
        scope=PitfallScope.GLOBAL,
        evidence=PitfallEvidence(session_ids=["session-1"], message_refs=["tool_0001"]),
        confidence=0.9,
        tags=["tooling_environment", "python"],
        created_at=utc_now(),
        updated_at=utc_now(),
    )

    write_pitfalls_markdown(path, [record])

    content = path.read_text(encoding="utf-8")
    assert "# Pitfall Library" in content
    assert "## tooling_environment" in content
    assert "Use .venv/bin/python." in content


def test_cursor_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "cursor.json"
    cursor = CursorState(last_session_path="/tmp/session.jsonl", updated_at=utc_now())

    write_cursor(path, cursor)
    loaded = read_cursor(path)

    assert loaded.last_session_path == "/tmp/session.jsonl"
