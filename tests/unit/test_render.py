from __future__ import annotations

import json
from pathlib import Path

from consolidate_agent.extraction.pipeline import read_cursor, write_candidates, write_cursor
from consolidate_agent.types import CursorState, PitfallCandidate, PitfallCategory, PitfallScope, utc_now


def test_cursor_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "cursor.json"
    cursor = CursorState(last_session_path="/tmp/session.jsonl", updated_at=utc_now())

    write_cursor(path, cursor)
    loaded = read_cursor(path)

    assert loaded.last_session_path == "/tmp/session.jsonl"


def make_candidate() -> PitfallCandidate:
    return PitfallCandidate(
        candidate_id="candidate-1",
        session_id="session-1",
        title="Validate JSON before parsing",
        category=PitfallCategory.EXECUTION_STRATEGY,
        trigger="The model returns structured data.",
        failure_mode="Invalid structure reaches downstream code.",
        impact="The workflow fails late.",
        preventive_rule="Validate JSON payloads against a schema before parsing model output.",
        scope=PitfallScope.GLOBAL,
        evidence_refs=["msg_0001"],
        confidence=0.9,
    )


def test_write_candidates_creates_loadable_json_file(tmp_path: Path) -> None:
    path = tmp_path / "out" / "candidates.json"

    write_candidates(path, [make_candidate()])

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))[0]["candidate_id"] == "candidate-1"


def test_write_candidates_empty_list_is_valid_json_array(tmp_path: Path) -> None:
    path = tmp_path / "candidates.json"

    write_candidates(path, [])

    with path.open("r", encoding="utf-8") as handle:
        assert json.load(handle) == []


def test_write_candidates_preserves_candidate_fields(tmp_path: Path) -> None:
    path = tmp_path / "candidates.json"
    candidate = make_candidate()

    write_candidates(path, [candidate])

    with path.open("r", encoding="utf-8") as handle:
        [payload] = json.load(handle)
    assert payload == candidate.model_dump(mode="json")
