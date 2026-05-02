from __future__ import annotations

import json
from pathlib import Path

from consolidate_agent.config import Settings
from consolidate_agent.extraction.pipeline import ConsolidationGraph
from consolidate_agent.knowledge.store import KnowledgeStore
from consolidate_agent.types import PitfallCandidate, PitfallCategory, PitfallScope, PipelineState, ProcessedStatus


def write_session(path: Path, session_id: str = "session-1", text: str = "please fix startup") -> None:
    events = [
        {
            "timestamp": "2026-04-14T09:15:26.524Z",
            "type": "session_meta",
            "payload": {"id": session_id, "timestamp": "2026-04-14T09:15:26.524Z", "cwd": "/tmp/project"},
        },
        {
            "timestamp": "2026-04-14T09:15:27.000Z",
            "type": "response_item",
            "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
        },
        {
            "timestamp": "2026-04-14T09:15:28.000Z",
            "type": "response_item",
            "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Use uv run instead of plain python."}]},
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


class FakeExtractor:
    calls = 0

    def __init__(self, settings: Settings):
        self.settings = settings

    def extract_chunk(self, chunk):
        FakeExtractor.calls += 1
        return [
            PitfallCandidate(
                candidate_id=f"candidate-{chunk.chunk_index}",
                session_id=chunk.session_id,
                title="Default interpreter assumption breaks validation",
                category=PitfallCategory.TOOLING_ENVIRONMENT,
                trigger="The workflow assumes python is available in PATH.",
                failure_mode="Validation fails before the real task is checked.",
                impact="Adds noisy debugging rounds.",
                preventive_rule="Use the project virtual environment interpreter for validation.",
                scope=PitfallScope.GLOBAL,
                evidence_refs=[chunk.messages[0].ref],
                confidence=0.8,
                chunk_id=chunk.chunk_id,
            )
        ]


class RaisingExtractor:
    def __init__(self, settings: Settings):
        self.settings = settings

    def extract_chunk(self, chunk):
        raise RuntimeError("boom")


def make_state(tmp_path: Path, input_dir: Path) -> PipelineState:
    output_dir = tmp_path / "outputs"
    return PipelineState(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        cursor_path=str(output_dir / "cursor.json"),
        processed_index_path=str(output_dir / "processed-index.json"),
        knowledge_db_path=str(output_dir / "knowledge.db"),
        session_index_path=str(tmp_path / "missing-index.jsonl"),
        max_chunk_chars=10000,
        overlap_messages=1,
    )


def test_graph_writes_processed_index_and_skips_unchanged_sessions(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "sessions"
    input_dir.mkdir()
    write_session(input_dir / "session.jsonl")
    monkeypatch.setattr("consolidate_agent.extraction.pipeline.PitfallExtractor", FakeExtractor)
    FakeExtractor.calls = 0

    settings = Settings(DASHSCOPE_API_KEY="dummy")
    first = ConsolidationGraph(settings).invoke(make_state(tmp_path, input_dir))
    second = ConsolidationGraph(settings).invoke(make_state(tmp_path, input_dir))

    assert first.stats.processed_sessions == 1
    assert first.stats.accepted_count == 1
    assert first.processed_index.sessions["session-1"].status == ProcessedStatus.PROCESSED
    assert second.stats.skipped_sessions == 1
    assert second.stats.processed_sessions == 0
    assert second.stats.accepted_count == 0
    assert FakeExtractor.calls == 1
    assert not (tmp_path / "outputs" / "pitfalls.json").exists()
    assert not (tmp_path / "outputs" / "pitfalls.md").exists()

    store = KnowledgeStore(tmp_path / "outputs" / "knowledge.db")
    try:
        records = store.list_all_source_records()
    finally:
        store.close()
    assert len(records) == 1
    assert records[0].title == "Default interpreter assumption breaks validation"


def test_graph_marks_failed_session_without_processing_it(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "sessions"
    input_dir.mkdir()
    write_session(input_dir / "session.jsonl")
    monkeypatch.setattr("consolidate_agent.extraction.pipeline.PitfallExtractor", RaisingExtractor)

    settings = Settings(DASHSCOPE_API_KEY="dummy")
    result = ConsolidationGraph(settings).invoke(make_state(tmp_path, input_dir))

    session_state = result.processed_index.sessions["session-1"]
    assert result.stats.failed_sessions == 1
    assert result.stats.processed_sessions == 0
    assert session_state.status == ProcessedStatus.FAILED
    assert session_state.error == "boom"
