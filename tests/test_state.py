from __future__ import annotations

from pathlib import Path

from consolidate_agent.state import read_processed_index, write_processed_index
from consolidate_agent.types import ProcessedIndex, ProcessedSessionState, ProcessedStatus


def test_processed_index_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "processed-index.json"
    index = ProcessedIndex(
        sessions={
            "session-1": ProcessedSessionState(
                session_id="session-1",
                path="/tmp/session.jsonl",
                normalized_hash="abc",
                status=ProcessedStatus.PROCESSED,
                chunk_count=2,
            )
        }
    )

    write_processed_index(path, index)
    loaded = read_processed_index(path)

    assert loaded.sessions["session-1"].status == ProcessedStatus.PROCESSED
    assert loaded.sessions["session-1"].chunk_count == 2
    assert loaded.updated_at is not None
